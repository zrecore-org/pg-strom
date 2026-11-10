/*
 * gpu_preagg.c
 *
 * Aggregation and Group-By with GPU acceleration
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "gpuserv/scan_caps.h"

/* static variables */
/* a text / varchar / bpchar GROUP BY key: a plain column (RelabelType varchar -> text stripped) under a deterministic collation */
static bool
pgstrom_group_string_key_ok(Expr *expr)
{
	Node	   *node = (Node *) expr;

	while (node && IsA(node, RelabelType))
		node = (Node *) ((RelabelType *) node)->arg;
	if (!node || !IsA(node, Var))
		return false;
	if (exprType(node) == UUIDOID)
		return true;				/* no collation: 16 bytes */
	return (OidIsValid(exprCollation((Node *) expr)) &&
			get_collation_isdeterministic(exprCollation((Node *) expr)));
}

static create_upper_paths_hook_type	create_upper_paths_next = NULL;
static CustomPathMethods	gpupreagg_path_methods;
static CustomScanMethods	gpupreagg_plan_methods;
static CustomExecMethods	gpupreagg_exec_methods;
static bool					pgstrom_enable_gpupreagg = false;
static bool					pgstrom_enable_partitionwise_gpupreagg = false;
static bool					pgstrom_enable_numeric_aggfuncs;
static int					pgstrom_gpupreagg_max_groups;	/* GUC */
static int					pgstrom_gpupreagg_max_groups_heap;	/* GUC */
bool						pgstrom_enable_gpusort = false;

/*
 * pgstrom_is_gpupreagg_path
 */
bool
pgstrom_is_gpupreagg_path(const Path *path)
{
	if (IsA(path, CustomPath))
	{
		const CustomPath *cpath = (const CustomPath *)path;

		return (cpath->methods == &gpupreagg_path_methods);
	}
	return false;
}

/*
 * pgstrom_is_gpupreagg_plan
 */
bool
pgstrom_is_gpupreagg_plan(const Plan *plan)
{
	if (IsA(plan, CustomScan))
	{
		const CustomScan *cscan = (const CustomScan *)plan;

		return (cscan->methods == &gpupreagg_plan_methods);
	}
	return false;
}

/*
 * pgstrom_is_gpupreagg_state
 */
bool
pgstrom_is_gpupreagg_state(const PlanState *ps)
{
	if (IsA(ps, CustomScanState))
	{
		const CustomScanState *css = (const CustomScanState *)ps;

		return (css->methods == &gpupreagg_exec_methods);
	}
	return false;
}

/*
 * List of supported aggregate functions
 */
typedef struct
{
	/* aggregate function can be preprocessed */
	const char *aggfn_signature;
	/*
	 * A pair of final/partial function will generate same result.
	 * Its prefix indicates the schema that stores these functions.
	 * c: pg_catalog ... the system default
	 * s: pgstrom    ... PG-Strom's special ones
	 */
	const char *finalfn_agg_signature;		/* used by Agg(CPU) */
	const char *partfn_signature;			/* used by GpuPreAgg */
	const char *finalfn_proj_signature;		/* used by GpuAgg */
	int			partfn_action;	/* any of KAGG_ACTION__* */
	int			finalfn_code;	/* any of KAGG_FINAL__* */
	bool		numeric_aware;	/* ignored, if !enable_numeric_aggfuncs */
	int			gpusort_keykind;			/* used by GPU-Sort */
} aggfunc_catalog_t;

static aggfunc_catalog_t	aggfunc_catalog_array[] = {
	/* COUNT(*) = SUM(NROWS()) */
	{"count()",
	 "s:fcount(int8)",
	 "s:nrows()",
	 NULL,		/* use nrows() as is */
	 KAGG_ACTION__NROWS_ANY,
	 KAGG_FINAL__SIMPLE_VREF,
	 false,
	 KSORT_KEY_KIND__VREF
	},
	/* COUNT(X) = SUM(NROWS(X)) */
	{"count(any)",
	 "s:fcount(int8)",
	 "s:nrows(any)",
	 NULL,		/* use nrows() as is */
	 KAGG_ACTION__NROWS_COND,
	 KAGG_FINAL__SIMPLE_VREF,
	 false,
	 KSORT_KEY_KIND__VREF
	},
	/*
	 * MIN(X) = MIN(PMIN(X))
	 */
	{"min(int1)",
	 "s:min_i1(bytea)",
	 "s:pmin(int4)",
	 "s:fmin_i1(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_INT8,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(int2)",
	 "s:min_i2(bytea)",
	 "s:pmin(int4)",
	 "s:fmin_i2(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_INT16,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(int4)",
	 "s:min_i4(bytea)",
	 "s:pmin(int4)",
	 "s:fmin_i4(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_INT32,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(int8)",
	 "s:min_i8(bytea)",
	 "s:pmin(int8)",
	 "s:fmin_i8(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_INT64,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(float2)",
	 "s:min_f2(bytea)",
	 "s:pmin(float8)",
	 "s:fmin_f2(bytea)",
	 KAGG_ACTION__PMIN_FP64,
	 KAGG_FINAL__FMINMAX_FP16,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"min(float4)",
	 "s:min_f4(bytea)",
	 "s:pmin(float8)",
	 "s:fmin_f4(bytea)",
	 KAGG_ACTION__PMIN_FP64,
	 KAGG_FINAL__FMINMAX_FP32,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"min(float8)",
	 "s:min_f8(bytea)",
	 "s:pmin(float8)",
	 "s:fmin_f8(bytea)",
	 KAGG_ACTION__PMIN_FP64,
	 KAGG_FINAL__FMINMAX_FP64,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	/*
	 * min / max of a numeric of a known scale (numeric(p,s) columns and what follows from them): the Vulkan service takes the
	 * extreme of the unscaled int64 values, and its partial state is the numeric itself in a bytea (pgstrom.pmin(numeric),
	 * pgstrom.min_numx(bytea)): exact, unlike the float8 state of the CUDA code. The action is the integer one; the typmod of
	 * the argument (a numeric typmod, see below) tells the service the scale.
	 */
	{"min(numeric)",
	 "s:min_numx(bytea)",
	 "s:pmin(numeric)",
	 "s:fmin_numx(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_NUMX,
	 true,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"min(money)",
	 "s:min_cash(bytea)",
	 "s:pmin(money)",
	 "s:fmin_cash(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_CASH,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(date)",
	 "s:min_date(bytea)",
	 "s:pmin(date)",
	 "s:fmin_date(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_DATE,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(time)",
	 "s:min_time(bytea)",
	 "s:pmin(time)",
	 "s:fmin_time(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_TIME,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(timestamp)",
	 "s:min_ts(bytea)",
	 "s:pmin(timestamp)",
	 "s:fmin_ts(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_TIMESTAMP,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"min(timestamptz)",
	 "s:min_tstz(bytea)",
	 "s:pmin(timestamptz)",
	 "s:fmin_tstz(bytea)",
	 KAGG_ACTION__PMIN_INT64,
	 KAGG_FINAL__FMINMAX_TIMESTAMPTZ,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	/*
	 * BOOL_AND(X) = EVERY(X) = MIN(X), BOOL_OR(X) = MAX(X) of the 0/1 values: the minimum / maximum states of the integers
	 */
	{"bool_and(bool)",
	 "s:min_bool(bytea)",
	 "s:pmin(bool)",
	 "s:fmin_bool(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_BOOL,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"every(bool)",
	 "s:min_bool(bytea)",
	 "s:pmin(bool)",
	 "s:fmin_bool(bytea)",
	 KAGG_ACTION__PMIN_INT32,
	 KAGG_FINAL__FMINMAX_BOOL,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"bool_or(bool)",
	 "s:max_bool(bytea)",
	 "s:pmax(bool)",
	 "s:fmax_bool(bytea)",
	 KAGG_ACTION__PMAX_INT32,
	 KAGG_FINAL__FMINMAX_BOOL,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	/*
	 * MAX(X) = MAX(PMAX(X))
	 */
	{"max(int1)",
	 "s:max_i1(bytea)",
	 "s:pmax(int4)",
	 "s:fmax_i1(bytea)",
	 KAGG_ACTION__PMAX_INT32,
	 KAGG_FINAL__FMINMAX_INT8,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(int2)",
	 "s:max_i2(bytea)",
	 "s:pmax(int4)",
	 "s:fmax_i2(bytea)",
	 KAGG_ACTION__PMAX_INT32,
	 KAGG_FINAL__FMINMAX_INT16,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(int4)",
	 "s:max_i4(bytea)",
	 "s:pmax(int4)",
	 "s:fmax_i4(bytea)",
	 KAGG_ACTION__PMAX_INT32,
	 KAGG_FINAL__FMINMAX_INT32,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(int8)",
	 "s:max_i8(bytea)",
	 "s:pmax(int8)",
	 "s:fmax_i8(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_INT64,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(float2)",
	 "s:max_f2(bytea)",
	 "s:pmax(float8)",
	 "s:fmax_f2(bytea)",
	 KAGG_ACTION__PMAX_FP64,
	 KAGG_FINAL__FMINMAX_FP16,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"max(float4)",
	 "s:max_f4(bytea)",
	 "s:pmax(float8)",
	 "s:fmax_f4(bytea)",
	 KAGG_ACTION__PMAX_FP64,
	 KAGG_FINAL__FMINMAX_FP32,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"max(float8)",
	 "s:max_f8(bytea)",
	 "s:pmax(float8)",
	 "s:fmax_f8(bytea)",
	 KAGG_ACTION__PMAX_FP64,
	 KAGG_FINAL__FMINMAX_FP64,
	 false,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"max(numeric)",
	 "s:max_numx(bytea)",
	 "s:pmax(numeric)",
	 "s:fmax_numx(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_NUMX,
	 true,
	 KSORT_KEY_KIND__PMINMAX_FP64
	},
	{"max(money)",
	 "s:max_cash(bytea)",
	 "s:pmax(money)",
	 "s:fmax_cash(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_CASH,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(date)",
	 "s:max_date(bytea)",
	 "s:pmax(date)",
	 "s:fmax_date(bytea)",
	 KAGG_ACTION__PMAX_INT32,
	 KAGG_FINAL__FMINMAX_DATE,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(time)",
	 "s:max_time(bytea)",
	 "s:pmax(time)",
	 "s:fmax_time(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_TIME,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(timestamp)",
	 "s:max_ts(bytea)",
	 "s:pmax(timestamp)",
	 "s:fmax_ts(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_TIMESTAMP,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	{"max(timestamptz)",
	 "s:max_tstz(bytea)",
	 "s:pmax(timestamptz)",
	 "s:fmax_tstz(bytea)",
	 KAGG_ACTION__PMAX_INT64,
	 KAGG_FINAL__FMINMAX_TIMESTAMPTZ,
	 false,
	 KSORT_KEY_KIND__PMINMAX_INT64
	},
	/*
	 * SUM(X) = SUM(PSUM(X))
	 */
	{"sum(int1)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 "s:fsum_int(bytea)",
	 KAGG_ACTION__PSUM_INT,
	 KAGG_FINAL__FSUM_INT,
	 false,
	 KSORT_KEY_KIND__PSUM_INT64
	},
	{"sum(int2)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 "s:fsum_int(bytea)",
	 KAGG_ACTION__PSUM_INT,
	 KAGG_FINAL__FSUM_INT,
	 false,
	 KSORT_KEY_KIND__PSUM_INT64
	},
	{"sum(int4)",
	 "s:sum_int(bytea)",
	 "s:psum(int8)",
	 "s:fsum_int(bytea)",
	 KAGG_ACTION__PSUM_INT,
	 KAGG_FINAL__FSUM_INT,
	 false,
	 KSORT_KEY_KIND__PSUM_INT64
	},
	{"sum(int8)",
	 "s:sum_int64(bytea)",
	 "s:psum64(int8)",
	 "s:fsum_int64(bytea)",
	 KAGG_ACTION__PSUM_INT64,
	 KAGG_FINAL__FSUM_INT64,
	 false,
	 KSORT_KEY_KIND__PSUM_NUMERIC
	},
	{"sum(float2)",
	 "s:sum_fp64(bytea)",
	 "s:psum(float8)",
	 "s:fsum_fp64(bytea)",
	 KAGG_ACTION__PSUM_FP,
	 KAGG_FINAL__FSUM_FP64,
	 false,
	 KSORT_KEY_KIND__PSUM_FP64
	},
	{"sum(float4)",
	 "s:sum_fp32(bytea)",
	 "s:psum(float8)",
	 "s:fsum_fp32(bytea)",
	 KAGG_ACTION__PSUM_FP32,
	 KAGG_FINAL__FSUM_FP32,
	 false,
	 KSORT_KEY_KIND__PSUM_FP64
	},
	{"sum(float8)",
	 "s:sum_fp64(bytea)",
	 "s:psum(float8)",
	 "s:fsum_fp64(bytea)",
	 KAGG_ACTION__PSUM_FP,
	 KAGG_FINAL__FSUM_FP64,
	 false,
	 KSORT_KEY_KIND__PSUM_FP64
	},
	{"sum(numeric)",
	 "s:sum_numeric(bytea)",
	 "s:psum(numeric)",
	 "s:fsum_numeric(bytea)",
	 KAGG_ACTION__PSUM_NUMERIC,
	 KAGG_FINAL__FSUM_NUMERIC,
	 true,
	 KSORT_KEY_KIND__PSUM_NUMERIC
	},
	{"sum(money)",
	 "s:sum_cash(bytea)",
	 "s:psum(money)",
	 "s:fsum_cach(bytea)",
	 KAGG_ACTION__PSUM_INT,
	 KAGG_FINAL__FSUM_CASH,
	 false,
	 KSORT_KEY_KIND__PSUM_INT64
	},
	/*
	 * AVG(X) = EX_AVG(NROWS(X), PSUM(X))
	 */
	{"avg(int1)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 "s:favg_int(bytea)",
	 KAGG_ACTION__PAVG_INT,
	 KAGG_FINAL__FAVG_INT,
	 false,
	 KSORT_KEY_KIND__PAVG_INT64
	},
	{"avg(int2)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 "s:favg_int(bytea)",
	 KAGG_ACTION__PAVG_INT,
	 KAGG_FINAL__FAVG_INT,
	 false,
	 KSORT_KEY_KIND__PAVG_INT64
	},
	{"avg(int4)",
	 "s:avg_int(bytea)",
	 "s:pavg(int8)",
	 "s:favg_int(bytea)",
	 KAGG_ACTION__PAVG_INT,
	 KAGG_FINAL__FAVG_INT,
	 false,
	 KSORT_KEY_KIND__PAVG_INT64
	},
	{"avg(int8)",
	 "s:avg_int64(bytea)",
	 "s:pavg64(int8)",
	 "s:favg_int64(bytea)",
	 KAGG_ACTION__PAVG_INT64,
	 KAGG_FINAL__FAVG_INT64,
	 false,
	 KSORT_KEY_KIND__PAVG_NUMERIC
	},
	{"avg(float2)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 "s:favg_fp(bytea)",
	 KAGG_ACTION__PAVG_FP,
	 KAGG_FINAL__FAVG_FP64,
	 false,
	 KSORT_KEY_KIND__PAVG_FP64
	},
	{"avg(float4)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 "s:favg_fp(bytea)",
	 KAGG_ACTION__PAVG_FP,
	 KAGG_FINAL__FAVG_FP64,
	 false,
	 KSORT_KEY_KIND__PAVG_FP64
	},
	{"avg(float8)",
	 "s:avg_fp(bytea)",
	 "s:pavg(float8)",
	 "s:favg_fp(bytea)",
	 KAGG_ACTION__PAVG_FP,
	 KAGG_FINAL__FAVG_FP64,
	 false,
	 KSORT_KEY_KIND__PAVG_FP64
	},
	{"avg(numeric)",
	 "s:avg_numeric(bytea)",
	 "s:pavg(numeric)",
	 "s:favg_numeric(bytea)",
	 KAGG_ACTION__PAVG_NUMERIC,
	 KAGG_FINAL__FAVG_NUMERIC,
	 true,
	 KSORT_KEY_KIND__PAVG_NUMERIC
	},
	/*
	 * STDDEV(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev(int1)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(int2)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(int4)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(int8)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(float2)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(float4)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(float8)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev(numeric)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	/*
	 * STDDEV_SAMP(X) = EX_STDDEV_SAMP(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev_samp(int1)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(int2)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(int4)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(int8)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(float2)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(float4)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(float8)",
	 "s:stddev_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"stddev_samp(numeric)",
	 "s:stddev_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_SAMP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	/*
	 * STDDEV_POP(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X))
	 */
	{"stddev_pop(int1)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP,
	},
	{"stddev_pop(int2)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(int4)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(int8)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(float2)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(float4)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(float8)",
	 "s:stddev_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FSTDDEV_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"stddev_pop(numeric)",
	 "s:stddev_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fstddev_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FSTDDEV_POP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	/*
	 * VARIANCE(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"variance(int1)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(int2)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(int4)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(int8)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(float2)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(float4)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(float8)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"variance(numeric)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	/*
	 * VAR_SAMP(X) = VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"var_samp(int1)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(int2)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(int4)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(int8)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(float2)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(float4)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(float8)",
	 "s:var_sampf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_sampf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_SAMPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	{"var_samp(numeric)",
	 "s:var_samp(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_samp(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_SAMP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_SAMP
	},
	/*
	 * VAR_POP(X)  = VAR_POP(NROWS(), PSUM(X),PSUM(X^2))
	 */
	{"var_pop(int1)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(int2)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(int4)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(int8)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_POP,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(float2)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(float4)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(float8)",
	 "s:var_popf(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_popf(bytea)",
	 KAGG_ACTION__STDDEV,
	 KAGG_FINAL__FVAR_POPF,
	 false,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	{"var_pop(numeric)",
	 "s:var_pop(bytea)",
	 "s:pvariance(float8)",
	 "s:fvar_pop(bytea)",
	 KAGG_ACTION__STDDEV_NUM,
	 KAGG_FINAL__FVAR_POP,
	 true,
	 KSORT_KEY_KIND__PVARIANCE_POP
	},
	/*
	 * CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
	 *                          PCOV_X(X,Y),  PCOV_Y(X,Y)
	 *                          PCOV_X2(X,Y), PCOV_Y2(X,Y),
	 *                          PCOV_XY(X,Y))
	 */
	{"corr(float8,float8)",
	 "s:corr(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fcorr(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FCORR,
	 false,
	 KSORT_KEY_KIND__PCOVAR_CORR
	},
	{"covar_samp(float8,float8)",
	 "s:covar_samp(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fcovar_samp(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FCOVAR_SAMP,
	 false,
	 KSORT_KEY_KIND__PCOVAR_SAMP
	},
	{"covar_pop(float8,float8)",
	 "s:covar_pop(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fcovar_pop(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FCOVAR_POP,
	 false,
	 KSORT_KEY_KIND__PCOVAR_POP
	},
	/*
	 * Aggregation to support least squares method
	 *
	 * That takes PSUM_X, PSUM_Y, PSUM_X2, PSUM_Y2, PSUM_XY according
	 * to the function
	 */
	{"regr_avgx(float8,float8)",
	 "s:regr_avgx(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_avgx(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_AVGX,
	 false,
	 KSORT_KEY_KIND__PCOVAR_AVGX
	},
	{"regr_avgy(float8,float8)",
	 "s:regr_avgy(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_avgy(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_AVGY,
	 false,
	 KSORT_KEY_KIND__PCOVAR_AVGY
	},
	{"regr_count(float8,float8)",
	 "s:regr_count(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_count(bytea)",
	 KAGG_ACTION__COVAR_CNT,
	 KAGG_FINAL__FREGR_COUNT,
	 false,
	 KSORT_KEY_KIND__PCOVAR_COUNT
	},
	{"regr_intercept(float8,float8)",
	 "s:regr_intercept(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_intercept(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_INTERCEPT,
	 false,
	 KSORT_KEY_KIND__PCOVAR_INTERCEPT
	},
	{"regr_r2(float8,float8)",
	 "s:regr_r2(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_r2(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_R2,
	 false,
	 KSORT_KEY_KIND__PCOVAR_REGR_R2
	},
	{"regr_slope(float8,float8)",
	 "s:regr_slope(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_slope(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_SLOPE,
	 false,
	 KSORT_KEY_KIND__PCOVAR_REGR_SLOPE
	},
	{"regr_sxx(float8,float8)",
	 "s:regr_sxx(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_sxx(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_SXX,
	 false,
	 KSORT_KEY_KIND__PCOVAR_REGR_SXX
	},
	{"regr_sxy(float8,float8)",
	 "s:regr_sxy(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_sxy(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_SXY,
	 false,
	 KSORT_KEY_KIND__PCOVAR_REGR_SXY
	},
	{"regr_syy(float8,float8)",
	 "s:regr_syy(bytea)",
	 "s:pcovar(float8,float8)",
	 "s:fregr_syy(bytea)",
	 KAGG_ACTION__COVAR,
	 KAGG_FINAL__FREGR_SYY,
	 false,
	 KSORT_KEY_KIND__PCOVAR_REGR_SYY
	},
	{ NULL, NULL, NULL, NULL, -1, -1, false, -1 },
};

/*
 * aggfunc_catalog_entry; hashed catalog entry
 */
typedef struct
{
	Oid		aggfn_oid;
	Oid		final_agg_func_oid;
	Oid		final_proj_func_oid;
	Oid		partial_func_oid;
	Oid		partial_func_filtered_oid;
	Oid		partial_func_rettype;
	int		partial_func_nargs;
	int		partial_func_action;
	int		partial_func_bufsz;
	bool	numeric_aware;
	bool	is_valid_entry;
} aggfunc_catalog_entry;

typedef struct
{
	Oid		finalfn_oid;
	int		finalfn_code;
	bool	is_valid_entry;
} finalfunc_catalog_entry;

static HTAB	   *aggfunc_catalog_htable = NULL;
static HTAB	   *finalfunc_catalog_htable = NULL;

static void
aggfunc_catalog_htable_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	hash_destroy(aggfunc_catalog_htable);
	aggfunc_catalog_htable = NULL;
	hash_destroy(finalfunc_catalog_htable);
	finalfunc_catalog_htable = NULL;
}

static Oid
__aggfunc_resolve_func_signature(const char *signature, bool with_filtered_clause)
{
	char	   *fn_name = alloca(strlen(signature));
	Oid			fn_namespace;
	oidvector  *fn_argtypes;
	int			fn_nargs = 0;
	Oid			fn_oid;
	Oid			type_oid;
	char	   *base, *tok, *pos;

	if (strncmp(signature, "c:", 2) == 0)
		fn_namespace = PG_CATALOG_NAMESPACE;
	else if (strncmp(signature, "s:", 2) == 0)
		fn_namespace = get_namespace_oid("pgstrom", false);
	else
		elog(ERROR, "wrong function signature: %s", signature);

	strcpy(fn_name, signature + 2);
	base = strchr(fn_name, '(');
	if (!base)
		elog(ERROR, "wrong function signature: %s", signature);
	*base++ = '\0';
	pos = strchr(base, ')');
	if (!pos)
		elog(ERROR, "wrong function signature: %s", signature);
	*pos = '\0';

	fn_argtypes = alloca(offsetof(oidvector, values[80]));
	fn_argtypes->ndim = 1;
	fn_argtypes->dataoffset = 0;
	fn_argtypes->elemtype = OIDOID;
	fn_argtypes->dim1 = 0;
	fn_argtypes->lbound1 = 0;
	for (tok = strtok_r(base, ",", &pos);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &pos))
	{
		type_oid = GetSysCacheOid2(TYPENAMENSP,
								   Anum_pg_type_oid,
								   CStringGetDatum(tok),
								   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
		if (!OidIsValid(type_oid))
			elog(ERROR, "cache lookup failed for type '%s'", tok);
		fn_argtypes->values[fn_nargs++] = type_oid;
	}
	if (with_filtered_clause)
		fn_argtypes->values[fn_nargs++] = BOOLOID;
	fn_argtypes->dim1 = fn_nargs;
	SET_VARSIZE(fn_argtypes, offsetof(oidvector, values[fn_nargs]));

	fn_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
							 Anum_pg_proc_oid,
							 CStringGetDatum(fn_name),
							 PointerGetDatum(fn_argtypes),
							 ObjectIdGetDatum(fn_namespace));
	if (!OidIsValid(fn_oid))
		elog(ERROR, "Catalog corruption? '%s' was not found",
			 funcname_signature_string(fn_name,
									   fn_argtypes->dim1,
									   NIL,
									   fn_argtypes->values));
	return fn_oid;
}

static void
__aggfunc_resolve_partial_func(aggfunc_catalog_entry *entry,
							   const char *partfn_signature,
							   int partfn_action)
{
	Oid		func_oid = __aggfunc_resolve_func_signature(partfn_signature, false);
	Oid		func_filtered_oid = __aggfunc_resolve_func_signature(partfn_signature, true);
	Oid		type_oid;
	int		func_nargs;
	int		partfn_bufsz;

	switch (partfn_action)
	{
		case KAGG_ACTION__NROWS_ANY:
			func_nargs = 0;
			type_oid = INT8OID;
			partfn_bufsz = sizeof(int64_t);
			break;
		case KAGG_ACTION__NROWS_COND:
			func_nargs = 1;
			type_oid = INT8OID;
			partfn_bufsz = sizeof(int64_t);
			break;
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:
		case KAGG_ACTION__PMAX_INT32:
		case KAGG_ACTION__PMAX_INT64:
			func_nargs = 1;
            type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__pminmax_int64_packed);
			break;

		case KAGG_ACTION__PMIN_FP64:
		case KAGG_ACTION__PMAX_FP64:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__pminmax_fp64_packed);
			break;

		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PSUM_INT:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__psum_int_packed);
			break;

		case KAGG_ACTION__PAVG_FP:
		case KAGG_ACTION__PSUM_FP:
		case KAGG_ACTION__PSUM_FP32:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__psum_fp_packed);
			break;

		case KAGG_ACTION__PAVG_INT64:
		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_NUMERIC:
		case KAGG_ACTION__PSUM_NUMERIC:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__psum_numeric_packed);
			break;

		case KAGG_ACTION__STDDEV:
		case KAGG_ACTION__STDDEV_NUM:
			func_nargs = 1;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__stddev_packed);
			break;
		case KAGG_ACTION__COVAR:
		case KAGG_ACTION__COVAR_CNT:
			func_nargs = 2;
			type_oid = BYTEAOID;
			partfn_bufsz = sizeof(kagg_state__covar_packed);
			break;
		default:
			elog(ERROR, "Catalog corruption? unknown action: %d", partfn_action);
			break;
	}
	entry->partial_func_oid = func_oid;
	entry->partial_func_filtered_oid = func_filtered_oid;
	entry->partial_func_rettype = get_func_rettype(func_oid);
	entry->partial_func_nargs = get_func_nargs(func_oid);
	entry->partial_func_action = partfn_action;
	entry->partial_func_bufsz  = partfn_bufsz;

	if (entry->partial_func_rettype != type_oid ||
		entry->partial_func_nargs != func_nargs ||
		entry->partial_func_bufsz != MAXALIGN(partfn_bufsz))
		elog(ERROR, "Catalog curruption? partial function mismatch: %s",
			 partfn_signature);
}

static void
__setup_aggfunc_catalog_entry(aggfunc_catalog_entry *entry,
							  const aggfunc_catalog_t *cat,
							  Form_pg_proc agg_proc)
{
	Oid			func_oid;
	HeapTuple	htup;
	Form_pg_proc proc;

	/* partial agg function */
	__aggfunc_resolve_partial_func(entry,
								   cat->partfn_signature,
								   cat->partfn_action);
	/* final agg function (used by Agg node) */
	Assert(cat->finalfn_agg_signature != NULL);
	func_oid = __aggfunc_resolve_func_signature(cat->finalfn_agg_signature, false);
	if (!SearchSysCacheExists1(AGGFNOID, ObjectIdGetDatum(func_oid)))
		elog(ERROR, "Catalog corruption? final function mismatch: %s",
			 format_procedure(func_oid));
	if (get_func_rettype(func_oid) != agg_proc->prorettype)
	{
		/* regr_count: the catalog's final aggregate returns float8, PostgreSQL's aggregate bigint. Not offered (it used to be an error) */
		elog(DEBUG2, "the final aggregate %s does not return the type of the original aggregate: not a device aggregate",
			 format_procedure(func_oid));
		return;
	}
	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	if (proc->pronargs != 1 ||
		proc->proargtypes.dim1 != 1 ||
		proc->proargtypes.values[0] != entry->partial_func_rettype)
		elog(ERROR, "Catalog corruption? final function mismatch: %s",
			 format_procedure(func_oid));
	ReleaseSysCache(htup);
	entry->final_agg_func_oid = func_oid;

	/* final cscan function (usec by GpuAgg node) */
	if (cat->finalfn_proj_signature != NULL)
		func_oid = __aggfunc_resolve_func_signature(cat->finalfn_proj_signature, false);
	else
		func_oid = InvalidOid;
	entry->final_proj_func_oid = func_oid;
	entry->numeric_aware = cat->numeric_aware;
	entry->is_valid_entry = true;
}

static const aggfunc_catalog_entry *
aggfunc_catalog_lookup_by_oid(Oid aggfn_oid)
{
	aggfunc_catalog_entry *entry;
	bool		found;

	/* fast path by the hashtable */
	if (!aggfunc_catalog_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(aggfunc_catalog_entry);
		hctl.hcxt = CacheMemoryContext;
		aggfunc_catalog_htable = hash_create("GROUP-BY Catalog Hash by agg function",
											 256,
											 &hctl,
											 HASH_ELEM |
											 HASH_BLOBS |
											 HASH_CONTEXT);
	}
	entry = hash_search(aggfunc_catalog_htable,
						&aggfn_oid,
						HASH_ENTER,
						&found);
	if (!found)
	{
		Form_pg_proc proc;
		HeapTuple htup;

		entry->is_valid_entry = false;
		PG_TRY();
		{
			htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfn_oid));
			if (!HeapTupleIsValid(htup))
				elog(ERROR, "cache lookup failed for function %u", aggfn_oid);
			proc = (Form_pg_proc) GETSTRUCT(htup);
			if (proc->pronamespace == PG_CATALOG_NAMESPACE &&
				proc->pronargs <= 2)
			{
				char	buf[3*NAMEDATALEN+100];
				int		off;

				off = sprintf(buf, "%s(", NameStr(proc->proname));
				for (int j=0; j < proc->pronargs; j++)
				{
					Oid		type_oid = proc->proargtypes.values[j];
					char   *type_name = get_type_name(type_oid, false);

					off += sprintf(buf + off, "%s%s",
								   (j>0 ? "," : ""),
								   type_name);
				}
				off += sprintf(buf + off, ")");

				for (int i=0; aggfunc_catalog_array[i].aggfn_signature != NULL; i++)
				{
					const aggfunc_catalog_t *cat = &aggfunc_catalog_array[i];

					if (strcmp(buf, cat->aggfn_signature) == 0)
					{
						/*
						 * min/max/sum(money) share the int8 actions, but the Vulkan
						 * VM has no money type: the service would reject the query
						 * at run time instead of leaving it to the CPU.
						 */
						if (strstr(cat->aggfn_signature, "(money)") != NULL)
							break;
						/*
						 * float2 has no device type. (sum(float4) adds in float4 on the CPU: it has its own action,
						 * KAGG_ACTION__PSUM_FP32, which rounds every device addition to float4.)
						 */
						if (strstr(cat->aggfn_signature, "(float2)") != NULL)
							break;
						__setup_aggfunc_catalog_entry(entry, cat, proc);
						break;
					}
				}
			}
			ReleaseSysCache(htup);
		}
		PG_CATCH();
		{
			hash_search(aggfunc_catalog_htable, &aggfn_oid, HASH_REMOVE, NULL);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	if (!entry->is_valid_entry)
		return NULL;
	if (entry->numeric_aware && !pgstrom_enable_numeric_aggfuncs)
		return NULL;
	/* only the aggregates the Vulkan service implements */
	if (!scan_vm_agg_action_supported(entry->partial_func_action))
		return NULL;
	return entry;
}

static int
finalfunc_catalog_lookup_by_oid(Oid func_oid)
{
	finalfunc_catalog_entry *entry;
	bool		found;

	/* fast path by the hashtable */
	if (!finalfunc_catalog_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(finalfunc_catalog_entry);
		hctl.hcxt = CacheMemoryContext;
		finalfunc_catalog_htable = hash_create("GROUP-BY Catalog Hash by final function",
											   256,
											   &hctl,
											   HASH_ELEM |
											   HASH_BLOBS |
											   HASH_CONTEXT);
	}
	entry = hash_search(finalfunc_catalog_htable,
						&func_oid,
						HASH_ENTER,
						&found);
	if (!found)
	{
		Form_pg_proc proc;
		HeapTuple htup;

		entry->is_valid_entry = false;
		entry->finalfn_code = -1;
		
		htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
		if (!HeapTupleIsValid(htup))
			elog(ERROR, "cache lookup failed for function %u", func_oid);
		proc = (Form_pg_proc) GETSTRUCT(htup);
		if ((proc->pronamespace == PG_CATALOG_NAMESPACE ||
			 proc->pronamespace == get_namespace_oid("pgstrom", false)) &&
			proc->pronargs <= 2)
		{
			char	buf[3*NAMEDATALEN+100];
			int		off;

			off = sprintf(buf, "%c:%s(",
						  proc->pronamespace == PG_CATALOG_NAMESPACE ? 'c' : 's',
						  NameStr(proc->proname));
			for (int j=0; j < proc->pronargs; j++)
			{
				Oid		type_oid = proc->proargtypes.values[j];
				char   *type_name = get_type_name(type_oid, false);

				if (j > 0)
					buf[off++] = ',';
				off += sprintf(buf + off, "%s", type_name);
			}
			off += sprintf(buf + off, ")");

			for (int i=0; aggfunc_catalog_array[i].aggfn_signature != NULL; i++)
			{
				const aggfunc_catalog_t *cat = &aggfunc_catalog_array[i];

				if (cat->finalfn_proj_signature &&
					strcmp(buf, cat->finalfn_proj_signature) == 0)
				{
					entry->finalfn_code = cat->finalfn_code;
					entry->is_valid_entry = true;
					break;
				}
			}
		}
		ReleaseSysCache(htup);
	}
	return (entry->is_valid_entry ? entry->finalfn_code : -1);
}

/*
 * xpugroupby_build_path_context
 */
typedef struct
{
	bool			device_executable;
	PlannerInfo	   *root;
	UpperRelationKind upper_stage;
	RelOptInfo	   *group_rel;
	RelOptInfo	   *input_rel;
	ParamPathInfo  *param_info;
	double			num_groups;
	int				max_keys;			/* GROUP BY keys: SCAN_GROUP_KEYS for a scan, SCAN_GROUP_KEYS_JOIN over a join */
	bool			try_parallel;
	bool			has_aggfuncs;
	PathTarget	   *target_upper;
	PathTarget	   *target_partial;
	PathTarget	   *target_agg_final;
	PathTarget	   *target_proj_final;
	AggClauseCosts	final_agg_clause_costs;
	QualCost		final_proj_clause_costs;
	pgstromPlanInfo *pp_info;
	List		   *inner_paths_list;
	List		   *inner_target_list;
	List		   *groupby_keys;
	List		   *groupby_keys_refno;
	List		   *havingAggQuals;		/* only if GpuPreAgg + Agg */
	List		   *havingProjQuals;	/* only if GpuAgg */
} xpugroupby_build_path_context;

/*
 * make_expr_typecast - constructor of type cast
 */
static Expr *
make_expr_typecast(Expr *expr, Oid target_type)
{
	Oid		source_type = exprType((Node *) expr);
	HeapTuple htup;
	Form_pg_cast cast;

	if (target_type == source_type ||
		target_type == ANYOID)
		return expr;

	htup = SearchSysCache2(CASTSOURCETARGET,
						   ObjectIdGetDatum(source_type),
						   ObjectIdGetDatum(target_type));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for cast (%s -> %s)",
			 format_type_be(source_type),
			 format_type_be(target_type));
	cast = (Form_pg_cast) GETSTRUCT(htup);
	if (cast->castmethod == COERCION_METHOD_BINARY)
	{
		RelabelType    *relabel = makeNode(RelabelType);

		relabel->arg = expr;
		relabel->resulttype = target_type;
		relabel->resulttypmod = exprTypmod((Node *) expr);
		relabel->resultcollid = exprCollation((Node *) expr);
		relabel->relabelformat = COERCE_IMPLICIT_CAST;
		relabel->location = -1;

		expr = (Expr *) relabel;
	}
	else if (cast->castmethod == COERCION_METHOD_FUNCTION)
	{
		Assert(OidIsValid(cast->castfunc));
		expr = (Expr *)makeFuncExpr(cast->castfunc,
									target_type,
									list_make1(expr),
									InvalidOid,		/* always right? */
									exprCollation((Node *) expr),
									COERCE_IMPLICIT_CAST);
	}
	else
	{
		elog(ERROR, "cast-method '%c' is not supported in the kernel mode",
			 cast->castmethod);
	}
	ReleaseSysCache(htup);

	return expr;
}

/*
 * make_alternative_aggref
 *
 * It makes an alternative final aggregate function towards the supplied
 * Aggref, and append its arguments on the target_partial/target_device.
 */
static Node *
make_alternative_aggref(xpugroupby_build_path_context *con,
						Aggref *aggref,
						bool final_function_by_aggregate)
{
	const aggfunc_catalog_entry *aggfn_cat;
	PathTarget *target_partial = con->target_partial;
	pgstromPlanInfo *pp_info = con->pp_info;
	List	   *partfn_args = NIL;
	Expr	   *partfn;
	Node	   *final_fn;
	Oid			partial_func_oid;
	HeapTuple	htup;
	Form_pg_proc proc;
	ListCell   *lc;
	int32_t		j, groupby_typmod = -1;

	if (aggref->aggorder != NIL || aggref->aggdistinct != NIL)
	{
		elog(DEBUG2, "Aggregate with ORDER BY/DISTINCT is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}
	if (AGGKIND_IS_ORDERED_SET(aggref->aggkind))
	{
		elog(DEBUG2, "ORDERED SET Aggregation is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}

	/*
	 * Lookup properties of aggregate function
	 */
	aggfn_cat = aggfunc_catalog_lookup_by_oid(aggref->aggfnoid);
	if (!aggfn_cat)
	{
		elog(DEBUG2, "Aggregate function '%s' is not device executable",
			 format_procedure(aggref->aggfnoid));
		return NULL;
	}
	/* sanity checks */
	Assert(aggref->aggkind == AGGKIND_NORMAL && !aggref->aggvariadic);

	/*
	 * Build partial-aggregate function
	 */
	if (!aggref->aggfilter)
		partial_func_oid = aggfn_cat->partial_func_oid;
	else
		partial_func_oid = aggfn_cat->partial_func_filtered_oid;
	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(partial_func_oid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u",
			 partial_func_oid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	Assert(!aggref->aggfilter
		   ? proc->pronargs == list_length(aggref->args)
		   : proc->pronargs == list_length(aggref->args) + 1);
	Assert(proc->prorettype == aggfn_cat->partial_func_rettype);
	j = 0;
	foreach (lc, aggref->args)
	{
		TargetEntry *tle = lfirst(lc);
		Expr   *expr = tle->expr;
		Oid		type_oid = exprType((Node *)expr);
		Oid		dest_oid = proc->proargtypes.values[j++];

		if (type_oid != dest_oid)
			expr = make_expr_typecast(expr, dest_oid);
		if (!pgstrom_xpu_expression(expr,
									pp_info->xpu_task_flags,
									pp_info->base_relid,
									con->inner_target_list,
									NULL))
		{
			elog(DEBUG2, "Partial aggregate argument is not executable: %s",
				 nodeToString(expr));
			ReleaseSysCache(htup);
			return NULL;
		}
		partfn_args = lappend(partfn_args, expr);
		if (lc == list_head(aggref->args))
		{
			groupby_typmod = exprTypmod((Node *)expr);
			if (aggfn_cat->partial_func_action == KAGG_ACTION__PSUM_NUMERIC ||
				aggfn_cat->partial_func_action == KAGG_ACTION__PAVG_NUMERIC ||
				((aggfn_cat->partial_func_action == KAGG_ACTION__PMIN_INT64 ||
				  aggfn_cat->partial_func_action == KAGG_ACTION__PMAX_INT64) &&
				 exprType((Node *) expr) == NUMERICOID))
			{
				/*
				 * The Vulkan service sums numerics as unscaled integers of one scale: it must be known
				 * from the plan (numeric(p,s) columns and what follows from them). The typmod of the
				 * action carries it, encoded as a numeric typmod.
				 */
				int		dscale = pgstrom_numeric_static_dscale((Node *) expr);

				/* (numerics are not on the GPU inside a join program yet) */
				if (con->inner_target_list != NIL || dscale < 0 || dscale > 255)
				{
					elog(DEBUG2, "the scale of the numeric aggregate argument is not known: %s",
						 nodeToString(expr));
					ReleaseSysCache(htup);
					return NULL;
				}
				groupby_typmod = (38 << 16) + dscale + VARHDRSZ;
			}
		}
	}
	/* last argument for filtering */
	if (aggref->aggfilter)
	{
		/* the device evaluates the arguments of rows the FILTER rejects: they must not be able to raise an error */
		foreach (lc, partfn_args)
		{
			if (!pgstrom_vulkan_expr_cannot_fail((Node *) lfirst(lc)))
			{
				elog(DEBUG2, "Aggregate argument with a FILTER-clause may raise an error: %s",
					 nodeToString(lfirst(lc)));
				ReleaseSysCache(htup);
				return NULL;
			}
		}
		if (!pgstrom_xpu_expression(aggref->aggfilter,
									pp_info->xpu_task_flags,
									pp_info->base_relid,
									con->inner_target_list,
									NULL))
		{
			elog(DEBUG2, "FILTER-clause of aggregate function is not executable: %s",
				 nodeToString(aggref->aggfilter));
			ReleaseSysCache(htup);
			return NULL;
		}
		partfn_args = lappend(partfn_args, aggref->aggfilter);
	}
	partfn = (Expr *)makeFuncExpr(partial_func_oid,
								  proc->prorettype,
								  partfn_args,
								  aggref->aggcollid,
								  aggref->inputcollid,
								  COERCE_EXPLICIT_CALL);
	ReleaseSysCache(htup);
	/* see add_new_column_to_pathtarget */
	if (!list_member(target_partial->exprs, partfn))
	{
		int		__action = aggfn_cat->partial_func_action;

		if (aggref->aggfilter)
			__action |= __KAGG_ACTION__USE_FILTER;
		add_column_to_pathtarget(target_partial, partfn, 0);
		pp_info->groupby_actions = lappend_int(pp_info->groupby_actions,
											   __action);
		pp_info->groupby_typmods = lappend_int(pp_info->groupby_typmods,
											   groupby_typmod);
		pp_info->groupby_prepfn_bufsz += aggfn_cat->partial_func_bufsz;
	}

	if (final_function_by_aggregate)
	{
		/*
		 * Build final-aggregate function (for GpuPreAgg + Agg)
		 */
		Form_pg_aggregate agg;
		Aggref	   *__aggref;
		Oid			__argtype = exprType((Node *)partfn);
		TargetEntry *__tle = makeTargetEntry(partfn, 1, NULL, false);

		htup = SearchSysCache1(AGGFNOID, ObjectIdGetDatum(aggfn_cat->final_agg_func_oid));
		if (!HeapTupleIsValid(htup))
			elog(ERROR, "cache lookup failed for pg_aggregate %u", aggfn_cat->final_agg_func_oid);
		agg = (Form_pg_aggregate) GETSTRUCT(htup);

		__aggref = makeNode(Aggref);
		__aggref->aggfnoid      = aggfn_cat->final_agg_func_oid;
		__aggref->aggtype       = aggref->aggtype;
		__aggref->aggcollid     = aggref->aggcollid;
		__aggref->inputcollid   = aggref->inputcollid;
		__aggref->aggtranstype  = agg->aggtranstype;
		__aggref->aggargtypes   = list_make1_oid(__argtype);
		__aggref->aggdirectargs = NIL;	/* see sanity checks */
		__aggref->args          = list_make1(__tle);
		__aggref->aggorder      = NIL;  /* see sanity check */
		__aggref->aggdistinct   = NIL;  /* see sanity check */
		__aggref->aggfilter     = NULL; /* processed in partial-function */
		__aggref->aggstar       = false;
		__aggref->aggvariadic   = false;
		__aggref->aggkind       = AGGKIND_NORMAL;   /* see sanity check */
		__aggref->agglevelsup   = 0;
		__aggref->aggsplit      = AGGSPLIT_SIMPLE;
		__aggref->aggno         = aggref->aggno;
		__aggref->aggtransno    = aggref->aggno;
		__aggref->location      = aggref->location;
		/*
		 * MEMO: nodeAgg.c creates AggStatePerTransData for each aggtransno
		 * (that is unique ID of transition state in the Agg). This is a kind
		 * of optimization for the case when multiple aggregate function has
		 * identical transition state.
		 * However, its impact is not large for GpuPreAgg because most of
		 * reduction works are already executed at the xPU device side.
		 * So, we simply assign aggref->aggno (unique ID within the Agg node)
		 * to construct transition state for each alternative aggregate function.
		 *
		 * See the issue #614 to reproduce the problem in the future version.
		 */

		/*
		 * Update the cost factor
		 */
		if (OidIsValid(agg->aggtransfn))
			add_function_cost(con->root,
							  agg->aggtransfn,
							  NULL,
							  &con->final_agg_clause_costs.transCost);
		if (OidIsValid(agg->aggfinalfn))
			add_function_cost(con->root,
							  agg->aggfinalfn,
							  NULL,
							  &con->final_agg_clause_costs.finalCost);
		ReleaseSysCache(htup);

		final_fn = (Node *)__aggref;
	}
	else
	{
		/*
		 * Build final-scalar function (for GpuAgg)
		 */
		Oid		final_proj_func_oid = aggfn_cat->final_proj_func_oid;
		if (!OidIsValid(final_proj_func_oid))
			final_fn = (Node *)partfn;
		else
		{
			final_fn = (Node *)makeFuncExpr(final_proj_func_oid,
											aggref->aggtype,
											list_make1(partfn),
											aggref->aggcollid,
											aggref->inputcollid,
											COERCE_EXPLICIT_CALL);
			cost_qual_eval_node(&con->final_proj_clause_costs,
								final_fn, con->root);
		}
	}
	return final_fn;
}

/*
 * replace_expression_by_[agg|proj]_altfuncs
 */
static Node *
__replace_expression_by_altfunc_common(Node *node, void *__priv,
									   tree_mutator_callback walker)
{
	xpugroupby_build_path_context *con = __priv;
	RelOptInfo *input_rel = con->input_rel;
	PathTarget *target_input = input_rel->reltarget;
	ListCell   *lc;

	/* must be processed in the early half */
	Assert(node != NULL && !IsA(node, Aggref));
	/* grouping key? */
	foreach (lc, con->groupby_keys)
	{
		Expr   *key = lfirst(lc);

		if (equal(node, key))
			return copyObject(node);
	}
	/*
	 * Elsewhere, non-grouping-key columns if GROUP BY <primary key> is used,
	 * because it is equivalent to GROUP BY <all the columns>.
	 * Also, the device code don't need to understand its data type, so we
	 * have no device type checks here.
	 */
	foreach (lc, target_input->exprs)
	{
		Expr   *expr = lfirst(lc);

		if (equal(node, expr))
		{
			/*
			 * this expression shall be attached onto the target-partial later,
			 * however, it don't need to be added to the target-final, because
			 * this expression is consumed within HAVING quals, thus not exists
			 * on the final Aggregate results.
			 */
			con->groupby_keys = lappend(con->groupby_keys, expr);
			con->groupby_keys_refno = lappend_int(con->groupby_keys_refno, 0);
			return copyObject(node);
		}
	}

	if (IsA(node, Var) || IsA(node, PlaceHolderVar))
	{
		elog(ERROR, "Bug? referenced variable is grouping-key nor its dependent key: %s",
			 nodeToString(node));
	}
	return expression_tree_mutator(node, walker, con);
}

static Node *
replace_expression_by_agg_altfuncs(Node *node, void *__priv)
{
	xpugroupby_build_path_context *con = __priv;
	ListCell   *lc;

	if (!node)
		return NULL;
	if (IsA(node, Aggref))
	{
		Node   *aggfn = make_alternative_aggref(con, (Aggref *)node, true);

		if (aggfn)
			con->has_aggfuncs = true;
		else
		{
			elog(DEBUG2, "no alternative (agg) function for: %s",
				 nodeToString(node));
			con->device_executable = false;
		}
		return aggfn;
	}
	/* check whether it is grouping-key expression */
	foreach (lc, con->groupby_keys)
	{
		Node   *groupby_key = lfirst(lc);

		if (equal(groupby_key, node))
			return copyObject(node);
	}
	/* if not a grouping key, it must be an expression that eventually contains grouping key */
	if (IsA(node, Var))
	{
		con->device_executable = false;
		return NULL;
	}
	return __replace_expression_by_altfunc_common(node, __priv,
												  replace_expression_by_agg_altfuncs);
}

static Node *
replace_expression_by_proj_altfuncs(Node *node, void *__priv)
{
	xpugroupby_build_path_context *con = __priv;
	ListCell   *lc;

	if (!node)
		return NULL;
	if (IsA(node, Aggref))
	{
		Node   *aggfn = make_alternative_aggref(con, (Aggref *)node, false);

		if (aggfn)
			con->has_aggfuncs = true;
		else
		{
			elog(DEBUG2, "no alternative (proj) function for: %s",
				 nodeToString(node));
			con->device_executable = false;
		}
		return aggfn;
	}
	/* check whether it is grouping-key expression */
	foreach (lc, con->groupby_keys)
	{
		Node   *groupby_key = lfirst(lc);

		if (equal(groupby_key, node))
			return copyObject(node);
	}
	/* if not a grouping key, it must be an expression that eventually contains grouping key */
	if (IsA(node, Var))
	{
		con->device_executable = false;
		return NULL;
	}
	return __replace_expression_by_altfunc_common(node, __priv,
												  replace_expression_by_proj_altfuncs);
}

static bool
xpugroupby_build_path_target(xpugroupby_build_path_context *con)
{
	PlannerInfo	   *root = con->root;
	Query		   *parse = root->parse;
	pgstromPlanInfo *pp_info = con->pp_info;
	PathTarget	   *target_upper = con->target_upper;
	ListCell	   *cell;
	ListCell	   *lc1, *lc2;
	int				i = 0;

	/*
	 * 1st step: Picks up all grouping keys or non-key distinct clauses.
	 * These expressions are used in the next step.
	 */
	foreach (cell, target_upper->exprs)
	{
		Expr   *expr = lfirst(cell);
		Index	sortgroupref = get_pathtarget_sortgroupref(target_upper, i++);

		if (sortgroupref &&
			((parse->groupClause &&
			  get_sortgroupref_clause_noerr(sortgroupref,
											parse->groupClause) != NULL) ||
			 (parse->distinctClause &&
			  get_sortgroupref_clause_noerr(sortgroupref,
											parse->distinctClause) != NULL)))
		{
			/* Grouping Key */
			devtype_info *dtype;
			Oid		type_oid = exprType((Node *)expr);
			Oid		coll_oid;

			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype || !dtype->type_hashfunc)
			{
				elog(DEBUG2, "GROUP BY contains unsupported type (%s): %s",
					 format_type_be(type_oid),
					 nodeToString(expr));
				return false;
			}
			/*
			 * the Vulkan service groups on up to SCAN_GROUP_KEYS integer, bool or numeric (of a known scale) columns,
			 * and on text / varchar / bpchar columns of the scanned relation (not over a join): the kernel hashes and
			 * compares the bytes, which is what the equality of a deterministic collation does
			 */
			if ((!scan_vm_groupkey_type_supported(dtype->type_code) &&
				 !(dtype->type_code == TypeOpCode__numeric &&
				   con->inner_target_list == NIL &&
				   pgstrom_numeric_static_dscale((Node *) expr) >= 0 &&
				   pgstrom_numeric_static_dscale((Node *) expr) <= 255) &&
				 !((dtype->type_code == TypeOpCode__text ||
					dtype->type_code == TypeOpCode__bpchar ||
					dtype->type_code == TypeOpCode__uuid) &&
				   con->inner_target_list == NIL &&
				   pgstrom_group_string_key_ok(expr)) &&
				 /* float4 / float8 keys of the scanned relation: the program groups on x + 0.0 (-0 = +0, NaN = NaN) */
				 !(dtype->type_code == TypeOpCode__float4 ||
				   dtype->type_code == TypeOpCode__float8)) ||
				list_length(con->groupby_keys) >= con->max_keys)
			{
				elog(DEBUG2, "GROUP BY key not supported by the Vulkan service: %s",
					 nodeToString(expr));
				return false;
			}
			coll_oid = exprCollation((Node *)expr);
			if (dtype->type_code != TypeOpCode__text &&
				dtype->type_code != TypeOpCode__bpchar &&
				dtype->type_code != TypeOpCode__uuid &&
				devtype_lookup_equal_func(dtype, coll_oid) == NULL)
			{
				elog(DEBUG2, "GROUP BY contains unsupported device type (%s): %s",
					 format_type_be(type_oid),
					 nodeToString(expr));
				return false;
			}
			/* grouping-key must be device executable. */
			if (!pgstrom_xpu_expression(expr,
										pp_info->xpu_task_flags,
										pp_info->base_relid,
										con->inner_target_list,
										NULL))
			{
				elog(DEBUG2, "Grouping-key must be device executable: %s",
					 nodeToString(expr));
				return false;
			}
			con->groupby_keys = lappend(con->groupby_keys, expr);
			con->groupby_keys_refno = lappend_int(con->groupby_keys_refno,
												  sortgroupref);
		}
		else if (parse->distinctClause)
		{
			/* non-key distinct results must be a simple Var or device executable  */
			if (!IsA(expr, Var) &&
				!pgstrom_xpu_expression(expr,
										pp_info->xpu_task_flags,
										pp_info->base_relid,
										con->inner_target_list,
										NULL))
			{
				elog(DEBUG2, "Distinct output must be a smple Var or device executable: %s",
					 nodeToString(expr));
				return false;
			}
			con->groupby_keys = lappend(con->groupby_keys, expr);
			con->groupby_keys_refno = lappend_int(con->groupby_keys_refno, 0);
		}
	}
	/*
	 * 2nd step: construction of target_proj_final (that can contain CPU-only
	 * expressions refering the grouping-key or aggregation), and the earlier
	 * half of the target_agg_file from Aggref node.
	 */
	foreach (cell, target_upper->exprs)
	{
		Node   *expr = lfirst(cell);
		Node   *agg_expr = expr;	/* for Agg(CPU) + GPU-PreAgg (traditional) */
		Node   *proj_expr = expr;	/* for Projection + GPU-PreAgg (merged in device) */
		Index	sortgroupref = 0;

		/*
		 * if expression is grouping-key, sortgroupref must be set.
		 * (no need to compare them using equal(), pointer comparison is enough.
		 */
		forboth (lc1, con->groupby_keys,
				 lc2, con->groupby_keys_refno)
		{
			Node   *groupby_key = lfirst(lc1);

			if (groupby_key == expr)
			{
				sortgroupref = lfirst_int(lc2);
				goto found;
			}
		}
		agg_expr = replace_expression_by_agg_altfuncs(expr, con);
		proj_expr = replace_expression_by_proj_altfuncs(expr, con);
		if (!agg_expr || !proj_expr || !con->device_executable)
		{
			elog(DEBUG2, "unexpected expression on the upper-tlist: %s",
				 nodeToString(expr));
			return false;
		}
	found:
		add_column_to_pathtarget(con->target_agg_final, (Expr *)agg_expr, sortgroupref);
		add_column_to_pathtarget(con->target_proj_final, (Expr *)proj_expr, sortgroupref);
	}

	/*
	 * HAVING clause
	 */
	if (parse->havingQual)
	{
		Assert(IsA(parse->havingQual, List));
		con->havingAggQuals = (List *)
			replace_expression_by_agg_altfuncs(parse->havingQual, con);
		con->havingProjQuals = (List *)
			replace_expression_by_proj_altfuncs(parse->havingQual, con);
		if (con->havingAggQuals == NIL ||
			con->havingProjQuals == NIL ||
			!con->device_executable)
		{
			elog(DEBUG2, "unable to replace HAVING to alternative aggregation: %s",
				 nodeToString(parse->havingQual));
			return false;
		}
	}

	/*
	 * Due to data alignment on the tuple on the kds_final, grouping-keys must
	 * be located after the aggregate functions.
	 */
	forboth (lc1, con->groupby_keys,
			 lc2, con->groupby_keys_refno)
	{
		Expr   *key = lfirst(lc1);
		Index	keyref = lfirst_int(lc2);

		add_column_to_pathtarget(con->target_partial, key, keyref);
		pp_info->groupby_actions = lappend_int(pp_info->groupby_actions,
											   keyref == 0
											   ? KAGG_ACTION__VREF_NOKEY
											   : KAGG_ACTION__VREF);
		pp_info->groupby_typmods = lappend_int(pp_info->groupby_typmods,
											   exprType((Node *)key) == NUMERICOID
											   ? (38 << 16) + pgstrom_numeric_static_dscale((Node *) key) + VARHDRSZ
											   : exprTypmod((Node *)key));
	}
	set_pathtarget_cost_width(root, con->target_proj_final);
	set_pathtarget_cost_width(root, con->target_agg_final);
	set_pathtarget_cost_width(root, con->target_partial);

	return true;
}

/*
 * try_add_final_groupby_paths
 */
static void
try_add_final_groupby_paths(xpugroupby_build_path_context *con, Path *part_path)
{
	Query	   *parse = con->root->parse;
	Path	   *__path;

	if (parse->groupClause)
	{
		Assert(grouping_is_hashable(parse->groupClause));
		__path = (Path *)create_agg_path(con->root,
										 con->group_rel,
										 part_path,
										 con->target_agg_final,
										 AGG_HASHED,
										 AGGSPLIT_SIMPLE,
										 parse->groupClause,
										 con->havingAggQuals,
										 &con->final_agg_clause_costs,
										 con->num_groups);
		if (con->has_aggfuncs)
			__path = pgstrom_create_dummy_path(con->root, __path);
		add_path(con->group_rel, __path);
	}
	else if (parse->distinctClause)
	{
		Assert(grouping_is_hashable(parse->distinctClause));
		__path = (Path *)create_agg_path(con->root,
										 con->group_rel,
										 part_path,
										 con->target_agg_final,
										 AGG_HASHED,
										 AGGSPLIT_SIMPLE,
										 parse->distinctClause,
										 con->havingAggQuals,
										 &con->final_agg_clause_costs,
										 con->num_groups);
		Assert(!con->has_aggfuncs);
		add_path(con->group_rel, __path);
	}
	else
	{
		__path = (Path *)create_agg_path(con->root,
										 con->group_rel,
										 part_path,
										 con->target_agg_final,
										 AGG_PLAIN,
										 AGGSPLIT_SIMPLE,
										 parse->groupClause,
										 con->havingAggQuals,
										 &con->final_agg_clause_costs,
										 con->num_groups);
		if (con->has_aggfuncs)
			__path = pgstrom_create_dummy_path(con->root, __path);
		add_path(con->group_rel, __path);
	}
}

/*
 * __buildXpuPreAggCustomPath
 */
static CustomPath *
__buildXpuPreAggCustomPath(xpugroupby_build_path_context *con)
{
	Query	   *parse = con->root->parse;
	CustomPath *cpath = makeNode(CustomPath);
	PathTarget *target_partial = con->target_partial;
	pgstromPlanInfo *pp_info = copy_pgstrom_plan_info(con->pp_info);
	double		input_nrows = PP_INFO_NUM_ROWS(pp_info);
	double		num_group_keys;

	pp_info->xpu_task_flags &= ~DEVTASK__MASK;
	if (parse->groupClause || parse->distinctClause)
		pp_info->xpu_task_flags |= (DEVTASK__PREAGG | DEVTASK__PINNED_HASH_RESULTS);
	else
		pp_info->xpu_task_flags |= (DEVTASK__PREAGG | DEVTASK__PINNED_ROW_RESULTS);
	/* TODO: more precise cost factors */
	pp_info->final_nrows = con->num_groups;

	/* No tuples shall be generated until child JOIN/SCAN path completion */
	pp_info->startup_cost = (pp_info->startup_cost +
							 pp_info->inner_cost +
							 pp_info->run_cost);
	/*
	 * The run cost of a GpuJoin contains the DMA receive of the joined rows (gpu_tuple_cost per
	 * row); with the aggregation on the device they never leave it.
	 */
	if (pp_info->num_inner_rels > 0)
	{
		pp_info->startup_cost -= (pgstrom_gpu_tuple_cost * input_nrows /
								  pp_info->parallel_divisor);
		if (pp_info->startup_cost < 0.0)
			pp_info->startup_cost = 0.0;
	}
	/* Cost estimation for grouping */
	num_group_keys = list_length(parse->groupClause);
	pp_info->startup_cost += (pgstrom_gpu_operator_cost *
							  num_group_keys *
							  input_nrows);
	/* Cost estimation for aggregate function */
	pp_info->startup_cost += (target_partial->cost.per_tuple * input_nrows +
							  target_partial->cost.startup) * pgstrom_gpu_operator_ratio();
	/* Cost for DMA receive (xPU --> Host) */
	pp_info->run_cost = (con->target_partial->cost.per_tuple +
						 pgstrom_gpu_tuple_cost) * con->num_groups / pp_info->parallel_divisor;
	pp_info->final_cost = 0.0;

	cpath->path.pathtype         = T_CustomScan;
	cpath->path.parent           = con->input_rel;
	cpath->path.pathtarget       = con->target_partial;
	cpath->path.param_info       = con->param_info;
	cpath->path.parallel_aware   = con->try_parallel;
	cpath->path.parallel_safe    = con->input_rel->consider_parallel;
	cpath->path.parallel_workers = pp_info->parallel_nworkers;
	cpath->path.rows             = con->num_groups;
	cpath->path.startup_cost     = pp_info->startup_cost;
	cpath->path.total_cost       = (pp_info->startup_cost +
									pp_info->run_cost +
									pp_info->final_cost);
	cpath->path.pathkeys         = NIL;
	cpath->flags                 = 0;	/* prevent Result pushdown before plan creation */
	cpath->custom_paths          = con->inner_paths_list;
	cpath->custom_private        = list_make3(pp_info, NULL, NULL);
	cpath->methods               = &gpupreagg_path_methods;
	return cpath;
}

/*
 * lookup_gpusort_keykind
 */
static int
lookup_gpusort_keykind(Node *f_expr, PathTarget *part_target, int *p_action)
{
	ListCell	   *lc;

	*p_action = 0;

	/* whether it is a final function? */
	if (IsA(f_expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *)f_expr;
		Node	   *farg = NULL;
		Oid			fnsp;
		char		namebuf[NAMEDATALEN * 2 + 64];
		int			off = 0;

		if (list_length(func->args) > 1)
			goto skip;
		fnsp = get_func_namespace(func->funcid);
		if (list_length(func->args) == 1)
			farg = linitial(func->args);
		/* setup signature */
		if (fnsp == PG_CATALOG_NAMESPACE)
			off += sprintf(namebuf+off, "c:");
		else if (fnsp == get_namespace_oid("pgstrom", false))
			off += sprintf(namebuf+off, "s:");
		else
			goto skip;
		off += sprintf(namebuf+off, "%s(", get_func_name(func->funcid));
		if (farg)
		{
			char   *type_name = get_type_name(exprType(farg), true);

			if (type_name)
				off += sprintf(namebuf+off, "%s", type_name);
		}
		off += sprintf(namebuf+off, ")");
		/* lookup catalog */
		//TODO: make a hash table
		for (int i=0; aggfunc_catalog_array[i].aggfn_signature != NULL; i++)
		{
			const char *signature = aggfunc_catalog_array[i].finalfn_proj_signature;

			/*
			 * count(*) does not use CPU-final function (final-projection == NULL),
			 * and it is identical to the partial function nrows(). in this case,
			 * we use signature of the partial function.
			 */
			if (!signature)
				signature = aggfunc_catalog_array[i].partfn_signature;
			if (strcmp(namebuf, signature) == 0)
			{
				*p_action = aggfunc_catalog_array[i].partfn_action;
				return aggfunc_catalog_array[i].gpusort_keykind;
			}
		}
	}
skip:
	/* elsewhere, f_expr exactly matches any of part_target? */
	foreach (lc, part_target->exprs)
	{
		Node   *p_expr = lfirst(lc);

		if (equal(f_expr, p_expr))
			return KSORT_KEY_KIND__VREF;
	}
	return -1;
}

/*
 * consider_sorted_groupby_path
 */
#define LOG2(x)		(log(x) / 0.693147180559945)
static bool
consider_sorted_groupby_path(PlannerInfo *root,
							 CustomPath *cpath,
							 PathTarget *upper_target,
							 PathTarget *final_target,
							 double ntuples,
							 List **p_sortkeys_upper,
							 List **p_sortkeys_expr,
							 List **p_sortkeys_kind,
							 List **p_sortkeys_refs,
							 Cost *p_gpusort_cost)
{
	pgstromPlanInfo	*pp_info = linitial(cpath->custom_private);
	Cost		comparison_cost = (2.0 * pgstrom_gpu_operator_cost);
	List	   *sortkeys_upper = NIL;
	List	   *sortkeys_expr = NIL;
	List	   *sortkeys_kind = NIL;
	List	   *sortkeys_refs = NIL;
	List	   *inner_target_list = NIL;
	ListCell   *lc1, *lc2, *lc3, *lc4;

	if (!pgstrom_enable_gpusort)
	{
		elog(DEBUG1, "gpusort: disabled by pg_strom.enable_gpusort");
		return false;
	}
	if (pgstrom_cpu_fallback_elevel < ERROR)
	{
		elog(DEBUG1, "gpusort: disabled by pg_strom.cpu_fallback");
		return false;
	}
	if ((pp_info->xpu_task_flags & DEVKIND__NVIDIA_GPU) == 0)
	{
		elog(DEBUG1, "gpusort: disabled, because only GPUs are supported (flags: %08x)",
			 pp_info->xpu_task_flags);
		return false;	/* feture available on GPU only */
	}
	/* pick up upper sortkeys */
	if (root->window_pathkeys != NIL)
		sortkeys_upper = root->window_pathkeys;
	else if (root->distinct_pathkeys != NIL)
		sortkeys_upper = root->distinct_pathkeys;
	else if (root->sort_pathkeys != NIL)
		sortkeys_upper = root->sort_pathkeys;
	else if (root->query_pathkeys != NIL &&
			 root->query_pathkeys != root->group_pathkeys)
		sortkeys_upper = root->query_pathkeys;
	else
	{
		elog(DEBUG1, "gpusort: disabled because no sortable pathkeys");
		return false;
	}
	/* preparation for pgstrom_xpu_expression */
	foreach (lc1, cpath->custom_paths)
	{
		inner_target_list = lappend(inner_target_list,
									((Path *)lfirst(lc1))->pathtarget);
	}

	foreach (lc1, sortkeys_upper)
	{
		PathKey	   *pk = lfirst(lc1);
		EquivalenceClass *ec = pk->pk_eclass;

		/* walk on the ec_members to find out identical key expression */
		foreach (lc2, ec->ec_members)
		{
			EquivalenceMember *em = lfirst(lc2);
			Expr   *em_expr = em->em_expr;

			/* strip Relabel for equal() comparison */
			while (IsA(em_expr, RelabelType))
				em_expr = ((RelabelType *)em_expr)->arg;
			/* ok, lookup the sorting key */
			forboth (lc3, upper_target->exprs,
					 lc4, final_target->exprs)
			{
				Node   *u_expr = lfirst(lc3);
				Node   *f_expr = lfirst(lc4);

				if (equal(u_expr, em_expr))
				{
					int		action;
					int		kind = lookup_gpusort_keykind(f_expr, cpath->path.pathtarget, &action);

					if (kind < 0)
					{
						elog(DEBUG1, "gpusort: key expression is not supported: %s",
							 nodeToString(f_expr));
						continue;
					}
					/*
					 * A GROUP BY key (action 0) must be device executable; the Vulkan service sorts
					 * the integer representation of bool, int1-8 and the date/time types, so the
					 * device compare handler of the type is not needed. count(*)/count(x) are VREF
					 * kinds of an aggregate action: nothing to check.
					 */
					if (kind == KSORT_KEY_KIND__VREF && action == 0 &&
						!pgstrom_xpu_expression((Expr *)f_expr,
												pp_info->xpu_task_flags,
												pp_info->base_relid,
												inner_target_list,
												NULL))
					{
						elog(DEBUG1, "gpusort: key expression is not device executable: %s",
							 nodeToString(f_expr));
						continue;
					}
					/* the Vulkan service sorts on a subset of the keys (the others stay on the CPU) */
					{
						devtype_info *kdtype = pgstrom_devtype_lookup(exprType((Node *)em_expr));

						if (!scan_vm_gpusort_key_supported(kind, action,
														   kdtype ? kdtype->type_code : 0))
						{
							elog(DEBUG1, "gpusort: key is not supported by the Vulkan service: %s",
								 nodeToString(f_expr));
							continue;
						}
					}
					if (__PathKeyUseNullsFirstCompare(pk))
						kind |= KSORT_KEY_ATTR__NULLS_FIRST;
					if (__PathKeyUseLessThanCompare(pk))
						kind |= KSORT_KEY_ATTR__ORDER_ASC;
					else if (!__PathKeyUseGreaterThanCompare(pk))
					{
						elog(DEBUG1, "Bug? PathKey has neither ASC nor DESC sorting order");
						return false;	/* should not happen */
					}
					/* ok, found the sorting key */
					sortkeys_expr = lappend(sortkeys_expr, f_expr);
					sortkeys_kind = lappend_int(sortkeys_kind, kind);
					sortkeys_refs = lappend_int(sortkeys_refs, ec->ec_sortref);
					goto found_one_key;
				}
			}
		}
		elog(DEBUG1, "gpusort: sort-key was not found in the result set");
		return false;	/* not found */
	found_one_key:
		Assert(list_length(sortkeys_expr) == list_length(sortkeys_kind) &&
			   list_length(sortkeys_expr) == list_length(sortkeys_refs));
	}
	*p_sortkeys_upper = sortkeys_upper;
	*p_sortkeys_expr  = sortkeys_expr;
	*p_sortkeys_kind  = sortkeys_kind;
	*p_sortkeys_refs  = sortkeys_refs;
	*p_gpusort_cost   = 10.0 + comparison_cost * ntuples * LOG2(ntuples);
	return true;
}

/*
 * tryAddGpuSortedGroupByPath
 *
 * GPU-Sort for GROUP BY ... ORDER BY / window-rank: the service returns each group once, final and in
 * sort order, so the path is Projection (final functions on the CPU) -> GpuPreAgg without an Aggregate
 * node. That needs a single session to see every row, so it is built for the non-parallel path only
 * (parallel workers have a session each and PostgreSQL's Aggregate combines them), and the service
 * fails the query if the CPU would have had to evaluate rows itself (pg_strom.cpu_fallback must be
 * 'off': consider_sorted_groupby_path checks that).
 */
static void
tryAddGpuSortedGroupByPath(xpugroupby_build_path_context *con, CustomPath *cpath_agg,
						   double num_groups, bool be_parallel)
{
	PlannerInfo *root = con->root;
	Query	   *parse = root->parse;
	pgstromPlanInfo *pp_info;
	CustomPath *cpath;
	Path	   *__path;
	List	   *sortkeys_upper = NIL;
	List	   *sortkeys_expr = NIL;
	List	   *sortkeys_kind = NIL;
	List	   *sortkeys_refs = NIL;
	Cost		gpusort_cost = 0.0;
	Cost		per_tuple;

	if (be_parallel || !pgstrom_enable_gpusort)
		return;
	/*
	 * A13: a geometry GiST join sets undecidable outer rows aside as KDS_FORMAT_RECHECK rows that the backend
	 * joins and aggregates on the CPU, and GPU-Sort needs the service to see every row (it refuses RECHECK).
	 */
	{
		pgstromPlanInfo *pp_join = linitial(cpath_agg->custom_private);

		for (int i=0; i < pp_join->num_inner_rels; i++)
		{
			if (OidIsValid(pp_join->inners[i].gist_func_oid))
			{
				elog(DEBUG2, "GpuPreAgg: GPU-Sort is not used over a GiST join with a geometry qual");
				return;
			}
			/* the unmatched inner rows are emitted by the backend after the sorted groups */
			if (pp_join->inners[i].join_type == JOIN_RIGHT ||
				pp_join->inners[i].join_type == JOIN_FULL)
			{
				elog(DEBUG2, "GpuPreAgg: GPU-Sort is not used over a RIGHT/FULL join");
				return;
			}
		}
	}
	if (!parse->groupClause || parse->distinctClause || con->havingProjQuals != NIL)
		return;
	if (!consider_sorted_groupby_path(root,
									  cpath_agg,
									  con->group_rel->reltarget,
									  con->target_proj_final,
									  num_groups,
									  &sortkeys_upper,
									  &sortkeys_expr,
									  &sortkeys_kind,
									  &sortkeys_refs,
									  &gpusort_cost))
		return;
	per_tuple = (cpath_agg->path.pathtarget->cost.per_tuple + pgstrom_gpu_tuple_cost);
	cpath = (CustomPath *)pgstrom_copy_pathnode(&cpath_agg->path);
	pp_info = copy_pgstrom_plan_info(linitial(cpath_agg->custom_private));
	pp_info->xpu_task_flags |= DEVTASK__MERGE_FINAL_BUFFER;
	pp_info->gpusort_keys_expr = sortkeys_expr;
	pp_info->gpusort_keys_kind = sortkeys_kind;
	pp_info->gpusort_keys_refs = sortkeys_refs;
	linitial(cpath->custom_private) = pp_info;
	lsecond(cpath->custom_private) = con->target_proj_final;
	lthird(cpath->custom_private) = con->havingProjQuals;
	cpath->path.startup_cost = (cpath->path.total_cost
								- per_tuple * cpath->path.rows / 2.0
								+ gpusort_cost);
	cpath->path.total_cost = (cpath->path.startup_cost
							  + per_tuple * cpath->path.rows / 2.0);
	/* the projection computes the final functions of the aggregates from the partial states */
	__path = (Path *)create_projection_path(root,
											con->group_rel,
											&cpath->path,
											con->target_proj_final);
	__path->pathkeys = sortkeys_upper;
	if (con->has_aggfuncs)
		__path = pgstrom_create_dummy_path(root, __path);
	add_path(con->group_rel, __path);
}

/*
 * tryAddSimpleGpuPreAggPath
 */
static void
tryAddSimpleGpuPreAggPath(PlannerInfo *root,
						  UpperRelationKind upper_stage,
						  RelOptInfo *group_rel,
						  RelOptInfo *input_rel,
						  CustomPath *input_path,
						  GroupPathExtraData *gp_extra,
						  bool be_parallel)
{
	xpugroupby_build_path_context con;
	pgstromPlanInfo *pp_input = linitial(input_path->custom_private);
	Query	   *parse = root->parse;
	List	   *inner_target_list = NIL;
	ListCell   *lc;
	CustomPath *cpath;
	double		num_groups = 1.0;
	bool		heap_scan;

	/*
	 * GpuPreAgg on top of a GpuJoin: the join kernel aggregates the joined rows on the device
	 * (jagg.slang / jgagg.slang). RIGHT and FULL joins work too: the kernels flag the matched inner rows
	 * in the outer-join map, and after the final reply the backend joins each unmatched inner row with
	 * NULL outer columns on the CPU and projects it to a partial-aggregate row (the poison path of a
	 * GiST join uses the same code). The combination with a GiST join stays on the CPU: its undecidable
	 * outer rows are joined by the backend after the service replied, and those set map flags too.
	 */
	if (input_path->custom_paths != NIL)
	{
		bool		has_right = false;
		bool		has_gist = false;

		for (int i=0; i < pp_input->num_inner_rels; i++)
		{
			JoinType	jtype = pp_input->inners[i].join_type;

			if (jtype == JOIN_RIGHT || jtype == JOIN_FULL)
				has_right = true;
			if (OidIsValid(pp_input->inners[i].gist_index_oid) ||
				OidIsValid(pp_input->inners[i].gist_func_oid))
				has_gist = true;
		}
		if (has_right && has_gist)
		{
			elog(DEBUG2, "GpuPreAgg: aggregation over a RIGHT/FULL join with a GiST join stays on the CPU");
			return;
		}
	}
	elog(DEBUG2, "GpuPreAgg: trying on top of %s (%d inner relations, parallel=%d)",
		 pp_input->num_inner_rels > 0 ? "GpuJoin" : "GpuScan",
		 pp_input->num_inner_rels, (int) be_parallel);
	/* estimate number of groups */
	if (parse->groupClause)
	{
		List   *groupExprs;

		groupExprs = get_sortgrouplist_exprs(parse->groupClause,
											 gp_extra->targetList);
		num_groups = estimate_num_groups(root, groupExprs,
										 pp_input->final_nrows,
										 NULL, NULL);
	}
	else if (parse->distinctClause)
	{
		List   *distinctExprs;

		distinctExprs = get_sortgrouplist_exprs(parse->distinctClause,
												parse->targetList);
		num_groups = estimate_num_groups(root, distinctExprs,
										 pp_input->final_nrows,
										 NULL, NULL);
	}
	/*
	 * A GROUP BY over a scan (heap, Arrow_Fdw, GPU-Cache) keeps its groups in one hash table in device memory for the whole query
	 * (gtab.slang) and scales to many groups. A join below still uses the workgroups' tables in shared memory; with many more groups than fit, most rows would be aggregated on the CPU, so leave such queries to
	 * PostgreSQL.
	 */
	heap_scan = (input_path->custom_paths == NIL);
	{
		int			limit;

		limit = heap_scan ? pgstrom_gpupreagg_max_groups_heap : pgstrom_gpupreagg_max_groups;
		if (parse->groupClause && num_groups > limit)
		{
			elog(DEBUG2, "GpuPreAgg: estimated %.0f groups exceed pg_strom.gpupreagg_max_groups%s",
				 num_groups, heap_scan ? "_heap" : "");
			return;
		}
	}
	/* setup inner_target_list */
	foreach (lc, input_path->custom_paths)
	{
		Path	   *i_path = lfirst(lc);
		inner_target_list = lappend(inner_target_list, i_path->pathtarget);
	}
	/* setup context */
	memset(&con, 0, sizeof(con));
	con.device_executable = true;
	con.root           = root;
	con.upper_stage    = upper_stage;
	con.group_rel      = group_rel;
	con.input_rel      = input_rel;
	con.param_info     = input_path->path.param_info;
	con.num_groups     = num_groups;
	con.max_keys       = heap_scan ? SCAN_GROUP_KEYS : SCAN_GROUP_KEYS_JOIN;
	con.try_parallel   = be_parallel;
	con.target_upper   = root->upper_targets[upper_stage];
	con.target_partial = create_empty_pathtarget();
	con.target_agg_final = create_empty_pathtarget();
	con.target_proj_final = create_empty_pathtarget();
	con.pp_info        = pp_input;
	con.inner_paths_list = input_path->custom_paths;
	con.inner_target_list = inner_target_list;
	/* construction of the target-list for each level */
	if (!xpugroupby_build_path_target(&con))
		return;
	/*
	 * Limits of the Vulkan service (scan_caps.h): a program it cannot translate fails the query.
	 * The partial result has at most SCAN_AGG_MAX columns (keys and partial aggregates), and the
	 * kernel variables (referenced columns of the outer side and of every inner relation, plus the
	 * computed arguments and keys) must fit its registers.
	 */
	{
		/*
		 * kernel variables: the referenced columns of every relation, and one slot per computed
		 * argument or filter of a partial aggregate (a plain column argument uses its column's slot)
		 */
		List	   *pvars = list_concat_unique(NIL, pull_var_clause((Node *) con.target_partial->exprs, 0));
		int			nslots = list_length(pvars);	/* columns the aggregates and keys read */
		int			nexprs = 0;		/* computed arguments/filters/keys: a slot each, and a register that stays */
		List	   *seen = NIL;
		ListCell   *lc3;
		int			bit = -1;

		/* plus the columns only the lower node's quals and join keys read (not listed in pvars) */
		while ((bit = bms_next_member(pp_input->outer_refs, bit)) >= 0)
		{
			AttrNumber	attno = bit + FirstLowInvalidHeapAttributeNumber;
			bool		found = false;

			foreach (lc, pvars)
			{
				Var	   *v = lfirst(lc);

				if (v->varno == pp_input->base_relid && v->varattno == attno)
				{
					found = true;
					break;
				}
			}
			if (!found)
				nslots++;
		}
		foreach (lc, inner_target_list)
		{
			ListCell   *lc5;

			foreach (lc5, ((PathTarget *) lfirst(lc))->exprs)
				if (!list_member(pvars, lfirst(lc5)))
					nslots++;
		}
		foreach (lc3, con.target_partial->exprs)
		{
			Node   *e = lfirst(lc3);
			List   *cands = (IsA(e, FuncExpr) ? ((FuncExpr *) e)->args : list_make1(e));
			ListCell *lc4;

			/* equal expressions share one slot and one register (codegen reuses them) */
			foreach (lc4, cands)
			{
				Node   *a = lfirst(lc4);

				if (IsA(a, Var) || list_member(seen, a))
					continue;
				seen = lappend(seen, a);
				nexprs++;
			}
		}
		/* registers: the slots, one more per computed value (scan_prog.c keeps them above the slots), temporaries */
		nslots += nexprs;
		if (!pgstrom_debug_no_limits &&
			(list_length(con.pp_info->groupby_actions) > SCAN_AGG_MAX ||
			 list_length(con.target_partial->exprs) > SCAN_AGG_MAX ||
			 nslots + nexprs + SCAN_VM_TEMP_BUDGET > SCAN_MAX_REGS))
		{
			elog(DEBUG2, "GpuPreAgg: too many columns for the Vulkan service (%d output columns, %d kernel variables)",
				 list_length(con.target_partial->exprs), nslots);
			return;
		}
	}
	/*
	 * float aggregates with GROUP BY (A10) run in the shared-memory kernels (the slot owners add the rows in order): at most
	 * SCAN_GROUP_KEYS_SHARED keys, and the groups must fit the workgroups' tables; the session-wide table of gtab.slang
	 * (many groups, three or four keys) has no float accumulators
	 */
	if (parse->groupClause)
	{
		ListCell   *lcf;
		bool		has_float = false;

		foreach (lcf, con.pp_info->groupby_actions)
		{
			switch (lfirst_int(lcf))
			{
				case KAGG_ACTION__PMIN_FP64:
				case KAGG_ACTION__PMAX_FP64:
				case KAGG_ACTION__PSUM_FP:
				case KAGG_ACTION__PAVG_FP:
				case KAGG_ACTION__PSUM_FP32:
				case KAGG_ACTION__STDDEV:
				case KAGG_ACTION__COVAR:
					has_float = true;
					break;
				default:
					break;
			}
		}
		if (has_float &&
			(num_groups > pgstrom_gpupreagg_max_groups ||
			 list_length(con.groupby_keys) > (heap_scan ? SCAN_GROUP_KEYS_SHARED : SCAN_GROUP_KEYS_JOIN)))
		{
			elog(DEBUG2, "GpuPreAgg: float aggregates with GROUP BY need at most %d keys and %d estimated groups (%.0f groups, %d keys)",
				 heap_scan ? SCAN_GROUP_KEYS_SHARED : SCAN_GROUP_KEYS_JOIN, pgstrom_gpupreagg_max_groups, num_groups, list_length(con.groupby_keys));
			return;
		}
	}
	/* build GpuPreAgg path */
	cpath = __buildXpuPreAggCustomPath(&con);
	elog(DEBUG2, "GpuPreAgg: path built, startup %.2f total %.2f rows %.0f",
		 cpath->path.startup_cost, cpath->path.total_cost, cpath->path.rows);

	/* Agg(CPU) [+ Gather] + GpuPreAgg,
	 * If CPU fallback may happen, because a part of results are kept in
	 * CPU-side, thus GPU-Service cannot generate final aggregation result
	 * by itself, or
	 * If no-grouping aggregation, because it must generate an empty result
	 * when input set is empty. The no-grouping aggregation is a relatively
	 * lightweight operation because CPU processes up to one tuple, so we
	 * can ignore this overhead.
	 * Elsewhere, no CPU-based Agg-node is needed.
	 */
	/*
	 * The Vulkan service returns partial aggregate states only, and PostgreSQL's own
	 * Agg node always combines and finalizes them (the service has no final buffer).
	 */
	if (true)
	{
		Path   *__path = &cpath->path;

		if (be_parallel)
		{
			__path = (Path *)
				create_gather_path(root,
								   group_rel,
								   __path,
								   __path->pathtarget,
								   NULL,
								   &num_groups);
		}
		try_add_final_groupby_paths(&con, __path);
	}

	/*
	 * GPU-Sort: the same GpuPreAgg path, with the final groups sorted by the service and no Aggregate
	 * above (non-parallel only: the service keeps the groups of one session, see tryAddGpuSortedGroupByPath).
	 */
	tryAddGpuSortedGroupByPath(&con, cpath, num_groups, be_parallel);
}

static void
tryGpuPreAggAddCustomPath(PlannerInfo *root,
						  UpperRelationKind upper_stage,
						  RelOptInfo *input_rel,
						  RelOptInfo *group_rel,
						  void *extra)
{
	Query	   *parse = root->parse;

	/* quick bailout if not supported */
	if (parse->groupingSets != NIL)
	{
		elog(DEBUG2, "GROUPING SET is not supported");
		return;
	}
	else if (!grouping_is_hashable(parse->groupClause) ||
			 !grouping_is_hashable(parse->distinctClause))
	{
		elog(DEBUG2, "GROUPING KEY is not hashable");
		return;
	}
	else if (parse->distinctClause && (parse->groupClause ||
									   parse->groupingSets ||
									   parse->hasAggs ||
									   root->hasHavingQual))
	{
		elog(DEBUG2, "Unable to use GROUP BY and DISTINCT together");
		return;
	}

	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		CustomPath *input_path;

		input_path = pgstrom_find_custom_path(root,
											  input_rel,
											  try_parallel > 0);
		if (!input_path)
			continue;
		/* an Arrow_Fdw / Parquet input is aggregated by the service as well (arrow_exec.c) */
		/*
		 * unless pg_strom.enable_partitionwise_gpupreagg is enabled,
		 * we don't make a path for partitioned outer relation.
		 */
		if (!pgstrom_enable_partitionwise_gpupreagg)
		{
			pgstromPlanInfo *pp_input = linitial(input_path->custom_private);
			if ((pp_input->xpu_task_flags & DEVTASK__IS_PARTITION_WISE) != 0)
				continue;
		}
		/* try add GPU-PreAgg path */
		tryAddSimpleGpuPreAggPath(root,
								  upper_stage,
								  group_rel,
								  input_rel,
								  input_path,
								  extra,
								  try_parallel > 0);
	}
}

/*
 * __duplicateWindowAggPath
 */
static Path *
__duplicateWindowAggPath(PlannerInfo *root,
						 WindowAggPath *wpath,
						 Path *subpath)
{
	Query		   *parse = root->parse;
	WindowClause   *__wc = wpath->winclause;
	WindowFuncLists *wflists
		= find_window_functions((Node *)root->processed_tlist,
								list_length(parse->windowClause));
	return (Path *)create_windowagg_path(root,
										 wpath->path.parent,
										 subpath,
										 wpath->path.pathtarget,
										 wflists->windowFuncs[__wc->winref],
										 wpath->runCondition,
										 __wc,
										 wpath->qual,
										 wpath->topwindow);
}

/*
 * tryGpuSortWithLimitPath
 */
static Path *
tryGpuSortWithLimitPath(PlannerInfo *root,
						FinalPathExtraData *extra,
						Path *path,
						bool be_parallel)
{
	Path   *subpath;

	switch (path->type)
	{
		case T_LimitPath: {
			LimitPath  *lpath = (LimitPath *)path;

			if (extra->limit_needed &&
				extra->limit_tuples > 0.0)
			{
				subpath = tryGpuSortWithLimitPath(root, extra, lpath->subpath, be_parallel);
				if (subpath)
				{
					return (Path *)create_limit_path(root,
													 lpath->path.parent,
													 subpath,
													 lpath->limitOffset,
													 lpath->limitCount,
													 lpath->limitOption,
													 extra->offset_est,
													 extra->count_est);
				}
			}
			break;
		}
		case T_GatherPath: {
			GatherPath *gpath = (GatherPath *)path;

			subpath = tryGpuSortWithLimitPath(root, extra, gpath->subpath, true);
			if (subpath)
			{
				GatherPath *result
					= create_gather_path(root,
										 path->parent,
										 subpath,
										 path->pathtarget,
										 NULL,
										 &subpath->rows);
				result->path.pathkeys = path->pathkeys;
				return &result->path;
			}
			break;
		}
		case T_GatherMergePath: {
			GatherMergePath *gmpath = (GatherMergePath *)path;

			subpath = tryGpuSortWithLimitPath(root, extra, gmpath->subpath, true);
			if (subpath)
			{
				GatherPath *result
					= create_gather_path(root,
										 path->parent,
										 subpath,
										 path->pathtarget,
										 NULL,
										 &subpath->rows);
				result->path.pathkeys = path->pathkeys;
				return &result->path;
			}
			break;
		}
		case T_ProjectionPath: {
			ProjectionPath *pjpath = (ProjectionPath *)path;

			subpath = tryGpuSortWithLimitPath(root, extra, pjpath->subpath, be_parallel);
			if (subpath)
			{
				ProjectionPath *result
					= create_projection_path(root,
											 path->parent,
											 subpath,
											 path->pathtarget);
				result->path.pathkeys = path->pathkeys;
				return &result->path;
			}
			break;
		}
		case T_AppendPath:
			if (IS_PARTITIONED_REL(path->parent))
			{
				CustomPath *cpath = pgstrom_find_custom_path(root,
															 path->parent,
															 be_parallel);
				if (cpath)
					path = &cpath->path;
			}
			__attribute__ ((fallthrough));
		case T_CustomPath:
			if (pgstrom_is_dummy_path(path))
			{
				CustomPath *cpath = (CustomPath *)path;

				subpath = linitial(cpath->custom_paths);
				subpath = tryGpuSortWithLimitPath(root, extra, subpath, be_parallel);
				if (subpath)
					return pgstrom_create_dummy_path(root, subpath);
			}
			else if (pgstrom_is_gpuscan_path(path) ||
					 pgstrom_is_gpujoin_path(path) ||
					 pgstrom_is_gpupreagg_path(path))
			{
				CustomPath *cpath = (CustomPath *)path;
				pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
				Cost		__startup_cost;
				Cost		__per_tuple = (cpath->path.pathtarget->cost.per_tuple +
										   pgstrom_gpu_tuple_cost);
				if (pp_info->gpusort_keys_expr == NIL ||
					pp_info->gpusort_keys_kind == NIL)
					return NULL;	/* no sort */
				if (extra->limit_tuples < 0.0 ||
					extra->limit_tuples > (double)INT_MAX)
					return NULL;	/* makes no sense */
				/* duplicate CustomPath */
				cpath = pmemdup(cpath, sizeof(CustomPath));
				cpath->custom_private = list_copy(cpath->custom_private);
				pp_info = copy_pgstrom_plan_info(pp_info);
				pp_info->gpusort_limit_count = extra->limit_tuples;
				linitial(cpath->custom_private) = pp_info;
				/* adjust the final cost factor */
				__startup_cost = cpath->path.total_cost
					- __per_tuple * cpath->path.rows
					+ __per_tuple * extra->limit_tuples / 2.0;
				cpath->path.startup_cost = __startup_cost;
				cpath->path.total_cost = __startup_cost
					+ __per_tuple * extra->limit_tuples / 2.0;
				cpath->path.rows = extra->limit_tuples;
				return &cpath->path;
			}
			break;

		default:
			break;
	}
	return NULL;
}
#if 0
static Path *
tryGpuSortWithLimitPath(PlannerInfo *root, FinalPathExtraData *extra, Path *path)
{
	Query	   *parse = root->parse;
	Const	   *con = (Const *)parse->limitCount;

	if (con && IsA(con, Const) &&
		con->consttype == INT8OID &&
		!con->constisnull)
	{
		int64_t		limit_count = DatumGetInt64(con->constvalue);

		if (parse->limitOffset)
		{
			con = (Const *)parse->limitOffset;
			if (IsA(con, Const) &&
				con->consttype == INT8OID &&
				!con->constisnull)
				limit_count += DatumGetInt64(con->constvalue);
			else
				return NULL;
		}

		if (limit_count > 0 && limit_count < INT_MAX)
			return __tryGpuSortWithLimitPath(root, path, limit_count, false);
	}
	return NULL;
}
#endif

/*
 * tryGpuSortWithWindowRankPath
 */
static Path *
__attachGpuSortWithWindowRankPath(PlannerInfo *root,
								  WindowAggPath *wpath,
								  CustomPath *cpath)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	WindowClause *wc = wpath->winclause;
	List	   *runCondition = __windowAggPathGetRunCondition(wpath);
	int			window_func = 0;
	int64_t		window_limit = -1;
	int			window_partby_nkeys = 0;
	int			window_orderby_nkeys = 0;
	ListCell   *lc, *cell;

	if (pp_info->gpusort_keys_expr == NIL ||
		pp_info->gpusort_keys_kind == NIL)
		return NULL;	/* no sort */
	/*
	 * Check whether the partition-by / order-by fits sorting keys
	 */
	cell = list_head(pp_info->gpusort_keys_refs);
	foreach (lc, wc->partitionClause)
	{
		SortGroupClause *sgc = lfirst(lc);

		if (!cell || lfirst_int(cell) != sgc->tleSortGroupRef)
		{
			elog(DEBUG2, "window-rank: GPU-Sort partition-keys mismatch");
			return NULL;
		}
		window_partby_nkeys++;
		cell = lnext(pp_info->gpusort_keys_refs, cell);
	}
	foreach (lc, wc->orderClause)
	{
		SortGroupClause *sgc = lfirst(lc);

		if (!cell || lfirst_int(cell) != sgc->tleSortGroupRef)
		{
			elog(DEBUG2, "window-rank: GPU-Sort ordering-keys mismatch");
			return NULL;
		}
		window_orderby_nkeys++;
		cell = lnext(pp_info->gpusort_keys_refs, cell);
	}

	/*
	 * Check whether the 'runCondition' contains supported filter
	 */
	foreach (lc, runCondition)
	{
		OpExpr	   *op = lfirst(lc);
		WindowFunc *func = NULL;
		Const	   *con = NULL;
		StrategyNumber strategy;

		if (!IsA(op, OpExpr) || list_length(op->args) != 2)
			continue;
		strategy = get_op_opfamily_strategy(op->opno, INTEGER_BTREE_FAM_OID);
		if (strategy ==  BTLessStrategyNumber ||
			strategy ==  BTLessEqualStrategyNumber)
		{
			func = linitial(op->args);
			con  = lsecond(op->args);
		}
		else if (strategy == BTGreaterStrategyNumber ||
				 strategy == BTGreaterEqualStrategyNumber)
		{
			con  = linitial(op->args);
			func = lsecond(op->args);
		}
		else
		{
			continue;
		}

		if (IsA(func, WindowFunc) && IsA(con, Const) && !con->constisnull)
		{
			int			__window_func;
			int64_t		__window_limit;

			switch (func->winfnoid)
			{
				case F_ROW_NUMBER:
					__window_func = KSORT_WINDOW_FUNC__ROW_NUMBER;
					break;
				case F_RANK_:
					__window_func = KSORT_WINDOW_FUNC__RANK;
					break;
				case F_DENSE_RANK_:
					__window_func = KSORT_WINDOW_FUNC__DENSE_RANK;
					break;
				default:
					goto skip;
			}

			switch (con->consttype)
			{
				case INT2OID:
					__window_limit = DatumGetInt16(con->constvalue);
					break;
				case INT4OID:
					__window_limit = DatumGetInt32(con->constvalue);
					break;
				case INT8OID:
					__window_limit = DatumGetInt64(con->constvalue);
					break;
				default:
					goto skip;
			}

			if (strategy == BTLessEqualStrategyNumber ||
				strategy == BTGreaterEqualStrategyNumber)
			{
				if (__window_limit <= 0)
				{
					//XXX to be replaced to empty results?
					elog(DEBUG2, "window-rank: rank() less than or equal to zero or negative shall always generate empty results");
					return NULL;
				}
				__window_limit++;
			}
			else
			{
				if (__window_limit <= 1)
				{
					elog(DEBUG2, "window-rank: rank() less than zero or negative shall always generate empty results");
					return NULL;
				}
			}

			if (window_func == 0)
			{
				window_func  = __window_func;
				window_limit = __window_limit;
			}
			else if (window_func == __window_func)
			{
				if (window_limit > __window_limit)
					window_limit = __window_limit;
			}
			else
			{
				elog(DEBUG2, "window-rank: different rank() functions are mixed, so unable to determine how much rows shall be filtered out");
				return NULL;
			}
		}
	skip:
		;
	}

	if (window_func)
	{
		double	ngroups;
		double	nrows;
		Cost	__startup_cost;
		Cost	__per_tuple;
		List   *__partby_keys = NIL;
		int		__count = 0;

		foreach (lc, pp_info->gpusort_keys_expr)
		{
			if (__count++ >= window_partby_nkeys)
				break;
			__partby_keys = lappend(__partby_keys, lfirst(lc));
		}
		ngroups = estimate_num_groups(root,
									  __partby_keys,
									  cpath->path.rows,
									  NULL, NULL);
		list_free(__partby_keys);

		nrows = (double)window_limit * ngroups;
		if (nrows >= cpath->path.rows)
		{
			elog(DEBUG2, "window-rank: estimated number of tuples reduction is not sufficient (partitions: %.0f, input-rows: %.0f, output-rows: %.0f",
				 ngroups, cpath->path.rows, nrows);
			return NULL;		/* no benefit */
		}
		cpath = pmemdup(cpath, sizeof(CustomPath));
		cpath->custom_private = list_copy(cpath->custom_private);
		pp_info = copy_pgstrom_plan_info(pp_info);
		pp_info->window_rank_func  = window_func;
		pp_info->window_rank_limit = window_limit;
		pp_info->window_partby_nkeys = window_partby_nkeys;
		pp_info->window_orderby_nkeys = window_orderby_nkeys;
		linitial(cpath->custom_private) = pp_info;

		__per_tuple = cpath->path.pathtarget->cost.per_tuple + pgstrom_gpu_tuple_cost;
		__startup_cost = cpath->path.total_cost
			- cpath->path.rows * __per_tuple
			+ cpath->path.rows * pgstrom_gpu_operator_cost
			+ nrows * __per_tuple;
		cpath->path.startup_cost = __startup_cost;
        cpath->path.total_cost = __startup_cost + __per_tuple * nrows;
		cpath->path.rows = nrows;
		return &cpath->path;
	}

	if (runCondition != NIL)
		elog(DEBUG2, "window-rank: not supported run-condition of the first window-agg node: %s",
			 nodeToString(runCondition));
	return NULL;
}

static Path *
tryGpuSortWithWindowRankPath(PlannerInfo *root,
							 WindowAggPath *wpath,
							 Path *__path,
							 bool be_parallel)
{
	Path   *subpath;

	switch (__path->type)
	{
		case T_WindowAggPath:
			{
				WindowAggPath  *__wpath = (WindowAggPath *)__path;

				subpath = tryGpuSortWithWindowRankPath(root,
													   __wpath,
													   __wpath->subpath,
													   be_parallel);
				if (subpath)
					return __duplicateWindowAggPath(root, __wpath, subpath);
			}
			break;

		case T_GatherPath:
			subpath = ((GatherPath *)__path)->subpath;
			subpath = tryGpuSortWithWindowRankPath(root, wpath, subpath, true);
			if (subpath)
			{
				GatherPath *gpath
					= create_gather_path(root,
										 __path->parent,
										 subpath,
										 __path->pathtarget,
										 NULL,
										 &subpath->rows);
				gpath->path.pathkeys = __path->pathkeys;
				//elog(INFO, "Gpath startup=%f total=%f rows=%.0f", gpath->path.startup_cost, gpath->path.total_cost, gpath->path.rows);
				return &gpath->path;
			}
			break;

		case T_GatherMergePath:
			subpath = ((GatherMergePath *)__path)->subpath;
			subpath = tryGpuSortWithWindowRankPath(root, wpath, subpath, true);
			if (subpath)
			{
				GatherPath *gpath
					= create_gather_path(root,
										 __path->parent,
										 subpath,
										 __path->pathtarget,
										 NULL,
										 &subpath->rows);
				gpath->path.pathkeys = __path->pathkeys;
				//elog(INFO, "Gpath startup=%f total=%f rows=%.0f", gpath->path.startup_cost, gpath->path.total_cost, gpath->path.rows);
				return &gpath->path;
			}
			break;

		case T_ProjectionPath:
			subpath = ((ProjectionPath *)__path)->subpath;
			subpath = tryGpuSortWithWindowRankPath(root, wpath, subpath, be_parallel);
			if (subpath)
			{
				ProjectionPath *ppath
					= create_projection_path(root,
											 __path->parent,
											 subpath,
											 __path->pathtarget);
				ppath->path.pathkeys = __path->pathkeys;
				//elog(INFO, "Proj-path startup=%f total=%f rows=%.0f", ppath->path.startup_cost, ppath->path.total_cost, ppath->path.rows);
				return &ppath->path;
			}
			break;

		case T_AppendPath:
			if (IS_PARTITIONED_REL(__path->parent))
			{
				CustomPath *cpath = pgstrom_find_custom_path(root,
															 __path->parent,
															 be_parallel);
				if (cpath)
					__path = &cpath->path;	/* fallback to T_CustomPath below */
			}
			__attribute__ ((fallthrough));
		case T_CustomPath:
			if (pgstrom_is_dummy_path(__path))
			{
				CustomPath *cpath = (CustomPath *)__path;

				subpath = linitial(cpath->custom_paths);
				subpath = tryGpuSortWithWindowRankPath(root, wpath, subpath, be_parallel);
				if (subpath)
					return pgstrom_create_dummy_path(root, subpath);
			}
			else if (wpath && (pgstrom_is_gpuscan_path(__path) ||
							   pgstrom_is_gpujoin_path(__path) ||
							   pgstrom_is_gpupreagg_path(__path)))
			{
				CustomPath *cpath = (CustomPath *)__path;

				return __attachGpuSortWithWindowRankPath(root, wpath, cpath);
			}
			break;

		default:
			break;
	}
	return NULL;
}

/*
 * tryInjectPartitionedGpuScanPath
 *
 * See issue #1026 for details.
 *
 * Once the PostgreSQL optimizer has built Scan/Join paths, if the
 * top-level relation is a partitioned table, apply_scanjoin_target_to_paths()
 * removes all paths registered on that relation and replaces them with an
 * Append path that aggregates scan paths from the child relations.
 *
 * Therefore, even if a CustomPath such as GpuScan is available on the
 * partitioned table itself, it will not be selected simply because it has
 * the lowest cost. We must explicitly re-inject the CustomPath after this
 * rewrite; otherwise it has already been discarded before the final path
 * selection.
 */
static Path *
__tryAddPartitionedGpuScanPath(PlannerInfo *root,
							   FinalPathExtraData *extra,
							   const Path *path, bool be_parallel)
{
	Path   *subpath;

	switch (path->type)
	{
		case T_WindowAggPath: {
			WindowAggPath  *wpath = (WindowAggPath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 wpath->subpath,
													 be_parallel);
			if (subpath)
				return __duplicateWindowAggPath(root, wpath, subpath);
			break;
		}
		case T_GatherPath: {
			GatherPath	   *gpath = (GatherPath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 gpath->subpath, true);
			if (subpath)
				return (Path *)create_gather_path(root,
												  gpath->path.parent,
												  subpath,
												  gpath->path.pathtarget,
												  NULL,
												  &subpath->rows);
			break;
		}
		case T_GatherMergePath: {
			GatherMergePath *gmpath = (GatherMergePath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 gmpath->subpath, true);
			if (subpath)
				return (Path *)create_gather_merge_path(root,
														gmpath->path.parent,
														subpath,
														gmpath->path.pathtarget,
														gmpath->path.pathkeys,
														NULL,
														&subpath->rows);
			break;
		}
		case T_ProjectionPath: {
			ProjectionPath *pjpath = (ProjectionPath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 pjpath->subpath, be_parallel);
			if (subpath)
				return (Path *)create_projection_path(root,
													  pjpath->path.parent,
													  subpath,
													  pjpath->path.pathtarget);

			break;
		}
		case T_SortPath: {
			SortPath	   *spath = (SortPath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 spath->subpath, be_parallel);
			if (subpath)
				return (Path *)create_sort_path(root,
												spath->path.parent,
												subpath,
												spath->path.pathkeys,
												root->limit_tuples);
			break;
		}
		case T_LimitPath: {
			LimitPath	   *lpath = (LimitPath *)path;

			subpath = __tryAddPartitionedGpuScanPath(root, extra,
													 lpath->subpath, be_parallel);
			if (subpath)
				return (Path *)create_limit_path(root,
												 lpath->path.parent,
												 subpath,
												 lpath->limitOffset,
												 lpath->limitCount,
												 lpath->limitOption,
												 extra->offset_est,
												 extra->count_est);
			break;
		}
		case T_AppendPath:
			if (IS_PARTITIONED_REL(path->parent))
			{
				CustomPath *cpath = pgstrom_find_custom_path(root,
															 path->parent,
															 be_parallel);
				if (cpath)
					return (Path *)pmemdup(cpath, sizeof(CustomPath));
			}
			break;
		default:
			break;
	}
	return NULL;
}

static void
tryAddPartitionedGpuScanPath(PlannerInfo *root,
							 FinalPathExtraData *extra,
							 RelOptInfo *input_rel)
{
	List	   *pathlist = NIL;
	ListCell   *lc;

	foreach (lc, input_rel->pathlist)
	{
		Path   *oldpath = lfirst(lc);
		Path   *newpath;

		newpath = __tryAddPartitionedGpuScanPath(root, extra, oldpath, false);
		if (newpath)
			pathlist = lappend(pathlist, newpath);
	}
	foreach (lc, pathlist)
		add_path(input_rel, (Path *)lfirst(lc));
}

/*
 * GpuPreAggAddCustomPath
 */
static void
GpuPreAggAddCustomPath(PlannerInfo *root,
					   UpperRelationKind upper_stage,
					   RelOptInfo *input_rel,
					   RelOptInfo *upper_rel,
					   void *extra)
{
	List	   *pathlist = NIL;
	ListCell   *lc;

	if (create_upper_paths_next)
		create_upper_paths_next(root,
								upper_stage,
								input_rel,
								upper_rel,
								extra);
	if (!pgstrom_enabled())
		return;

	switch (upper_stage)
	{
		case UPPERREL_GROUP_AGG:
		case UPPERREL_DISTINCT:
			if (pgstrom_enable_gpupreagg && gpuserv_ready_accept())
			{
				tryGpuPreAggAddCustomPath(root,
										  upper_stage,
										  input_rel,
										  upper_rel,
										  extra);
			}
			break;
		case UPPERREL_WINDOW:
			if (pgstrom_enable_gpusort)
			{
				foreach (lc, upper_rel->pathlist)
				{
					Path   *__path = lfirst(lc);

					__path = tryGpuSortWithWindowRankPath(root, NULL, __path, false);
					if (__path)
						pathlist = lappend(pathlist, __path);
				}
				foreach (lc, pathlist)
					add_path(upper_rel, (Path *)lfirst(lc));
			}
			break;
		case UPPERREL_FINAL:
			if (pgstrom_enable_gpusort)
			{
				foreach (lc, input_rel->pathlist)
				{
					Path   *__path = lfirst(lc);

					FinalPathExtraData *fextra = extra;
					Query	   *parse = root->parse;

					/* WITH TIES needs the rows after the limit too */
					if (parse->limitOption == LIMIT_OPTION_WITH_TIES)
						continue;
					__path = tryGpuSortWithLimitPath(root, extra, __path, false);
					if (__path)
					{
						/*
						 * The sorter keeps the first limit+offset rows; the Limit node (the planner has
						 * not put one on this path: it only adds it to the paths of its own list) drops
						 * the OFFSET rows and enforces the count.
						 */
						if (fextra->limit_needed)
							__path = (Path *)create_limit_path(root,
															   upper_rel,
															   __path,
															   parse->limitOffset,
															   parse->limitCount,
															   parse->limitOption,
															   fextra->offset_est,
															   fextra->count_est);
						pathlist = lappend(pathlist, __path);
					}
				}
				foreach (lc, pathlist)
					add_path(upper_rel, (Path *)lfirst(lc));
			}
			tryAddPartitionedGpuScanPath(root, extra, upper_rel);
			break;
		default:
			break;
	}
}

/*
 * PlanGpuPreAggPath
 */
static Plan *
PlanGpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *joinrel,
				  CustomPath *cpath,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanGpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &gpupreagg_plan_methods);
	if ((pp_info->xpu_task_flags & DEVTASK__MERGE_FINAL_BUFFER) != 0)
	{
		List	   *having_proj_quals = lthird(cpath->custom_private);

		/* HAVING clause */
		if (having_proj_quals && pp_info->gpusort_keys_expr)
			elog(ERROR, "Bug? GPU-Sort and HAVING clause are not usable together");
		cscan->scan.plan.qual = having_proj_quals;
	}
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/*
 * tryAddSelectIntoDirectProjection
 */
bool
tryAddSelectIntoDirectProjection(pgstromTaskState *pts,
								 List **p_select_into_proj)
{
	ProjectionInfo *projInfo = pts->css.ss.ps.ps_ProjInfo;
	List	   *select_into_proj = NIL;

	if (projInfo)
	{
		CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
		List	   *tlist_cpu = (List *)projInfo->pi_state.expr;
		ListCell   *lc;
		int			nvalids = 0;
		bool		meet_resjunk = false;
		bool		compatible;

		/* number of valid attributes in tlist_dev */
		foreach (lc, cscan->custom_scan_tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (tle->resjunk)
				meet_resjunk = true;
			else if (!meet_resjunk)
				nvalids++;
			else
				elog(ERROR, "Bug? attribute exists after junks: %s",
					 nodeToString(cscan->custom_scan_tlist));
		}
		/* check for each final projection */
		compatible = (list_length(tlist_cpu) == nvalids);
		foreach (lc, tlist_cpu)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Var))
			{
				Var	   *var = (Var *)tle->expr;

				if (var->varno == INDEX_VAR &&
					var->varattno > 0 &&
					var->varattno <= nvalids)
				{
					TargetEntry *__tle = list_nth(cscan->custom_scan_tlist, var->varattno-1);
					int		code = (KAGG_FINAL__SIMPLE_VREF << 16) | var->varattno;

					Assert(var->vartype == exprType((Node *)__tle->expr) &&
						   var->vartypmod == exprTypmod((Node *)__tle->expr));
					if (var->varattno != tle->resno)
						compatible = false;
					select_into_proj = lappend_int(select_into_proj, code);
					continue;
				}
			}
			else if (IsA(tle->expr, FuncExpr))
			{
				FuncExpr   *func = (FuncExpr *)tle->expr;
				int			code = finalfunc_catalog_lookup_by_oid(func->funcid);

				if (code >= 0 && list_length(func->args) == 1)
				{
					Var	   *var = linitial(func->args);

					if (IsA(var, Var) &&
						var->varno == INDEX_VAR &&
						var->varattno > 0 &&
						var->varattno <= nvalids)
					{
						TargetEntry *__tle = list_nth(cscan->custom_scan_tlist,
													  var->varattno-1);
						Assert(var->vartype == exprType((Node *)__tle->expr) &&
							   var->vartypmod == exprTypmod((Node *)__tle->expr));
						code = (code << 16) | var->varattno;
						select_into_proj = lappend_int(select_into_proj, code);
						compatible = false;
						continue;
					}
				}
			}
			elog(DEBUG2, "SELECT INTO Direct disabled: not a supported expression: %s",
				 nodeToString(tle->expr));
			return false;
		}
		/* no need to run heavy projection */
		if (compatible)
			select_into_proj = NIL;
	}
	*p_select_into_proj = select_into_proj;
	return true;
}

/*
 * CreateGpuPreAggScanState
 */
static Node *
CreateGpuPreAggScanState(CustomScan *cscan)
{
	Assert(cscan->methods == &gpupreagg_plan_methods);
	return pgstromCreateTaskState(cscan, &gpupreagg_exec_methods);
}

/*
 * Entrypoint of GpuPreAgg
 */
void
pgstrom_init_gpu_preagg(void)
{
	/* turn on/off GpuPreAgg */
	DefineCustomBoolVariable("pg_strom.enable_gpupreagg",
							 "Enables the use of GPU-PreAgg",
							 NULL,
							 &pgstrom_enable_gpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.gpupreagg_max_groups */
	DefineCustomIntVariable("pg_strom.gpupreagg_max_groups",
							"Largest estimated number of groups for which GROUP BY runs on the GPU",
							"The GPU wins for up to about a hundred groups (the service keeps each workgroup's groups in shared memory, in a table sized for the groups it has seen); with more groups most rows are aggregated on the CPU",
							&pgstrom_gpupreagg_max_groups,
							128,
							1,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* pg_strom.gpupreagg_max_groups_heap */
	DefineCustomIntVariable("pg_strom.gpupreagg_max_groups_heap",
							"Largest estimated number of groups for which GROUP BY over a scan (heap, Arrow_Fdw, GPU-Cache) runs on the GPU",
							"A GROUP BY over a heap table keeps its groups in a hash table in device memory for the whole query; it beat 8-worker PostgreSQL from about a thousand groups up to a million. Other sources are limited by pg_strom.gpupreagg_max_groups",
							&pgstrom_gpupreagg_max_groups_heap,
							2000000,
							1,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* pg_strom.enable_numeric_aggfuncs */
	DefineCustomBoolVariable("pg_strom.enable_numeric_aggfuncs",
							 "Enable aggregate functions on numeric type",
							 NULL,
							 &pgstrom_enable_numeric_aggfuncs,
							 true,
							 PGC_USERSET,
							 GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_partitionwise_gpugroupby */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpupreagg",
							 "Enabled Enables partition wise GPU-PreAgg",
							 NULL,
							 &pgstrom_enable_partitionwise_gpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off GPU-Sort */
	/* (GPU-Sort comes back with milestone A7; the Vulkan service does not implement it yet) */
	DefineCustomBoolVariable("pg_strom.enable_gpusort",
							 "Enables to use GPU-Sort on top of GPU-Projection/PreAgg",
							 NULL,
							 &pgstrom_enable_gpusort,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* initialization of path method table */
	memset(&gpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	gpupreagg_path_methods.CustomName          = "GpuPreAgg";
	gpupreagg_path_methods.PlanCustomPath      = PlanGpuPreAggPath;

	/* initialization of plan method table */
	memset(&gpupreagg_plan_methods, 0, sizeof(CustomScanMethods));
	gpupreagg_plan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_plan_methods.CreateCustomScanState = CreateGpuPreAggScanState;
	RegisterCustomScanMethods(&gpupreagg_plan_methods);

	/* initialization of exec method table */
	memset(&gpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
	gpupreagg_exec_methods.CustomName          = "GpuPreAgg";
	gpupreagg_exec_methods.BeginCustomScan     = pgstromExecInitTaskState;
	gpupreagg_exec_methods.ExecCustomScan      = pgstromExecTaskState;
	gpupreagg_exec_methods.EndCustomScan       = pgstromExecEndTaskState;
	gpupreagg_exec_methods.ReScanCustomScan    = pgstromExecResetTaskState;
	gpupreagg_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
	gpupreagg_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	gpupreagg_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	gpupreagg_exec_methods.ShutdownCustomScan  = pgstromSharedStateShutdownDSM;
	gpupreagg_exec_methods.ExplainCustomScan   = pgstromExplainTaskState;

	/* hook registration */
	create_upper_paths_next = create_upper_paths_hook;
	create_upper_paths_hook = GpuPreAggAddCustomPath;
	CacheRegisterSyscacheCallback(PROCOID, aggfunc_catalog_htable_invalidator, 0);
}
