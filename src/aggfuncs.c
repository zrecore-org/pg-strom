/*
 * aggfuncs.c
 *
 * Definition of self-defined aggregate functions, used by GpuPreAgg
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "float2.h"

/*
 * Functions Declaration
 */
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows);
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows_filtered);

PG_FUNCTION_INFO_V1(pgstrom_partial_minmax_int64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_bool);
PG_FUNCTION_INFO_V1(pgstrom_partial_minmax_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fmin_trans_int64);
PG_FUNCTION_INFO_V1(pgstrom_fmin_trans_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fmax_trans_int64);
PG_FUNCTION_INFO_V1(pgstrom_fmax_trans_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int8);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int16);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int32);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp16);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp32);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_numeric);

PG_FUNCTION_INFO_V1(pgstrom_partial_sum_int);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_int64);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_fp);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_numeric);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_cash);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_int);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_fp);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int_as_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int_as_cash);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp32);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp64_as_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_numeric);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_int);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_fp);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_num);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_numeric);

PG_FUNCTION_INFO_V1(pgstrom_partial_variance);
PG_FUNCTION_INFO_V1(pgstrom_stddev_trans);
PG_FUNCTION_INFO_V1(pgstrom_stddev_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_sampf_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_pop_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_popf_final);
PG_FUNCTION_INFO_V1(pgstrom_var_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_var_sampf_final);
PG_FUNCTION_INFO_V1(pgstrom_var_pop_final);
PG_FUNCTION_INFO_V1(pgstrom_var_popf_final);

PG_FUNCTION_INFO_V1(pgstrom_partial_covar);
PG_FUNCTION_INFO_V1(pgstrom_covar_accum);
PG_FUNCTION_INFO_V1(pgstrom_correlation_final);
PG_FUNCTION_INFO_V1(pgstrom_covar_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_covar_pop_final);

PG_FUNCTION_INFO_V1(pgstrom_regr_avgx_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_avgy_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_count_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_intercept_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_r2_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_slope_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_sxx_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_sxy_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_syy_final);

/*
 * float8 validator
 */
static inline void
check_float8_value(float8 value, bool inf_is_valid, bool zero_is_valid)
{
	if (isinf(value) && !inf_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: overflow")));
	if (value == 0.0 && !zero_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: underflow")));
}

/*
 * NROWS
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_nrows(PG_FUNCTION_ARGS)
{
	int		nargs = PG_NARGS();;

	for (int i=0; i < nargs; i++)
	{
		if (PG_ARGISNULL(i))
			PG_RETURN_INT64(0);
	}
	PG_RETURN_INT64(1);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_nrows_filtered(PG_FUNCTION_ARGS)
{
	int		nargs = PG_NARGS();

	Assert(nargs > 0);
	if (PG_ARGISNULL(nargs-1) || !PG_GETARG_BOOL(nargs-1))
		PG_RETURN_INT64(0);
	for (int i=1; i < nargs; i++)
	{
		if (PG_ARGISNULL(i-1))
			PG_RETURN_INT64(0);
	}
	PG_RETURN_INT64(1);
}

/*
 * MIN(X) and MAX(X) functions
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_minmax_int64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__pminmax_int64_packed));
	r->attrs = __PAGG_MINMAX_ATTRS__VALID;
	r->value = PG_GETARG_INT64(0);
	SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_minmax_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__pminmax_fp64_packed));
	r->attrs = __PAGG_MINMAX_ATTRS__VALID;
	r->value = PG_GETARG_FLOAT8(0);
	SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));

	PG_RETURN_POINTER(r);
}

#define __MINMAX_TRANS_TEMPLATE(TYPE,OPER)								\
	kagg_state__pminmax_##TYPE##_packed *state;							\
	kagg_state__pminmax_##TYPE##_packed *arg;							\
	MemoryContext	aggcxt;												\
																		\
	if (!AggCheckCallContext(fcinfo, &aggcxt))							\
		elog(ERROR, "aggregate function called in non-aggregate context"); \
	if (PG_ARGISNULL(0))												\
	{																	\
		if (PG_ARGISNULL(1))											\
			PG_RETURN_NULL();											\
		arg = (kagg_state__pminmax_##TYPE##_packed *)					\
			PG_GETARG_BYTEA_P(1);										\
		state = MemoryContextAlloc(aggcxt, sizeof(*state));				\
		memcpy(state, arg, sizeof(*state));								\
	}																	\
	else																\
	{																	\
		state = (kagg_state__pminmax_##TYPE##_packed *)					\
			PG_GETARG_BYTEA_P(0);										\
		if (!PG_ARGISNULL(1))											\
		{																\
			arg = (kagg_state__pminmax_##TYPE##_packed *)				\
				PG_GETARG_BYTEA_P(1);									\
			if ((arg->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)			\
			{															\
				if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)	\
					memcpy(state, arg, sizeof(*state));					\
				else													\
					state->value = OPER(state->value, arg->value);		\
			}															\
		}																\
	}																	\
	PG_RETURN_POINTER(state);

PUBLIC_FUNCTION(Datum)
pgstrom_fmin_trans_int64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(int64,Min);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmin_trans_fp64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(fp64,Min);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmax_trans_int64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(int64,Max);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmax_trans_fp64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(fp64,Max);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int8(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	/*
	 * fmin_i1 / fmin_i2 / fmin_i4 / fmin_i8 (and fmax_*) are all bound to this symbol by the old SQL
	 * scripts, so the width comes from the declared result type. (Before, every one of them range-checked
	 * as int1: the final projection of a GpuPreAgg without an Aggregate node, i.e. GPU-Sort, failed
	 * for min/max of int2/int4/int8.)
	 */
	switch (get_fn_expr_rettype(fcinfo->flinfo))
	{
		case INT2OID:
			if (state->value < SHRT_MIN || state->value > SHRT_MAX)
				elog(ERROR, "min/max(int2) out of range");
			PG_RETURN_INT16((int16) state->value);
		case INT4OID:
			if (state->value < INT_MIN || state->value > INT_MAX)
				elog(ERROR, "min/max(int4) out of range");
			PG_RETURN_INT32((int32) state->value);
		case INT8OID:
			PG_RETURN_INT64(state->value);
		default:
			break;		/* int1 */
	}
	if (state->value < SCHAR_MIN || state->value > SCHAR_MAX)
		elog(ERROR, "min/max(int1) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_bool(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_BOOL(state->value != 0);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int16(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	if (state->value < SHRT_MIN || state->value > SHRT_MAX)
		elog(ERROR, "min(int16) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int32(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	if (state->value < INT_MIN || state->value > INT_MAX)
		elog(ERROR, "min(int32) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_INT64(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp16(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_UINT16(__half_as_short__(fp64_to_fp16(state->value)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp32(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT4(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	return DirectFunctionCall1(float8_numeric,
							   Float8GetDatum(state->value));
}

/*
 * SUM(X),AVG(X) functions
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_int_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_INT64(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_int_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_int64(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_numeric_packed));
	r->attrs = 0;
	r->nitems = 1;
	__store_int128_packed(&r->sum, PG_GETARG_INT64(0));
	SET_VARSIZE(r, sizeof(kagg_state__psum_numeric_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_fp_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_FLOAT8(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_fp_packed));

	PG_RETURN_POINTER(r);
}

/*
 * sum(numeric), avg(numeric) partial state (kagg_state__psum_numeric_packed).
 *
 * attrs & WEIGHT is the display scale (dscale) of the sum, which PostgreSQL's sum()/avg() return: the maximum
 * dscale of the inputs. 'sum' is the exact sum as an unscaled 128-bit integer at that scale. The GPU service
 * (a GROUP BY or a whole-table aggregate over numeric(p,s) columns or expressions of a known scale) builds
 * the same state. A sum that does not fit 128 bits, or an input that does not, switches the state to the BIG
 * form: the struct is followed by the sum as a numeric varlena (attrs & BIG). NaN, +Infinity and -Infinity
 * are flags (they are never rows of the GPU: such rows are evaluated by the backend).
 */
#define __PAGG_NUMERIC_ATTRS__BIG		0x080000U

static inline int
__psum_numeric_dscale(const kagg_state__psum_numeric_packed *st)
{
	return (int)(st->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
}

/* the largest power of ten that fits an int128 is 10^38 */
static bool
__int128_mul_pow10(int128_t *v, int n)
{
	while (n-- > 0)
		if (__builtin_mul_overflow(*v, (int128_t) 10, v))
			return false;
	return true;
}

/* an unscaled int128 and its scale as a numeric (through the text form, which keeps the dscale) */
static Datum
__int128_to_numeric(int128_t value, int dscale)
{
	char		buf[160];
	char		digits[48];
	int			nd = 0, pos = 0;
	bool		neg = (value < 0);
	unsigned __int128 u = neg ? -(unsigned __int128) value : (unsigned __int128) value;

	do {
		digits[nd++] = '0' + (int)(u % 10);
		u /= 10;
	} while (u != 0);
	while (nd <= dscale)
		digits[nd++] = '0';					/* at least one digit before the decimal point */
	if (neg)
		buf[pos++] = '-';
	for (int i = nd - 1; i >= 0; i--)
	{
		buf[pos++] = digits[i];
		if (i == dscale && dscale > 0)
			buf[pos++] = '.';
	}
	buf[pos] = '\0';
	return DirectFunctionCall3(numeric_in, CStringGetDatum(buf), ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1));
}

/* the sum of a state as a numeric */
static Datum
__psum_numeric_to_datum(const kagg_state__psum_numeric_packed *st)
{
	if (st->attrs & __PAGG_NUMERIC_ATTRS__BIG)
	{
		const struct varlena *vl = (const struct varlena *)((const char *) st + sizeof(*st));
		struct varlena *copy = palloc(VARSIZE_ANY(vl));

		memcpy(copy, vl, VARSIZE_ANY(vl));
		return PointerGetDatum(copy);
	}
	return __int128_to_numeric(__fetch_int128_packed(&st->sum), __psum_numeric_dscale(st));
}

/* a BIG state from a numeric datum */
static kagg_state__psum_numeric_packed *
__psum_numeric_make_big(MemoryContext cxt, Datum numeric, uint32_t attrs, uint64_t nitems)
{
	struct varlena *vl = (struct varlena *) PG_DETOAST_DATUM(numeric);
	Size		len = VARSIZE_ANY(vl);
	kagg_state__psum_numeric_packed *r;
	MemoryContext oldcxt = MemoryContextSwitchTo(cxt);

	r = palloc0(sizeof(*r) + len);
	MemoryContextSwitchTo(oldcxt);
	r->attrs = (attrs | __PAGG_NUMERIC_ATTRS__BIG);
	r->nitems = nitems;
	memcpy((char *) r + sizeof(*r), vl, len);
	SET_VARSIZE(r, sizeof(*r) + len);
	return r;
}

/*
 * Decode a numeric into an unscaled int128 at its display scale: 0 if it is a plain value, 1 if it is
 * NaN / +Inf / -Inf (*p_special is the flag), 2 if it does not fit 128 bits (or has digits beyond its dscale).
 */
static int
__numeric_to_scaled_int128(Numeric num, uint32_t *p_dscale, int128_t *p_value, uint32_t *p_special)
{
	const varlena *addr = (const varlena *) num;
	uint32_t	len = VARSIZE_ANY_EXHDR(addr);
	NumericChoice *nc = (NumericChoice *) VARDATA_ANY(addr);
	uint16_t	n_head;
	int128_t	acc = 0;
	int			nd, e;
	NumericDigit *digits;

	if (len < sizeof(uint16_t))
		elog(ERROR, "corrupted numeric header");
	n_head = __Fetch(&nc->n_header);
	*p_special = 0;
	*p_dscale = 0;
	*p_value = 0;
	if (NUMERIC_IS_SPECIAL(n_head))
	{
		if (NUMERIC_IS_NAN(n_head))
			*p_special = __PAGG_NUMERIC_ATTRS__NAN;
		else if (NUMERIC_IS_PINF(n_head))
			*p_special = __PAGG_NUMERIC_ATTRS__PINF;
		else if (NUMERIC_IS_NINF(n_head))
			*p_special = __PAGG_NUMERIC_ATTRS__NINF;
		else
			elog(ERROR, "unknown special numeric value");
		return 1;
	}
	digits = NUMERIC_DIGITS(nc, n_head);
	nd = NUMERIC_NDIGITS(n_head, len);
	*p_dscale = NUMERIC_DSCALE(nc, n_head);
	if (*p_dscale > __PAGG_NUMERIC_ATTRS__WEIGHT)
		return 2;
	for (int i = 0; i < nd; i++)
	{
		NumericDigit dig = __Fetch(&digits[i]);

		if (__builtin_mul_overflow(acc, (int128_t) PG_NBASE, &acc) ||
			__builtin_add_overflow(acc, (int128_t) dig, &acc))
			return 2;
	}
	e = PG_DEC_DIGITS * (NUMERIC_WEIGHT(nc, n_head) - nd + 1) + (int) *p_dscale;
	if (acc != 0)
	{
		if (e >= 0)
		{
			if (!__int128_mul_pow10(&acc, e))
				return 2;
		}
		else
		{
			int128_t	p = 1;

			if (-e > 38 || !__int128_mul_pow10(&p, -e) || acc % p != 0)
				return 2;
			acc /= p;
		}
	}
	if (NUMERIC_SIGN(n_head) == NUMERIC_NEG)
		acc = -acc;
	*p_value = acc;
	return 0;
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *r;
	Numeric		num;
	uint32_t	dscale, special;
	int128_t	value;
	int			rc;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	num = PG_GETARG_NUMERIC(0);
	rc = __numeric_to_scaled_int128(num, &dscale, &value, &special);
	if (rc == 2)
		PG_RETURN_POINTER(__psum_numeric_make_big(CurrentMemoryContext, NumericGetDatum(num),
												  (dscale & __PAGG_NUMERIC_ATTRS__WEIGHT), 1));
	r = palloc0(sizeof(kagg_state__psum_numeric_packed));
	if (rc == 1)
		r->attrs = special;
	else
	{
		r->attrs = (dscale & __PAGG_NUMERIC_ATTRS__WEIGHT);
		__store_int128_packed(&r->sum, value);
	}
	r->nitems = 1;
	SET_VARSIZE(r, sizeof(kagg_state__psum_numeric_packed));
	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_cash(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) && !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_int_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_CASH(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_int_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;
	kagg_state__psum_int_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(1);
			state->nitems += arg->nitems;
			state->sum    += arg->sum;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;
	kagg_state__psum_fp_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(1);
			double		r = state->sum + arg->sum;

			/* as float8pl(): an infinite sum of finite values is an overflow (the CPU replay of a failed
			 * GPU node (A14) must raise the error that PostgreSQL's own plan raises) */
			if (isinf(r) && !isinf(state->sum) && !isinf(arg->sum))
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("value out of range: overflow")));
			state->nitems += arg->nitems;
			state->sum    = r;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	kagg_state__psum_numeric_packed *arg;
	MemoryContext		aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, VARSIZE(arg));
		memcpy(state, arg, VARSIZE(arg));
	}
	else
	{
		state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			uint32_t	special;
			int			ds_s, ds_a, ds;
			int128_t	vs = 0, va = 0;
			bool		big;

			arg = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(1);
			special = (arg->attrs & __PAGG_NUMERIC_ATTRS__MASK);
			ds_s = __psum_numeric_dscale(state);
			ds_a = __psum_numeric_dscale(arg);
			ds = Max(ds_s, ds_a);
			if (special != 0)
			{
				state->attrs |= special;			/* NaN / Infinity: the value does not matter any more */
				state->nitems += arg->nitems;
				PG_RETURN_POINTER(state);
			}
			/* the exact sum at the larger scale; anything that does not fit 128 bits goes to numeric */
			big = ((state->attrs | arg->attrs) & __PAGG_NUMERIC_ATTRS__BIG) != 0;
			if (!big)
			{
				vs = __fetch_int128_packed(&state->sum);
				va = __fetch_int128_packed(&arg->sum);
				if (!__int128_mul_pow10(&vs, ds - ds_s) ||
					!__int128_mul_pow10(&va, ds - ds_a) ||
					__builtin_add_overflow(vs, va, &vs))
					big = true;
			}
			if (!big)
			{
				state->attrs = ((state->attrs & ~__PAGG_NUMERIC_ATTRS__WEIGHT) | (uint32_t) ds);
				__store_int128_packed(&state->sum, vs);
				state->nitems += arg->nitems;
			}
			else
			{
				Datum		a = __psum_numeric_to_datum(state);
				Datum		b = __psum_numeric_to_datum(arg);
				Datum		sum = DirectFunctionCall2(numeric_add, a, b);
				uint32_t	attrs = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK) | (uint32_t) ds;

				state = __psum_numeric_make_big(aggcxt, sum, attrs, state->nitems + arg->nitems);
			}
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_INT64(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int_as_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(int8_numeric,
										Int64GetDatum(state->sum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int_as_cash(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_CASH(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp32(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	/* float4pl raises an error for an infinite sum of finite values; the partials are float4 values (Vulkan service) */
	if (!isinf(state->sum) && !isnan(state->sum) && isinf((float) state->sum))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: overflow")));
	PG_RETURN_FLOAT4(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp64_as_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric,
										Float8GetDatum(state->sum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	uint32_t	special;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	if (special != 0)
	{
		const char *str;

		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			str = "+Inf";
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			str = "-Inf";
		else
			str = "NaN";			/* NaN, or +Infinity and -Infinity together */
		PG_RETURN_DATUM(DirectFunctionCall3(numeric_in, CStringGetDatum(str),
											ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1)));
	}
	PG_RETURN_DATUM(__psum_numeric_to_datum(state));
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;
	Datum	n, sum;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	n = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->nitems));
	sum = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->sum));

	PG_RETURN_DATUM(DirectFunctionCall2(numeric_div, sum, n));
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;

	state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	/*
	 * avg(float8) starts from +0 (initcond '{0,0,0}'), sum(float8) from the first value: the sum of a -0 alone is -0 for
	 * sum, +0 for avg. "+ 0.0" turns a -0 into +0 and changes nothing else.
	 */
	PG_RETURN_FLOAT8(((double)state->sum + 0.0) / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_num(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;
	Datum	n, sum;

	state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	n = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->nitems));
	sum = DirectFunctionCall1(float8_numeric, Float8GetDatum(state->sum));

	PG_RETURN_DATUM(DirectFunctionCall2(numeric_div, sum, n));
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	uint32_t	special;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	if (special != 0)
	{
		const char *str;

		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			str = "+Inf";
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			str = "-Inf";
		else
			str = "NaN";
		PG_RETURN_DATUM(DirectFunctionCall3(numeric_in, CStringGetDatum(str),
											ObjectIdGetDatum(InvalidOid), Int32GetDatum(-1)));
	}
	/* as numeric_avg(): the sum (at the display scale of the inputs) divided by the number of inputs */
	PG_RETURN_DATUM(DirectFunctionCall2(numeric_div,
										__psum_numeric_to_datum(state),
										NumericGetDatum(int64_to_numeric((int64) state->nitems))));
}

/*
 * STDDEV/VARIANCE, COVAR/REGR_* (A10)
 *
 * The partial state is PostgreSQL's own transition state, not a sum of powers: N, Sx and Sxx of float8_accum (the Youngs-Cramer
 * update; Sxx is the sum of squared deviations, so there is no cancellation for data with a large mean), or float8_regr_accum's
 * N, Sx, Sxx, Sy, Syy, Sxy. The GPU service computes it with the same operations (kernels_vm/vagg.inc); the combine and final
 * functions below call PostgreSQL's float8_combine, float8_regr_combine and float8_var_samp ... float8_regr_syy on a float8[]
 * built from it, so they are the server's own code, version and rounding included. The packed (bytea) form is only the carrier
 * between GpuPreAgg and the Agg node: kagg_state__stddev_packed {nitems = N, sum_x = Sx, sum_x2 = Sxx} and
 * kagg_state__covar_packed {nitems = N, sum_x = Sx, sum_xx = Sxx, sum_y = Sy, sum_yy = Syy, sum_xy = Sxy}. regr_*(Y, X): the
 * first argument is Y, as in PostgreSQL's float8_regr_accum(state, Y, X). The variance of integers and numerics keeps PostgreSQL's
 * exact (int128 / numeric) state and is not offered to the GPU (KAGG_ACTION__STDDEV_NUM).
 */
static ArrayType *
__float8_array(const double *v, int n)
{
	Datum		d[6];

	for (int i = 0; i < n; i++)
		d[i] = Float8GetDatum(v[i]);
	return construct_array(d, n, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);
}

static ArrayType *
__stddev_to_array(const kagg_state__stddev_packed *st)
{
	double		v[3] = {(double) st->nitems, st->sum_x, st->sum_x2};

	return __float8_array(v, 3);
}

static ArrayType *
__covar_to_array(const kagg_state__covar_packed *st)
{
	double		v[6] = {(double) st->nitems, st->sum_x, st->sum_xx, st->sum_y, st->sum_yy, st->sum_xy};

	return __float8_array(v, 6);
}

static void
__array_to_stddev(kagg_state__stddev_packed *st, ArrayType *a)
{
	const float8 *v = (const float8 *) ARR_DATA_PTR(a);

	st->nitems = (int64_t) v[0];
	st->sum_x = v[1];
	st->sum_x2 = v[2];
}

static void
__array_to_covar(kagg_state__covar_packed *st, ArrayType *a)
{
	const float8 *v = (const float8 *) ARR_DATA_PTR(a);

	st->nitems = (int64_t) v[0];
	st->sum_x = v[1];
	st->sum_xx = v[2];
	st->sum_y = v[3];
	st->sum_yy = v[4];
	st->sum_xy = v[5];
}

/* call a PostgreSQL final function of a float8[] state (it returns NULL for too few rows: *isnull) */
static Datum
__call_float8_final(PGFunction fn, ArrayType *a, bool *isnull)
{
	LOCAL_FCINFO(fcinfo, 1);
	Datum		datum;

	InitFunctionCallInfoData(*fcinfo, NULL, 1, InvalidOid, NULL, NULL);
	fcinfo->args[0].value = PointerGetDatum(a);
	fcinfo->args[0].isnull = false;
	datum = fn(fcinfo);
	*isnull = fcinfo->isnull;
	return datum;
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_variance(PG_FUNCTION_ARGS)
{
	kagg_state__stddev_packed *r;
	static const double zero[3] = {0.0, 0.0, 0.0};
	Datum		st;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	/* float8_accum on the empty state: the state of one row (Sxx is NaN for an infinite or NaN value) */
	st = DirectFunctionCall2(float8_accum, PointerGetDatum(__float8_array(zero, 3)), Float8GetDatum(PG_GETARG_FLOAT8(0)));
	r = palloc0(sizeof(kagg_state__stddev_packed));
	r->attrs = 0;
	__array_to_stddev(r, DatumGetArrayTypeP(st));
	SET_VARSIZE(r, sizeof(kagg_state__stddev_packed));
	PG_RETURN_POINTER(r);
}

/* the combine function of the Agg node (float8_combine of the states) */
PUBLIC_FUNCTION(Datum)
pgstrom_stddev_trans(PG_FUNCTION_ARGS)
{
	kagg_state__stddev_packed *state;
	kagg_state__stddev_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			Datum		d;

			arg = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(1);
			d = DirectFunctionCall2(float8_combine,
									PointerGetDatum(__stddev_to_array(state)),
									PointerGetDatum(__stddev_to_array(arg)));
			__array_to_stddev(state, DatumGetArrayTypeP(d));
		}
	}
	PG_RETURN_POINTER(state);
}

#define STDDEV_FINAL(NAME, PGFN)											\
PUBLIC_FUNCTION(Datum)														\
NAME(PG_FUNCTION_ARGS)														\
{																			\
	bool		isnull;														\
	Datum		datum;														\
																			\
	if (PG_ARGISNULL(0))													\
		PG_RETURN_NULL();													\
	datum = __call_float8_final(PGFN, __stddev_to_array(					\
						(kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(0)),	\
						&isnull);											\
	if (isnull)																\
		PG_RETURN_NULL();													\
	PG_RETURN_DATUM(datum);													\
}

STDDEV_FINAL(pgstrom_var_sampf_final, float8_var_samp)
STDDEV_FINAL(pgstrom_var_popf_final, float8_var_pop)
STDDEV_FINAL(pgstrom_stddev_sampf_final, float8_stddev_samp)
STDDEV_FINAL(pgstrom_stddev_popf_final, float8_stddev_pop)

/* the numeric-typed finals (the variance of integers and numerics: not offered to the GPU, kept for the catalog) */
#define STDDEV_FINAL_NUMERIC(NAME, FLOATFN)									\
PUBLIC_FUNCTION(Datum)														\
NAME(PG_FUNCTION_ARGS)														\
{																			\
	Datum	datum = FLOATFN(fcinfo);										\
																			\
	if (fcinfo->isnull)														\
		PG_RETURN_NULL();													\
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric, datum));			\
}

STDDEV_FINAL_NUMERIC(pgstrom_var_samp_final, pgstrom_var_sampf_final)
STDDEV_FINAL_NUMERIC(pgstrom_var_pop_final, pgstrom_var_popf_final)
STDDEV_FINAL_NUMERIC(pgstrom_stddev_samp_final, pgstrom_stddev_sampf_final)
STDDEV_FINAL_NUMERIC(pgstrom_stddev_pop_final, pgstrom_stddev_popf_final)

/*
 * COVAR/REGR_*
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_covar(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *r;
	static const double zero[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	Datum		st;

	if (PG_NARGS() > 2 && (PG_ARGISNULL(2) || !PG_GETARG_BOOL(2)))
		PG_RETURN_NULL();
	/* float8_regr_accum(state, Y, X) on the empty state */
	st = DirectFunctionCall3(float8_regr_accum, PointerGetDatum(__float8_array(zero, 6)),
							 Float8GetDatum(PG_GETARG_FLOAT8(0)), Float8GetDatum(PG_GETARG_FLOAT8(1)));
	r = palloc0(sizeof(kagg_state__covar_packed));
	r->attrs = 0;
	__array_to_covar(r, DatumGetArrayTypeP(st));
	SET_VARSIZE(r, sizeof(kagg_state__covar_packed));
	PG_RETURN_POINTER(r);
}

/* the combine function of the Agg node (float8_regr_combine of the states) */
PUBLIC_FUNCTION(Datum)
pgstrom_covar_accum(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state;
	kagg_state__covar_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			Datum		d;

			arg = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(1);
			d = DirectFunctionCall2(float8_regr_combine,
									PointerGetDatum(__covar_to_array(state)),
									PointerGetDatum(__covar_to_array(arg)));
			__array_to_covar(state, DatumGetArrayTypeP(d));
		}
	}
	PG_RETURN_POINTER(state);
}

#define COVAR_FINAL(NAME, PGFN)												\
PUBLIC_FUNCTION(Datum)														\
NAME(PG_FUNCTION_ARGS)														\
{																			\
	bool		isnull;														\
	Datum		datum;														\
																			\
	if (PG_ARGISNULL(0))													\
		PG_RETURN_NULL();													\
	datum = __call_float8_final(PGFN, __covar_to_array(						\
						(kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0)),	\
						&isnull);											\
	if (isnull)																\
		PG_RETURN_NULL();													\
	PG_RETURN_DATUM(datum);													\
}

COVAR_FINAL(pgstrom_correlation_final, float8_corr)
COVAR_FINAL(pgstrom_covar_samp_final, float8_covar_samp)
COVAR_FINAL(pgstrom_covar_pop_final, float8_covar_pop)
COVAR_FINAL(pgstrom_regr_avgx_final, float8_regr_avgx)
COVAR_FINAL(pgstrom_regr_avgy_final, float8_regr_avgy)
COVAR_FINAL(pgstrom_regr_intercept_final, float8_regr_intercept)
COVAR_FINAL(pgstrom_regr_r2_final, float8_regr_r2)
COVAR_FINAL(pgstrom_regr_slope_final, float8_regr_slope)
COVAR_FINAL(pgstrom_regr_sxx_final, float8_regr_sxx)
COVAR_FINAL(pgstrom_regr_sxy_final, float8_regr_sxy)
COVAR_FINAL(pgstrom_regr_syy_final, float8_regr_syy)

PUBLIC_FUNCTION(Datum)
pgstrom_regr_count_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	PG_RETURN_FLOAT8((double)state->nitems);
}

#if 0
/*
 * ----------------------------------------------------------------
 *
 * Hyper-Log-Log support functions
 *
 * ----------------------------------------------------------------
 */

/*
 * Hash-function based on Sip-Hash
 *
 * See https://en.wikipedia.org/wiki/SipHash
 *     and https://github.com/veorq/SipHash
 */
/* default: SipHash-2-4 */
#define cROUNDS 2
#define dROUNDS 4
#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U8TO64_LE(p)											\
    (((uint64_t)((p)[0]))       | ((uint64_t)((p)[1]) <<  8) |	\
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |	\
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |	\
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND					\
    do {							\
        v0 += v1;					\
        v1 = ROTL(v1, 13);			\
        v1 ^= v0;					\
        v0 = ROTL(v0, 32);			\
        v2 += v3;					\
        v3 = ROTL(v3, 16);			\
        v3 ^= v2;					\
        v0 += v3;					\
        v3 = ROTL(v3, 21);			\
        v3 ^= v0;					\
        v2 += v1;					\
        v1 = ROTL(v1, 17);			\
        v1 ^= v2;					\
        v2 = ROTL(v2, 32);			\
    } while (0)

static uint64_t
__pgstrom_hll_siphash_value(const void *ptr, const size_t len)
{
	const unsigned char *ni = (const unsigned char *)ptr;
	uint64_t	v0 = 0x736f6d6570736575UL;
	uint64_t	v1 = 0x646f72616e646f6dUL;
	uint64_t	v2 = 0x6c7967656e657261UL;
	uint64_t	v3 = 0x7465646279746573UL;
	uint64_t	k0 = 0x9c38151cda15a76bUL;	/* random key-0 */
	uint64_t	k1 = 0xfb4ff68fbd3e6658UL;	/* random key-1 */
	uint64_t	m;
	int			i;
    const unsigned char *end = ni + len - (len % sizeof(uint64_t));
    const int	left = len & 7;
    uint64_t	b = ((uint64_t)len) << 56;

    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

	for (; ni != end; ni += 8)
	{
		m = U8TO64_LE(ni);
		v3 ^= m;

		for (i = 0; i < cROUNDS; ++i)
			SIPROUND;

		v0 ^= m;
	}

#if 1
	if (left > 0)
	{
		uint64_t	temp = 0;

		memcpy(&temp, ni, left);
		b |= (temp & ((1UL << (BITS_PER_BYTE * left)) - 1));
	}
#else
	/* original code */
	switch (left)
	{
		case 7:
			b |= ((uint64_t)ni[6]) << 48;		__attribute__ ((fallthrough));
		case 6:
			b |= ((uint64_t)ni[5]) << 40;		__attribute__ ((fallthrough));
		case 5:
			b |= ((uint64_t)ni[4]) << 32;		__attribute__ ((fallthrough));
		case 4:
			b |= ((uint64_t)ni[3]) << 24;		__attribute__ ((fallthrough));
		case 3:
			b |= ((uint64_t)ni[2]) << 16;		__attribute__ ((fallthrough));
		case 2:
			b |= ((uint64_t)ni[1]) << 8;		__attribute__ ((fallthrough));
		case 1:
			b |= ((uint64_t)ni[0]);
			break;
		case 0:
			break;
    }
#endif

    v3 ^= b;
	for (i = 0; i < cROUNDS; ++i)
		SIPROUND;

	v0 ^= b;

	v2 ^= 0xff;

	for (i = 0; i < dROUNDS; ++i)
		SIPROUND;

	b = v0 ^ v1 ^ v2 ^ v3;

	return b;
}

/*
 * pgstrom_hll_hash_xxxx functions
 */
static uint64
__pgstrom_hll_hash_int1(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int8));
}

static uint64
__pgstrom_hll_hash_int2(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int16));
}

static uint64
__pgstrom_hll_hash_int4(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int32));
}

static uint64
__pgstrom_hll_hash_int8(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int64));
}

static uint64
__pgstrom_hll_hash_numeric(Datum datum)
{
	xpu_numeric_t	num;
	const char	   *emsg;

	memset(&num, 0, sizeof(num));
	emsg = __xpu_numeric_from_varlena(&num, (struct varlena *)datum);
	if (emsg)
		elog(ERROR, "failed on hash calculation of device numeric: %s", emsg);
	return __pgstrom_hll_siphash_value(&num.weight,
									   offsetof(xpu_numeric_t, value)
									   + sizeof(int128_t)
									   - offsetof(xpu_numeric_t, weight));
}

static uint64
__pgstrom_hll_hash_date(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(DateADT));
}

static uint64
__pgstrom_hll_hash_time(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(TimeADT));
}

static uint64
__pgstrom_hll_hash_timetz(Datum datum)
{
	return __pgstrom_hll_siphash_value(DatumGetPointer(datum), sizeof(TimeTzADT));
}

static uint64
__pgstrom_hll_hash_timestamp(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(Timestamp));
}

static uint64
__pgstrom_hll_hash_timestamptz(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(TimestampTz));
}

static uint64
__pgstrom_hll_hash_bpchar(Datum datum)
{
	BpChar	   *val = DatumGetBpCharPP(datum);
	int			len = bpchartruelen(VARDATA_ANY(val),
									VARSIZE_ANY_EXHDR(val));
	return __pgstrom_hll_siphash_value(VARDATA_ANY(val), len);
}

static uint64
__pgstrom_hll_hash_varlena(Datum datum)
{
	struct varlena *val = PG_DETOAST_DATUM(datum);

	return __pgstrom_hll_siphash_value(VARDATA_ANY(val), VARSIZE_ANY_EXHDR(val));
}

static uint64
__pgstrom_hll_hash_uuid(Datum datum)
{
	return __pgstrom_hll_siphash_value(DatumGetUUIDP(datum), sizeof(pg_uuid_t));
}

static bytea *
__pgstrom_hll_sketch_update_common(PG_FUNCTION_ARGS, uint64 hash)
{
	MemoryContext	aggcxt;
	bytea		   *hll_state;
	uint8		   *hll_regs;
	uint64			nrooms;
	uint32			index;
	uint32			count;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	nrooms = (1UL << pgstrom_hll_register_bits);
	if (PG_ARGISNULL(0))
	{
		size_t	sz = VARHDRSZ + sizeof(uint8) * nrooms;
		hll_state = MemoryContextAllocZero(aggcxt, sz);
		SET_VARSIZE(hll_state, sz);
	}
	else
	{
		hll_state = PG_GETARG_BYTEA_P(0);
	}
	Assert(VARSIZE(hll_state) == VARHDRSZ + sizeof(uint8) * nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	index = hash & (nrooms - 1);
	count = __builtin_ctzll(hash >> pgstrom_hll_register_bits) + 1;
	if (hll_regs[index] < count)
		hll_regs[index] = count;
	return hll_state;
}

#define PGSTROM_HLL_HANDLER_TEMPLATE(NAME)								\
	PG_FUNCTION_INFO_V1(pgstrom_hll_hash_##NAME);						\
	PG_FUNCTION_INFO_V1(pgstrom_hll_sketch_update_##NAME);				\
	PUBLIC_FUNCTION(Datum)												\
	pgstrom_hll_hash_##NAME(PG_FUNCTION_ARGS)							\
	{																	\
		Datum	arg = PG_GETARG_DATUM(0);								\
		PG_RETURN_UINT64(__pgstrom_hll_hash_##NAME(arg));				\
	}																	\
	PUBLIC_FUNCTION(Datum)												\
	pgstrom_hll_sketch_update_##NAME(PG_FUNCTION_ARGS)					\
	{																	\
		if (PG_ARGISNULL(1))											\
		{																\
			if (PG_ARGISNULL(0))										\
				PG_RETURN_NULL();										\
			PG_RETURN_DATUM(PG_GETARG_DATUM(0));						\
		}																\
		else															\
		{																\
			Datum	arg = PG_GETARG_DATUM(1);							\
			uint64  hash = __pgstrom_hll_hash_##NAME(arg);				\
			bytea  *state;												\
																		\
			state = __pgstrom_hll_sketch_update_common(fcinfo, hash);	\
			PG_RETURN_BYTEA_P(state);									\
		}																\
	}

PGSTROM_HLL_HANDLER_TEMPLATE(int1)
PGSTROM_HLL_HANDLER_TEMPLATE(int2)
PGSTROM_HLL_HANDLER_TEMPLATE(int4)
PGSTROM_HLL_HANDLER_TEMPLATE(int8)
PGSTROM_HLL_HANDLER_TEMPLATE(numeric)
PGSTROM_HLL_HANDLER_TEMPLATE(date)
PGSTROM_HLL_HANDLER_TEMPLATE(time)
PGSTROM_HLL_HANDLER_TEMPLATE(timetz)
PGSTROM_HLL_HANDLER_TEMPLATE(timestamp)
PGSTROM_HLL_HANDLER_TEMPLATE(timestamptz)
PGSTROM_HLL_HANDLER_TEMPLATE(bpchar)
PGSTROM_HLL_HANDLER_TEMPLATE(varlena)
PGSTROM_HLL_HANDLER_TEMPLATE(uuid)

/*
 * pgstrom_hll_sketch_new
 */
PUBLIC_FUNCTION(Datum)
pgstrom_hll_sketch_new(PG_FUNCTION_ARGS)
{
	uint64		nrooms = (1UL << pgstrom_hll_register_bits);
	uint64		hll_hash = DatumGetUInt64(PG_GETARG_DATUM(0));
	bytea	   *hll_state;
	uint8	   *hll_regs;
	uint32		count;
	uint32		index;

	hll_state = palloc0(VARHDRSZ + sizeof(uint8) * nrooms);
	SET_VARSIZE(hll_state, VARHDRSZ + sizeof(uint8) * nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	index = hll_hash & (nrooms - 1);
	Assert(index < nrooms);
	count = __builtin_ctzll(hll_hash >> pgstrom_hll_register_bits) + 1;
	if (hll_regs[index] < count)
		hll_regs[index] = count;

	PG_RETURN_BYTEA_P(hll_state);
}

/*
 * pgstrom_hll_sketch_merge
 */
PUBLIC_FUNCTION(Datum)
pgstrom_hll_sketch_merge(PG_FUNCTION_ARGS)
{
	MemoryContext	aggcxt;
	bytea		   *hll_state = NULL;
	uint8		   *hll_regs;
	bytea		   *new_state;
	uint8		   *new_regs;
	uint32			nrooms;
	uint32			index;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		new_state = PG_GETARG_BYTEA_P(1);
		nrooms = VARSIZE_ANY_EXHDR(new_state);
		if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
			elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
		hll_state = MemoryContextAllocZero(aggcxt, VARHDRSZ + nrooms);
		SET_VARSIZE(hll_state, VARHDRSZ + nrooms);
		memcpy(VARDATA_ANY(hll_state), VARDATA_ANY(new_state), nrooms);
	}
	else
	{
		hll_state = PG_GETARG_BYTEA_P(0);
		nrooms = VARSIZE_ANY_EXHDR(hll_state);
		if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
			elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
		if (!PG_ARGISNULL(1))
		{
			new_state = PG_GETARG_BYTEA_P(1);
			if (VARSIZE_ANY_EXHDR(hll_state) != VARSIZE_ANY_EXHDR(new_state))
				elog(ERROR, "incompatible HLL sketch");
			hll_regs = (uint8 *)VARDATA_ANY(hll_state);
			new_regs = (uint8 *)VARDATA_ANY(new_state);
			for (index=0; index < nrooms; index++)
			{
				if (hll_regs[index] < new_regs[index])
                    hll_regs[index] = new_regs[index];
			}
		}
	}
	PG_RETURN_POINTER(hll_state);
}

/*
 * pgstrom_hll_count_final
 */
PUBLIC_FUNCTION(Datum)
pgstrom_hll_count_final(PG_FUNCTION_ARGS)
{
	bytea	   *hll_state;
	uint8	   *hll_regs;
	uint32		nrooms;
	uint32		index;
	double		divider = 0.0;
	double		weight;
	double		estimate;

#if 0
	/*
	 * MEMO: Here to no reason to prohibit to use pgstrom.hll_count_final()
	 * towards preliminary calculated HLL sketch.
	 */
	if (!AggCheckCallContext(fcinfo, NULL))
		elog(ERROR, "aggregate function called in non-aggregate context");
#endif
	if (PG_ARGISNULL(0))
		PG_RETURN_INT64(0);
	/*
	 * MEMO: Hyper-Log-Log merge algorithm
	 * https://ja.wikiqube.net/wiki/HyperLogLog
	 */
	hll_state = PG_GETARG_BYTEA_P(0);
	nrooms = VARSIZE_ANY_EXHDR(hll_state);
	if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
		elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	for (index = 0; index < nrooms; index++)
		divider += 1.0 / (double)(1UL << hll_regs[index]);
	if (nrooms <= 16)
		weight = 0.673;
	else if (nrooms <= 32)
		weight = 0.697;
	else if (nrooms <= 64)
		weight = 0.709;
	else
		weight = 0.7213 / (1.0 + 1.079 / (double)nrooms);

	estimate = (weight * (double)nrooms * (double)nrooms) / divider;
	PG_RETURN_INT64((int64)estimate);
}



/*
 * pgstrom_hll_sketch_histogram
 */
PUBLIC_FUNCTION(Datum)
pgstrom_hll_sketch_histogram(PG_FUNCTION_ARGS)
{
	bytea	   *hll_state = PG_GETARG_BYTEA_P(0);
	uint8	   *hll_regs;
	uint32		nrooms;
	uint32		index;
	Datum		hll_hist[64];
	int			max_hist = -1;
	ArrayType  *result;

	nrooms = VARSIZE_ANY_EXHDR(hll_state);
	if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
		elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	memset(hll_hist, 0, sizeof(hll_hist));
	for (index=0; index < nrooms; index++)
	{
		int		value = (int)hll_regs[index];

		if (value < 0 || value >= 64)
			elog(ERROR, "HLL sketch looks corrupted");
		hll_hist[value]++;
		if (max_hist < value)
			max_hist = value;
	}

	if (max_hist < 0)
		PG_RETURN_NULL();

	result = construct_array(hll_hist,
							 max_hist + 1,
							 INT4OID,
							 sizeof(int32),
							 true,
							 'i');
	PG_RETURN_POINTER(result);
}
#endif
