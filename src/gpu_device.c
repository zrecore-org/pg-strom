/*
 * gpu_device.c
 *
 * What the planner knows about the GPU: the device the GPU service selected at start-up
 * (the probe's result, see gpu_launcher.c), the cost parameters, and the SQL function that
 * describes the devices.
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "funcapi.h"
#include "gpuserv/gpuserv_proto.h"

#define COMPONENT_LABEL		"gpu-device"
/* variable declarations */
GpuDevAttributes *gpuDevAttrs = NULL;
int				numGpuDevAttrs = 0;
double			pgstrom_gpu_setup_cost;			/* GUC */
double			pgstrom_gpu_tuple_cost;			/* GUC */
double			pgstrom_gpu_operator_cost;		/* GUC */
double			pgstrom_gpu_direct_seq_page_cost; /* GUC */

/*
 * pgstrom_collect_gpu_devices
 *
 * The devices were probed by the GPU service program at start-up; the planner only needs
 * a description of the selected one.
 */
static void
pgstrom_collect_gpu_devices(void)
{
	const gpuserv_control *ctl = pgstrom_gpuserv_control();
	const gpuserv_devcaps *dc;
	GpuDevAttributes *dattrs;

	numGpuDevAttrs = 0;
	gpuDevAttrs = NULL;
	if (!ctl)
		return;
	dc = &ctl->devcaps[ctl->selected_device];
	dattrs = calloc(1, sizeof(GpuDevAttributes));
	if (!dattrs)
		elog(ERROR, "out of memory");
	dattrs->DEV_ID = 0;
	snprintf(dattrs->DEV_NAME, sizeof(dattrs->DEV_NAME), "%s", dc->name);
	snprintf(dattrs->DEV_UUID, sizeof(dattrs->DEV_UUID), "%s", dc->uuid);
	dattrs->DEV_TOTAL_MEMSZ = dc->device_memory;
	dattrs->MAX_THREADS_PER_BLOCK = dc->max_wg_invocations;
	dattrs->MAX_SHARED_MEMORY_PER_BLOCK = dc->max_shared_memory;
	dattrs->WARP_SIZE = (dc->subgroup_size_observed > 0
						 ? dc->subgroup_size_observed
						 : dc->subgroup_size_max);
	numGpuDevAttrs = 1;
	gpuDevAttrs = dattrs;
	elog(LOG, "PG-Strom: GPU0 %s (Vulkan, tier %s), RAM %zuMB",
		 dc->name, gpuserv_tier_name(dc->tier),
		 (size_t)(dc->device_memory >> 20));
}

/*
 * pgstrom_gpu_operator_ratio
 */
double
pgstrom_gpu_operator_ratio(void)
{
	if (cpu_operator_cost > 0.0)
	{
		return pgstrom_gpu_operator_cost / cpu_operator_cost;
	}
	return (pgstrom_gpu_operator_cost == 0.0 ? 1.0 : disable_cost);
}

/*
 * GetOptimalGpuForXXX - which GPUs are close to the storage of a file or relation, so that
 * GPU-Direct SQL could read it. There is no GPU-Direct here (the CPU reads the storage), so
 * the answer is always "none"; the GPU service is then fed by the CPU.
 */
gpumask_t
GetOptimalGpuForFile(const char *pathname)
{
	return 0UL;
}

gpumask_t
GetOptimalGpuForRelation(Relation relation)
{
	return 0UL;
}

gpumask_t
GetOptimalGpuForBaseRel(PlannerInfo *root, RelOptInfo *baserel)
{
	return 0UL;
}

/*
 * GetSystemAvailableGpus - returns bitmap of all available GPUs
 */
gpumask_t
GetSystemAvailableGpus(void)
{
	gpumask_t	available_gpus = 0;

	for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
		available_gpus |= (1UL << dindex);

	return available_gpus;
}

/*
 * GetGpuMinimalDeviceMemorySize
 */
size_t
GetGpuMinimalDeviceMemorySize(void)
{
	size_t	total_sz = 0;

	for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
	{
		size_t	dmem_sz = gpuDevAttrs[dindex].DEV_TOTAL_MEMSZ;

		if (dindex == 0 || total_sz > dmem_sz)
			total_sz = dmem_sz;
	}
	return total_sz;
}

/*
 * GetGpuTotalDeviceMemorySize
 */
size_t
GetGpuTotalDeviceMemorySize(void)
{
	size_t	total_sz = 0;

	for (int dindex=0; dindex < numGpuDevAttrs; dindex++)
		total_sz += gpuDevAttrs[dindex].DEV_TOTAL_MEMSZ;
	return total_sz;
}

/*
 * pgstrom_init_gpu_options
 */
static void
pgstrom_init_gpu_options(void)
{
	/* cost factor for GPU setup */
	DefineCustomRealVariable("pg_strom.gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 100 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for each Gpu task */
	DefineCustomRealVariable("pg_strom.gpu_tuple_cost",
							 "Default cost to transfer GPU<->Host per tuple",
							 NULL,
							 &pgstrom_gpu_tuple_cost,
							 DEFAULT_CPU_TUPLE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for GPU operator */
	DefineCustomRealVariable("pg_strom.gpu_operator_cost",
							 "Cost of processing each operators by GPU",
							 NULL,
							 &pgstrom_gpu_operator_cost,
							 DEFAULT_CPU_OPERATOR_COST / 16.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost of a page the GPU service reads straight from the file (Arrow_Fdw, all-visible heap pages) */
	DefineCustomRealVariable("pg_strom.gpu_direct_seq_page_cost",
							 "Cost for sequential page read by the GPU service",
							 NULL,
							 &pgstrom_gpu_direct_seq_page_cost,
							 DEFAULT_SEQ_PAGE_COST / 4.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/*
 * pgstrom_init_gpu_device
 */
bool
pgstrom_init_gpu_device(void)
{
	pgstrom_collect_gpu_devices();
	if (numGpuDevAttrs > 0)
	{
		pgstrom_init_gpu_options();
		return true;
	}
	return false;
}

/*
 * pgstrom_gpu_device_info - pgstrom.gpu_device_info()
 *
 * One row per attribute of every Vulkan device the probe found (see also
 * pgstrom.gpu_devcaps(), which has the same information in columns).
 */
PG_FUNCTION_INFO_V1(pgstrom_gpu_device_info);
PUBLIC_FUNCTION(Datum)
pgstrom_gpu_device_info(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	const gpuserv_control *ctl = pgstrom_gpuserv_control();

	InitMaterializedSRF(fcinfo, 0);
	if (!ctl)
		return (Datum) 0;
	for (uint32_t i = 0; i < ctl->ndevices; i++)
	{
		const gpuserv_devcaps *dc = &ctl->devcaps[i];
		StringInfoData caps;
		const char *rows[][2] = {
			{"DEV_NAME", NULL},
			{"DEV_UUID", NULL},
			{"DRIVER", NULL},
			{"TIER", NULL},
			{"CAPABILITIES", NULL},
			{"DEV_TOTAL_MEMSZ", NULL},
			{"MAX_STORAGE_BUFFER_RANGE", NULL},
			{"MAX_SHARED_MEMORY_PER_BLOCK", NULL},
			{"MAX_THREADS_PER_BLOCK", NULL},
			{"SUBGROUP_SIZE", NULL},
			{"SELECTED", NULL},
		};
		const char *descs[] = {
			"GPU Device Name", "GPU Device UUID", "Vulkan driver",
			"Capability tier (none, base or desktop)", "Capabilities that passed the probe kernels",
			"GPU Total RAM Size", "Largest storage buffer a kernel can address",
			"Shared memory per workgroup", "Invocations per workgroup",
			"Subgroup size observed in a kernel", "Device used by the GPU service",
		};

		initStringInfo(&caps);
		for (int b = 0; b < GPUCAP_NUM; b++)
			if (dc->caps & (1ULL << b))
				appendStringInfo(&caps, "%s%s", caps.len > 0 ? "," : "", gpuserv_cap_name(b));
		rows[0][1] = dc->name;
		rows[1][1] = dc->uuid;
		rows[2][1] = psprintf("%s %s", dc->driver_name, dc->driver_info);
		rows[3][1] = gpuserv_tier_name(dc->tier);
		rows[4][1] = caps.data;
		rows[5][1] = format_bytesz(dc->device_memory);
		rows[6][1] = format_bytesz(dc->max_storage_buffer_range);
		rows[7][1] = format_bytesz(dc->max_shared_memory);
		rows[8][1] = psprintf("%u", dc->max_wg_invocations);
		rows[9][1] = psprintf("%u", dc->subgroup_size_observed);
		rows[10][1] = (i == ctl->selected_device ? "true" : "false");
		for (int r = 0; r < lengthof(rows); r++)
		{
			Datum		values[4];
			bool		isnull[4] = {false, false, false, false};

			values[0] = Int32GetDatum((int32) i);
			values[1] = CStringGetTextDatum(rows[r][0]);
			values[2] = CStringGetTextDatum(rows[r][1]);
			values[3] = CStringGetTextDatum(descs[r]);
			tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, isnull);
		}
		pfree(caps.data);
	}
	return (Datum) 0;
}

/*
 * pgstrom_license_query - kept so that the older extension scripts can create
 * pgstrom.license_query(); the 7.0 script drops it again.
 */
PG_FUNCTION_INFO_V1(pgstrom_license_query);
PUBLIC_FUNCTION(Datum)
pgstrom_license_query(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}
