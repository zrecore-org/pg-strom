/*
 * gpu_cache.c
 *
 * GPU-Cache is not available in the Vulkan build yet (milestone A7 of
 * .claude/hw-acceleration.md). This file keeps the symbols the rest of the module and the
 * SQL scripts refer to: no table has a GPU-Cache, so the planner never offers one, and
 * the SQL functions of the extension report that the feature is unavailable. The CUDA
 * implementation is on the master branch.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

bool
baseRelHasGpuCache(PlannerInfo *root, RelOptInfo *baserel)
{
	return false;
}

bool
pgstromGpuCacheExecInit(pgstromTaskScanState *ptss)
{
	return false;
}

XpuCommand *
pgstromScanChunkGpuCache(pgstromTaskState *pts,
						 pgstromTaskScanState *ptss,
						 struct iovec *xcmd_iov,
						 int *xcmd_iovcnt)
{
	elog(ERROR, "GPU-Cache is not available in this build");
	return NULL;
}

void
pgstromGpuCacheExecReset(pgstromTaskScanState *ptss)
{
}

void
pgstromGpuCacheInitDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
}

void
pgstromGpuCacheAttachDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
}

void
pgstromGpuCacheShutdownDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
}

void
pgstromGpuCacheExplain(pgstromTaskScanState *ptss, ExplainState *es, const char *prefix)
{
}

void
pgstrom_init_gpu_cache(void)
{
}

/* ----------------------------------------------------------------
 * SQL functions created by the extension scripts
 * ----------------------------------------------------------------
 */
#define GPUCACHE_UNAVAILABLE(NAME)											\
	PG_FUNCTION_INFO_V1(NAME);												\
	PUBLIC_FUNCTION(Datum)													\
	NAME(PG_FUNCTION_ARGS)													\
	{																		\
		elog(ERROR, "GPU-Cache is not available in this build (%s)", #NAME);	\
		PG_RETURN_NULL();													\
	}
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_sync_trigger)
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_apply_redo)
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_compaction)
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_recovery)
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_info_one)
GPUCACHE_UNAVAILABLE(pgstrom_gpucache_info)
