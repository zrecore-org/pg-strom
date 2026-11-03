/*
 * parquet_cache.c
 *
 * The Parquet disk cache (decoded column chunks kept on local storage and read by
 * GPU-Direct SQL) is not available in the Vulkan build: it depends on GPU-Direct. It comes
 * back with the Arrow/Parquet milestone (A5), reading the cache through the GPU service's
 * host buffers. Until then lookups miss and written chunks are dropped. The CUDA
 * implementation is on the master branch.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

void *
parquetCacheLookup(const struct stat *pq_fstat, int32_t rg_index, int32_t field_id)
{
	return NULL;
}

void
parquetCacheRelease(void *entry)
{
}

size_t
parquetCacheLength(const void *entry)
{
	return 0;
}

void
parquetCacheWriteAsync(parquetCacheWriteAsyncData *data)
{
	releaseParquetCacheWriteAsyncData(data);
}

bool
pgstrom_parquet_cache_enabled(void)
{
	return false;
}

void
pgstrom_init_parquet_cache(void)
{
}

/*
 * pgstrom_parquet_cache_info - pgstrom.parquet_cache_info()
 */
PG_FUNCTION_INFO_V1(pgstrom_parquet_cache_info);
PUBLIC_FUNCTION(Datum)
pgstrom_parquet_cache_info(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(cstring_to_text("{ \"cache_path\" : null }"));
}
