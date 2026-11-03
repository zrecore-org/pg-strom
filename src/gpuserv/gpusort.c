/*
 * gpusort.c
 *
 * GPU-Sort: see gpusort.h. The rows are sorted by a bitonic network on the device (kernels_vm/gsort.slang):
 *
 *   keys   n rows of nwords uint32 (encoded so that the unsigned word order is the sort order)
 *   perm   npad = next power of two (at least GS_TILE) row indexes; the ones >= n are padding
 *
 * and one command buffer holds every pass (tile sort, then for each merge size the global steps and the
 * in-tile steps), so the whole sort costs one submit. Inputs below gpusort_min_rows (the cost of the
 * submit and of the upload is not worth it), above the buffer limits, and any device failure use the same
 * encoded keys with a CPU merge sort; the result is identical because the row number is the last
 * tie-breaker, which makes the order total.
 *
 * Thresholds (env PGSTROM_GPUSERV_SORT_MIN overrides the first; PGSTROM_GPUSERV_SORT=cpu|gpu forces an engine):
 *   GPUSORT_MIN_ROWS_DEFAULT  rows needed to use the device
 *   GPUSORT_MAX_ROWS          rows the device sorts (npad up to 2^22: 16 MB of indexes)
 *   keys buffer               at most 512 MB and at most the device's maxStorageBufferRange
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpusort.h"
#include "kernels_spv.h"

#define GS_TILE					512
#define GPUSORT_MIN_ROWS_DEFAULT	2048
#define GPUSORT_MAX_ROWS		(1U << 22)
#define GPUSORT_MAX_KEY_BYTES	((size_t) 512 << 20)

typedef unsigned __int128 u128;

/* ---------------------------------------------------------------- key encoding */
void
gpusort_spec_layout(gpusort_spec *spec)
{
	uint32_t	off = 0;

	for (int k = 0; k < spec->nkeys; k++)
	{
		spec->key[k].woff = off;
		off += 1 + (spec->key[k].wide ? 4 : 2);
	}
	spec->nwords = off;
}

void
gpusort_encode(const gpusort_spec *spec, int k, uint32_t *row, int isnull, __int128 v)
{
	uint32_t   *w = row + spec->key[k].woff;
	int			nv = spec->key[k].wide ? 4 : 2;

	/* the NULL word: NULLs first (0) or last (1), whatever the direction of the value */
	if (spec->key[k].nulls_first)
		w[0] = isnull ? 0 : 1;
	else
		w[0] = isnull ? 1 : 0;
	if (isnull)
	{
		for (int i = 0; i < nv; i++)
			w[1 + i] = 0;
		return;
	}
	if (spec->key[k].wide)
	{
		u128		u = (u128) v ^ ((u128) 1 << 127);

		if (!spec->key[k].asc)
			u = ~u;
		w[1] = (uint32_t)(u >> 96);
		w[2] = (uint32_t)(u >> 64);
		w[3] = (uint32_t)(u >> 32);
		w[4] = (uint32_t) u;
	}
	else
	{
		uint64_t	u = (uint64_t)(int64_t) v ^ ((uint64_t) 1 << 63);

		if (!spec->key[k].asc)
			u = ~u;
		w[1] = (uint32_t)(u >> 32);
		w[2] = (uint32_t) u;
	}
}

/* ---------------------------------------------------------------- CPU sort */
static int
row_cmp(const uint32_t *keys, uint32_t nw, uint32_t a, uint32_t b)
{
	const uint32_t *x = keys + (size_t) a * nw;
	const uint32_t *y = keys + (size_t) b * nw;

	for (uint32_t w = 0; w < nw; w++)
		if (x[w] != y[w])
			return x[w] < y[w] ? -1 : 1;
	return a < b ? -1 : (a > b ? 1 : 0);
}

/* bottom-up merge sort of the row numbers (n log n, no recursion, no qsort_r portability problem) */
static int
cpu_sort(const uint32_t *keys, uint32_t nw, uint32_t n, uint32_t *perm)
{
	uint32_t   *tmp = malloc(sizeof(uint32_t) * (n ? n : 1));
	uint32_t   *src = perm, *dst = tmp;

	if (!tmp)
		return -1;
	for (uint32_t i = 0; i < n; i++)
		perm[i] = i;
	for (uint32_t width = 1; width < n; width *= 2)
	{
		for (uint32_t lo = 0; lo < n; lo += 2 * width)
		{
			uint32_t	mid = lo + width < n ? lo + width : n;
			uint32_t	hi = lo + 2 * width < n ? lo + 2 * width : n;
			uint32_t	a = lo, b = mid, o = lo;

			while (a < mid && b < hi)
				dst[o++] = (row_cmp(keys, nw, src[a], src[b]) <= 0) ? src[a++] : src[b++];
			while (a < mid)
				dst[o++] = src[a++];
			while (b < hi)
				dst[o++] = src[b++];
		}
		{
			uint32_t   *t = src;

			src = dst;
			dst = t;
		}
	}
	if (src != perm)
		memcpy(perm, src, sizeof(uint32_t) * n);
	free(tmp);
	return 0;
}

/* ---------------------------------------------------------------- device sort */
static double
sort_now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct
{
	uint32_t	n, nw, k, j, mode, pad;
} gs_pc;

/* 0 on success (perm[0..n) is the order); -1 with a message in err */
static int
gpu_sort(scan_engine *eng, const uint32_t *keys, uint32_t nw, uint32_t n, uint32_t *perm,
		 char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	uint32_t	npad = GS_TILE;
	gpu_vk_buffer bkeys, bperm;
	gpu_vk_buffer *bufs[2];
	gpu_vk_step *steps = NULL;
	gs_pc	   *pcs = NULL;
	uint32_t	nsteps = 0, maxsteps;
	size_t		keys_bytes = (size_t) n * nw * sizeof(uint32_t);
	int			have_keys = 0, have_perm = 0, rv = -1;
	double		ms = 0.0;

	while (npad < n)
		npad <<= 1;
	if (!eng->sort_pipe_ready)
	{
		if (gpu_vk_pipeline_create(vk, &eng->sort_pipe, kspv_gsort, kspv_gsort_len, 2, sizeof(gs_pc)) != 0)
		{
			snprintf(err, errlen, "cannot create the GPU-Sort pipeline: %s", vk->err);
			return -1;
		}
		eng->sort_pipe_ready = 1;
	}
	if (gpu_vk_buffer_create(vk, &bkeys, keys_bytes ? keys_bytes : 4, 0) != 0)
	{
		snprintf(err, errlen, "GPU-Sort key buffer: %s", vk->err);
		return -1;
	}
	have_keys = 1;
	if (gpu_vk_buffer_create(vk, &bperm, (size_t) npad * sizeof(uint32_t), 0) != 0)
	{
		snprintf(err, errlen, "GPU-Sort index buffer: %s", vk->err);
		goto done;
	}
	have_perm = 1;
	memcpy(bkeys.map, keys, keys_bytes);
	for (uint32_t i = 0; i < npad; i++)
		((uint32_t *) bperm.map)[i] = i;
	/* the passes: a tile sort, then for k = 2 * tile .. npad the global steps and the in-tile steps */
	maxsteps = 1;
	for (uint32_t k = 2 * GS_TILE; k <= npad; k <<= 1)
		for (uint32_t j = k >> 1; j >= GS_TILE; j >>= 1)
			maxsteps += 1 + (j == GS_TILE ? 1 : 0);
	maxsteps += 64;
	steps = calloc(maxsteps, sizeof(*steps));
	pcs = calloc(maxsteps, sizeof(*pcs));
	if (!steps || !pcs)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	bufs[0] = &bkeys;
	bufs[1] = &bperm;
#define GS_STEP(m, kk, jj) do { \
		pcs[nsteps] = (gs_pc){ n, nw, (kk), (jj), (m), 0 }; \
		steps[nsteps] = (gpu_vk_step){ &eng->sort_pipe, bufs, &pcs[nsteps], npad / GS_TILE, 1, 1 }; \
		nsteps++; \
	} while (0)
	GS_STEP(0, 0, 0);
	for (uint32_t k = 2 * GS_TILE; k <= npad; k <<= 1)
	{
		for (uint32_t j = k >> 1; j >= GS_TILE; j >>= 1)
			GS_STEP(1, k, j);
		GS_STEP(2, k, 0);
	}
#undef GS_STEP
	if (gpu_vk_dispatch_seq(vk, steps, nsteps, &ms) != 0)
	{
		snprintf(err, errlen, "GPU-Sort failed: %s", vk->err);
		goto done;
	}
	memcpy(perm, bperm.map, sizeof(uint32_t) * n);
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: gpu-sort: %u rows, %u key words, %u passes, %.2f ms\n",
				n, nw, nsteps, ms);
	rv = 0;
done:
	free(steps);
	free(pcs);
	if (have_perm)
		gpu_vk_buffer_destroy(vk, &bperm);
	if (have_keys)
		gpu_vk_buffer_destroy(vk, &bkeys);
	return rv;
}

void
gpusort_engine_fini(scan_engine *eng)
{
	if (eng->sort_pipe_ready)
		gpu_vk_pipeline_destroy(eng->vk, &eng->sort_pipe);
	eng->sort_pipe_ready = 0;
}

/* ---------------------------------------------------------------- window rank and top-N */
static int
words_differ(const uint32_t *a, const uint32_t *b, uint32_t from, uint32_t to)
{
	for (uint32_t w = from; w < to; w++)
		if (a[w] != b[w])
			return 1;
	return 0;
}

/* compact perm[0..n) to the rows the window filter keeps (in place); returns the count */
static uint32_t
window_filter(const gpusort_spec *spec, const uint32_t *keys, uint32_t n, uint32_t *perm)
{
	uint32_t	nw = spec->nwords;
	uint32_t	part_end = (spec->partby_nkeys > 0)
		? (spec->partby_nkeys < spec->nkeys ? spec->key[spec->partby_nkeys].woff : nw) : 0;
	int			oend_k = spec->partby_nkeys + spec->orderby_nkeys;
	uint32_t	ord_end = oend_k < spec->nkeys ? spec->key[oend_k].woff : nw;
	uint64_t	rn = 0, rank = 0, dense = 0;
	uint32_t	nout = 0;
	const uint32_t *prev = NULL;

	for (uint32_t i = 0; i < n; i++)
	{
		const uint32_t *cur = keys + (size_t) perm[i] * nw;
		uint64_t	v;

		if (!prev || words_differ(prev, cur, 0, part_end))
		{
			rn = 1;
			rank = 1;
			dense = 1;
		}
		else
		{
			rn++;
			if (words_differ(prev, cur, part_end, ord_end))
			{
				rank = rn;
				dense++;
			}
		}
		prev = cur;
		switch (spec->window_func)
		{
			case KSORT_WINDOW_FUNC__ROW_NUMBER:	v = rn; break;
			case KSORT_WINDOW_FUNC__RANK:		v = rank; break;
			default:							v = dense; break;
		}
		if (v < spec->window_limit)
			perm[nout++] = perm[i];
	}
	return nout;
}

/* ---------------------------------------------------------------- entry */
int
gpusort_run(scan_engine *eng, const gpusort_spec *spec, const uint32_t *keys, uint32_t n,
			uint32_t **p_order, uint32_t *p_nout, int *p_used_gpu, char *err, size_t errlen)
{
	uint32_t	nw = spec->nwords;
	uint32_t   *perm = malloc(sizeof(uint32_t) * (n ? n : 1));
	const char *mode = getenv("PGSTROM_GPUSERV_SORT");
	const char *minenv = getenv("PGSTROM_GPUSERV_SORT_MIN");
	uint32_t	min_rows = minenv ? (uint32_t) strtoul(minenv, NULL, 10) : GPUSORT_MIN_ROWS_DEFAULT;
	int			use_gpu = 0;
	double		t0 = sort_now_ms();
	uint32_t	nout = n;

	*p_used_gpu = 0;
	if (!perm)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	if (n >= 2 && n >= min_rows && n <= GPUSORT_MAX_ROWS &&
		(size_t) n * nw * sizeof(uint32_t) <= GPUSORT_MAX_KEY_BYTES &&
		(size_t) n * nw * sizeof(uint32_t) <= (size_t) eng->vk->limits.maxStorageBufferRange &&
		!(mode && strcmp(mode, "cpu") == 0))
		use_gpu = 1;
	if (n >= 2 && mode && strcmp(mode, "gpu") == 0 && n <= GPUSORT_MAX_ROWS)
		use_gpu = 1;
	if (use_gpu)
	{
		char		gerr[256];

		if (gpu_sort(eng, keys, nw, n, perm, gerr, sizeof(gerr)) == 0)
			*p_used_gpu = 1;
		else
		{
			fprintf(stderr, "pgstrom-gpuserv: GPU-Sort fell back to the CPU: %s\n", gerr);
			use_gpu = 0;
		}
	}
	if (!use_gpu)
	{
		if (cpu_sort(keys, nw, n, perm) != 0)
		{
			free(perm);
			snprintf(err, errlen, "out of memory");
			return -1;
		}
		if (eng->trace)
			fprintf(stderr, "pgstrom-gpuserv: cpu-sort: %u rows, %u key words, %.2f ms\n",
					n, nw, sort_now_ms() - t0);
	}
	if (spec->window_func)
		nout = window_filter(spec, keys, n, perm);
	else if (spec->limit > 0 && (uint64_t) spec->limit < nout)
		nout = (uint32_t) spec->limit;
	*p_order = perm;
	*p_nout = nout;
	return 0;
}
