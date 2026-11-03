/*
 * gpusort_check.c
 *
 * Self-check of GPU-Sort (make -C src/gpuserv gpusort_check && ./gpusort_check [device index]): random
 * keys of every shape the service builds (64-bit and 128-bit values, ASC/DESC, NULLs first/last, many
 * ties) are sorted on the device and by the CPU merge sort and must give the same order; the order is
 * also verified against a direct comparison of neighbours, and the top-N cut and the window-rank filter
 * are checked against a naive reference. Exit status 0: all passed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpusort.h"

static uint64_t rng = 88172645463325252ULL;
static uint64_t
rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return rng;
}

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int
rowcmp(const gpusort_spec *spec, const uint32_t *keys, uint32_t a, uint32_t b)
{
	for (uint32_t w = 0; w < spec->nwords; w++)
	{
		uint32_t	x = keys[(size_t) a * spec->nwords + w], y = keys[(size_t) b * spec->nwords + w];

		if (x != y)
			return x < y ? -1 : 1;
	}
	return a < b ? -1 : (a > b ? 1 : 0);
}

/* naive window reference: quadratic per row, from the definitions */
static uint32_t
window_ref(const gpusort_spec *spec, const uint32_t *keys, const uint32_t *sorted, uint32_t n, uint32_t *out)
{
	uint32_t	nw = spec->nwords;
	uint32_t	pe = spec->partby_nkeys > 0 ? spec->key[spec->partby_nkeys].woff : 0;
	int			ok = spec->partby_nkeys + spec->orderby_nkeys;
	uint32_t	oe = ok < spec->nkeys ? spec->key[ok].woff : nw;
	uint32_t	nout = 0;

	for (uint32_t i = 0; i < n; i++)
	{
		const uint32_t *ci = keys + (size_t) sorted[i] * nw;
		uint64_t	rn = 0, less = 0;
		uint64_t	dense = 0;

		for (uint32_t j = 0; j < n; j++)
		{
			const uint32_t *cj = keys + (size_t) sorted[j] * nw;
			int			same_part = (memcmp(ci, cj, pe * 4) == 0);

			if (!same_part)
				continue;
			if (j <= i)
				rn++;
			if (memcmp(cj + pe, ci + pe, (oe - pe) * 4) != 0 && j < i)
				less++;
		}
		/* rank = 1 + rows of the partition that sort before the first peer of this row */
		{
			uint64_t	first_peer = 0;

			for (uint32_t j = 0; j <= i; j++)
			{
				const uint32_t *cj = keys + (size_t) sorted[j] * nw;

				if (memcmp(ci, cj, pe * 4) == 0 && memcmp(cj + pe, ci + pe, (oe - pe) * 4) == 0)
				{
					first_peer = j;
					break;
				}
			}
			less = 0;
			for (uint32_t j = 0; j < first_peer; j++)
				if (memcmp(ci, keys + (size_t) sorted[j] * nw, pe * 4) == 0)
					less++;
		}
		{
			/* dense rank: distinct order values before this one in the partition, plus one */
			for (uint32_t j = 0; j <= i; j++)
			{
				const uint32_t *cj = keys + (size_t) sorted[j] * nw;
				int			isfirst = 1;

				if (memcmp(ci, cj, pe * 4) != 0)
					continue;
				for (uint32_t m = 0; m < j; m++)
				{
					const uint32_t *cm = keys + (size_t) sorted[m] * nw;

					if (memcmp(cj, cm, pe * 4) == 0 && memcmp(cj + pe, cm + pe, (oe - pe) * 4) == 0)
					{
						isfirst = 0;
						break;
					}
				}
				dense += isfirst;
			}
		}
		{
			uint64_t	v = (spec->window_func == KSORT_WINDOW_FUNC__ROW_NUMBER ? rn :
							 spec->window_func == KSORT_WINDOW_FUNC__RANK ? less + 1 : dense);

			if (v < spec->window_limit)
				out[nout++] = sorted[i];
		}
	}
	return nout;
}

static void
run_case(scan_engine *eng, int nkeys, const int *wide, const int *asc, const int *nf, uint32_t n,
		 uint32_t range, int window, int limit)
{
	gpusort_spec spec;
	uint32_t   *keys, *order = NULL, *cpu_order = NULL, nout = 0, cnout = 0;
	int			used_gpu = 0;
	char		err[256] = "";

	memset(&spec, 0, sizeof(spec));
	spec.nkeys = nkeys;
	for (int k = 0; k < nkeys; k++)
	{
		spec.key[k].asc = asc[k];
		spec.key[k].nulls_first = nf[k];
		spec.key[k].wide = wide[k];
	}
	gpusort_spec_layout(&spec);
	if (window)
	{
		spec.window_func = window;
		spec.window_limit = 3;
		spec.partby_nkeys = 1;
		spec.orderby_nkeys = nkeys > 1 ? 1 : 0;
		if (nkeys < 2)
			spec.window_func = 0;
	}
	spec.limit = limit;
	keys = malloc(sizeof(uint32_t) * spec.nwords * (n ? n : 1));
	for (uint32_t r = 0; r < n; r++)
		for (int k = 0; k < nkeys; k++)
		{
			int			isnull = (rnd() % 13 == 0);
			__int128	v = (__int128)(int64_t)(rnd() % range) - (__int128)(range / 2);

			if (wide[k] && rnd() % 5 == 0)
				v = v * ((__int128) 1 << 70) + (__int128)(rnd() % 3);
			if (!wide[k] && rnd() % 7 == 0)
				v = (rnd() & 1) ? INT64_MIN : INT64_MAX;
			gpusort_encode(&spec, k, keys + (size_t) r * spec.nwords, isnull, v);
		}
	setenv("PGSTROM_GPUSERV_SORT", "gpu", 1);
	if (gpusort_run(eng, &spec, keys, n, &order, &nout, &used_gpu, err, sizeof(err)) != 0)
	{
		CHECK(0, "gpu run n=%u: %s", n, err);
		free(keys);
		return;
	}
	setenv("PGSTROM_GPUSERV_SORT", "cpu", 1);
	if (gpusort_run(eng, &spec, keys, n, &cpu_order, &cnout, &used_gpu, err, sizeof(err)) != 0)
	{
		CHECK(0, "cpu run n=%u: %s", n, err);
		free(keys);
		free(order);
		return;
	}
	CHECK(nout == cnout && (nout == 0 || memcmp(order, cpu_order, sizeof(uint32_t) * nout) == 0),
		  "GPU and CPU orders differ (n=%u nkeys=%d window=%d limit=%d nout=%u/%u)", n, nkeys, window, limit, nout, cnout);
	if (!spec.window_func && limit == 0)
	{
		CHECK(nout == n, "row count %u != %u", nout, n);
		for (uint32_t i = 1; i < nout; i++)
			if (rowcmp(&spec, keys, order[i - 1], order[i]) >= 0)
			{
				CHECK(0, "order[%u] is not above order[%u] (n=%u)", i, i - 1, n);
				break;
			}
	}
	if (limit > 0 && !spec.window_func)
		CHECK(nout == (n < (uint32_t) limit ? n : (uint32_t) limit), "limit count %u", nout);
	if (spec.window_func && n <= 1500)
	{
		uint32_t   *full = NULL, fn = 0, *ref = malloc(sizeof(uint32_t) * (n ? n : 1)), rn;
		gpusort_spec s2 = spec;

		s2.window_func = 0;
		setenv("PGSTROM_GPUSERV_SORT", "cpu", 1);
		gpusort_run(eng, &s2, keys, n, &full, &fn, &used_gpu, err, sizeof(err));
		rn = window_ref(&spec, keys, full, n, ref);
		CHECK(rn == nout && (rn == 0 || memcmp(ref, order, sizeof(uint32_t) * rn) == 0),
			  "window filter differs from the reference (func %c, n=%u: %u vs %u rows)", spec.window_func, n, nout, rn);
		free(full);
		free(ref);
	}
	free(keys);
	free(order);
	free(cpu_order);
}

int
main(int argc, char **argv)
{
	gpu_vk_ctx	vk;
	scan_engine	eng;
	static const uint32_t sizes[] = {2, 3, 100, 511, 512, 513, 1000, 1024, 1025, 3000, 4096, 5000, 20000, 70000};

	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, argc > 1 ? atoi(argv[1]) : 0) != 0)
	{
		printf("cannot open the Vulkan device: %s\n", vk.err);
		return 2;
	}
	scan_engine_init(&eng, &vk);
	eng.trace = getenv("GPUSORT_CHECK_TRACE") != NULL;
	for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
	{
		uint32_t	n = sizes[si];
		int			w1[1] = {0}, a1[1] = {1}, f1[1] = {0};
		int			w2[2] = {0, 1}, a2[2] = {0, 1}, f2[2] = {1, 0};
		int			w3[3] = {1, 0, 0}, a3[3] = {1, 0, 1}, f3[3] = {0, 0, 1};

		run_case(&eng, 1, w1, a1, f1, n, 10, 0, 0);
		run_case(&eng, 1, w1, a1, f1, n, 1000000, 0, 0);
		run_case(&eng, 2, w2, a2, f2, n, 5, 0, 0);
		run_case(&eng, 2, w2, a2, f2, n, 1000, 0, 7);
		run_case(&eng, 3, w3, a3, f3, n, 3, 0, 0);
		run_case(&eng, 3, w3, a3, f3, n, 50, 0, 100);
		run_case(&eng, 2, w2, a2, f2, n, 4, KSORT_WINDOW_FUNC__ROW_NUMBER, 0);
		run_case(&eng, 2, w2, a2, f2, n, 4, KSORT_WINDOW_FUNC__RANK, 0);
		run_case(&eng, 2, w2, a2, f2, n, 4, KSORT_WINDOW_FUNC__DENSE_RANK, 0);
		run_case(&eng, 3, w3, a3, f3, n, 3, KSORT_WINDOW_FUNC__RANK, 0);
		printf("n=%u done\n", n);
	}
	scan_engine_fini(&eng);
	gpu_vk_close(&vk);
	if (fails)
	{
		printf("gpusort_check: %d FAILED\n", fails);
		return 1;
	}
	printf("gpusort_check: all passed\n");
	return 0;
}
