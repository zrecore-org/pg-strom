/*
 * rsort_check.c
 *
 * Self-check of the radix sort kernels (kernels_vm/rsort.slang, ported from the mongo fork under Apache-2.0):
 * make -C src/gpuserv rsort_check && src/gpuserv/rsort_check [device index]. Random 64-bit keys of several shapes
 * (full range, few distinct values, already sorted, reversed, all equal) and sizes around the block size (4096) are
 * sorted on the device (16 passes of count / scan / scatter, ping-pong buffers) and must equal the stable order of
 * qsort on (key, index). One line per case with the device time; exit status 0: all passed.
 * Run it under the GPU lock with PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=<an empty directory>: opening the device otherwise
 * recompiles the whole persisted pipeline cache (1.6 GB of full-tier kernels, minutes on Metal).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "gpu_vk.h"
#include "kernels_spv.h"

typedef struct { uint64_t k; uint32_t i; } kv;
typedef struct { uint32_t n, B, pass, pad; } rs_pc;

static uint64_t rng = 88172645463325252ULL;
static uint64_t
rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return rng;
}

static int
kvcmp(const void *a, const void *b)
{
	const kv   *x = a, *y = b;

	if (x->k != y->k)
		return x->k < y->k ? -1 : 1;
	return x->i < y->i ? -1 : (x->i > y->i ? 1 : 0);
}

int
main(int argc, char **argv)
{
	gpu_vk_ctx	vk;
	gpu_vk_pipeline pipe[2][3];		/* [pass parity][count, scan, scatter] */
	const unsigned char *spv[3] = {kspv_rsort_count, kspv_rsort_scan, kspv_rsort_scatter};
	const unsigned int *len[3] = {&kspv_rsort_count_len, &kspv_rsort_scan_len, &kspv_rsort_scatter_len};
	static const uint32_t sizes[] = {1, 2, 255, 4095, 4096, 4097, 8192, 100000, 1000000};
	static const char *shapes[] = {"full range", "few distinct", "sorted", "reversed", "all equal"};
	int			dev = argc > 1 ? atoi(argv[1]) : 0;
	int			fails = 0;
	size_t		maxn = 1000000, maxB = (maxn + 4095) / 4096;
	gpu_vk_buffer K[2], I[2], H;

	setvbuf(stdout, NULL, _IOLBF, 0);
	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, dev) != 0)
	{
		printf("rsort_check: cannot open Vulkan device %d: %s\n", dev, vk.err);
		return 1;
	}
	printf("opened\n");
	for (int j = 0; j < 2; j++)
		if (gpu_vk_buffer_create(&vk, &K[j], maxn * 8, 0) != 0 || gpu_vk_buffer_create(&vk, &I[j], maxn * 4, 0) != 0)
		{
			printf("rsort_check: buffer: %s\n", vk.err);
			return 1;
		}
	if (gpu_vk_buffer_create(&vk, &H, 16 * maxB * 4, 0) != 0)
	{
		printf("rsort_check: buffer: %s\n", vk.err);
		return 1;
	}
	printf("buffers\n");
	for (int par = 0; par < 2; par++)
		for (int s = 0; s < 3; s++)
			if (gpu_vk_pipeline_create(&vk, &pipe[par][s], spv[s], *len[s], 5, sizeof(rs_pc)) != 0)
			{
				printf("rsort_check: pipeline %d: %s\n", s, vk.err);
				return 1;
			}
	printf("pipelines\n");
	for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
	{
		uint32_t	n = sizes[si], B = (n + 4095) / 4096;

		for (int sh = 0; sh < 5; sh++)
		{
			kv		   *ref = malloc(sizeof(kv) * n);
			uint64_t   *kin = (uint64_t *) K[0].map;
			gpu_vk_step *steps = calloc(48, sizeof(*steps));
			rs_pc	   *pcs = calloc(48, sizeof(*pcs));
			gpu_vk_buffer *bufs[2][5] = {{&K[0], &H, &I[0], &K[1], &I[1]}, {&K[1], &H, &I[1], &K[0], &I[0]}};
			double		ms = 0;
			int			ns = 0, bad = 0;

			for (uint32_t i = 0; i < n; i++)
			{
				uint64_t	v;

				switch (sh)
				{
					case 0: v = rnd(); break;
					case 1: v = (rnd() % 7) * 0x0101010101010101ULL; break;
					case 2: v = (uint64_t) i * 3; break;
					case 3: v = (uint64_t) (n - i) * 5; break;
					default: v = 42; break;
				}
				kin[i] = v;
				ref[i].k = v;
				ref[i].i = i;
			}
			qsort(ref, n, sizeof(kv), kvcmp);
			for (uint32_t p = 0; p < 16; p++)
				for (int s = 0; s < 3; s++)
				{
					pcs[ns] = (rs_pc){n, B, p, 0};
					steps[ns] = (gpu_vk_step){&pipe[p & 1][s], bufs[p & 1], &pcs[ns], s == 1 ? 1 : B, 1, 1};
					ns++;
				}
			if (gpu_vk_dispatch_seq(&vk, steps, ns, &ms) != 0)
			{
				printf("rsort_check: dispatch failed: %s\n", vk.err);
				return 1;
			}
			/* 16 passes: the result is back in K[0], I[0] */
			for (uint32_t i = 0; i < n && bad < 3; i++)
				if (((uint64_t *) K[0].map)[i] != ref[i].k || ((uint32_t *) I[0].map)[i] != ref[i].i)
				{
					printf("  FAIL n=%u %s: position %u: got key %llx idx %u, want key %llx idx %u\n", n, shapes[sh], i,
						   (unsigned long long) ((uint64_t *) K[0].map)[i], ((uint32_t *) I[0].map)[i],
						   (unsigned long long) ref[i].k, ref[i].i);
					bad++;
				}
			fails += bad ? 1 : 0;
			printf("  %s n=%-8u %-13s %7.2f ms\n", bad ? "FAIL" : "ok  ", n, shapes[sh], ms);
			free(ref);
			free(steps);
			free(pcs);
		}
	}
	printf("rsort_check: %s\n", fails ? "FAILED" : "all passed");
	return fails ? 1 : 0;
}
