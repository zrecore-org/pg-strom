/*
 * pcache_check.c
 *
 * Time to create the interpreter pipelines of the service (the Metal first-compile cost) with the persisted
 * pipeline cache (gpu_vk.c, pcache_open): make -C src/gpuserv pcache_check &&
 * src/gpuserv/pcache_check [device index [kernel ...]]; test/gpuserv/pcache_test.sh runs it cold and warm.
 * Kernels: gpujoin16, gpujoin16_lite, gpujoin16_nogeom, gpujoin32 (and _lite, _nogeom), gpuscan16, gpuscan16_lite (default: gpujoin16, gpujoin16_lite).
 * Each pipeline is created and given an empty dispatch (0 workgroups, which makes the driver finish its lazy
 * compile); one line "pcache <kernel> create_ms=<n> dispatch_ms=<n>" per kernel.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_vk.h"
#include "kernels_spv.h"

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const struct
{
	const char *name;
	const unsigned char *spv;
	const unsigned int *len;
	uint32_t	nbind, pc;
} ktab[] = {
	{"gpujoin16", kspv_gpujoin16, &kspv_gpujoin16_len, 6, 20},
	{"gpujoin16_lite", kspv_gpujoin16_lite, &kspv_gpujoin16_lite_len, 6, 20},
	{"gpujoin32", kspv_gpujoin32, &kspv_gpujoin32_len, 6, 20},
	{"gpujoin32_lite", kspv_gpujoin32_lite, &kspv_gpujoin32_lite_len, 6, 20},
	{"gpujoin16_nogeom", kspv_gpujoin16_nogeom, &kspv_gpujoin16_nogeom_len, 6, 20},
	{"gpujoin32_nogeom", kspv_gpujoin32_nogeom, &kspv_gpujoin32_nogeom_len, 6, 20},
	{"gpuscan16", kspv_gpuscan16, &kspv_gpuscan16_len, 5, 16},
	{"gpuscan16_lite", kspv_gpuscan16_lite, &kspv_gpuscan16_lite_len, 5, 16},
	{"gpuscan16_nogeom", kspv_gpuscan16_nogeom, &kspv_gpuscan16_nogeom_len, 5, 16},
};

int
main(int argc, char **argv)
{
	gpu_vk_ctx	vk;
	gpu_vk_buffer bufs[6];
	gpu_vk_buffer *bp[6];
	int			dev = argc > 1 ? atoi(argv[1]) : 0;
	const char *defk[] = {"gpujoin16", "gpujoin16_lite"};
	int			nk = argc > 2 ? argc - 2 : 2;
	char		pc[32];
	int			rc = 0;

	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, dev) != 0)
	{
		printf("pcache: cannot open Vulkan device %d: %s\n", dev, vk.err);
		return 1;
	}
	for (int i = 0; i < 6; i++)
	{
		if (gpu_vk_buffer_create(&vk, &bufs[i], 4096, 0) != 0)
		{
			printf("pcache: cannot allocate a buffer: %s\n", vk.err);
			return 1;
		}
		bp[i] = &bufs[i];
	}
	memset(pc, 0, sizeof(pc));
	for (int k = 0; k < nk; k++)
	{
		const char *name = argc > 2 ? argv[2 + k] : defk[k];
		size_t		i;
		gpu_vk_pipeline p;
		double		t0, t1, t2 = 0;

		for (i = 0; i < sizeof(ktab) / sizeof(ktab[0]); i++)
			if (!strcmp(ktab[i].name, name))
				break;
		if (i == sizeof(ktab) / sizeof(ktab[0]))
		{
			printf("pcache: unknown kernel %s\n", name);
			rc = 1;
			continue;
		}
		memset(&p, 0, sizeof(p));
		t0 = now_ms();
		if (gpu_vk_pipeline_create(&vk, &p, ktab[i].spv, *ktab[i].len, ktab[i].nbind, ktab[i].pc) != 0)
		{
			printf("pcache: %s rejected: %s\n", name, vk.err);
			rc = 1;
			continue;
		}
		t1 = now_ms();
		if (gpu_vk_dispatch(&vk, &p, bp, pc, 0, 1, 1, NULL) == 0)
			t2 = now_ms() - t1;
		printf("pcache %s create_ms=%.0f dispatch_ms=%.0f\n", name, t1 - t0, t2);
		gpu_vk_pipeline_destroy(&vk, &p);
	}
	gpu_vk_close(&vk);
	return rc;
}
