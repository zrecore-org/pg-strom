/*
 * Self-check of the string hash of the join kernels (kernels_vm/hash_bytes.inc): hash_bytes_src() on the device
 * against the values PostgreSQL's hashtext() gives (hash_any of the bytes) for strings of 0..40 bytes, each placed
 * at the four byte alignments of the buffer. The expected values were taken from PostgreSQL 17:
 *   select hashtext(s) from the strings s(len) = chr(97 + (j * 7 + len) % 26), j = 0 .. len - 1.
 * Run: sh test/gpuserv/hash_check.sh [device index]   (no cluster, no test lock)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gpu_vk.h"

static const uint32_t expected[41] = {
	2817148525u, 4226146208u, 708958924u, 3206444747u, 819764728u, 4224083010u, 251923159u, 2817713120u, 2227928894u,
	267974497u, 1937479660u, 2282924951u, 2921133474u, 4073427855u, 3877415075u, 788684301u, 2106967128u, 3881878423u,
	1709970857u, 986273737u, 2108414062u, 2070550555u, 1196058611u, 3001894885u, 2989711582u, 2920530200u, 2697247785u,
	1342531128u, 2639016508u, 655986102u, 1648442431u, 802713113u, 1712185080u, 3139702578u, 1568803884u, 2629137554u,
	1785075746u, 3088435480u, 3427257329u, 3937369481u, 2569799146u
};

int
main(int argc, char **argv)
{
	int			dev = argc > 1 ? atoi(argv[1]) : 0;
	const char *spvpath = argc > 2 ? argv[2] : "build/hash_check.spv";
	const uint32_t ncase = 41 * 4;
	gpu_vk_ctx	vk;
	gpu_vk_buffer ba, br;
	gpu_vk_pipeline pl;
	gpu_vk_buffer *bufs[2];
	FILE	   *fp;
	void	   *spv;
	long		spvlen;
	int			bad = 0;
	uint32_t	pc[1];

	if (!(fp = fopen(spvpath, "rb")))
	{
		printf("cannot open %s\n", spvpath);
		return 2;
	}
	fseek(fp, 0, SEEK_END);
	spvlen = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	spv = malloc(spvlen);
	if (fread(spv, 1, spvlen, fp) != (size_t) spvlen)
		return 2;
	fclose(fp);
	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, dev) != 0)
	{
		printf("cannot open Vulkan device %d: %s\n", dev, vk.err);
		return 2;
	}
	if (gpu_vk_buffer_create(&vk, &ba, ncase * 64 + 16, 0) || gpu_vk_buffer_create(&vk, &br, ncase * 4, 0))
	{
		printf("cannot allocate buffers: %s\n", vk.err);
		return 2;
	}
	if (gpu_vk_pipeline_create(&vk, &pl, spv, spvlen, 2, 4) != 0)
	{
		printf("the device's shader compiler rejected the kernel: %s\n", vk.err);
		return 2;
	}
	memset(ba.map, 0xEE, ncase * 64 + 16);		/* bytes outside the string must not matter */
	for (uint32_t c = 0; c < ncase; c++)
	{
		uint32_t	len = c >> 2, al = c & 3;
		unsigned char *p = (unsigned char *) ba.map + c * 64 + al;

		for (uint32_t j = 0; j < len; j++)
			p[j] = (unsigned char) ('a' + (j * 7 + len) % 26);
	}
	bufs[0] = &ba;
	bufs[1] = &br;
	pc[0] = ncase;
	memset(br.map, 0xA5, ncase * 4);
	if (gpu_vk_dispatch(&vk, &pl, bufs, pc, (ncase + 63) / 64, 1, 1, NULL) != 0)
	{
		printf("dispatch failed: %s\n", vk.err);
		return 2;
	}
	for (uint32_t c = 0; c < ncase; c++)
	{
		uint32_t	h = ((uint32_t *) br.map)[c];

		if (h != expected[c >> 2])
		{
			printf("  FAIL len %u align %u: device %u, PostgreSQL %u\n", c >> 2, c & 3, h, expected[c >> 2]);
			bad++;
		}
	}
	gpu_vk_pipeline_destroy(&vk, &pl);
	gpu_vk_buffer_destroy(&vk, &ba);
	gpu_vk_buffer_destroy(&vk, &br);
	gpu_vk_close(&vk);
	printf(bad ? "HASH CHECK: %d FAILED\n" : "HASH CHECK: ALL PASSED (%d cases)\n", bad ? bad : (int) ncase);
	return bad ? 1 : 0;
}
