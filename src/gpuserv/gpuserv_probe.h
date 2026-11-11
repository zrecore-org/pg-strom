/*
 * gpuserv_probe.h
 *
 * Measure what each Vulkan device really supports by running small probe kernels.
 */
#ifndef GPUSERV_PROBE_H
#define GPUSERV_PROBE_H
#include <stdio.h>
#include "gpuserv_proto.h"

/*
 * Enumerate and probe every Vulkan device (at most 'max', in enumeration order), filling
 * every field of gpuserv_devcaps including the tier. CPU devices (e.g. lavapipe) are
 * probed but get tier NONE unless allow_cpu is non-zero. 'log' (may be NULL) receives
 * one human-readable line per probe. Returns the number of devices (>= 0, zero if there
 * is no Vulkan driver) or -1 on a hard failure (see gpuserv_probe_last_error()).
 */
extern int gpuserv_probe_devices(gpuserv_devcaps *out, int max, int allow_cpu, FILE *log);

/* message of the last hard failure (static buffer, not thread safe) */
extern const char *gpuserv_probe_last_error(void);

/* Compute the tier and tier_reason of dc from dc->caps and the limits. */
extern void gpuserv_devcaps_compute_tier(gpuserv_devcaps *dc, int allow_cpu);

/*
 * A10 part 6: 0 when the device computes the error-free transformations of kernels_vm/df64.inc bit for bit as the host (the
 * df64 prefilter of float8 comparisons may be used), -1 otherwise (message in why). Needs the device opened (gpu_vk.h).
 */
struct gpu_vk_ctx;
extern int gpuserv_df64_selfcheck(struct gpu_vk_ctx *ctx, char *why, size_t whylen);

#endif	/* GPUSERV_PROBE_H */
