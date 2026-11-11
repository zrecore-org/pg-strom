/*
 * kprewarm.h
 *
 * A9 part 3: first-use kernel compile. The GPU driver compiles each SPIR-V kernel for the user's GPU when its pipeline is
 * first created (Metal: seconds for the lite and nogeom tiers, minutes for the full ones) and keeps the result in the
 * persisted pipeline cache (gpu_vk.c). This module
 *   - tracks which interpreter kernels are compiled ("ready": creating their pipeline takes milliseconds), in the
 *     control segment (gpuserv_prewarm_state) and in a manifest file next to the pipeline cache file;
 *   - runs a low-priority thread that compiles the selected set one pipeline at a time, never while a task runs;
 *   - installs the readiness gate of gpu_vk_pipeline_create(): a task whose kernel is not ready waits for its
 *     compile within the session's budget and is then refused, so that the backend continues on the CPU (A14) instead
 *     of stalling for minutes.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef GPUSERV_KPREWARM_H
#define GPUSERV_KPREWARM_H
#include <stdint.h>
#include "gpu_vk.h"
#include "gpuserv_proto.h"

/* session budget meaning "no gate: create the pipeline in place and wait as long as it takes" */
#define KPW_WAIT_NONE			UINT32_MAX

/* "off", "lite", "geometry", "all" -> GPUSERV_PREWARM_*; -1 for anything else */
int		kpw_parse_mode(const char *s);

/*
 * Start the prewarm thread and install the gate on 'vk' (service side). 'st' is the state in the control segment (may
 * be a private struct). Does nothing and returns 0 without a persisted pipeline cache (nothing could be kept).
 * Environment: PGSTROM_GPUSERV_PREWARM (off|lite|geometry|all) wins over 'mode', PGSTROM_GPUSERV_PREWARM_IDLE_MS (the
 * quiet time before the thread compiles the next kernel of the set, default 2000), PGSTROM_GPUSERV_PREWARM_DELAY_MS (before the
 * first, default 3000), PGSTROM_GPUSERV_KERNEL_WAIT_MS (-1: no gate whatever the sessions say).
 */
int		kpw_start(gpu_vk_ctx *vk, gpuserv_prewarm_state *st, int mode);
/* Stop the thread. Returns 1 if it ended, 0 if it is still inside a compile (the caller must not close the device). */
int		kpw_stop(int wait_ms);
/* the budget of the session being served (kern_session_info.kernel_wait_ms: 0 = no gate, else the budget in ms + 1) */
void	kpw_set_session_wait(uint32_t session_field);
/* a task starts / ends: the thread compiles the set only while the service is quiet */
void	kpw_task_begin(void);
void	kpw_task_end(void);

/*
 * Command line (--prewarm-all): compile a set one kernel after the other in this process, print progress, keep the
 * manifest. 'list_only' prints what would be compiled and whether it is ready. Returns 0 if every kernel is ready.
 */
int		kpw_cli(gpu_vk_ctx *vk, int mode, int list_only);

#endif	/* GPUSERV_KPREWARM_H */
