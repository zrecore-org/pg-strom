/*
 * gpuserv_proto.h
 *
 * Contract shared by the PostgreSQL module (pg_strom) and the standalone GPU service
 * (pgstrom-gpuserv): the device capability record, the shared control segment, and the
 * text format in which a probe reports its results. Plain C, libc only: it is compiled
 * into both sides, which must not depend on each other's headers.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef GPUSERV_PROTO_H
#define GPUSERV_PROTO_H
#include <stdint.h>
#include <stdio.h>

#define GPUSERV_PROTO_VERSION		1
#define GPUSERV_MAX_DEVICES			8

/*
 * Capability bits. A bit is set in gpuserv_devcaps.caps only if the feature flag, the
 * needed extension AND a probe kernel run on the device all agree: feature flags alone
 * over-report (MoltenVK claims int64 subgroup arithmetic that Metal's compiler rejects).
 * gpuserv_devcaps.reported_caps keeps what the API claimed, to show the difference.
 */
#define GPUCAP_INT64				(1ULL <<  0)	/* 64-bit integer arithmetic */
#define GPUCAP_STORAGE8				(1ULL <<  1)	/* 8-bit loads from storage buffers */
#define GPUCAP_SUBGROUP_BALLOT		(1ULL <<  2)	/* ballot, prefix count, elect, broadcast */
#define GPUCAP_SUBGROUP_ARITH32		(1ULL <<  3)	/* 32-bit subgroup add / min / max */
#define GPUCAP_SUBGROUP_ARITH64		(1ULL <<  4)	/* int64 subgroup add */
#define GPUCAP_ATOMIC_I64_ADD		(1ULL <<  5)	/* 64-bit atomic add on storage buffers */
#define GPUCAP_ATOMIC_I64_MINMAX	(1ULL <<  6)	/* 64-bit atomic min / max */
#define GPUCAP_ATOMIC_I64_CAS		(1ULL <<  7)	/* 64-bit atomic compare-exchange */
#define GPUCAP_FP64					(1ULL <<  8)	/* double precision arithmetic */
#define GPUCAP_ATOMIC_F32_ADD		(1ULL <<  9)	/* float32 atomic add */
#define GPUCAP_BDA					(1ULL << 10)	/* buffer device address (raw pointers) */
#define GPUCAP_MEMORY_MODEL			(1ULL << 11)	/* Vulkan memory model, device scope */
#define GPUCAP_HOST_IMPORT			(1ULL << 12)	/* VK_EXT_external_memory_host works */
#define GPUCAP_UNIFIED_MEMORY		(1ULL << 13)	/* device-local + host-visible + coherent */
#define GPUCAP_FP16					(1ULL << 14)	/* shaderFloat16 */
#define GPUCAP_NUM					15

/* tiers, see the plan: base = Apple-class, desktop = base + 64-bit atomics + fp64 */
#define GPUSERV_TIER_NONE			0
#define GPUSERV_TIER_BASE			1
#define GPUSERV_TIER_DESKTOP		2

typedef struct gpuserv_devcaps
{
	char		name[128];
	char		driver_name[64];
	char		driver_info[96];
	char		uuid[36];				/* 32 hex digits */
	char		tier_reason[160];		/* why tier is NONE, or a note */
	uint32_t	vendor_id;
	uint32_t	device_id;
	uint32_t	device_type;			/* VkPhysicalDeviceType: 1 integrated, 2 discrete, 4 cpu */
	uint32_t	api_version;			/* VK_MAKE_API_VERSION */
	uint32_t	driver_version;
	uint32_t	tier;					/* GPUSERV_TIER_* */
	uint64_t	caps;					/* probed, usable */
	uint64_t	reported_caps;			/* what feature flags / extensions claimed */
	uint32_t	subgroup_size_min;
	uint32_t	subgroup_size_max;
	uint32_t	subgroup_size_observed;	/* WaveGetLaneCount() measured in a kernel; 0 if unknown */
	uint32_t	max_shared_memory;		/* bytes per workgroup */
	uint32_t	max_wg_invocations;
	uint32_t	max_push_constants;
	uint32_t	host_import_alignment;	/* 0 if host import is unavailable */
	uint32_t	probes_ok;
	uint32_t	probes_failed;
	uint64_t	max_storage_buffer_range;
	uint64_t	max_buffer_size;		/* 0 if not reported */
	uint64_t	device_memory;			/* bytes in the largest device-local heap */
} gpuserv_devcaps;

/*
 * Shared control segment: a named POSIX shared-memory object created by the postmaster
 * (no threads needed), inherited by the launcher and the backends through fork(), and
 * attached by name by the exec'd service. Fields written by more than one process are
 * accessed with __atomic builtins.
 */
#define GPUSERV_CONTROL_MAGIC		0x50475356U		/* 'PGSV' */

#define GPUSERV_STATE_STOPPED		0
#define GPUSERV_STATE_STARTING		1
#define GPUSERV_STATE_READY			2
#define GPUSERV_STATE_FAILED		3

typedef struct gpuserv_control
{
	uint32_t	magic;
	uint32_t	version;				/* GPUSERV_PROTO_VERSION */
	uint32_t	ndevices;
	uint32_t	selected_device;		/* index served by pgstrom-gpuserv, or UINT32_MAX */
	gpuserv_devcaps	devcaps[GPUSERV_MAX_DEVICES];
	/* service state: the launcher and the service write, backends read */
	int32_t		state;					/* GPUSERV_STATE_* */
	int32_t		service_pid;
	uint32_t	restart_count;
	uint32_t	generation;				/* bumped on each (re)start of the service */
	char		socket_path[104];
	char		last_error[256];
} gpuserv_control;

/*
 * Text format of a probe (one 'key=value' per line, then a line 'end'):
 *   proto=1
 *   ndevices=<n>
 *   dev<i>.<field>=<value>       for every field of gpuserv_devcaps
 *   dev<i>.caps=int64,storage8   probed capabilities by name (also dev<i>.reported=)
 *   end
 */
extern const char *gpuserv_cap_name(int bit);
extern const char *gpuserv_tier_name(uint32_t tier);
extern void gpuserv_devcaps_write(FILE *out, int index, const gpuserv_devcaps *dc);
extern void gpuserv_probe_write_header(FILE *out, int ndevices);
extern void gpuserv_probe_write_footer(FILE *out);
/*
 * Parse a complete probe text. Returns the number of devices (0 or more) or -1 if the
 * text is malformed, truncated (no 'end') or of another protocol version. At most
 * 'max' devices are stored.
 */
extern int	gpuserv_probe_parse(const char *text, gpuserv_devcaps *devs, int max);

/*
 * Socket protocol between backends and the service. A1 implements the handshake only;
 * the task commands arrive with the scan milestone (A2).
 */
#define GPUSERV_SOCK_MAGIC			0x50534b54U		/* 'PSKT' */
#define GPUSERV_CMD_PING			1

typedef struct gpuserv_sock_req
{
	uint32_t	magic;					/* GPUSERV_SOCK_MAGIC */
	uint32_t	cmd;					/* GPUSERV_CMD_* */
} gpuserv_sock_req;

typedef struct gpuserv_sock_pong
{
	uint32_t	magic;
	uint32_t	status;					/* 0 = ok */
	int32_t		pid;					/* pid of the service */
	uint32_t	generation;
	uint32_t	device;					/* device index being served */
	uint32_t	tier;					/* GPUSERV_TIER_* of that device */
} gpuserv_sock_pong;

#endif	/* GPUSERV_PROTO_H */
