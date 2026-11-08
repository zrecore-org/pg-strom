/*
 * gpu_client.c
 *
 * The backend's side of the GPU service: whether it accepts work, how many tasks a backend
 * may have in flight, and the connection to it. The service itself is the exec'd
 * pgstrom-gpuserv program run by the launcher (gpu_launcher.c).
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "gpuserv/gpuserv_proto.h"
#include <fcntl.h>

static int		pgstrom_max_async_tasks_guc;		/* GUC */
int				pgstrom_debug_level = 0;			/* GUC */

int
pgstrom_max_async_tasks(void)
{
	return pgstrom_max_async_tasks_guc;
}

/* does the GPU service accept sessions now? (its state is in the control segment) */
bool
gpuserv_ready_accept(void)
{
	const gpuserv_control *ctl = pgstrom_gpuserv_control();

	return (ctl && __atomic_load_n(&ctl->state, __ATOMIC_SEQ_CST) == GPUSERV_STATE_READY);
}

/*
 * gpuClientOpenSession - connect to the service and open the session of a task state
 */
void
gpuClientOpenSession(pgstromTaskState *pts, const XpuCommand *session)
{
	const gpuserv_control *ctl = pgstrom_gpuserv_control();
	struct sockaddr_un addr;
	pgsocket	sockfd;

	char		reason[MAXPGPATH + 100];

	if (!ctl || __atomic_load_n(&ctl->state, __ATOMIC_SEQ_CST) != GPUSERV_STATE_READY)
	{
		if (pts->cpu_fallback_ok)
		{
			pgstromCpuModeBegin(pts, "the GPU service is not ready");
			return;
		}
		elog(ERROR, "the GPU service is not ready");
	}
	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", ctl->socket_path);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		int		save_errno = errno;

		close(sockfd);
		errno = save_errno;
		if (pts->cpu_fallback_ok)
		{
			snprintf(reason, sizeof(reason), "failed on connect('%s'): %s",
					 ctl->socket_path, strerror(save_errno));
			pgstromCpuModeBegin(pts, reason);
			return;
		}
		elog(ERROR, "failed on connect('%s'): %m", ctl->socket_path);
	}
	__xpuClientOpenSession(pts, session, sockfd);
}

/*
 * gpuClientConnectControl - a non-blocking connection to the service for requests that need no
 * session (GPU-Cache maintenance), or -1 (no error is reported)
 */
int
gpuClientConnectControl(uint32_t *p_generation)
{
	const gpuserv_control *ctl = pgstrom_gpuserv_control();
	struct sockaddr_un addr;
	int			sockfd;
	int			flags;

	if (!ctl || __atomic_load_n(&ctl->state, __ATOMIC_SEQ_CST) != GPUSERV_STATE_READY)
		return -1;
	*p_generation = __atomic_load_n(&ctl->generation, __ATOMIC_SEQ_CST);
	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		return -1;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", ctl->socket_path);
	if (connect(sockfd, (struct sockaddr *)&addr, sizeof(addr)) != 0)
	{
		close(sockfd);
		return -1;
	}
	(void) fcntl(sockfd, F_SETFD, FD_CLOEXEC);
	flags = fcntl(sockfd, F_GETFL, 0);
	if (flags >= 0)
		(void) fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
	return sockfd;
}

void
pgstrom_init_gpu_client(void)
{
	DefineCustomIntVariable("pg_strom.max_async_tasks",
							"Limit of concurrent GPU task execution",
							NULL,
							&pgstrom_max_async_tasks_guc,
							16,
							1,
							256,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.gpu_error_fallback",
							 "A node continues on the CPU when the GPU service refuses or fails its program",
							 "On: a program the GPU service refuses or cannot run, a lost device and a service that is "
							 "not ready or goes away do not fail the query; the node reads its relation itself and "
							 "returns the rows the CPU plan would.  Off: the error is raised.",
							 &pgstrom_gpu_error_fallback,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.debug_no_program_limits",
							 "Test only: the planner does not keep programs that may exceed the GPU service's limits on the CPU",
							 "The service then refuses such a program at run time, which exercises the CPU fallback.",
							 &pgstrom_debug_no_limits,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.scan_chunk_blocks",
							"Test only: heap blocks per GPU task (0: as many as fit into 64MB)",
							NULL,
							&pgstrom_scan_chunk_blocks,
							0,
							0,
							1000000,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);
	DefineCustomStringVariable("pg_strom.gpuserv_fault",
							   "Fault injection for tests of the CPU fallback (see src/gpuserv/wire.c)",
							   NULL,
							   &pgstrom_gpuserv_fault,
							   "",
							   PGC_SUSET,
							   GUC_NOT_IN_SAMPLE | GUC_SUPERUSER_ONLY,
							   NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.debug_level",
							"Verbosity of PG-Strom's planner and executor debug messages",
							NULL,
							&pgstrom_debug_level,
							0,
							0,
							2,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
}
