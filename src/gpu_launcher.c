/*
 * gpu_launcher.c
 *
 * Starts, watches and restarts the GPU service, and publishes what it learned about
 * the devices.
 *
 * The GPU service is a separately executed program (pgstrom-gpuserv), not a PostgreSQL
 * background worker: on macOS a process forked without exec cannot use Metal's shader
 * compiler, and PostgreSQL 17 refuses a multithreaded postmaster. So:
 *
 *   postmaster     creates the control segment (named POSIX shm), runs
 *                  'pgstrom-gpuserv --probe' to learn each device's real capabilities,
 *                  and registers the launcher
 *   launcher       a background worker that never touches Vulkan: fork+exec the service,
 *                  reap it, restart it with a back-off, tell it to stop at shutdown
 *   service        attaches the control segment by name, serves the socket
 *   backends       read the control segment they inherited through fork()
 *
 * A crash or hang of the GPU driver therefore kills only the service; the launcher
 * brings it back and PostgreSQL keeps running.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "funcapi.h"
#include "libpq/libpq.h"
#include "gpuserv/gpuserv_proto.h"

/* GUC variables */
static char	   *gpuserv_path = NULL;
static bool		gpu_allow_cpu_device = false;
static int		gpuserv_probe_timeout = 60000;		/* ms */

static char	   *gpu_visible_devices = NULL;		/* GUC */
static gpuserv_control *gpuserv_ctl = NULL;
static char		gpuserv_ctl_name[64];
static volatile sig_atomic_t launcher_got_sigterm = false;

#define GPUSERV_PROBE_BUFSZ		(256 * 1024)

/* ----------------------------------------------------------------
 *
 * Control segment
 *
 * ----------------------------------------------------------------
 */
static void
gpuserv_control_cleanup(int code, Datum arg)
{
	/* runs in the postmaster only: children reset the exit callbacks after fork() */
	if (gpuserv_ctl_name[0])
		shm_unlink(gpuserv_ctl_name);
}

static bool
gpuserv_control_create(void)
{
	size_t	len = TYPEALIGN(sysconf(_SC_PAGESIZE), sizeof(gpuserv_control));
	int		fdesc;
	void   *addr;

	snprintf(gpuserv_ctl_name, sizeof(gpuserv_ctl_name), "/pgsctl.%d", PostPortNumber);
	(void) shm_unlink(gpuserv_ctl_name);	/* stale object of a crashed postmaster */
	fdesc = shm_open(gpuserv_ctl_name, O_RDWR | O_CREAT | O_EXCL, 0600);
	if (fdesc < 0)
	{
		elog(LOG, "PG-Strom: failed on shm_open('%s'): %m", gpuserv_ctl_name);
		gpuserv_ctl_name[0] = '\0';
		return false;
	}
	if (pgstromPreallocate(fdesc, len) != 0)
	{
		elog(LOG, "PG-Strom: failed to size the control segment: %m");
		goto error;
	}
	addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fdesc, 0);
	if (addr == MAP_FAILED)
	{
		elog(LOG, "PG-Strom: failed on mmap of the control segment: %m");
		goto error;
	}
	close(fdesc);
	memset(addr, 0, len);
	gpuserv_ctl = (gpuserv_control *) addr;
	gpuserv_ctl->magic = GPUSERV_CONTROL_MAGIC;
	gpuserv_ctl->version = GPUSERV_PROTO_VERSION;
	gpuserv_ctl->selected_device = UINT32_MAX;
	gpuserv_ctl->state = GPUSERV_STATE_STOPPED;
	on_proc_exit(gpuserv_control_cleanup, 0);
	return true;

error:
	close(fdesc);
	shm_unlink(gpuserv_ctl_name);
	gpuserv_ctl_name[0] = '\0';
	return false;
}

/* ----------------------------------------------------------------
 *
 * Running the service binary
 *
 * ----------------------------------------------------------------
 */

/* in the forked child: keep fds 0-2 (stdin to /dev/null) and 'keep_fd' (as 3), close the rest */
static void
exec_service_child(const char *path, char *const argv[], int keep_fd)
{
	int		fd, maxfd = (int) sysconf(_SC_OPEN_MAX);

	if (maxfd < 0 || maxfd > 4096)
		maxfd = 4096;
	fd = open("/dev/null", O_RDONLY);
	if (fd >= 0 && fd != 0)
	{
		dup2(fd, 0);
		close(fd);
	}
	if (keep_fd >= 0 && keep_fd != 3)
	{
		dup2(keep_fd, 3);
		keep_fd = 3;
	}
	for (fd = 3 + (keep_fd >= 0 ? 1 : 0); fd < maxfd; fd++)
		close(fd);
	execv(path, argv);
	_exit(127);
}

/*
 * Run 'pgstrom-gpuserv --probe' and collect its output. The probe runs in a child so a
 * misbehaving driver cannot take the postmaster down, and it is killed after the timeout.
 * Returns the text (malloc'd), or NULL on failure.
 */
static char *
gpuserv_run_probe(void)
{
	char	   *buf = malloc(GPUSERV_PROBE_BUFSZ);
	size_t		used = 0;
	int			pfd[2];
	pid_t		pid;
	int			status;
	TimestampTz	deadline;
	char	   *argv[4];
	int			argc = 0;
	bool		timed_out = false;

	if (!buf)
		return NULL;
	argv[argc++] = gpuserv_path;
	argv[argc++] = "--probe";
	if (gpu_allow_cpu_device)
		argv[argc++] = "--allow-cpu";
	argv[argc] = NULL;

	if (pipe(pfd) != 0)
	{
		elog(LOG, "PG-Strom: failed on pipe(2): %m");
		free(buf);
		return NULL;
	}
	pid = fork();
	if (pid < 0)
	{
		elog(LOG, "PG-Strom: failed on fork(2): %m");
		close(pfd[0]);
		close(pfd[1]);
		free(buf);
		return NULL;
	}
	if (pid == 0)
	{
		close(pfd[0]);
		dup2(pfd[1], 1);
		if (pfd[1] != 1)
			close(pfd[1]);
		exec_service_child(gpuserv_path, argv, -1);
	}
	close(pfd[1]);
	deadline = GetCurrentTimestamp() + (TimestampTz) gpuserv_probe_timeout * 1000;
	for (;;)
	{
		struct pollfd	pfds = {pfd[0], POLLIN, 0};
		long	remain = (deadline - GetCurrentTimestamp()) / 1000;
		ssize_t	nbytes;

		if (remain <= 0)
		{
			timed_out = true;
			break;
		}
		if (poll(&pfds, 1, (int) Min(remain, 1000)) < 0)
		{
			if (errno == EINTR)
				continue;
			break;
		}
		if (pfds.revents == 0)
			continue;
		if (used + 1 >= GPUSERV_PROBE_BUFSZ)
			break;
		nbytes = read(pfd[0], buf + used, GPUSERV_PROBE_BUFSZ - used - 1);
		if (nbytes > 0)
			used += nbytes;
		else if (nbytes == 0)
			break;			/* EOF: the probe finished */
		else if (errno != EINTR)
			break;
	}
	close(pfd[0]);
	if (timed_out)
	{
		elog(LOG, "PG-Strom: GPU probe did not finish within %d ms, killed it",
			 gpuserv_probe_timeout);
		kill(pid, SIGKILL);
	}
	while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
		;
	buf[used] = '\0';
	if (!timed_out && !(WIFEXITED(status) && WEXITSTATUS(status) == 0))
	{
		elog(LOG, "PG-Strom: GPU probe '%s --probe' failed (status 0x%x)",
			 gpuserv_path, status);
		free(buf);
		return NULL;
	}
	if (timed_out)
	{
		free(buf);
		return NULL;
	}
	return buf;
}

/* ----------------------------------------------------------------
 *
 * Launcher background worker
 *
 * ----------------------------------------------------------------
 */
static void
launcher_sigterm(SIGNAL_ARGS)
{
	int		save_errno = errno;

	launcher_got_sigterm = true;
	SetLatch(MyLatch);
	errno = save_errno;
}

static pid_t
launcher_spawn_service(int *p_lifeline_write)
{
	char		devbuf[32];
	char		fdbuf[16];
	char	   *argv[16];
	int			argc = 0;
	int			lifeline[2];
	pid_t		pid;

	if (pipe(lifeline) != 0)
		return -1;
	fcntl(lifeline[1], F_SETFD, FD_CLOEXEC);	/* only the launcher holds the write end */
	snprintf(devbuf, sizeof(devbuf), "%u", gpuserv_ctl->selected_device);
	snprintf(fdbuf, sizeof(fdbuf), "3");		/* the child puts the read end on fd 3 */
	argv[argc++] = gpuserv_path;
	argv[argc++] = "--serve";
	argv[argc++] = "--control";
	argv[argc++] = gpuserv_ctl_name;
	argv[argc++] = "--device";
	argv[argc++] = devbuf;
	argv[argc++] = "--lifeline";
	argv[argc++] = fdbuf;
	argv[argc++] = "--socket";
	argv[argc++] = gpuserv_ctl->socket_path;
	if (gpu_allow_cpu_device)
		argv[argc++] = "--allow-cpu";
	argv[argc] = NULL;

	pid = fork();
	if (pid < 0)
	{
		close(lifeline[0]);
		close(lifeline[1]);
		return -1;
	}
	if (pid == 0)
		exec_service_child(gpuserv_path, argv, lifeline[0]);
	close(lifeline[0]);
	*p_lifeline_write = lifeline[1];
	return pid;
}

static void
launcher_set_error(const char *fmt, ...) pg_attribute_printf(1, 2);

static void
launcher_set_error(const char *fmt, ...)
{
	va_list		ap;

	va_start(ap, fmt);
	vsnprintf(gpuserv_ctl->last_error, sizeof(gpuserv_ctl->last_error), fmt, ap);
	va_end(ap);
}

/* ask the service to stop, escalate to SIGKILL after 3 s */
static void
launcher_stop_service(pid_t pid)
{
	int		status, i;

	kill(pid, SIGTERM);
	for (i = 0; i < 60; i++)
	{
		if (waitpid(pid, &status, WNOHANG) == pid)
			return;
		pg_usleep(50 * 1000L);
	}
	elog(LOG, "PG-Strom: GPU service (pid %d) did not stop, sending SIGKILL", (int) pid);
	kill(pid, SIGKILL);
	while (waitpid(pid, &status, 0) < 0 && errno == EINTR)
		;
}

PGDLLEXPORT void pgstrom_gpu_launcher_main(Datum arg);

void
pgstrom_gpu_launcher_main(Datum arg)
{
	long		backoff_ms = 1000;

	pqsignal(SIGTERM, launcher_sigterm);
	BackgroundWorkerUnblockSignals();

	while (!launcher_got_sigterm)
	{
		TimestampTz	started;
		pid_t		pid;
		int			lifeline_wr = -1;
		int			status = 0;
		bool		exited = false;

		pid = launcher_spawn_service(&lifeline_wr);
		if (pid < 0)
		{
			launcher_set_error("could not start '%s': %s", gpuserv_path, strerror(errno));
			__atomic_store_n(&gpuserv_ctl->state, GPUSERV_STATE_FAILED, __ATOMIC_SEQ_CST);
			elog(LOG, "PG-Strom: %s", gpuserv_ctl->last_error);
		}
		else
		{
			gpuserv_ctl->last_error[0] = '\0';
			__atomic_store_n(&gpuserv_ctl->service_pid, (int32_t) pid, __ATOMIC_SEQ_CST);
			__atomic_add_fetch(&gpuserv_ctl->generation, 1, __ATOMIC_SEQ_CST);
			__atomic_store_n(&gpuserv_ctl->state, GPUSERV_STATE_STARTING, __ATOMIC_SEQ_CST);
			elog(LOG, "PG-Strom: GPU service started (pid %d, generation %u)",
				 (int) pid, gpuserv_ctl->generation);
			started = GetCurrentTimestamp();

			while (!launcher_got_sigterm)
			{
				if (waitpid(pid, &status, WNOHANG) == pid)
				{
					exited = true;
					break;
				}
				(void) WaitLatch(MyLatch,
								 WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
								 500L, PG_WAIT_EXTENSION);
				ResetLatch(MyLatch);
			}
			if (!exited)
			{
				/* shutting down: closing the lifeline pipe also tells the service */
				launcher_stop_service(pid);
				close(lifeline_wr);
				__atomic_store_n(&gpuserv_ctl->service_pid, 0, __ATOMIC_SEQ_CST);
				__atomic_store_n(&gpuserv_ctl->state, GPUSERV_STATE_STOPPED, __ATOMIC_SEQ_CST);
				break;
			}
			close(lifeline_wr);
			__atomic_store_n(&gpuserv_ctl->service_pid, 0, __ATOMIC_SEQ_CST);
			if (WIFSIGNALED(status))
				launcher_set_error("service (pid %d) killed by signal %d",
								   (int) pid, WTERMSIG(status));
			else
				launcher_set_error("service (pid %d) exited with status %d",
								   (int) pid, WEXITSTATUS(status));
			__atomic_store_n(&gpuserv_ctl->state, GPUSERV_STATE_FAILED, __ATOMIC_SEQ_CST);
			__atomic_add_fetch(&gpuserv_ctl->restart_count, 1, __ATOMIC_SEQ_CST);
			elog(LOG, "PG-Strom: GPU service stopped: %s", gpuserv_ctl->last_error);
			/* a service that ran for a minute was healthy: start the back-off over */
			if (GetCurrentTimestamp() - started > 60 * USECS_PER_SEC)
				backoff_ms = 1000;
		}
		/* back off before the next attempt, but wake up for SIGTERM */
		if (!launcher_got_sigterm)
		{
			(void) WaitLatch(MyLatch,
							 WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
							 backoff_ms, PG_WAIT_EXTENSION);
			ResetLatch(MyLatch);
			backoff_ms = Min(backoff_ms * 2, 30000L);
		}
	}
	proc_exit(0);
}

/* ----------------------------------------------------------------
 *
 * Initialization (postmaster)
 *
 * ----------------------------------------------------------------
 */
static void
gpuserv_socket_path(char *buf, size_t bufsz)
{
	char	   *dir = NULL, *comma;

	if (Unix_socket_directories && Unix_socket_directories[0])
	{
		dir = pstrdup(Unix_socket_directories);
		if ((comma = strchr(dir, ',')) != NULL)
			*comma = '\0';
	}
	snprintf(buf, bufsz, "%s/.pg_strom.%d.gpuserv.sock",
			 (dir && dir[0]) ? dir : DataDir, PostPortNumber);
	if (strlen(buf) >= 100)		/* sun_path is 104 bytes on macOS */
		snprintf(buf, bufsz, "/tmp/.pg_strom.%d.gpuserv.sock", PostPortNumber);
}

/* is device 'index' listed in pg_strom.gpu_visible_devices? (empty = all of them) */
static bool
gpu_device_is_visible(int index)
{
	const char *p = gpu_visible_devices;

	if (!p || !p[0])
		return true;
	while (*p)
	{
		char	   *end;
		long		v = strtol(p, &end, 10);

		if (end == p)
			break;
		if (v == index)
			return true;
		p = (*end == ',' ? end + 1 : end);
		while (*p == ' ')
			p++;
	}
	return false;
}

void
pgstrom_init_gpu_launcher(void)
{
	char		pkglib[MAXPGPATH];
	char		defpath[MAXPGPATH];
	char	   *text;
	int			ndev, i, selected = -1;
	BackgroundWorker worker;

	get_pkglib_path(my_exec_path, pkglib);
	snprintf(defpath, sizeof(defpath), "%s/pgstrom-gpuserv", pkglib);
	DefineCustomStringVariable("pg_strom.gpuserv_path",
							   "Path of the GPU service executable",
							   NULL,
							   &gpuserv_path,
							   defpath,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	DefineCustomStringVariable("pg_strom.gpu_visible_devices",
							   "Vulkan devices PG-Strom may use: device indexes separated by commas (empty: all)",
							   NULL,
							   &gpu_visible_devices,
							   NULL,
							   PGC_POSTMASTER,
							   GUC_NOT_IN_SAMPLE,
							   NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.gpu_allow_cpu_device",
							 "Allow Vulkan devices that run on the CPU (lavapipe) to be used",
							 "Meant for testing; such a device is never faster than the CPU",
							 &gpu_allow_cpu_device,
							 false,
							 PGC_POSTMASTER,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpuserv_probe_timeout",
							"Time limit of the GPU capability probe at startup",
							NULL,
							&gpuserv_probe_timeout,
							60000,
							1000,
							600000,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MS,
							NULL, NULL, NULL);

	if (access(gpuserv_path, X_OK) != 0)
	{
		elog(LOG, "PG-Strom: GPU service '%s' is not available (%m); running without GPU",
			 gpuserv_path);
		return;
	}
	if (!gpuserv_control_create())
		return;
	text = gpuserv_run_probe();
	if (!text)
		return;
	ndev = gpuserv_probe_parse(text, gpuserv_ctl->devcaps, GPUSERV_MAX_DEVICES);
	free(text);
	if (ndev < 0)
	{
		elog(LOG, "PG-Strom: the GPU probe returned a malformed report; running without GPU");
		return;
	}
	gpuserv_ctl->ndevices = ndev;
	for (i = 0; i < ndev; i++)
	{
		const gpuserv_devcaps *dc = &gpuserv_ctl->devcaps[i];

		elog(LOG, "PG-Strom: Vulkan device %d: %s (%s), tier %s%s%s",
			 i, dc->name, dc->driver_name, gpuserv_tier_name(dc->tier),
			 dc->tier_reason[0] ? " - " : "", dc->tier_reason);
		if (selected < 0 && dc->tier != GPUSERV_TIER_NONE && gpu_device_is_visible(i))
			selected = i;
	}
	if (selected < 0)
	{
		elog(LOG, "PG-Strom: no usable Vulkan device; running without GPU");
		return;
	}
	gpuserv_ctl->selected_device = selected;
	gpuserv_socket_path(gpuserv_ctl->socket_path, sizeof(gpuserv_ctl->socket_path));

	memset(&worker, 0, sizeof(worker));
	snprintf(worker.bgw_name, BGW_MAXLEN, "PG-Strom GPU service launcher");
	snprintf(worker.bgw_type, BGW_MAXLEN, "PG-Strom GPU service launcher");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 5;
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_strom");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "pgstrom_gpu_launcher_main");
	RegisterBackgroundWorker(&worker);
}

/*
 * The control segment as seen by the postmaster and (through fork) every backend.
 * NULL if there is no usable device.
 */
const gpuserv_control *
pgstrom_gpuserv_control(void)
{
	if (gpuserv_ctl && gpuserv_ctl->selected_device < gpuserv_ctl->ndevices)
		return gpuserv_ctl;
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * SQL-callable functions
 *
 * ----------------------------------------------------------------
 */
static char *
caps_to_text(uint64_t caps)
{
	StringInfoData	buf;
	int		i;

	initStringInfo(&buf);
	for (i = 0; i < GPUCAP_NUM; i++)
		if (caps & (1ULL << i))
			appendStringInfo(&buf, "%s%s", buf.len > 0 ? "," : "", gpuserv_cap_name(i));
	return buf.data;
}

static const char *
device_type_name(uint32_t t)
{
	switch (t)
	{
		case 1:		return "integrated";
		case 2:		return "discrete";
		case 3:		return "virtual";
		case 4:		return "cpu";
		default:	return "other";
	}
}

/* pgstrom.gpu_devcaps() */
PG_FUNCTION_INFO_V1(pgstrom_gpu_devcaps);
Datum
pgstrom_gpu_devcaps(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	uint32		i;

	InitMaterializedSRF(fcinfo, 0);
	if (!gpuserv_ctl)
		return (Datum) 0;
	for (i = 0; i < gpuserv_ctl->ndevices; i++)
	{
		const gpuserv_devcaps *dc = &gpuserv_ctl->devcaps[i];
		Datum	values[15];
		bool	nulls[15];

		memset(nulls, 0, sizeof(nulls));
		values[0]  = Int32GetDatum(i);
		values[1]  = CStringGetTextDatum(dc->name);
		values[2]  = CStringGetTextDatum(gpuserv_tier_name(dc->tier));
		values[3]  = CStringGetTextDatum(dc->tier_reason);
		values[4]  = CStringGetTextDatum(device_type_name(dc->device_type));
		values[5]  = CStringGetTextDatum(dc->driver_name);
		values[6]  = CStringGetTextDatum(caps_to_text(dc->caps));
		values[7]  = CStringGetTextDatum(caps_to_text(dc->reported_caps));
		values[8]  = Int32GetDatum(dc->subgroup_size_min);
		values[9]  = Int32GetDatum(dc->subgroup_size_max);
		values[10] = Int32GetDatum(dc->subgroup_size_observed);
		values[11] = Int32GetDatum(dc->max_shared_memory);
		values[12] = Int64GetDatum((int64) dc->max_storage_buffer_range);
		values[13] = Int64GetDatum((int64) dc->device_memory);
		values[14] = BoolGetDatum(i == gpuserv_ctl->selected_device);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	return (Datum) 0;
}

/* pgstrom.gpu_service_state() */
PG_FUNCTION_INFO_V1(pgstrom_gpu_service_state);
Datum
pgstrom_gpu_service_state(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[7];
	bool		nulls[7];
	const char *state;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	tupdesc = BlessTupleDesc(tupdesc);
	memset(nulls, 0, sizeof(nulls));
	if (!gpuserv_ctl || gpuserv_ctl->selected_device == UINT32_MAX)
	{
		values[0] = CStringGetTextDatum("unavailable");
		nulls[1] = nulls[2] = nulls[3] = nulls[4] = nulls[5] = nulls[6] = true;
	}
	else
	{
		switch (__atomic_load_n(&gpuserv_ctl->state, __ATOMIC_SEQ_CST))
		{
			case GPUSERV_STATE_STARTING:	state = "starting"; break;
			case GPUSERV_STATE_READY:		state = "ready"; break;
			case GPUSERV_STATE_FAILED:		state = "failed"; break;
			default:						state = "stopped"; break;
		}
		values[0] = CStringGetTextDatum(state);
		values[1] = Int32GetDatum(__atomic_load_n(&gpuserv_ctl->service_pid, __ATOMIC_SEQ_CST));
		values[2] = Int64GetDatum(__atomic_load_n(&gpuserv_ctl->generation, __ATOMIC_SEQ_CST));
		values[3] = Int32GetDatum(__atomic_load_n(&gpuserv_ctl->restart_count, __ATOMIC_SEQ_CST));
		values[4] = Int32GetDatum(gpuserv_ctl->selected_device);
		values[5] = CStringGetTextDatum(gpuserv_ctl->socket_path);
		if (gpuserv_ctl->last_error[0])
			values[6] = CStringGetTextDatum(gpuserv_ctl->last_error);
		else
			nulls[6] = true;
	}
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

/* pgstrom.gpu_service_ping(): connect to the service and read its handshake reply */
PG_FUNCTION_INFO_V1(pgstrom_gpu_service_ping);
Datum
pgstrom_gpu_service_ping(PG_FUNCTION_ARGS)
{
	struct sockaddr_un addr;
	struct timeval tmo = {2, 0};
	gpuserv_sock_req req;
	gpuserv_sock_pong pong;
	int			sockfd;
	ssize_t		nbytes;

	if (!gpuserv_ctl || gpuserv_ctl->selected_device == UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("no GPU service is configured")));
	if (__atomic_load_n(&gpuserv_ctl->state, __ATOMIC_SEQ_CST) != GPUSERV_STATE_READY)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("the GPU service is not ready"),
				 gpuserv_ctl->last_error[0] ? errdetail("%s", gpuserv_ctl->last_error) : 0));
	sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (sockfd < 0)
		elog(ERROR, "failed on socket(2): %m");
	setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tmo, sizeof(tmo));
	setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tmo, sizeof(tmo));
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", gpuserv_ctl->socket_path);
	if (connect(sockfd, (struct sockaddr *) &addr, sizeof(addr)) != 0)
	{
		int		save_errno = errno;

		close(sockfd);
		errno = save_errno;
		elog(ERROR, "failed on connect('%s'): %m", gpuserv_ctl->socket_path);
	}
	req.magic = GPUSERV_SOCK_MAGIC;
	req.cmd = GPUSERV_CMD_PING;
	if (write(sockfd, &req, sizeof(req)) != (ssize_t) sizeof(req))
	{
		close(sockfd);
		elog(ERROR, "failed to send the handshake: %m");
	}
	nbytes = read(sockfd, &pong, sizeof(pong));
	close(sockfd);
	if (nbytes != (ssize_t) sizeof(pong) || pong.magic != GPUSERV_SOCK_MAGIC || pong.status != 0)
		elog(ERROR, "unexpected reply from the GPU service (%zd bytes)", nbytes);
	PG_RETURN_TEXT_P(cstring_to_text(psprintf("ok pid=%d generation=%u device=%u tier=%s",
											  (int) pong.pid, pong.generation, pong.device,
											  gpuserv_tier_name(pong.tier))));
}
