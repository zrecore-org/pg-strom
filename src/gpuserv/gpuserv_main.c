/*
 * gpuserv_main.c
 *
 * pgstrom-gpuserv: the GPU service program.
 *
 *   pgstrom-gpuserv --probe [--allow-cpu] [--verbose]
 *       measure what every Vulkan device really supports and print the report
 *       (text format of gpuserv_proto.h) on stdout. Run by the postmaster at start-up.
 *
 *   pgstrom-gpuserv --serve --control NAME --device N --lifeline FD --socket PATH
 *       serve one device. Started and restarted by the launcher background worker:
 *       attaches the control segment NAME, opens the device, listens on PATH, and
 *       exits when the lifeline pipe (the launcher's) reaches EOF or on SIGTERM.
 *
 *   pgstrom-gpuserv --prewarm NBIND PCSIZE FILE.spv... [--device N]
 *       create the compute pipeline of each SPIR-V file (NBIND storage buffers, PCSIZE bytes of push constants)
 *       and exit: the driver's compiler runs once and the pipeline cache file keeps the result
 *       (test/gpuserv/prewarm_kernels.sh chooses the kernels and the arguments).
 *
 * Not a PostgreSQL process: no PostgreSQL headers, no fork() before exec().
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include "gpuserv_proto.h"
#include "gpu_vk.h"
#include "gpuserv_probe.h"
#include "gpu_vk.h"
#include "vm_selftest.h"
extern int scan_prog_df64;		/* scan_prog.h: the df64 prefilter of float8 comparisons (A10 part 6) */
#include "wire.h"
#include "kprewarm.h"

static volatile sig_atomic_t	got_signal = 0;
static int		prewarm_mode = GPUSERV_PREWARM_LITE;	/* --prewarm-set (pg_strom.gpu_prewarm) */

static void
log_msg(const char *fmt, ...)
{
	va_list		ap;

	fprintf(stderr, "pgstrom-gpuserv[%d]: ", (int) getpid());
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	fflush(stderr);
}

static void
handle_signal(int signo)
{
	got_signal = signo;
}

/* ----------------------------------------------------------------
 * --probe
 * ----------------------------------------------------------------
 */
static int
do_probe(int allow_cpu, int verbose)
{
	gpuserv_devcaps *devs = calloc(GPUSERV_MAX_DEVICES, sizeof(gpuserv_devcaps));
	int		i, n;

	if (!devs)
		return 1;
	n = gpuserv_probe_devices(devs, GPUSERV_MAX_DEVICES, allow_cpu, verbose ? stderr : NULL);
	if (n < 0)
	{
		log_msg("device probe failed");
		return 1;
	}
	gpuserv_probe_write_header(stdout, n);
	for (i = 0; i < n; i++)
		gpuserv_devcaps_write(stdout, i, &devs[i]);
	gpuserv_probe_write_footer(stdout);
	free(devs);
	return 0;
}

/* ----------------------------------------------------------------
 * --serve
 * ----------------------------------------------------------------
 */
typedef struct
{
	gpuserv_control *ctl;
	int			listen_fd;
	int			lifeline_fd;
	const char *socket_path;
	uint32_t	device;
	gpu_vk_ctx	vk;
	wire_server	wire;
} service_state;

static gpuserv_control *
attach_control(const char *name)
{
	size_t	len = (sizeof(gpuserv_control) + 16383) & ~(size_t) 16383;
	int		fd = shm_open(name, O_RDWR, 0600);
	void   *addr;

	if (fd < 0)
	{
		log_msg("cannot open the control segment '%s': %s", name, strerror(errno));
		return NULL;
	}
	addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (addr == MAP_FAILED)
	{
		log_msg("cannot map the control segment: %s", strerror(errno));
		return NULL;
	}
	if (((gpuserv_control *) addr)->magic != GPUSERV_CONTROL_MAGIC ||
		((gpuserv_control *) addr)->version != GPUSERV_PROTO_VERSION)
	{
		log_msg("the control segment has an unexpected magic or version");
		return NULL;
	}
	return addr;
}

static int
open_listen_socket(const char *path)
{
	struct sockaddr_un addr;
	int		fd = socket(AF_UNIX, SOCK_STREAM, 0);

	if (fd < 0)
		return -1;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	if (strlen(path) >= sizeof(addr.sun_path))
	{
		close(fd);
		errno = ENAMETOOLONG;
		return -1;
	}
	strcpy(addr.sun_path, path);
	unlink(path);
	if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0 ||
		chmod(path, 0600) != 0 ||
		listen(fd, 32) != 0)
	{
		int		save_errno = errno;

		close(fd);
		errno = save_errno;
		return -1;
	}
	return fd;
}

static int
do_serve(const char *control_name, uint32_t device, int lifeline_fd,
		 const char *socket_path)
{
	service_state ss;
	struct sigaction sa;

	memset(&ss, 0, sizeof(ss));
	ss.lifeline_fd = lifeline_fd;
	ss.socket_path = socket_path;
	ss.device = device;
	ss.ctl = attach_control(control_name);
	if (!ss.ctl)
		return 2;
	if (device >= ss.ctl->ndevices)
	{
		log_msg("device %u does not exist (%u devices)", device, ss.ctl->ndevices);
		return 2;
	}

	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = handle_signal;
	sigaction(SIGTERM, &sa, NULL);
	sigaction(SIGINT, &sa, NULL);
	signal(SIGPIPE, SIG_IGN);

	/*
	 * Open the device and prove the VM kernel works on it before saying "ready": a device
	 * that fails here is reported as failed rather than accepting work it would get wrong.
	 */
	memset(&ss.vk, 0, sizeof(ss.vk));
	if (gpu_vk_open(&ss.vk, (int) device) != 0)
	{
		log_msg("cannot open Vulkan device %u: %s", device, ss.vk.err);
		snprintf(ss.ctl->last_error, sizeof(ss.ctl->last_error),
				 "cannot open Vulkan device %u: %s", device, ss.vk.err);
		return 3;
	}
	{
		char	msg[256];

		gpu_vk_describe(&ss.vk, msg, sizeof(msg));
		log_msg("opened '%s': %s", ss.vk.info.name, msg);
		if (gpuserv_vm_selftest(&ss.vk, msg, sizeof(msg)) != 0)
		{
			log_msg("VM self-test FAILED: %s", msg);
			snprintf(ss.ctl->last_error, sizeof(ss.ctl->last_error), "VM self-test failed: %s", msg);
			gpu_vk_close(&ss.vk);
			return 3;
		}
		log_msg("VM self-test %s", msg);
		/*
		 * A10 part 6: the df64 prefilter of float8 comparisons needs exact error-free transformations. It is validated on MoltenVK only
		 * (gpu_vk.c switches its fast math off); on another driver it needs PGSTROM_GPUSERV_DF64=on, and in every case the device must
		 * pass gpuserv_df64_selfcheck. PGSTROM_GPUSERV_DF64=off keeps the exact path only.
		 */
		{
			const char *e = getenv("PGSTROM_GPUSERV_DF64");
			int			mvk = strstr(ss.vk.info.driver_name, "MoltenVK") != NULL || strstr(ss.vk.info.name, "MoltenVK") != NULL;

			if (e && !strcmp(e, "off"))
				log_msg("df64 prefilter off (PGSTROM_GPUSERV_DF64=off)");
			else if (!mvk && !(e && !strcmp(e, "on")))
				log_msg("df64 prefilter off: validated on MoltenVK only (PGSTROM_GPUSERV_DF64=on to try it after the self-check)");
			else if (gpuserv_df64_selfcheck(&ss.vk, msg, sizeof(msg)) != 0)
				log_msg("df64 prefilter off: the device does not keep the error-free transformations exact (%s)", msg);
			else
			{
				scan_prog_df64 = 1;
				log_msg("df64 prefilter on");
			}
		}
	}

	ss.listen_fd = open_listen_socket(socket_path);
	if (ss.listen_fd < 0)
	{
		log_msg("cannot listen on '%s': %s", socket_path, strerror(errno));
		return 2;
	}
	/* A9 part 3: the readiness gate and the background compile of the kernels (a no-op without a pipeline cache); before the
	 * first task can arrive */
	kpw_start(&ss.vk, &ss.ctl->prewarm, prewarm_mode);
	__atomic_store_n(&ss.ctl->state, GPUSERV_STATE_READY, __ATOMIC_SEQ_CST);
	log_msg("ready: device %u (%s), tier %s, socket %s",
			device, ss.ctl->devcaps[device].name,
			gpuserv_tier_name(ss.ctl->devcaps[device].tier), socket_path);

	wire_server_init(&ss.wire, ss.ctl, &ss.vk, device);
	int			lost = 0;
	while (!got_signal)
	{
		struct pollfd	pfds[2 + WIRE_MAX_CLIENTS];
		int				nfixed = 0, nclients, rc;

		pfds[nfixed].fd = ss.listen_fd;
		pfds[nfixed].events = POLLIN;
		pfds[nfixed++].revents = 0;
		if (ss.lifeline_fd >= 0)		/* EOF = the launcher is gone */
		{
			pfds[nfixed].fd = ss.lifeline_fd;
			pfds[nfixed].events = POLLIN;
			pfds[nfixed++].revents = 0;
		}
		nclients = wire_server_pollfds(&ss.wire, pfds + nfixed, WIRE_MAX_CLIENTS);
		rc = poll(pfds, nfixed + nclients, 1000);
		if (rc < 0)
		{
			if (errno == EINTR)
				continue;
			log_msg("poll failed: %s", strerror(errno));
			break;
		}
		if (rc == 0)
			continue;
		if (ss.lifeline_fd >= 0 && pfds[1].revents != 0)
		{
			char	c;

			if (read(ss.lifeline_fd, &c, 1) <= 0)
			{
				log_msg("the launcher went away, exiting");
				break;
			}
		}
		wire_server_service(&ss.wire, pfds + nfixed, nclients);
		if (ss.vk.lost)
		{
			/* every later task would fail with "device is lost": exit, the launcher starts a fresh service */
			log_msg("the GPU device was lost (%s): exiting so that the launcher restarts the service", ss.vk.err);
			lost = 1;
			break;
		}
		if (pfds[0].revents & POLLIN)
			wire_server_accept(&ss.wire, ss.listen_fd);
	}
	if (got_signal)
		log_msg("terminating on signal %d", (int) got_signal);
	wire_server_fini(&ss.wire);
	close(ss.listen_fd);
	unlink(socket_path);
	/* a thread inside a driver compile cannot be interrupted: leave it to the exit of the process, do not destroy the device under it */
	if (kpw_stop(2000))
		gpu_vk_close(&ss.vk);
	return lost ? 3 : 0;
}

/* --prewarm: create the pipelines once so that the pipeline cache file holds them */
static int
do_prewarm(long device, uint32_t nbind, uint32_t pcsize, char **files, int nfiles)
{
	gpu_vk_ctx *ctx = calloc(1, sizeof(*ctx));
	int			rc = 0;

	if (!ctx || gpu_vk_open(ctx, (int) device) != 0)
	{
		log_msg("prewarm: cannot open device %ld: %s", device, ctx ? ctx->err : "out of memory");
		return 1;
	}
	for (int i = 0; i < nfiles; i++)
	{
		FILE	   *fp = fopen(files[i], "rb");
		long		n = 0;
		void	   *buf = NULL;
		gpu_vk_pipeline pl;
		struct timespec t0, t1;

		if (!fp || fseek(fp, 0, SEEK_END) != 0 || (n = ftell(fp)) <= 0 || fseek(fp, 0, SEEK_SET) != 0 ||
			posix_memalign(&buf, 16, (size_t) n + 16) != 0 || fread(buf, 1, (size_t) n, fp) != (size_t) n)
		{
			log_msg("prewarm: cannot read %s", files[i]);
			rc = 1;
			if (fp)
				fclose(fp);
			free(buf);
			continue;
		}
		fclose(fp);
		clock_gettime(CLOCK_MONOTONIC, &t0);
		if (gpu_vk_pipeline_create(ctx, &pl, buf, (size_t) n, nbind, pcsize) != 0)
		{
			log_msg("prewarm: %s: %s", files[i], ctx->err);
			rc = 1;
		}
		else
		{
			clock_gettime(CLOCK_MONOTONIC, &t1);
			/*
			 * An empty dispatch (0 workgroups): measured 2026-11-07, a pipeline that was created but never dispatched in
			 * the process is not found by the next process (the compile of a 32-register aggregation kernel, 270 s, is
			 * repeated), while one that was dispatched once is (18 ms); the driver finishes its lazy compile and
			 * records the result at the first dispatch.
			 */
			{
				gpu_vk_buffer bufs[8];
				gpu_vk_buffer *bp[8];
				char		pc[64];
				uint32_t	nb = nbind < 8 ? nbind : 8;
				uint32_t	k;

				memset(bufs, 0, sizeof(bufs));
				memset(pc, 0, sizeof(pc));
				for (k = 0; k < nb; k++)
				{
					bp[k] = &bufs[k];
					if (gpu_vk_buffer_create(ctx, &bufs[k], 4096, 0) != 0)
						break;
				}
				if (k == nb)
					(void) gpu_vk_dispatch(ctx, &pl, bp, pc, 0, 1, 1, NULL);
				while (k-- > 0)
					gpu_vk_buffer_destroy(ctx, &bufs[k]);
			}
			clock_gettime(CLOCK_MONOTONIC, &t1);
			printf("prewarm %s: %.2f s\n", files[i],
				   (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
			fflush(stdout);
			gpu_vk_pipeline_destroy(ctx, &pl);
		}
		free(buf);
	}
	gpu_vk_close(ctx);
	free(ctx);
	return rc;
}

/* --prewarm-all / --prewarm-list: compile (or list) the kernel set in this process; the installer's hook */
static int
do_prewarm_all(long device, int mode, int list_only)
{
	gpu_vk_ctx *ctx = calloc(1, sizeof(*ctx));
	int			rc;

	if (!ctx || gpu_vk_open(ctx, (int) device) != 0)
	{
		log_msg("prewarm: cannot open device %ld: %s", device, ctx ? ctx->err : "out of memory");
		return 1;
	}
	if (!gpu_vk_pcache_path(ctx)[0])
		log_msg("prewarm: the pipeline cache is off (PGSTROM_GPUSERV_PIPELINE_CACHE, PGSTROM_GPUSERV_CACHE_DIR=none): nothing would be kept");
	rc = kpw_cli(ctx, mode, list_only);
	gpu_vk_close(ctx);
	free(ctx);
	return rc;
}

/* ----------------------------------------------------------------
 * main
 * ----------------------------------------------------------------
 */
int
main(int argc, char **argv)
{
	int			i, probe = 0, serve = 0, allow_cpu = 0, verbose = 0;
	const char *control = NULL, *socket_path = NULL;
	int			lifeline = -1;
	long		device = 0;
	int			prewarm_all = 0, prewarm_list = 0;

	for (i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--prewarm") && i + 2 < argc)
		{
			long		nb = strtol(argv[i + 1], NULL, 10);
			long		pc = strtol(argv[i + 2], NULL, 10);
			int			k;

			for (k = i + 3; k + 1 < argc; k++)
				if (!strcmp(argv[k], "--device"))
				{
					device = strtol(argv[k + 1], NULL, 10);
					memmove(&argv[k], &argv[k + 2], (argc - k - 2) * sizeof(char *));
					argc -= 2;
					break;
				}
			return do_prewarm(device, (uint32_t) nb, (uint32_t) pc, &argv[i + 3], argc - i - 3);
		}
		if (!strcmp(argv[i], "--probe"))
		{
			probe = 1;
			/*
			 * The postmaster waits for the probe for a bounded time (60 s) and the probe only runs a few tiny kernels,
			 * so it must not load the persisted pipeline caches (a cache of 1 GB made one start take 45 s on a busy
			 * machine and the postmaster killed the probe: "GPU probe did not finish within 60000 ms").
			 */
			setenv("PGSTROM_GPUSERV_PIPELINE_CACHE", "off", 1);
		}
		else if (!strcmp(argv[i], "--serve"))
			serve = 1;
		else if (!strcmp(argv[i], "--prewarm-all"))
			prewarm_all = 1;
		else if (!strcmp(argv[i], "--prewarm-list"))
			prewarm_list = 1;
		else if (!strcmp(argv[i], "--prewarm-set") && i + 1 < argc)
		{
			prewarm_mode = kpw_parse_mode(argv[++i]);		/* pg_strom.gpu_prewarm; --prewarm-all takes it too */
			if (prewarm_mode < 0)
			{
				fprintf(stderr, "--prewarm-set: off, lite, geometry or all\n");
				return 2;
			}
		}
		else if (!strcmp(argv[i], "--allow-cpu"))
			allow_cpu = 1;
		else if (!strcmp(argv[i], "--verbose"))
			verbose = 1;
		else if (!strcmp(argv[i], "--control") && i + 1 < argc)
			control = argv[++i];
		else if (!strcmp(argv[i], "--socket") && i + 1 < argc)
			socket_path = argv[++i];
		else if (!strcmp(argv[i], "--device") && i + 1 < argc)
			device = strtol(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--lifeline") && i + 1 < argc)
			lifeline = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--specialize") && i + 1 < argc)
			setenv("PGSTROM_GPUSERV_SPECIALIZE", argv[++i], 1);		/* pg_strom.gpu_specialize */
		else if (!strcmp(argv[i], "--group-memory") && i + 1 < argc)
			setenv("PGSTROM_GPUSERV_GROUP_MEM_MB", argv[++i], 1);		/* pg_strom.gpuserv_group_memory */
		else if (!strcmp(argv[i], "--kernel-cache-dir") && i + 1 < argc)
			setenv("PGSTROM_GPUSERV_CACHE_DIR", argv[++i], 1);		/* pg_strom.gpu_kernel_cache_dir */
		else
		{
			fprintf(stderr,
					"usage: %s --probe [--allow-cpu] [--verbose]\n"
					"       %s --serve --control NAME --device N --lifeline FD --socket PATH [--allow-cpu]\n"
					"       %s --prewarm-all [--prewarm-set lite|geometry|all] [--device N]\n"
					"       %s --prewarm-list [--prewarm-set lite|geometry|all] [--device N]\n",
					argv[0], argv[0], argv[0], argv[0]);
			return 2;
		}
	}
	if (prewarm_all || prewarm_list)
	{
		/* the same pipeline cache directory and fast-math setting as the service: run as the user the server runs as */
		setenv("PGSTROM_GPUSERV_PREWARM", "off", 1);
		return do_prewarm_all(device, prewarm_mode == GPUSERV_PREWARM_OFF ? GPUSERV_PREWARM_LITE : prewarm_mode, prewarm_list);
	}
	if (probe == serve)
	{
		fprintf(stderr, "exactly one of --probe or --serve is required\n");
		return 2;
	}
	if (probe)
		return do_probe(allow_cpu, verbose);
	if (!control || !socket_path || device < 0)
	{
		fprintf(stderr, "--serve needs --control, --device and --socket\n");
		return 2;
	}
	return do_serve(control, (uint32_t) device, lifeline, socket_path);
}
