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
#include "gpuserv_proto.h"
#include "gpuserv_probe.h"
#include "gpu_vk.h"
#include "vm_selftest.h"
#include "wire.h"

static volatile sig_atomic_t	got_signal = 0;

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
	}

	ss.listen_fd = open_listen_socket(socket_path);
	if (ss.listen_fd < 0)
	{
		log_msg("cannot listen on '%s': %s", socket_path, strerror(errno));
		return 2;
	}
	__atomic_store_n(&ss.ctl->state, GPUSERV_STATE_READY, __ATOMIC_SEQ_CST);
	log_msg("ready: device %u (%s), tier %s, socket %s",
			device, ss.ctl->devcaps[device].name,
			gpuserv_tier_name(ss.ctl->devcaps[device].tier), socket_path);

	wire_server_init(&ss.wire, ss.ctl, &ss.vk, device);
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
		if (pfds[0].revents & POLLIN)
			wire_server_accept(&ss.wire, ss.listen_fd);
	}
	if (got_signal)
		log_msg("terminating on signal %d", (int) got_signal);
	wire_server_fini(&ss.wire);
	close(ss.listen_fd);
	unlink(socket_path);
	gpu_vk_close(&ss.vk);
	return 0;
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

	for (i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--probe"))
			probe = 1;
		else if (!strcmp(argv[i], "--serve"))
			serve = 1;
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
		else
		{
			fprintf(stderr,
					"usage: %s --probe [--allow-cpu] [--verbose]\n"
					"       %s --serve --control NAME --device N --lifeline FD --socket PATH [--allow-cpu]\n",
					argv[0], argv[0]);
			return 2;
		}
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
