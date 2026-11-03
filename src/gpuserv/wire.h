/*
 * wire.h
 *
 * Backend <-> GPU service protocol (the XpuCommand frames of xpu_common.h), served by
 * the exec'd pgstrom-gpuserv. The wire format is unchanged from the CUDA service, so
 * executor.c / gpu_scan.c / relscan.c / fallback.c on the backend side do not change.
 */
#ifndef GPUSERV_WIRE_H
#define GPUSERV_WIRE_H
#include <poll.h>
#include <stddef.h>
#include <stdint.h>
#include "gpu_vk.h"
#include "gpuserv_proto.h"

#define WIRE_MAX_CLIENTS		64

typedef struct wire_client wire_client;

typedef struct wire_server
{
	gpuserv_control *ctl;
	gpu_vk_ctx	   *vk;
	uint32_t		device;
	int				trace;				/* PGSTROM_GPUSERV_TRACE */
	wire_client	   *clients[WIRE_MAX_CLIENTS];
	int				nclients;
} wire_server;

void	wire_server_init(wire_server *ws, gpuserv_control *ctl, gpu_vk_ctx *vk, uint32_t device);
/* accept() a connection from the listen socket and track it. */
void	wire_server_accept(wire_server *ws, int listen_fd);
/* Fill pfds[] (after the caller's own entries) with the open clients; returns the count. */
int		wire_server_pollfds(wire_server *ws, struct pollfd *pfds, int max);
/* Process readable/hung-up clients; pfds is the array filled by wire_server_pollfds(). */
void	wire_server_service(wire_server *ws, const struct pollfd *pfds, int n);
void	wire_server_close_all(wire_server *ws);

#endif	/* GPUSERV_WIRE_H */
