/*
 * pg_portable.c
 *
 * Implementation of the portability layer (see pg_portable.h).
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
/* libc only, on purpose: this file has no PostgreSQL dependency, so it can be tested alone */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "pg_portable.h"

#ifdef __linux__
/* ----------------------------------------------------------------
 * Linux: epoll
 * ----------------------------------------------------------------
 */
struct pgstromEventLoop
{
	int		epfd;
};

pgstromEventLoop *
pgstromEventLoopCreate(void)
{
	pgstromEventLoop *loop = malloc(sizeof(pgstromEventLoop));

	if (!loop)
	{
		errno = ENOMEM;
		return NULL;
	}
	loop->epfd = epoll_create(30);
	if (loop->epfd < 0)
	{
		int		saved_errno = errno;

		free(loop);
		errno = saved_errno;
		return NULL;
	}
	return loop;
}

int
pgstromEventLoopAdd(pgstromEventLoop *loop, int fdesc)
{
	struct epoll_event ev;

	memset(&ev, 0, sizeof(ev));
	ev.events = EPOLLIN;
	ev.data.fd = fdesc;
	return epoll_ctl(loop->epfd, EPOLL_CTL_ADD, fdesc, &ev);
}

int
pgstromEventLoopWait(pgstromEventLoop *loop,
					 int *p_fdesc, int *p_events, int timeout_ms)
{
	struct epoll_event ev;
	int		status = epoll_wait(loop->epfd, &ev, 1, timeout_ms);

	if (status > 0)
	{
		*p_fdesc = ev.data.fd;
		*p_events = 0;
		if ((ev.events & EPOLLIN) != 0)
			*p_events |= PGSTROM_EVENT_READABLE;
		if ((ev.events & ~EPOLLIN) != 0)
			*p_events |= PGSTROM_EVENT_ERROR;
	}
	return status;
}

void
pgstromEventLoopClose(pgstromEventLoop *loop)
{
	if (loop)
	{
		close(loop->epfd);
		free(loop);
	}
}

#else	/* !__linux__ */
/* ----------------------------------------------------------------
 * Elsewhere (macOS, BSD): poll(2)
 *
 * The set is tiny (the listening socket and the logger pipe), so a pollfd
 * array is as good as kqueue. The scan starts after the descriptor reported
 * last time, so one busy descriptor cannot starve the other.
 * ----------------------------------------------------------------
 */
#include <poll.h>

struct pgstromEventLoop
{
	struct pollfd *fds;
	int		nfds;
	int		nalloc;
	int		cursor;
};

pgstromEventLoop *
pgstromEventLoopCreate(void)
{
	pgstromEventLoop *loop = calloc(1, sizeof(pgstromEventLoop));

	if (!loop)
		errno = ENOMEM;
	return loop;
}

int
pgstromEventLoopAdd(pgstromEventLoop *loop, int fdesc)
{
	if (loop->nfds == loop->nalloc)
	{
		int		nalloc = (loop->nalloc == 0 ? 8 : 2 * loop->nalloc);
		struct pollfd *fds = realloc(loop->fds, sizeof(struct pollfd) * nalloc);

		if (!fds)
		{
			errno = ENOMEM;
			return -1;
		}
		loop->fds = fds;
		loop->nalloc = nalloc;
	}
	loop->fds[loop->nfds].fd = fdesc;
	loop->fds[loop->nfds].events = POLLIN;
	loop->fds[loop->nfds].revents = 0;
	loop->nfds++;
	return 0;
}

int
pgstromEventLoopWait(pgstromEventLoop *loop,
					 int *p_fdesc, int *p_events, int timeout_ms)
{
	int		i, status;

	status = poll(loop->fds, loop->nfds, timeout_ms);
	if (status <= 0)
		return status;
	for (i=0; i < loop->nfds; i++)
	{
		int		k = (loop->cursor + i) % loop->nfds;
		short	revents = loop->fds[k].revents;

		if (revents == 0)
			continue;
		*p_fdesc = loop->fds[k].fd;
		*p_events = 0;
		if ((revents & POLLIN) != 0)
			*p_events |= PGSTROM_EVENT_READABLE;
		if ((revents & ~POLLIN) != 0)
			*p_events |= PGSTROM_EVENT_ERROR;
		loop->cursor = (k + 1) % loop->nfds;
		return 1;
	}
	return 0;
}

void
pgstromEventLoopClose(pgstromEventLoop *loop)
{
	if (loop)
	{
		free(loop->fds);
		free(loop);
	}
}
#endif	/* __linux__ */
