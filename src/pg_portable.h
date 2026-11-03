/*
 * pg_portable.h
 *
 * Small portability layer so the host side builds on macOS as well as Linux.
 * Everything Linux-specific that PG-Strom used directly is wrapped here:
 *   - the event loop of the GPU service (epoll on Linux, poll(2) elsewhere)
 *   - SIGPOLL, st_mtim, O_DIRECT
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PG_PORTABLE_H
#define PG_PORTABLE_H
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __linux__
#include <sys/epoll.h>
#endif

#ifdef __APPLE__
#include <alloca.h>
#include <pthread.h>
#include <string.h>

/* BSD names for what Linux calls st_mtim, st_ctim and SIGPOLL */
#define st_mtim		st_mtimespec
#define st_ctim		st_ctimespec
#ifndef SIGPOLL
#define SIGPOLL		SIGIO
#endif

/* gettid(2) is a Linux system call; macOS has the thread id through pthread */
static inline pid_t
gettid(void)
{
	uint64_t	tid;

	(void) pthread_threadid_np(NULL, &tid);
	return (pid_t) tid;
}

/* strdupa() is a glibc extension */
#ifndef strdupa
#define strdupa(s)	strcpy((char *) alloca(strlen(s) + 1), (s))
#endif
#endif	/* __APPLE__ */

/*
 * Event loop: wait until one of a small set of descriptors becomes readable.
 * Level-triggered, one event per call (the semantics the GPU service had with
 * epoll_wait(maxevents=1)).
 */
#define PGSTROM_EVENT_READABLE		0x01
#define PGSTROM_EVENT_ERROR			0x02	/* error or hangup on the descriptor */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pgstromEventLoop		pgstromEventLoop;

/* returns NULL (errno set) on failure */
extern pgstromEventLoop *pgstromEventLoopCreate(void);
/* start watching 'fdesc' for readability; 0 on success, -1 (errno set) on failure */
extern int	pgstromEventLoopAdd(pgstromEventLoop *loop, int fdesc);
/*
 * Wait up to timeout_ms (negative: forever). Returns 1 and sets *p_fdesc and
 * *p_events (PGSTROM_EVENT_*) if a descriptor is ready, 0 on timeout, -1 on error
 * (errno set; EINTR is reported like epoll_wait did).
 */
extern int	pgstromEventLoopWait(pgstromEventLoop *loop,
								 int *p_fdesc, int *p_events, int timeout_ms);
extern void	pgstromEventLoopClose(pgstromEventLoop *loop);

#ifdef __cplusplus
}
#endif

/*
 * Open a file bypassing the OS cache where the platform allows it: O_DIRECT on
 * Linux, F_NOCACHE on macOS (which has no O_DIRECT and no alignment rules).
 */
static inline int
pgstromOpenDirect(const char *pathname, int flags, mode_t mode)
{
#ifdef O_DIRECT
	return open(pathname, flags | O_DIRECT, mode);
#else
	int		fdesc = open(pathname, flags, mode);

	if (fdesc >= 0)
		(void) fcntl(fdesc, F_NOCACHE, 1);
	return fdesc;
#endif
}

/*
 * Reserve 'size' bytes for a shared-memory object: fallocate(2) on Linux, so a
 * later page fault cannot fail for lack of memory; ftruncate(2) elsewhere (macOS
 * has no fallocate, and sizes a POSIX shm object only once).
 */
static inline int
pgstromPreallocate(int fdesc, size_t size)
{
#ifdef __linux__
	return fallocate(fdesc, 0, 0, size);
#else
	return ftruncate(fdesc, size);
#endif
}

#endif	/* PG_PORTABLE_H */
