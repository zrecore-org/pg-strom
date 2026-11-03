/*
 * port_test.c
 *
 * Standalone test of the portability layer (src/pg_portable.[ch]): the event loop,
 * pgstromOpenDirect, pgstromPreallocate on a POSIX shm object, gettid and the
 * st_mtim alias. Needs neither PostgreSQL headers nor a server. Builds and runs on
 * Linux and macOS:
 *
 *   make -C test/port check
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "pg_portable.h"

static int	nfail = 0;

#define CHECK(cond, ...)											\
	do {															\
		if (cond)													\
			printf("  ok    %s\n", #cond);							\
		else														\
		{															\
			printf("  FAIL  %s  (line %d) ", #cond, __LINE__);		\
			printf(__VA_ARGS__);									\
			printf("\n");											\
			nfail++;												\
		}															\
	} while (0)

static double
now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void
test_event_loop(void)
{
	pgstromEventLoop *loop = pgstromEventLoopCreate();
	int		pa[2], pb[2], fd, ev, status;
	int		first, second;
	char	c = 'x', buf[8];
	double	t0;

	printf("event loop\n");
	CHECK(loop != NULL, "create failed: %s", strerror(errno));
	CHECK(pipe(pa) == 0 && pipe(pb) == 0, "pipe failed");
	CHECK(pgstromEventLoopAdd(loop, pa[0]) == 0, "add a");
	CHECK(pgstromEventLoopAdd(loop, pb[0]) == 0, "add b");

	/* nothing ready: zero timeout returns 0, a 60 ms timeout really waits */
	status = pgstromEventLoopWait(loop, &fd, &ev, 0);
	CHECK(status == 0, "status=%d", status);
	t0 = now_ms();
	status = pgstromEventLoopWait(loop, &fd, &ev, 60);
	CHECK(status == 0 && now_ms() - t0 >= 50.0, "status=%d waited %.0f ms", status, now_ms() - t0);

	/* both readable and never drained: successive waits must not starve one of them */
	CHECK(write(pa[1], &c, 1) == 1 && write(pb[1], &c, 1) == 1, "write");
	status = pgstromEventLoopWait(loop, &first, &ev, 100);
	CHECK(status == 1 && (ev & PGSTROM_EVENT_READABLE) && !(ev & PGSTROM_EVENT_ERROR),
		  "status=%d ev=%d", status, ev);
	status = pgstromEventLoopWait(loop, &second, &ev, 100);
	CHECK(status == 1 && second != first, "first=%d second=%d (same descriptor twice)", first, second);
	CHECK((first == pa[0] && second == pb[0]) || (first == pb[0] && second == pa[0]),
		  "unexpected descriptors");

	/* drain both; closing the write end of a is reported as an error/hangup */
	CHECK(read(pa[0], buf, sizeof(buf)) == 1 && read(pb[0], buf, sizeof(buf)) == 1, "drain");
	close(pa[1]);
	status = pgstromEventLoopWait(loop, &fd, &ev, 100);
	CHECK(status == 1 && fd == pa[0] && (ev & PGSTROM_EVENT_ERROR), "status=%d fd=%d ev=%d", status, fd, ev);

	pgstromEventLoopClose(loop);
	close(pa[0]); close(pb[0]); close(pb[1]);
}

static void
test_open_direct(void)
{
	char	path[] = "/tmp/pgstrom_port_test_XXXXXX";
	int		fdesc = mkstemp(path);
	void   *wbuf = NULL, *rbuf = NULL;
	size_t	len = 16384;

	printf("pgstromOpenDirect\n");
	CHECK(fdesc >= 0, "mkstemp");
	close(fdesc);
	CHECK(posix_memalign(&wbuf, 4096, len) == 0 && posix_memalign(&rbuf, 4096, len) == 0, "alloc");
	memset(wbuf, 0x5a, len);
	memset(rbuf, 0, len);
	fdesc = pgstromOpenDirect(path, O_WRONLY | O_TRUNC, 0600);
	CHECK(fdesc >= 0, "open for write: %s", strerror(errno));
	CHECK(write(fdesc, wbuf, len) == (ssize_t) len, "write: %s", strerror(errno));
	close(fdesc);
	fdesc = pgstromOpenDirect(path, O_RDONLY, 0);
	CHECK(fdesc >= 0, "open for read: %s", strerror(errno));
	CHECK(read(fdesc, rbuf, len) == (ssize_t) len && memcmp(wbuf, rbuf, len) == 0, "read back differs");
	close(fdesc);
	unlink(path);
	free(wbuf);
	free(rbuf);
}

static void
test_shm_preallocate(void)
{
	/* macOS limits shm names to 31 bytes: the longest name pgstromShmemObjectName makes */
	const char *name = "/pgsb.65535.ffffffff";
	size_t	len = 3 * 16384 + 4096;
	struct stat st;
	int		fdesc;
	char   *p;

	printf("shared memory object\n");
	shm_unlink(name);
	CHECK(strlen(name) <= 31, "name too long for macOS");
	fdesc = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
	CHECK(fdesc >= 0, "shm_open: %s", strerror(errno));
	CHECK(pgstromPreallocate(fdesc, len) == 0, "preallocate: %s", strerror(errno));
	CHECK(fstat(fdesc, &st) == 0 && (size_t) st.st_size >= len, "st_size=%lld", (long long) st.st_size);
	p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fdesc, 0);
	CHECK(p != MAP_FAILED, "mmap: %s", strerror(errno));
	if (p != MAP_FAILED)
	{
		p[0] = 1;
		p[len - 1] = 2;
		CHECK(p[0] == 1 && p[len - 1] == 2, "memory not usable");
		munmap(p, len);
	}
	close(fdesc);
	CHECK(shm_unlink(name) == 0, "shm_unlink: %s", strerror(errno));
}

static void *
tid_thread(void *arg)
{
	*(pid_t *) arg = gettid();
	return NULL;
}

static void
test_misc(void)
{
	pthread_t	th;
	pid_t		main_tid = gettid(), other_tid = 0;
	struct stat	st;

	printf("gettid, st_mtim\n");
	CHECK(gettid() == main_tid && main_tid != 0, "unstable or zero thread id");
	CHECK(pthread_create(&th, NULL, tid_thread, &other_tid) == 0, "pthread_create");
	pthread_join(th, NULL);
	CHECK(other_tid != 0 && other_tid != main_tid, "main=%d other=%d", (int) main_tid, (int) other_tid);
	CHECK(stat("/tmp", &st) == 0 && st.st_mtim.tv_sec == st.st_mtime && st.st_ctim.tv_sec == st.st_ctime,
		  "st_mtim/st_ctim do not match st_mtime/st_ctime");
}

int
main(void)
{
	test_event_loop();
	test_open_direct();
	test_shm_preallocate();
	test_misc();
	printf("\n%s (%d failure%s)\n", nfail ? "FAILED" : "ALL PASSED", nfail, nfail == 1 ? "" : "s");
	return nfail ? 1 : 0;
}
