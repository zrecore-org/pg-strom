/*
 * gcache_check.c
 *
 * Self-check of the GPU-Cache store (gcache.c) without a GPU: the device buffers are plain memory here.
 * It applies redo logs the way the backends send them and checks what a scan would see (row slots, the
 * visibility bitmap of sessions with and without transactions of their own, the values the tuple former
 * reads, compaction, overflow).
 *
 *   make -C src/gpuserv gcache_check && ./gcache_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xpu_common.h"
#include "gcache_proto.h"
#include "gcache.h"
#include "scan_caps.h"

/* the Vulkan layer is replaced by malloc */
int
gpu_vk_buffer_create(gpu_vk_ctx *ctx, gpu_vk_buffer *b, size_t size, int want_bda)
{
	(void) ctx; (void) want_bda;
	memset(b, 0, sizeof(*b));
	b->map = calloc(1, size);
	b->size = size;
	return b->map ? 0 : -1;
}

int
gpu_vk_buffer_create_rd(gpu_vk_ctx *ctx, gpu_vk_buffer *b, size_t size, int want_bda)
{
	return gpu_vk_buffer_create(ctx, b, size, want_bda);
}

void
gpu_vk_buffer_destroy(gpu_vk_ctx *ctx, gpu_vk_buffer *b)
{
	(void) ctx;
	free(b->map);
	memset(b, 0, sizeof(*b));
}

static int	nfail = 0;
#define CHECK(cond, ...)	do { if (!(cond)) { printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); nfail++; } } while (0)

static gpu_vk_ctx	vk;
#define DB		1
#define REL		7
#define SIG		99

/* schema: id int4, f text, b bool, w float8 (fixed 8, not GPU readable), t text (var) */
enum { C_ID = 1, C_F, C_B, C_W };

static XpuCommand *
mkop(uint32_t op, const void *data, size_t datalen, uint32_t nrecords)
{
	size_t		len = offsetof(XpuCommand, u.gcop.data) + datalen;
	XpuCommand *x = calloc(1, len);

	x->magic = XpuCommandMagicNumber;
	x->tag = XpuCommandTag__GpuCacheOp;
	x->length = len;
	x->u.gcop.op = op;
	x->u.gcop.database_oid = DB;
	x->u.gcop.table_oid = REL;
	x->u.gcop.table_sig = SIG;
	x->u.gcop.nrecords = nrecords;
	if (datalen)
		memcpy(x->u.gcop.data, data, datalen);
	return x;
}

static uint32_t
rpc(uint32_t op, const void *data, size_t datalen, uint32_t nrecords)
{
	XpuCommand *x = mkop(op, data, datalen, nrecords);
	size_t		rl;
	XpuCommand *r = gcache_handle_op(x, &rl);
	uint32_t	st = 0xffff;

	if (r && r->tag == XpuCommandTag__Success)
		st = r->u.gcop.status;
	else if (r)
		printf("  (error reply: %s)\n", r->u.error.message);
	free(x);
	free(r);
	return st;
}

static void
setup(uint64_t max_rows, uint64_t cache_size)
{
	size_t		len = sizeof(gcache_setup_hdr) + 4 * sizeof(gcache_colspec);
	gcache_setup_hdr *sh = calloc(1, len);
	gcache_colspec *cs = (gcache_colspec *)(sh + 1);

	sh->ncols = 4;
	sh->max_rows = max_rows;
	sh->cache_size = cache_size;
	cs[0] = (gcache_colspec){ .attlen = 4, .attbyval = 1, .attalign = 4, .vt = GCVT_INT, .typid = 23 };
	cs[1] = (gcache_colspec){ .attlen = -1, .attalign = 4, .vt = GCVT_STR, .typid = 25, .avgwidth = 16 };
	cs[2] = (gcache_colspec){ .attlen = 1, .attbyval = 1, .attalign = 1, .vt = GCVT_BOOL, .typid = 16 };
	cs[3] = (gcache_colspec){ .attlen = 8, .attbyval = 1, .attalign = 8, .vt = GCVT_NONE, .typid = 701 };
	CHECK(rpc(GCOP_SETUP, sh, len, 0) == GCST_OK, "setup");
	free(sh);
}

/* log buffer */
static unsigned char lb[1 << 20];
static size_t lbl = 0;
static uint32_t lbn = 0;

static void
lb_reset(void)
{
	lbl = 0;
	lbn = 0;
}

static void
lb_insert(uint32_t blk, uint16_t off, uint32_t xmin, uint32_t xmax, int id, const char *f, int b, double w, int idnull)
{
	unsigned char *p = lb + lbl;
	gcache_rec_hdr *h = (gcache_rec_hdr *) p;
	gcache_rec_insert *ri = (gcache_rec_insert *)(p + sizeof(*h));
	unsigned char *tags = (unsigned char *)(ri + 1);
	unsigned char *d = tags + 8;

	memset(p, 0, 256 + (f ? strlen(f) : 0) + 16);
	h->type = GCREC_INSERT;
	h->natts = 4;
	ri->xmin = xmin; ri->xmax = xmax; ri->blkno = blk; ri->offnum = off;
	if (!idnull)
	{
		tags[0] = 1; memcpy(d, &id, 4); d += 4;
	}
	if (f)
	{
		size_t		sl = strlen(f);
		uint32_t	vl = (uint32_t)(sl < 127 ? sl + 1 : sl + 4);	/* a 1-byte or a 4-byte header */

		tags[1] = 2;
		memcpy(d, &vl, 4); d += 4;
		if (sl < 127)
			*d++ = (unsigned char)((vl << 1) | 1);
		else
		{
			uint32_t	h4 = vl << 2;

			memcpy(d, &h4, 4); d += 4;
		}
		memcpy(d, f, sl); d += sl;
	}
	if (b >= 0)
	{
		tags[2] = 1; *d++ = (unsigned char) b;
	}
	if (w == w)
	{
		tags[3] = 1; memcpy(d, &w, 8); d += 8;
	}
	h->len = (uint32_t)(((d - p) + 7) & ~7);
	lbl += h->len;
	lbn++;
}

static void
lb_delete(uint32_t blk, uint16_t off, uint32_t xid)
{
	unsigned char *p = lb + lbl;
	gcache_rec_hdr *h = (gcache_rec_hdr *) p;
	gcache_rec_delete *rd = (gcache_rec_delete *)(p + sizeof(*h));

	memset(p, 0, sizeof(*h) + sizeof(*rd));
	h->len = sizeof(*h) + sizeof(*rd);
	h->type = GCREC_DELETE;
	rd->xid = xid; rd->blkno = blk; rd->offnum = off;
	lbl += h->len;
	lbn++;
}

static void
lb_xact(uint32_t xid, int commit)
{
	unsigned char *p = lb + lbl;
	gcache_rec_hdr *h = (gcache_rec_hdr *) p;
	gcache_rec_xact *rx = (gcache_rec_xact *)(p + sizeof(*h));

	memset(p, 0, sizeof(*h) + sizeof(*rx));
	h->len = sizeof(*h) + sizeof(*rx);
	h->type = commit ? GCREC_COMMIT : GCREC_ABORT;
	rx->xid = xid;
	lbl += h->len;
	lbn++;
}

static uint32_t
lb_send(void)
{
	uint32_t	st = rpc(GCOP_LOG, lb, lbl, lbn);

	lb_reset();
	return st;
}

/* the ids of the rows a session with transaction ids xids sees, as a sorted string "1,2,3" */
static void
visible(gcache *g, const uint32_t *xids, int nx, char *out, size_t outlen)
{
	gpu_vk_buffer bm;
	int			own = 0;
	char		err[128];
	uint32_t	n = gcache_nrows(g);
	size_t		pos = 0;
	unsigned char tmp[8];

	out[0] = '\0';
	if (gcache_visibility(g, xids, nx, &bm, &own, err, sizeof(err)) != 0)
	{
		snprintf(out, outlen, "ERR %s", err);
		return;
	}
	for (uint32_t r = 0; r < n; r++)
	{
		const unsigned char *ptr;
		uint32_t	len;
		int			rv;

		if (!((((const uint8_t *) bm.map)[r >> 3] >> (r & 7)) & 1))
			continue;
		rv = gcache_fetch(g, C_ID, r, tmp, &ptr, &len, err, sizeof(err));
		if (rv == GCF_FIXED)
		{
			int32_t		id;

			memcpy(&id, ptr, 4);
			pos += snprintf(out + pos, outlen - pos, "%s%d", pos ? "," : "", id);
		}
		else
			pos += snprintf(out + pos, outlen - pos, "%sNULL", pos ? "," : "");
	}
	if (own)
		gpu_vk_buffer_destroy(&vk, &bm);
}

/* the ids in order of the slots may differ from id order: sort the list for a stable comparison */
static int
cmpint(const void *a, const void *b)
{
	return *(const int *) a - *(const int *) b;
}

static void
sorted(char *s)
{
	int			v[4096], n = 0;
	char	   *p = s, *q;
	size_t		pos = 0;

	if (!*s)
		return;
	while (*p)
	{
		v[n++] = (int) strtol(p, &q, 10);
		p = (*q == ',') ? q + 1 : q;
		if (q == p && *q)
			break;
	}
	qsort(v, n, sizeof(int), cmpint);
	s[0] = '\0';
	for (int i = 0; i < n; i++)
		pos += sprintf(s + pos, "%s%d", i ? "," : "", v[i]);
}

#define EXPECT(g, xids, nx, expected) do { char o[1024]; visible(g, xids, nx, o, sizeof(o)); sorted(o); \
	CHECK(strcmp(o, expected) == 0, "visible = [%s], expected [%s]", o, expected); } while (0)

int
main(void)
{
	gcache	   *g;
	const uint32_t none[1] = {0};
	uint32_t	x100[1] = {100}, x101[1] = {101}, x100_101[2] = {100, 101};

	memset(&vk, 0, sizeof(vk));
	vk.limits.maxStorageBufferRange = 1U << 30;
	gcache_init(&vk);

	/* 1. committed rows (an initial load) */
	setup(1000, 64UL << 20);
	g = gcache_find(DB, REL, SIG);
	CHECK(g != NULL, "the cache exists");
	lb_insert(0, 1, 2, 0, 1, "one", 1, 1.5, 0);
	lb_insert(0, 2, 2, 0, 2, "two", 0, 2.5, 0);
	lb_insert(0, 3, 2, 0, 3, NULL, -1, 3.5, 0);
	CHECK(lb_send() == GCST_OK, "initial rows");
	EXPECT(g, none, 0, "1,2,3");

	/* 2. a transaction 100 inserts 10 and deletes 2; another session sees the old state, 100 its own */
	lb_insert(0, 10, 100, 0, 10, "ten", 1, 10.5, 0);
	lb_delete(0, 2, 100);
	CHECK(lb_send() == GCST_OK, "txn 100 records");
	EXPECT(g, none, 0, "1,2,3");
	EXPECT(g, x100, 1, "1,3,10");
	EXPECT(g, x101, 1, "1,2,3");
	{
		/* values */
		char		err[64];
		unsigned char tmp[8];
		const unsigned char *ptr;
		uint32_t	len;
		int			rv;
		uint32_t	slot = 0;
		unsigned char ct[6];

		rv = gcache_fetch(g, C_F, slot, tmp, &ptr, &len, err, sizeof(err));
		CHECK(rv == GCF_RAW_VARLENA && len == 4 && memcmp(ptr + 1, "one", 3) == 0, "text of row 1 (rv=%d len=%u)", rv, len);
		rv = gcache_fetch(g, C_B, 0, tmp, &ptr, &len, err, sizeof(err));
		CHECK(rv == GCF_FIXED && len == 1 && ptr[0] == 1, "bool of row 1");
		rv = gcache_fetch(g, C_B, 1, tmp, &ptr, &len, err, sizeof(err));
		CHECK(rv == GCF_FIXED && ptr[0] == 0, "bool of row 2");
		rv = gcache_fetch(g, C_F, 2, tmp, &ptr, &len, err, sizeof(err));
		CHECK(rv == GCF_NULL, "NULL text of row 3");
		rv = gcache_fetch(g, C_W, 0, tmp, &ptr, &len, err, sizeof(err));
		{
			double		w;

			memcpy(&w, ptr, 8);
			CHECK(rv == GCF_FIXED && len == 8 && w == 1.5, "float8 of row 1");
		}
		gcache_ctid(g, 0, ct);
		CHECK(ct[4] == 1 && ct[2] == 0 && ct[3] == 0, "ctid of row 1");
	}
	lb_xact(100, 1);
	CHECK(lb_send() == GCST_OK, "commit 100");
	EXPECT(g, none, 0, "1,3,10");
	EXPECT(g, x101, 1, "1,3,10");

	/* 3. an aborted transaction leaves nothing; its deleted row comes back */
	lb_insert(0, 11, 101, 0, 11, "eleven", 1, 11.5, 0);
	lb_delete(0, 1, 101);
	CHECK(lb_send() == GCST_OK, "txn 101 records");
	EXPECT(g, none, 0, "1,3,10");
	EXPECT(g, x101, 1, "3,10,11");
	lb_xact(101, 0);
	CHECK(lb_send() == GCST_OK, "abort 101");
	EXPECT(g, none, 0, "1,3,10");
	EXPECT(g, x101, 1, "1,3,10");

	/* 4. two ids of one session (a subtransaction), an insert deleted in the same transaction */
	lb_insert(1, 1, 100, 0, 20, "twenty", 0, 20.5, 0);
	lb_insert(1, 2, 101, 0, 21, "twentyone", 0, 21.5, 0);
	lb_delete(1, 1, 101);						/* the subtransaction deletes the parent's row */
	CHECK(lb_send() == GCST_OK, "records of two ids");
	EXPECT(g, x100_101, 2, "1,3,10,21");
	EXPECT(g, x100, 1, "1,3,10,20");
	EXPECT(g, none, 0, "1,3,10");
	lb_xact(101, 1);
	lb_xact(100, 1);
	CHECK(lb_send() == GCST_OK, "commit both");
	EXPECT(g, none, 0, "1,3,10,21");

	/* 5. an UPDATE: delete + insert; a resent identical insert is harmless */
	lb_delete(0, 3, 100);
	lb_insert(0, 30, 100, 0, 3, "three!", 1, 3.75, 0);
	lb_insert(0, 30, 100, 0, 3, "three!", 1, 3.75, 0);
	CHECK(lb_send() == GCST_OK, "update and a resend");
	EXPECT(g, none, 0, "1,3,10,21");
	EXPECT(g, x100, 1, "1,3,10,21");
	lb_xact(100, 1);
	CHECK(lb_send() == GCST_OK, "commit the update");
	EXPECT(g, none, 0, "1,3,10,21");
	{
		/* a freed slot is reused */
		uint32_t	n0 = gcache_nrows(g);

		lb_insert(2, 1, 2, 0, 50, "fifty", 1, 5.0, 0);
		CHECK(lb_send() == GCST_OK, "reuse a slot");
		CHECK(gcache_nrows(g) == n0, "the hwm did not grow (%u -> %u)", n0, gcache_nrows(g));
		EXPECT(g, none, 0, "1,3,10,21,50");
	}

	/* 6. a delete of an unknown ctid, and an insert of a known ctid with another xmin, drop the cache */
	lb_delete(9, 9, 100);
	CHECK(lb_send() == GCST_INCONSISTENT, "unknown delete");
	CHECK(gcache_find(DB, REL, SIG) == NULL, "the cache is dropped");
	rpc(GCOP_SETUP, NULL, 0, 0);
	setup(1000, 64UL << 20);
	g = gcache_find(DB, REL, SIG);
	lb_insert(0, 1, 2, 0, 1, "one", 1, 1.5, 0);
	lb_insert(0, 1, 100, 0, 1, "uno", 1, 1.5, 0);
	CHECK(lb_send() == GCST_INCONSISTENT, "duplicate ctid with another xmin");

	/* 7. overflow of the row slots */
	setup(32, 64UL << 20);
	g = gcache_find(DB, REL, SIG);
	for (int i = 0; i < 40; i++)
		lb_insert(5, (uint16_t)(i + 1), 2, 0, i, "x", 1, 1.0, 0);
	CHECK(lb_send() == GCST_OVERFLOW, "more rows than max_rows");
	CHECK(gcache_find(DB, REL, SIG) == NULL, "dropped on overflow");

	/* 8. the data region fills with the strings of released rows: compaction on demand makes room */
	setup(2000, 4UL << 20);
	g = gcache_find(DB, REL, SIG);
	{
		char		big[1001];
		int			ok = 1;

		memset(big, 'z', sizeof(big));
		big[sizeof(big) - 1] = '\0';
		for (int round = 0; round < 40 && ok; round++)
		{
			uint32_t	xid = 1000 + 2 * round;

			for (int i = 0; i < 20; i++)
				lb_insert(8, (uint16_t)(1 + (round % 2) * 100 + i), xid, 0, round * 100 + i, big, 1, 1.0, 0);
			lb_xact(xid, 1);
			if (round > 0)
			{
				for (int i = 0; i < 20; i++)
					lb_delete(8, (uint16_t)(1 + ((round - 1) % 2) * 100 + i), xid + 1);
				lb_xact(xid + 1, 1);
			}
			if (lb_send() != GCST_OK)
			{
				ok = 0;
				CHECK(0, "round %d failed", round);
			}
		}
		CHECK(ok, "800 strings of 1000 bytes through a region of ~120KB");
		g = gcache_find(DB, REL, SIG);
		if (g)
		{
			char		o[8192];
			char		want[8192];
			size_t		pos = 0;

			visible(g, none, 0, o, sizeof(o));
			sorted(o);
			for (int i = 0; i < 20; i++)
				pos += snprintf(want + pos, sizeof(want) - pos, "%s%d", i ? "," : "", 3900 + i);
			CHECK(strcmp(o, want) == 0, "the last round is what is visible: [%s]", o);
		}
	}
	rpc(GCOP_ERASE, NULL, 0, 0);
	CHECK(gcache_find(DB, REL, SIG) == NULL, "erased");

	/* 8b. live strings beyond the first region size: the region grows, the other columns keep their data */
	setup(2000, 64UL << 20);
	g = gcache_find(DB, REL, SIG);
	{
		char		big[1001];
		int			bad = 0;

		memset(big, 'q', sizeof(big));
		big[sizeof(big) - 1] = '\0';
		for (int i = 0; i < 600; i++)
			lb_insert(9, (uint16_t)(i + 1), 2, 0, i, big, 1, i + 0.5, 0);
		CHECK(lb_send() == GCST_OK, "600 live strings of 1000 bytes (more than the first region holds)");
		g = gcache_find(DB, REL, SIG);
		if (g)
		{
			unsigned char tmp[8];
			const unsigned char *ptr;
			uint32_t	len;
			char		err[64];

			for (uint32_t r = 0; r < gcache_nrows(g); r++)
			{
				int			rv = gcache_fetch(g, C_F, r, tmp, &ptr, &len, err, sizeof(err));
				double		w;

				if (rv != GCF_RAW_VARLENA || len != 1004 || ptr[4] != 'q' || ptr[1003] != 'q')
				{
					if (bad < 3) printf("  bad string row %u rv=%d len=%u\n", r, rv, len);
					bad++;
				}
				rv = gcache_fetch(g, C_W, r, tmp, &ptr, &len, err, sizeof(err));
				memcpy(&w, ptr, 8);
				if (rv != GCF_FIXED || w != (double) r + 0.5)
					bad++;
			}
			CHECK(bad == 0, "after the growth every string and float8 is intact (%d bad)", bad);
		}
	}
	rpc(GCOP_ERASE, NULL, 0, 0);

	/* 9. compaction on a cache with released rows */
	setup(1000, 64UL << 20);
	g = gcache_find(DB, REL, SIG);
	for (int i = 0; i < 200; i++)
		lb_insert(3, (uint16_t)(i + 1), 100, 0, i, "abcdefghij", 1, 1.0, 0);
	CHECK(lb_send() == GCST_OK, "200 rows");
	lb_xact(100, 1);
	CHECK(lb_send() == GCST_OK, "commit");
	for (int i = 0; i < 200; i += 2)
		lb_delete(3, (uint16_t)(i + 1), 101);
	CHECK(lb_send() == GCST_OK, "delete half");
	lb_xact(101, 1);
	CHECK(lb_send() == GCST_OK, "commit the deletes");
	{
		gcache_info gi;
		XpuCommand *x = mkop(GCOP_INFO, NULL, 0, 0);
		size_t		rl;
		XpuCommand *r = gcache_handle_op(x, &rl);

		memcpy(&gi, r->u.gcop.data, sizeof(gi));
		CHECK(gi.nitems == 100 && gi.dead_space > 0, "100 rows left, dead space %lu", (unsigned long) gi.dead_space);
		free(x); free(r);
		CHECK(rpc(GCOP_COMPACT, NULL, 0, 0) == GCST_OK, "compaction");
		x = mkop(GCOP_INFO, NULL, 0, 0);
		r = gcache_handle_op(x, &rl);
		memcpy(&gi, r->u.gcop.data, sizeof(gi));
		CHECK(gi.nitems == 100 && gi.dead_space == 0 && gi.hwm <= 200, "after compaction: %lu rows, dead %lu, hwm %lu",
			  (unsigned long) gi.nitems, (unsigned long) gi.dead_space, (unsigned long) gi.hwm);
		free(x); free(r);
	}
	{
		char		o[4096];
		unsigned char tmp[8];
		const unsigned char *ptr;
		uint32_t	len;
		char		err[64];
		int			cnt = 0, bad = 0;

		g = gcache_find(DB, REL, SIG);
		visible(g, none, 0, o, sizeof(o));
		for (uint32_t r = 0; r < gcache_nrows(g); r++)
		{
			int			rv = gcache_fetch(g, C_F, r, tmp, &ptr, &len, err, sizeof(err));

			if (rv == GCF_RAW_VARLENA)
			{
				cnt++;
				if (len != 11 || memcmp(ptr + 1, "abcdefghij", 10) != 0)
					bad++;
			}
		}
		CHECK(bad == 0, "strings survived the compaction (%d bad of %d)", bad, cnt);
	}

	gcache_fini();
	printf(nfail == 0 ? "gcache_check: all passed\n" : "gcache_check: %d FAILED\n", nfail);
	return nfail ? 1 : 0;
}
