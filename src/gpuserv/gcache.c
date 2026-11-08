/*
 * gcache.c
 *
 * GPU-Cache of the Vulkan service.
 *
 * A cache is a columnar copy of one table, in one device buffer (host-visible, so the service writes
 * the redo log into it directly and the GPU reads it in place), plus the bookkeeping the service
 * keeps on the host:
 *
 *   row slots     0 ... max_rows-1; a slot holds one row version (a heap tuple's ctid, xmin, xmax). Free
 *                 slots are reused; 'hwm' (high-water mark) is what a scan has to look at.
 *   columns       per column a validity bitmap (bit 1 = not NULL) and, by kind
 *                   fixed   attlen bytes per slot (bool: one bit per slot)
 *                   varlena { offset, length } per slot (8 bytes) into the column's own data region, which
 *                           holds the complete datum (header included, detoasted by the backend) and is
 *                           compacted when it fills
 *                 Every type is stored, whether the GPU can compute with it or not: the kernels read the
 *                 columns the program names (bool, integers, date/time, float4, float8, text, geometry, numeric
 *                 and jsonb: the last three as a string register or decoded from the datum), and the result
 *                 tuples copy the others as they are.
 *   visibility    a bitmap 'vis': bit = the version's xmin is committed. The scan kernels skip rows whose
 *                 bit is 0. A version of a transaction in flight has its transaction id as xmin (or xmax)
 *                 and is listed under that id, so COMMIT / ABORT of the id finalizes it without a scan.
 *                 A session that has modified the table sees its own versions: gcache_visibility() patches a
 *                 copy of the bitmap for it.
 *   ctid hash     the redo log names rows by ctid.
 *
 * What a scan sees: the latest committed state, and the session's own uncommitted changes. This is what
 * the CUDA GPU-Cache offers (it is not a snapshot: a REPEATABLE READ transaction sees rows committed after
 * its snapshot).
 *
 * The service never talks to PostgreSQL. Everything it knows comes from the log records.
 */
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xpu_common.h"
#include "gcache_proto.h"
#include "gcache.h"
#include "scan_caps.h"

#define GC_INVALID_XID		0U
#define GC_FROZEN_XID		2U
#define GC_NO_SLOT			0xffffffffU
#define GC_XTAB_NSLOTS		256
#define GC_BDALIGN			16
#define GC_MIN_REGION		(64U << 10)

enum { GCK_NONE = 0, GCK_FIXED = 1, GCK_BOOL = 2, GCK_VAR = 3 };

typedef struct
{
	uint8_t		kind;
	uint8_t		vt;
	uint16_t	w;						/* fixed: bytes per value */
	uint16_t	attalign;
	int16_t		attlen;
	uint64_t	nm_off, nm_len;			/* validity bitmap */
	uint64_t	v_off, v_len;			/* values (fixed, bool bits, or the pairs of a varlena) */
	uint64_t	x_off, x_cap, x_used;	/* the data region of a varlena column */
	uint64_t	x_dead;					/* bytes of it that belong to released rows */
} gcache_col;

/* the slots a transaction id touched */
typedef struct gcache_xent
{
	struct gcache_xent *next;
	uint32_t	xid;
	uint32_t	nins, capins, ndel, capdel;
	uint32_t   *ins;
	uint32_t   *del;
} gcache_xent;

struct gcache
{
	struct gcache *next;
	uint32_t	database_oid, table_oid, table_sig;
	uint32_t	ncols;
	gcache_col *cols;
	uint32_t	max_rows;
	uint32_t	hwm;					/* slots 0 ... hwm-1 have been used */
	uint32_t	nlive;					/* slots holding a version */
	uint32_t	nfree;
	uint32_t   *free_stack;
	uint64_t   *ctid;					/* per slot, ~0 if the slot is free */
	uint32_t   *xmin, *xmax;
	uint32_t   *hnext;					/* ctid hash chain */
	uint32_t   *bucket;
	uint32_t	nbuckets;
	uint8_t	   *alloc;					/* slot is in use: one byte per slot */
	gpu_vk_buffer bd;					/* the columns */
	gpu_vk_buffer bvis;					/* the visibility bitmap */
	uint64_t	bd_used;
	uint64_t	budget;					/* cache_size */
	gcache_xent *xtab[GC_XTAB_NSLOTS];
	uint64_t	npending;				/* versions of transactions in flight */
	uint64_t	nrecords;
	uint64_t	last_used;
};

static char		gc_why[256];				/* why the last request failed (for the service's log) */
static gpu_vk_ctx  *gc_vk = NULL;
static gcache	   *gc_list = NULL;
static uint64_t		gc_clock = 0;
const kern_session_info *gcache_cur_session = NULL;

static inline uint64_t
align_up(uint64_t v, uint64_t a)
{
	return (v + a - 1) & ~(a - 1);
}

static void
gc_err(char *err, size_t errlen, const char *fmt, ...)
{
	va_list		ap;

	if (!err || errlen == 0)
		return;
	va_start(ap, fmt);
	vsnprintf(err, errlen, fmt, ap);
	va_end(ap);
}

void
gcache_init(gpu_vk_ctx *vk)
{
	gc_vk = vk;
}

/* ---------------------------------------------------------------- */
static void
xent_free_all(gcache *g)
{
	for (int i = 0; i < GC_XTAB_NSLOTS; i++)
	{
		gcache_xent *x = g->xtab[i];

		while (x)
		{
			gcache_xent *nx = x->next;

			free(x->ins);
			free(x->del);
			free(x);
			x = nx;
		}
		g->xtab[i] = NULL;
	}
	g->npending = 0;
}

static void
gcache_destroy(gcache *g)
{
	if (!g)
		return;
	xent_free_all(g);
	if (g->bd.size)
		gpu_vk_buffer_destroy(gc_vk, &g->bd);
	if (g->bvis.size)
		gpu_vk_buffer_destroy(gc_vk, &g->bvis);
	free(g->cols);
	free(g->free_stack);
	free(g->ctid);
	free(g->xmin);
	free(g->xmax);
	free(g->hnext);
	free(g->bucket);
	free(g->alloc);
	free(g);
}

void
gcache_fini(void)
{
	while (gc_list)
	{
		gcache *g = gc_list;

		gc_list = g->next;
		gcache_destroy(g);
	}
}

gcache *
gcache_find(uint32_t database_oid, uint32_t table_oid, uint32_t table_sig)
{
	for (gcache *g = gc_list; g; g = g->next)
	{
		if (g->database_oid == database_oid && g->table_oid == table_oid && g->table_sig == table_sig)
		{
			g->last_used = ++gc_clock;
			return g;
		}
	}
	return NULL;
}

static void
gcache_unlink(gcache *g)
{
	gcache **pp = &gc_list;

	while (*pp && *pp != g)
		pp = &(*pp)->next;
	if (*pp)
		*pp = g->next;
}

/* ---------------------------------------------------------------- */
static inline uint64_t
ctid_key(uint32_t blkno, uint16_t offnum)
{
	return ((uint64_t) blkno << 16) | offnum;
}

static inline uint32_t
ctid_hash(uint64_t key, uint32_t nbuckets)
{
	key ^= key >> 33;
	key *= 0xff51afd7ed558ccdULL;
	key ^= key >> 33;
	key *= 0xc4ceb9fe1a85ec53ULL;
	key ^= key >> 33;
	return (uint32_t)(key & (nbuckets - 1));
}

static uint32_t
gc_lookup(const gcache *g, uint64_t key)
{
	uint32_t	s = g->bucket[ctid_hash(key, g->nbuckets)];

	while (s != GC_NO_SLOT)
	{
		if (g->ctid[s] == key)
			return s;
		s = g->hnext[s];
	}
	return GC_NO_SLOT;
}

static void
gc_hash_remove(gcache *g, uint32_t slot)
{
	uint32_t	h = ctid_hash(g->ctid[slot], g->nbuckets);
	uint32_t   *pp = &g->bucket[h];

	while (*pp != GC_NO_SLOT)
	{
		if (*pp == slot)
		{
			*pp = g->hnext[slot];
			return;
		}
		pp = &g->hnext[*pp];
	}
}

static inline uint8_t *
gc_vis(gcache *g)
{
	return (uint8_t *) g->bvis.map;
}

static inline void
gc_bit_set(uint8_t *bm, uint32_t i, int v)
{
	if (v)
		bm[i >> 3] |= (uint8_t)(1U << (i & 7));
	else
		bm[i >> 3] &= (uint8_t) ~(1U << (i & 7));
}

static inline uint8_t *
gc_col_nm(const gcache *g, const gcache_col *c)
{
	return (uint8_t *) g->bd.map + c->nm_off;
}

/* ---------------------------------------------------------------- */
/* the transaction ids */
static gcache_xent *
xent_get(gcache *g, uint32_t xid, int create)
{
	gcache_xent **pp = &g->xtab[xid % GC_XTAB_NSLOTS];

	for (gcache_xent *x = *pp; x; x = x->next)
		if (x->xid == xid)
			return x;
	if (!create)
		return NULL;
	{
		gcache_xent *x = calloc(1, sizeof(gcache_xent));

		if (!x)
			return NULL;
		x->xid = xid;
		x->next = *pp;
		*pp = x;
		return x;
	}
}

static int
xent_add(uint32_t **arr, uint32_t *n, uint32_t *cap, uint32_t slot)
{
	if (*n == *cap)
	{
		uint32_t	ncap = *cap ? *cap * 2 : 16;
		uint32_t   *na = realloc(*arr, sizeof(uint32_t) * ncap);

		if (!na)
			return -1;
		*arr = na;
		*cap = ncap;
	}
	(*arr)[(*n)++] = slot;
	return 0;
}

static void
xent_remove(gcache *g, uint32_t xid)
{
	gcache_xent **pp = &g->xtab[xid % GC_XTAB_NSLOTS];

	while (*pp)
	{
		gcache_xent *x = *pp;

		if (x->xid == xid)
		{
			*pp = x->next;
			free(x->ins);
			free(x->del);
			free(x);
			return;
		}
		pp = &x->next;
	}
}

/* ---------------------------------------------------------------- */
/* slots */
static void
gc_slot_free(gcache *g, uint32_t s)
{
	if (!g->alloc[s])
		return;
	gc_hash_remove(g, s);
	g->alloc[s] = 0;
	g->ctid[s] = ~(uint64_t) 0;
	if (g->xmin[s] != GC_FROZEN_XID && g->xmin[s] != GC_INVALID_XID)
		g->npending--;
	if (g->xmax[s] != GC_INVALID_XID)
		g->npending--;
	g->xmin[s] = g->xmax[s] = GC_INVALID_XID;
	gc_bit_set(gc_vis(g), s, 0);
	for (uint32_t c = 0; c < g->ncols; c++)
	{
		gcache_col *col = &g->cols[c];

		if (col->kind == GCK_VAR && ((gc_col_nm(g, col)[s >> 3] >> (s & 7)) & 1))
		{
			uint64_t	pr;

			memcpy(&pr, (uint8_t *) g->bd.map + col->v_off + 8 * (uint64_t) s, 8);
			col->x_dead += (uint32_t)(pr >> 32);
		}
	}
	g->free_stack[g->nfree++] = s;
	g->nlive--;
}

static int
gc_slot_alloc(gcache *g, uint32_t *p_slot)
{
	if (g->nfree > 0)
		*p_slot = g->free_stack[--g->nfree];
	else if (g->hwm < g->max_rows)
		*p_slot = g->hwm++;
	else
		return -1;
	g->nlive++;
	return 0;
}

/* compact the data region of a varlena column: the live strings, one after the other */
static int
gc_compact_col(gcache *g, gcache_col *col)
{
	uint8_t	   *base = (uint8_t *) g->bd.map;
	uint8_t	   *tmp;
	uint64_t	pos = 0;
	const uint8_t *nm = gc_col_nm(g, col);

	if (col->x_used == 0)
		return 0;
	tmp = malloc(col->x_used);
	if (!tmp)
		return -1;
	for (uint32_t s = 0; s < g->hwm; s++)
	{
		uint64_t	pr;
		uint32_t	off, len;

		if (!g->alloc[s] || !((nm[s >> 3] >> (s & 7)) & 1))
			continue;
		memcpy(&pr, base + col->v_off + 8 * (uint64_t) s, 8);
		off = (uint32_t) pr;
		len = (uint32_t)(pr >> 32);
		if ((uint64_t) off + len > col->x_used)
		{
			free(tmp);
			return -1;
		}
		pos = align_up(pos, 4);
		memcpy(tmp + pos, base + col->x_off + off, len);
		pr = (uint64_t) pos | ((uint64_t) len << 32);
		memcpy(base + col->v_off + 8 * (uint64_t) s, &pr, 8);
		pos += len;
	}
	memcpy(base + col->x_off, tmp, pos);
	free(tmp);
	col->x_used = pos;
	col->x_dead = 0;
	return 0;
}

/*
 * Make the data region of column 'cidx' larger: a new device buffer, the old contents copied to it. The
 * service runs one task at a time, so nothing on the GPU uses the old buffer. The total stays within the
 * cache's budget (cache_size, or what the first layout needed if that was more) and the device's limit.
 */
static int
gc_grow_col(gcache *g, uint32_t cidx, uint64_t need)
{
	gcache_col *col = &g->cols[cidx];
	uint64_t	newcap = col->x_cap * 2;
	uint64_t	limit = gc_vk->limits.maxStorageBufferRange;
	uint64_t	budget = (g->budget > g->bd_used ? g->budget : g->bd_used);
	uint64_t	shift, total;
	gpu_vk_buffer nb;
	uint8_t	   *ob, *nbp;

	if (gc_vk->max_buffer_size != 0 && gc_vk->max_buffer_size < limit)
		limit = gc_vk->max_buffer_size;
	if (limit > 0xfffffff0ULL)
		limit = 0xfffffff0ULL;
	if (newcap < need + (need >> 2))
		newcap = need + (need >> 2);
	newcap = align_up(newcap, GC_BDALIGN);
	shift = newcap - col->x_cap;
	total = g->bd_used + shift;
	if (total > limit || total > budget)
	{
		/* grow as far as the budget allows */
		uint64_t	room = (limit < budget ? limit : budget);

		if (room <= g->bd_used)
			return -1;
		shift = (room - g->bd_used) & ~(uint64_t)(GC_BDALIGN - 1);
		newcap = col->x_cap + shift;
		total = g->bd_used + shift;
		if (newcap < need)
			return -1;
	}
	if (gpu_vk_buffer_create(gc_vk, &nb, (size_t) total, 0) != 0)
		return -1;
	ob = (uint8_t *) g->bd.map;
	nbp = (uint8_t *) nb.map;
	/* everything up to the end of this column's region as it is; the later regions move up by 'shift' */
	memcpy(nbp, ob, col->x_off + col->x_cap);
	for (uint32_t c = cidx + 1; c < g->ncols; c++)
	{
		gcache_col *oc = &g->cols[c];

		if (oc->kind == GCK_VAR)
		{
			memcpy(nbp + oc->x_off + shift, ob + oc->x_off, oc->x_cap);
			oc->x_off += shift;
		}
	}
	col->x_cap = newcap;
	gpu_vk_buffer_destroy(gc_vk, &g->bd);
	g->bd = nb;
	g->bd_used = total;
	return 0;
}

static int
gc_x_alloc(gcache *g, gcache_col *col, uint32_t len, uint32_t *p_off)
{
	uint64_t	pos = align_up(col->x_used, 4);

	if (pos + len > col->x_cap)
	{
		if (col->x_dead > 0 && gc_compact_col(g, col) != 0)
			return -1;
		pos = align_up(col->x_used, 4);
		if (pos + len > col->x_cap)
		{
			uint32_t	cidx = (uint32_t)(col - g->cols);

			if (gc_grow_col(g, cidx, pos + len) != 0)
			{
				snprintf(gc_why, sizeof(gc_why), "the data region of column %u is full (%llu bytes) and the cache cannot grow",
						 cidx + 1, (unsigned long long) col->x_cap);
				return -1;
			}
		}
	}
	*p_off = (uint32_t) pos;
	col->x_used = pos + len;
	return 0;
}

/* ---------------------------------------------------------------- */
/* SETUP */
static gcache *
gcache_create(const kern_gcache_op *op, const gcache_setup_hdr *sh, const gcache_colspec *spec,
			  char *err, size_t errlen)
{
	gcache	   *g;
	uint64_t	max_rows = sh->max_rows;
	uint64_t	cache_size = sh->cache_size;
	uint64_t	row_fixed = 0, var_demand_row = 0, fixed_total = 0, var_demand_total = 0, var_budget;
	uint64_t	pos = 0;
	uint64_t	limit;
	uint32_t	ncols = sh->ncols;

	g = calloc(1, sizeof(gcache));
	if (!g)
	{
		gc_err(err, errlen, "out of memory");
		return NULL;
	}
	g->database_oid = op->database_oid;
	g->table_oid = op->table_oid;
	g->table_sig = op->table_sig;
	g->ncols = ncols;
	g->cols = calloc(ncols ? ncols : 1, sizeof(gcache_col));
	if (!g->cols)
		goto oom;
	/* the shape of the columns, and how many rows fit the budget */
	for (uint32_t c = 0; c < ncols; c++)
	{
		const gcache_colspec *cs = &spec[c];
		gcache_col *col = &g->cols[c];

		col->attlen = cs->attlen;
		col->attalign = cs->attalign;
		col->vt = cs->vt;
		if (cs->dropped)
			col->kind = GCK_NONE;
		else if (cs->attlen == -1)
		{
			col->kind = GCK_VAR;
			row_fixed += 8;
			var_demand_row += (cs->avgwidth ? cs->avgwidth : 32) + 8;
		}
		else if (cs->attlen > 0 && cs->attlen <= 1024)
		{
			if (cs->vt == GCVT_BOOL && cs->attlen == 1)
			{
				col->kind = GCK_BOOL;
				row_fixed += 1;			/* a bit, rounded up */
			}
			else
			{
				col->kind = GCK_FIXED;
				col->w = (uint16_t) cs->attlen;
				row_fixed += (uint64_t) cs->attlen;
			}
		}
		else
		{
			gc_err(err, errlen, "gpucache: column %u has an unsupported length (%d)", c + 1, (int) cs->attlen);
			gcache_destroy(g);
			return NULL;
		}
		if (col->kind != GCK_NONE)
			row_fixed += 1;				/* the validity bit */
	}
	if (max_rows == 0 || max_rows > 0x7fffffffULL)
		max_rows = 0x7fffffffULL;
	{
		uint64_t	per_row = row_fixed + var_demand_row + var_demand_row / 4 + 24;
		uint64_t	fit = (cache_size * 9 / 10) / per_row;

		if (fit < max_rows)
			max_rows = fit;
	}
	if (max_rows < 16)
	{
		gc_err(err, errlen, "gpucache: cache_size (%llu bytes) is too small for this table", (unsigned long long) cache_size);
		gcache_destroy(g);
		return NULL;
	}
	g->max_rows = (uint32_t) max_rows;

	/* the layout of the columns buffer */
	for (uint32_t c = 0; c < ncols; c++)
	{
		gcache_col *col = &g->cols[c];

		if (col->kind == GCK_NONE)
			continue;
		col->nm_off = pos;
		col->nm_len = align_up((max_rows + 7) / 8, 8);
		pos = align_up(pos + col->nm_len, GC_BDALIGN);
		col->v_off = pos;
		if (col->kind == GCK_BOOL)
			col->v_len = align_up((max_rows + 7) / 8, 8);
		else if (col->kind == GCK_FIXED)
			col->v_len = align_up(max_rows * col->w, 8);
		else
			col->v_len = max_rows * 8;
		pos = align_up(pos + col->v_len, GC_BDALIGN);
	}
	fixed_total = pos;
	for (uint32_t c = 0; c < ncols; c++)
		if (g->cols[c].kind == GCK_VAR)
			var_demand_total += max_rows * ((spec[c].avgwidth ? spec[c].avgwidth : 32) + 8) * 5 / 4 + GC_MIN_REGION;
	if (cache_size > fixed_total)
		var_budget = cache_size - fixed_total;
	else
		var_budget = 0;
	for (uint32_t c = 0; c < ncols; c++)
	{
		gcache_col *col = &g->cols[c];
		uint64_t	demand, cap;

		if (col->kind != GCK_VAR)
			continue;
		demand = max_rows * ((spec[c].avgwidth ? spec[c].avgwidth : 32) + 8) * 5 / 4 + GC_MIN_REGION;
		cap = (var_demand_total > var_budget ? demand * var_budget / var_demand_total : demand);
		if (cap < GC_MIN_REGION)
			cap = GC_MIN_REGION;
		col->x_off = pos;
		col->x_cap = align_up(cap, GC_BDALIGN);
		pos = col->x_off + col->x_cap;
	}
	g->bd_used = pos;
	g->budget = cache_size;
	limit = gc_vk->limits.maxStorageBufferRange;
	if (gc_vk->max_buffer_size != 0 && gc_vk->max_buffer_size < limit)
		limit = gc_vk->max_buffer_size;
	if (limit > 0xfffffff0ULL)
		limit = 0xfffffff0ULL;
	if (pos > limit)
	{
		gc_err(err, errlen, "gpucache: the cache needs %llu bytes of device memory, the device's buffer limit is %llu; "
			   "lower max_num_rows or cache_size", (unsigned long long) pos, (unsigned long long) limit);
		gcache_destroy(g);
		return NULL;
	}

	g->free_stack = malloc(sizeof(uint32_t) * g->max_rows);
	g->ctid = malloc(sizeof(uint64_t) * g->max_rows);
	g->xmin = calloc(g->max_rows, sizeof(uint32_t));
	g->xmax = calloc(g->max_rows, sizeof(uint32_t));
	g->hnext = malloc(sizeof(uint32_t) * g->max_rows);
	g->alloc = calloc(g->max_rows, 1);
	g->nbuckets = 1024;
	while (g->nbuckets < g->max_rows)
		g->nbuckets <<= 1;
	g->bucket = malloc(sizeof(uint32_t) * g->nbuckets);
	if (!g->free_stack || !g->ctid || !g->xmin || !g->xmax || !g->hnext || !g->alloc || !g->bucket)
		goto oom;
	memset(g->bucket, 0xff, sizeof(uint32_t) * g->nbuckets);
	memset(g->ctid, 0xff, sizeof(uint64_t) * g->max_rows);
	if (gpu_vk_buffer_create(gc_vk, &g->bd, pos > 16 ? (size_t) pos : 16, 0) != 0)
	{
		gc_err(err, errlen, "gpucache: cannot allocate %llu bytes of device memory: %s",
			   (unsigned long long) pos, gc_vk->err);
		memset(&g->bd, 0, sizeof(g->bd));
		gcache_destroy(g);
		return NULL;
	}
	if (gpu_vk_buffer_create(gc_vk, &g->bvis, (size_t) align_up(((uint64_t) g->max_rows + 31) / 32 * 4, 16), 0) != 0)
	{
		gc_err(err, errlen, "gpucache: cannot allocate the visibility bitmap: %s", gc_vk->err);
		memset(&g->bvis, 0, sizeof(g->bvis));
		gcache_destroy(g);
		return NULL;
	}
	memset(g->bvis.map, 0, g->bvis.size);
	for (uint32_t c = 0; c < ncols; c++)
		if (g->cols[c].kind != GCK_NONE)
			memset((uint8_t *) g->bd.map + g->cols[c].nm_off, 0, g->cols[c].nm_len);
	g->last_used = ++gc_clock;
	return g;
oom:
	gc_err(err, errlen, "out of memory");
	gcache_destroy(g);
	return NULL;
}

/* ---------------------------------------------------------------- */
/* INSERT */
static int
gc_insert(gcache *g, const unsigned char *rec, uint32_t len, uint32_t natts)
{
	const gcache_rec_insert *ri;
	const unsigned char *tags, *p, *end = rec + len;
	uint64_t	key;
	uint32_t	slot;
	size_t		tagsz = (size_t)((natts + 7) & ~7U);

	if (len < sizeof(gcache_rec_hdr) + sizeof(gcache_rec_insert) + tagsz || natts != g->ncols)
		return GCST_BAD_REQUEST;
	ri = (const gcache_rec_insert *)(rec + sizeof(gcache_rec_hdr));
	tags = (const unsigned char *)(ri + 1);
	p = tags + tagsz;
	key = ctid_key(ri->blkno, ri->offnum);
	slot = gc_lookup(g, key);
	if (slot != GC_NO_SLOT)
	{
		/* a version with this ctid is there: only an identical resend is harmless */
		if (g->xmin[slot] != ri->xmin || g->xmax[slot] != ri->xmax)
		{
			snprintf(gc_why, sizeof(gc_why), "ctid (%u,%u) is inserted again with another version", ri->blkno, ri->offnum);
			return GCST_INCONSISTENT;
		}
		/* overwrite it below */
	}
	else
	{
		if (gc_slot_alloc(g, &slot) != 0)
		{
			snprintf(gc_why, sizeof(gc_why), "more row versions than max_num_rows");
			return GCST_OVERFLOW;
		}
		g->alloc[slot] = 1;
		g->ctid[slot] = key;
		{
			uint32_t	h = ctid_hash(key, g->nbuckets);

			g->hnext[slot] = g->bucket[h];
			g->bucket[h] = slot;
		}
		g->xmin[slot] = ri->xmin;
		g->xmax[slot] = ri->xmax;
		if (ri->xmin != GC_FROZEN_XID && ri->xmin != GC_INVALID_XID)
		{
			gcache_xent *x = xent_get(g, ri->xmin, 1);

			if (!x || xent_add(&x->ins, &x->nins, &x->capins, slot) != 0)
				return GCST_OVERFLOW;
			g->npending++;
		}
		if (ri->xmax != GC_INVALID_XID)
		{
			gcache_xent *x = xent_get(g, ri->xmax, 1);

			if (!x || xent_add(&x->del, &x->ndel, &x->capdel, slot) != 0)
				return GCST_OVERFLOW;
			g->npending++;
		}
	}
	for (uint32_t c = 0; c < g->ncols; c++)
	{
		gcache_col *col = &g->cols[c];
		uint8_t		tag = tags[c];

		if (col->kind == GCK_NONE)
		{
			if (tag != 0)
				return GCST_BAD_REQUEST;
			continue;
		}
		if (tag == 0)
		{
			gc_bit_set(gc_col_nm(g, col), slot, 0);
			if (col->kind == GCK_VAR)
			{
				/* a version being overwritten may have owned a string */
			}
			continue;
		}
		if (col->kind == GCK_VAR)
		{
			uint32_t	vl, off;
			uint64_t	pr;

			if (tag != 2 || p + 4 > end)
				return GCST_BAD_REQUEST;
			memcpy(&vl, p, 4);
			p += 4;
			if (vl == 0 || p + vl > end)
				return GCST_BAD_REQUEST;
			if (gc_x_alloc(g, col, vl, &off) != 0)
				return GCST_OVERFLOW;
			memcpy((uint8_t *) g->bd.map + col->x_off + off, p, vl);   /* the buffer may have moved */
			p += vl;
			pr = (uint64_t) off | ((uint64_t) vl << 32);
			memcpy((uint8_t *) g->bd.map + col->v_off + 8 * (uint64_t) slot, &pr, 8);
		}
		else
		{
			uint32_t	w = (col->kind == GCK_BOOL ? 1 : col->w);

			if (tag != 1 || p + w > end)
				return GCST_BAD_REQUEST;
			if (col->kind == GCK_BOOL)
				gc_bit_set((uint8_t *) g->bd.map + col->v_off, slot, p[0] != 0);
			else
				memcpy((uint8_t *) g->bd.map + col->v_off + (uint64_t) w * slot, p, w);
			p += w;
		}
		gc_bit_set(gc_col_nm(g, col), slot, 1);
	}
	/* the version is visible to everyone once its xmin is committed (an xmax in flight does not hide it) */
	gc_bit_set(gc_vis(g), slot, ri->xmin == GC_FROZEN_XID);
	return GCST_OK;
}

static int
gc_delete(gcache *g, const gcache_rec_delete *rd)
{
	uint32_t	slot = gc_lookup(g, ctid_key(rd->blkno, rd->offnum));
	gcache_xent *x;

	if (slot == GC_NO_SLOT)
	{
		snprintf(gc_why, sizeof(gc_why), "delete of an unknown ctid (%u,%u)", rd->blkno, rd->offnum);
		return GCST_INCONSISTENT;
	}
	if (g->xmax[slot] == rd->xid)
		return GCST_OK;
	if (g->xmax[slot] != GC_INVALID_XID)
		return GCST_INCONSISTENT;
	g->xmax[slot] = rd->xid;
	x = xent_get(g, rd->xid, 1);
	if (!x || xent_add(&x->del, &x->ndel, &x->capdel, slot) != 0)
		return GCST_OVERFLOW;
	g->npending++;
	return GCST_OK;
}

static void
gc_commit(gcache *g, uint32_t xid)
{
	gcache_xent *x = xent_get(g, xid, 0);

	if (!x)
		return;
	for (uint32_t i = 0; i < x->nins; i++)
	{
		uint32_t	s = x->ins[i];

		if (s < g->hwm && g->alloc[s] && g->xmin[s] == xid)
		{
			g->xmin[s] = GC_FROZEN_XID;
			g->npending--;
			gc_bit_set(gc_vis(g), s, 1);
		}
	}
	for (uint32_t i = 0; i < x->ndel; i++)
	{
		uint32_t	s = x->del[i];

		if (s < g->hwm && g->alloc[s] && g->xmax[s] == xid)
			gc_slot_free(g, s);		/* its version is gone for good */
	}
	xent_remove(g, xid);
}

static void
gc_abort(gcache *g, uint32_t xid)
{
	gcache_xent *x = xent_get(g, xid, 0);

	if (!x)
		return;
	for (uint32_t i = 0; i < x->nins; i++)
	{
		uint32_t	s = x->ins[i];

		if (s < g->hwm && g->alloc[s] && g->xmin[s] == xid)
			gc_slot_free(g, s);
	}
	for (uint32_t i = 0; i < x->ndel; i++)
	{
		uint32_t	s = x->del[i];

		if (s < g->hwm && g->alloc[s] && g->xmax[s] == xid)
		{
			g->xmax[s] = GC_INVALID_XID;
			g->npending--;
		}
	}
	xent_remove(g, xid);
}

/* apply the records of a LOG request; returns a GCST_* */
static int
gc_apply_log(gcache *g, const unsigned char *data, size_t datalen, uint32_t nrecords)
{
	size_t		off = 0;

	for (uint32_t i = 0; i < nrecords; i++)
	{
		const gcache_rec_hdr *h;
		int			rv = GCST_OK;

		if (off + sizeof(gcache_rec_hdr) > datalen)
			return GCST_BAD_REQUEST;
		h = (const gcache_rec_hdr *)(data + off);
		if (h->len < sizeof(gcache_rec_hdr) || (h->len & 7) != 0 || off + h->len > datalen)
			return GCST_BAD_REQUEST;
		switch (h->type)
		{
			case GCREC_INSERT:
				rv = gc_insert(g, data + off, h->len, h->natts);
				break;
			case GCREC_DELETE:
				if (h->len < sizeof(gcache_rec_hdr) + sizeof(gcache_rec_delete))
					return GCST_BAD_REQUEST;
				rv = gc_delete(g, (const gcache_rec_delete *)(data + off + sizeof(gcache_rec_hdr)));
				break;
			case GCREC_COMMIT:
			case GCREC_ABORT:
				if (h->len < sizeof(gcache_rec_hdr) + sizeof(gcache_rec_xact))
					return GCST_BAD_REQUEST;
				{
					const gcache_rec_xact *rx = (const gcache_rec_xact *)(data + off + sizeof(gcache_rec_hdr));

					if (h->type == GCREC_COMMIT)
						gc_commit(g, rx->xid);
					else
						gc_abort(g, rx->xid);
				}
				break;
			default:
				return GCST_BAD_REQUEST;
		}
		if (rv != GCST_OK)
			return rv;
		off += h->len;
		g->nrecords++;
	}
	return GCST_OK;
}

/* ---------------------------------------------------------------- */
static int
gc_compact(gcache *g)
{
	for (uint32_t c = 0; c < g->ncols; c++)
		if (g->cols[c].kind == GCK_VAR && gc_compact_col(g, &g->cols[c]) != 0)
			return -1;
	/* give back the free slots above the last used one */
	while (g->hwm > 0 && !g->alloc[g->hwm - 1])
		g->hwm--;
	g->nfree = 0;
	for (uint32_t s = 0; s < g->hwm; s++)
		if (!g->alloc[s])
			g->free_stack[g->nfree++] = s;
	return 0;
}

static void
gc_fill_info(const gcache *g, gcache_info *gi)
{
	uint64_t	usage = 0, dead = 0;

	memset(gi, 0, sizeof(*gi));
	for (uint32_t c = 0; c < g->ncols; c++)
		if (g->cols[c].kind == GCK_VAR)
		{
			usage += g->cols[c].x_used;
			dead += g->cols[c].x_dead;
		}
	gi->max_rows = g->max_rows;
	gi->nitems = g->nlive;
	gi->nfree = g->nfree;
	gi->hwm = g->hwm;
	gi->usage = usage;
	gi->dead_space = dead;
	gi->virtual_mem_sz = g->bd.size;
	gi->physical_mem_sz = g->bd_used - 0;
	gi->pending_rows = g->npending;
	gi->nrecords = g->nrecords;
}

/* ---------------------------------------------------------------- */
static XpuCommand *
gc_reply(uint32_t status, const void *data, size_t datalen, uint64_t arg, size_t *p_len)
{
	size_t		len = offsetof(XpuCommand, u.gcop.data) + datalen;
	XpuCommand *resp = calloc(1, len ? len : 1);

	if (!resp)
		return NULL;
	resp->magic = XpuCommandMagicNumber;
	resp->tag = XpuCommandTag__Success;
	resp->length = len;
	resp->u.gcop.status = status;
	resp->u.gcop.arg = arg;
	if (datalen > 0)
		memcpy(resp->u.gcop.data, data, datalen);
	*p_len = len;
	return resp;
}

static XpuCommand *
gc_reply_error(const char *msg, size_t *p_len)
{
	XpuCommand *resp = calloc(1, sizeof(XpuCommand));

	if (!resp)
		return NULL;
	resp->magic = XpuCommandMagicNumber;
	resp->tag = XpuCommandTag__Error;
	resp->length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	resp->u.error.errcode = ERRCODE_DEVICE_ERROR;
	snprintf(resp->u.error.filename, sizeof(resp->u.error.filename), "gcache");
	snprintf(resp->u.error.funcname, sizeof(resp->u.error.funcname), "gcache");
	snprintf(resp->u.error.message, sizeof(resp->u.error.message), "%s", msg);
	*p_len = resp->length;
	return resp;
}

XpuCommand *
gcache_handle_op(const XpuCommand *xcmd, size_t *p_len)
{
	const kern_gcache_op *op = &xcmd->u.gcop;
	size_t		datalen;
	const unsigned char *data;
	gcache	   *g;
	char		msg[256];

	if (!gc_vk || xcmd->length < offsetof(XpuCommand, u.gcop.data))
		return gc_reply_error("gpucache: malformed request", p_len);
	datalen = xcmd->length - offsetof(XpuCommand, u.gcop.data);
	data = (const unsigned char *) op->data;
	switch (op->op)
	{
		case GCOP_SETUP:
		{
			const gcache_setup_hdr *sh = (const gcache_setup_hdr *) data;
			gcache	   *ng;

			if (datalen < sizeof(*sh) || sh->ncols == 0 || sh->ncols > GCACHE_MAXCOLS ||
				datalen < sizeof(*sh) + sizeof(gcache_colspec) * sh->ncols)
				return gc_reply_error("gpucache: malformed SETUP request", p_len);
			/* a table has one cache: older definitions and the old copy of this one go */
			for (gcache **pp = &gc_list; *pp; )
			{
				gcache *o = *pp;

				if (o->database_oid == op->database_oid && o->table_oid == op->table_oid)
				{
					*pp = o->next;
					gcache_destroy(o);
				}
				else
					pp = &o->next;
			}
			msg[0] = '\0';
			ng = gcache_create(op, sh, (const gcache_colspec *)(sh + 1), msg, sizeof(msg));
			if (!ng)
				return gc_reply_error(msg[0] ? msg : "gpucache: cannot create the cache", p_len);
			ng->next = gc_list;
			gc_list = ng;
			{
				gcache_info gi;

				gc_fill_info(ng, &gi);
				return gc_reply(GCST_OK, &gi, sizeof(gi), ng->max_rows, p_len);
			}
		}
		case GCOP_LOG:
		{
			int			rv;

			g = gcache_find(op->database_oid, op->table_oid, op->table_sig);
			if (!g)
				return gc_reply(GCST_NOT_FOUND, NULL, 0, 0, p_len);
			gc_why[0] = '\0';
			rv = gc_apply_log(g, data, datalen, op->nrecords);
			if (rv == GCST_OVERFLOW || rv == GCST_INCONSISTENT || rv == GCST_BAD_REQUEST)
			{
				fprintf(stderr, "pgstrom-gpuserv: GPU-Cache of table %u dropped (status %d, %u rows in use of %u): %s\n",
						g->table_oid, rv, g->nlive, g->max_rows, gc_why[0] ? gc_why : "see the request");
				/* the cache no longer mirrors the table: drop it */
				gcache_unlink(g);
				gcache_destroy(g);
			}
			return gc_reply(rv, NULL, 0, 0, p_len);
		}
		case GCOP_ERASE:
		{
			int			n = 0;

			for (gcache **pp = &gc_list; *pp; )
			{
				gcache *o = *pp;

				if (o->database_oid == op->database_oid && o->table_oid == op->table_oid &&
					(op->table_sig == 0 || o->table_sig == op->table_sig))
				{
					*pp = o->next;
					gcache_destroy(o);
					n++;
				}
				else
					pp = &o->next;
			}
			return gc_reply(GCST_OK, NULL, 0, n, p_len);
		}
		case GCOP_INFO:
		case GCOP_STAT:
		{
			gcache_info gi;

			g = gcache_find(op->database_oid, op->table_oid, op->table_sig);
			if (!g)
				return gc_reply(GCST_NOT_FOUND, NULL, 0, 0, p_len);
			gc_fill_info(g, &gi);
			return gc_reply(GCST_OK, &gi, sizeof(gi), g->nlive, p_len);
		}
		case GCOP_COMPACT:
			g = gcache_find(op->database_oid, op->table_oid, op->table_sig);
			if (!g)
				return gc_reply(GCST_NOT_FOUND, NULL, 0, 0, p_len);
			if (gc_compact(g) != 0)
			{
				gcache_unlink(g);
				gcache_destroy(g);
				return gc_reply(GCST_INCONSISTENT, NULL, 0, 0, p_len);
			}
			return gc_reply(GCST_OK, NULL, 0, 0, p_len);
		default:
			break;
	}
	return gc_reply_error("gpucache: unknown request", p_len);
}

/* ---------------------------------------------------------------- */
/* what a task reads */
uint32_t
gcache_nrows(const gcache *g)
{
	return g->hwm;
}

uint32_t
gcache_nlive(const gcache *g)
{
	return g->nlive;
}

uint32_t
gcache_ncols(const gcache *g)
{
	return g->ncols;
}

gpu_vk_buffer *
gcache_data_buffer(gcache *g)
{
	return &g->bd;
}

static inline int
xid_in(const uint32_t *xids, int nxids, uint32_t xid)
{
	for (int i = 0; i < nxids; i++)
		if (xids[i] == xid)
			return 1;
	return 0;
}

int
gcache_visibility(gcache *g, const uint32_t *xids, int nxids, gpu_vk_buffer *bitmap, int *own,
				  char *err, size_t errlen)
{
	int			npatch = 0;
	uint8_t	   *bm = NULL;

	*own = 0;
	if (g->npending > 0)
	{
		for (int i = 0; i < nxids; i++)
		{
			const gcache_xent *x;

			if (xids[i] == GC_INVALID_XID || xids[i] == GC_FROZEN_XID)
				continue;
			x = xent_get(g, xids[i], 0);
			if (!x)
				continue;
			for (uint32_t k = 0; k < x->nins && !npatch; k++)
				if (x->ins[k] < g->hwm && g->alloc[x->ins[k]] && g->xmin[x->ins[k]] == x->xid)
					npatch = 1;
			for (uint32_t k = 0; k < x->ndel && !npatch; k++)
				if (x->del[k] < g->hwm && g->alloc[x->del[k]] && g->xmax[x->del[k]] == x->xid)
					npatch = 1;
		}
	}
	if (!npatch)
	{
		*bitmap = g->bvis;				/* the cache's own: not to be destroyed */
		return 0;
	}
	if (gpu_vk_buffer_create(gc_vk, bitmap, g->bvis.size, 0) != 0)
	{
		gc_err(err, errlen, "gpucache: cannot allocate a bitmap: %s", gc_vk->err);
		return -1;
	}
	*own = 1;
	bm = (uint8_t *) bitmap->map;
	memcpy(bm, g->bvis.map, g->bvis.size);
	for (int i = 0; i < nxids; i++)
	{
		const gcache_xent *x;

		if (xids[i] == GC_INVALID_XID || xids[i] == GC_FROZEN_XID)
			continue;
		x = xent_get(g, xids[i], 0);
		if (!x)
			continue;
		for (uint32_t k = 0; k < x->nins; k++)
		{
			uint32_t	s = x->ins[k];

			if (s < g->hwm && g->alloc[s] && g->xmin[s] == x->xid &&
				!(g->xmax[s] != GC_INVALID_XID && xid_in(xids, nxids, g->xmax[s])))
				gc_bit_set(bm, s, 1);
		}
		for (uint32_t k = 0; k < x->ndel; k++)
		{
			uint32_t	s = x->del[k];

			if (s < g->hwm && g->alloc[s] && g->xmax[s] == x->xid)
				gc_bit_set(bm, s, 0);
		}
	}
	return 0;
}

int
gcache_build_table(const gcache *g, const scan_prog *prog, uint32_t *tab, char *err, size_t errlen)
{
	uint32_t	coloff = prog->words[0];
	uint32_t	ncols = prog->words[3];

	if (ncols > g->ncols)
	{
		gc_err(err, errlen, "the program reads more columns than the GPU-Cache has");
		return -1;
	}
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = prog->words[coloff + c];
		uint32_t	slot = (desc >> 16) & 0xff;
		uint32_t	vt = (desc >> 12) & 0xf;
		const gcache_col *col = &g->cols[c];
		uint32_t   *t = tab + 8 * c;

		t[0] = 0xffffffffU;
		if (slot == 0xff)
			continue;
		if (col->kind == GCK_NONE)
		{
			t[4] = 1;						/* a dropped column: NULL */
			continue;
		}
		t[0] = (uint32_t) col->nm_off;
		t[1] = (uint32_t) col->nm_len;
		if (vt == SOP_VT_NULLONLY)
			continue;
		switch (vt)
		{
			case SOP_VT_BOOL:
				if (col->kind != GCK_BOOL)
					goto badtype;
				t[2] = (uint32_t) col->v_off;
				t[3] = 0;
				break;
			case SOP_VT_INT:
				if (col->kind != GCK_FIXED || col->vt != GCVT_INT ||
					(col->w != 1 && col->w != 2 && col->w != 4 && col->w != 8))
					goto badtype;
				t[2] = (uint32_t) col->v_off;
				t[3] = col->w;
				break;
			case SOP_VT_FLOAT4:
				if (col->kind != GCK_FIXED || col->vt != GCVT_FLOAT4 || col->w != 4)
					goto badtype;
				t[2] = (uint32_t) col->v_off;
				t[3] = 4;
				break;
			case SOP_VT_FLOAT8:
				if (col->kind != GCK_FIXED || col->vt != GCVT_FLOAT8 || col->w != 8)
					goto badtype;
				t[2] = (uint32_t) col->v_off;
				t[3] = 8;
				break;
			case SOP_VT_STR:
			case SOP_VT_JSON:
			case SOP_VT_NUM:
				if (col->kind != GCK_VAR || col->vt != GCVT_STR)
					goto badtype;
				t[2] = (uint32_t) col->v_off;
				t[3] = 8;
				t[4] = GCACHE_ACM_PAIRS;
				t[5] = (uint32_t) col->x_off;
				t[6] = (uint32_t) col->x_used;
				break;
			default:
				goto badtype;
		}
		t[7] = (uint32_t) col->v_len;
		continue;
badtype:
		gc_err(err, errlen, "column %u: the expression reads a value that the GPU cannot load from a GPU-Cache "
			   "(numeric and other types are only passed through)", c + 1);
		return -1;
	}
	return 0;
}

int
gcache_fetch(const gcache *g, int colno, uint32_t row, unsigned char *tmp,
			 const unsigned char **p_ptr, uint32_t *p_len, char *err, size_t errlen)
{
	const gcache_col *col;
	const uint8_t *base = (const uint8_t *) g->bd.map;

	if (colno < 1 || (uint32_t) colno > g->ncols || row >= g->hwm)
	{
		gc_err(err, errlen, "column %d: no such column in the GPU-Cache", colno);
		return -1;
	}
	col = &g->cols[colno - 1];
	if (col->kind == GCK_NONE)
		return GCF_NULL;
	if (!((gc_col_nm(g, col)[row >> 3] >> (row & 7)) & 1))
		return GCF_NULL;
	if (col->kind == GCK_BOOL)
	{
		tmp[0] = (base[col->v_off + (row >> 3)] >> (row & 7)) & 1;
		*p_ptr = tmp;
		*p_len = 1;
		return GCF_FIXED;
	}
	if (col->kind == GCK_FIXED)
	{
		*p_ptr = base + col->v_off + (uint64_t) col->w * row;
		*p_len = col->w;
		return GCF_FIXED;
	}
	{
		uint64_t	pr;
		uint32_t	off, len;

		memcpy(&pr, base + col->v_off + 8 * (uint64_t) row, 8);
		off = (uint32_t) pr;
		len = (uint32_t)(pr >> 32);
		if (len == 0 || (uint64_t) off + len > col->x_used)
		{
			gc_err(err, errlen, "column %d: corrupted GPU-Cache (row %u)", colno, row);
			return -1;
		}
		*p_ptr = base + col->x_off + off;
		*p_len = len;
		return GCF_RAW_VARLENA;
	}
}

void
gcache_ctid(const gcache *g, uint32_t row, unsigned char *out)
{
	uint64_t	key = g->ctid[row];
	uint32_t	blkno = (uint32_t)(key >> 16);
	uint16_t	offnum = (uint16_t) key;
	uint16_t	hi = (uint16_t)(blkno >> 16), lo = (uint16_t) blkno;

	memcpy(out, &hi, 2);
	memcpy(out + 2, &lo, 2);
	memcpy(out + 4, &offnum, 2);
}

/* pg_attribute.attalign ('c', 's', 'i', 'd') in bytes */
static size_t
gc_typalign(uint16_t attalign, size_t dflt)
{
	switch (attalign)
	{
		case 'c':
			return 1;
		case 's':
			return 2;
		case 'i':
			return 4;
		case 'd':
			return 8;
		default:
			return dflt;
	}
}

/*
 * A12: the heap tuple of a cached row, for the backend's CPU re-evaluation of a row the GPU could not decide
 * (a KDS_FORMAT_RECHECK item of kind heap). The cache is not a snapshot (it holds the latest committed state and the
 * session's own changes), so the backend cannot fetch the version from the heap by ctid; the tuple is built from the
 * cache's columns instead: every column the cache holds (a column that is not cached is NULL), frozen xmin, the
 * slot's ctid. On success *p_buf is a malloc'ed buffer of *p_len bytes (the caller frees it) and *p_blkno / *p_offnum
 * the ctid; -1 with a message in err otherwise.
 */
int
gcache_form_tuple(const gcache *g, uint32_t row, unsigned char **p_buf, uint32_t *p_len,
				  uint32_t *p_blkno, uint32_t *p_offnum, char *err, size_t errlen)
{
	uint32_t	natts = g->ncols;
	size_t		hoff = MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) + (natts + 7) / 8);
	size_t		cap = hoff + 256, pos = hoff;
	unsigned char *buf = calloc(1, cap);
	HeapTupleHeaderData *th;
	uint16_t	infomask = HEAP_XMIN_FROZEN | HEAP_XMAX_INVALID;
	unsigned char ct[6];
	uint16_t	hi, lo, off;

	if (!buf)
	{
		gc_err(err, errlen, "out of memory");
		return -1;
	}
	for (uint32_t c = 0; c < natts; c++)
	{
		const gcache_col *col = &g->cols[c];
		unsigned char tmp[8];
		const unsigned char *ptr = NULL;
		uint32_t	len = 0;
		size_t		al = 1, npos;
		int			rv = gcache_fetch(g, (int)(c + 1), row, tmp, &ptr, &len, err, errlen);

		if (rv < 0)
		{
			free(buf);
			return -1;
		}
		if (rv == GCF_NULL)
		{
			infomask |= HEAP_HASNULL;
			continue;
		}
		buf[offsetof(HeapTupleHeaderData, t_bits) + (c >> 3)] |= (unsigned char)(1U << (c & 7));
		if (rv == GCF_RAW_VARLENA)
		{
			infomask |= HEAP_HASVARWIDTH;
			if ((ptr[0] & 1) == 0)				/* a 4-byte header is aligned (a short one is not) */
				al = gc_typalign(col->attalign, 4);
		}
		else
			al = gc_typalign(col->attalign, 1);
		npos = TYPEALIGN(al, pos);
		if (npos + len > cap)
		{
			unsigned char *nb;

			cap = (npos + len) * 2;
			nb = realloc(buf, cap);
			if (!nb)
			{
				free(buf);
				gc_err(err, errlen, "out of memory");
				return -1;
			}
			buf = nb;
		}
		memset(buf + pos, 0, npos - pos);
		memcpy(buf + npos, ptr, len);
		pos = npos + len;
	}
	th = (HeapTupleHeaderData *) buf;
	th->t_choice.t_heap.t_xmin = GC_FROZEN_XID;
	th->t_choice.t_heap.t_xmax = 0;
	gcache_ctid(g, row, ct);
	memcpy(&hi, ct, 2);
	memcpy(&lo, ct + 2, 2);
	memcpy(&off, ct + 4, 2);
	th->t_ctid.ip_blkid.bi_hi = hi;
	th->t_ctid.ip_blkid.bi_lo = lo;
	th->t_ctid.ip_posid = off;
	th->t_infomask2 = (uint16_t) natts;
	th->t_infomask = infomask;
	th->t_hoff = (uint8_t) hoff;
	*p_buf = buf;
	*p_len = (uint32_t) pos;
	*p_blkno = ((uint32_t) hi << 16) | lo;
	*p_offnum = off;
	return 0;
}
