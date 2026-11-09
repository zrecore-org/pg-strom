/*
 * scan_exec.c
 *
 * GpuScan task execution for the Vulkan service. The GPU decides which tuples qualify and
 * computes the projection expressions; the service copies the pass-through columns and
 * forms the result tuples (kern_tupitem, the layout of MinimalTuple plus a row id).
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "xpu_common.h"
#include "kernels_spv.h"
#define F64SOFT_C 1
#include "kernels_vm/f64_soft.inc"
#include "scan_exec.h"
#include "scan_caps.h"
#include "cpu_vm.h"
#include "gpusort.h"
#include "specialize.h"

#define SCAN_WG_MAX		65535U
#define AGG_PAGES_PER_WG	32U		/* pages one aggregation workgroup runs over */

/* PGSTROM_GPUSERV_GROUP_PPW overrides the pages per workgroup of the GROUP BY kernels (tuning experiments) */
static uint32_t
group_ppw(uint32_t dflt)
{
	static int	init = 0;
	static uint32_t v = 0;

	if (!init)
	{
		const char *s = getenv("PGSTROM_GPUSERV_GROUP_PPW");

		v = (s && atoi(s) > 0) ? (uint32_t) atoi(s) : 0;
		init = 1;
	}
	return v ? v : dflt;
}

typedef strom_io_chunk	io_chunk;
typedef strom_io_vector	io_vector;

static double
now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void
scan_engine_init(scan_engine *eng, gpu_vk_ctx *vk)
{
	memset(eng, 0, sizeof(*eng));
	eng->vk = vk;
	eng->no_map = (getenv("PGSTROM_GPUSERV_NO_MAP") != NULL);
	{
		const char *tiers = getenv("PGSTROM_GPUSERV_TIERS");

		scan_tier_full_only = (tiers && strcmp(tiers, "full") == 0);
	}
	{
		const char *tr = getenv("PGSTROM_GPUSERV_TRACE");

		eng->spec = spec_mgr_create(vk, tr && tr[0] && strcmp(tr, "0") != 0);
	}
}


/* a task's dispatch: the specialized kernel or the VM's (spec_dispatch, spec_verify.c) */
static int
kdispatch(scan_engine *eng, scan_prog *prog, int kind, int agg_max, int ga_c,
		  gpu_vk_pipeline *vm, gpu_vk_pipeline *sp, gpu_vk_buffer *const *bufs, const void *pc,
		  uint32_t groups)
{
	spec_shape	sh;

	memset(&sh, 0, sizeof(sh));
	sh.kind = kind;
	sh.agg_max = agg_max;
	sh.ga_c = ga_c;
	return spec_dispatch(eng, prog, kind, &sh, vm, sp, bufs, pc, groups);
}

/* ---------------------------------------------------------------- */
typedef struct
{
	const unsigned char *htup;			/* tuple header */
	uint32_t	natts;
	int			hasnull;
} src_tuple;

/* size of the varlena at p (the datum as stored, including its header) */
static uint32_t
varlena_size(const unsigned char *p)
{
	if (p[0] == 0x01)					/* external: on-disk pointer or in-memory reference */
		return (p[1] == 18) ? 18 : 10;
	if (p[0] & 0x01)					/* 1-byte header */
		return (p[0] >> 1) & 0x7f;
	{
		uint32_t	h;

		memcpy(&h, p, 4);
		return (h >> 2) & 0x3fffffffU;
	}
}

/*
 * Deform attributes 1..maxres of a heap tuple: isnull[i] and the address of each value.
 * (A value that is not NULL always has data; 'lens' is its size for varlena columns.)
 */
static void
deform_src(const kern_data_store *kds, const unsigned char *htup, int maxres,
		   uint8_t *isnull, const unsigned char **ptrs, uint32_t *lens)
{
	uint16_t	infomask, infomask2;
	uint32_t	off = htup[22];
	uint32_t	natts;
	int			hasnull;

	memcpy(&infomask2, htup + 18, 2);
	memcpy(&infomask, htup + 20, 2);
	natts = infomask2 & 0x7ff;
	hasnull = (infomask & 1) != 0;
	for (int i = 0; i < maxres; i++)
	{
		const kern_colmeta *cm = &kds->colmeta[i];

		if ((uint32_t) i >= natts ||
			(hasnull && !((htup[23 + (i >> 3)] >> (i & 7)) & 1)))
		{
			isnull[i] = 1;
			ptrs[i] = NULL;
			lens[i] = 0;
			continue;
		}
		isnull[i] = 0;
		if (cm->attlen == -1)
		{
			if (htup[off] == 0)
				off = (off + cm->attalign - 1) & ~(uint32_t)(cm->attalign - 1);
			ptrs[i] = htup + off;
			lens[i] = varlena_size(htup + off);
			off += lens[i];
		}
		else
		{
			off = (off + cm->attalign - 1) & ~(uint32_t)(cm->attalign - 1);
			ptrs[i] = htup + off;
			lens[i] = (cm->attlen > 0 ? (uint32_t) cm->attlen : 0);
			off += lens[i];
		}
	}
}

typedef struct
{
	const kern_data_store *kds_dst_head;
	const kern_data_store *kds_src;
	const scan_prog *prog;
	int			maxres;
	const kern_data_store *kds_in[SCAN_JOIN_MAXDEPTH + 1];	/* a join: the inner relations' stores (index 1 ...) */
	int			maxres_inner[SCAN_JOIN_MAXDEPTH + 1];
	int			ndepth;
} form_ctx;

/*
 * Form (or just measure, if titem is NULL) one result tuple. 'comp' points at the GPU's
 * results of this row: (isnull, lo, hi) per computed attribute. Returns t_len, or -1.
 */
static int
form_tuple(const form_ctx *fc, const unsigned char *htup, const unsigned char *const *htup_in,
		   const uint32_t *comp, uint32_t blkno, uint32_t posid,
		   kern_tupitem *titem, uint32_t rowid, char *err, size_t errlen)
{
	const scan_prog *prog = fc->prog;
	int			nattrs = prog->nattrs;
	int			nsrc = fc->maxres + 1;
	uint8_t		isnull_src[nsrc];
	const unsigned char *ptrs[nsrc];
	uint32_t	lens[nsrc];
	int			nin = 1;
	uint32_t	t_hoff;
	uint16_t	infomask = 0;
	int			hasnull = 0;

	if (fc->maxres > 1664)
	{
		snprintf(err, errlen, "too many columns");
		return -1;
	}
	if (fc->maxres > 0)
		deform_src(fc->kds_src, htup, fc->maxres, isnull_src, ptrs, lens);
	for (int d = 1; d <= fc->ndepth; d++)
		if (fc->maxres_inner[d] + 1 > nin)
			nin = fc->maxres_inner[d] + 1;
	uint8_t		isnull_in[SCAN_JOIN_MAXDEPTH + 1][nin];
	const unsigned char *ptrs_in[SCAN_JOIN_MAXDEPTH + 1][nin];
	uint32_t	lens_in[SCAN_JOIN_MAXDEPTH + 1][nin];

	for (int d = 1; d <= fc->ndepth; d++)
	{
		if (htup_in && htup_in[d] && fc->maxres_inner[d] > 0)
			deform_src(fc->kds_in[d], htup_in[d], fc->maxres_inner[d], isnull_in[d], ptrs_in[d], lens_in[d]);
		else
		{
			/* no inner tuple (an outer row without a match): all its columns are NULL */
			memset(isnull_in[d], 1, nin);
			memset(ptrs_in[d], 0, sizeof(ptrs_in[d]));
			memset(lens_in[d], 0, sizeof(lens_in[d]));
		}
	}

	for (int j = 0; j < nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];

		if (pa->kind == SCAN_PROJ_SYS)
			continue;
		if (pa->kind == SCAN_PROJ_SRC ? isnull_src[pa->resno - 1] :
			pa->kind == SCAN_PROJ_INNER ? isnull_in[pa->depth][pa->resno - 1] : ((comp[3 * pa->comp] & 1u) != 0))
		{
			hasnull = 1;
			infomask |= 1;				/* HEAP_HASNULL */
			break;
		}
	}
	t_hoff = offsetof(kern_tupitem, t_bits);
	if (hasnull)
		t_hoff += (nattrs + 7) / 8;
	t_hoff = MAXALIGN(t_hoff);
	if (titem)
	{
		memset(titem, 0, t_hoff);
		titem->t_infomask2 = (uint16_t)(nattrs & 0x7ff);
		titem->t_hoff = (uint8_t)(t_hoff + MINIMAL_TUPLE_OFFSET);
	}
	for (int j = 0; j < nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];
		const kern_colmeta *cm = &fc->kds_dst_head->colmeta[j];
		const unsigned char *src = NULL;
		uint32_t	sz = 0, t_next;
		unsigned char tmp[8];
		unsigned char numbuf[SCAN_NUMERIC_MAXLEN];

		if (pa->kind == SCAN_PROJ_SYS)
		{
			if (pa->resno == -1)				/* ctid: block number (hi, lo), then offset */
			{
				uint16_t	t[3] = {(uint16_t)(blkno >> 16), (uint16_t)(blkno & 0xffff), (uint16_t) posid};

				memcpy(tmp, t, 6);
				sz = 6;
			}
			else								/* tableoid */
			{
				uint32_t	oid = fc->kds_src->table_oid;

				memcpy(tmp, &oid, 4);
				sz = 4;
			}
			src = tmp;
			if (cm->attlen != (int) sz)
			{
				snprintf(err, errlen, "column %d: system column of %u bytes does not fit the result type (len %d)",
						 j + 1, sz, cm->attlen);
				return -1;
			}
		}
		else if (pa->kind == SCAN_PROJ_INNER)
		{
			if (isnull_in[pa->depth][pa->resno - 1])
				continue;
			src = ptrs_in[pa->depth][pa->resno - 1];
			sz = lens_in[pa->depth][pa->resno - 1];
			if (cm->attlen != fc->kds_in[pa->depth]->colmeta[pa->resno - 1].attlen)
			{
				snprintf(err, errlen, "column %d: result type length %d differs from the inner relation (%d)",
						 j + 1, cm->attlen, fc->kds_in[pa->depth]->colmeta[pa->resno - 1].attlen);
				return -1;
			}
		}
		else if (pa->kind == SCAN_PROJ_SRC)
		{
			if (isnull_src[pa->resno - 1])
				continue;
			src = ptrs[pa->resno - 1];
			sz = lens[pa->resno - 1];
			if (cm->attlen != fc->kds_src->colmeta[pa->resno - 1].attlen)
			{
				snprintf(err, errlen, "column %d: result type length %d differs from the source (%d)",
						 j + 1, cm->attlen, fc->kds_src->colmeta[pa->resno - 1].attlen);
				return -1;
			}
		}
		else
		{
			uint32_t	lo = comp[3 * pa->comp + 1];
			uint32_t	hi = comp[3 * pa->comp + 2];
			uint64_t	v = (uint64_t) lo | ((uint64_t) hi << 32);

			if ((comp[3 * pa->comp] & 1u) != 0)
				continue;				/* NULL (the scale of a numeric rides above bit 0) */
			if (pa->comp_numeric)
			{
				if (cm->attlen != -1)
				{
					snprintf(err, errlen, "column %d: computed numeric does not fit the result type (len %d)",
							 j + 1, cm->attlen);
					return -1;
				}
				sz = scan_vm_numeric_encode((int64_t) v, comp[3 * pa->comp] >> 8, numbuf);
				src = numbuf;
			}
			else if (cm->attlen != pa->comp_bytes || !cm->attbyval)
			{
				snprintf(err, errlen, "column %d: computed %d-byte value does not fit the result type (len %d)",
						 j + 1, pa->comp_bytes, cm->attlen);
				return -1;
			}
			if (!pa->comp_numeric)
			{
				memcpy(tmp, &v, 8);			/* little endian */
				src = tmp;
				sz = pa->comp_bytes;
			}
		}
		if (cm->attlen == -1)
		{
			infomask |= 2;				/* HEAP_HASVARWIDTH */
			if (src[0] == 0x01)
				infomask |= 4;			/* HEAP_HASEXTERNAL */
			/* a short (1-byte) header needs no alignment */
			t_next = (src[0] & 0x01) ? t_hoff : TYPEALIGN(cm->attalign, t_hoff);
		}
		else
			t_next = TYPEALIGN(cm->attalign, t_hoff);
		if (titem)
		{
			if (t_next > t_hoff)
				memset((char *) titem + t_hoff, 0, t_next - t_hoff);
			memcpy((char *) titem + t_next, src, sz);
			if (hasnull)
				titem->t_bits[j >> 3] |= (1 << (j & 7));
		}
		t_hoff = t_next + sz;
	}
	t_hoff += ROWID_SIZE;
	if (titem)
	{
		titem->t_len = t_hoff;
		titem->t_infomask = infomask;
		KERN_TUPITEM_SET_ROWID(titem, rowid);
	}
	return (int) t_hoff;
}

/* ---------------------------------------------------------------- */
/* the message of a kernel error code (FuncOpCode | reason << 16, see scan_vm_ops.inc) */
void
describe_kernel_error(uint32_t err, uint32_t page, uint32_t lp, char *buf, size_t buflen)
{
	if (err & 0x80000000U)
		snprintf(buf, buflen, "malformed GPU program (instruction %u)", err & 0x7fffffffU);
	else if (err == 0x7ffffffeU)
		snprintf(buf, buflen, "a compressed or external string reached the CPU re-evaluation of block index %u, line pointer %u", page, lp + 1);
	else if (err == 0x7ffffffdU)
		snprintf(buf, buflen, "GiST join: the probe point of block index %u, line pointer %u has a non-finite coordinate", page, lp + 1);
	else if (err == 0x7ffffffcU)
		snprintf(buf, buflen, "GiST join: the index walk of block index %u, line pointer %u exceeded pg_strom.gpugist_max_steps", page, lp + 1);
	else if (err == 0x7fffffffU)
		snprintf(buf, buflen, "unsupported attribute layout in block index %u, line pointer %u", page, lp + 1);
	else
	{
		uint32_t	func = err & 0xffffU;
		uint32_t	why = (err >> 16) & 0xffU;
		scan_func_class fc = scan_vm_classify_func(func);
		const char *what = "evaluation error";

		switch (why)
		{
			case SOP_ERR_ZERODIV:
				what = "division by zero";
				break;
			case SOP_ERR_INTRANGE:
				what = (fc.width == 8 ? "tinyint out of range" :
						fc.width == 16 ? "smallint out of range" :
						fc.width == 32 ? "integer out of range" : "bigint out of range");
				break;
			case SOP_ERR_FLTOVER:
				what = "value out of range: overflow";
				break;
			case SOP_ERR_FLTUNDER:
				what = "value out of range: underflow";
				break;
			case SOP_ERR_DATERANGE:
				what = "date out of range";
				break;
			case SOP_ERR_TSRANGE:
				what = "date out of range for timestamp";
				break;
			case SOP_ERR_INFSUB:
				what = "cannot subtract infinite dates";
				break;
			case SOP_ERR_SQRTNEG:
				what = "cannot take square root of a negative number";
				break;
		}
		snprintf(buf, buflen, "%s (in %s, block index %u, line pointer %u)",
				 what, scan_vm_opcode_name(func), page, lp + 1);
	}
}

static int
cmp_u64(const void *a, const void *b)
{
	uint64_t	x = *(const uint64_t *) a, y = *(const uint64_t *) b;

	return (x < y ? -1 : x > y ? 1 : 0);
}

/* ---------------------------------------------------------------- */
/* rows for the backend to evaluate (KDS_FORMAT_RECHECK) */
static int
rck_reserve(recheck_buf *r, size_t isz)
{
	if (r->len + isz > r->cap)
	{
		size_t		ncap = (r->cap ? r->cap * 2 : 4096);
		unsigned char *nb;

		while (ncap < r->len + isz)
			ncap *= 2;
		nb = realloc(r->buf, ncap);
		if (!nb)
			return -1;
		r->buf = nb;
		r->cap = ncap;
	}
	return 0;
}

/* an item of kind RECHECK_KIND_ARROW (16 bytes, no tuple): row `row` of record batch `rb_flat` of scan relation `relidx` */
int
rck_add_arrow(recheck_buf *r, uint32_t rb_flat, uint32_t relidx, uint32_t row)
{
	kern_recheckitem *it;

	if (relidx > 0xffffU || rck_reserve(r, offsetof(kern_recheckitem, t_data)) != 0)
		return -1;
	it = (kern_recheckitem *)(r->buf + r->len);
	memset(it, 0, offsetof(kern_recheckitem, t_data));
	it->blkno = rb_flat;
	it->offnum = (uint16_t) relidx;
	it->kind = RECHECK_KIND_ARROW;
	it->arrow_row = row;
	r->len += offsetof(kern_recheckitem, t_data);
	r->nitems++;
	return 0;
}

int
rck_add(recheck_buf *r, uint32_t blkno, uint32_t offnum, const unsigned char *htup, uint32_t len)
{
	size_t		isz = MAXALIGN(offsetof(kern_recheckitem, t_data) + len);
	kern_recheckitem *it;

	if (rck_reserve(r, isz) != 0)
		return -1;
	it = (kern_recheckitem *)(r->buf + r->len);
	memset(it, 0, offsetof(kern_recheckitem, t_data));
	it->t_len = len;
	it->blkno = blkno;
	it->offnum = (uint16_t) offnum;
	memcpy(it->t_data, htup, len);
	memset(it->t_data + len, 0, isz - offsetof(kern_recheckitem, t_data) - len);
	r->len += isz;
	r->nitems++;
	return 0;
}

/* the size of the chunk of the rows (0 if none), and filling it in at 'kds' */
size_t
rck_chunk_size(const kern_data_store *kds_head, const recheck_buf *r)
{
	return r->nitems == 0 ? 0 : KDS_HEAD_LENGTH(kds_head) + r->len;
}
void
rck_chunk_fill(kern_data_store *kds, const kern_data_store *kds_head, const recheck_buf *r)
{
	size_t		head_sz = KDS_HEAD_LENGTH(kds_head);

	memcpy(kds, kds_head, head_sz);
	kds->length = head_sz + r->len;
	kds->usage = r->len;
	kds->nitems = r->nitems;
	kds->format = KDS_FORMAT_RECHECK;
	kds->hash_nslots = 0;
	memcpy((char *) kds + head_sz, r->buf, r->len);
}

/* ---------------------------------------------------------------- */
/*
 * The pages of one task come from two places:
 *   - the pages the backend loaded and visibility-checked (copied into a small buffer), and
 *   - the pages of the relation file, which are not copied at all: a window of the file is
 *     mapped and imported into Vulkan, and kept for the next tasks (importing is what costs
 *     time, not reading). If a window cannot be used (pages past the end of the file, a span
 *     that does not fit, no host import) every page is copied into one buffer instead.
 * ptab[] tells the kernel where each page is; bit 31 selects the loaded-pages buffer.
 */
#define PTE_LOADED			0x80000000U
#define MAP_WINDOW			((int64_t) 512 << 20)
#define MAP_WINDOW_STEP		((int64_t) 256 << 20)

typedef struct
{
	gpu_vk_buffer	bufF;			/* the file window (cache-owned) or the copied pages */
	int				ownF;			/* bufF is ours to destroy */
	gpu_vk_buffer	bufL;			/* the loaded pages (or a placeholder) */
	uint32_t	   *ptab;
	unsigned char  *baseF, *baseL;	/* the buffers as the CPU sees them */
	int				mapped;
} page_set;

#define PAGE_AT(ps, i)											\
	(((ps)->ptab[i] & PTE_LOADED)								\
	 ? (ps)->baseL + ((ps)->ptab[i] & ~PTE_LOADED)				\
	 : (ps)->baseF + (ps)->ptab[i])

static void
page_set_release(gpu_vk_ctx *vk, page_set *ps)
{
	if (ps->ownF)
		gpu_vk_buffer_destroy(vk, &ps->bufF);
	if (ps->bufL.buf)
		gpu_vk_buffer_destroy(vk, &ps->bufL);
	free(ps->ptab);
	memset(ps, 0, sizeof(*ps));
}

static void
map_entry_drop(scan_engine *eng, scan_map_entry *me)
{
	if (me->map)
	{
		gpu_vk_buffer_destroy(eng->vk, &me->buf);
		munmap(me->map, me->maplen);
	}
	memset(me, 0, sizeof(*me));
}

void
scan_engine_fini(scan_engine *eng)
{
	for (int i = 0; i < SCAN_MAP_CACHE; i++)
		map_entry_drop(eng, &eng->maps[i]);
	arrow_engine_fini(eng);
	gpusort_engine_fini(eng);
	spec_mgr_destroy(eng->spec);
	eng->spec = NULL;
	SCAN_PIPES_DESTROY(eng, pipe, pipe_ready);
	SCAN_PIPES_DESTROY(eng, join_pipe, join_pipe_ready);
	SCAN_PIPES_DESTROY(eng, agg_pipe, agg_pipe_ready);
	SCAN_PIPES_DESTROY(eng, gagg_pipe, gagg_pipe_ready);
	SCAN_PIPES_DESTROY(eng, gtab_pipe, gtab_pipe_ready);
	SCAN_PIPES_DESTROY(eng, agtab_pipe, agtab_pipe_ready);
	SCAN_PIPES_DESTROY(eng, jagg_pipe, jagg_pipe_ready);
	SCAN_PIPES_DESTROY(eng, jgagg_pipe, jgagg_pipe_ready);
}

void
scan_pipes_destroy(gpu_vk_ctx *vk, gpu_vk_pipeline *pipes, int *ready, size_t npipes)
{
	for (size_t i = 0; i < npipes; i++)
	{
		if (ready[i])
			gpu_vk_pipeline_destroy(vk, &pipes[i]);
		ready[i] = 0;
	}
}

/* a mapped and imported window of the file that covers [lo, hi) */
static scan_map_entry *
map_window(scan_engine *eng, const char *pathname, int64_t lo, int64_t hi)
{
	gpu_vk_ctx *vk = eng->vk;
	struct stat	st;
	int			fd;
	int64_t		ws = (lo / MAP_WINDOW_STEP) * MAP_WINDOW_STEP;
	int64_t		wlen;
	size_t		align = vk->host_import_alignment;
	scan_map_entry *me = NULL, *victim = NULL;
	void	   *map;

	if (align == 0 || hi > ws + MAP_WINDOW)
		return NULL;
	if ((size_t) sysconf(_SC_PAGESIZE) > align)
		align = sysconf(_SC_PAGESIZE);
	fd = open(pathname, O_RDONLY);
	if (fd < 0)
		return NULL;
	if (fstat(fd, &st) != 0 || hi > st.st_size)		/* blocks that exist only in shared_buffers so far */
	{
		close(fd);
		return NULL;
	}
	eng->map_clock++;
	for (int i = 0; i < SCAN_MAP_CACHE; i++)
	{
		scan_map_entry *e = &eng->maps[i];

		if (e->map && e->dev == (uint64_t) st.st_dev && e->ino == (uint64_t) st.st_ino &&
			e->win_start == ws)
		{
			if (e->file_size == (int64_t) st.st_size)
			{
				e->last_used = eng->map_clock;
				close(fd);
				return e;
			}
			map_entry_drop(eng, e);			/* the file changed size: map it again */
			break;
		}
	}
	for (int i = 0; i < SCAN_MAP_CACHE; i++)
	{
		scan_map_entry *e = &eng->maps[i];

		if (!e->map)
		{
			me = e;
			break;
		}
		if (!victim || e->last_used < victim->last_used)
			victim = e;
	}
	if (!me)
	{
		map_entry_drop(eng, victim);
		me = victim;
	}
	wlen = st.st_size - ws;
	if (wlen > MAP_WINDOW)
		wlen = MAP_WINDOW;
	wlen = (wlen + (int64_t) align - 1) & ~(int64_t)(align - 1);
	if ((uint64_t) wlen > vk->limits.maxStorageBufferRange ||
		(vk->max_buffer_size != 0 && (uint64_t) wlen > vk->max_buffer_size))
	{
		close(fd);
		return NULL;
	}
	map = mmap(NULL, wlen, PROT_READ, MAP_SHARED, fd, ws);
	close(fd);
	if (map == MAP_FAILED)
		return NULL;
	if (gpu_vk_buffer_import_host(vk, &me->buf, map, wlen) != 0)
	{
		munmap(map, wlen);
		return NULL;
	}
	me->map = map;
	me->maplen = wlen;
	me->dev = st.st_dev;
	me->ino = st.st_ino;
	me->win_start = ws;
	me->file_size = st.st_size;
	me->last_used = eng->map_clock;
	return me;
}

/* 1: the file pages come from a mapped window, 0: not possible here, -1: error */
static int
page_set_map(scan_engine *eng, page_set *ps, const kern_data_store *kds_src,
			 const char *pathname, const io_vector *iovec)
{
	uint32_t	npages = kds_src->nitems;
	uint32_t	nloaded = kds_src->block_nloaded;
	int64_t	   *poff;
	int64_t		lo = INT64_MAX, hi = 0;
	scan_map_entry *me;

	poff = malloc(sizeof(int64_t) * (npages ? npages : 1));
	if (!poff)
		return -1;
	for (uint32_t i = nloaded; i < npages; i++)
		poff[i] = -1;
	for (unsigned int c = 0; c < iovec->nr_chunks; c++)
	{
		const io_chunk *ioc = &iovec->ioc[c];

		for (unsigned int k = 0; k < ioc->nr_pages / PAGES_PER_BLOCK; k++)
		{
			uint64_t	j = nloaded + ioc->m_offset / BLCKSZ + k;
			int64_t		off = (int64_t) ioc->fchunk_id * PAGE_SIZE + (int64_t) k * BLCKSZ;

			if (j >= npages || (ioc->m_offset % BLCKSZ) != 0)
				goto notpossible;
			poff[j] = off;
			if (off < lo)
				lo = off;
			if (off + BLCKSZ > hi)
				hi = off + BLCKSZ;
		}
	}
	for (uint32_t j = nloaded; j < npages; j++)
		if (poff[j] < 0)
			goto notpossible;			/* a page nobody provides: the copy path zero-fills it */
	if (lo > hi)
		goto notpossible;
	me = map_window(eng, pathname, lo, hi);
	if (!me)
		goto notpossible;
	ps->bufF = me->buf;
	ps->ownF = 0;
	ps->baseF = (unsigned char *) me->map;
	ps->ptab = malloc(sizeof(uint32_t) * (npages ? npages : 1));
	if (!ps->ptab)
	{
		free(poff);
		return -1;
	}
	for (uint32_t i = 0; i < nloaded; i++)
		ps->ptab[i] = i * BLCKSZ | PTE_LOADED;
	for (uint32_t j = nloaded; j < npages; j++)
		ps->ptab[j] = (uint32_t)(poff[j] - me->win_start);
	free(poff);
	return 1;

notpossible:
	free(poff);
	return 0;
}

/* the copy path: one buffer, loaded pages first, then the file's pages read in */
static int
page_set_copy(gpu_vk_ctx *vk, page_set *ps, const kern_data_store *kds_src,
			  const char *pathname, const io_vector *iovec, char *err, size_t errlen)
{
	uint32_t	npages = kds_src->nitems;
	uint32_t	nloaded = kds_src->block_nloaded;
	size_t		pages_sz = (npages ? (size_t) npages : 1) * BLCKSZ;

	if (gpu_vk_buffer_create(vk, &ps->bufF, pages_sz, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate %zu bytes of device memory: %s", pages_sz, vk->err);
		return -1;
	}
	ps->ownF = 1;
	ps->baseF = (unsigned char *) ps->bufF.map;
	ps->ptab = malloc(sizeof(uint32_t) * (npages ? npages : 1));
	if (!ps->ptab)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (uint32_t i = 0; i < npages; i++)
		ps->ptab[i] = i * BLCKSZ;
	if (nloaded > 0)
		memcpy(ps->baseF, (const char *) kds_src + kds_src->block_offset, (size_t) nloaded * BLCKSZ);
	if (nloaded < npages)
	{
		unsigned char *dst0 = ps->baseF + (size_t) nloaded * BLCKSZ;
		size_t		dst_len = (size_t)(npages - nloaded) * BLCKSZ;
		int			fd;
		struct stat	st;

		memset(dst0, 0, dst_len);
		if (!pathname || !iovec)
		{
			snprintf(err, errlen, "pages are missing and no file to read them from");
			return -1;
		}
		fd = open(pathname, O_RDONLY);
		if (fd < 0 || fstat(fd, &st) != 0)
		{
			snprintf(err, errlen, "cannot open '%s': %s", pathname, strerror(errno));
			if (fd >= 0)
				close(fd);
			return -1;
		}
		for (unsigned int i = 0; i < iovec->nr_chunks; i++)
		{
			const io_chunk *ioc = &iovec->ioc[i];
			size_t		remained = (size_t) ioc->nr_pages * PAGE_SIZE;
			off_t		file_pos = (off_t) ioc->fchunk_id * PAGE_SIZE;
			unsigned char *dest = dst0 + ioc->m_offset;

			if (ioc->m_offset + remained > dst_len)
			{
				snprintf(err, errlen, "iovec chunk %u exceeds the buffer", i);
				close(fd);
				return -1;
			}
			if (file_pos >= st.st_size)
				continue;
			if (file_pos + (off_t) remained > st.st_size)
				remained = st.st_size - file_pos;
			while (remained > 0)
			{
				ssize_t		n = pread(fd, dest, remained, file_pos);

				if (n < 0 && errno == EINTR)
					continue;
				if (n <= 0)
				{
					snprintf(err, errlen, "failed to read '%s': %s", pathname,
							 n < 0 ? strerror(errno) : "unexpected end of file");
					close(fd);
					return -1;
				}
				dest += n;
				file_pos += n;
				remained -= n;
			}
		}
		close(fd);
	}
	return 0;
}

/* the pages of a task: from a mapped file window when possible, else copied (see page_set) */
static int
task_pages_open(scan_engine *eng, page_set *ps, const kern_data_store *kds_src,
				const char *pathname, const io_vector *iovec, int *p_mapped,
				char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	uint32_t	npages = kds_src->nitems;
	uint32_t	nloaded = kds_src->block_nloaded;

	(void) npages;
	*p_mapped = 0;
	if (nloaded < npages && pathname && iovec && !eng->no_map)
	{
		int		rv = page_set_map(eng, ps, kds_src, pathname, iovec);

		if (rv < 0)
		{
			snprintf(err, errlen, "out of memory");
			return -1;
		}
		*p_mapped = rv;
	}
	if (*p_mapped)
	{
		/* the loaded pages: a small buffer next to the file window */
		size_t		l_sz = (nloaded ? (size_t) nloaded : 1) * BLCKSZ;

		if (gpu_vk_buffer_create(vk, &ps->bufL, l_sz, 0) != 0)
		{
			snprintf(err, errlen, "cannot allocate the page buffer: %s", vk->err);
			return -1;
		}
		ps->baseL = (unsigned char *) ps->bufL.map;
		if (nloaded > 0)
			memcpy(ps->baseL, (const char *) kds_src + kds_src->block_offset, (size_t) nloaded * BLCKSZ);
	}
	else
	{
		if (page_set_copy(vk, ps, kds_src, pathname, iovec, err, errlen) != 0)
			return -1;
		if (gpu_vk_buffer_create(vk, &ps->bufL, 16, 0) != 0)		/* placeholder binding */
		{
			snprintf(err, errlen, "cannot allocate the page buffer: %s", vk->err);
			return -1;
		}
		ps->baseL = (unsigned char *) ps->bufL.map;
	}
	return 0;
}

XpuCommand *
scan_exec_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
			   const XpuCommand *task, size_t *p_len, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped;
	gpu_vk_pipeline *sp;
	size_t		btab_sz;
	uint32_t	npages, nloaded, total_lps = 0, ncomp, rec_words;
	size_t		out_words;
	uint32_t   *out;
	XpuCommand *resp = NULL;
	uint32_t   *lp_base = NULL;
	const uint32_t *flags, *vals;
	uint32_t	nmatch = 0, nrck = 0;
	recheck_buf	rck = {0};
	form_ctx	fc;
	int			variant;
	double		t0 = now_ms(), t1 = 0, t2 = 0, t3 = 0, t4 = 0;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return NULL;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format == KDS_FORMAT_ARROW || task->tag == XpuCommandTag__XpuTaskExecGpuCache)
		return scan_exec_arrow_task(eng, session_cmd, prog, task, p_len, err, errlen);
	if (kds_src->format != KDS_FORMAT_BLOCK)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuScan yet",
				 kds_src->format);
		return NULL;
	}
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	npages = kds_src->nitems;
	nloaded = kds_src->block_nloaded;
	if (nloaded > npages ||
		kds_src->block_offset + (size_t) nloaded * BLCKSZ > task->length)
	{
		snprintf(err, errlen, "corrupted block data store");
		return NULL;
	}

	if (!prog->words)
	{
		if (scan_prog_build(session, kds_src, prog, err, errlen) != 0)
			return NULL;
	}
	ncomp = prog->ncomp;
	rec_words = 2 + 3 * ncomp;

	/* variant: 0-2 the registers (8/16/32), + 3 * tier (scan_caps.h: full, lite, nogeom) */
	variant = (prog->nregs <= 8 ? 0 : prog->nregs <= 16 ? 1 : 2) + 3 * scan_feat_tier(prog->feat);
	/* the VM kernel is only created when it is going to run (or to be compared with) */
	sp = spec_pipe(eng, prog, SPEC_SCAN, 0, 0, NULL, NULL, &eng->pipe[variant]);
	if ((sp == &eng->pipe[variant] || spec_check_enabled(eng)) && !eng->pipe_ready[variant])
	{
		static const struct { const unsigned char *spv; const unsigned int *len; } stab[] = {
			{kspv_gpuscan8, &kspv_gpuscan8_len}, {kspv_gpuscan16, &kspv_gpuscan16_len}, {kspv_gpuscan32, &kspv_gpuscan32_len},
			{kspv_gpuscan8_lite, &kspv_gpuscan8_lite_len}, {kspv_gpuscan16_lite, &kspv_gpuscan16_lite_len},
			{kspv_gpuscan32_lite, &kspv_gpuscan32_lite_len},
			{kspv_gpuscan8_nogeom, &kspv_gpuscan8_nogeom_len}, {kspv_gpuscan16_nogeom, &kspv_gpuscan16_nogeom_len},
			{kspv_gpuscan32_nogeom, &kspv_gpuscan32_nogeom_len},
		};
		_Static_assert(sizeof(stab) / sizeof(stab[0]) == SCAN_NV3, "stab: one entry per register variant and tier");
		const unsigned char *spv = stab[variant].spv;
		size_t		spv_len = *stab[variant].len;

		if (gpu_vk_pipeline_create(vk, &eng->pipe[variant], spv, spv_len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the scan pipeline: %s", vk->err);
			return NULL;
		}
		eng->pipe_ready[variant] = 1;
	}

	/* 1. the pages */
	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto error;
	t1 = now_ms();
	/* line pointers per page, as the kernel counts them: the base of each page's flags */
	lp_base = malloc(sizeof(uint32_t) * (npages ? npages : 1));
	if (!lp_base)
	{
		snprintf(err, errlen, "out of memory");
		goto error;
	}
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		lp_base[i] = total_lps;
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
	}

	/* 2. the page table, the output and the program */
	btab_sz = sizeof(uint32_t) * 2 * (npages ? npages : 1);
	if (gpu_vk_buffer_create(vk, &btab, btab_sz, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto error;
	}
	have_tab = 1;
	for (uint32_t i = 0; i < npages; i++)
	{
		((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
		((uint32_t *) btab.map)[2 * i + 1] = lp_base[i];
	}
	out_words = 4 + (size_t) total_lps * (1 + 3 * (size_t) ncomp);
	if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
		goto error;
	}
	have_out = 1;
	out = (uint32_t *) bout.map;
	memset(out, 0, 16);
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto error;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);

	t2 = now_ms();
	/* 3. run */
	{
		gpu_vk_buffer *bufs[5] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL};

		for (uint32_t base = 0; base < npages; base += SCAN_WG_MAX)
		{
			uint32_t	groups = npages - base;
			uint32_t	pc[4];

			if (groups > SCAN_WG_MAX)
				groups = SCAN_WG_MAX;
			pc[0] = npages;
			pc[1] = base;
			pc[2] = BLCKSZ;
			pc[3] = total_lps;
			if (kdispatch(eng, prog, SPEC_SCAN, 0, 0, &eng->pipe[variant], sp, bufs, pc, groups) != 0)
			{
				snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
				goto error;
			}
		}
	}
	t3 = now_ms();
	if (out[1] != 0)
	{
		describe_kernel_error(out[1], out[2], out[3], err, errlen);
		goto error;
	}
	flags = out + 4;
	vals = out + 4 + total_lps;

	/* 4. the result chunk, in heap order, in one pass; tuples are stored forward and the
	 * row index is fixed up once the final length is known */
	memset(&fc, 0, sizeof(fc));
	fc.kds_dst_head = kds_dst_head;
	fc.kds_src = kds_src;
	fc.prog = prog;
	fc.maxres = prog->max_src_resno;
	{
		size_t		head_sz = KDS_HEAD_LENGTH(kds_dst_head);
		size_t		resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
		size_t		idx_sz, cap, usage = 0, chunk_len;
		uint64_t   *rowindex;
		char	   *tup_area;
		kern_data_store *kds;

		for (uint32_t i = 0; i < total_lps; i++)
		{
			nmatch += (flags[i] == 1);
			nrck += (flags[i] == 2);
		}
		idx_sz = MAXALIGN(sizeof(uint64_t) * nmatch);
		/* a first guess of the tuples' size; grown on demand */
		cap = resp_sz + head_sz + idx_sz + (size_t) nmatch * 64 + 4096;
		resp = calloc(1, cap);
		if (!resp)
		{
			snprintf(err, errlen, "out of memory");
			goto error;
		}
		for (uint32_t i = 0, row = 0, nrck_seen = 0; i < npages && (row < nmatch || nrck_seen < nrck); i++)
		{
			const unsigned char *page = PAGE_AT(&ps, i);
			uint32_t	nlp = ((i + 1 < npages ? lp_base[i + 1] : total_lps) - lp_base[i]);
			BlockNumber	blkno = ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[i];

			for (uint32_t lp = 0; lp < nlp; lp++)
			{
				uint32_t	g = lp_base[i] + lp;
				uint32_t	lpd;
				int			sz;
				char	   *dest;

				if (flags[g] == 2)		/* a string the GPU cannot read: the backend evaluates the row */
				{
					memcpy(&lpd, page + 24 + (size_t) lp * 4, 4);
					if (rck_add(&rck, blkno, lp + 1, page + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
					{
						snprintf(err, errlen, "out of memory");
						goto error;
					}
					nrck_seen++;
					continue;
				}
				if (flags[g] != 1)
					continue;
				memcpy(&lpd, page + 24 + (size_t) lp * 4, 4);
				/* the tuple needs at most its source's length plus the header, the row id and padding */
				if (resp_sz + head_sz + idx_sz + usage + (lpd >> 17) + 256 + 16 * (size_t) prog->nattrs > cap)
				{
					size_t		ncap = cap * 2 + (lpd >> 17) + 4096;
					XpuCommand *nr = realloc(resp, ncap);

					if (!nr)
					{
						snprintf(err, errlen, "out of memory");
						goto error;
					}
					resp = nr;
					cap = ncap;
				}
				tup_area = (char *) resp + resp_sz + head_sz + idx_sz;
				dest = tup_area + usage;
				sz = form_tuple(&fc, page + (lpd & 0x7fff), NULL, vals + (size_t) g * 3 * ncomp,
								blkno, lp + 1, (kern_tupitem *) dest, row, err, errlen);
				if (sz < 0)
					goto error;
				/* forward position for now (see below) */
				((uint64_t *)((char *) resp + resp_sz + head_sz))[row] = usage;
				usage += MAXALIGN(sz);
				row++;
			}
		}
		chunk_len = head_sz + idx_sz + usage;
		kds = (kern_data_store *)((char *) resp + resp_sz);
		memcpy(kds, kds_dst_head, head_sz);
		kds->length = chunk_len;
		kds->usage = usage;
		kds->nitems = nmatch;
		kds->format = KDS_FORMAT_ROW;
		kds->hash_nslots = 0;
		/* row index: the distance of a tuple's start from the end of the chunk */
		rowindex = (uint64_t *)((char *) kds + head_sz);
		for (uint32_t r = 0; r < nmatch; r++)
			rowindex[r] = usage - rowindex[r];
		resp->magic = XpuCommandMagicNumber;
		resp->tag = XpuCommandTag__Success;
		resp->length = resp_sz + chunk_len;
		resp->u.results.scan_relidx = task->u.task.scan_relidx;
		resp->u.results.chunks_offset = resp_sz;
		resp->u.results.chunks_nitems = 1;
		resp->u.results.nitems_raw = total_lps;
		resp->u.results.nitems_in = nmatch;
		resp->u.results.nitems_out = nmatch;
		resp->u.results.npages_vfs_read = (npages - nloaded) * PAGES_PER_BLOCK;
		if (rck.nitems > 0)
		{
			/* a second chunk: the rows the backend has to evaluate */
			size_t		csz = rck_chunk_size(kds_dst_head, &rck);
			XpuCommand *nr = realloc(resp, resp->length + csz);

			if (!nr)
			{
				snprintf(err, errlen, "out of memory");
				goto error;
			}
			resp = nr;
			rck_chunk_fill((kern_data_store *)((char *) resp + resp->length), kds_dst_head, &rck);
			resp->length += csz;
			resp->u.results.chunks_nitems = 2;
		}
		*p_len = resp->length;
	}
	t4 = now_ms();
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: scan task: %u pages (%u from file) pages-in %.1f ms%s, setup %.1f ms, gpu %.1f ms, results %.1f ms (%u rows)\n",
				npages, npages - nloaded, t1 - t0, mapped ? " (mapped)" : "", t2 - t1, t3 - t2, t4 - t3, nmatch);
	free(lp_base);
	free(rck.buf);
	gpu_vk_buffer_destroy(vk, &bprog);
	gpu_vk_buffer_destroy(vk, &bout);
	gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	return resp;

error:
	free(lp_base);
	free(rck.buf);
	free(resp);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (err[0] == '\0')
		snprintf(err, errlen, "scan failed");
	return NULL;
}

/* ----------------------------------------------------------------
 * GpuPreAgg without GROUP BY
 * ----------------------------------------------------------------
 */
static int
ensure_agg_pipeline(scan_engine *eng, int variant, int avariant, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } tab[][3] = {
		{{kspv_gpuagg8_4, &kspv_gpuagg8_4_len}, {kspv_gpuagg8_12, &kspv_gpuagg8_12_len},
		 {kspv_gpuagg8_12, &kspv_gpuagg8_12_len}},
		{{kspv_gpuagg16_4, &kspv_gpuagg16_4_len}, {kspv_gpuagg16_12, &kspv_gpuagg16_12_len},
		 {kspv_gpuagg16_12, &kspv_gpuagg16_12_len}},
		{{kspv_gpuagg32_4, &kspv_gpuagg32_4_len}, {kspv_gpuagg32_12, &kspv_gpuagg32_12_len},
		 {kspv_gpuagg32_12, &kspv_gpuagg32_12_len}},
		{{kspv_gpuagg8_4_lite, &kspv_gpuagg8_4_lite_len}, {kspv_gpuagg8_12_lite, &kspv_gpuagg8_12_lite_len},
		 {kspv_gpuagg8_12_lite, &kspv_gpuagg8_12_lite_len}},
		{{kspv_gpuagg16_4_lite, &kspv_gpuagg16_4_lite_len}, {kspv_gpuagg16_12_lite, &kspv_gpuagg16_12_lite_len},
		 {kspv_gpuagg16_12_lite, &kspv_gpuagg16_12_lite_len}},
		{{kspv_gpuagg32_4_lite, &kspv_gpuagg32_4_lite_len}, {kspv_gpuagg32_12_lite, &kspv_gpuagg32_12_lite_len},
		 {kspv_gpuagg32_12_lite, &kspv_gpuagg32_12_lite_len}},
		{{kspv_gpuagg8_4_nogeom, &kspv_gpuagg8_4_nogeom_len}, {kspv_gpuagg8_12_nogeom, &kspv_gpuagg8_12_nogeom_len},
		 {kspv_gpuagg8_12_nogeom, &kspv_gpuagg8_12_nogeom_len}},
		{{kspv_gpuagg16_4_nogeom, &kspv_gpuagg16_4_nogeom_len}, {kspv_gpuagg16_12_nogeom, &kspv_gpuagg16_12_nogeom_len},
		 {kspv_gpuagg16_12_nogeom, &kspv_gpuagg16_12_nogeom_len}},
		{{kspv_gpuagg32_4_nogeom, &kspv_gpuagg32_4_nogeom_len}, {kspv_gpuagg32_12_nogeom, &kspv_gpuagg32_12_nogeom_len},
		 {kspv_gpuagg32_12_nogeom, &kspv_gpuagg32_12_nogeom_len}},
	};
	_Static_assert(sizeof(tab) / sizeof(tab[0]) == SCAN_NV3, "tab: one entry per register variant and tier");

	if (!eng->agg_pipe_ready[variant][avariant])
	{
		if (gpu_vk_pipeline_create(vk, &eng->agg_pipe[variant][avariant], tab[variant][avariant].spv,
								   *tab[variant][avariant].len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the aggregation pipeline: %s", vk->err);
			return -1;
		}
		eng->agg_pipe_ready[variant][avariant] = 1;
	}
	return 0;
}

/*
 * float8 aggregates (A10): classes 6 (sum, avg), 7 (min), 8 (max), 9 (sum of float4) of scan_common.slang's fagg_merge(), on the
 * host: add the partial (n2, a2) (the bits of a double) into (n, val). A sum of finite values that overflows sets *ovf (PostgreSQL
 * raises "value out of range: overflow" for it).
 */
static void
fagg_host_merge(uint32_t cls, uint64_t *n, int64_t *val, uint8_t *ovf, uint64_t n2, int64_t a2)
{
	if (n2 == 0)
		return;
	if (*n == 0)
	{
		*n = n2;
		*val = a2;
		return;
	}
	if (cls == 6 || cls == 9)
	{
		uint64_t	r = (cls == 9) ? f64_addf4((uint64_t) *val, (uint64_t) a2) : f64_add((uint64_t) *val, (uint64_t) a2);

		if (f64_arith_err(0, (uint64_t) *val, (uint64_t) a2, r) != 0)
			*ovf = 1;
		*val = (int64_t) r;
		*n += n2;
		return;
	}
	{
		uint64_t	kn = f64_key((uint64_t) a2), ko = f64_key((uint64_t) *val);

		if (cls == 7 ? kn < ko : kn > ko)
			*val = a2;
		*n = 1;
	}
}

/* the class of a float8 aggregate action (0: not one) */
static uint32_t
fagg_class_of_action(uint32_t action)
{
	switch (action)
	{
		case KAGG_ACTION__PSUM_FP:
		case KAGG_ACTION__PAVG_FP:		return 6;
		case KAGG_ACTION__PMIN_FP64:	return 7;
		case KAGG_ACTION__PMAX_FP64:	return 8;
		case KAGG_ACTION__PSUM_FP32:	return 9;
		default:						return 0;
	}
}

/*
 * Add the per-workgroup records of an aggregation kernel without GROUP BY (agg.slang, jagg.slang) to
 * the session's aggregates: 6 words per aggregate and workgroup, from word 4 of the output.
 */
static void
agg_merge_workgroups(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total)
{
	uint32_t	nagg = prog->nagg;

	for (uint32_t pg = 0; pg < ngroups_total; pg++)
	{
		st->nitems_in += out[4 + (size_t) pg * nagg * 6 + 1];
		for (uint32_t k = 0; k < nagg; k++)
		{
			const uint32_t *rec = &out[4 + ((size_t) pg * nagg + k) * 6];
			int64_t		a = (int64_t)((uint64_t) rec[2] | ((uint64_t) rec[3] << 32));
			int64_t		b = (int64_t)((uint64_t) rec[4] | ((uint64_t) rec[5] << 32));

			switch (prog->agg_action[k])
			{
				case KAGG_ACTION__NROWS_ANY:
				case KAGG_ACTION__NROWS_COND:
					st->n[k] += rec[0];
					break;
				case KAGG_ACTION__PSUM_INT:
				case KAGG_ACTION__PSUM_INT64:
				case KAGG_ACTION__PAVG_INT:
				case KAGG_ACTION__PAVG_INT64:
		case KAGG_ACTION__PSUM_NUMERIC:
		case KAGG_ACTION__PAVG_NUMERIC:
					st->n[k] += rec[0];
					st->sum[k] += (__int128) b * 4294967296LL + (__int128) a;
					break;
				case KAGG_ACTION__PMIN_INT32:
				case KAGG_ACTION__PMIN_INT64:
					if (rec[0] && (!st->n[k] || a < st->val[k]))
						st->val[k] = a;
					if (rec[0])
						st->n[k] = 1;
					break;
				case KAGG_ACTION__PSUM_FP:		/* A10: float8, the partial of a workgroup is the bits of a double */
				case KAGG_ACTION__PAVG_FP:
				case KAGG_ACTION__PSUM_FP32:
				case KAGG_ACTION__PMIN_FP64:
				case KAGG_ACTION__PMAX_FP64:
					fagg_host_merge(fagg_class_of_action(prog->agg_action[k]), &st->n[k], &st->val[k], &st->fovf[k], rec[0], a);
					break;
				default:		/* PMAX_* */
					if (rec[0] && (!st->n[k] || a > st->val[k]))
						st->val[k] = a;
					if (rec[0])
						st->n[k] = 1;
					break;
			}
		}
	}
}

int
scan_exec_agg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
				   agg_state *st, const XpuCommand *task, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped, variant, avariant;
	gpu_vk_pipeline *sp;
	uint32_t	npages, nloaded, nagg, ngroups_total, total_lps = 0, flags_base;
	uint32_t   *lp_base = NULL;
	size_t		out_words;
	const uint32_t *out;
	int			rv = -1;
	double		t0 = now_ms(), t1 = 0, t2 = 0, t3 = 0;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return -1;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format != KDS_FORMAT_BLOCK)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuPreAgg yet",
				 kds_src->format);
		return -1;
	}
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	npages = kds_src->nitems;
	nloaded = kds_src->block_nloaded;
	if (nloaded > npages ||
		kds_src->block_offset + (size_t) nloaded * BLCKSZ > task->length)
	{
		snprintf(err, errlen, "corrupted block data store");
		return -1;
	}
	if (!prog->words && scan_prog_build(session, kds_src, prog, err, errlen) != 0)
		return -1;
	if (!prog->agg_mode)
	{
		snprintf(err, errlen, "not an aggregation program");
		return -1;
	}
	nagg = prog->nagg;
	if (st->nagg == 0)
	{
		st->nagg = nagg;
		memset(st->n, 0, sizeof(st->n));
		memset(st->sum, 0, sizeof(st->sum));
		memset(st->val, 0, sizeof(st->val));
		memset(st->fovf, 0, sizeof(st->fovf));
	}
	variant = (prog->nregs <= 8 ? 0 : prog->nregs <= 16 ? 1 : 2) + 3 * scan_feat_tier(prog->feat);
	avariant = scan_agg_variant(nagg);
	sp = spec_pipe(eng, prog, SPEC_AGG, 4 * (avariant + 1), 0, NULL, NULL, &eng->agg_pipe[variant][avariant]);
	if ((sp == &eng->agg_pipe[variant][avariant] || spec_check_enabled(eng)) &&
		ensure_agg_pipeline(eng, variant, avariant, err, errlen) != 0)
		return -1;
	if (npages == 0)
		return 0;

	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto done;
	t1 = now_ms();
	/* page table: the byte offset of each page (the second word, line pointer bases, is unused) */
	if (gpu_vk_buffer_create(vk, &btab, sizeof(uint32_t) * 2 * npages, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto done;
	}
	have_tab = 1;
	lp_base = malloc(sizeof(uint32_t) * npages);
	if (!lp_base)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
		((uint32_t *) btab.map)[2 * i + 1] = total_lps;
		lp_base[i] = total_lps;
		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
	}
	st->nitems_raw += total_lps;
	ngroups_total = (npages + AGG_PAGES_PER_WG - 1) / AGG_PAGES_PER_WG;
	/* after the per-workgroup records: one word per line pointer, 2 = the backend evaluates the row */
	flags_base = 4 + (uint32_t)((size_t) ngroups_total * nagg * 6);
	out_words = (size_t) flags_base + total_lps;
	if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
		goto done;
	}
	have_out = 1;
	out = (const uint32_t *) bout.map;
	memset(bout.map, 0, 16);
	memset((uint32_t *) bout.map + flags_base, 0, sizeof(uint32_t) * total_lps);
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto done;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);
	t2 = now_ms();
	{
		gpu_vk_buffer *bufs[5] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL};

		for (uint32_t base = 0; base < ngroups_total; base += SCAN_WG_MAX)
		{
			uint32_t	groups = ngroups_total - base;
			uint32_t	pc[4];

			if (groups > SCAN_WG_MAX)
				groups = SCAN_WG_MAX;
			pc[0] = npages;
			pc[1] = base;				/* first workgroup of this dispatch */
			pc[2] = AGG_PAGES_PER_WG;
			pc[3] = flags_base;
			if (kdispatch(eng, prog, SPEC_AGG, 4 * (avariant + 1), 0, &eng->agg_pipe[variant][avariant], sp, bufs, pc, groups) != 0)
			{
				snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
				goto done;
			}
		}
	}
	t3 = now_ms();
	if (out[1] != 0)
	{
		describe_kernel_error(out[1], out[2], out[3], err, errlen);
		goto done;
	}
	/* rows with a string the GPU cannot read: the backend evaluates them */
	for (uint32_t i = 0; i < npages; i++)
	{
		uint32_t	nlp = (i + 1 < npages ? lp_base[i + 1] : total_lps) - lp_base[i];
		const unsigned char *page = PAGE_AT(&ps, i);
		BlockNumber	blkno = ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[i];

		for (uint32_t lp = 0; lp < nlp; lp++)
		{
			uint32_t	lpd;

			if (out[flags_base + lp_base[i] + lp] != 2)
				continue;
			memcpy(&lpd, page + 24 + (size_t) lp * 4, 4);
			if (rck_add(&st->rck, blkno, lp + 1, page + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
			{
				snprintf(err, errlen, "out of memory");
				goto done;
			}
		}
	}
	/* add the per-page records to the session's aggregates */
	agg_merge_workgroups(prog, st, out, ngroups_total);
	st->npages_vfs_read += (npages - nloaded) * PAGES_PER_BLOCK;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: agg task: %u pages, pages-in %.1f ms%s, setup %.1f ms, gpu %.1f ms, merge %.1f ms\n",
				npages, t1 - t0, mapped ? " (mapped)" : "", t2 - t1, t3 - t2, now_ms() - t3);
	rv = 0;
done:
	free(lp_base);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "aggregation failed");
	return rv;
}

/* ---------------------------------------------------------------- */
/*
 * Aggregate state helpers shared by aggregation with and without GROUP BY
 */
typedef struct
{
	uint64_t	n;
	__int128	sum;
	int64_t		val;
	int			fovf;			/* a float8 sum overflowed */
} agg_acc;

/* add the contribution of one GPU record { n, A, B } to an accumulator */
static void
acc_merge_record(agg_acc *acc, uint32_t action, const uint32_t *rec)
{
	int64_t		a = (int64_t)((uint64_t) rec[2] | ((uint64_t) rec[3] << 32));
	int64_t		b = (int64_t)((uint64_t) rec[4] | ((uint64_t) rec[5] << 32));

	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:
		case KAGG_ACTION__NROWS_COND:
			acc->n += rec[0];
			break;
		case KAGG_ACTION__PSUM_INT:
		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PAVG_INT64:
		case KAGG_ACTION__PSUM_NUMERIC:
		case KAGG_ACTION__PAVG_NUMERIC:
			acc->n += rec[0];
			acc->sum += (__int128) b * 4294967296LL + (__int128) a;
			break;
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:
			if (rec[0] && (!acc->n || a < acc->val))
				acc->val = a;
			if (rec[0])
				acc->n = 1;
			break;
		default:		/* PMAX_* */
			if (rec[0] && (!acc->n || a > acc->val))
				acc->val = a;
			if (rec[0])
				acc->n = 1;
			break;
	}
}

/* add one record of the GROUP BY kernel (gagg.slang: counts, 16-bit limbs of biased sums, biased min/max) */
static void
acc_merge_group_record(agg_acc *acc, uint32_t action, const uint32_t *rec)
{
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:
		case KAGG_ACTION__NROWS_COND:
			acc->n += rec[0];
			break;
		case KAGG_ACTION__PSUM_INT:
		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PAVG_INT64:
		case KAGG_ACTION__PSUM_NUMERIC:
		case KAGG_ACTION__PAVG_NUMERIC:
		{
			__int128	u = (__int128) rec[2] + ((__int128) rec[3] << 16) +
				((__int128) rec[4] << 32) + ((__int128) rec[5] << 48);

			acc->n += rec[0];
			acc->sum += u - ((__int128) rec[0] << 63);
			break;
		}
		case KAGG_ACTION__PSUM_FP:		/* A10: the slot holds the bits of a double (words 2, 3), see gagg.slang */
		case KAGG_ACTION__PAVG_FP:
		case KAGG_ACTION__PSUM_FP32:
		case KAGG_ACTION__PMIN_FP64:
		case KAGG_ACTION__PMAX_FP64:
		{
			uint8_t		ovf = (uint8_t) acc->fovf;

			fagg_host_merge(fagg_class_of_action(action), &acc->n, &acc->val, &ovf, rec[0],
							(int64_t)((uint64_t) rec[2] | ((uint64_t) rec[3] << 32)));
			acc->fovf = ovf;
			break;
		}
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:
		{
			int64_t		v = (int32_t)(rec[2] ^ 0x80000000u);

			if (rec[0] && (!acc->n || v < acc->val))
				acc->val = v;
			if (rec[0])
				acc->n = 1;
			break;
		}
		default:		/* PMAX_* */
		{
			int64_t		v = (int32_t)(rec[2] ^ 0x80000000u);

			if (rec[0] && (!acc->n || v > acc->val))
				acc->val = v;
			if (rec[0])
				acc->n = 1;
			break;
		}
	}
}

/* agg_scales_ok() of scan_common.slang for the CPU path: every numeric aggregate argument and key has the plan's scale */
static int
cpu_scales_ok(const uint32_t *w, uint32_t aggoff, uint32_t nagg, uint32_t keyoff, uint32_t nkeys, const vm_reg *regs)
{
	for (uint32_t k = 0; k < nagg; k++)
	{
		uint32_t	sc1 = w[aggoff + 4 * k + 3], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		if (sc1 == 0)
			continue;
		if (rf != 0xffffffffu && (regs[rf].isnull || regs[rf].v == 0))
			continue;
		if (!regs[ra].isnull && regs[ra].sc != sc1 - 1)
			return 0;
	}
	for (uint32_t k = 0; k < nkeys; k++)
	{
		uint32_t	sc1 = w[keyoff + nkeys + k];

		if (sc1 == 0)
			continue;
		if (!regs[w[keyoff + k]].isnull && regs[w[keyoff + k]].sc != sc1 - 1)
			return 0;
	}
	return 1;
}

/* add one row's argument to an accumulator (the CPU path of GROUP BY overflow rows) */
static void
acc_add_value(agg_acc *acc, uint32_t cls, int64_t v)
{
	switch (cls)
	{
		case 1:			/* count(*) */
		case 2:			/* count(x) */
			acc->n += 1;
			break;
		case 3:			/* sum, avg */
			acc->n += 1;
			acc->sum += v;
			break;
		case 4:			/* min */
			if (!acc->n || v < acc->val)
				acc->val = v;
			acc->n = 1;
			break;
		case 6:			/* float8 sum, avg; 7 min, 8 max; 9 sum(float4) (A10) */
		case 7:
		case 8:
		case 9:
		{
			uint8_t		ovf = (uint8_t) acc->fovf;

			fagg_host_merge(cls, &acc->n, &acc->val, &ovf, 1, v);
			acc->fovf = ovf;
			break;
		}
		default:		/* max */
			if (!acc->n || v > acc->val)
				acc->val = v;
			acc->n = 1;
			break;
	}
}

/* the value of one output column of a partial aggregate row */
typedef struct
{
	int			isnull;
	uint32_t	len;
	unsigned char data[40];
	const unsigned char *ext;	/* a string key: the bytes (form_partial_tuple writes the varlena header) */
} col_value;

static int
agg_col_value(uint32_t action, int scale, const agg_acc *acc, col_value *cv, char *err, size_t errlen)
{
	int64_t		i64;

	memset(cv, 0, sizeof(*cv));
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:
		case KAGG_ACTION__NROWS_COND:
			i64 = (int64_t) acc->n;
			memcpy(cv->data, &i64, 8);
			cv->len = 8;
			break;

		case KAGG_ACTION__PSUM_INT:
		case KAGG_ACTION__PAVG_INT:
		{
			kagg_state__psum_int_packed *r = (kagg_state__psum_int_packed *) cv->data;

			if (acc->n == 0)
			{
				cv->isnull = 1;
				break;
			}
			if (acc->sum > (__int128) INT64_MAX || acc->sum < (__int128) INT64_MIN)
			{
				snprintf(err, errlen, "bigint out of range (sum of integers)");
				return -1;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->nitems = (int64_t) acc->n;
			r->sum = (int64_t) acc->sum;
			cv->len = sizeof(*r);
			break;
		}

		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_INT64:
		{
			kagg_state__psum_numeric_packed *r = (kagg_state__psum_numeric_packed *) cv->data;

			if (acc->n == 0)
			{
				cv->isnull = 1;
				break;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->attrs = 0;
			r->nitems = acc->n;
			__store_int128_packed(&r->sum, (int128_t) acc->sum);
			cv->len = sizeof(*r);
			break;
		}

		case KAGG_ACTION__PSUM_NUMERIC:
		case KAGG_ACTION__PAVG_NUMERIC:
		{
			/* the same state as the sum of int8, with the display scale of the inputs in attrs */
			kagg_state__psum_numeric_packed *r = (kagg_state__psum_numeric_packed *) cv->data;

			if (acc->n == 0)
			{
				cv->isnull = 1;
				break;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->attrs = (uint32_t)(scale < 0 ? 0 : scale) & __PAGG_NUMERIC_ATTRS__WEIGHT;
			r->nitems = acc->n;
			__store_int128_packed(&r->sum, (int128_t) acc->sum);
			cv->len = sizeof(*r);
			break;
		}

		case KAGG_ACTION__PSUM_FP:
		case KAGG_ACTION__PAVG_FP:
		case KAGG_ACTION__PSUM_FP32:
		{
			kagg_state__psum_fp_packed *r = (kagg_state__psum_fp_packed *) cv->data;

			if (acc->n == 0)
			{
				cv->isnull = 1;
				break;
			}
			if (acc->fovf)
			{
				snprintf(err, errlen, "value out of range: overflow");
				return -1;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->attrs = 0;
			r->nitems = (int64_t) acc->n;
			memcpy(&r->sum, &acc->val, sizeof(r->sum));
			cv->len = sizeof(*r);
			break;
		}

		case KAGG_ACTION__PMIN_FP64:
		case KAGG_ACTION__PMAX_FP64:
		{
			kagg_state__pminmax_fp64_packed *r = (kagg_state__pminmax_fp64_packed *) cv->data;

			if (!acc->n)
			{
				cv->isnull = 1;
				break;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->attrs = __PAGG_MINMAX_ATTRS__VALID;
			memcpy(&r->value, &acc->val, sizeof(r->value));
			cv->len = sizeof(*r);
			break;
		}

		default:		/* PMIN_*, PMAX_* */
		{
			kagg_state__pminmax_int64_packed *r = (kagg_state__pminmax_int64_packed *) cv->data;

			if (!acc->n)
			{
				cv->isnull = 1;
				break;
			}
			r->vl_len_ = (int32_t)(sizeof(*r) << 2);
			r->attrs = __PAGG_MINMAX_ATTRS__VALID;
			r->value = acc->val;
			cv->len = sizeof(*r);
			break;
		}
	}
	return 0;
}

/*
 * Form a tuple (kern_tupitem) of ncols values into tup (capacity tupsz). Returns t_len or -1.
 * The column types come from the result store's column metadata.
 */
static int
form_partial_tuple(const kern_data_store *kds_dst_head, const col_value *cols, uint32_t ncols,
				   unsigned char *tup, size_t tupsz, uint32_t rowid, char *err, size_t errlen)
{
	kern_tupitem *titem = (kern_tupitem *) tup;
	uint32_t	t_hoff;
	uint16_t	infomask = 0;
	int			hasnull = 0;

	for (uint32_t k = 0; k < ncols; k++)
		if (cols[k].isnull)
			hasnull = 1;
	if (hasnull)
		infomask |= 1;
	t_hoff = offsetof(kern_tupitem, t_bits);
	if (hasnull)
		t_hoff += (ncols + 7) / 8;
	t_hoff = MAXALIGN(t_hoff);
	memset(tup, 0, t_hoff);
	titem->t_infomask2 = (uint16_t)(ncols & 0x7ff);
	titem->t_hoff = (uint8_t)(t_hoff + MINIMAL_TUPLE_OFFSET);
	for (uint32_t k = 0; k < ncols; k++)
	{
		const kern_colmeta *cm = &kds_dst_head->colmeta[k];
		uint32_t	t_next;

		if (cols[k].isnull)
			continue;
		uint32_t	vlen = cols[k].len + (cols[k].ext ? 4 : 0);

		if (cm->attlen == -1)
			infomask |= 2;						/* HEAP_HASVARWIDTH */
		t_next = TYPEALIGN(cm->attalign, t_hoff);
		if (t_next + vlen + ROWID_SIZE > tupsz)
		{
			snprintf(err, errlen, "the partial aggregate row is too large");
			return -1;
		}
		if (t_next > t_hoff)
			memset(tup + t_hoff, 0, t_next - t_hoff);	/* the padding of a varlena must be zero bytes (heap_deform_tuple) */
		if (cols[k].ext)
		{
			uint32_t	hdr = vlen << 2;		/* a 4-byte varlena header (little endian) */

			memcpy(tup + t_next, &hdr, 4);
			memcpy(tup + t_next + 4, cols[k].ext, cols[k].len);
		}
		else
			memcpy(tup + t_next, cols[k].data, cols[k].len);
		if (hasnull)
			titem->t_bits[k >> 3] |= (1 << (k & 7));
		t_hoff = t_next + vlen;
	}
	t_hoff += ROWID_SIZE;
	if (t_hoff > tupsz)
	{
		snprintf(err, errlen, "the partial aggregate row is too large");
		return -1;
	}
	titem->t_len = t_hoff;
	titem->t_infomask = infomask;
	KERN_TUPITEM_SET_ROWID(titem, rowid);
	return (int) t_hoff;
}

/* ---------------------------------------------------------------- */
/*
 * String GROUP BY keys (A6): the session interns every distinct key value into a dictionary and a group's key is the id, so the
 * group table below stays a table of int64 keys. A bpchar value is identified by its bytes without the trailing blanks (the
 * stored value is the first one seen, whole, so a padded char(n) is returned as it is stored).
 */
typedef struct str_dict
{
	unsigned char *buf;
	size_t		len, cap;
	struct str_dict_ent
	{
		size_t		off;
		uint32_t	len;		/* stored bytes */
		uint32_t	klen;		/* bytes that identify it (bpchar: without trailing blanks) */
		uint32_t	bp;			/* a bpchar value (a text and a bpchar value never share an id: the stored bytes differ) */
	}		   *ents;
	uint32_t	nents, ecap;
	uint32_t   *htab;
	uint32_t	hcap;
	uint32_t	maxlen;
} str_dict;

static uint32_t
sd_hash(const unsigned char *p, uint32_t len)
{
	uint64_t	h = 0xcbf29ce484222325ULL;

	for (uint32_t i = 0; i < len; i++)
		h = (h ^ p[i]) * 0x100000001b3ULL;
	return (uint32_t)(h ^ (h >> 32));
}

static void
sd_free(str_dict *d)
{
	if (d)
	{
		free(d->buf);
		free(d->ents);
		free(d->htab);
		free(d);
	}
}

/* the id of a value, -1 on out of memory or when the dictionary alone exceeds the group memory */
static int64_t
sd_intern(agg_state *st, const unsigned char *p, uint32_t len, int bpchar)
{
	str_dict   *d = st->sdict;
	uint32_t	klen = len, h, i;

	if (!d)
	{
		d = calloc(1, sizeof(*d));
		if (!d)
			return -1;
		st->sdict = d;
	}
	if (bpchar)
		while (klen > 0 && p[klen - 1] == 0x20)
			klen--;
	if (d->nents * 2 >= d->hcap)
	{
		uint32_t	ncap = d->hcap ? d->hcap * 2 : 1024;
		uint32_t   *nh = calloc(ncap, sizeof(uint32_t));

		if (!nh)
			return -1;
		for (uint32_t e = 0; e < d->nents; e++)
		{
			uint32_t	x = (sd_hash(d->buf + d->ents[e].off, d->ents[e].klen) + d->ents[e].bp) & (ncap - 1);

			while (nh[x])
				x = (x + 1) & (ncap - 1);
			nh[x] = e + 1;
		}
		free(d->htab);
		d->htab = nh;
		d->hcap = ncap;
	}
	h = (sd_hash(p, klen) + (bpchar ? 1 : 0)) & (d->hcap - 1);
	for (i = h; d->htab[i]; i = (i + 1) & (d->hcap - 1))
	{
		const struct str_dict_ent *e = &d->ents[d->htab[i] - 1];

		if (e->klen == klen && e->bp == (uint32_t) bpchar && !memcmp(d->buf + e->off, p, klen))
			return (int64_t) d->htab[i] - 1;
	}
	if (d->nents == d->ecap)
	{
		uint32_t	ncap = d->ecap ? d->ecap * 2 : 256;
		void	   *ne = realloc(d->ents, sizeof(*d->ents) * ncap);

		if (!ne)
			return -1;
		d->ents = ne;
		d->ecap = ncap;
	}
	if (d->len + len > d->cap)
	{
		size_t		ncap = d->cap ? d->cap * 2 : 4096;
		unsigned char *nb;

		while (ncap < d->len + len)
			ncap *= 2;
		if (ncap > ((size_t) 1 << 30))
			return -1;
		nb = realloc(d->buf, ncap);
		if (!nb)
			return -1;
		d->buf = nb;
		d->cap = ncap;
	}
	memcpy(d->buf + d->len, p, len);
	d->ents[d->nents].off = d->len;
	d->ents[d->nents].len = len;
	d->ents[d->nents].klen = klen;
	d->ents[d->nents].bp = bpchar ? 1 : 0;
	d->len += len;
	if (len > d->maxlen)
		d->maxlen = len;
	d->htab[i] = ++d->nents;
	return (int64_t) d->nents - 1;
}

static const unsigned char *
sd_get(const agg_state *st, int64_t id, uint32_t *p_len)
{
	const str_dict *d = st->sdict;

	if (!d || id < 0 || id >= (int64_t) d->nents)
	{
		*p_len = 0;
		return (const unsigned char *) "";
	}
	*p_len = d->ents[id].len;
	return d->buf + d->ents[id].off;
}

/*
 * The string keys of a group key as the kernel or the CPU VM staged them (string registers { offset | length << 32 | source << 60 })
 * into dictionary ids. base[s] is the buffer source s names. 0 on success, -1 on out of memory.
 */
static int
group_str_keys(agg_state *st, const scan_prog *prog, int64_t *key, uint32_t knull, const unsigned char *const *base)
{
	for (uint32_t k = 0; k < (uint32_t) prog->nkeys && k < SCAN_GROUP_KEYS; k++)
	{
		uint64_t	v;
		int64_t		id;

		if (!prog->key_str[k] || (knull & (1u << k)))
			continue;
		v = (uint64_t) key[k];
		if (!base[(v >> 60) & 3])
			return -1;
		id = sd_intern(st, base[(v >> 60) & 3] + (v & 0xffffffffu), (uint32_t)((v >> 32) & 0x0fffffffu), prog->key_str[k] == 2);
		if (id < 0)
			return -1;
		key[k] = id;
	}
	return 0;
}

/* ---------------------------------------------------------------- */
/*
 * The groups of a session
 */
typedef struct
{
	int64_t		key[SCAN_GROUP_KEYS];
	uint32_t	knull;
	agg_acc		acc[];				/* nagg of them (group_table.esize) */
} group_entry;

struct group_table
{
	group_entry	   *ents;			/* nents entries of esize bytes */
	size_t			esize;
	uint32_t		nents, cap;
	uint32_t	   *htab;			/* index + 1, 0 = empty */
	uint32_t		hcap;
};

static uint32_t
group_hash(const int64_t *key, uint32_t knull)
{
	uint64_t	h = 0xcbf29ce484222325ULL;

	for (int i = 0; i < SCAN_GROUP_KEYS; i++)
	{
		h = (h ^ (uint64_t) key[i]) * 0x100000001b3ULL;
		h ^= h >> 29;
	}
	h = (h ^ knull) * 0x100000001b3ULL;
	return (uint32_t)(h ^ (h >> 32));
}

#define GENT(gt, i)	((group_entry *)((char *)(gt)->ents + (size_t)(i) * (gt)->esize))

/* PGSTROM_GPUSERV_GROUP_MEM_MB (pg_strom.gpuserv_group_memory): the memory one session may use for its groups */
static size_t
group_mem_limit(void)
{
	static size_t v = 0;

	if (!v)
	{
		const char *s = getenv("PGSTROM_GPUSERV_GROUP_MEM_MB");
		long		mb = s ? atol(s) : 1024;

		v = (size_t)(mb > 0 ? mb : 1024) << 20;
	}
	return v;
}

/* NULL on out of memory, or when the session's groups would exceed the limit (the caller reports "out of memory") */
static group_entry *
group_lookup(struct group_table **p_gt, uint32_t nagg, const int64_t *key, uint32_t knull)
{
	struct group_table *gt = *p_gt;
	uint32_t	h, i;

	if (!gt)
	{
		gt = calloc(1, sizeof(*gt));
		if (!gt)
			return NULL;
		gt->esize = sizeof(group_entry) + (size_t) nagg * sizeof(agg_acc);
		*p_gt = gt;
	}
	if (gt->nents * 2 >= gt->hcap)
	{
		uint32_t	ncap = gt->hcap ? gt->hcap * 2 : 1024;
		uint32_t   *nh = calloc(ncap, sizeof(uint32_t));

		if (!nh)
			return NULL;
		for (uint32_t e = 0; e < gt->nents; e++)
		{
			uint32_t	x = group_hash(GENT(gt, e)->key, GENT(gt, e)->knull) & (ncap - 1);

			while (nh[x])
				x = (x + 1) & (ncap - 1);
			nh[x] = e + 1;
		}
		free(gt->htab);
		gt->htab = nh;
		gt->hcap = ncap;
	}
	h = group_hash(key, knull) & (gt->hcap - 1);
	for (i = h; gt->htab[i]; i = (i + 1) & (gt->hcap - 1))
	{
		group_entry *e = GENT(gt, gt->htab[i] - 1);

		if (e->knull == knull && !memcmp(e->key, key, sizeof(e->key)))
			return e;
	}
	if (gt->nents == gt->cap)
	{
		uint32_t	ncap = gt->cap ? gt->cap * 2 : 256;
		group_entry *ne;

		if ((size_t) ncap * gt->esize > group_mem_limit())
			return NULL;
		ne = realloc(gt->ents, gt->esize * ncap);

		if (!ne)
			return NULL;
		gt->ents = ne;
		gt->cap = ncap;
	}
	{
		group_entry *e = GENT(gt, gt->nents);

		memset(e, 0, gt->esize);
		memcpy(e->key, key, sizeof(e->key));
		e->knull = knull;
		gt->htab[i] = ++gt->nents;
		return e;
	}
}

static void gtab_destroy(agg_state *st);

void
agg_state_free(agg_state *st)
{
	gtab_destroy(st);
	free(st->rck.buf);
	st->rck.buf = NULL;
	st->rck.len = st->rck.cap = st->rck.nitems = 0;
	sd_free(st->sdict);
	st->sdict = NULL;
	if (st->groups)
	{
		free(st->groups->ents);
		free(st->groups->htab);
		free(st->groups);
		st->groups = NULL;
	}
}

/* the capacity of a workgroup's table and the aggregate variant, as compiled into gagg.slang */
static const int ga_variant_na[3] = {4, 12, 12};
static const int ga_variant_cap[3] = {128, 64, 64};

static int
ensure_group_pipeline(scan_engine *eng, int variant, int avariant, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } tab[][3] = {
		{{kspv_gagg16_4, &kspv_gagg16_4_len}, {kspv_gagg16_12, &kspv_gagg16_12_len},
		 {kspv_gagg16_12, &kspv_gagg16_12_len}},
		{{kspv_gagg32_4, &kspv_gagg32_4_len}, {kspv_gagg32_12, &kspv_gagg32_12_len},
		 {kspv_gagg32_12, &kspv_gagg32_12_len}},
		{{kspv_gagg16_4_lite, &kspv_gagg16_4_lite_len}, {kspv_gagg16_12_lite, &kspv_gagg16_12_lite_len},
		 {kspv_gagg16_12_lite, &kspv_gagg16_12_lite_len}},
		{{kspv_gagg32_4_lite, &kspv_gagg32_4_lite_len}, {kspv_gagg32_12_lite, &kspv_gagg32_12_lite_len},
		 {kspv_gagg32_12_lite, &kspv_gagg32_12_lite_len}},
		{{kspv_gagg16_4_nogeom, &kspv_gagg16_4_nogeom_len}, {kspv_gagg16_12_nogeom, &kspv_gagg16_12_nogeom_len},
		 {kspv_gagg16_12_nogeom, &kspv_gagg16_12_nogeom_len}},
		{{kspv_gagg32_4_nogeom, &kspv_gagg32_4_nogeom_len}, {kspv_gagg32_12_nogeom, &kspv_gagg32_12_nogeom_len},
		 {kspv_gagg32_12_nogeom, &kspv_gagg32_12_nogeom_len}},
	};
	_Static_assert(sizeof(tab) / sizeof(tab[0]) == SCAN_NV2, "tab: one entry per register variant and tier");

	if (!eng->gagg_pipe_ready[variant][avariant])
	{
		if (gpu_vk_pipeline_create(vk, &eng->gagg_pipe[variant][avariant], tab[variant][avariant].spv,
								   *tab[variant][avariant].len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the GROUP BY pipeline: %s", vk->err);
			return -1;
		}
		eng->gagg_pipe_ready[variant][avariant] = 1;
	}
	return 0;
}

/*
 * Merge the workgroups' group tables of a GROUP BY kernel (gagg.slang, jgagg.slang) into the session's
 * groups. Each block is { hits, cap records of rec_words }. -1 on out of memory.
 */
static int
group_merge_blocks(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total,
				   uint32_t cap, uint32_t rec_words, uint32_t blk_words)
{
	uint32_t	nagg = prog->nagg;
	uint32_t	agg_col[SCAN_AGG_MAX];

	for (uint32_t a = 0; a < nagg; a++)
	{
		uint32_t	col;

		for (col = 0; col < (uint32_t) prog->nattrs; col++)
			if (prog->agg_of_col[col] == (int8_t) a)
				break;
		agg_col[a] = col;
	}
	for (uint32_t wg = 0; wg < ngroups_total; wg++)
	{
		const uint32_t *blk = &out[4 + (size_t) wg * blk_words];

		st->nitems_in += blk[0];
		for (uint32_t s = 0; s < cap; s++)
		{
			const uint32_t *rec = blk + 1 + (size_t) s * rec_words;
			int64_t		key[SCAN_GROUP_KEYS] = {0};
			group_entry *e;

			if (!rec[0])
				continue;
			for (int i = 0; i < SCAN_GROUP_KEYS_SHARED; i++)
				key[i] = (int64_t)((uint64_t) rec[2 + 2 * i] | ((uint64_t) rec[3 + 2 * i] << 32));
			e = group_lookup(&st->groups, nagg, key, rec[1]);
			if (!e)
				return -1;
			for (uint32_t a = 0; a < nagg; a++)
				acc_merge_group_record(&e->acc[a], prog->agg_action[agg_col[a]],
									   rec + 2 + 2 * SCAN_GROUP_KEYS_SHARED + 6 * a);
		}
	}
	return 0;
}

/*
 * The rows of a GROUP BY task that the kernel did not aggregate (flags, one word per line pointer: 1 the CPU aggregates the row
 * here, 2 the backend evaluates it): evaluated by the VM on the CPU into the session's groups, or kept for the backend.
 * -1 on error (err is set).
 */
static int
group_cpu_rows(const scan_prog *prog, agg_state *st, const kern_data_store *kds_src, const page_set *ps,
			   const uint32_t *lp_base, uint32_t npages, uint32_t total_lps, const uint32_t *flags,
			   uint64_t *p_noverflow, char *err, size_t errlen)
{
	const uint32_t *w = prog->words;
	uint32_t	nagg = prog->nagg;
	uint64_t	n_overflow = 0;
	uint32_t	aggoff = w[11], keyoff = w[13];
	uint32_t	nkeys = prog->nkeys;

	for (uint32_t i = 0; i < npages; i++)
	{
		uint32_t	nlp = (i + 1 < npages ? lp_base[i + 1] : total_lps) - lp_base[i];
		const unsigned char *page = PAGE_AT(ps, i);

		for (uint32_t lp = 0; lp < nlp; lp++)
		{
			vm_reg		regs[SCAN_MAX_REGS];
			uint32_t	lpd, e = 0, knull = 0;
			int64_t		key[SCAN_GROUP_KEYS] = {0};
			group_entry *g;

			if (!flags[lp_base[i] + lp])
				continue;
			memcpy(&lpd, page + 24 + (size_t) lp * 4, 4);
			if (flags[lp_base[i] + lp] == 2)		/* a string the GPU cannot read: the backend */
			{
				if (rck_add(&st->rck, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[i], lp + 1,
							page + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
				{
					snprintf(err, errlen, "out of memory");
					return -1;
				}
				continue;
			}
			n_overflow++;
			if (cpu_eval_row(prog, page + (lpd & 0x7fff), regs, &e) != 1)
			{
				if (e == SOP_RECHECK_CODE)			/* a numeric the VM cannot evaluate: the backend */
				{
					if (rck_add(&st->rck, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[i], lp + 1,
								page + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
					{
						snprintf(err, errlen, "out of memory");
						return -1;
					}
					continue;
				}
				if (e != 0)
				{
					describe_kernel_error(e, i, lp, err, errlen);
					return -1;
				}
				continue;
			}
			if (!cpu_scales_ok(w, aggoff, nagg, keyoff, nkeys, regs))
			{
				/* a numeric of another scale than the plan's: the backend evaluates the row */
				if (rck_add(&st->rck, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[i], lp + 1,
							page + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
				{
					snprintf(err, errlen, "out of memory");
					return -1;
				}
				continue;
			}
			for (uint32_t k = 0; k < nkeys; k++)
			{
				const vm_reg *r = &regs[w[keyoff + k]];

				if (r->isnull)
					knull |= (1u << k);
				else
					key[k] = r->v;
			}
			if (prog->has_str_key)
			{
				/* the CPU VM's string registers are offsets from the tuple header (source 0) */
				const unsigned char *sb[4] = { page + (lpd & 0x7fff), NULL, NULL, (const unsigned char *) (w + w[20]) };

				if (group_str_keys(st, prog, key, knull, sb) != 0)
				{
					snprintf(err, errlen, "out of memory for the GROUP BY groups (pg_strom.gpuserv_group_memory)");
					return -1;
				}
			}
			g = group_lookup(&st->groups, nagg, key, knull);
			if (!g)
			{
				snprintf(err, errlen, "out of memory for the GROUP BY groups (pg_strom.gpuserv_group_memory)");
				return -1;
			}
			for (uint32_t a = 0; a < nagg; a++)
			{
				uint32_t	cls = w[aggoff + 4 * a], ra = w[aggoff + 4 * a + 1], rf = w[aggoff + 4 * a + 2];

				if (rf != 0xffffffffu && (regs[rf].isnull || regs[rf].v == 0))
					continue;
				if (cls != 1 && regs[ra].isnull)
					continue;
				acc_add_value(&g->acc[a], cls, (cls == 1 || cls == 2) ? 0 : regs[ra].v);
			}
		}
	}
	*p_noverflow += n_overflow;
	return 0;
}

/* ---------------------------------------------------------------- */
/*
 * GROUP BY over many groups: one hash table in device memory for the whole session (kernels_vm/gtab.slang, which describes
 * the layout). The tasks add to it; it is read once, when it has to grow and at the end of the session (scan_group_flush),
 * and merged into the session's groups, where the rows aggregated on the CPU are too.
 */
#define GT_HDR			8
#define GT_KEYW			(2 + 2 * SCAN_GROUP_KEYS)
#define GT_MAX_BYTES	(512U << 20)	/* the table of a session */
#define GT_INIT_SLOTS	(1U << 20)		/* the largest table created from the planner's estimate (it grows if needed) */

static uint64_t
gt_max_bytes(void)
{
	uint64_t	lim = group_mem_limit() / 2;		/* the other half is for the session's groups on the host */

	return lim < GT_MAX_BYTES ? lim : GT_MAX_BYTES;
}

typedef struct gtab_state
{
	gpu_vk_ctx *vk;
	gpu_vk_buffer buf;
	uint32_t	cap;				/* slots (a power of 2) */
	uint32_t	stride;				/* words per slot */
	uint32_t	maxcap;
	uint32_t	last_used;			/* slots the last task of a string-key table claimed (read before the flush cleared them) */
} gtab_state;

static uint32_t
gt_width(uint32_t cls)
{
	return (cls == 1 || cls == 2) ? 1 : (cls == 3 ? 10 : 2);
}

static uint32_t
gt_stride(const scan_prog *prog)
{
	uint32_t	stride = GT_KEYW;

	for (uint32_t a = 0; a < prog->nagg; a++)
		stride += gt_width(prog->words[prog->words[11] + 4 * a]);
	return stride;
}

/*
 * PGSTROM_GPUSERV_GTAB: 0 never, 1 always (tests); default: once a task of the shared-memory kernel has found that most of
 * its rows overflow (scan_exec_group_task), the session uses the table. The planner's estimate is no guide (PostgreSQL
 * estimates an expression such as a % 10 by the distinct values of a), it only sizes the table.
 */
/* float aggregates (A10) are accumulated in order by the slot owners of gagg.slang / jgagg.slang, not with the atomics of gtab.slang */
int
scan_prog_has_float_agg(const scan_prog *prog)
{
	for (uint32_t a = 0; a < prog->nagg; a++)
		if (prog->words[prog->words[11] + 4 * a] >= 6)
			return 1;
	return 0;
}

int
scan_gtab_forced(const agg_state *st, const scan_prog *prog)
{
	if (scan_prog_has_float_agg(prog))
		return 0;				/* (the planner offers float aggregates with at most SCAN_GROUP_KEYS_SHARED keys) */
	static int	mode = -1;

	if (mode < 0)
	{
		const char *s = getenv("PGSTROM_GPUSERV_GTAB");

		mode = s ? atoi(s) : 2;
	}
	if (prog->has_str_key)
		return 1;				/* the shared-memory kernels hold int64 keys (a string key is compared by its bytes in the table) */
	if (prog->nkeys > SCAN_GROUP_KEYS_SHARED)
		return 1;				/* the shared-memory kernels hold two keys */
	if (mode == 0)
		return 0;
	return mode == 1 || st->gt != NULL;
}

static void
gtab_destroy(agg_state *st)
{
	if (st->gt)
	{
		if (st->gt->cap)
			gpu_vk_buffer_destroy(st->gt->vk, &st->gt->buf);
		free(st->gt);
		st->gt = NULL;
	}
}

/* a (new) table of at least 'want' slots; the memory limit caps it */
static int
gtab_create(gpu_vk_ctx *vk, const scan_prog *prog, agg_state *st, uint64_t want, char *err, size_t errlen)
{
	gtab_state *gt = st->gt;
	uint32_t	stride = gt_stride(prog), cap = 4096, maxcap = 4096;
	size_t		bytes;
	uint32_t   *hdr;

	while ((uint64_t)(maxcap * 2) * stride * 4 <= gt_max_bytes())
		maxcap *= 2;
	while (cap < want && cap < maxcap)
		cap *= 2;
	if (!gt)
	{
		gt = calloc(1, sizeof(*gt));
		if (!gt)
		{
			snprintf(err, errlen, "out of memory");
			return -1;
		}
		st->gt = gt;
	}
	bytes = ((size_t) GT_HDR + (size_t) cap * stride) * 4;
	if (gpu_vk_buffer_create(vk, &gt->buf, bytes, 0) != 0)
	{
		gt->cap = 0;
		snprintf(err, errlen, "cannot allocate the group table: %s", vk->err);
		return -1;
	}
	memset(gt->buf.map, 0, bytes);
	hdr = (uint32_t *) gt->buf.map;
	hdr[2] = cap - 1;
	hdr[3] = stride;
	gt->vk = vk;
	gt->cap = cap;
	gt->stride = stride;
	gt->maxcap = maxcap;
	return 0;
}

/* one slot's accumulators of an aggregate into an accumulator (the words are described in gtab.slang) */
static void
acc_merge_gtab(agg_acc *acc, uint32_t action, const uint32_t *w)
{
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:
		case KAGG_ACTION__NROWS_COND:
			acc->n += w[0];
			break;
		case KAGG_ACTION__PSUM_INT:
		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PAVG_INT64:
		case KAGG_ACTION__PSUM_NUMERIC:
		case KAGG_ACTION__PAVG_NUMERIC:
		{
			__int128	pos = (__int128) w[1] + ((__int128)(uint64_t) w[2] << 32);
			__int128	neg = (__int128) w[3] + ((__int128)(uint64_t) w[4] << 32);
			__int128	xl = (__int128) w[6] + ((__int128)(uint64_t) w[7] << 32);
			__int128	xh = (__int128) w[8] + ((__int128)(uint64_t) w[9] << 32);

			acc->n += w[0];
			acc->sum += pos - neg + xl + (xh << 32) - ((__int128) w[5] << 63);
			break;
		}
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:
		{
			int64_t		v = (int32_t)(~w[1] ^ 0x80000000u);

			if (w[0] && (!acc->n || v < acc->val))
				acc->val = v;
			if (w[0])
				acc->n = 1;
			break;
		}
		default:		/* PMAX_* */
		{
			int64_t		v = (int32_t)(w[1] ^ 0x80000000u);

			if (w[0] && (!acc->n || v > acc->val))
				acc->val = v;
			if (w[0])
				acc->n = 1;
			break;
		}
	}
}

/* merge the device table into the session's groups; 'clear' empties the table */
static int
gtab_flush(const scan_prog *prog, agg_state *st, int clear, const unsigned char *const *srcs, char *err, size_t errlen)
{
	gtab_state *gt = st->gt;
	uint32_t	nagg = prog->nagg;
	uint32_t	agg_col[SCAN_AGG_MAX], agg_off[SCAN_AGG_MAX];
	uint32_t   *base = gt ? (uint32_t *) gt->buf.map : NULL;
	uint32_t	off = GT_KEYW;

	if (!gt || !gt->cap)
		return 0;
	for (uint32_t a = 0; a < nagg; a++)
	{
		uint32_t	col;

		for (col = 0; col < (uint32_t) prog->nattrs; col++)
			if (prog->agg_of_col[col] == (int8_t) a)
				break;
		agg_col[a] = col;
		agg_off[a] = off;
		off += gt_width(prog->words[prog->words[11] + 4 * a]);
	}
	for (uint32_t s = 0; s < gt->cap; s++)
	{
		const uint32_t *w = base + GT_HDR + (size_t) s * gt->stride;
		int64_t		key[SCAN_GROUP_KEYS];
		group_entry *e;

		if (!w[0])
			continue;
		for (int i = 0; i < SCAN_GROUP_KEYS; i++)
			key[i] = (int64_t)((uint64_t) w[2 + 2 * i] | ((uint64_t) w[3 + 2 * i] << 32));
		if (prog->has_str_key && (!srcs || group_str_keys(st, prog, key, w[1], srcs) != 0))
		{
			snprintf(err, errlen, "out of memory for the GROUP BY groups (pg_strom.gpuserv_group_memory)");
			return -1;
		}
		e = group_lookup(&st->groups, nagg, key, w[1]);
		if (!e)
		{
			snprintf(err, errlen, "out of memory for the GROUP BY groups (pg_strom.gpuserv_group_memory)");
			return -1;
		}
		for (uint32_t a = 0; a < nagg; a++)
			acc_merge_gtab(&e->acc[a], prog->agg_action[agg_col[a]], w + agg_off[a]);
	}
	if (clear)
	{
		memset(base + GT_HDR, 0, (size_t) gt->cap * gt->stride * 4);
		base[0] = 0;
	}
	return 0;
}

/* the end of the session: the table's groups into the session's groups (the reply is built from those) */
int
scan_group_flush(const scan_prog *prog, agg_state *st, char *err, size_t errlen)
{
	if (!st->gt)
		return 0;
	if (gtab_flush(prog, st, 0, NULL, err, errlen) != 0)
		return -1;
	gtab_destroy(st);
	return 0;
}

int
scan_gtab_switch_allowed(void)
{
	const char *s = getenv("PGSTROM_GPUSERV_GTAB");

	return !s || atoi(s) == 2;
}

/* the kernel of the session-wide table: arrow = 0 heap pages (gtab.slang), 1 an Arrow record batch (-DARROW_SRC) */
gpu_vk_pipeline *
scan_gtab_vm_pipe(scan_engine *eng, int arrow, int variant)
{
	return arrow ? &eng->agtab_pipe[variant] : &eng->gtab_pipe[variant];
}

/* creates the interpreter's pipeline (the specialized kernel, if there is one, does not need it) */
int
scan_gtab_pipeline(scan_engine *eng, int arrow, int variant, gpu_vk_pipeline **p_pipe, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } tab[] = {
		{kspv_gtab16, &kspv_gtab16_len}, {kspv_gtab32, &kspv_gtab32_len},
		{kspv_gtab16_lite, &kspv_gtab16_lite_len}, {kspv_gtab32_lite, &kspv_gtab32_lite_len},
		{kspv_gtab16_nogeom, &kspv_gtab16_nogeom_len}, {kspv_gtab32_nogeom, &kspv_gtab32_nogeom_len},
	};
	static const struct { const unsigned char *spv; const unsigned int *len; } atab[] = {
		{kspv_agtab16, &kspv_agtab16_len}, {kspv_agtab32, &kspv_agtab32_len},
		{kspv_agtab16_lite, &kspv_agtab16_lite_len}, {kspv_agtab32_lite, &kspv_agtab32_lite_len},
		{kspv_agtab16_nogeom, &kspv_agtab16_nogeom_len}, {kspv_agtab32_nogeom, &kspv_agtab32_nogeom_len},
	};
	_Static_assert(sizeof(tab) / sizeof(tab[0]) == SCAN_NV2, "tab: one entry per register variant and tier");
	_Static_assert(sizeof(atab) / sizeof(atab[0]) == SCAN_NV2, "atab: one entry per register variant and tier");
	gpu_vk_pipeline *pipe = arrow ? &eng->agtab_pipe[variant] : &eng->gtab_pipe[variant];
	int		   *ready = arrow ? &eng->agtab_pipe_ready[variant] : &eng->gtab_pipe_ready[variant];

	if (!*ready)
	{
		if (gpu_vk_pipeline_create(vk, pipe, arrow ? atab[variant].spv : tab[variant].spv,
								   arrow ? *atab[variant].len : *tab[variant].len, 6, arrow ? 20 : 16) != 0)
		{
			snprintf(err, errlen, "cannot create the group table pipeline: %s", vk->err);
			return -1;
		}
		*ready = 1;
	}
	*p_pipe = pipe;
	return 0;
}

/* the table of the session, created at its first task (twice the planner's estimate, a table that exists is kept) */
int
scan_gtab_open(gpu_vk_ctx *vk, const kern_session_info *session, const scan_prog *prog, agg_state *st,
			   char *err, size_t errlen)
{
	double		est = session->groupby_ngroups_estimation;

	if (st->gt && st->gt->cap)
		return 0;
	if (!(est >= 0))
		est = 0;
	if (est * 2 > GT_INIT_SLOTS)
		est = GT_INIT_SLOTS / 2;
	/* the planner's estimate, and at least what the session has seen (twice: the table is kept below half full) */
	if (gtab_create(vk, prog, st, (uint64_t)(est * 2) > 4096 ? (uint64_t)(est * 2) : 4096, err, errlen) != 0)
		return -1;
	if (st->groups && (uint64_t) st->groups->nents * 2 > st->gt->cap)
	{
		gtab_destroy(st);
		if (gtab_create(vk, prog, st, (uint64_t) st->groups->nents * 4, err, errlen) != 0)
			return -1;
	}
	return 0;
}

gpu_vk_buffer *
scan_gtab_buffer(agg_state *st)
{
	return &st->gt->buf;
}

/* a string-key table: the groups of the task into the session's, the table empty for the next task (see scan_exec.h) */
int
scan_gtab_task_flush(const scan_prog *prog, agg_state *st, const unsigned char *const *srcs, char *err, size_t errlen)
{
	uint32_t   *hdr = (uint32_t *) st->gt->buf.map;

	st->gt->last_used = hdr[0];
	return gtab_flush(prog, st, 1, srcs, err, errlen);
}

/* after a task: a table more than half full grows (its groups move to the session's groups, the next task starts a bigger one) */
int
scan_gtab_task_done(gpu_vk_ctx *vk, const scan_prog *prog, agg_state *st, char *err, size_t errlen)
{
	uint32_t   *hdr = (uint32_t *) st->gt->buf.map;
	uint32_t	used = prog->has_str_key ? st->gt->last_used : hdr[0];

	if (st->gt->cap < st->gt->maxcap && (uint64_t) used * 2 > st->gt->cap)
	{
		uint32_t	ncap = st->gt->cap * 4;

		if (gtab_flush(prog, st, 0, NULL, err, errlen) != 0)
			return -1;
		gtab_destroy(st);
		if (gtab_create(vk, prog, st, ncap, err, errlen) != 0)
			return -1;
	}
	return 0;
}

/* rows an Arrow task could not aggregate on the table (mode 3 of agtab: key words, null mask and values staged by the kernel) */
int
scan_gtab_merge_overflow(const scan_prog *prog, agg_state *st, const uint32_t *recs, uint32_t count,
						 const unsigned char *const *srcs)
{
	uint32_t	nagg = prog->nagg;
	const uint32_t *w = prog->words;
	uint32_t	aggoff = w[11];
	uint32_t	ov_rec = 1 + 2 * SCAN_GROUP_KEYS + 3 * 12;

	for (uint32_t r = 0; r < count; r++)
	{
		const uint32_t *rec = recs + (size_t) r * ov_rec;
		int64_t		key[SCAN_GROUP_KEYS];
		group_entry *g;

		for (int i = 0; i < SCAN_GROUP_KEYS; i++)
			key[i] = (int64_t)((uint64_t) rec[1 + 2 * i] | ((uint64_t) rec[2 + 2 * i] << 32));
		if (prog->has_str_key && (!srcs || group_str_keys(st, prog, key, rec[0], srcs) != 0))
			return -1;
		g = group_lookup(&st->groups, nagg, key, rec[0]);
		if (!g)
			return -1;
		for (uint32_t a = 0; a < nagg; a++)
		{
			const uint32_t *av = rec + 1 + 2 * SCAN_GROUP_KEYS + 3 * a;

			if (!av[0])
				continue;
			acc_add_value(&g->acc[a], w[aggoff + 4 * a],
						  (int64_t)((uint64_t) av[1] | ((uint64_t) av[2] << 32)));
		}
	}
	return 0;
}

/*
 * A GROUP BY task on the session's device table: up to three dispatches (gtab.slang: look up and add, insert the keys that
 * were missing, add those rows), then the rows no slot could take on the CPU.
 */
static int
group_task_gtab(scan_engine *eng, const kern_session_info *session, scan_prog *prog, agg_state *st,
				const XpuCommand *task, const kern_data_store *kds_src, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped, variant, rv = -1;
	uint32_t	npages = kds_src->nitems, nloaded = kds_src->block_nloaded, total_lps = 0, ngroups_total, nmiss = 0;
	uint32_t   *lp_base = NULL, *hdr;
	size_t		out_words;
	const uint32_t *out;
	double		t0 = now_ms(), t1, t2 = 0, t3 = 0;
	uint64_t	n_overflow = 0;
	uint32_t	hits0;
	gpu_vk_pipeline *gpipe, *sp;
	spec_shape	sh;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	variant = (prog->nregs <= 16 ? 0 : 1) + 2 * scan_feat_tier(prog->feat);
	gpipe = scan_gtab_vm_pipe(eng, 0, variant);
	memset(&sh, 0, sizeof(sh));
	sp = spec_pipe(eng, prog, SPEC_GTAB, 0, 0, NULL, &sh, gpipe);
	if ((sp == gpipe || spec_check_enabled(eng)) && scan_gtab_pipeline(eng, 0, variant, &gpipe, err, errlen) != 0)
		return -1;
	if (scan_gtab_open(vk, session, prog, st, err, errlen) != 0)
		return -1;
	if (npages == 0)
		return 0;
	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto done;
	lp_base = malloc(sizeof(uint32_t) * npages);
	if (!lp_base)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	t1 = now_ms();
	if (gpu_vk_buffer_create(vk, &btab, sizeof(uint32_t) * 2 * npages, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto done;
	}
	have_tab = 1;
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		lp_base[i] = total_lps;
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
		((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
		((uint32_t *) btab.map)[2 * i + 1] = lp_base[i];
	}
	st->nitems_raw += total_lps;
	ngroups_total = (npages + group_ppw(AGG_PAGES_PER_WG) - 1) / group_ppw(AGG_PAGES_PER_WG);
	out_words = 4 + (size_t) total_lps;
	if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
		goto done;
	}
	have_out = 1;
	out = (const uint32_t *) bout.map;
	memset(bout.map, 0, 16);
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto done;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);
	hdr = (uint32_t *) st->gt->buf.map;
	hits0 = hdr[1] = 0;
	{
		gpu_vk_buffer *bufs[6] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL, &st->gt->buf};
		const uint32_t *flags = out + 4;

		for (uint32_t mode = 0; mode < 3; mode++)
		{
			if (mode == 1 && nmiss == 0)
				continue;
			if (mode == 2 && nmiss == 0)
				continue;
			for (uint32_t base = 0; base < ngroups_total; base += SCAN_WG_MAX)
			{
				uint32_t	groups = ngroups_total - base;
				uint32_t	pc[4];

				if (groups > SCAN_WG_MAX)
					groups = SCAN_WG_MAX;
				pc[0] = npages;
				pc[1] = base | (mode << 28);
				pc[2] = group_ppw(AGG_PAGES_PER_WG);
				pc[3] = 4;
				if (spec_dispatch(eng, prog, SPEC_GTAB, &sh, gpipe, sp, bufs, pc, groups) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto done;
				}
			}
			if (out[1] != 0)
			{
				describe_kernel_error(out[1], out[2], out[3], err, errlen);
				goto done;
			}
			if (mode == 0)
			{
				for (uint32_t i = 0; i < total_lps; i++)
					nmiss += (flags[i] == 4);
				t2 = now_ms();
			}
		}
	}
	st->nitems_in += hdr[1] - hits0;
	t3 = now_ms();
	/* the rows that found no slot: on the CPU */
	if (group_cpu_rows(prog, st, kds_src, &ps, lp_base, npages, total_lps, out + 4, &n_overflow, err, errlen) != 0)
		goto done;
	if (prog->has_str_key)
	{
		const unsigned char *srcs[4] = { ps.baseF, ps.baseL, NULL, NULL };

		if (scan_gtab_task_flush(prog, st, srcs, err, errlen) != 0)
			goto done;
	}
	st->npages_vfs_read += (npages - nloaded) * PAGES_PER_BLOCK;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: group task: %u pages, gpu %.1f ms (lookup %.1f), new-key rows %u, table %u/%u slots, overflow rows %llu (%.1f ms)\n",
				npages, t3 - t1, t2 - t1, nmiss, hdr[0], st->gt->cap, (unsigned long long) n_overflow, now_ms() - t3);
	if (scan_gtab_task_done(vk, prog, st, err, errlen) != 0)
		goto done;
	(void) t0;
	rv = 0;
done:
	free(lp_base);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "GROUP BY aggregation failed");
	return rv;
}


int
scan_exec_group_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
					 agg_state *st, const XpuCommand *task, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped, variant, avariant;
	uint32_t	npages, nloaded, nagg, ngroups_total, total_lps = 0, cap, rec_words, blk_words;
	gpu_vk_pipeline *sp;
	spec_shape	sh;
	spec_keystats ks;
	uint32_t   *lp_base = NULL;
	size_t		out_words;
	const uint32_t *out;
	int			rv = -1, redo = 0;
	const uint32_t *w;
	double		t0 = now_ms(), t1 = 0, t2 = 0, t3 = 0;
	uint64_t	n_overflow = 0;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return -1;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format != KDS_FORMAT_BLOCK)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuPreAgg yet",
				 kds_src->format);
		return -1;
	}
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	npages = kds_src->nitems;
	nloaded = kds_src->block_nloaded;
	if (nloaded > npages ||
		kds_src->block_offset + (size_t) nloaded * BLCKSZ > task->length)
	{
		snprintf(err, errlen, "corrupted block data store");
		return -1;
	}
	if (!prog->words && scan_prog_build(session, kds_src, prog, err, errlen) != 0)
		return -1;
	if (!prog->agg_mode || prog->nkeys == 0)
	{
		snprintf(err, errlen, "not a GROUP BY program");
		return -1;
	}
	if (scan_gtab_forced(st, prog))
		return group_task_gtab(eng, session, prog, st, task, kds_src, err, errlen);
	nagg = prog->nagg;
	w = prog->words;
	variant = (prog->nregs <= 16 ? 0 : 1) + 2 * scan_feat_tier(prog->feat);
	avariant = scan_agg_variant(nagg);
	/* the kernel of this task: the specialized one has its own table capacity and aggregate count (spec_shape) */
	sh.agg_max = ga_variant_na[avariant];
	sh.ga_c = ga_variant_cap[avariant];
	scan_group_keystats(st, prog->nkeys, &ks);
	sp = spec_pipe(eng, prog, SPEC_GAGG, sh.agg_max, sh.ga_c, &ks, &sh, &eng->gagg_pipe[variant][avariant]);
	cap = sh.ga_c;
	rec_words = 2 + 2 * SCAN_GROUP_KEYS_SHARED + 6 * sh.agg_max;
	blk_words = 1 + cap * rec_words;
	if ((sp == &eng->gagg_pipe[variant][avariant] || spec_check_enabled(eng)) &&
		ensure_group_pipeline(eng, variant, avariant, err, errlen) != 0)
		return -1;
	if (npages == 0)
		return 0;

	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto done;
	lp_base = malloc(sizeof(uint32_t) * npages);
	if (!lp_base)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	t1 = now_ms();
	if (gpu_vk_buffer_create(vk, &btab, sizeof(uint32_t) * 2 * npages, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto done;
	}
	have_tab = 1;
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		lp_base[i] = total_lps;
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
		((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
		((uint32_t *) btab.map)[2 * i + 1] = lp_base[i];
	}
	st->nitems_raw += total_lps;
	ngroups_total = (npages + group_ppw(AGG_PAGES_PER_WG) - 1) / group_ppw(AGG_PAGES_PER_WG);
	out_words = 4 + (size_t) ngroups_total * blk_words + total_lps;
	if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
		goto done;
	}
	have_out = 1;
	out = (const uint32_t *) bout.map;
	memset(bout.map, 0, 16);
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto done;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);
	{
		gpu_vk_buffer *bufs[5] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL};
		uint32_t	flags_base = 4 + ngroups_total * blk_words;

		for (uint32_t base = 0; base < ngroups_total; base += SCAN_WG_MAX)
		{
			uint32_t	groups = ngroups_total - base;
			uint32_t	pc[4];

			if (groups > SCAN_WG_MAX)
				groups = SCAN_WG_MAX;
			pc[0] = npages;
			pc[1] = base;
			pc[2] = group_ppw(AGG_PAGES_PER_WG);
			pc[3] = flags_base;
			if (kdispatch(eng, prog, SPEC_GAGG, sh.agg_max, sh.ga_c, &eng->gagg_pipe[variant][avariant], sp, bufs, pc, groups) != 0)
			{
				snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
				goto done;
			}
		}
	}
	if (out[1] != 0)
	{
		describe_kernel_error(out[1], out[2], out[3], err, errlen);
		goto done;
	}
	t2 = now_ms();
	/*
	 * Most rows found no slot in the workgroups' tables (many groups): the CPU would aggregate them one by one. Run the task
	 * again on the session-wide table (gtab.slang) and keep using it. (The groups so far, if any, stay in the session's groups.)
	 */
	if (scan_gtab_switch_allowed() && !scan_prog_has_float_agg(prog))
	{
		const uint32_t *flags = out + 4 + (size_t) ngroups_total * blk_words;
		uint64_t	nov = 0, nhits = 0;

		for (uint32_t i = 0; i < total_lps; i++)
			nov += (flags[i] == 1);
		for (uint32_t wg = 0; wg < ngroups_total; wg++)
			nhits += out[4 + (size_t) wg * blk_words];
		if (nov > 1000 && nov * 16 > nhits)
		{
			st->nitems_raw -= total_lps;
			redo = 1;
			rv = 0;
			goto done;
		}
	}
	/* merge the workgroups' tables */
	if (group_merge_blocks(prog, st, out, ngroups_total, cap, rec_words, blk_words) != 0)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	t3 = now_ms();
	/* the rows that found no slot: on the CPU */
	if (group_cpu_rows(prog, st, kds_src, &ps, lp_base, npages, total_lps,
					   out + 4 + (size_t) ngroups_total * blk_words, &n_overflow, err, errlen) != 0)
		goto done;
	st->npages_vfs_read += (npages - nloaded) * PAGES_PER_BLOCK;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: group task: %u pages, gpu %.1f ms, merge %.1f ms, overflow rows %llu (%.1f ms), groups so far %u\n",
				npages, t2 - t1, t3 - t2, (unsigned long long) n_overflow, now_ms() - t3,
				st->groups ? st->groups->nents : 0);
	rv = 0;
done:
	free(lp_base);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "GROUP BY aggregation failed");
	if (redo)
		return group_task_gtab(eng, session, prog, st, task, kds_src, err, errlen);
	return rv;
}

/*
 * The reply to XpuTaskFinal: the partial aggregate rows in the format PostgreSQL's Agg node
 * combines (kagg_state__* of xpu_common.h in bytea; counts as int8): one row without GROUP BY,
 * one row per group with it.
 */
/*
 * 'order' (GPU-Sort): the groups to return, in this order, nrows_out of them (NULL: all, in table order).
 */
static XpuCommand *
agg_final_build(const XpuCommand *session_cmd, const scan_prog *prog,
				const agg_state *st, const uint32_t *order, uint32_t nrows_out,
				size_t *p_len, char *err, size_t errlen)
{
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	const kern_expression *kagg = SESSION_KEXP_GROUPBY_ACTIONS(session);
	uint32_t	ncols = kagg ? kagg->u.pagg.nattrs : 0;
	int			grouped = (session->xpucode_groupby_keyhash != 0);
	uint32_t	nrows = grouped ? (st->groups ? st->groups->nents : 0) : 1;
	col_value	cols[SCAN_AGG_MAX];
	size_t		tupsz = sizeof(kern_tupitem) + 64 + SCAN_AGG_MAX * 48;
	unsigned char *tup;
	size_t		head_sz, resp_sz, cap, usage = 0, chunk_len;
	XpuCommand *resp;
	kern_data_store *kds;
	uint64_t   *rowindex;

	if (order)
		nrows = nrows_out;
	if (!kds_dst_head || !kagg || ncols > SCAN_AGG_MAX || kds_dst_head->ncols < ncols)
	{
		snprintf(err, errlen, "the result store has %u columns, %u are needed",
				 kds_dst_head ? kds_dst_head->ncols : 0, ncols);
		return NULL;
	}
	head_sz = KDS_HEAD_LENGTH(kds_dst_head);
	resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
	if (st->sdict && prog && prog->has_str_key)
		tupsz += (size_t) SCAN_GROUP_KEYS * (st->sdict->maxlen + 8);	/* the string keys of a row */
	cap = resp_sz + head_sz + MAXALIGN(sizeof(uint64_t) * nrows) + (size_t) nrows * 96 + 4096;
	if (st->sdict && prog && prog->has_str_key)
		cap += (size_t) nrows * (st->sdict->len / (st->sdict->nents ? st->sdict->nents : 1) + 8);
	resp = calloc(1, cap);
	tup = malloc(tupsz);
	if (!resp || !tup)
	{
		free(resp);
		free(tup);
		snprintf(err, errlen, "out of memory");
		return NULL;
	}
	rowindex = (uint64_t *)((char *) resp + resp_sz + head_sz);
	for (uint32_t r = 0; r < nrows; r++)
	{
		int			sz;
		size_t		idx_sz = MAXALIGN(sizeof(uint64_t) * nrows);
		const group_entry *g = (grouped ? GENT(st->groups, order ? order[r] : r) : NULL);

		for (uint32_t j = 0; j < ncols; j++)
		{
			uint32_t	action = kagg->u.pagg.desc[j].action;

			if (action == KAGG_ACTION__VREF && grouped && prog && prog->words)
			{
				int			k = prog->key_of_col[j];
				const kern_colmeta *cm = &kds_dst_head->colmeta[j];

				memset(&cols[j], 0, sizeof(cols[j]));
				if (k >= 0 && prog->key_str[k] && cm->attlen == -1)
				{
					/* a string key: the value the dictionary holds under the group's id */
					if (g->knull & (1u << k))
						cols[j].isnull = 1;
					else
						cols[j].ext = sd_get(st, g->key[k], &cols[j].len);
					continue;
				}
				if (k >= 0 && prog->key_scale[k] >= 0 && cm->attlen == -1)
				{
					/* a numeric key: { unscaled, scale } formed as PostgreSQL stores it */
					if (g->knull & (1u << k))
						cols[j].isnull = 1;
					else
						cols[j].len = scan_vm_numeric_encode(g->key[k], (uint32_t) prog->key_scale[k], cols[j].data);
					continue;
				}
				if (k < 0 || (cm->attlen != 1 && cm->attlen != 2 && cm->attlen != 4 && cm->attlen != 8))
				{
					snprintf(err, errlen, "GROUP BY column %u has an unsupported result type", j + 1);
					free(resp); free(tup);
					return NULL;
				}
				if (g->knull & (1u << k))
					cols[j].isnull = 1;
				else
				{
					int64_t		v = g->key[k];

					memcpy(cols[j].data, &v, 8);		/* little endian: the low bytes */
					cols[j].len = cm->attlen;
				}
			}
			else
			{
				agg_acc		acc, *pacc;

				if (grouped)
				{
					int		a = prog->agg_of_col[j];

					pacc = &g->acc[a < 0 ? 0 : a];
				}
				else
				{
					memset(&acc, 0, sizeof(acc));
					acc.n = st->n[j];
					acc.sum = st->sum[j];
					acc.val = st->val[j];
					acc.fovf = st->fovf[j];
					pacc = &acc;
				}
				if (agg_col_value(action, (prog && prog->words && prog->agg_of_col[j] >= 0) ? prog->agg_scale[(int) prog->agg_of_col[j]] : -1, pacc, &cols[j], err, errlen) != 0)
				{
					free(resp); free(tup);
					return NULL;
				}
			}
		}
		/* append the tuple, growing the buffer if necessary */
		if (resp_sz + head_sz + idx_sz + usage + tupsz > cap)
		{
			size_t		ncap = cap * 2 + tupsz;
			XpuCommand *nr = realloc(resp, ncap);

			if (!nr)
			{
				snprintf(err, errlen, "out of memory");
				free(resp); free(tup);
				return NULL;
			}
			resp = nr;
			cap = ncap;
			rowindex = (uint64_t *)((char *) resp + resp_sz + head_sz);
		}
		sz = form_partial_tuple(kds_dst_head, cols, ncols, tup, tupsz, r, err, errlen);
		if (sz < 0)
		{
			free(resp); free(tup);
			return NULL;
		}
		memcpy((char *) resp + resp_sz + head_sz + idx_sz + usage, tup, sz);
		rowindex[r] = usage;				/* forward position, fixed up below */
		usage += MAXALIGN(sz);
	}
	chunk_len = head_sz + MAXALIGN(sizeof(uint64_t) * nrows) + usage;
	kds = (kern_data_store *)((char *) resp + resp_sz);
	memcpy(kds, kds_dst_head, head_sz);
	kds->length = chunk_len;
	kds->usage = usage;
	kds->nitems = nrows;
	kds->format = KDS_FORMAT_ROW;
	kds->hash_nslots = 0;
	for (uint32_t r = 0; r < nrows; r++)
		rowindex[r] = usage - rowindex[r];		/* distance from the end of the chunk */
	resp->magic = XpuCommandMagicNumber;
	resp->tag = XpuCommandTag__Success;
	resp->length = resp_sz + chunk_len;
	resp->u.results.scan_relidx = -1;
	resp->u.results.chunks_offset = resp_sz;
	resp->u.results.chunks_nitems = (nrows > 0 ? 1 : 0);
	resp->u.results.final_plan_task = true;
	resp->u.results.final_nitems = nrows;
	resp->u.results.final_usage = usage;
	resp->u.results.final_total = chunk_len;
	if (nrows == 0)
		resp->length = resp_sz;
	if (st->rck.nitems > 0)
	{
		/* the rows the backend evaluates (strings the GPU could not read) */
		size_t		csz = rck_chunk_size(kds_dst_head, &st->rck);
		XpuCommand *nr = realloc(resp, resp->length + csz);

		if (!nr)
		{
			snprintf(err, errlen, "out of memory");
			free(resp); free(tup);
			return NULL;
		}
		resp = nr;
		rck_chunk_fill((kern_data_store *)((char *) resp + resp->length), kds_dst_head, &st->rck);
		resp->length += csz;
		resp->u.results.chunks_nitems++;
	}
	*p_len = resp->length;
	free(tup);
	return resp;
}

/*
 * GPU-Sort of the groups: the session carries the sort keys (xpucode_gpusort_keydesc). Builds the key
 * words of every group from its keys and aggregate states (the final value of a sum/min/max/count is
 * known here because this session saw every row: the plan has no Aggregate node above), sorts, and
 * applies the top-N cut or the window-rank filter. Returns the group numbers in output order.
 */
static int
group_sort_order(scan_engine *eng, const kern_session_info *session, const scan_prog *prog,
				 const kern_expression *kagg, const struct group_table *gt,
				 uint32_t **p_order, uint32_t *p_nout, char *err, size_t errlen)
{
	const kern_expression *ksort = SESSION_KEXP_GPUSORT_KEYDESC(session);
	uint32_t	ncols = kagg->u.pagg.nattrs;
	uint32_t	n = gt->nents;
	gpusort_spec spec;
	uint32_t   *keys;
	int			used_gpu = 0, rv;

	if (ksort->u.sort.nkeys < 1 || ksort->u.sort.nkeys > GPUSORT_MAX_KEYS)
	{
		snprintf(err, errlen, "GPU-Sort supports 1 to %d sort keys, the plan has %d",
				 GPUSORT_MAX_KEYS, (int) ksort->u.sort.nkeys);
		return -1;
	}
	memset(&spec, 0, sizeof(spec));
	spec.nkeys = ksort->u.sort.nkeys;
	for (int s = 0; s < spec.nkeys; s++)
	{
		const kern_sortkey_desc *d = &ksort->u.sort.desc[s];

		spec.key[s].asc = d->order_asc;
		spec.key[s].nulls_first = d->nulls_first;
		spec.key[s].wide = (d->kind == KSORT_KEY_KIND__PSUM_INT64 ||
							d->kind == KSORT_KEY_KIND__PSUM_NUMERIC);
	}
	gpusort_spec_layout(&spec);
	spec.window_func = ksort->u.sort.window_rank_func;
	spec.window_limit = ksort->u.sort.window_rank_limit;
	spec.partby_nkeys = ksort->u.sort.window_partby_nkeys;
	spec.orderby_nkeys = ksort->u.sort.window_orderby_nkeys;
	spec.limit = session->gpusort_limit_count;
	if (spec.window_func &&
		spec.partby_nkeys + spec.orderby_nkeys > spec.nkeys)
	{
		snprintf(err, errlen, "GPU-Sort: the window keys exceed the sort keys");
		return -1;
	}
	keys = malloc(sizeof(uint32_t) * spec.nwords * (n ? n : 1));
	if (!keys)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (uint32_t r = 0; r < n; r++)
	{
		const group_entry *g = GENT(gt, r);
		uint32_t   *row = keys + (size_t) r * spec.nwords;

		for (int s = 0; s < spec.nkeys; s++)
		{
			const kern_sortkey_desc *d = &ksort->u.sort.desc[s];
			uint32_t	j = (uint32_t) d->src_anum - 1;
			uint32_t	action;
			int			isnull = 0;
			__int128	v = 0;
			int			a;

			if (d->src_anum < 1 || j >= ncols)
			{
				snprintf(err, errlen, "GPU-Sort key %d refers to column %u of a %u-column result",
						 s + 1, (unsigned) d->src_anum, ncols);
				free(keys);
				return -1;
			}
			action = kagg->u.pagg.desc[j].action;
			a = prog->agg_of_col[j];
			if (d->kind == KSORT_KEY_KIND__VREF && action == KAGG_ACTION__VREF && prog->key_of_col[j] >= 0)
			{
				int			k = prog->key_of_col[j];

				if (g->knull & (1u << k))
					isnull = 1;
				else
					v = g->key[k];
			}
			else if (d->kind == KSORT_KEY_KIND__VREF &&
					 (action == KAGG_ACTION__NROWS_ANY || action == KAGG_ACTION__NROWS_COND) && a >= 0)
				v = (__int128) g->acc[a].n;
			else if ((d->kind == KSORT_KEY_KIND__PSUM_INT64 || d->kind == KSORT_KEY_KIND__PSUM_NUMERIC) &&
					 (action == KAGG_ACTION__PSUM_INT || action == KAGG_ACTION__PSUM_INT64) && a >= 0)
			{
				if (g->acc[a].n == 0)
					isnull = 1;
				else
					v = g->acc[a].sum;
			}
			else if (d->kind == KSORT_KEY_KIND__PMINMAX_INT64 &&
					 (action == KAGG_ACTION__PMIN_INT32 || action == KAGG_ACTION__PMIN_INT64 ||
					  action == KAGG_ACTION__PMAX_INT32 || action == KAGG_ACTION__PMAX_INT64) && a >= 0)
			{
				if (!g->acc[a].n)
					isnull = 1;
				else
					v = g->acc[a].val;
			}
			else
			{
				snprintf(err, errlen, "GPU-Sort does not support sort key kind %d on result column %u (action %u)",
						 (int) d->kind, j + 1, action);
				free(keys);
				return -1;
			}
			gpusort_encode(&spec, s, row, isnull, v);
		}
	}
	rv = gpusort_run(eng, &spec, keys, n, p_order, p_nout, &used_gpu, err, errlen);
	free(keys);
	return rv;
}

/*
 * The reply to XpuTaskFinal of a GpuPreAgg with GPU-Sort: the groups, sorted and cut. The backend's plan
 * has no Aggregate above, so rows the backend would evaluate itself (rechecks: compressed or external
 * values, numeric overflow) cannot be combined: the query fails with a message instead of returning
 * groups that are wrong or duplicated.
 */
XpuCommand *
scan_exec_agg_final_sorted(scan_engine *eng, const XpuCommand *session_cmd, const scan_prog *prog,
						   const agg_state *st, size_t *p_len, char *err, size_t errlen)
{
	const kern_session_info *session = &session_cmd->u.session;
	const kern_expression *kagg = SESSION_KEXP_GROUPBY_ACTIONS(session);
	uint32_t   *order = NULL;
	uint32_t	nout = 0;
	XpuCommand *resp;

	if (!session->xpucode_gpusort_keydesc || !session->xpucode_groupby_keyhash || !kagg ||
		!st->groups || st->groups->nents == 0)
		return agg_final_build(session_cmd, prog, st, NULL, 0, p_len, err, errlen);
	if (st->rck.nitems > 0)
	{
		snprintf(err, errlen, "GPU-Sort cannot combine %u rows that the CPU has to evaluate "
				 "(compressed or external values); run the query with pg_strom.enable_gpusort = off",
				 st->rck.nitems);
		return NULL;
	}
	if (group_sort_order(eng, session, prog, kagg, st->groups, &order, &nout, err, errlen) != 0)
		return NULL;
	resp = agg_final_build(session_cmd, prog, st, order, nout, p_len, err, errlen);
	free(order);
	return resp;
}

XpuCommand *
scan_exec_agg_final(const XpuCommand *session_cmd, const scan_prog *prog,
					const agg_state *st, size_t *p_len, char *err, size_t errlen)
{
	return agg_final_build(session_cmd, prog, st, NULL, 0, p_len, err, errlen);
}

/* ----------------------------------------------------------------
 * GpuJoin
 * ----------------------------------------------------------------
 */
/*
 * RIGHT/FULL joins: the backend emits the inner rows that found no partner when the scan is over; it
 * reads a map of one byte per inner row in the shared buffer (right_outer, ojmap_offset). The kernel
 * flags matched rows in a per-task word map, and the task merges it into the shared bytes (only ever
 * 0 -> 1, so tasks of several workers may merge at the same time). Maps the writable part here.
 */
static int
join_open_ojmap(join_inner *ji, const char *name, const kern_multirels *kmrels, off_t file_size,
				char *err, size_t errlen)
{
	uint64_t	lo = UINT64_MAX, hi = 0;
	size_t		pg = (size_t) sysconf(_SC_PAGESIZE), base, len;
	int			fd;

	for (int i = 0; i < ji->nrels; i++)
	{
		uint64_t	off, end;

		if (!kmrels->chunks[i].right_outer)
			continue;
		off = kmrels->chunks[i].ojmap_offset;
		end = off + ji->rels[i].kds->nitems;
		if (off == 0 || end > (uint64_t) file_size)
		{
			snprintf(err, errlen, "the outer-join map of inner relation %d is outside the buffer", i + 1);
			return -1;
		}
		if (off < lo)
			lo = off;
		if (end > hi)
			hi = end;
	}
	if (hi == 0)
		return 0;
	fd = shm_open(name, O_RDWR, 0);
	if (fd < 0)
	{
		snprintf(err, errlen, "cannot open the inner buffer '%s' for writing: %s", name, strerror(errno));
		return -1;
	}
	base = (size_t) lo & ~(pg - 1);
	len = (((size_t) hi - base) + pg - 1) & ~(pg - 1);
	ji->ojmap = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t) base);
	close(fd);
	if (ji->ojmap == MAP_FAILED)
	{
		ji->ojmap = NULL;
		snprintf(err, errlen, "cannot map the outer-join map: %s", strerror(errno));
		return -1;
	}
	ji->ojmaplen = len;
	for (int i = 0; i < ji->nrels; i++)
		if (kmrels->chunks[i].right_outer)
			ji->oj[i + 1] = (uint8_t *) ji->ojmap + (kmrels->chunks[i].ojmap_offset - base);
	return 0;
}

/* merge the flags a task set (its output buffer's map, which starts at word `base`) into the backend's map */
static void
join_merge_ojmap(const scan_prog *prog, const join_inner *ji, const uint32_t *out, size_t base)
{
	for (int d = 1; d <= prog->ndepth; d++)
	{
		volatile uint8_t *dst = ji->oj[d];
		const uint32_t *src;

		if (prog->oj_off[d] == 0xffffffffU || !dst)
			continue;
		src = out + base + prog->oj_off[d];
		for (uint32_t r = 0; r < prog->oj_nitems[d]; r++)
			if (src[r])
				dst[r] = 1;
	}
	__sync_synchronize();
}

int
scan_exec_join_open(scan_engine *eng, const XpuCommand *session_cmd, join_inner *ji,
					char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	char		name[64];
	int			fd;
	struct stat	st;
	size_t		align, maplen;
	void	   *map;
	const kern_multirels *kmrels;
	const kern_data_store *kds;
	uint64_t	kds_offset;

	memset(ji, 0, sizeof(*ji));
	if (session->join_inner_handle == 0)
	{
		snprintf(err, errlen, "the join has no inner buffer");
		return -1;
	}
#ifdef __APPLE__
	snprintf(name, sizeof(name), "/pgsb.%u.%x", session->pgsql_port_number, session->join_inner_handle);
#else
	snprintf(name, sizeof(name), "/.pgstrom_shmbuf_%u_%d", session->pgsql_port_number,
			 (int) session->join_inner_handle);
#endif
	fd = shm_open(name, O_RDONLY, 0);
	if (fd < 0)
	{
		snprintf(err, errlen, "cannot open the inner buffer '%s': %s", name, strerror(errno));
		return -1;
	}
	if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(kern_multirels))
	{
		snprintf(err, errlen, "the inner buffer '%s' is empty", name);
		close(fd);
		return -1;
	}
	align = vk->host_import_alignment ? vk->host_import_alignment : 4096;
	if ((size_t) sysconf(_SC_PAGESIZE) > align)
		align = sysconf(_SC_PAGESIZE);
	maplen = ((size_t) st.st_size + align - 1) & ~(align - 1);
	map = mmap(NULL, maplen, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (map == MAP_FAILED)
	{
		snprintf(err, errlen, "cannot map the inner buffer: %s", strerror(errno));
		return -1;
	}
	kmrels = (const kern_multirels *) map;
	if (kmrels->num_inner_rels < 1 || kmrels->num_inner_rels > SCAN_JOIN_MAXDEPTH ||
		kmrels->length > (size_t) st.st_size || kmrels->kbuf_part_offset != 0)
	{
		snprintf(err, errlen, "the inner buffer holds %u relations (1 to %d are supported)",
				 kmrels->num_inner_rels, SCAN_JOIN_MAXDEPTH);
		munmap(map, maplen);
		return -1;
	}
	ji->nrels = kmrels->num_inner_rels;
	for (int i = 0; i < ji->nrels; i++)
	{
		const kern_buffer_partitions *dummy = NULL;
		uint64_t	kds_offset = kmrels->chunks[i].kds_offset;
		const kern_data_store *kds = (const kern_data_store *)((const char *) map + kds_offset);

		(void) dummy;
		if (kmrels->chunks[i].is_nestloop || kmrels->chunks[i].pinned_buffer)
		{
			snprintf(err, errlen, "only hash joins and GiST-index joins are implemented by the Vulkan service");
			munmap(map, maplen);
			return -1;
		}
		if (kds_offset == 0 || kds_offset + sizeof(kern_data_store) > (uint64_t) st.st_size ||
			kds_offset + kds->length > (uint64_t) st.st_size ||
			kds->format != KDS_FORMAT_HASH || kds->hash_nslots == 0 || kds_offset + kds->length >= 0xffffffffULL)
		{
			snprintf(err, errlen, "inner relation %d is not a hash store the service can use", i + 1);
			munmap(map, maplen);
			return -1;
		}
		if (kmrels->chunks[i].gist_offset != 0)
		{
			uint64_t	goff = kmrels->chunks[i].gist_offset;
			const kern_data_store *kg = (const kern_data_store *)((const char *) map + goff);

			if (goff + sizeof(kern_data_store) > (uint64_t) st.st_size || kg->format != KDS_FORMAT_BLOCK ||
				goff + kg->block_offset + (uint64_t) kg->nitems * BLCKSZ > (uint64_t) st.st_size ||
				kg->nitems == 0 || kg->nitems >= (1U << 21) - 1 || goff + kg->length >= 0xffffffffULL ||
				kg->block_offset % 4 != 0)
			{
				snprintf(err, errlen, "inner relation %d has no index the service can walk", i + 1);
				munmap(map, maplen);
				return -1;
			}
			ji->rels[i].gist_page0 = (uint32_t)(goff + kg->block_offset);
			ji->rels[i].gist_nblocks = kg->nitems;
		}
		ji->rels[i].kds = kds;
		ji->rels[i].kds_end = (uint32_t)(kds_offset + kds->length);
		ji->rels[i].hslot_word = (uint32_t)((kds_offset + KDS_HEAD_LENGTH(kds)) / 4);
		ji->rels[i].nslots = kds->hash_nslots;
		ji->rels[i].left_outer = kmrels->chunks[i].left_outer ? 1 : 0;
		ji->rels[i].semi = kmrels->chunks[i].is_semi ? 1 : 0;
		ji->rels[i].anti = kmrels->chunks[i].is_anti ? 1 : 0;
		ji->rels[i].right_outer = kmrels->chunks[i].right_outer ? 1 : 0;
	}
	if (join_open_ojmap(ji, name, kmrels, st.st_size, err, errlen) != 0)
	{
		munmap(map, maplen);
		return -1;
	}
	if (maplen > vk->limits.maxStorageBufferRange ||
		gpu_vk_buffer_import_host(vk, &ji->buf, map, maplen) != 0)
	{
		/* copy instead */
		if (gpu_vk_buffer_create(vk, &ji->buf, maplen, 0) != 0)
		{
			snprintf(err, errlen, "cannot allocate %zu bytes for the inner buffer: %s", maplen, vk->err);
			if (ji->ojmap)
				munmap(ji->ojmap, ji->ojmaplen);
			munmap(map, maplen);
			return -1;
		}
		memcpy(ji->buf.map, map, st.st_size);
		ji->own_buf = 1;
	}
	ji->map = map;
	ji->maplen = maplen;
	ji->ready = 1;
	return 0;
}

void
join_inner_free(scan_engine *eng, join_inner *ji)
{
	if (!ji->ready)
		return;
	gpu_vk_buffer_destroy(eng->vk, &ji->buf);
	if (ji->ojmap)
		munmap(ji->ojmap, ji->ojmaplen);
	munmap(ji->map, ji->maplen);
	memset(ji, 0, sizeof(*ji));
}

/* the VM join pipeline of a variant (registers 8/16/32, + 3 * tier; scan_join_variant3), created on first use */
static int
ensure_join_pipeline(scan_engine *eng, int variant, char *err, size_t errlen)
{
	static const struct { const unsigned char *spv; const unsigned int *len; } jtab[] = {
		{kspv_gpujoin8, &kspv_gpujoin8_len}, {kspv_gpujoin16, &kspv_gpujoin16_len}, {kspv_gpujoin32, &kspv_gpujoin32_len},
		{kspv_gpujoin8_lite, &kspv_gpujoin8_lite_len}, {kspv_gpujoin16_lite, &kspv_gpujoin16_lite_len},
		{kspv_gpujoin32_lite, &kspv_gpujoin32_lite_len},
		{kspv_gpujoin8_nogeom, &kspv_gpujoin8_nogeom_len}, {kspv_gpujoin16_nogeom, &kspv_gpujoin16_nogeom_len},
		{kspv_gpujoin32_nogeom, &kspv_gpujoin32_nogeom_len},
		{kspv_gpujoin16_gpip, &kspv_gpujoin16_gpip_len}, {kspv_gpujoin16_gpip, &kspv_gpujoin16_gpip_len}, {kspv_gpujoin32_gpip, &kspv_gpujoin32_gpip_len},
		{kspv_gpujoin16_gdw, &kspv_gpujoin16_gdw_len}, {kspv_gpujoin16_gdw, &kspv_gpujoin16_gdw_len}, {kspv_gpujoin32_gdw, &kspv_gpujoin32_gdw_len},
		{kspv_gpujoin16_gwalk, &kspv_gpujoin16_gwalk_len}, {kspv_gpujoin16_gwalk, &kspv_gpujoin16_gwalk_len}, {kspv_gpujoin32_gwalk, &kspv_gpujoin32_gwalk_len},
	};
	_Static_assert(sizeof(jtab) / sizeof(jtab[0]) == SCAN_JV3, "jtab: one entry per register variant and tier");

	if (!eng->join_pipe_ready[variant])
	{
		if (gpu_vk_pipeline_create(eng->vk, &eng->join_pipe[variant], jtab[variant].spv, *jtab[variant].len, 6, 20) != 0)
		{
			snprintf(err, errlen, "cannot create the join pipeline: %s", eng->vk->err);
			return -1;
		}
		eng->join_pipe_ready[variant] = 1;
	}
	return 0;
}

XpuCommand *
scan_exec_join_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
					join_inner *ji, const XpuCommand *task,
					size_t *p_len, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped, variant;
	gpu_vk_pipeline *sp = NULL;
	uint32_t	npages, nloaded, total_lps = 0, ncomp, rec_words, cap, nmatch = 0, nout = 0;
	size_t		out_words;
	const uint32_t *out = NULL;
	XpuCommand *resp = NULL;
	form_ctx	fc;
	double		t0 = now_ms(), t1 = 0, t2 = 0;
	recheck_buf	rck = {0};
	uint64_t   *pz = NULL;				/* the outer rows the join program could not decide: (page << 32) | lp, sorted */
	uint32_t	npz = 0;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (!ji->ready)
	{
		snprintf(err, errlen, "the inner buffer is not attached");
		return NULL;
	}
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return NULL;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format != KDS_FORMAT_BLOCK)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuJoin yet", kds_src->format);
		return NULL;
	}
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	npages = kds_src->nitems;
	nloaded = kds_src->block_nloaded;
	if (nloaded > npages || kds_src->block_offset + (size_t) nloaded * BLCKSZ > task->length)
	{
		snprintf(err, errlen, "corrupted block data store");
		return NULL;
	}
	if (!prog->words &&
		scan_prog_build_join(session, kds_src, ji->rels, ji->nrels, prog, err, errlen) != 0)
		return NULL;
	if (!prog->join_mode)
	{
		snprintf(err, errlen, "not a join program");
		return NULL;
	}
	ncomp = prog->ncomp;
	rec_words = 2 + prog->ndepth + 3 * ncomp;
	/* variant: 0-2 the registers (8/16/32), + 3 * tier (scan_caps.h) */
	variant = scan_join_variant3(prog->nregs, scan_join_tier(prog->feat));
	sp = spec_pipe(eng, prog, SPEC_JOIN, 0, 0, NULL, NULL, &eng->join_pipe[variant]);
	if ((sp == &eng->join_pipe[variant] || spec_check_enabled(eng)) && !eng->join_pipe_ready[variant])
	{
		if (ensure_join_pipeline(eng, variant, err, errlen) != 0)
			return NULL;
	}
	if (npages == 0)
	{
		/* an empty task: an empty reply */
		resp = calloc(1, MAXALIGN(offsetof(XpuCommand, u.results.stats[0])));
		if (!resp)
			return NULL;
		resp->magic = XpuCommandMagicNumber;
		resp->tag = XpuCommandTag__Success;
		resp->length = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
		resp->u.results.scan_relidx = task->u.task.scan_relidx;
		*p_len = resp->length;
		return resp;
	}
	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto error;
	t1 = now_ms();
	if (gpu_vk_buffer_create(vk, &btab, sizeof(uint32_t) * 2 * npages, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto error;
	}
	have_tab = 1;
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
		((uint32_t *) btab.map)[2 * i + 1] = 0;
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
	}
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto error;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);

	/* run; if there are more matches than the output holds, run again with room for all */
	cap = (total_lps > 4096 ? total_lps : 4096);
	for (int attempt = 0; attempt < 2; attempt++)
	{
		gpu_vk_buffer *bufs[6] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL, &ji->buf};
		uint32_t	pc[5];

		out_words = 4 + (size_t) prog->oj_words + (size_t) cap * rec_words;
		if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
		{
			snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
			goto error;
		}
		have_out = 1;
		out = (const uint32_t *) bout.map;
		memset(bout.map, 0, 16 + (size_t) prog->oj_words * 4);
		for (uint32_t base = 0; base < npages; base += SCAN_WG_MAX)
		{
			uint32_t	groups = npages - base;

			if (groups > SCAN_WG_MAX)
				groups = SCAN_WG_MAX;
			pc[0] = npages;
			pc[1] = base;
			pc[2] = BLCKSZ;
			pc[3] = cap;
			pc[4] = 0;
			if (kdispatch(eng, prog, SPEC_JOIN, 0, 0, &eng->join_pipe[variant], sp, bufs, pc, groups) != 0)
			{
				snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
				goto error;
			}
		}
		if (out[1] != 0)
		{
			describe_kernel_error(out[1], out[2], out[3], err, errlen);
			goto error;
		}
		nmatch = out[0];
		if (nmatch <= cap)
			break;
		cap = nmatch;						/* the exact number of matches */
		gpu_vk_buffer_destroy(vk, &bout);
		have_out = 0;
	}
	if (nmatch > cap)
	{
		snprintf(err, errlen, "the join produced more matches than the output could hold");
		goto error;
	}
	t2 = now_ms();
	if (prog->oj_words > 0)
		join_merge_ojmap(prog, ji, out, 4);

	/* the result chunk, in one pass */
	memset(&fc, 0, sizeof(fc));
	fc.kds_dst_head = kds_dst_head;
	fc.kds_src = kds_src;
	fc.prog = prog;
	fc.maxres = prog->max_src_resno;
	fc.ndepth = prog->ndepth;
	for (int d = 1; d <= prog->ndepth; d++)
	{
		fc.kds_in[d] = ji->rels[d - 1].kds;
		fc.maxres_inner[d] = prog->max_inner_resno[d];
	}
	{
		size_t		head_sz = KDS_HEAD_LENGTH(kds_dst_head);
		size_t		resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
		size_t		idx_sz = MAXALIGN(sizeof(uint64_t) * nmatch);
		size_t		cap_b = resp_sz + head_sz + idx_sz + (size_t) nmatch * 64 + 4096;
		size_t		usage = 0, chunk_len;
		uint64_t   *rowindex;
		kern_data_store *kds;

		resp = calloc(1, cap_b);
		if (!resp)
		{
			snprintf(err, errlen, "out of memory");
			goto error;
		}
		/*
		 * Poison records (item word of depth 1 = 0xfffffffe): the kernel could not decide the outer row (a geometry
		 * it cannot read, an instruction that needs the CPU). Whatever matches it appended before are dropped and the
		 * outer tuple goes back to the backend, which joins it on the CPU (KDS_FORMAT_RECHECK).
		 */
		for (uint32_t r = 0; r < nmatch; r++)
		{
			const uint32_t *rec = &out[4 + (size_t) prog->oj_words + (size_t) r * rec_words];

			if (rec[2] != 0xfffffffeu)
				continue;
			if (rec[0] >= npages)
			{
				snprintf(err, errlen, "corrupt match record");
				goto error;
			}
			if (!pz)
			{
				pz = malloc(sizeof(uint64_t) * (size_t) nmatch);
				if (!pz)
				{
					snprintf(err, errlen, "out of memory");
					goto error;
				}
			}
			pz[npz++] = ((uint64_t) rec[0] << 32) | rec[1];
		}
		if (npz > 1)
			qsort(pz, npz, sizeof(uint64_t), cmp_u64);
		for (uint32_t i = 0; i < npz; i++)
		{
			uint32_t	page = (uint32_t)(pz[i] >> 32), lp = (uint32_t) pz[i], lpd;
			const unsigned char *pg = PAGE_AT(&ps, page);

			if (i > 0 && pz[i] == pz[i - 1])
				continue;
			memcpy(&lpd, pg + 24 + (size_t) lp * 4, 4);
			if (rck_add(&rck, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[page], lp + 1,
						pg + (lpd & 0x7fff), (lpd >> 17) & 0x7fff) != 0)
			{
				snprintf(err, errlen, "out of memory");
				goto error;
			}
		}
		for (uint32_t r = 0; r < nmatch; r++)
		{
			const uint32_t *rec = &out[4 + (size_t) prog->oj_words + (size_t) r * rec_words];
			uint32_t	page = rec[0], lp = rec[1], lpd;
			const unsigned char *pg = PAGE_AT(&ps, page);
			const unsigned char *inner[SCAN_JOIN_MAXDEPTH + 1];
			int			sz;

			if (page >= npages)
			{
				snprintf(err, errlen, "corrupt match record");
				goto error;
			}
			if (rec[2] == 0xfffffffeu)
				continue;
			if (npz > 0)
			{
				uint64_t	key = ((uint64_t) page << 32) | lp;

				if (bsearch(&key, pz, npz, sizeof(uint64_t), cmp_u64))
					continue;				/* the backend joins this row */
			}
			inner[0] = NULL;
			for (int d = 1; d <= prog->ndepth; d++)
			{
				uint32_t	item = rec[1 + d];

				if (item != 0xffffffffu && item >= ji->maplen)
				{
					snprintf(err, errlen, "corrupt match record");
					goto error;
				}
				inner[d] = (item == 0xffffffffu ? NULL : (const unsigned char *) ji->map + item);
			}
			memcpy(&lpd, pg + 24 + (size_t) lp * 4, 4);
			if (resp_sz + head_sz + idx_sz + usage + (lpd >> 17) + 1024 > cap_b)
			{
				size_t		ncap = cap_b * 2 + (lpd >> 17) + 4096;
				XpuCommand *nr = realloc(resp, ncap);

				if (!nr)
				{
					snprintf(err, errlen, "out of memory");
					goto error;
				}
				resp = nr;
				cap_b = ncap;
			}
			sz = form_tuple(&fc, pg + (lpd & 0x7fff), inner,
							rec + 2 + prog->ndepth, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[page], lp + 1,
							(kern_tupitem *)((char *) resp + resp_sz + head_sz + idx_sz + usage),
							nout, err, errlen);
			if (sz < 0)
				goto error;
			((uint64_t *)((char *) resp + resp_sz + head_sz))[nout] = usage;
			usage += MAXALIGN(sz);
			nout++;
		}
		chunk_len = head_sz + idx_sz + usage;
		kds = (kern_data_store *)((char *) resp + resp_sz);
		memcpy(kds, kds_dst_head, head_sz);
		kds->length = chunk_len;
		kds->usage = usage;
		kds->nitems = nout;
		kds->format = KDS_FORMAT_ROW;
		kds->hash_nslots = 0;
		rowindex = (uint64_t *)((char *) kds + head_sz);
		for (uint32_t r = 0; r < nout; r++)
			rowindex[r] = usage - rowindex[r];
		resp->magic = XpuCommandMagicNumber;
		resp->tag = XpuCommandTag__Success;
		resp->length = resp_sz + chunk_len;
		resp->u.results.scan_relidx = task->u.task.scan_relidx;
		resp->u.results.chunks_offset = resp_sz;
		resp->u.results.chunks_nitems = (nout > 0 ? 1 : 0);
		resp->u.results.nitems_raw = total_lps;
		resp->u.results.nitems_in = total_lps;
		resp->u.results.nitems_out = nout;
		resp->u.results.npages_vfs_read = (npages - nloaded) * PAGES_PER_BLOCK;
		if (nout == 0)
			resp->length = resp_sz;
		if (rck.nitems > 0)
		{
			/* the rows the backend has to join itself, as one more chunk */
			size_t		csz = rck_chunk_size(kds_dst_head, &rck);
			XpuCommand *nr = realloc(resp, resp->length + csz);

			if (!nr)
			{
				snprintf(err, errlen, "out of memory");
				goto error;
			}
			resp = nr;
			rck_chunk_fill((kern_data_store *)((char *) resp + resp->length), kds_dst_head, &rck);
			resp->length += csz;
			resp->u.results.chunks_nitems++;
		}
		*p_len = resp->length;
	}
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: join task: %u pages, pages-in %.1f ms, gpu %.1f ms, results %.1f ms (%u matches)\n",
				npages, t1 - t0, t2 - t1, now_ms() - t2, nmatch);
	free(pz);
	free(rck.buf);
	gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	return resp;

error:
	free(resp);
	free(pz);
	free(rck.buf);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (err[0] == '\0')
		snprintf(err, errlen, "join failed");
	return NULL;
}

/* ----------------------------------------------------------------
 * GpuPreAgg on top of GpuJoin: the join kernel aggregates the joined rows itself (jagg.slang without
 * GROUP BY, jgagg.slang with it), so they never leave the device; the results merge into the
 * session's agg_state exactly like those of a scan.
 * ----------------------------------------------------------------
 */
#define JAGG_PAGES_PER_WG	8U		/* outer pages one such workgroup runs over (without GROUP BY) */
#define JGAGG_PAGES_PER_WG	32U		/* ... with GROUP BY (the group table is set up and written out per workgroup) */
/*
 * A walk of the join tree can be long (an inner side with many equal or NULL hash values makes a chain of thousands of
 * items), and the device is marked lost when one dispatch runs past the watchdog: a join aggregation is therefore
 * dispatched in slices of this many workgroups (the slices add up to the same work).
 */
#define JOINAGG_DISPATCH_GROUPS	128U

static int
ensure_joinagg_pipeline(scan_engine *eng, int grouped, int variant, int avariant, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } jtab[][3] = {
		{{kspv_jagg16_4, &kspv_jagg16_4_len}, {kspv_jagg16_12, &kspv_jagg16_12_len},
		 {kspv_jagg16_12, &kspv_jagg16_12_len}},
		{{kspv_jagg32_4, &kspv_jagg32_4_len}, {kspv_jagg32_12, &kspv_jagg32_12_len},
		 {kspv_jagg32_12, &kspv_jagg32_12_len}},
		{{kspv_jagg16_4_lite, &kspv_jagg16_4_lite_len}, {kspv_jagg16_12_lite, &kspv_jagg16_12_lite_len},
		 {kspv_jagg16_12_lite, &kspv_jagg16_12_lite_len}},
		{{kspv_jagg32_4_lite, &kspv_jagg32_4_lite_len}, {kspv_jagg32_12_lite, &kspv_jagg32_12_lite_len},
		 {kspv_jagg32_12_lite, &kspv_jagg32_12_lite_len}},
		{{kspv_jagg16_4_nogeom, &kspv_jagg16_4_nogeom_len}, {kspv_jagg16_12_nogeom, &kspv_jagg16_12_nogeom_len},
		 {kspv_jagg16_12_nogeom, &kspv_jagg16_12_nogeom_len}},
		{{kspv_jagg32_4_nogeom, &kspv_jagg32_4_nogeom_len}, {kspv_jagg32_12_nogeom, &kspv_jagg32_12_nogeom_len},
		 {kspv_jagg32_12_nogeom, &kspv_jagg32_12_nogeom_len}},
		{{kspv_jagg16_4_gpip, &kspv_jagg16_4_gpip_len}, {kspv_jagg16_12_gpip, &kspv_jagg16_12_gpip_len},
		 {kspv_jagg16_12_gpip, &kspv_jagg16_12_gpip_len}},
		{{kspv_jagg32_4_gpip, &kspv_jagg32_4_gpip_len}, {kspv_jagg32_12_gpip, &kspv_jagg32_12_gpip_len},
		 {kspv_jagg32_12_gpip, &kspv_jagg32_12_gpip_len}},
		{{kspv_jagg16_4_gdw, &kspv_jagg16_4_gdw_len}, {kspv_jagg16_12_gdw, &kspv_jagg16_12_gdw_len},
		 {kspv_jagg16_12_gdw, &kspv_jagg16_12_gdw_len}},
		{{kspv_jagg32_4_gdw, &kspv_jagg32_4_gdw_len}, {kspv_jagg32_12_gdw, &kspv_jagg32_12_gdw_len},
		 {kspv_jagg32_12_gdw, &kspv_jagg32_12_gdw_len}},
		{{kspv_jagg16_4_gwalk, &kspv_jagg16_4_gwalk_len}, {kspv_jagg16_12_gwalk, &kspv_jagg16_12_gwalk_len},
		 {kspv_jagg16_12_gwalk, &kspv_jagg16_12_gwalk_len}},
		{{kspv_jagg32_4_gwalk, &kspv_jagg32_4_gwalk_len}, {kspv_jagg32_12_gwalk, &kspv_jagg32_12_gwalk_len},
		 {kspv_jagg32_12_gwalk, &kspv_jagg32_12_gwalk_len}},
	};
	_Static_assert(sizeof(jtab) / sizeof(jtab[0]) == SCAN_JV2, "jtab: one entry per register variant and tier");
	static const struct { const unsigned char *spv; const unsigned int *len; } gtab[][3] = {
		{{kspv_jgagg16_4, &kspv_jgagg16_4_len}, {kspv_jgagg16_12, &kspv_jgagg16_12_len},
		 {kspv_jgagg16_12, &kspv_jgagg16_12_len}},
		{{kspv_jgagg32_4, &kspv_jgagg32_4_len}, {kspv_jgagg32_12, &kspv_jgagg32_12_len},
		 {kspv_jgagg32_12, &kspv_jgagg32_12_len}},
		{{kspv_jgagg16_4_lite, &kspv_jgagg16_4_lite_len}, {kspv_jgagg16_12_lite, &kspv_jgagg16_12_lite_len},
		 {kspv_jgagg16_12_lite, &kspv_jgagg16_12_lite_len}},
		{{kspv_jgagg32_4_lite, &kspv_jgagg32_4_lite_len}, {kspv_jgagg32_12_lite, &kspv_jgagg32_12_lite_len},
		 {kspv_jgagg32_12_lite, &kspv_jgagg32_12_lite_len}},
		{{kspv_jgagg16_4_nogeom, &kspv_jgagg16_4_nogeom_len}, {kspv_jgagg16_12_nogeom, &kspv_jgagg16_12_nogeom_len},
		 {kspv_jgagg16_12_nogeom, &kspv_jgagg16_12_nogeom_len}},
		{{kspv_jgagg32_4_nogeom, &kspv_jgagg32_4_nogeom_len}, {kspv_jgagg32_12_nogeom, &kspv_jgagg32_12_nogeom_len},
		 {kspv_jgagg32_12_nogeom, &kspv_jgagg32_12_nogeom_len}},
		{{kspv_jgagg16_4_gpip, &kspv_jgagg16_4_gpip_len}, {kspv_jgagg16_12_gpip, &kspv_jgagg16_12_gpip_len},
		 {kspv_jgagg16_12_gpip, &kspv_jgagg16_12_gpip_len}},
		{{kspv_jgagg32_4_gpip, &kspv_jgagg32_4_gpip_len}, {kspv_jgagg32_12_gpip, &kspv_jgagg32_12_gpip_len},
		 {kspv_jgagg32_12_gpip, &kspv_jgagg32_12_gpip_len}},
		{{kspv_jgagg16_4_gdw, &kspv_jgagg16_4_gdw_len}, {kspv_jgagg16_12_gdw, &kspv_jgagg16_12_gdw_len},
		 {kspv_jgagg16_12_gdw, &kspv_jgagg16_12_gdw_len}},
		{{kspv_jgagg32_4_gdw, &kspv_jgagg32_4_gdw_len}, {kspv_jgagg32_12_gdw, &kspv_jgagg32_12_gdw_len},
		 {kspv_jgagg32_12_gdw, &kspv_jgagg32_12_gdw_len}},
		{{kspv_jgagg16_4_gwalk, &kspv_jgagg16_4_gwalk_len}, {kspv_jgagg16_12_gwalk, &kspv_jgagg16_12_gwalk_len},
		 {kspv_jgagg16_12_gwalk, &kspv_jgagg16_12_gwalk_len}},
		{{kspv_jgagg32_4_gwalk, &kspv_jgagg32_4_gwalk_len}, {kspv_jgagg32_12_gwalk, &kspv_jgagg32_12_gwalk_len},
		 {kspv_jgagg32_12_gwalk, &kspv_jgagg32_12_gwalk_len}},
	};
	_Static_assert(sizeof(gtab) / sizeof(gtab[0]) == SCAN_JV2, "gtab: one entry per register variant and tier");
	int		   *ready = (grouped ? &eng->jgagg_pipe_ready[variant][avariant]
							  : &eng->jagg_pipe_ready[variant][avariant]);
	gpu_vk_pipeline *pipe = (grouped ? &eng->jgagg_pipe[variant][avariant]
									 : &eng->jagg_pipe[variant][avariant]);

	if (!*ready)
	{
		const unsigned char *spv = (grouped ? gtab[variant][avariant].spv : jtab[variant][avariant].spv);
		unsigned int len = (grouped ? *gtab[variant][avariant].len : *jtab[variant][avariant].len);

		if (gpu_vk_pipeline_create(vk, pipe, spv, len, 6, 20) != 0)
		{
			snprintf(err, errlen, "cannot create the join aggregation pipeline: %s", vk->err);
			return -1;
		}
		*ready = 1;
	}
	return 0;
}

/* heap tuple `lp` (0-based) of page `page` of the task's page set, as a row for the backend's CPU path */
static int
joinagg_rck_row(const page_set *ps, const kern_data_store *kds_src, uint32_t page, uint32_t lp, recheck_buf *rck)
{
	const unsigned char *pg = PAGE_AT(ps, page);
	uint32_t	lpd;

	memcpy(&lpd, pg + 24 + (size_t) lp * 4, 4);
	return rck_add(rck, ((const BlockNumber *) KDS_BODY_ADDR(kds_src))[page], lp + 1,
				   pg + (lpd & 0x7fff), (lpd >> 17) & 0x7fff);
}

/*
 * The outer rows the aggregating kernel flagged because their scan quals could not be decided on the device
 * (A14: bit = line pointers before the page + lp, in a bitmap of total_lps bits behind the results). They
 * contributed nothing to the aggregates, so they go to the backend as they are. skip: the bitmap of the rows
 * the scout pass has handed over already (NULL: no scout pass).
 */
static int
joinagg_harvest_flags(const page_set *ps, const kern_data_store *kds_src, uint32_t npages, uint32_t total_lps,
					  const uint32_t *ptab, const uint32_t *flags, const uint32_t *skip, recheck_buf *rck)
{
	uint32_t	pg = 0;

	for (uint32_t wi = 0; wi * 32 < total_lps; wi++)
	{
		uint32_t	m = flags[wi];

		if (skip)
			m &= ~skip[wi];
		while (m != 0)
		{
			uint32_t	bi = wi * 32 + (uint32_t) __builtin_ctz(m);

			m &= m - 1;
			if (bi >= total_lps)
				break;
			while (pg + 1 < npages && ptab[2 * (pg + 1) + 1] <= bi)
				pg++;
			if (joinagg_rck_row(ps, kds_src, pg, bi - ptab[2 * pg + 1], rck) != 0)
				return -1;
		}
	}
	return 0;
}

int
scan_exec_joinagg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
					   join_inner *ji, agg_state *st, const XpuCommand *task, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	gpu_vk_buffer bout, bprog, btab;
	page_set	ps;
	int			have_out = 0, have_prog = 0, have_tab = 0, mapped, variant, avariant, grouped, gist = 0;
	uint32_t	npages, nloaded, nagg, ngroups_total, total_lps = 0, ppw;
	uint32_t	cap = 0, rec_words = 0, blk_words = 0, ov_rec = 0, ovf_cap = 0, ovf_count = 0;
	gpu_vk_pipeline *sp, *vmpipe;
	spec_shape	sh;
	spec_keystats ks;
	size_t		out_words, ovf_base = 0, oj_base = 0;
	const uint32_t *out = NULL;
	int			rv = -1;
	double		t0 = now_ms(), t1 = 0, t2 = 0, t3 = 0;
	uint32_t	bm_words = 0, flag_words = 0, flag_base = 0;
	int			scout;

	memset(&ps, 0, sizeof(ps));
	err[0] = '\0';
	if (!ji->ready)
	{
		snprintf(err, errlen, "the inner buffer is not attached");
		return -1;
	}
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return -1;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format != KDS_FORMAT_BLOCK)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuPreAgg over GpuJoin yet",
				 kds_src->format);
		return -1;
	}
	if (task->u.task.kds_src_pathname)
		pathname = (const char *) task + task->u.task.kds_src_pathname;
	if (task->u.task.kds_src_iovec)
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	npages = kds_src->nitems;
	nloaded = kds_src->block_nloaded;
	if (nloaded > npages || kds_src->block_offset + (size_t) nloaded * BLCKSZ > task->length)
	{
		snprintf(err, errlen, "corrupted block data store");
		return -1;
	}
	if (!prog->words &&
		scan_prog_build_join(session, kds_src, ji->rels, ji->nrels, prog, err, errlen) != 0)
		return -1;
	if (!prog->join_mode || !prog->agg_mode)
	{
		snprintf(err, errlen, "not a join aggregation program");
		return -1;
	}
	nagg = prog->nagg;
	grouped = (prog->nkeys > 0);
	if (st->nagg == 0)
	{
		st->nagg = nagg;
		memset(st->n, 0, sizeof(st->n));
		memset(st->sum, 0, sizeof(st->sum));
		memset(st->val, 0, sizeof(st->val));
		memset(st->fovf, 0, sizeof(st->fovf));
	}
	variant = (prog->nregs <= 16 ? 0 : 1) + 2 * scan_join_tier(prog->feat);
	avariant = scan_agg_variant(nagg);
	vmpipe = (grouped ? &eng->jgagg_pipe[variant][avariant] : &eng->jagg_pipe[variant][avariant]);
	sh.agg_max = ga_variant_na[avariant];
	sh.ga_c = ga_variant_cap[avariant];
	scan_group_keystats(st, prog->nkeys, &ks);
	/* a GiST-index join runs on the interpreter (specialize.c refuses it) after a scout pass, see below */
	for (int i = 0; i < ji->nrels; i++)
		if (ji->rels[i].gist_page0 != 0)
			gist = 1;
	if (gist)
		sp = vmpipe;
	else
		sp = spec_pipe(eng, prog, grouped ? SPEC_JGAGG : SPEC_JAGG, sh.agg_max, sh.ga_c, &ks, &sh, vmpipe);
	if (grouped)
	{
		cap = sh.ga_c;
		rec_words = 2 + 2 * SCAN_GROUP_KEYS_SHARED + 6 * sh.agg_max;
		blk_words = 1 + cap * rec_words;
		ov_rec = 1 + 2 * SCAN_GROUP_KEYS_SHARED + 3 * sh.agg_max;
	}
	if ((sp == vmpipe || spec_check_enabled(eng)) &&
		ensure_joinagg_pipeline(eng, grouped, variant, avariant, err, errlen) != 0)
		return -1;
	if (npages == 0)
		return 0;

	if (task_pages_open(eng, &ps, kds_src, pathname, iovec, &mapped, err, errlen) != 0)
		goto done;
	t1 = now_ms();
	for (uint32_t i = 0; i < npages; i++)
	{
		uint16_t	pd_lower;

		memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
		if (pd_lower > 24)
			total_lps += (pd_lower - 24) / 4;
	}
	/*
	 * A program whose walk or projection can poison an outer row (rck_walk: a GiST join, numeric / float4 / jsonb
	 * instructions): the skip bitmap (one bit per line pointer) follows the 2 words per page. A program whose
	 * scan quals alone can recheck needs no scout pass: the kernel flags those rows in a bitmap behind its results.
	 */
	scout = prog->rck_walk;
	if (scout)
		bm_words = (total_lps + 31) / 32;
	else if (prog->rck_outer)
		flag_words = (total_lps + 31) / 32;
	if (gpu_vk_buffer_create(vk, &btab, sizeof(uint32_t) * ((size_t) 2 * npages + bm_words), 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the page table: %s", vk->err);
		goto done;
	}
	have_tab = 1;
	memset(btab.map, 0, sizeof(uint32_t) * ((size_t) 2 * npages + bm_words));
	{
		uint32_t	lps_before = 0;

		for (uint32_t i = 0; i < npages; i++)
		{
			uint16_t	pd_lower;

			memcpy(&pd_lower, PAGE_AT(&ps, i) + 12, 2);
			((uint32_t *) btab.map)[2 * i] = ps.ptab[i];
			((uint32_t *) btab.map)[2 * i + 1] = lps_before;	/* line pointers before this page: the bitmap index */
			if (pd_lower > 24)
				lps_before += (pd_lower - 24) / 4;
		}
	}
	st->nitems_raw += total_lps;
	ppw = (grouped ? group_ppw(JGAGG_PAGES_PER_WG) : JAGG_PAGES_PER_WG);
	ngroups_total = (npages + ppw - 1) / ppw;
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto done;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);

	/*
	 * An outer row the join program cannot decide (a GiST join's geometry the kernel cannot read, an instruction
	 * that needs the CPU: SOP_RECHECK_CODE) is poison. The aggregating kernels cannot take back what they
	 * accumulated for the row's earlier matches, so the plain join kernel runs first as a scout (push constant
	 * word 4 = 1: append the poison records only). Every poisoned outer row goes to the backend as a
	 * KDS_FORMAT_RECHECK row (joined and aggregated on the CPU, combined by the Agg node above) and into the skip
	 * bitmap behind the page table (bit = line pointers before the page + lp), which the aggregating pass
	 * honours (join_walk.inc). Programs that cannot poison (no instruction that may recheck after the scan quals)
	 * skip the pass.
	 */
	if (scout)
	{
		int			jvariant = scan_join_variant3(prog->nregs, scan_join_tier(prog->feat));
		size_t		rw = 2 + prog->ndepth + 3 * prog->ncomp;
		uint32_t	scap = (total_lps < 4096 ? total_lps : 4096);

		if (ensure_join_pipeline(eng, jvariant, err, errlen) != 0)
			goto done;
		for (int attempt = 0; attempt < 2; attempt++)
		{
			gpu_vk_buffer sbout;
			gpu_vk_buffer *sbufs[6] = {&ps.bufF, &sbout, &bprog, &btab, &ps.bufL, &ji->buf};
			size_t		sout_words = 4 + (size_t) prog->oj_words + (size_t) scap * rw;
			const uint32_t *sout;
			uint32_t	spc[5], snrec;

			if (gpu_vk_buffer_create(vk, &sbout, sout_words * 4, 0) != 0)
			{
				snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
				goto done;
			}
			sout = (const uint32_t *) sbout.map;
			memset(sbout.map, 0, 16 + (size_t) prog->oj_words * 4);
			for (uint32_t base = 0; base < npages; base += SCAN_WG_MAX)
			{
				uint32_t	groups = npages - base;

				if (groups > SCAN_WG_MAX)
					groups = SCAN_WG_MAX;
				spc[0] = npages;
				spc[1] = base;
				spc[2] = BLCKSZ;
				spc[3] = scap;				/* record capacity */
				spc[4] = 1;
				if (gpu_vk_dispatch(vk, &eng->join_pipe[jvariant], sbufs, spc, groups, 1, 1, NULL) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					gpu_vk_buffer_destroy(vk, &sbout);
					goto done;
				}
			}
			snrec = sout[0];
			if (sout[1] != 0)
				describe_kernel_error(sout[1], sout[2], sout[3], err, errlen);
			else if (snrec > scap && attempt == 0 && snrec <= total_lps)
			{
				scap = snrec;				/* the exact number of poison records */
				gpu_vk_buffer_destroy(vk, &sbout);
				continue;
			}
			else if (snrec > scap)
				snprintf(err, errlen, "the join scout produced more poison records than line pointers");
			else
			{
				uint32_t   *bitmap = (uint32_t *) btab.map + 2 * (size_t) npages;
				const uint32_t *lpsb = (const uint32_t *) btab.map;

				memset(bitmap, 0, sizeof(uint32_t) * bm_words);
				for (uint32_t r = 0; r < snrec; r++)
				{
					const uint32_t *rec = &sout[4 + (size_t) prog->oj_words + (size_t) r * rw];
					uint32_t	page = rec[0], lp = rec[1], bi;

					if (rec[2] != 0xfffffffeu)
						continue;				/* (only poison records are appended in scout mode) */
					if (page >= npages)
					{
						snprintf(err, errlen, "corrupt poison record");
						break;
					}
					bi = lpsb[2 * page + 1] + lp;
					if (bi >= total_lps)
					{
						snprintf(err, errlen, "corrupt poison record");
						break;
					}
					if ((bitmap[bi >> 5] >> (bi & 31)) & 1u)
						continue;
					bitmap[bi >> 5] |= 1u << (bi & 31);
					if (joinagg_rck_row(&ps, kds_src, page, lp, &st->rck) != 0)
					{
						snprintf(err, errlen, "out of memory");
						break;
					}
				}
			}
			gpu_vk_buffer_destroy(vk, &sbout);
			break;
		}
		if (err[0] != '\0')
			goto done;
	}


	/*
	 * Run. With GROUP BY the rows that found no slot in their workgroup's table are appended to an
	 * overflow list; if it was too small, run again with room for all (nothing is merged before).
	 */
	ovf_cap = 4096;
	for (int attempt = 0; attempt < 4; attempt++)
	{
		gpu_vk_pipeline *pipe = vmpipe;
		gpu_vk_buffer *bufs[6] = {&ps.bufF, &bout, &bprog, &btab, &ps.bufL, &ji->buf};

		if (grouped)
		{
			ovf_base = 4 + (size_t) ngroups_total * blk_words;
			out_words = ovf_base + 1 + (size_t) ovf_cap * ov_rec;
		}
		else
			out_words = 4 + (size_t) ngroups_total * nagg * 6;
		/* RIGHT/FULL joins: the outer-join map follows the results (the kernels flag the matched inner rows) */
		oj_base = out_words;
		out_words += prog->oj_words;
		/* the bitmap of the outer rows whose scan quals the kernel could not decide (program header word 19) */
		flag_base = (uint32_t) out_words;
		out_words += flag_words;
		if (out_words > 0xffffffffULL / 8 ||
			gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
		{
			snprintf(err, errlen, "cannot allocate the output buffer (%zu words): %s", out_words, vk->err);
			goto done;
		}
		have_out = 1;
		out = (const uint32_t *) bout.map;
		memset(bout.map, 0, 16);
		if (grouped)
			((uint32_t *) bout.map)[ovf_base] = 0;
		if (prog->oj_words > 0)
		{
			memset((uint32_t *) bout.map + oj_base, 0, (size_t) prog->oj_words * 4);
			memcpy(bprog.map, prog->words, prog->nwords * 4);
			scan_prog_ojmap_place(prog, (uint32_t *) bprog.map, oj_base);
		}
		if (flag_words > 0)
		{
			memset((uint32_t *) bout.map + flag_base, 0, (size_t) flag_words * 4);
			((uint32_t *) bprog.map)[19] = flag_base;
		}
		for (uint32_t base = 0; base < ngroups_total; base += JOINAGG_DISPATCH_GROUPS)
		{
			uint32_t	groups = ngroups_total - base;
			uint32_t	pc[5];

			if (groups > JOINAGG_DISPATCH_GROUPS)
				groups = JOINAGG_DISPATCH_GROUPS;
			pc[0] = npages;
			pc[1] = base;
			pc[2] = ppw;
			pc[3] = (uint32_t) ovf_base;
			pc[4] = ovf_cap;
			if (kdispatch(eng, prog, grouped ? SPEC_JGAGG : SPEC_JAGG, sh.agg_max, sh.ga_c, pipe, sp, bufs, pc, groups) != 0)
			{
				snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
				goto done;
			}
		}
		if (out[1] != 0)
		{
			describe_kernel_error(out[1], out[2], out[3], err, errlen);
			goto done;
		}
		ovf_count = grouped ? out[ovf_base] : 0;
		if (ovf_count <= ovf_cap)
			break;
		ovf_cap = ovf_count + ovf_count / 2 + 1024;		/* the count may vary a little between runs */
		gpu_vk_buffer_destroy(vk, &bout);
		have_out = 0;
	}
	if (!have_out)
	{
		snprintf(err, errlen, "the join aggregation overflowed its list repeatedly");
		goto done;
	}
	t2 = now_ms();
	if (flag_words > 0 &&
		joinagg_harvest_flags(&ps, kds_src, npages, total_lps, (const uint32_t *) btab.map,
							  out + flag_base, NULL, &st->rck) != 0)
	{
		snprintf(err, errlen, "out of memory");
		goto done;
	}
	if (prog->oj_words > 0)
		join_merge_ojmap(prog, ji, out, oj_base);
	if (!grouped)
		agg_merge_workgroups(prog, st, out, ngroups_total);
	else
	{
		const uint32_t *w = prog->words;
		uint32_t	aggoff = w[11];

		if (group_merge_blocks(prog, st, out, ngroups_total, cap, rec_words, blk_words) != 0)
		{
			snprintf(err, errlen, "out of memory");
			goto done;
		}
		/* the joined rows that found no slot: their staged key and argument values */
		for (uint32_t r = 0; r < ovf_count; r++)
		{
			const uint32_t *rec = out + ovf_base + 1 + (size_t) r * ov_rec;
			int64_t		key[SCAN_GROUP_KEYS] = {0};
			group_entry *g;

			for (int i = 0; i < SCAN_GROUP_KEYS_SHARED; i++)
				key[i] = (int64_t)((uint64_t) rec[1 + 2 * i] | ((uint64_t) rec[2 + 2 * i] << 32));
			g = group_lookup(&st->groups, nagg, key, rec[0]);
			if (!g)
			{
				snprintf(err, errlen, "out of memory for the GROUP BY groups (pg_strom.gpuserv_group_memory)");
				goto done;
			}
			for (uint32_t a = 0; a < nagg; a++)
			{
				const uint32_t *av = rec + 1 + 2 * SCAN_GROUP_KEYS_SHARED + 3 * a;

				if (!av[0])
					continue;
				acc_add_value(&g->acc[a], w[aggoff + 4 * a],
							  (int64_t)((uint64_t) av[1] | ((uint64_t) av[2] << 32)));
			}
		}
	}
	t3 = now_ms();
	st->npages_vfs_read += (npages - nloaded) * PAGES_PER_BLOCK;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: join%s agg task: %u pages, pages-in %.1f ms, gpu %.1f ms, merge %.1f ms, overflow rows %u\n",
				grouped ? " group" : "", npages, t1 - t0, t2 - t1, t3 - t2, ovf_count);
	rv = 0;
done:
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "aggregation over a join failed");
	return rv;
}

/* ----------------------------------------------------------------
 * Exports for the aggregation and join tasks over Arrow record batches (arrow_exec.c), which share
 * the merge of the kernels' results with the heap tasks above
 * ----------------------------------------------------------------
 */
void
scan_agg_state_prepare(agg_state *st, int nagg)
{
	if (st->nagg == 0)
	{
		st->nagg = nagg;
		memset(st->n, 0, sizeof(st->n));
		memset(st->sum, 0, sizeof(st->sum));
		memset(st->val, 0, sizeof(st->val));
		memset(st->fovf, 0, sizeof(st->fovf));
	}
}

void
scan_agg_merge_workgroups(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total)
{
	agg_merge_workgroups(prog, st, out, ngroups_total);
}

void
scan_group_keystats(const agg_state *st, int nkeys, spec_keystats *ks)
{
	memset(ks, 0, sizeof(*ks));
	ks->nkeys = nkeys;
	if (!st->groups)
		return;
	ks->ngroups = st->groups->nents;
	for (uint32_t i = 0; i < st->groups->nents; i++)
	{
		const group_entry *e = GENT(st->groups, i);

		for (int k = 0; k < nkeys && k < 2; k++)
		{
			if (e->knull & (1u << k))
				continue;
			if (!ks->has_value[k] || e->key[k] < ks->kmin[k])
				ks->kmin[k] = e->key[k];
			if (!ks->has_value[k] || e->key[k] > ks->kmax[k])
				ks->kmax[k] = e->key[k];
			ks->has_value[k] = 1;
		}
	}
}

void
scan_group_variant(int avariant, uint32_t *cap, uint32_t *rec_words, uint32_t *ov_rec)
{
	*cap = ga_variant_cap[avariant];
	*rec_words = 2 + 2 * SCAN_GROUP_KEYS_SHARED + 6 * ga_variant_na[avariant];
	*ov_rec = 1 + 2 * SCAN_GROUP_KEYS_SHARED + 3 * ga_variant_na[avariant];
}

int
scan_group_merge_blocks(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total,
						uint32_t cap, uint32_t rec_words, uint32_t blk_words)
{
	return group_merge_blocks(prog, st, out, ngroups_total, cap, rec_words, blk_words);
}

/* the rows that found no slot in their workgroup's table: { null mask, keys, (ok, value) * aggregates } each */
int
scan_group_merge_overflow(const scan_prog *prog, agg_state *st, const uint32_t *recs, uint32_t count,
						  uint32_t ov_rec)
{
	const uint32_t *w = prog->words;
	uint32_t	aggoff = w[11];
	uint32_t	nagg = prog->nagg;

	for (uint32_t r = 0; r < count; r++)
	{
		const uint32_t *rec = recs + (size_t) r * ov_rec;
		int64_t		key[SCAN_GROUP_KEYS] = {0};
		group_entry *g;

		for (int i = 0; i < SCAN_GROUP_KEYS_SHARED; i++)
			key[i] = (int64_t)((uint64_t) rec[1 + 2 * i] | ((uint64_t) rec[2 + 2 * i] << 32));
		g = group_lookup(&st->groups, nagg, key, rec[0]);
		if (!g)
			return -1;
		for (uint32_t a = 0; a < nagg; a++)
		{
			const uint32_t *av = rec + 1 + 2 * SCAN_GROUP_KEYS_SHARED + 3 * a;

			if (!av[0])
				continue;
			acc_add_value(&g->acc[a], w[aggoff + 4 * a],
						  (int64_t)((uint64_t) av[1] | ((uint64_t) av[2] << 32)));
		}
	}
	return 0;
}

/* merge the outer-join map a join kernel filled (RIGHT/FULL joins) into the backend's */
void
scan_join_merge_ojmap(const scan_prog *prog, const join_inner *ji, const uint32_t *out, size_t base)
{
	join_merge_ojmap(prog, ji, out, base);
}
