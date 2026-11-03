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
#include "scan_exec.h"
#include "scan_caps.h"

#define SCAN_WG_MAX		65535U

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
} form_ctx;

/*
 * Form (or just measure, if titem is NULL) one result tuple. 'comp' points at the GPU's
 * results of this row: (isnull, lo, hi) per computed attribute. Returns t_len, or -1.
 */
static int
form_tuple(const form_ctx *fc, const unsigned char *htup, const uint32_t *comp,
		   uint32_t blkno, uint32_t posid,
		   kern_tupitem *titem, uint32_t rowid, char *err, size_t errlen)
{
	const scan_prog *prog = fc->prog;
	int			nattrs = prog->nattrs;
	int			nsrc = fc->maxres + 1;
	uint8_t		isnull_src[nsrc];
	const unsigned char *ptrs[nsrc];
	uint32_t	lens[nsrc];
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

	for (int j = 0; j < nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];

		if (pa->kind == SCAN_PROJ_SYS)
			continue;
		if (pa->kind == SCAN_PROJ_SRC ? isnull_src[pa->resno - 1] : (comp[3 * pa->comp] != 0))
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

			if (comp[3 * pa->comp] != 0)
				continue;				/* NULL */
			if (cm->attlen != pa->comp_bytes || !cm->attbyval)
			{
				snprintf(err, errlen, "column %d: computed %d-byte value does not fit the result type (len %d)",
						 j + 1, pa->comp_bytes, cm->attlen);
				return -1;
			}
			memcpy(tmp, &v, 8);			/* little endian */
			src = tmp;
			sz = pa->comp_bytes;
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
static void
describe_kernel_error(uint32_t err, uint32_t page, uint32_t lp, char *buf, size_t buflen)
{
	if (err & 0x80000000U)
		snprintf(buf, buflen, "malformed GPU program (instruction %u)", err & 0x7fffffffU);
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
		}
		snprintf(buf, buflen, "%s (in %s, block index %u, line pointer %u)",
				 what, scan_vm_opcode_name(func), page, lp + 1);
	}
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
	for (int i = 0; i < 3; i++)
	{
		if (eng->pipe_ready[i])
			gpu_vk_pipeline_destroy(eng->vk, &eng->pipe[i]);
		eng->pipe_ready[i] = 0;
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
	size_t		btab_sz;
	uint32_t	npages, nloaded, total_lps = 0, ncomp, rec_words;
	size_t		out_words;
	uint32_t   *out;
	XpuCommand *resp = NULL;
	uint32_t   *lp_base = NULL;
	const uint32_t *flags, *vals;
	uint32_t	nmatch = 0;
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

	variant = (prog->nregs <= 8 ? 0 : prog->nregs <= 16 ? 1 : 2);
	if (!eng->pipe_ready[variant])
	{
		const unsigned char *spv = (variant == 0 ? kspv_gpuscan8 : variant == 1 ? kspv_gpuscan16 : kspv_gpuscan32);
		size_t		spv_len = (variant == 0 ? kspv_gpuscan8_len : variant == 1 ? kspv_gpuscan16_len : kspv_gpuscan32_len);

		if (gpu_vk_pipeline_create(vk, &eng->pipe[variant], spv, spv_len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the scan pipeline: %s", vk->err);
			return NULL;
		}
		eng->pipe_ready[variant] = 1;
	}

	/* 1. the pages */
	mapped = 0;
	if (nloaded < npages && pathname && iovec && !eng->no_map)
	{
		int		rv = page_set_map(eng, &ps, kds_src, pathname, iovec);

		if (rv < 0)
		{
			snprintf(err, errlen, "out of memory");
			goto error;
		}
		mapped = rv;
	}
	if (mapped)
	{
		/* the loaded pages: a small buffer next to the file window */
		size_t		l_sz = (nloaded ? (size_t) nloaded : 1) * BLCKSZ;

		if (gpu_vk_buffer_create(vk, &ps.bufL, l_sz, 0) != 0)
		{
			snprintf(err, errlen, "cannot allocate the page buffer: %s", vk->err);
			goto error;
		}
		ps.baseL = (unsigned char *) ps.bufL.map;
		if (nloaded > 0)
			memcpy(ps.baseL, (const char *) kds_src + kds_src->block_offset, (size_t) nloaded * BLCKSZ);
	}
	else
	{
		if (page_set_copy(vk, &ps, kds_src, pathname, iovec, err, errlen) != 0)
			goto error;
		if (gpu_vk_buffer_create(vk, &ps.bufL, 16, 0) != 0)		/* placeholder binding */
		{
			snprintf(err, errlen, "cannot allocate the page buffer: %s", vk->err);
			goto error;
		}
		ps.baseL = (unsigned char *) ps.bufL.map;
	}
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
			if (gpu_vk_dispatch(vk, &eng->pipe[variant], bufs, pc, groups, 1, 1, NULL) != 0)
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
			nmatch += flags[i];
		idx_sz = MAXALIGN(sizeof(uint64_t) * nmatch);
		/* a first guess of the tuples' size; grown on demand */
		cap = resp_sz + head_sz + idx_sz + (size_t) nmatch * 64 + 4096;
		resp = calloc(1, cap);
		if (!resp)
		{
			snprintf(err, errlen, "out of memory");
			goto error;
		}
		for (uint32_t i = 0, row = 0; i < npages && row < nmatch; i++)
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

				if (!flags[g])
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
				sz = form_tuple(&fc, page + (lpd & 0x7fff), vals + (size_t) g * 3 * ncomp,
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
		*p_len = resp->length;
	}
	t4 = now_ms();
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: scan task: %u pages (%u from file) pages-in %.1f ms%s, setup %.1f ms, gpu %.1f ms, results %.1f ms (%u rows)\n",
				npages, npages - nloaded, t1 - t0, mapped ? " (mapped)" : "", t2 - t1, t3 - t2, t4 - t3, nmatch);
	free(lp_base);
	gpu_vk_buffer_destroy(vk, &bprog);
	gpu_vk_buffer_destroy(vk, &bout);
	gpu_vk_buffer_destroy(vk, &btab);
	page_set_release(vk, &ps);
	return resp;

error:
	free(lp_base);
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
