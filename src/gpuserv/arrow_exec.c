/*
 * arrow_exec.c
 *
 * GpuScan over Apache Arrow record batches (Arrow_Fdw) for the Vulkan service.
 *
 * The backend sends, per record batch, a KDS_FORMAT_ARROW header (kern_colmeta of every column
 * with the offsets of its validity bitmap, values and extra buffers) and an i/o vector of the file
 * pages that hold the referenced columns; the service reads them itself. The offsets in the colmeta
 * are relative to the start of the store, the i/o vector places file pages in the region behind
 * the header (KDS_HEAD_LENGTH + arrow_virtual_usage).
 *
 * Two ways to give the columns to the GPU:
 *   - a window of the Arrow file is mapped and imported into Vulkan (and kept between tasks, as for
 *     heap files): the kernel reads the page cache directly, nothing is copied;
 *   - otherwise the file pages are read into a buffer laid out as the store.
 * The kernel (kernels_vm/arrow.slang) runs the scan VM with one thread per row, over passes of at
 * most ARROW_PASS_ROWS rows, and flags the qualifying rows and stores the computed values. The
 * service then forms the result tuples from the Arrow buffers (pass-through columns) and the
 * GPU's values (computed columns).
 *
 * Column types that are copied as they are: int1/2/4/8, float2/4/8, bool, utf8 and binary (32 and
 * 64 bit offsets), char(n), date (days), time, timestamp. The planner refuses to offer a GpuScan
 * for a table with any other referenced column (arrowFdwGpuScanSupported() in arrow_fdw.c).
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
#include "specialize.h"
#include "scan_exec.h"
#include "scan_caps.h"
#include "gcache.h"

#define ARROW_PASS_ROWS		(1U << 20)
#define ARROW_PASS_WORDS	(16U << 20)		/* words of the GPU's output buffer per pass, at most */
#define ARROW_WG_MAX		65535U
#define ARROW_WG			64				/* SCAN_WG of arrow.slang */
#define ARROW_TS_OFFSET		946684800000000LL	/* 1970-01-01 -> 2000-01-01 in microseconds */
#define ARROW_DATE_OFFSET	10957			/* ... in days */
#define BPCHAR_OID			1042

typedef strom_io_vector	io_vector;

static double
arrow_now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ---------------------------------------------------------------- */
/* the columns' buffers, as the CPU and the GPU see them */
typedef struct
{
	int			mode;				/* 0: normal, 1: all NULL, 2: constant (virtual column) */
	int64_t		nm_off;				/* validity bitmap, -1: none */
	uint64_t	nm_len;
	int64_t		v_off;				/* values, -1: none */
	uint64_t	v_len;
	int64_t		x_off;				/* extra (the bytes of utf8 / binary), -1: none */
	uint64_t	x_len;
	const unsigned char *vconst;	/* mode 2: the datum, NULL if NULL */
} arrow_col;

typedef struct
{
	const kern_data_store *kds;
	const unsigned char *base;		/* the data as the CPU sees it */
	int64_t		head;				/* copy: the buffer starts at this offset of the store */
	int			mapped;
	int64_t		win_start;			/* mapped: file offset the buffer starts at */
	const io_vector *iovec;
} arrow_src;

/* the offset in the buffer of the store's offset 'off', -1 if no chunk covers it */
static int64_t
arrow_xlate(const arrow_src *as, uint64_t off)
{
	if (!as->mapped)
	{
		if ((int64_t) off < as->head)
			return -1;
		return (int64_t) off - as->head;
	}
	for (unsigned int c = 0; c < as->iovec->nr_chunks; c++)
	{
		const strom_io_chunk *ioc = &as->iovec->ioc[c];
		uint64_t	lo = as->head + ioc->m_offset;
		uint64_t	hi = lo + (uint64_t) ioc->nr_pages * PAGE_SIZE;

		if (off >= lo && off < hi)
			return (int64_t) ioc->fchunk_id * PAGE_SIZE + (int64_t)(off - lo) - as->win_start;
	}
	return -1;
}

static int
arrow_col_setup(const arrow_src *as, const kern_colmeta *cm, arrow_col *ac)
{
	memset(ac, 0, sizeof(*ac));
	ac->nm_off = ac->v_off = ac->x_off = -1;
	if (cm->virtual_offset != 0)
	{
		if (cm->virtual_offset < 0)
			ac->mode = 1;
		else
		{
			ac->mode = 2;
			ac->vconst = (const unsigned char *) as->kds + cm->virtual_offset;
		}
		return 0;
	}
	if (cm->atttypkind == TYPE_KIND__NULL)
		return -1;				/* not loaded */
	if (cm->nullmap_offset != 0 && cm->nullmap_length > 0)
	{
		ac->nm_off = arrow_xlate(as, cm->nullmap_offset);
		if (ac->nm_off < 0)
			return -1;
		ac->nm_len = cm->nullmap_length;
	}
	if (cm->values_offset != 0 && cm->values_length > 0)
	{
		ac->v_off = arrow_xlate(as, cm->values_offset);
		if (ac->v_off < 0)
			return -1;
		ac->v_len = cm->values_length;
	}
	if (cm->extra_offset != 0 && cm->extra_length > 0)
	{
		ac->x_off = arrow_xlate(as, cm->extra_offset);
		if (ac->x_off < 0)
			return -1;
		ac->x_len = cm->extra_length;
	}
	return 0;
}

/* ---------------------------------------------------------------- */
/* a column's datum for the result tuple (the same conversions as pg_datum_arrow_ref() of arrow_fdw.c) */
enum { FETCH_NULL = 0, FETCH_FIXED = 1, FETCH_VARLENA = 2, FETCH_RAW_VARLENA = 3 };

static inline uint32_t
rd32(const unsigned char *p)
{
	uint32_t	v;

	memcpy(&v, p, 4);
	return v;
}

static inline uint64_t
rd64(const unsigned char *p)
{
	uint64_t	v;

	memcpy(&v, p, 8);
	return v;
}

static uint32_t
arrow_varlena_size(const unsigned char *p)
{
	if (p[0] == 0x01)
		return (p[1] == 18) ? 18 : 10;
	if (p[0] & 0x01)
		return (p[0] >> 1) & 0x7f;
	return (rd32(p) >> 2) & 0x3fffffffU;
}

/*
 * Returns FETCH_NULL, FETCH_FIXED (p_ptr and p_len: the value's bytes, in 'tmp' for converted
 * types), FETCH_VARLENA (the payload, to be given a header) or FETCH_RAW_VARLENA (a complete
 * datum, virtual columns), or -1 on an error.
 */
static int
arrow_fetch(const arrow_src *as, const arrow_col *ac, const kern_colmeta *cm, uint32_t row,
			unsigned char *tmp, const unsigned char **p_ptr, uint32_t *p_len,
			char *err, size_t errlen)
{
	const unsigned char *base = as->base;
	const unsigned char *vals;

	if (ac->mode == 1)
		return FETCH_NULL;
	if (ac->mode == 2)
	{
		if (!ac->vconst)
			return FETCH_NULL;
		*p_ptr = ac->vconst;
		if (cm->attlen > 0)
		{
			*p_len = cm->attlen;
			return FETCH_FIXED;
		}
		*p_len = arrow_varlena_size(ac->vconst);
		return FETCH_RAW_VARLENA;
	}
	if (ac->nm_off >= 0)
	{
		if ((row >> 3) >= ac->nm_len || !((base[ac->nm_off + (row >> 3)] >> (row & 7)) & 1))
			return FETCH_NULL;
	}
	vals = (ac->v_off >= 0 ? base + ac->v_off : NULL);
	switch (cm->attopts.tag)
	{
		case ArrowType__Int:
		case ArrowType__FloatingPoint:
		{
			uint32_t	w = cm->attopts.unitsz;

			if (w == 0 || w > 8 || !vals || (uint64_t) w * (row + 1) > ac->v_len)
				goto corrupted;
			if ((int) w != cm->attlen)
			{
				snprintf(err, errlen, "column %d: Arrow value of %u bytes does not fit the type (len %d)",
						 cm->attnum, w, cm->attlen);
				return -1;
			}
			*p_ptr = vals + (size_t) w * row;
			*p_len = w;
			return FETCH_FIXED;
		}
		case ArrowType__Bool:
			if (!vals || (row >> 3) >= ac->v_len)
				goto corrupted;
			tmp[0] = (vals[row >> 3] >> (row & 7)) & 1;
			*p_ptr = tmp;
			*p_len = 1;
			return FETCH_FIXED;
		case ArrowType__Date:
		{
			int32_t		dt;

			if (cm->attopts.date.unit != ArrowDateUnit__Day || !vals || 4 * (uint64_t)(row + 1) > ac->v_len)
				goto corrupted;
			dt = (int32_t) rd32(vals + 4 * (size_t) row);
			dt -= ARROW_DATE_OFFSET;
			memcpy(tmp, &dt, 4);
			*p_ptr = tmp;
			*p_len = 4;
			return FETCH_FIXED;
		}
		case ArrowType__Time:
		{
			int64_t		tm;

			switch (cm->attopts.time.unit)
			{
				case ArrowTimeUnit__Second:
					if (!vals || 4 * (uint64_t)(row + 1) > ac->v_len)
						goto corrupted;
					tm = (int64_t) rd32(vals + 4 * (size_t) row) * 1000000L;
					break;
				case ArrowTimeUnit__MilliSecond:
					if (!vals || 4 * (uint64_t)(row + 1) > ac->v_len)
						goto corrupted;
					tm = (int64_t) rd32(vals + 4 * (size_t) row) * 1000L;
					break;
				case ArrowTimeUnit__MicroSecond:
					if (!vals || 8 * (uint64_t)(row + 1) > ac->v_len)
						goto corrupted;
					tm = (int64_t) rd64(vals + 8 * (size_t) row);
					break;
				default:
					if (!vals || 8 * (uint64_t)(row + 1) > ac->v_len)
						goto corrupted;
					tm = (int64_t)(rd64(vals + 8 * (size_t) row) / 1000L);
					break;
			}
			memcpy(tmp, &tm, 8);
			*p_ptr = tmp;
			*p_len = 8;
			return FETCH_FIXED;
		}
		case ArrowType__Timestamp:
		{
			uint64_t	ts;

			if (!vals || 8 * (uint64_t)(row + 1) > ac->v_len)
				goto corrupted;
			ts = rd64(vals + 8 * (size_t) row);
			switch (cm->attopts.timestamp.unit)
			{
				case ArrowTimeUnit__Second:
					ts *= 1000000UL;
					break;
				case ArrowTimeUnit__MilliSecond:
					ts *= 1000UL;
					break;
				case ArrowTimeUnit__MicroSecond:
					break;
				default:
					ts /= 1000UL;
					break;
			}
			ts -= (uint64_t) ARROW_TS_OFFSET;
			memcpy(tmp, &ts, 8);
			*p_ptr = tmp;
			*p_len = 8;
			return FETCH_FIXED;
		}
		case ArrowType__Utf8:
		case ArrowType__Binary:
		{
			uint32_t	o1, o2;

			if (!vals || 4 * (uint64_t)(row + 1) > ac->v_len)
				return FETCH_NULL;				/* as KDS_ARROW_REF_VARLENA32_DATUM() */
			o1 = rd32(vals + 4 * (size_t) row);
			o2 = (4 * (uint64_t)(row + 2) <= ac->v_len) ? rd32(vals + 4 * ((size_t) row + 1)) : 0;
			if (4 * (uint64_t)(row + 2) > ac->v_len || o1 > o2 || o2 > ac->x_len || o2 - o1 > VARATT_MAX)
				return FETCH_NULL;
			*p_ptr = (o2 > o1 ? base + ac->x_off + o1 : tmp);
			*p_len = o2 - o1;
			return FETCH_VARLENA;
		}
		case ArrowType__LargeUtf8:
		case ArrowType__LargeBinary:
		{
			uint64_t	o1, o2;

			if (!vals || 8 * (uint64_t)(row + 2) > ac->v_len)
				return FETCH_NULL;
			o1 = rd64(vals + 8 * (size_t) row);
			o2 = rd64(vals + 8 * ((size_t) row + 1));
			if (o1 > o2 || o2 > ac->x_len || o2 - o1 > VARATT_MAX)
				return FETCH_NULL;
			*p_ptr = (o2 > o1 ? base + ac->x_off + o1 : tmp);
			*p_len = (uint32_t)(o2 - o1);
			return FETCH_VARLENA;
		}
		case ArrowType__FixedSizeBinary:
		{
			uint32_t	w = cm->attopts.fixed_size_binary.byteWidth;

			if (cm->atttypid != BPCHAR_OID || w == 0 || !vals || (uint64_t) w * (row + 1) > ac->v_len)
				goto corrupted;
			*p_ptr = vals + (size_t) w * row;
			*p_len = w;
			return FETCH_VARLENA;
		}
		default:
			break;
	}
	snprintf(err, errlen, "column %d: Arrow type %s is not supported by the Vulkan GpuScan",
			 cm->attnum, ArrowTypeTagAsCString(cm->attopts.tag));
	return -1;
corrupted:
	snprintf(err, errlen, "column %d: corrupted or unsupported Arrow buffer (row %u)", cm->attnum, row);
	return -1;
}

/* ---------------------------------------------------------------- */
typedef struct
{
	const kern_data_store *kds_dst_head;
	const kern_data_store *kds_src;
	const scan_prog *prog;
	const arrow_src *as;
	const arrow_col *cols;			/* per source column */
	gcache	   *gc;					/* a GPU-Cache task: the columns come from the cache (cols is unused) */
	/* scratch of the row being formed (one entry per result column) */
	uint8_t	   *kind;
	const unsigned char **ptr;
	uint32_t   *len;
	unsigned char (*tmpv)[8];
	unsigned char (*numv)[SCAN_NUMERIC_MAXLEN];		/* a computed numeric */
	/* a join: the inner relations' stores, and the inner items of the row being formed (NULL: none) */
	int			ndepth;
	const kern_data_store *kds_in[SCAN_JOIN_MAXDEPTH + 1];
	const unsigned char *inner[SCAN_JOIN_MAXDEPTH + 1];
} arrow_form;

/*
 * Column `resno` (1 ...) of an inner tuple of a join (a kern_hashitem: a heap tuple behind an 8-byte chain
 * pointer): its bytes and length, or -1 if it is NULL.
 */
static int
arrow_inner_col(const kern_data_store *kds, const unsigned char *htup, int resno,
				const unsigned char **p_ptr, uint32_t *p_len)
{
	uint16_t	infomask, infomask2;
	uint32_t	off = htup[22];
	uint32_t	natts;
	int			hasnull;

	memcpy(&infomask2, htup + 18, 2);
	memcpy(&infomask, htup + 20, 2);
	natts = infomask2 & 0x7ff;
	hasnull = (infomask & 1) != 0;
	for (int i = 0; i < resno; i++)
	{
		const kern_colmeta *cm = &kds->colmeta[i];
		uint32_t	sz;

		if ((uint32_t) i >= natts || (hasnull && !((htup[23 + (i >> 3)] >> (i & 7)) & 1)))
		{
			if (i + 1 == resno)
				return -1;
			continue;
		}
		if (cm->attlen == -1)
		{
			if (htup[off] == 0)
				off = (off + cm->attalign - 1) & ~(uint32_t)(cm->attalign - 1);
			sz = arrow_varlena_size(htup + off);
		}
		else
		{
			off = (off + cm->attalign - 1) & ~(uint32_t)(cm->attalign - 1);
			sz = (cm->attlen > 0 ? (uint32_t) cm->attlen : 0);
		}
		if (i + 1 == resno)
		{
			*p_ptr = htup + off;
			*p_len = sz;
			return 0;
		}
		off += sz;
	}
	return -1;
}

/*
 * Form (or only measure, if titem is NULL) the result tuple of one row. 'comp' holds the GPU's
 * values of the row: (isnull, lo, hi) per computed column. A call with fetch = 1 reads the row's
 * columns into the scratch arrays of 'af'; with fetch = 0 it formats what the previous call read
 * (measure, then write). Returns the tuple's length, -1 on error.
 */
static int
arrow_form_tuple(arrow_form *af, uint32_t row, const uint32_t *comp, int fetch,
				 kern_tupitem *titem, uint32_t rowid, char *err, size_t errlen)
{
	const scan_prog *prog = af->prog;
	int			nattrs = prog->nattrs;
	uint8_t	   *kind = af->kind;
	const unsigned char **ptr = af->ptr;
	uint32_t   *len = af->len;
	unsigned char (*tmpv)[8] = af->tmpv;
	uint32_t	t_hoff;
	uint16_t	infomask = 0;
	int			hasnull = 0;

	for (int j = 0; fetch && j < nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];
		const kern_colmeta *cm = &af->kds_dst_head->colmeta[j];

		len[j] = 0;
		ptr[j] = NULL;
		if (pa->kind == SCAN_PROJ_SYS && af->gc && pa->resno == -1)
		{
			gcache_ctid(af->gc, row, tmpv[j]);
			ptr[j] = tmpv[j];
			len[j] = 6;
			kind[j] = FETCH_FIXED;
			if (cm->attlen != 6)
			{
				snprintf(err, errlen, "column %d: ctid does not fit the result type (len %d)", j + 1, cm->attlen);
				return -1;
			}
		}
		else if (pa->kind == SCAN_PROJ_SYS)
		{
			if (pa->resno != -6)
			{
				snprintf(err, errlen, "column %d: system column %d of an Arrow table", j + 1, pa->resno);
				return -1;
			}
			memcpy(tmpv[j], &af->kds_src->table_oid, 4);
			ptr[j] = tmpv[j];
			len[j] = 4;
			kind[j] = FETCH_FIXED;
			if (cm->attlen != 4)
			{
				snprintf(err, errlen, "column %d: tableoid does not fit the result type (len %d)", j + 1, cm->attlen);
				return -1;
			}
		}
		else if (pa->kind == SCAN_PROJ_SRC)
		{
			int		rv = (af->gc
						  ? gcache_fetch(af->gc, pa->resno, row, tmpv[j], &ptr[j], &len[j], err, errlen)
						  : arrow_fetch(af->as, &af->cols[pa->resno - 1], &af->kds_src->colmeta[pa->resno - 1],
										row, tmpv[j], &ptr[j], &len[j], err, errlen));

			if (rv < 0)
				return -1;
			kind[j] = (uint8_t) rv;
			if (rv == FETCH_FIXED && cm->attlen != (int) len[j])
			{
				snprintf(err, errlen, "column %d: result type length %d differs from the source (%u)",
						 j + 1, cm->attlen, len[j]);
				return -1;
			}
			if ((rv == FETCH_VARLENA || rv == FETCH_RAW_VARLENA) && cm->attlen != -1)
			{
				snprintf(err, errlen, "column %d: a variable-length source for a fixed-length result (len %d)",
						 j + 1, cm->attlen);
				return -1;
			}
		}
		else if (pa->kind == SCAN_PROJ_INNER)
		{
			const unsigned char *it = (pa->depth >= 1 && pa->depth <= af->ndepth ? af->inner[pa->depth] : NULL);
			const kern_data_store *ki = af->kds_in[pa->depth];

			kind[j] = FETCH_NULL;
			if (it)
			{
				const unsigned char *vp;
				uint32_t	vl;

				if (arrow_inner_col(ki, it, pa->resno, &vp, &vl) < 0)
					continue;				/* NULL */
				if (cm->attlen != ki->colmeta[pa->resno - 1].attlen)
				{
					snprintf(err, errlen, "column %d: result type length %d differs from the inner relation (%d)",
							 j + 1, cm->attlen, ki->colmeta[pa->resno - 1].attlen);
					return -1;
				}
				ptr[j] = vp;
				len[j] = vl;
				kind[j] = (cm->attlen == -1 ? FETCH_RAW_VARLENA : FETCH_FIXED);
			}
		}
		else if (pa->kind == SCAN_PROJ_COMP && pa->comp_numeric)
		{
			uint32_t	lo = comp[3 * pa->comp + 1];
			uint32_t	hi = comp[3 * pa->comp + 2];
			uint64_t	v = (uint64_t) lo | ((uint64_t) hi << 32);

			if ((comp[3 * pa->comp] & 1u) != 0)
				kind[j] = FETCH_NULL;
			else
			{
				if (cm->attlen != -1)
				{
					snprintf(err, errlen, "column %d: computed numeric does not fit the result type (len %d)",
							 j + 1, cm->attlen);
					return -1;
				}
				len[j] = scan_vm_numeric_encode((int64_t) v, comp[3 * pa->comp] >> 8, af->numv[j]);
				ptr[j] = af->numv[j];
				kind[j] = FETCH_RAW_VARLENA;
			}
		}
		else if (pa->kind == SCAN_PROJ_COMP)
		{
			uint32_t	lo = comp[3 * pa->comp + 1];
			uint32_t	hi = comp[3 * pa->comp + 2];
			uint64_t	v = (uint64_t) lo | ((uint64_t) hi << 32);

			if ((comp[3 * pa->comp] & 1u) != 0)
				kind[j] = FETCH_NULL;
			else
			{
				if (cm->attlen != pa->comp_bytes || !cm->attbyval)
				{
					snprintf(err, errlen, "column %d: computed %d-byte value does not fit the result type (len %d)",
							 j + 1, pa->comp_bytes, cm->attlen);
					return -1;
				}
				memcpy(tmpv[j], &v, 8);
				ptr[j] = tmpv[j];
				len[j] = pa->comp_bytes;
				kind[j] = FETCH_FIXED;
			}
		}
		else
		{
			snprintf(err, errlen, "column %d: a projection of kind %d on an Arrow table", j + 1, pa->kind);
			return -1;
		}
	}
	for (int j = 0; j < nattrs; j++)
		if (kind[j] == FETCH_NULL)
		{
			hasnull = 1;
			infomask |= 1;				/* HEAP_HASNULL */
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
		const kern_colmeta *cm = &af->kds_dst_head->colmeta[j];
		uint32_t	t_next, hdr = 0, sz;

		if (kind[j] == FETCH_NULL)
			continue;
		if (kind[j] == FETCH_FIXED)
		{
			t_next = TYPEALIGN(cm->attalign, t_hoff);
			sz = len[j];
		}
		else if (kind[j] == FETCH_RAW_VARLENA)
		{
			infomask |= 2;				/* HEAP_HASVARWIDTH */
			if (ptr[j][0] == 0x01)
				infomask |= 4;			/* HEAP_HASEXTERNAL */
			t_next = (ptr[j][0] & 0x01) ? t_hoff : TYPEALIGN(cm->attalign, t_hoff);
			sz = len[j];
		}
		else
		{
			infomask |= 2;
			if (len[j] + 1 <= 127)		/* a 1-byte header, as heap_form_tuple() */
			{
				hdr = 1;
				t_next = t_hoff;
				sz = len[j] + 1;
			}
			else
			{
				hdr = 4;
				t_next = TYPEALIGN(cm->attalign, t_hoff);
				sz = len[j] + 4;
			}
		}
		if (titem)
		{
			unsigned char *dest;

			if (t_next > t_hoff)
				memset((char *) titem + t_hoff, 0, t_next - t_hoff);
			dest = (unsigned char *) titem + t_next;
			if (hdr == 1)
			{
				dest[0] = (unsigned char)((sz << 1) | 1);
				if (len[j] > 0)
					memcpy(dest + 1, ptr[j], len[j]);
			}
			else if (hdr == 4)
			{
				uint32_t	h = sz << 2;

				memcpy(dest, &h, 4);
				memcpy(dest + 4, ptr[j], len[j]);
			}
			else
				memcpy(dest, ptr[j], sz);
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
/*
 * The data of a task as the CPU sees it: the task's own bytes (a Parquet row-group the backend
 * decoded), a read-only mapping of the file range the i/o vector covers (the usual case: pass-through
 * columns are read from the page cache, nothing is copied), or, if the mapping fails, the pages read
 * into a buffer laid out as the store. The GPU gets its own packed copy of just the columns the
 * kernel reads (arrow_pack_gpu).
 */
typedef struct
{
	void		   *map;
	size_t			maplen;
	unsigned char  *copy;
	arrow_src		as;
} arrow_data;

static int
arrow_data_open(arrow_data *ad, const kern_data_store *kds, const char *pathname,
				const io_vector *iovec, char *err, size_t errlen)
{
	int64_t		head = (int64_t) KDS_HEAD_LENGTH(kds) + kds->arrow_virtual_usage;
	int64_t		lo = INT64_MAX, hi = 0;
	int			fd;
	struct stat	st;
	size_t		len;

	memset(ad, 0, sizeof(*ad));
	ad->as.kds = kds;
	ad->as.head = head;
	ad->as.iovec = iovec;
	if (!iovec || iovec->nr_chunks == 0)
	{
		/* the buffers are in the task (a Parquet row-group), or no column has a buffer */
		ad->as.head = (!iovec && !pathname) ? 0 : head;
		ad->as.base = (const unsigned char *) kds;
		return 0;
	}
	if (!pathname)
	{
		snprintf(err, errlen, "no file to read the Arrow columns from");
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
	for (unsigned int c = 0; c < iovec->nr_chunks; c++)
	{
		const strom_io_chunk *ioc = &iovec->ioc[c];
		int64_t		a = (int64_t) ioc->fchunk_id * PAGE_SIZE;
		int64_t		b = a + (int64_t) ioc->nr_pages * PAGE_SIZE;

		if (a < lo)
			lo = a;
		if (b > hi)
			hi = b;
	}
	{
		int64_t		pg = sysconf(_SC_PAGESIZE);
		int64_t		lo_al = lo & ~(pg - 1);

		if (lo_al < (int64_t) st.st_size && !getenv("PGSTROM_GPUSERV_NO_MAP"))
		{
			void	   *map = mmap(NULL, (size_t)(hi - lo_al), PROT_READ, MAP_PRIVATE, fd, (off_t) lo_al);

			if (map != MAP_FAILED)
			{
				ad->map = map;
				ad->maplen = (size_t)(hi - lo_al);
				ad->as.mapped = 1;
				ad->as.win_start = lo_al;
				ad->as.base = (const unsigned char *) map;
				close(fd);
				return 0;
			}
		}
	}
	/* no mapping: read the pages into a buffer laid out as the store behind its header */
	if (kds->length < (uint64_t) head)
	{
		snprintf(err, errlen, "corrupted Arrow data store");
		close(fd);
		return -1;
	}
	len = (size_t)(kds->length - head);
	ad->copy = calloc(1, len ? len : 1);
	if (!ad->copy)
	{
		snprintf(err, errlen, "out of memory");
		close(fd);
		return -1;
	}
	ad->as.base = ad->copy;
	for (unsigned int i = 0; i < iovec->nr_chunks; i++)
	{
		const strom_io_chunk *ioc = &iovec->ioc[i];
		size_t		remained = (size_t) ioc->nr_pages * PAGE_SIZE;
		off_t		file_pos = (off_t) ioc->fchunk_id * PAGE_SIZE;
		unsigned char *dest = ad->copy + ioc->m_offset;

		if (ioc->m_offset + remained > len)
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
	return 0;
}

static void
arrow_data_close(arrow_data *ad)
{
	if (ad->map)
		munmap(ad->map, ad->maplen);
	free(ad->copy);
	memset(ad, 0, sizeof(*ad));
}

/* ---------------------------------------------------------------- */
void
arrow_engine_fini(scan_engine *eng)
{
	for (int i = 0; i < 6; i++)
	{
		if (eng->arrow_pipe_ready[i])
			gpu_vk_pipeline_destroy(eng->vk, &eng->arrow_pipe[i]);
		eng->arrow_pipe_ready[i] = 0;
	}
	for (int v = 0; v < 4; v++)
	{
		for (int a = 0; a < 3; a++)
		{
			if (eng->aagg_ready[v][a])
				gpu_vk_pipeline_destroy(eng->vk, &eng->aagg_pipe[v][a]);
			if (eng->agagg_ready[v][a])
				gpu_vk_pipeline_destroy(eng->vk, &eng->agagg_pipe[v][a]);
			if (eng->ajagg_ready[v][a])
				gpu_vk_pipeline_destroy(eng->vk, &eng->ajagg_pipe[v][a]);
			if (eng->ajgagg_ready[v][a])
				gpu_vk_pipeline_destroy(eng->vk, &eng->ajgagg_pipe[v][a]);
			eng->aagg_ready[v][a] = eng->agagg_ready[v][a] = eng->ajagg_ready[v][a] = eng->ajgagg_ready[v][a] = 0;
		}
		if (eng->ajoin_ready[v])
			gpu_vk_pipeline_destroy(eng->vk, &eng->ajoin_pipe[v]);
		eng->ajoin_ready[v] = 0;
	}
}

/* a Vulkan type of the column the program expects, for a message */
static const char *
vt_name(uint32_t vt)
{
	switch (vt)
	{
		case SOP_VT_BOOL:	return "bool";
		case SOP_VT_INT:	return "integer";
		case SOP_VT_FLOAT4:	return "float4";
		case SOP_VT_FLOAT8:	return "float8";
		case SOP_VT_STR:	return "text";
		default:			return "?";
	}
}

/* modes and conversions of the column table (kernels_vm/arrow_load.inc) */
enum { ACM_NORMAL = 0, ACM_NULL = 1, ACM_CONST = 2, ACM_STR = 3, ACM_FIXSTR = 4 };
enum { ACV_NONE = 0, ACV_DATE = 1, ACV_TIME_S = 2, ACV_TIME_MS = 3, ACV_TIME_US = 4, ACV_TIME_NS = 5,
	   ACV_TS_S = 6, ACV_TS_MS = 7, ACV_TS_US = 8, ACV_TS_NS = 9 };

/* table words per column, and the words after the table (rows of the pass, first row of the pass) */
#define ARROW_TAB_COLWORDS	8
#define ARROW_TAB_EXTRA		4		/* rows of the pass, first row, row filter flag (GPU-Cache), spare */

/*
 * Fill the column table of the kernel (8 words per column of the program's column list) and check
 * that the Arrow columns have the types the program reads them as.
 */
static int
arrow_build_table(const arrow_src *as, const kern_data_store *kds, const scan_prog *prog,
				  const arrow_col *cols, uint32_t *tab, char *err, size_t errlen)
{
	uint32_t	coloff = prog->words[0];
	uint32_t	ncols = prog->words[3];

	memset(tab, 0, sizeof(uint32_t) * (ARROW_TAB_COLWORDS * ncols + ARROW_TAB_EXTRA));
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = prog->words[coloff + c];
		uint32_t	slot = (desc >> 16) & 0xff;
		uint32_t	vt = (desc >> 12) & 0xf;
		const kern_colmeta *cm = &kds->colmeta[c];
		const arrow_col *ac = &cols[c];
		uint32_t   *t = tab + ARROW_TAB_COLWORDS * c;
		uint32_t	w = 0, conv = ACV_NONE, mode = ACM_NORMAL;

		t[0] = 0xffffffffU;
		if (slot == 0xff)
			continue;
		if (vt != 0 && vt != SOP_VT_BOOL && vt != SOP_VT_INT && vt != SOP_VT_FLOAT4 && vt != SOP_VT_FLOAT8 &&
			vt != SOP_VT_NULLONLY && vt != SOP_VT_STR)
		{
			snprintf(err, errlen, "column %u: the expression reads a value type the Arrow scan does not load", c + 1);
			return -1;
		}
		if (ac->mode == 1)
		{
			t[4] = ACM_NULL;
			continue;
		}
		if (ac->mode == 2)
		{
			uint64_t	v = 0;

			if (!ac->vconst)
			{
				t[4] = ACM_NULL;
				continue;
			}
			if (vt == SOP_VT_NULLONLY)
			{
				t[4] = ACM_CONST;
				continue;
			}
			if (vt == SOP_VT_STR || !cm->attbyval || cm->attlen > 8 || cm->attlen <= 0)
			{
				snprintf(err, errlen, "column %u: virtual column of an unsupported type", c + 1);
				return -1;
			}
			memcpy(&v, ac->vconst, cm->attlen);
			if (vt == SOP_VT_INT)
			{
				int			sh = 64 - 8 * cm->attlen;

				v = (uint64_t)(((int64_t)(v << sh)) >> sh);
			}
			t[4] = ACM_CONST;
			t[5] = (uint32_t) v;
			t[6] = (uint32_t)(v >> 32);
			continue;
		}
		if (ac->nm_off >= 0)
		{
			t[0] = (uint32_t) ac->nm_off;
			t[1] = (uint32_t) ac->nm_len;
		}
		if (vt == SOP_VT_NULLONLY)
			continue;
		switch (vt)
		{
			case SOP_VT_BOOL:
				if (cm->attopts.tag != ArrowType__Bool)
					goto badtype;
				w = 0;
				break;
			case SOP_VT_INT:
				if (cm->attopts.tag == ArrowType__Int)
				{
					if (cm->attopts.unitsz != cm->attlen ||
						(cm->attlen != 1 && cm->attlen != 2 && cm->attlen != 4 && cm->attlen != 8))
						goto badtype;
					w = cm->attlen;
				}
				else if (cm->attopts.tag == ArrowType__Date)
				{
					if (cm->attopts.date.unit != ArrowDateUnit__Day || cm->attlen != 4)
						goto badtype;
					w = 4;
					conv = ACV_DATE;
				}
				else if (cm->attopts.tag == ArrowType__Time)
				{
					if (cm->attlen != 8)
						goto badtype;
					switch (cm->attopts.time.unit)
					{
						case ArrowTimeUnit__Second:		w = 4; conv = ACV_TIME_S; break;
						case ArrowTimeUnit__MilliSecond: w = 4; conv = ACV_TIME_MS; break;
						case ArrowTimeUnit__MicroSecond: w = 8; conv = ACV_TIME_US; break;
						default:						w = 8; conv = ACV_TIME_NS; break;
					}
				}
				else if (cm->attopts.tag == ArrowType__Timestamp)
				{
					if (cm->attlen != 8)
						goto badtype;
					w = 8;
					switch (cm->attopts.timestamp.unit)
					{
						case ArrowTimeUnit__Second:		conv = ACV_TS_S; break;
						case ArrowTimeUnit__MilliSecond: conv = ACV_TS_MS; break;
						case ArrowTimeUnit__MicroSecond: conv = ACV_TS_US; break;
						default:						conv = ACV_TS_NS; break;
					}
				}
				else
					goto badtype;
				break;
			case SOP_VT_STR:
				if (cm->attopts.tag == ArrowType__Utf8)
				{
					w = 4;
					mode = ACM_STR;
				}
				else if (cm->attopts.tag == ArrowType__LargeUtf8)
				{
					w = 8;
					mode = ACM_STR;
				}
				else if (cm->attopts.tag == ArrowType__FixedSizeBinary && cm->atttypid == BPCHAR_OID &&
						 cm->attopts.fixed_size_binary.byteWidth > 0)
				{
					w = cm->attopts.fixed_size_binary.byteWidth;
					mode = ACM_FIXSTR;
				}
				else
					goto badtype;
				break;
			case SOP_VT_FLOAT8:		/* an Arrow double: the 64 bits are the register */
				if (cm->attopts.tag != ArrowType__FloatingPoint || cm->attopts.unitsz != 8 || cm->attlen != 8)
					goto badtype;
				w = 8;
				break;
			default:	/* SOP_VT_FLOAT4 */
				if (cm->attopts.tag != ArrowType__FloatingPoint || cm->attopts.unitsz != 4)
					goto badtype;
				w = 4;
				break;
		}
		if (ac->v_off < 0)
		{
			t[4] = ACM_NULL;		/* no values at all: every row is NULL */
			continue;
		}
		t[2] = (uint32_t) ac->v_off;
		t[3] = w;
		t[4] = mode;
		t[5] = conv;
		t[7] = (uint32_t) ac->v_len;
		continue;
badtype:
		snprintf(err, errlen, "column %u: Arrow type %s cannot be read as %s by the GPU scan",
				 c + 1, ArrowTypeTagAsCString(cm->attopts.tag), vt_name(vt));
		return -1;
	}
	(void) as;
	return 0;
}

/*
 * The GPU's copy: the validity bitmaps, values and (strings) bytes of the columns the kernel reads,
 * packed one after the other (8-byte aligned) in one buffer; the table gets the offsets in it.
 * Columns that are only passed through are not copied (the CPU reads them from the file).
 */
static int
arrow_pack_gpu(gpu_vk_ctx *vk, const arrow_src *as, const scan_prog *prog, const arrow_col *cols,
			   uint32_t *tab, gpu_vk_buffer *bg, char *err, size_t errlen)
{
	uint32_t	coloff = prog->words[0];
	uint32_t	ncols = prog->words[3];
	size_t		total = 16;
	unsigned char *dst = NULL;

	for (int pass = 0; pass < 2; pass++)
	{
		size_t		pos = 0;

		for (uint32_t c = 0; c < ncols; c++)
		{
			uint32_t	slot = (prog->words[coloff + c] >> 16) & 0xff;
			uint32_t   *t = tab + ARROW_TAB_COLWORDS * c;
			const arrow_col *ac = &cols[c];

			if (slot == 0xff || t[4] == ACM_NULL || t[4] == ACM_CONST)
				continue;
			if (t[0] != 0xffffffffU)
			{
				if (pass)
				{
					memcpy(dst + pos, as->base + ac->nm_off, ac->nm_len);
					t[0] = (uint32_t) pos;
				}
				pos += (ac->nm_len + 7) & ~(size_t) 7;
			}
			if (t[7] > 0)
			{
				if (pass)
				{
					memcpy(dst + pos, as->base + ac->v_off, ac->v_len);
					t[2] = (uint32_t) pos;
				}
				pos += (ac->v_len + 7) & ~(size_t) 7;
			}
			if (t[4] == ACM_STR)
			{
				if (pass)
				{
					if (ac->x_off >= 0 && ac->x_len > 0)
						memcpy(dst + pos, as->base + ac->x_off, ac->x_len);
					t[5] = (uint32_t) pos;
					t[6] = (uint32_t) ac->x_len;
				}
				pos += (ac->x_len + 7) & ~(size_t) 7;
			}
		}
		if (pass == 0)
		{
			total = pos > 16 ? pos : 16;
			if (total > vk->limits.maxStorageBufferRange || total >= 0xffffffffU ||
				(vk->max_buffer_size != 0 && total > vk->max_buffer_size))
			{
				snprintf(err, errlen, "the columns of the record batch (%zu bytes) exceed the device's storage buffer limit (%llu)",
						 total, (unsigned long long) vk->limits.maxStorageBufferRange);
				return -1;
			}
			if (gpu_vk_buffer_create(vk, bg, total, 0) != 0)
			{
				snprintf(err, errlen, "cannot allocate %zu bytes of device memory: %s", total, vk->err);
				return -1;
			}
			dst = (unsigned char *) bg->map;
		}
	}
	return 0;
}

/* ---------------------------------------------------------------- */
/*
 * What a task over an Arrow record batch needs: the data as the CPU sees it, the column table and the
 * GPU's packed copy of the columns the kernel reads, the program and a placeholder buffer for the
 * binding of the kernels that is unused here (the second set of heap pages).
 */
#define ARROW_VP			256U		/* AR_VP of arrow_load.inc: rows of a virtual page */
#define ARROW_AGG_PAGES_PER_WG	32U		/* virtual pages one aggregation workgroup runs over */
#define ARROW_JAGG_PAGES_PER_WG	8U
#define ARROW_JGAGG_PAGES_PER_WG	32U
#define ARROW_JOINAGG_DISPATCH_GROUPS	128U	/* workgroups per dispatch of a join aggregation (see JOINAGG_DISPATCH_GROUPS in scan_exec.c) */
#define ARROW_JOIN_PASS_ROWS	(1U << 20)	/* rows of a join pass (the output buffer holds the matches) */

typedef struct
{
	const kern_data_store *kds_src;
	const char *pathname;
	const io_vector *iovec;
	arrow_data	ad;
	arrow_col  *cols;
	uint32_t   *tab;
	uint32_t	ncols, nrows;
	gcache	   *gc;					/* a GPU-Cache task: bg is the cache's buffer (not owned), bdummy its row filter */
	gpu_vk_buffer bg, btab, bprog, bdummy;
	int			have_data, have_g, have_tab, have_prog, have_dummy;
} arrow_task;

static void
arrow_task_close(gpu_vk_ctx *vk, arrow_task *at)
{
	if (at->have_g)
		gpu_vk_buffer_destroy(vk, &at->bg);
	if (at->have_tab)
		gpu_vk_buffer_destroy(vk, &at->btab);
	if (at->have_prog)
		gpu_vk_buffer_destroy(vk, &at->bprog);
	if (at->have_dummy)
		gpu_vk_buffer_destroy(vk, &at->bdummy);
	if (at->have_data)
		arrow_data_close(&at->ad);
	free(at->cols);
	free(at->tab);
	memset(at, 0, sizeof(*at));
}

/* the record batch of a task: its store, the file and the i/o vector. 0 on success. */
static int
arrow_task_parse(const XpuCommand *task, arrow_task *at, char *err, size_t errlen)
{
	memset(at, 0, sizeof(*at));
	if (!task->u.task.kds_src_offset)
	{
		snprintf(err, errlen, "no source data store in the task");
		return -1;
	}
	at->kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (task->tag == XpuCommandTag__XpuTaskExecGpuCache)
	{
		/* GPU-Cache: the data is the service's own */
		if (task->u.task.kds_src_offset + sizeof(kern_data_store) > task->length ||
			task->length < offsetof(XpuCommand, u.gc_task.data))
		{
			snprintf(err, errlen, "corrupted GPU-Cache task");
			return -1;
		}
		at->gc = gcache_find(task->u.gc_task.database_oid, task->u.gc_task.table_oid, task->u.gc_task.table_sig);
		if (!at->gc)
		{
			snprintf(err, errlen, "gpucache: the cache of table %u is not in the GPU service (it was restarted?)",
					 task->u.gc_task.table_oid);
			return -1;
		}
		at->nrows = gcache_nrows(at->gc);
		return 0;
	}
	if (at->kds_src->format != KDS_FORMAT_ARROW)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan service for this task",
				 at->kds_src->format);
		return -1;
	}
	if (task->u.task.kds_src_iovec)
	{
		/* the file's pages (Arrow IPC) */
		if (task->u.task.kds_src_pathname)
			at->pathname = (const char *) task + task->u.task.kds_src_pathname;
		at->iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	}
	else if (task->u.task.kds_src_offset + at->kds_src->length > task->length)
	{
		snprintf(err, errlen, "corrupted Arrow data store");
		return -1;
	}
	at->nrows = at->kds_src->nitems;
	return 0;
}

/*
 * A GPU-Cache task: the cache's device buffer is the GPU's data (nothing is copied), the column table
 * points into it, and the row filter is the visibility bitmap of the session.
 */
static int
arrow_task_setup_gcache(gpu_vk_ctx *vk, arrow_task *at, const scan_prog *prog, char *err, size_t errlen)
{
	size_t		tabbytes = sizeof(uint32_t) * (ARROW_TAB_COLWORDS * (size_t) at->ncols + ARROW_TAB_EXTRA);
	const kern_session_info *session = gcache_cur_session;
	const SerializedTransactionState *xs = NULL;
	int			own = 0;

	if (session && session->session_xact_state)
		xs = (const SerializedTransactionState *)((const char *) session + session->session_xact_state);
	at->tab = calloc(1, tabbytes);
	if (!at->tab)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (uint32_t c = 0; c < at->ncols; c++)
		at->tab[ARROW_TAB_COLWORDS * c] = 0xffffffffU;
	if (gcache_build_table(at->gc, prog, at->tab, err, errlen) != 0)
		return -1;
	at->tab[ARROW_TAB_COLWORDS * at->ncols + 2] = 1;			/* row filter in the fifth buffer */
	at->bg = *gcache_data_buffer(at->gc);
	if (gcache_visibility(at->gc, xs ? (const uint32_t *) xs->parallelCurrentXids : NULL,
						  xs ? xs->nParallelCurrentXids : 0, &at->bdummy, &own, err, errlen) != 0)
		return -1;
	at->have_dummy = own;				/* a temporary bitmap is ours to destroy */
	if (gpu_vk_buffer_create(vk, &at->btab, tabbytes, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate a buffer: %s", vk->err);
		return -1;
	}
	at->have_tab = 1;
	memcpy(at->btab.map, at->tab, tabbytes);
	if (gpu_vk_buffer_create(vk, &at->bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		return -1;
	}
	at->have_prog = 1;
	memcpy(at->bprog.map, prog->words, prog->nwords * 4);
	return 0;
}

/* open the data, build the column table, pack the GPU's copy; the program is already built */
static int
arrow_task_setup(gpu_vk_ctx *vk, arrow_task *at, const scan_prog *prog, char *err, size_t errlen)
{
	const kern_data_store *kds_src = at->kds_src;
	size_t		tabbytes;

	at->ncols = prog->words[3];
	if (at->gc)
		return arrow_task_setup_gcache(vk, at, prog, err, errlen);
	if (at->ncols > kds_src->ncols || prog->max_src_resno > (int) kds_src->ncols)
	{
		snprintf(err, errlen, "the program reads more columns than the Arrow store has");
		return -1;
	}
	tabbytes = sizeof(uint32_t) * (ARROW_TAB_COLWORDS * (size_t) at->ncols + ARROW_TAB_EXTRA);
	if (arrow_data_open(&at->ad, kds_src, at->pathname, at->iovec, err, errlen) != 0)
		return -1;
	at->have_data = 1;
	at->cols = calloc(kds_src->ncols ? kds_src->ncols : 1, sizeof(arrow_col));
	at->tab = calloc(1, tabbytes);
	if (!at->cols || !at->tab)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	for (uint32_t c = 0; c < kds_src->ncols; c++)
	{
		const kern_colmeta *cm = &kds_src->colmeta[c];

		if (cm->virtual_offset == 0 && cm->atttypkind == TYPE_KIND__NULL)
			continue;					/* not referenced */
		if (arrow_col_setup(&at->ad.as, cm, &at->cols[c]) != 0)
		{
			snprintf(err, errlen, "column %u: its Arrow buffers are not among the loaded pages", c + 1);
			return -1;
		}
	}
	if (arrow_build_table(&at->ad.as, kds_src, prog, at->cols, at->tab, err, errlen) != 0)
		return -1;
	for (int j = 0; prog->attrs && j < prog->nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];

		if (pa->kind == SCAN_PROJ_SRC && kds_src->colmeta[pa->resno - 1].virtual_offset == 0 &&
			kds_src->colmeta[pa->resno - 1].atttypkind == TYPE_KIND__NULL)
		{
			snprintf(err, errlen, "column %d is projected but was not loaded", pa->resno);
			return -1;
		}
	}
	if (arrow_pack_gpu(vk, &at->ad.as, prog, at->cols, at->tab, &at->bg, err, errlen) != 0)
		return -1;
	at->have_g = 1;
	if (gpu_vk_buffer_create(vk, &at->btab, tabbytes, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate a buffer: %s", vk->err);
		return -1;
	}
	at->have_tab = 1;
	memcpy(at->btab.map, at->tab, tabbytes);
	if (gpu_vk_buffer_create(vk, &at->bdummy, 16, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate a buffer: %s", vk->err);
		return -1;
	}
	at->have_dummy = 1;
	if (gpu_vk_buffer_create(vk, &at->bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		return -1;
	}
	at->have_prog = 1;
	memcpy(at->bprog.map, prog->words, prog->nwords * 4);
	return 0;
}

/* the kernels (ARROW_SRC) see a pass of n rows starting at row 'first' */
static void
arrow_task_pass(arrow_task *at, uint32_t first, uint32_t n)
{
	uint32_t   *m = (uint32_t *) at->btab.map;

	m[ARROW_TAB_COLWORDS * at->ncols] = n;
	m[ARROW_TAB_COLWORDS * at->ncols + 1] = first;
}

/* the message of a kernel error; row = the batch row it names */
static void
arrow_kernel_error(uint32_t code, uint32_t row, char *err, size_t errlen)
{
	char	   *p;

	describe_kernel_error(code, row, 0, err, errlen);
	p = strstr(err, ", block index");
	if (p)
		snprintf(p, errlen - (size_t)(p - err), ", row %u)", row);
}

/* ---------------------------------------------------------------- */
/* the result chunk of a task, which grows as tuples are added */
typedef struct
{
	const kern_data_store *kds_dst_head;
	XpuCommand *resp;
	size_t		cap, usage, resp_sz, head_sz, idx_cap;
	uint64_t   *rowindex;
	uint32_t	nmatch;
} arrow_result;

static int
arrow_result_init(arrow_result *rs, const kern_data_store *kds_dst_head, char *err, size_t errlen)
{
	memset(rs, 0, sizeof(*rs));
	rs->kds_dst_head = kds_dst_head;
	rs->head_sz = KDS_HEAD_LENGTH(kds_dst_head);
	rs->resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
	rs->idx_cap = 1024;
	rs->cap = rs->resp_sz + rs->head_sz + 8 * rs->idx_cap + 64 * rs->idx_cap + 4096;
	rs->resp = calloc(1, rs->cap);
	rs->rowindex = malloc(sizeof(uint64_t) * rs->idx_cap);
	if (!rs->resp || !rs->rowindex)
	{
		snprintf(err, errlen, "out of memory");
		free(rs->resp);
		free(rs->rowindex);
		memset(rs, 0, sizeof(*rs));
		return -1;
	}
	return 0;
}

static void
arrow_result_free(arrow_result *rs)
{
	free(rs->resp);
	free(rs->rowindex);
	memset(rs, 0, sizeof(*rs));
}

/* form the tuple of `row` ('comp': the GPU's values of the row) and append it; -1 on error */
static int
arrow_result_add(arrow_result *rs, arrow_form *af, uint32_t row, const uint32_t *comp,
				 char *err, size_t errlen)
{
	int			sz;
	char	   *dest;

	if (rs->nmatch == rs->idx_cap)
	{
		uint64_t   *ni = realloc(rs->rowindex, sizeof(uint64_t) * rs->idx_cap * 2);

		if (!ni)
		{
			snprintf(err, errlen, "out of memory");
			return -1;
		}
		rs->rowindex = ni;
		rs->idx_cap *= 2;
	}
	/* measure, then make room (the tuple area starts after the row index, whose size is not known
	 * yet: tuples are kept in their own buffer region and moved when the chunk is assembled) */
	sz = arrow_form_tuple(af, row, comp, 1, NULL, rs->nmatch, err, errlen);
	if (sz < 0)
		return -1;
	if (rs->resp_sz + rs->head_sz + rs->usage + MAXALIGN(sz) + 16 > rs->cap)
	{
		size_t		ncap = rs->cap * 2 + MAXALIGN(sz);
		XpuCommand *nr = realloc(rs->resp, ncap);

		if (!nr)
		{
			snprintf(err, errlen, "out of memory");
			return -1;
		}
		rs->resp = nr;
		rs->cap = ncap;
	}
	dest = (char *) rs->resp + rs->resp_sz + rs->head_sz + rs->usage;
	if (arrow_form_tuple(af, row, comp, 0, (kern_tupitem *) dest, rs->nmatch, err, errlen) < 0)
		return -1;
	rs->rowindex[rs->nmatch] = rs->usage;
	rs->usage += MAXALIGN(sz);
	rs->nmatch++;
	return 0;
}

/* assemble the chunk (header, row index, tuples); returns the reply, which the caller owns */
static XpuCommand *
arrow_result_finish(arrow_result *rs, const XpuCommand *task, uint32_t nitems_raw, uint32_t nitems_in,
					uint32_t npages_vfs_read, int keep_empty, size_t *p_len, char *err, size_t errlen)
{
	size_t		idx_sz = MAXALIGN(sizeof(uint64_t) * rs->nmatch);
	size_t		chunk_len = rs->head_sz + idx_sz + rs->usage;
	kern_data_store *kds;
	char	   *tup_area;
	XpuCommand *resp;

	if (rs->resp_sz + chunk_len > rs->cap)
	{
		XpuCommand *nr = realloc(rs->resp, rs->resp_sz + chunk_len);

		if (!nr)
		{
			snprintf(err, errlen, "out of memory");
			return NULL;
		}
		rs->resp = nr;
		rs->cap = rs->resp_sz + chunk_len;
	}
	tup_area = (char *) rs->resp + rs->resp_sz + rs->head_sz;
	memmove(tup_area + idx_sz, tup_area, rs->usage);
	kds = (kern_data_store *)((char *) rs->resp + rs->resp_sz);
	memcpy(kds, rs->kds_dst_head, rs->head_sz);
	kds->length = chunk_len;
	kds->usage = rs->usage;
	kds->nitems = rs->nmatch;
	kds->format = KDS_FORMAT_ROW;
	kds->hash_nslots = 0;
	/* row index: the distance of a tuple's start from the end of the chunk */
	{
		uint64_t   *ri = (uint64_t *)((char *) kds + rs->head_sz);

		for (uint32_t r = 0; r < rs->nmatch; r++)
			ri[r] = rs->usage - rs->rowindex[r];
	}
	resp = rs->resp;
	resp->magic = XpuCommandMagicNumber;
	resp->tag = XpuCommandTag__Success;
	resp->length = rs->resp_sz + chunk_len;
	resp->u.results.scan_relidx = task->u.task.scan_relidx;
	resp->u.results.chunks_offset = rs->resp_sz;
	resp->u.results.chunks_nitems = (rs->nmatch > 0 || keep_empty) ? 1 : 0;
	resp->u.results.nitems_raw = nitems_raw;
	resp->u.results.nitems_in = nitems_in;
	resp->u.results.nitems_out = rs->nmatch;
	resp->u.results.npages_vfs_read = npages_vfs_read;
	if (resp->u.results.chunks_nitems == 0)
		resp->length = rs->resp_sz;
	*p_len = resp->length;
	rs->resp = NULL;
	free(rs->rowindex);
	rs->rowindex = NULL;
	return resp;
}

/* the form context of a task (the scratch arrays of the tuple being formed) */
static int
arrow_form_open(arrow_form *af, const kern_data_store *kds_dst_head, const arrow_task *at,
				const scan_prog *prog, char *err, size_t errlen)
{
	size_t		n = (prog->nattrs ? prog->nattrs : 1);

	memset(af, 0, sizeof(*af));
	af->kds_dst_head = kds_dst_head;
	af->kds_src = at->kds_src;
	af->prog = prog;
	af->as = &at->ad.as;
	af->cols = at->cols;
	af->gc = at->gc;
	af->kind = malloc(n);
	af->ptr = malloc(sizeof(void *) * n);
	af->len = malloc(sizeof(uint32_t) * n);
	af->tmpv = malloc(8 * n);
	af->numv = malloc(SCAN_NUMERIC_MAXLEN * n);
	if (!af->kind || !af->ptr || !af->len || !af->tmpv || !af->numv)
	{
		snprintf(err, errlen, "out of memory");
		return -1;
	}
	return 0;
}

static void
arrow_form_close(arrow_form *af)
{
	free(af->kind);
	free(af->ptr);
	free(af->len);
	free(af->tmpv);
	free(af->numv);
	memset(af, 0, sizeof(*af));
}

/* ---------------------------------------------------------------- */
XpuCommand *
scan_exec_arrow_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
					 const XpuCommand *task, size_t *p_len, char *err, size_t errlen)
{
	gpu_vk_pipeline *sp;
	spec_shape	sh;
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	arrow_task	at;
	gpu_vk_buffer bout;
	int			have_out = 0, have_result = 0, have_form = 0;
	uint32_t	nrows, ncomp, pass_rows;
	int			variant;
	XpuCommand *resp = NULL;
	arrow_result rs;
	arrow_form	af;
	double		t0 = arrow_now_ms(), t1 = 0, tg = 0, tf = 0;

	err[0] = '\0';
	memset(&af, 0, sizeof(af));
	if ((session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__SCAN)
	{
		snprintf(err, errlen, "only GpuScan reads Arrow files in the Vulkan service so far");
		return NULL;
	}
	if (arrow_task_parse(task, &at, err, errlen) != 0)
		return NULL;
	nrows = at.nrows;
	if (!kds_dst_head)
	{
		snprintf(err, errlen, "the scan has no projection");
		return NULL;
	}
	if (!prog->words && scan_prog_build(session, at.kds_src, prog, err, errlen) != 0)
		return NULL;
	ncomp = prog->ncomp;

	variant = (prog->nregs <= 8 ? 0 : prog->nregs <= 16 ? 1 : 2) + (prog->feat == 0 ? 3 : 0);	/* 3-5: lite */
	memset(&sh, 0, sizeof(sh));
	sp = spec_pipe(eng, prog, SPEC_SCAN | SPEC_ARROW, 0, 0, NULL, NULL, &eng->arrow_pipe[variant]);
	if ((sp == &eng->arrow_pipe[variant] || spec_check_enabled(eng)) && !eng->arrow_pipe_ready[variant])
	{
		static const struct { const unsigned char *spv; const unsigned int *len; } atab[6] = {
			{kspv_gpuarrow8, &kspv_gpuarrow8_len}, {kspv_gpuarrow16, &kspv_gpuarrow16_len}, {kspv_gpuarrow32, &kspv_gpuarrow32_len},
			{kspv_gpuarrow8_lite, &kspv_gpuarrow8_lite_len}, {kspv_gpuarrow16_lite, &kspv_gpuarrow16_lite_len},
			{kspv_gpuarrow32_lite, &kspv_gpuarrow32_lite_len},
		};
		const unsigned char *spv = atab[variant].spv;
		size_t		spv_len = *atab[variant].len;

		if (gpu_vk_pipeline_create(vk, &eng->arrow_pipe[variant], spv, spv_len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the Arrow scan pipeline: %s", vk->err);
			return NULL;
		}
		eng->arrow_pipe_ready[variant] = 1;
	}

	/* 1. the data, the column table, the GPU's copy */
	if (arrow_task_setup(vk, &at, prog, err, errlen) != 0)
		goto error;
	pass_rows = ARROW_PASS_ROWS;
	if (pass_rows > ARROW_PASS_WORDS / (1 + 3 * ncomp))
		pass_rows = ARROW_PASS_WORDS / (1 + 3 * ncomp);
	if (pass_rows > nrows)
		pass_rows = nrows ? nrows : 1;
	if (gpu_vk_buffer_create(vk, &bout, ((size_t) 4 + (size_t) pass_rows * (1 + 3 * ncomp)) * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
		goto error;
	}
	have_out = 1;
	t1 = arrow_now_ms();

	/* 2. the passes; the result chunk grows as rows qualify */
	if (arrow_form_open(&af, kds_dst_head, &at, prog, err, errlen) != 0)
		goto error;
	have_form = 1;
	if (arrow_result_init(&rs, kds_dst_head, err, errlen) != 0)
		goto error;
	have_result = 1;
	for (uint32_t base = 0; base < nrows; base += pass_rows)
	{
		uint32_t	n = nrows - base;
		uint32_t   *out = (uint32_t *) bout.map;
		const uint32_t *flags = out + 4;
		const uint32_t *vals;
		double		tp = arrow_now_ms();

		if (n > pass_rows)
			n = pass_rows;
		vals = out + 4 + n;
		memset(out, 0, 16);
		{
			gpu_vk_buffer *bufs[5] = {&at.bg, &bout, &at.bprog, &at.btab, &at.bdummy};

			for (uint32_t g0 = 0; g0 * ARROW_WG < n; g0 += ARROW_WG_MAX)
			{
				uint32_t	groups = (n + ARROW_WG - 1) / ARROW_WG - g0;
				uint32_t	pc[4];

				if (groups > ARROW_WG_MAX)
					groups = ARROW_WG_MAX;
				pc[0] = n;
				pc[1] = g0;
				pc[2] = base;
				pc[3] = n;
				if (spec_dispatch(eng, prog, SPEC_SCAN | SPEC_ARROW, &sh, &eng->arrow_pipe[variant], sp, bufs, pc, groups) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto error;
				}
			}
		}
		tg += arrow_now_ms() - tp;
		if (out[1] != 0)
		{
			arrow_kernel_error(out[1], out[2], err, errlen);
			goto error;
		}
		tp = arrow_now_ms();
		for (uint32_t r = 0; r < n; r++)
		{
			if (!flags[r])
				continue;
			if (arrow_result_add(&rs, &af, base + r, vals + (size_t) r * 3 * ncomp, err, errlen) != 0)
				goto error;
		}
		tf += arrow_now_ms() - tp;
	}
	/* 3. assemble the chunk */
	resp = arrow_result_finish(&rs, task, nrows, rs.nmatch,
							   (uint32_t)((at.kds_src->length > (uint64_t) at.ad.as.head
										   ? at.kds_src->length - at.ad.as.head : 0) / BLCKSZ),
							   1, p_len, err, errlen);
	if (!resp)
		goto error;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: arrow task: %u rows, %s, setup %.1f ms, gpu %.1f ms, results %.1f ms (%u rows)\n",
				nrows, at.ad.as.mapped ? "mapped" : "read", t1 - t0, tg, tf, rs.nmatch);
	arrow_form_close(&af);
	arrow_result_free(&rs);
	gpu_vk_buffer_destroy(vk, &bout);
	arrow_task_close(vk, &at);
	return resp;

error:
	if (have_form)
		arrow_form_close(&af);
	if (have_result)
		arrow_result_free(&rs);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	arrow_task_close(vk, &at);
	if (err[0] == '\0')
		snprintf(err, errlen, "Arrow scan failed");
	return NULL;
}

/* ---------------------------------------------------------------- */
/*
 * GpuPreAgg over an Arrow record batch, with and without GROUP BY: the aggregation kernels of the heap
 * path (agg.slang, gagg.slang) built with -DARROW_SRC, whose "pages" are virtual pages of ARROW_VP
 * rows. Without GROUP BY every workgroup writes one record per aggregate; with GROUP BY a workgroup
 * writes its table, and the rows whose key found no slot go to an overflow list (there is no heap
 * tuple to evaluate again on the CPU). The service merges the results into the session's agg_state
 * exactly as it does for heap tasks.
 */
static int
arrow_ensure_agg_pipeline(scan_engine *eng, int grouped, int variant, int avariant, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } ptab[4][3] = {
		{{kspv_aagg16_4, &kspv_aagg16_4_len}, {kspv_aagg16_8, &kspv_aagg16_8_len},
		 {kspv_aagg16_12, &kspv_aagg16_12_len}},
		{{kspv_aagg32_4, &kspv_aagg32_4_len}, {kspv_aagg32_8, &kspv_aagg32_8_len},
		 {kspv_aagg32_12, &kspv_aagg32_12_len}},
		{{kspv_aagg16_4_lite, &kspv_aagg16_4_lite_len}, {kspv_aagg16_8_lite, &kspv_aagg16_8_lite_len},
		 {kspv_aagg16_12_lite, &kspv_aagg16_12_lite_len}},
		{{kspv_aagg32_4_lite, &kspv_aagg32_4_lite_len}, {kspv_aagg32_8_lite, &kspv_aagg32_8_lite_len},
		 {kspv_aagg32_12_lite, &kspv_aagg32_12_lite_len}},
	};
	static const struct { const unsigned char *spv; const unsigned int *len; } gtab[4][3] = {
		{{kspv_agagg16_4, &kspv_agagg16_4_len}, {kspv_agagg16_8, &kspv_agagg16_8_len},
		 {kspv_agagg16_12, &kspv_agagg16_12_len}},
		{{kspv_agagg32_4, &kspv_agagg32_4_len}, {kspv_agagg32_8, &kspv_agagg32_8_len},
		 {kspv_agagg32_12, &kspv_agagg32_12_len}},
		{{kspv_agagg16_4_lite, &kspv_agagg16_4_lite_len}, {kspv_agagg16_8_lite, &kspv_agagg16_8_lite_len},
		 {kspv_agagg16_12_lite, &kspv_agagg16_12_lite_len}},
		{{kspv_agagg32_4_lite, &kspv_agagg32_4_lite_len}, {kspv_agagg32_8_lite, &kspv_agagg32_8_lite_len},
		 {kspv_agagg32_12_lite, &kspv_agagg32_12_lite_len}},
	};
	int		   *ready = (grouped ? &eng->agagg_ready[variant][avariant] : &eng->aagg_ready[variant][avariant]);
	gpu_vk_pipeline *pipe = (grouped ? &eng->agagg_pipe[variant][avariant] : &eng->aagg_pipe[variant][avariant]);

	if (!*ready)
	{
		const unsigned char *spv = (grouped ? gtab[variant][avariant].spv : ptab[variant][avariant].spv);
		unsigned int len = (grouped ? *gtab[variant][avariant].len : *ptab[variant][avariant].len);

		if (gpu_vk_pipeline_create(vk, pipe, spv, len, 5, 20) != 0)
		{
			snprintf(err, errlen, "cannot create the Arrow aggregation pipeline: %s", vk->err);
			return -1;
		}
		*ready = 1;
	}
	return 0;
}

int
scan_exec_arrow_agg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
						 agg_state *st, const XpuCommand *task, char *err, size_t errlen)
{
	gpu_vk_pipeline *sp;
	spec_shape	sh;
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	arrow_task	at;
	gpu_vk_buffer bout;
	int			have_out = 0, variant, avariant, grouped, rv = -1;
	gpu_vk_pipeline *vmpipe;
	spec_keystats ks;
	uint32_t	nrows, nagg, ppw, pass_rows;
	uint32_t	cap = 0, rec_words = 0, blk_words = 0, ov_rec = 0, ovf_cap = 0, ovf_count = 0;
	uint64_t	n_overflow = 0;
	double		t0 = arrow_now_ms(), t1 = 0, tg = 0, tm = 0;

	err[0] = '\0';
	if (arrow_task_parse(task, &at, err, errlen) != 0)
		return -1;
	nrows = at.nrows;
	if (!prog->words && scan_prog_build(session, at.kds_src, prog, err, errlen) != 0)
		return -1;
	if (!prog->agg_mode)
	{
		snprintf(err, errlen, "not an aggregation program");
		return -1;
	}
	nagg = prog->nagg;
	grouped = (prog->nkeys > 0);
	scan_agg_state_prepare(st, nagg);
	variant = (prog->nregs <= 16 ? 0 : 1) + (prog->feat == 0 ? 2 : 0);		/* 2-3: lite */
	avariant = (nagg <= 4 ? 0 : nagg <= 8 ? 1 : 2);
	if (grouped)
		scan_group_variant(avariant, &cap, &rec_words, &ov_rec);
	vmpipe = (grouped ? &eng->agagg_pipe[variant][avariant] : &eng->aagg_pipe[variant][avariant]);
	memset(&sh, 0, sizeof(sh));
	sh.agg_max = 4 * (avariant + 1);
	sh.ga_c = cap;
	scan_group_keystats(st, prog->nkeys, &ks);
	sp = spec_pipe(eng, prog, (grouped ? SPEC_GAGG : SPEC_AGG) | SPEC_ARROW, sh.agg_max, sh.ga_c, &ks, &sh, vmpipe);
	if (grouped)
	{
		cap = sh.ga_c;
		rec_words = 2 + 2 * SCAN_GROUP_KEYS + 6 * sh.agg_max;
		ov_rec = 1 + 2 * SCAN_GROUP_KEYS + 3 * sh.agg_max;
		blk_words = 1 + cap * rec_words;
	}
	if ((sp == vmpipe || spec_check_enabled(eng)) &&
		arrow_ensure_agg_pipeline(eng, grouped, variant, avariant, err, errlen) != 0)
		return -1;
	if (nrows == 0)
		return 0;
	if (arrow_task_setup(vk, &at, prog, err, errlen) != 0)
		goto done;
	t1 = arrow_now_ms();
	ppw = ARROW_AGG_PAGES_PER_WG;
	st->nitems_raw += nrows;
	/* a pass is limited by the size of the per-workgroup records */
	pass_rows = nrows;
	if (grouped)
	{
		uint64_t	max_wg = (uint64_t)(64U << 20) / blk_words;		/* at most 256 MB of tables */
		uint64_t	max_rows = max_wg * ppw * ARROW_VP;

		if (pass_rows > max_rows)
			pass_rows = (uint32_t) max_rows;
	}
	for (uint32_t first = 0; first < nrows; first += pass_rows)
	{
		uint32_t	n = nrows - first, pages, ngroups;
		size_t		out_words, ovf_base = 0;
		const uint32_t *out = NULL;
		gpu_vk_pipeline *pipe = vmpipe;
		gpu_vk_buffer *bufs[5];
		double		tp;

		if (n > pass_rows)
			n = pass_rows;
		pages = (n + ARROW_VP - 1) / ARROW_VP;
		ngroups = (pages + ppw - 1) / ppw;
		arrow_task_pass(&at, first, n);
		bufs[0] = &at.bg;
		bufs[1] = &bout;
		bufs[2] = &at.bprog;
		bufs[3] = &at.btab;
		bufs[4] = &at.bdummy;
		ovf_cap = 4096;
		have_out = 0;
		tp = arrow_now_ms();
		for (int attempt = 0; attempt < 4; attempt++)
		{
			if (grouped)
			{
				ovf_base = 4 + (size_t) ngroups * blk_words;
				out_words = ovf_base + 1 + (size_t) ovf_cap * ov_rec;
			}
			else
				out_words = 4 + (size_t) ngroups * nagg * 6;
			if (out_words > 0xffffffffULL / 8 ||
				gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
			{
				snprintf(err, errlen, "cannot allocate the output buffer (%zu words): %s", out_words, vk->err);
				goto done;
			}
			have_out = 1;
			bufs[1] = &bout;
			out = (const uint32_t *) bout.map;
			memset(bout.map, 0, 16);
			if (grouped)
				((uint32_t *) bout.map)[ovf_base] = 0;
			for (uint32_t base = 0; base < ngroups; base += ARROW_WG_MAX)
			{
				uint32_t	groups = ngroups - base;
				uint32_t	pc[5];

				if (groups > ARROW_WG_MAX)
					groups = ARROW_WG_MAX;
				pc[0] = pages;
				pc[1] = base;				/* first workgroup of this dispatch */
				pc[2] = ppw;
				pc[3] = (uint32_t) ovf_base;
				pc[4] = ovf_cap;
				if (spec_dispatch(eng, prog, (grouped ? SPEC_GAGG : SPEC_AGG) | SPEC_ARROW, &sh, pipe, sp, bufs, pc, groups) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto done;
				}
			}
			if (out[1] != 0)
			{
				arrow_kernel_error(out[1], first + out[2] * ARROW_VP + out[3], err, errlen);
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
			snprintf(err, errlen, "the aggregation overflowed its list repeatedly");
			goto done;
		}
		tg += arrow_now_ms() - tp;
		tp = arrow_now_ms();
		if (!grouped)
			scan_agg_merge_workgroups(prog, st, out, ngroups);
		else
		{
			if (scan_group_merge_blocks(prog, st, out, ngroups, cap, rec_words, blk_words) != 0 ||
				scan_group_merge_overflow(prog, st, out + ovf_base + 1, ovf_count, ov_rec) != 0)
			{
				snprintf(err, errlen, "out of memory");
				goto done;
			}
			n_overflow += ovf_count;
		}
		tm += arrow_now_ms() - tp;
		gpu_vk_buffer_destroy(vk, &bout);
		have_out = 0;
	}
	st->npages_vfs_read += (uint32_t)((at.kds_src->length > (uint64_t) at.ad.as.head
									   ? at.kds_src->length - at.ad.as.head : 0) / BLCKSZ);
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: arrow%s agg task: %u rows, %s, setup %.1f ms, gpu %.1f ms, merge %.1f ms, overflow rows %llu\n",
				grouped ? " group" : "", nrows, at.ad.as.mapped ? "mapped" : "read", t1 - t0, tg, tm,
				(unsigned long long) n_overflow);
	rv = 0;
done:
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	arrow_task_close(vk, &at);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "Arrow aggregation failed");
	return rv;
}

/* ---------------------------------------------------------------- */
/*
 * GpuJoin with an Arrow record batch as the outer relation: join.slang built with -DARROW_SRC. The
 * match records name the row (virtual page, row in it) and the inner items; the result tuples are
 * formed from the Arrow buffers (outer columns), the inner items and the GPU's computed values. The
 * batch is joined in passes of ARROW_JOIN_PASS_ROWS rows (the output holds one pass's matches).
 */
XpuCommand *
scan_exec_arrow_join_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
						  join_inner *ji, const XpuCommand *task,
						  size_t *p_len, char *err, size_t errlen)
{
	gpu_vk_pipeline *sp;
	spec_shape	sh;
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	arrow_task	at;
	gpu_vk_buffer bout;
	int			have_out = 0, have_result = 0, have_form = 0, variant;
	uint32_t	nrows, ncomp, rec_words;
	XpuCommand *resp = NULL;
	arrow_result rs;
	arrow_form	af;
	double		t0 = arrow_now_ms(), t1 = 0, tg = 0, tf = 0;

	err[0] = '\0';
	memset(&af, 0, sizeof(af));
	if (!ji->ready)
	{
		snprintf(err, errlen, "the inner buffer is not attached");
		return NULL;
	}
	if (arrow_task_parse(task, &at, err, errlen) != 0)
		return NULL;
	nrows = at.nrows;
	if (!kds_dst_head)
	{
		snprintf(err, errlen, "the join has no projection");
		return NULL;
	}
	if (!prog->words &&
		scan_prog_build_join(session, at.kds_src, ji->rels, ji->nrels, prog, err, errlen) != 0)
		return NULL;
	if (!prog->join_mode)
	{
		snprintf(err, errlen, "not a join program");
		return NULL;
	}
	ncomp = prog->ncomp;
	rec_words = 2 + prog->ndepth + 3 * ncomp;
	/* variants 0-1: the full kernels by register count; 2-3: the lite kernels (see scan_exec.c) */
	variant = (prog->nregs <= 16 ? 0 : 1) + (prog->feat == 0 ? 2 : 0);
	memset(&sh, 0, sizeof(sh));
	sp = spec_pipe(eng, prog, SPEC_JOIN | SPEC_ARROW, 0, 0, NULL, NULL, &eng->ajoin_pipe[variant]);
	if ((sp == &eng->ajoin_pipe[variant] || spec_check_enabled(eng)) && !eng->ajoin_ready[variant])
	{
		const unsigned char *spv = (variant == 0 ? kspv_ajoin16 : variant == 1 ? kspv_ajoin32 :
									variant == 2 ? kspv_ajoin16_lite : kspv_ajoin32_lite);
		size_t		spv_len = (variant == 0 ? kspv_ajoin16_len : variant == 1 ? kspv_ajoin32_len :
							   variant == 2 ? kspv_ajoin16_lite_len : kspv_ajoin32_lite_len);

		if (gpu_vk_pipeline_create(vk, &eng->ajoin_pipe[variant], spv, spv_len, 6, 20) != 0)
		{
			snprintf(err, errlen, "cannot create the Arrow join pipeline: %s", vk->err);
			return NULL;
		}
		eng->ajoin_ready[variant] = 1;
	}
	if (arrow_task_setup(vk, &at, prog, err, errlen) != 0)
		goto error;
	t1 = arrow_now_ms();
	if (arrow_form_open(&af, kds_dst_head, &at, prog, err, errlen) != 0)
		goto error;
	have_form = 1;
	af.ndepth = prog->ndepth;
	for (int d = 1; d <= prog->ndepth; d++)
		af.kds_in[d] = ji->rels[d - 1].kds;
	if (arrow_result_init(&rs, kds_dst_head, err, errlen) != 0)
		goto error;
	have_result = 1;

	for (uint32_t first = 0; first < nrows; first += ARROW_JOIN_PASS_ROWS)
	{
		uint32_t	n = nrows - first, pages, cap, nmatch = 0;
		const uint32_t *out = NULL;
		double		tp = arrow_now_ms();

		if (n > ARROW_JOIN_PASS_ROWS)
			n = ARROW_JOIN_PASS_ROWS;
		pages = (n + ARROW_VP - 1) / ARROW_VP;
		arrow_task_pass(&at, first, n);
		/* run; if there are more matches than the output holds, run again with room for all */
		cap = (n > 4096 ? n : 4096);
		for (int attempt = 0; attempt < 2; attempt++)
		{
			gpu_vk_buffer *bufs[6] = {&at.bg, &bout, &at.bprog, &at.btab, &at.bdummy, &ji->buf};
			size_t		out_words = 4 + (size_t) prog->oj_words + (size_t) cap * rec_words;

			if (gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
			{
				snprintf(err, errlen, "cannot allocate the output buffer: %s", vk->err);
				goto error;
			}
			have_out = 1;
			out = (const uint32_t *) bout.map;
			memset(bout.map, 0, 16 + (size_t) prog->oj_words * 4);
			for (uint32_t base = 0; base < pages; base += ARROW_WG_MAX)
			{
				uint32_t	groups = pages - base;
				uint32_t	pc[5];

				if (groups > ARROW_WG_MAX)
					groups = ARROW_WG_MAX;
				pc[0] = pages;
				pc[1] = base;
				pc[2] = 0;
				pc[3] = cap;
				pc[4] = 0;
				if (spec_dispatch(eng, prog, SPEC_JOIN | SPEC_ARROW, &sh, &eng->ajoin_pipe[variant], sp, bufs, pc, groups) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto error;
				}
			}
			if (out[1] != 0)
			{
				arrow_kernel_error(out[1], first + out[2] * ARROW_VP + out[3], err, errlen);
				goto error;
			}
			nmatch = out[0];
			if (nmatch <= cap)
				break;
			cap = nmatch;						/* the exact number of matches */
			gpu_vk_buffer_destroy(vk, &bout);
			have_out = 0;
		}
		if (!have_out)
		{
			snprintf(err, errlen, "the join produced more matches than the output could hold");
			goto error;
		}
		tg += arrow_now_ms() - tp;
		tp = arrow_now_ms();
		if (prog->oj_words > 0)
			scan_join_merge_ojmap(prog, ji, out, 4);
		for (uint32_t r = 0; r < nmatch; r++)
		{
			const uint32_t *rec = &out[4 + (size_t) prog->oj_words + (size_t) r * rec_words];
			uint32_t	row = first + rec[0] * ARROW_VP + rec[1];

			if (rec[0] >= pages || rec[1] >= ARROW_VP || row >= nrows)
			{
				snprintf(err, errlen, "corrupt match record");
				goto error;
			}
			af.inner[0] = NULL;
			for (int d = 1; d <= prog->ndepth; d++)
			{
				uint32_t	item = rec[1 + d];

				if (item != 0xffffffffu && item >= ji->maplen)
				{
					snprintf(err, errlen, "corrupt match record");
					goto error;
				}
				af.inner[d] = (item == 0xffffffffu ? NULL : (const unsigned char *) ji->map + item);
			}
			if (arrow_result_add(&rs, &af, row, rec + 2 + prog->ndepth, err, errlen) != 0)
				goto error;
		}
		tf += arrow_now_ms() - tp;
		gpu_vk_buffer_destroy(vk, &bout);
		have_out = 0;
	}
	resp = arrow_result_finish(&rs, task, nrows, nrows,
							   (uint32_t)((at.kds_src->length > (uint64_t) at.ad.as.head
										   ? at.kds_src->length - at.ad.as.head : 0) / BLCKSZ),
							   0, p_len, err, errlen);
	if (!resp)
		goto error;
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: arrow join task: %u rows, %s, setup %.1f ms, gpu %.1f ms, results %.1f ms (%u matches)\n",
				nrows, at.ad.as.mapped ? "mapped" : "read", t1 - t0, tg, tf, rs.nmatch);
	arrow_form_close(&af);
	arrow_result_free(&rs);
	arrow_task_close(vk, &at);
	return resp;

error:
	if (have_form)
		arrow_form_close(&af);
	if (have_result)
		arrow_result_free(&rs);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	arrow_task_close(vk, &at);
	if (err[0] == '\0')
		snprintf(err, errlen, "Arrow join failed");
	return NULL;
}

/* ---------------------------------------------------------------- */
/* GpuPreAgg on a join over an Arrow record batch: jagg.slang / jgagg.slang built with -DARROW_SRC */
static int
arrow_ensure_joinagg_pipeline(scan_engine *eng, int grouped, int variant, int avariant, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	static const struct { const unsigned char *spv; const unsigned int *len; } jtab[4][3] = {
		{{kspv_ajagg16_4, &kspv_ajagg16_4_len}, {kspv_ajagg16_8, &kspv_ajagg16_8_len},
		 {kspv_ajagg16_12, &kspv_ajagg16_12_len}},
		{{kspv_ajagg32_4, &kspv_ajagg32_4_len}, {kspv_ajagg32_8, &kspv_ajagg32_8_len},
		 {kspv_ajagg32_12, &kspv_ajagg32_12_len}},
		{{kspv_ajagg16_4_lite, &kspv_ajagg16_4_lite_len}, {kspv_ajagg16_8_lite, &kspv_ajagg16_8_lite_len},
		 {kspv_ajagg16_12_lite, &kspv_ajagg16_12_lite_len}},
		{{kspv_ajagg32_4_lite, &kspv_ajagg32_4_lite_len}, {kspv_ajagg32_8_lite, &kspv_ajagg32_8_lite_len},
		 {kspv_ajagg32_12_lite, &kspv_ajagg32_12_lite_len}},
	};
	static const struct { const unsigned char *spv; const unsigned int *len; } gtab[4][3] = {
		{{kspv_ajgagg16_4, &kspv_ajgagg16_4_len}, {kspv_ajgagg16_8, &kspv_ajgagg16_8_len},
		 {kspv_ajgagg16_12, &kspv_ajgagg16_12_len}},
		{{kspv_ajgagg32_4, &kspv_ajgagg32_4_len}, {kspv_ajgagg32_8, &kspv_ajgagg32_8_len},
		 {kspv_ajgagg32_12, &kspv_ajgagg32_12_len}},
		{{kspv_ajgagg16_4_lite, &kspv_ajgagg16_4_lite_len}, {kspv_ajgagg16_8_lite, &kspv_ajgagg16_8_lite_len},
		 {kspv_ajgagg16_12_lite, &kspv_ajgagg16_12_lite_len}},
		{{kspv_ajgagg32_4_lite, &kspv_ajgagg32_4_lite_len}, {kspv_ajgagg32_8_lite, &kspv_ajgagg32_8_lite_len},
		 {kspv_ajgagg32_12_lite, &kspv_ajgagg32_12_lite_len}},
	};
	int		   *ready = (grouped ? &eng->ajgagg_ready[variant][avariant] : &eng->ajagg_ready[variant][avariant]);
	gpu_vk_pipeline *pipe = (grouped ? &eng->ajgagg_pipe[variant][avariant] : &eng->ajagg_pipe[variant][avariant]);

	if (!*ready)
	{
		const unsigned char *spv = (grouped ? gtab[variant][avariant].spv : jtab[variant][avariant].spv);
		unsigned int len = (grouped ? *gtab[variant][avariant].len : *jtab[variant][avariant].len);

		if (gpu_vk_pipeline_create(vk, pipe, spv, len, 6, 20) != 0)
		{
			snprintf(err, errlen, "cannot create the Arrow join aggregation pipeline: %s", vk->err);
			return -1;
		}
		*ready = 1;
	}
	return 0;
}

int
scan_exec_arrow_joinagg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
							 join_inner *ji, agg_state *st, const XpuCommand *task,
							 char *err, size_t errlen)
{
	gpu_vk_pipeline *sp;
	spec_shape	sh;
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	arrow_task	at;
	gpu_vk_buffer bout;
	int			have_out = 0, variant, avariant, grouped, rv = -1;
	gpu_vk_pipeline *vmpipe;
	spec_keystats ks;
	uint32_t	nrows, nagg, ppw, pass_rows;
	uint32_t	cap = 0, rec_words = 0, blk_words = 0, ov_rec = 0, ovf_cap = 0, ovf_count = 0;
	uint64_t	n_overflow = 0;
	double		t0 = arrow_now_ms(), t1 = 0, tg = 0, tm = 0;

	err[0] = '\0';
	if (!ji->ready)
	{
		snprintf(err, errlen, "the inner buffer is not attached");
		return -1;
	}
	if (arrow_task_parse(task, &at, err, errlen) != 0)
		return -1;
	nrows = at.nrows;
	if (!prog->words &&
		scan_prog_build_join(session, at.kds_src, ji->rels, ji->nrels, prog, err, errlen) != 0)
		return -1;
	if (!prog->join_mode || !prog->agg_mode)
	{
		snprintf(err, errlen, "not a join aggregation program");
		return -1;
	}
	nagg = prog->nagg;
	grouped = (prog->nkeys > 0);
	scan_agg_state_prepare(st, nagg);
	variant = (prog->nregs <= 16 ? 0 : 1) + (prog->feat == 0 ? 2 : 0);		/* 2-3: lite */
	avariant = (nagg <= 4 ? 0 : nagg <= 8 ? 1 : 2);
	if (grouped)
		scan_group_variant(avariant, &cap, &rec_words, &ov_rec);
	vmpipe = (grouped ? &eng->ajgagg_pipe[variant][avariant] : &eng->ajagg_pipe[variant][avariant]);
	memset(&sh, 0, sizeof(sh));
	sh.agg_max = 4 * (avariant + 1);
	sh.ga_c = cap;
	scan_group_keystats(st, prog->nkeys, &ks);
	sp = spec_pipe(eng, prog, (grouped ? SPEC_JGAGG : SPEC_JAGG) | SPEC_ARROW, sh.agg_max, sh.ga_c, &ks, &sh, vmpipe);
	if (grouped)
	{
		cap = sh.ga_c;
		rec_words = 2 + 2 * SCAN_GROUP_KEYS + 6 * sh.agg_max;
		ov_rec = 1 + 2 * SCAN_GROUP_KEYS + 3 * sh.agg_max;
		blk_words = 1 + cap * rec_words;
	}
	if ((sp == vmpipe || spec_check_enabled(eng)) &&
		arrow_ensure_joinagg_pipeline(eng, grouped, variant, avariant, err, errlen) != 0)
		return -1;
	if (nrows == 0)
		return 0;
	if (arrow_task_setup(vk, &at, prog, err, errlen) != 0)
		goto done;
	t1 = arrow_now_ms();
	ppw = (grouped ? ARROW_JGAGG_PAGES_PER_WG : ARROW_JAGG_PAGES_PER_WG);
	st->nitems_raw += nrows;
	pass_rows = nrows;
	if (grouped)
	{
		uint64_t	max_wg = (uint64_t)(64U << 20) / blk_words;
		uint64_t	max_rows = max_wg * ppw * ARROW_VP;

		if (pass_rows > max_rows)
			pass_rows = (uint32_t) max_rows;
	}
	for (uint32_t first = 0; first < nrows; first += pass_rows)
	{
		uint32_t	n = nrows - first, pages, ngroups;
		size_t		out_words, ovf_base = 0, oj_base = 0;
		const uint32_t *out = NULL;
		gpu_vk_pipeline *pipe = vmpipe;
		gpu_vk_buffer *bufs[6];
		double		tp;

		if (n > pass_rows)
			n = pass_rows;
		pages = (n + ARROW_VP - 1) / ARROW_VP;
		ngroups = (pages + ppw - 1) / ppw;
		arrow_task_pass(&at, first, n);
		bufs[0] = &at.bg;
		bufs[2] = &at.bprog;
		bufs[3] = &at.btab;
		bufs[4] = &at.bdummy;
		bufs[5] = &ji->buf;
		ovf_cap = 4096;
		have_out = 0;
		tp = arrow_now_ms();
		for (int attempt = 0; attempt < 4; attempt++)
		{
			if (grouped)
			{
				ovf_base = 4 + (size_t) ngroups * blk_words;
				out_words = ovf_base + 1 + (size_t) ovf_cap * ov_rec;
			}
			else
				out_words = 4 + (size_t) ngroups * nagg * 6;
			/* RIGHT/FULL joins: the outer-join map follows the results */
			oj_base = out_words;
			out_words += prog->oj_words;
			if (out_words > 0xffffffffULL / 8 ||
				gpu_vk_buffer_create(vk, &bout, out_words * 4, 0) != 0)
			{
				snprintf(err, errlen, "cannot allocate the output buffer (%zu words): %s", out_words, vk->err);
				goto done;
			}
			have_out = 1;
			bufs[1] = &bout;
			out = (const uint32_t *) bout.map;
			memset(bout.map, 0, 16);
			if (grouped)
				((uint32_t *) bout.map)[ovf_base] = 0;
			if (prog->oj_words > 0)
			{
				memset((uint32_t *) bout.map + oj_base, 0, (size_t) prog->oj_words * 4);
				memcpy(at.bprog.map, prog->words, prog->nwords * 4);
				scan_prog_ojmap_place(prog, (uint32_t *) at.bprog.map, oj_base);
			}
			for (uint32_t base = 0; base < ngroups; base += ARROW_JOINAGG_DISPATCH_GROUPS)
			{
				uint32_t	groups = ngroups - base;
				uint32_t	pc[5];

				if (groups > ARROW_JOINAGG_DISPATCH_GROUPS)
					groups = ARROW_JOINAGG_DISPATCH_GROUPS;
				pc[0] = pages;
				pc[1] = base;
				pc[2] = ppw;
				pc[3] = (uint32_t) ovf_base;
				pc[4] = ovf_cap;
				if (spec_dispatch(eng, prog, (grouped ? SPEC_JGAGG : SPEC_JAGG) | SPEC_ARROW, &sh, pipe, sp, bufs, pc, groups) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto done;
				}
			}
			if (out[1] != 0)
			{
				arrow_kernel_error(out[1], first + out[2] * ARROW_VP + out[3], err, errlen);
				goto done;
			}
			ovf_count = grouped ? out[ovf_base] : 0;
			if (ovf_count <= ovf_cap)
				break;
			ovf_cap = ovf_count + ovf_count / 2 + 1024;
			gpu_vk_buffer_destroy(vk, &bout);
			have_out = 0;
		}
		if (!have_out)
		{
			snprintf(err, errlen, "the join aggregation overflowed its list repeatedly");
			goto done;
		}
		tg += arrow_now_ms() - tp;
		tp = arrow_now_ms();
		if (prog->oj_words > 0)
			scan_join_merge_ojmap(prog, ji, out, oj_base);
		if (!grouped)
			scan_agg_merge_workgroups(prog, st, out, ngroups);
		else
		{
			if (scan_group_merge_blocks(prog, st, out, ngroups, cap, rec_words, blk_words) != 0 ||
				scan_group_merge_overflow(prog, st, out + ovf_base + 1, ovf_count, ov_rec) != 0)
			{
				snprintf(err, errlen, "out of memory");
				goto done;
			}
			n_overflow += ovf_count;
		}
		tm += arrow_now_ms() - tp;
		gpu_vk_buffer_destroy(vk, &bout);
		have_out = 0;
	}
	st->npages_vfs_read += (uint32_t)((at.kds_src->length > (uint64_t) at.ad.as.head
									   ? at.kds_src->length - at.ad.as.head : 0) / BLCKSZ);
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: arrow join%s agg task: %u rows, %s, setup %.1f ms, gpu %.1f ms, merge %.1f ms, overflow rows %llu\n",
				grouped ? " group" : "", nrows, at.ad.as.mapped ? "mapped" : "read", t1 - t0, tg, tm,
				(unsigned long long) n_overflow);
	rv = 0;
done:
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	arrow_task_close(vk, &at);
	if (rv != 0 && err[0] == '\0')
		snprintf(err, errlen, "Arrow aggregation over a join failed");
	return rv;
}
