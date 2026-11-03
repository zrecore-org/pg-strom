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
#include "scan_exec.h"
#include "scan_caps.h"

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
	/* scratch of the row being formed (one entry per result column) */
	uint8_t	   *kind;
	const unsigned char **ptr;
	uint32_t   *len;
	unsigned char (*tmpv)[8];
} arrow_form;

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
		if (pa->kind == SCAN_PROJ_SYS)
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
			int		rv = arrow_fetch(af->as, &af->cols[pa->resno - 1], &af->kds_src->colmeta[pa->resno - 1],
									 row, tmpv[j], &ptr[j], &len[j], err, errlen);

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
		else if (pa->kind == SCAN_PROJ_COMP)
		{
			uint32_t	lo = comp[3 * pa->comp + 1];
			uint32_t	hi = comp[3 * pa->comp + 2];
			uint64_t	v = (uint64_t) lo | ((uint64_t) hi << 32);

			if (comp[3 * pa->comp] != 0)
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
	for (int i = 0; i < 3; i++)
	{
		if (eng->arrow_pipe_ready[i])
			gpu_vk_pipeline_destroy(eng->vk, &eng->arrow_pipe[i]);
		eng->arrow_pipe_ready[i] = 0;
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
		default:			return "?";
	}
}

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

	memset(tab, 0, sizeof(uint32_t) * 8 * (ncols ? ncols : 1));
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = prog->words[coloff + c];
		uint32_t	slot = (desc >> 16) & 0xff;
		uint32_t	vt = (desc >> 12) & 0xf;
		const kern_colmeta *cm = &kds->colmeta[c];
		const arrow_col *ac = &cols[c];
		uint32_t   *t = tab + 8 * c;
		uint32_t	w = 0;

		t[0] = 0xffffffffU;
		if (slot == 0xff)
			continue;
		if (vt != 0 && vt != SOP_VT_BOOL && vt != SOP_VT_INT && vt != SOP_VT_FLOAT4 && vt != SOP_VT_NULLONLY)
		{
			snprintf(err, errlen, "column %u: the expression reads a value type the Arrow scan does not load", c + 1);
			return -1;
		}
		if (ac->mode == 1)
		{
			t[4] = 1;
			continue;
		}
		if (ac->mode == 2)
		{
			uint64_t	v = 0;

			if (!ac->vconst)
			{
				t[4] = 1;
				continue;
			}
			if (vt == SOP_VT_NULLONLY)
			{
				t[4] = 2;
				continue;
			}
			if (!cm->attbyval || cm->attlen > 8 || cm->attlen <= 0)
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
			t[4] = 2;
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
				if (cm->attopts.tag != ArrowType__Int || cm->attopts.unitsz != cm->attlen ||
					(cm->attlen != 1 && cm->attlen != 2 && cm->attlen != 4 && cm->attlen != 8))
					goto badtype;
				w = cm->attlen;
				break;
			default:	/* SOP_VT_FLOAT4 */
				if (cm->attopts.tag != ArrowType__FloatingPoint || cm->attopts.unitsz != 4)
					goto badtype;
				w = 4;
				break;
		}
		if (ac->v_off < 0)
		{
			t[4] = 1;				/* no values at all: every row is NULL */
			continue;
		}
		t[2] = (uint32_t) ac->v_off;
		t[3] = w;
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
 * The GPU's copy: the validity bitmaps and values of the columns the kernel reads, packed one after
 * the other (8-byte aligned) in one buffer; the table gets the offsets in it. Columns that are only
 * passed through are not copied (the CPU reads them from the file).
 */
static int
arrow_pack_gpu(gpu_vk_ctx *vk, const arrow_src *as, const scan_prog *prog, const arrow_col *cols,
			   uint32_t *tab, gpu_vk_buffer *bg, char *err, size_t errlen)
{
	uint32_t	coloff = prog->words[0];
	uint32_t	ncols = prog->words[3];
	size_t		total = 16;
	unsigned char *dst;

	for (int pass = 0; pass < 2; pass++)
	{
		size_t		pos = 0;

		for (uint32_t c = 0; c < ncols; c++)
		{
			uint32_t	slot = (prog->words[coloff + c] >> 16) & 0xff;
			uint32_t   *t = tab + 8 * c;
			const arrow_col *ac = &cols[c];

			if (slot == 0xff || t[4] != 0)
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

XpuCommand *
scan_exec_arrow_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
					 const XpuCommand *task, size_t *p_len, char *err, size_t errlen)
{
	gpu_vk_ctx *vk = eng->vk;
	const kern_session_info *session = &session_cmd->u.session;
	const kern_data_store *kds_dst_head = SESSION_KDS_DST_HEAD(session);
	const kern_data_store *kds_src;
	const char *pathname = NULL;
	const io_vector *iovec = NULL;
	arrow_data	ad;
	arrow_col  *cols = NULL;
	uint32_t   *tab = NULL;
	gpu_vk_buffer bout, bprog, btab, bdummy, bg;
	int			have_data = 0, have_g = 0, have_out = 0, have_prog = 0, have_tab = 0, have_dummy = 0;
	uint32_t	nrows, ncomp, ncols, pass_rows, nmatch = 0;
	int			variant;
	XpuCommand *resp = NULL;
	size_t		head_sz, resp_sz, cap, usage = 0, idx_cap;
	uint64_t   *rowindex = NULL;
	arrow_form	af;
	double		t0 = arrow_now_ms(), t1 = 0, tg = 0, tf = 0;

	err[0] = '\0';
	if ((session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__SCAN)
	{
		snprintf(err, errlen, "only GpuScan reads Arrow files in the Vulkan service so far");
		return NULL;
	}
	kds_src = (const kern_data_store *)((const char *) task + task->u.task.kds_src_offset);
	if (kds_src->format != KDS_FORMAT_ARROW)
	{
		snprintf(err, errlen, "source format '%c' is not implemented by the Vulkan GpuScan yet", kds_src->format);
		return NULL;
	}
	if (task->u.task.kds_src_iovec)
	{
		/* the file's pages (Arrow IPC) */
		if (task->u.task.kds_src_pathname)
			pathname = (const char *) task + task->u.task.kds_src_pathname;
		iovec = (const io_vector *)((const char *) task + task->u.task.kds_src_iovec);
	}
	else if (task->u.task.kds_src_offset + kds_src->length > task->length)
	{
		snprintf(err, errlen, "corrupted Arrow data store");
		return NULL;
	}
	nrows = kds_src->nitems;
	if (!kds_dst_head)
	{
		snprintf(err, errlen, "the scan has no projection");
		return NULL;
	}
	if (!prog->words && scan_prog_build(session, kds_src, prog, err, errlen) != 0)
		return NULL;
	ncomp = prog->ncomp;
	ncols = prog->words[3];
	if (ncols > kds_src->ncols || prog->max_src_resno > (int) kds_src->ncols)
	{
		snprintf(err, errlen, "the program reads more columns than the Arrow store has");
		return NULL;
	}

	variant = (prog->nregs <= 8 ? 0 : prog->nregs <= 16 ? 1 : 2);
	if (!eng->arrow_pipe_ready[variant])
	{
		const unsigned char *spv = (variant == 0 ? kspv_gpuarrow8 : variant == 1 ? kspv_gpuarrow16 : kspv_gpuarrow32);
		size_t		spv_len = (variant == 0 ? kspv_gpuarrow8_len : variant == 1 ? kspv_gpuarrow16_len : kspv_gpuarrow32_len);

		if (gpu_vk_pipeline_create(vk, &eng->arrow_pipe[variant], spv, spv_len, 5, 16) != 0)
		{
			snprintf(err, errlen, "cannot create the Arrow scan pipeline: %s", vk->err);
			return NULL;
		}
		eng->arrow_pipe_ready[variant] = 1;
	}

	/* 1. the data and the column table */
	if (arrow_data_open(&ad, kds_src, pathname, iovec, err, errlen) != 0)
		return NULL;
	have_data = 1;
	cols = calloc(kds_src->ncols ? kds_src->ncols : 1, sizeof(arrow_col));
	tab = calloc(8 * (size_t)(ncols ? ncols : 1), sizeof(uint32_t));
	if (!cols || !tab)
	{
		snprintf(err, errlen, "out of memory");
		goto error;
	}
	for (uint32_t c = 0; c < kds_src->ncols; c++)
	{
		const kern_colmeta *cm = &kds_src->colmeta[c];

		if (cm->virtual_offset == 0 && cm->atttypkind == TYPE_KIND__NULL)
			continue;					/* not referenced */
		if (arrow_col_setup(&ad.as, cm, &cols[c]) != 0)
		{
			snprintf(err, errlen, "column %u: its Arrow buffers are not among the loaded pages", c + 1);
			goto error;
		}
	}
	if (arrow_build_table(&ad.as, kds_src, prog, cols, tab, err, errlen) != 0)
		goto error;
	for (int j = 0; j < prog->nattrs; j++)
	{
		const scan_proj_attr *pa = &prog->attrs[j];

		if (pa->kind == SCAN_PROJ_SRC && kds_src->colmeta[pa->resno - 1].virtual_offset == 0 &&
			kds_src->colmeta[pa->resno - 1].atttypkind == TYPE_KIND__NULL)
		{
			snprintf(err, errlen, "column %d is projected but was not loaded", pa->resno);
			goto error;
		}
	}
	if (arrow_pack_gpu(vk, &ad.as, prog, cols, tab, &bg, err, errlen) != 0)
		goto error;
	have_g = 1;
	if (gpu_vk_buffer_create(vk, &btab, 4 * 8 * (size_t)(ncols ? ncols : 1), 0) != 0 ||
		(have_tab = 1, gpu_vk_buffer_create(vk, &bdummy, 16, 0) != 0))
	{
		snprintf(err, errlen, "cannot allocate a buffer: %s", vk->err);
		goto error;
	}
	have_dummy = 1;
	memcpy(btab.map, tab, 4 * 8 * (size_t)(ncols ? ncols : 1));
	if (gpu_vk_buffer_create(vk, &bprog, prog->nwords * 4, 0) != 0)
	{
		snprintf(err, errlen, "cannot allocate the program buffer: %s", vk->err);
		goto error;
	}
	have_prog = 1;
	memcpy(bprog.map, prog->words, prog->nwords * 4);

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

	/* 2. the passes; the result chunk grows as rows qualify (tuples forward, the row index fixed at the end) */
	memset(&af, 0, sizeof(af));
	af.kds_dst_head = kds_dst_head;
	af.kds_src = kds_src;
	af.prog = prog;
	af.as = &ad.as;
	af.cols = cols;
	af.kind = malloc(prog->nattrs ? prog->nattrs : 1);
	af.ptr = malloc(sizeof(void *) * (prog->nattrs ? prog->nattrs : 1));
	af.len = malloc(sizeof(uint32_t) * (prog->nattrs ? prog->nattrs : 1));
	af.tmpv = malloc(8 * (size_t)(prog->nattrs ? prog->nattrs : 1));
	if (!af.kind || !af.ptr || !af.len || !af.tmpv)
	{
		snprintf(err, errlen, "out of memory");
		goto error;
	}
	head_sz = KDS_HEAD_LENGTH(kds_dst_head);
	resp_sz = MAXALIGN(offsetof(XpuCommand, u.results.stats[0]));
	idx_cap = 1024;
	cap = resp_sz + head_sz + 8 * idx_cap + 64 * (size_t) idx_cap + 4096;
	resp = calloc(1, cap);
	rowindex = malloc(sizeof(uint64_t) * idx_cap);
	if (!resp || !rowindex)
	{
		snprintf(err, errlen, "out of memory");
		goto error;
	}
	for (uint32_t base = 0; base < nrows; base += pass_rows)
	{
		uint32_t	n = nrows - base;
		uint32_t   *out = (uint32_t *) bout.map;
		const uint32_t *flags = out + 4;
		const uint32_t *vals = out + 4 + n;
		double		tp = arrow_now_ms();

		if (n > pass_rows)
			n = pass_rows;
		vals = out + 4 + n;
		memset(out, 0, 16);
		{
			gpu_vk_buffer *bufs[5] = {&bg, &bout, &bprog, &btab, &bdummy};

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
				if (gpu_vk_dispatch(vk, &eng->arrow_pipe[variant], bufs, pc, groups, 1, 1, NULL) != 0)
				{
					snprintf(err, errlen, "GPU dispatch failed: %s", vk->err);
					goto error;
				}
			}
		}
		tg += arrow_now_ms() - tp;
		if (out[1] != 0)
		{
			char	*p;

			describe_kernel_error(out[1], out[2], 0, err, errlen);
			p = strstr(err, ", block index");
			if (p)
				snprintf(p, errlen - (size_t)(p - err), ", row %u)", out[2]);
			goto error;
		}
		tp = arrow_now_ms();
		for (uint32_t r = 0; r < n; r++)
		{
			int			sz;
			char	   *dest;

			if (!flags[r])
				continue;
			if (nmatch == idx_cap)
			{
				uint64_t   *ni = realloc(rowindex, sizeof(uint64_t) * idx_cap * 2);

				if (!ni)
				{
					snprintf(err, errlen, "out of memory");
					goto error;
				}
				rowindex = ni;
				idx_cap *= 2;
			}
			/* measure, then make room (the tuple area starts after the row index, whose size is not known
			 * yet: tuples are kept in their own buffer region and moved when the chunk is assembled) */
			sz = arrow_form_tuple(&af, base + r, vals + (size_t) r * 3 * ncomp, 1, NULL, nmatch, err, errlen);
			if (sz < 0)
				goto error;
			if (resp_sz + head_sz + usage + MAXALIGN(sz) + 16 > cap)
			{
				size_t		ncap = cap * 2 + MAXALIGN(sz);
				XpuCommand *nr = realloc(resp, ncap);

				if (!nr)
				{
					snprintf(err, errlen, "out of memory");
					goto error;
				}
				resp = nr;
				cap = ncap;
			}
			dest = (char *) resp + resp_sz + head_sz + usage;
			if (arrow_form_tuple(&af, base + r, vals + (size_t) r * 3 * ncomp, 0,
								 (kern_tupitem *) dest, nmatch, err, errlen) < 0)
				goto error;
			rowindex[nmatch] = usage;
			usage += MAXALIGN(sz);
			nmatch++;
		}
		tf += arrow_now_ms() - tp;
	}
	/* 3. assemble the chunk: header, row index, tuples */
	{
		size_t		idx_sz = MAXALIGN(sizeof(uint64_t) * nmatch);
		size_t		chunk_len = head_sz + idx_sz + usage;
		kern_data_store *kds;
		char	   *tup_area;

		if (resp_sz + chunk_len > cap)
		{
			XpuCommand *nr = realloc(resp, resp_sz + chunk_len);

			if (!nr)
			{
				snprintf(err, errlen, "out of memory");
				goto error;
			}
			resp = nr;
		}
		tup_area = (char *) resp + resp_sz + head_sz;
		memmove(tup_area + idx_sz, tup_area, usage);
		kds = (kern_data_store *)((char *) resp + resp_sz);
		memcpy(kds, kds_dst_head, head_sz);
		kds->length = chunk_len;
		kds->usage = usage;
		kds->nitems = nmatch;
		kds->format = KDS_FORMAT_ROW;
		kds->hash_nslots = 0;
		/* row index: the distance of a tuple's start from the end of the chunk */
		{
			uint64_t   *ri = (uint64_t *)((char *) kds + head_sz);

			for (uint32_t r = 0; r < nmatch; r++)
				ri[r] = usage - rowindex[r];
		}
		resp->magic = XpuCommandMagicNumber;
		resp->tag = XpuCommandTag__Success;
		resp->length = resp_sz + chunk_len;
		resp->u.results.scan_relidx = task->u.task.scan_relidx;
		resp->u.results.chunks_offset = resp_sz;
		resp->u.results.chunks_nitems = 1;
		resp->u.results.nitems_raw = nrows;
		resp->u.results.nitems_in = nmatch;
		resp->u.results.nitems_out = nmatch;
		resp->u.results.npages_vfs_read = (uint32_t)((kds_src->length > (uint64_t) ad.as.head
													  ? kds_src->length - ad.as.head : 0) / BLCKSZ);
		*p_len = resp->length;
	}
	if (eng->trace)
		fprintf(stderr, "pgstrom-gpuserv: arrow task: %u rows, %s, setup %.1f ms, gpu %.1f ms, results %.1f ms (%u rows)\n",
				nrows, ad.as.mapped ? "mapped" : "read", t1 - t0, tg, tf, nmatch);
	free(af.kind); free(af.ptr); free(af.len); free(af.tmpv);
	free(rowindex);
	free(cols);
	free(tab);
	gpu_vk_buffer_destroy(vk, &bout);
	gpu_vk_buffer_destroy(vk, &bprog);
	gpu_vk_buffer_destroy(vk, &btab);
	gpu_vk_buffer_destroy(vk, &bdummy);
	gpu_vk_buffer_destroy(vk, &bg);
	arrow_data_close(&ad);
	return resp;

error:
	free(af.kind); free(af.ptr); free(af.len); free(af.tmpv);
	free(rowindex);
	free(resp);
	free(cols);
	free(tab);
	if (have_out)
		gpu_vk_buffer_destroy(vk, &bout);
	if (have_prog)
		gpu_vk_buffer_destroy(vk, &bprog);
	if (have_tab)
		gpu_vk_buffer_destroy(vk, &btab);
	if (have_dummy)
		gpu_vk_buffer_destroy(vk, &bdummy);
	if (have_g)
		gpu_vk_buffer_destroy(vk, &bg);
	if (have_data)
		arrow_data_close(&ad);
	if (err[0] == '\0')
		snprintf(err, errlen, "Arrow scan failed");
	return NULL;
}
