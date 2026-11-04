/*
 * gcache_proto.h
 *
 * GPU-Cache: the contract between the backends (src/gpu_cache.c) and the GPU service
 * (src/gpuserv/gcache.c). Plain C, no PostgreSQL headers: it is compiled into both sides.
 *
 * A backend talks to the service with XpuCommand frames of tag XpuCommandTag__GpuCacheOp. The
 * request is a kern_gcache_op (xpu_common.h) followed by 'data'; the service answers with a frame of
 * tag Success (status in kern_gcache_op.status) or Error. Requests are synchronous and need no
 * session, so every backend keeps one connection for them.
 *
 *   SETUP   data = gcache_setup_hdr + ncols * gcache_colspec: (re)create the cache of a table
 *   LOG     data = a sequence of records (gcache_rec_hdr and body), applied in order
 *   ERASE   forget the cache of the table (all signatures if sig == 0)
 *   INFO    reply data = gcache_info
 *   COMPACT compact the variable-length regions of the cache
 *   STAT    reply data = gcache_info (the cache exists or status NOT_FOUND)
 *
 * The cache holds the latest committed state of the table plus the rows of transactions in flight
 * (their xmin / xmax are the transactions' ids): a record INSERT carries the row version with the
 * xmin / xmax the backend knows, DELETE sets the xmax, and COMMIT / ABORT of a transaction id
 * finalizes every row that transaction touched. A scan sees the rows whose xmin is committed (or one of
 * its own transaction ids) and whose xmax is not committed or one of its own ids.
 */
#ifndef GCACHE_PROTO_H
#define GCACHE_PROTO_H
#include <stdint.h>

#define GCOP_SETUP			1
#define GCOP_LOG			2
#define GCOP_ERASE			3
#define GCOP_INFO			4
#define GCOP_COMPACT		5
#define GCOP_STAT			6

/* kern_gcache_op.status of a reply */
#define GCST_OK				0
#define GCST_NOT_FOUND		1		/* no such cache (never built, or the service restarted) */
#define GCST_OVERFLOW		2		/* max_num_rows or the data region is full: the cache was dropped */
#define GCST_INCONSISTENT	3		/* a log record did not fit the cache's state: the cache was dropped */
#define GCST_BAD_REQUEST	4

#define GCACHE_MAXCOLS		512

/* value types the GPU kernels read (the others are only passed through) */
#define GCVT_NONE			0
#define GCVT_BOOL			1
#define GCVT_INT			2		/* int1/2/4/8, date, time, timestamp, timestamptz */
#define GCVT_FLOAT4			3
#define GCVT_STR			4		/* text, varchar, bpchar */
#define GCVT_FLOAT8			5		/* float8 */

typedef struct
{
	int16_t		attlen;				/* > 0: fixed, -1: varlena */
	uint8_t		attbyval;
	uint8_t		attalign;
	uint8_t		dropped;
	uint8_t		vt;					/* GCVT_* */
	uint16_t	reserved;
	uint32_t	typid;
	uint32_t	avgwidth;			/* estimated bytes per value (varlena) */
} gcache_colspec;

typedef struct
{
	uint32_t	ncols;
	uint32_t	flags;
	uint64_t	max_rows;			/* rows the cache can hold (including versions in flight) */
	uint64_t	cache_size;			/* byte budget of the cache */
	/* ncols * gcache_colspec follow */
} gcache_setup_hdr;

/* log records: 8-byte aligned, len includes the header */
#define GCREC_INSERT		1
#define GCREC_DELETE		2
#define GCREC_COMMIT		3
#define GCREC_ABORT			4

typedef struct
{
	uint32_t	len;
	uint16_t	type;
	uint16_t	natts;				/* INSERT: attributes in the body */
} gcache_rec_hdr;

/* INSERT body: tags[natts] padded to 8 bytes, then per non-NULL attribute its value:
 *   tag 0: NULL (or dropped), no value
 *   tag 1: a fixed-length value of attlen bytes
 *   tag 2: a varlena: uint32 length, then the complete datum (header included, detoasted) */
typedef struct
{
	uint32_t	xmin;
	uint32_t	xmax;
	uint32_t	blkno;
	uint16_t	offnum;
	uint16_t	reserved;
} gcache_rec_insert;

/* DELETE body */
typedef struct
{
	uint32_t	xid;
	uint32_t	blkno;
	uint16_t	offnum;
	uint16_t	reserved;
	uint32_t	reserved2;
} gcache_rec_delete;

/* COMMIT / ABORT body */
typedef struct
{
	uint32_t	xid;
	uint32_t	reserved;
} gcache_rec_xact;

/* reply of INFO / STAT */
typedef struct
{
	uint64_t	max_rows;
	uint64_t	nitems;				/* rows holding a version (committed or in flight) */
	uint64_t	nfree;				/* free row slots below the high-water mark */
	uint64_t	hwm;				/* high-water mark of row slots */
	uint64_t	usage;				/* bytes of the variable-length regions in use */
	uint64_t	dead_space;			/* bytes of those regions that belong to released rows */
	uint64_t	virtual_mem_sz;		/* size of the cache's device buffer */
	uint64_t	physical_mem_sz;	/* bytes of it in use */
	uint64_t	pending_rows;		/* versions of transactions in flight */
	uint64_t	nrecords;			/* log records applied so far */
} gcache_info;

#endif	/* GCACHE_PROTO_H */
