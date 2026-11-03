/*
 * gcache.h
 *
 * GPU-Cache in the GPU service: a columnar copy of a table that lives in a device buffer, kept
 * current by the redo log the backends send (gcache_proto.h), and read by the Arrow kernels as a
 * "record batch" of row slots (arrow_exec.c). See gcache.c.
 */
#ifndef GPUSERV_GCACHE_H
#define GPUSERV_GCACHE_H
#include <stddef.h>
#include <stdint.h>
#include "xpu_common.h"
#include "gpu_vk.h"
#include "scan_prog.h"

typedef struct gcache	gcache;

/* mode of a column in the kernel's column table for a string stored as { offset, length } pairs
 * (kernels_vm/arrow_load.inc) */
#define GCACHE_ACM_PAIRS	5

/* results of gcache_fetch(), the same numbers as FETCH_* of arrow_exec.c */
#define GCF_NULL			0
#define GCF_FIXED			1
#define GCF_RAW_VARLENA		3

/* the session of the task being run (set by the wire layer): its transaction ids decide what a scan sees */
extern const kern_session_info *gcache_cur_session;

void		gcache_init(gpu_vk_ctx *vk);
void		gcache_fini(void);
/* one GpuCacheOp request: returns the malloc'ed reply frame, NULL on a memory failure */
XpuCommand *gcache_handle_op(const XpuCommand *xcmd, size_t *p_len);

/* the cache a task reads, or NULL */
gcache	   *gcache_find(uint32_t database_oid, uint32_t table_oid, uint32_t table_sig);
/* rows (slots) a task has to look at */
uint32_t	gcache_nrows(const gcache *g);
uint32_t	gcache_nlive(const gcache *g);
gpu_vk_buffer *gcache_data_buffer(gcache *g);
/*
 * The row-visibility bitmap for a task of a session whose transaction ids are xids[0 ... nxids-1].
 * Returns 0 and fills *bitmap: the cache's own bitmap (own = 0), or a temporary buffer that the caller
 * must destroy (own = 1). A temporary one is made only if the session's own versions change what is
 * visible. -1 on failure.
 */
int			gcache_visibility(gcache *g, const uint32_t *xids, int nxids, gpu_vk_buffer *bitmap, int *own,
							  char *err, size_t errlen);
/* the column table (8 words per column of the program, see arrow_load.inc); -1 on a type the GPU cannot read */
int			gcache_build_table(const gcache *g, const scan_prog *prog, uint32_t *tab, char *err, size_t errlen);
/* a column's value of a row for the result tuple: GCF_*; the bytes are in *p_ptr (tmp is 8 bytes of scratch) */
int			gcache_fetch(const gcache *g, int colno, uint32_t row, unsigned char *tmp,
						 const unsigned char **p_ptr, uint32_t *p_len, char *err, size_t errlen);
/* ctid of a row (6 bytes) */
void		gcache_ctid(const gcache *g, uint32_t row, unsigned char *out);
uint32_t	gcache_ncols(const gcache *g);

#endif	/* GPUSERV_GCACHE_H */
