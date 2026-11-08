/*
 * scan_exec.h
 *
 * Executes one XpuTaskExec of a GpuScan on the Vulkan device: uploads the heap pages,
 * runs the scan kernel, and assembles the result tuples into a KDS_FORMAT_ROW chunk.
 */
#ifndef GPUSERV_SCAN_EXEC_H
#define GPUSERV_SCAN_EXEC_H
#include <stddef.h>
#include "xpu_common.h"
#include "gpu_vk.h"
#include "scan_prog.h"
#include "specialize.h"

#define SCAN_MAP_CACHE		4

/* a window of a relation file, mapped and imported into Vulkan, kept between tasks */
typedef struct scan_map_entry
{
	void		   *map;
	size_t			maplen;
	gpu_vk_buffer	buf;
	uint64_t		dev, ino;
	int64_t			win_start;
	int64_t			file_size;
	uint64_t		last_used;
} scan_map_entry;

/*
 * The VM kernel pipelines, created on first use. Every kernel family exists in SCAN_NTIERS tiers (scan_caps.h: full, lite,
 * nogeom), and the first index of an array is (register variant) + (register variants) * tier, with tier = scan_feat_tier(
 * prog->feat); NV3 is the count of the families with register files of 8/16/32, NV2 that of 16/32. Each pipeline array has a
 * "ready" twin of the same shape: scan_pipes_destroy() checks that at compile time.
 */
#define SCAN_NV3	(3 * SCAN_NTIERS)
#define SCAN_NV2	(2 * SCAN_NTIERS)

typedef struct scan_engine
{
	scan_map_entry	maps[SCAN_MAP_CACHE];
	uint64_t		map_clock;
	gpu_vk_ctx	   *vk;
	gpu_vk_pipeline	pipe[SCAN_NV3];			/* the scan kernel: [registers 8/16/32 + 3 * tier] */
	int				pipe_ready[SCAN_NV3];
	gpu_vk_pipeline	agg_pipe[SCAN_NV3][3];	/* the aggregation kernel: [registers 8/16/32 + 3 * tier][aggregates 4/8/12] */
	int				agg_pipe_ready[SCAN_NV3][3];
	gpu_vk_pipeline	join_pipe[SCAN_NV3];	/* the join kernel: [registers 8/16/32 + 3 * tier] */
	int				join_pipe_ready[SCAN_NV3];
	gpu_vk_pipeline	gagg_pipe[SCAN_NV2][3];	/* GROUP BY: [registers 16/32 + 2 * tier][aggregates 4/8/12] */
	int				gagg_pipe_ready[SCAN_NV2][3];
	gpu_vk_pipeline	gtab_pipe[SCAN_NV2];	/* GROUP BY over many groups: [registers 16/32 + 2 * tier] */
	int				gtab_pipe_ready[SCAN_NV2];
	gpu_vk_pipeline	agtab_pipe[SCAN_NV2];	/* the same on an Arrow record batch */
	int				agtab_pipe_ready[SCAN_NV2];
	gpu_vk_pipeline	jagg_pipe[SCAN_NV2][3];	/* aggregation over a join: the same */
	int				jagg_pipe_ready[SCAN_NV2][3];
	gpu_vk_pipeline	jgagg_pipe[SCAN_NV2][3];	/* GROUP BY over a join: the same */
	int				jgagg_pipe_ready[SCAN_NV2][3];
	gpu_vk_pipeline	arrow_pipe[SCAN_NV3];	/* the Arrow scan kernel: [registers 8/16/32 + 3 * tier] */
	int				arrow_pipe_ready[SCAN_NV3];
	gpu_vk_pipeline	sort_pipe;				/* GPU-Sort (gpusort.c) */
	int				sort_pipe_ready;

	/* Arrow record batches as the outer relation (registers 16/32 + 2 * tier; aggregates 4/8/12) */
	gpu_vk_pipeline	aagg_pipe[SCAN_NV2][3], agagg_pipe[SCAN_NV2][3], ajagg_pipe[SCAN_NV2][3], ajgagg_pipe[SCAN_NV2][3];
	int				aagg_ready[SCAN_NV2][3], agagg_ready[SCAN_NV2][3], ajagg_ready[SCAN_NV2][3], ajgagg_ready[SCAN_NV2][3];
	gpu_vk_pipeline	ajoin_pipe[SCAN_NV2];	/* registers 16/32 + 2 * tier */
	int				ajoin_ready[SCAN_NV2];
	int				trace;
	int				no_map;			/* PGSTROM_GPUSERV_NO_MAP: always copy the pages */
	struct spec_mgr *spec;			/* A6: specialized kernels (specialize.h) */
} scan_engine;

/*
 * The aggregation kernels exist for 4 and for 12 aggregate slots: a Metal first-compile of a full kernel costs about
 * 5 minutes whatever the slot count (measured 2026-11-07), so a third, 8-slot, set only doubled the kernels to warm.
 * Index 1 of the variant tables is kept (it points to the 12-slot kernel) and is never selected.
 */
static inline int
scan_agg_variant(int nagg)
{
	return nagg <= 4 ? 0 : 2;
}

void	describe_kernel_error(uint32_t err, uint32_t page, uint32_t lp, char *buf, size_t buflen);

/* Arrow_Fdw scans (arrow_exec.c) */
XpuCommand *scan_exec_arrow_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
								 const XpuCommand *task, size_t *p_len, char *err, size_t errlen);
void	arrow_engine_fini(scan_engine *eng);

void	scan_engine_init(scan_engine *eng, gpu_vk_ctx *vk);
void	scan_engine_fini(scan_engine *eng);

/*
 * Run a task. 'prog' caches the program of the session (built on first use). On success
 * returns a malloc'ed reply frame (length in *p_len); on failure returns NULL with a
 * message in err.
 */
XpuCommand *scan_exec_task(scan_engine *eng,
								  const XpuCommand *session_cmd,
								  scan_prog *prog,
								  const XpuCommand *task,
								  size_t *p_len, char *err, size_t errlen);

/*
 * Aggregation without GROUP BY: the session's running partial aggregates. Tasks add to it;
 * the reply to XpuTaskFinal is the one row of partial states.
 */
/*
 * Rows the GPU could not evaluate (a compressed or external string was needed): raw heap tuples
 * the backend evaluates on the CPU. Returned as a KDS_FORMAT_RECHECK chunk.
 */
typedef struct recheck_buf
{
	unsigned char *buf;
	size_t		len, cap;
	uint32_t	nitems;
} recheck_buf;

/* the rows for the backend (scan_exec.c): heap tuples, Arrow rows (rck_add_arrow) and GPU-Cache rows (a synthesized heap tuple) */
int		rck_add(recheck_buf *r, uint32_t blkno, uint32_t offnum, const unsigned char *htup, uint32_t len);
int		rck_add_arrow(recheck_buf *r, uint32_t rb_flat, uint32_t relidx, uint32_t row);
size_t	rck_chunk_size(const kern_data_store *kds_head, const recheck_buf *r);
void	rck_chunk_fill(kern_data_store *kds, const kern_data_store *kds_head, const recheck_buf *r);

struct group_table;
struct gtab_state;
typedef struct agg_state
{
	recheck_buf	rck;				/* rows for the backend, sent with the final reply */
	struct group_table *groups;		/* GROUP BY: the groups of the session (NULL otherwise) */
	struct gtab_state *gt;			/* GROUP BY over many groups: the hash table in device memory (gtab.slang), NULL if not used */
	int			nagg;
	uint64_t	n[SCAN_AGG_MAX];		/* count, number of values, or "has a value" (min/max) */
	__int128	sum[SCAN_AGG_MAX];		/* sums, exact */
	int64_t		val[SCAN_AGG_MAX];		/* current min/max (float8: the bits of the min, max or sum) */
	uint8_t		fovf[SCAN_AGG_MAX];		/* a float8 sum overflowed (PostgreSQL raises an error for it) */
	uint64_t	nitems_raw, nitems_in;	/* statistics */
	uint32_t	npages_vfs_read;
} agg_state;

/* Run one task of a GpuPreAgg session and add its results to st. 0 on success. */
int		scan_exec_agg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
						   agg_state *st, const XpuCommand *task, char *err, size_t errlen);
/* GROUP BY: the same, with the groups kept in st->groups. */
int		scan_exec_group_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
							 agg_state *st, const XpuCommand *task, char *err, size_t errlen);
/*
 * The reply to XpuTaskFinal: the partial aggregate row (one row per group with GROUP BY).
 * malloc'ed, NULL on error.
 */
XpuCommand *scan_exec_agg_final(const XpuCommand *session_cmd, const scan_prog *prog,
								const agg_state *st, size_t *p_len, char *err, size_t errlen);
/* GPU-Sort (gpusort.c): the same with the groups sorted, cut or window-filtered as the session's sort keys say */
XpuCommand *scan_exec_agg_final_sorted(scan_engine *eng, const XpuCommand *session_cmd, const scan_prog *prog,
									   const agg_state *st, size_t *p_len, char *err, size_t errlen);
void	agg_state_free(agg_state *st);
/* GROUP BY over many groups: the session-wide hash table (scan_exec.c, kernels_vm/gtab.slang) used by the Arrow task too */
int		scan_gtab_pipeline(scan_engine *eng, int arrow, int variant, gpu_vk_pipeline **p_pipe, char *err, size_t errlen);
int		scan_gtab_open(gpu_vk_ctx *vk, const kern_session_info *session, const scan_prog *prog, agg_state *st, char *err, size_t errlen);
gpu_vk_buffer *scan_gtab_buffer(agg_state *st);
int		scan_gtab_task_done(gpu_vk_ctx *vk, const scan_prog *prog, agg_state *st, char *err, size_t errlen);
int		scan_gtab_merge_overflow(const scan_prog *prog, agg_state *st, const uint32_t *recs, uint32_t count);
int		scan_gtab_forced(const agg_state *st, const scan_prog *prog);
int		scan_gtab_switch_allowed(void);
int		scan_group_flush(const scan_prog *prog, agg_state *st, char *err, size_t errlen);

/*
 * GpuJoin: the inner relation's buffer (the backend's shared-memory hash table), mapped once per
 * session and imported into Vulkan.
 */
typedef struct join_inner
{
	int			ready;
	void	   *map;
	size_t		maplen;
	gpu_vk_buffer buf;
	int			own_buf;			/* buf holds a copy (the import was not possible) */
	int			nrels;					/* inner relations (1 ...) */
	scan_join_rel rels[SCAN_JOIN_MAXDEPTH];
	/* RIGHT/FULL joins: the backend's outer-join map of each relation (writable shared memory) */
	void	   *ojmap;
	size_t		ojmaplen;
	uint8_t	   *oj[SCAN_JOIN_MAXDEPTH + 1];
} join_inner;

int		scan_exec_join_open(scan_engine *eng, const XpuCommand *session_cmd, join_inner *ji,
							char *err, size_t errlen);
void	join_inner_free(scan_engine *eng, join_inner *ji);
/*
 * GpuPreAgg on top of GpuJoin: the join kernel aggregates the joined rows on the device (with or
 * without GROUP BY, as the program says); the results are added to st. 0 on success.
 */
int		scan_exec_joinagg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
							   join_inner *ji, agg_state *st, const XpuCommand *task,
							   char *err, size_t errlen);
XpuCommand *scan_exec_join_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
								join_inner *ji, const XpuCommand *task,
								size_t *p_len, char *err, size_t errlen);

/* exports of scan_exec.c for the aggregation and join tasks over Arrow record batches (arrow_exec.c) */
void	scan_agg_state_prepare(agg_state *st, int nagg);
void	scan_agg_merge_workgroups(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total);
void	scan_group_variant(int avariant, uint32_t *cap, uint32_t *rec_words, uint32_t *ov_rec);
/* the keys of the groups the session has so far (what a specialized GROUP BY kernel is made for) */
void	scan_group_keystats(const agg_state *st, int nkeys, spec_keystats *ks);
int		scan_group_merge_blocks(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total,
								uint32_t cap, uint32_t rec_words, uint32_t blk_words);
int		scan_group_merge_overflow(const scan_prog *prog, agg_state *st, const uint32_t *recs, uint32_t count,
								  uint32_t ov_rec);
void	scan_join_merge_ojmap(const scan_prog *prog, const join_inner *ji, const uint32_t *out, size_t base);

/* Arrow record batches as the outer relation (arrow_exec.c): aggregation, GROUP BY, join, join aggregation */
int		scan_exec_arrow_agg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
								 agg_state *st, const XpuCommand *task, char *err, size_t errlen);
XpuCommand *scan_exec_arrow_join_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
									  join_inner *ji, const XpuCommand *task,
									  size_t *p_len, char *err, size_t errlen);
int		scan_exec_arrow_joinagg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
									 join_inner *ji, agg_state *st, const XpuCommand *task,
									 char *err, size_t errlen);

/* destroy the created pipelines of an array and its "ready" twin (any shape); both must have the same number of elements */
void	scan_pipes_destroy(gpu_vk_ctx *vk, gpu_vk_pipeline *pipes, int *ready, size_t npipes);
#define SCAN_PIPES_DESTROY(eng, pipes, ready) \
	do { \
		_Static_assert(sizeof((eng)->pipes) / sizeof(gpu_vk_pipeline) == sizeof((eng)->ready) / sizeof(int), \
					   #pipes " and " #ready " differ in size"); \
		scan_pipes_destroy((eng)->vk, (gpu_vk_pipeline *)(eng)->pipes, (int *)(eng)->ready, \
						   sizeof((eng)->ready) / sizeof(int)); \
	} while (0)

#endif	/* GPUSERV_SCAN_EXEC_H */
