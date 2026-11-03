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

typedef struct scan_engine
{
	scan_map_entry	maps[SCAN_MAP_CACHE];
	uint64_t		map_clock;
	gpu_vk_ctx	   *vk;
	gpu_vk_pipeline	pipe[3];			/* kernel variants with 8, 16 and 32 registers */
	int				pipe_ready[3];
	gpu_vk_pipeline	agg_pipe[3][3];		/* the aggregation kernel: [registers 8/16/32][aggregates 4/8/12] */
	int				agg_pipe_ready[3][3];
	gpu_vk_pipeline	join_pipe[3];			/* the join kernel: registers 8/16/32 */
	int				join_pipe_ready[3];
	gpu_vk_pipeline	gagg_pipe[2][3];		/* GROUP BY: [registers 16/32][aggregates 4/8/12] */
	int				gagg_pipe_ready[2][3];
	gpu_vk_pipeline	jagg_pipe[2][3];		/* aggregation over a join: [registers 16/32][aggregates 4/8/12] */
	int				jagg_pipe_ready[2][3];
	gpu_vk_pipeline	jgagg_pipe[2][3];		/* GROUP BY over a join: the same */
	int				jgagg_pipe_ready[2][3];
	gpu_vk_pipeline	arrow_pipe[3];			/* the Arrow scan kernel: registers 8/16/32 */
	int				arrow_pipe_ready[3];
	/* Arrow record batches as the outer relation (registers 16/32; aggregates 4/8/12) */
	gpu_vk_pipeline	aagg_pipe[2][3], agagg_pipe[2][3], ajagg_pipe[2][3], ajgagg_pipe[2][3];
	int				aagg_ready[2][3], agagg_ready[2][3], ajagg_ready[2][3], ajgagg_ready[2][3];
	gpu_vk_pipeline	ajoin_pipe[2];
	int				ajoin_ready[2];
	int				trace;
	int				no_map;			/* PGSTROM_GPUSERV_NO_MAP: always copy the pages */
} scan_engine;

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

struct group_table;
typedef struct agg_state
{
	recheck_buf	rck;				/* rows for the backend, sent with the final reply */
	struct group_table *groups;		/* GROUP BY: the groups of the session (NULL otherwise) */
	int			nagg;
	uint64_t	n[SCAN_AGG_MAX];		/* count, number of values, or "has a value" (min/max) */
	__int128	sum[SCAN_AGG_MAX];		/* sums, exact */
	int64_t		val[SCAN_AGG_MAX];		/* current min/max */
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
void	agg_state_free(agg_state *st);

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
int		scan_group_merge_blocks(const scan_prog *prog, agg_state *st, const uint32_t *out, uint32_t ngroups_total,
								uint32_t cap, uint32_t rec_words, uint32_t blk_words);
int		scan_group_merge_overflow(const scan_prog *prog, agg_state *st, const uint32_t *recs, uint32_t count,
								  uint32_t ov_rec);
void	scan_join_merge_ojmap(const scan_prog *prog, const join_inner *ji, const uint32_t *out);

/* Arrow record batches as the outer relation (arrow_exec.c): aggregation, GROUP BY, join, join aggregation */
int		scan_exec_arrow_agg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
								 agg_state *st, const XpuCommand *task, char *err, size_t errlen);
XpuCommand *scan_exec_arrow_join_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
									  join_inner *ji, const XpuCommand *task,
									  size_t *p_len, char *err, size_t errlen);
int		scan_exec_arrow_joinagg_task(scan_engine *eng, const XpuCommand *session_cmd, scan_prog *prog,
									 join_inner *ji, agg_state *st, const XpuCommand *task,
									 char *err, size_t errlen);

#endif	/* GPUSERV_SCAN_EXEC_H */
