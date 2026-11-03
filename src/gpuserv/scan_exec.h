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
	gpu_vk_pipeline	gagg_pipe[2][3];		/* GROUP BY: [registers 16/32][aggregates 4/8/12] */
	int				gagg_pipe_ready[2][3];
	int				trace;
	int				no_map;			/* PGSTROM_GPUSERV_NO_MAP: always copy the pages */
} scan_engine;

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
struct group_table;
typedef struct agg_state
{
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

#endif	/* GPUSERV_SCAN_EXEC_H */
