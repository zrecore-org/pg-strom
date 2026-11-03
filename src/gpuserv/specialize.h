/*
 * specialize.h
 *
 * A6: kernels generated from a program of the scan VM (see specialize.c).
 *
 * A session's program (scan_prog: instruction stream, column tables, aggregate tables) is turned into
 * Slang source in which the registers are constant-indexed and every instruction is straight-line code
 * (spec_gen.inc, included by the kernel at its SPEC_PROG hook), compiled by slangc in a background
 * thread while the VM serves the first tasks, cached on disk, and switched to when ready.
 */
#ifndef GPUSERV_SPECIALIZE_H
#define GPUSERV_SPECIALIZE_H
#include <stddef.h>
#include <stdint.h>
#include "gpu_vk.h"

struct scan_prog;
struct scan_engine;

/* which kernel a program runs in */
enum
{
	SPEC_SCAN = 1,			/* scan.slang */
	SPEC_AGG,				/* agg.slang */
	SPEC_GAGG,				/* gagg.slang */
	SPEC_JOIN,				/* join.slang */
	SPEC_JAGG,				/* jagg.slang */
	SPEC_JGAGG				/* jgagg.slang */
};

/* a kind or'ed with SPEC_ARROW: the same kernel with an Arrow record batch as the outer relation (-DARROW_SRC) */
#define SPEC_ARROW		0x10

typedef struct spec_shape
{
	int			kind;
	int			arrow;		/* the outer relation is an Arrow record batch */
	int			agg_max;	/* AGG_MAX / GA_NA of the kernel variant (the layout of the output) */
	int			ga_c;		/* GA_C: slots of a workgroup's group table */
	/*
	 * Direct-mapped GROUP BY (gagg_dm.slang, jgagg_dm.slang): the keys are integers from a small window, a
	 * group's slot is computed from its key, so no rows are staged and no barrier separates the rounds. Key k
	 * maps to slot index key - dm_base[k] in [0, dm_w[k]); NULL maps to dm_w[k]; a row with a key outside
	 * the window is aggregated on the CPU. ga_c is the product of the (dm_w + 1).
	 */
	int			dm;
	int			dm_nkeys;
	int64_t		dm_base[2];
	int			dm_w[2];
} spec_shape;

/* the keys a session's group table holds (the planner gives no ranges): what a direct-mapped kernel is made for */
typedef struct spec_keystats
{
	uint32_t	ngroups;
	int			nkeys;
	int64_t		kmin[2], kmax[2];
	int			has_value[2];	/* some group has a non-NULL key k */
} spec_keystats;

/*
 * Generate the sources for a program: *defs (the #defines, included before the kernel) and *body
 * (spec_gen.inc). Returns 0, or -1 with a reason in err when the program has to run on the VM
 * (an instruction the generator does not know, an inconsistent program ...). The strings are malloc'ed.
 */
int		spec_generate(const spec_shape *shape, const struct scan_prog *prog,
					  char **defs, char **body, char *err, size_t errlen);

/* the manager: one per service engine */
typedef struct spec_mgr spec_mgr;
spec_mgr *spec_mgr_create(gpu_vk_ctx *vk, int trace);
void	spec_mgr_destroy(spec_mgr *mgr);

/*
 * The pipeline a task of 'prog' should dispatch: the specialized one when it is ready, else vm_pipe.
 * Called once per task; the first calls of a session only count (a one-task query never compiles),
 * a later one queues the compilation, which a worker thread does while the VM serves the tasks.
 *
 * agg_max / ga_c: the VM variant's aggregate capacity and group table slots (the layout of its output).
 * A specialized GROUP BY kernel has the exact aggregate count and a table sized for the groups the session has
 * seen (groups_seen), so its output layout is its own: *out_shape tells the layout of the pipeline returned
 * (the variant's numbers for the VM kernel). out_shape may be NULL for the other kinds.
 */
gpu_vk_pipeline *spec_pipe(struct scan_engine *eng, struct scan_prog *prog, int kind,
						   int agg_max, int ga_c, const spec_keystats *ks, spec_shape *out_shape,
						   gpu_vk_pipeline *vm_pipe);
/* drop the program's reference to its compiled entry (scan_prog_free) */
void	spec_release(struct scan_prog *prog);

/* PGSTROM_GPUSERV_SPECIALIZE=check: every dispatch also runs the VM kernel and compares (spec_verify.c) */
int		spec_check_enabled(struct scan_engine *eng);

/*
 * Dispatch of a task's kernel: the specialized pipeline sp when it is not the VM's (sp != vm), else the VM
 * kernel; in check mode both, on the same input, and the outputs must agree (spec_verify.c), else the
 * dispatch fails. sh: the layout of the specialized kernel's output (spec_pipe's out_shape).
 */
int		spec_dispatch(struct scan_engine *eng, struct scan_prog *prog, int kind, const spec_shape *sh,
					  gpu_vk_pipeline *vm, gpu_vk_pipeline *sp, gpu_vk_buffer *const *bufs, const void *pc,
					  uint32_t groups);

#endif	/* GPUSERV_SPECIALIZE_H */
