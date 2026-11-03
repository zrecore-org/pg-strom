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

typedef struct spec_shape
{
	int			kind;
	int			agg_max;	/* AGG_MAX / GA_NA of the kernel variant (the layout of the output) */
	int			ga_c;		/* GA_C: slots of a workgroup's group table */
} spec_shape;

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
						   int agg_max, int ga_c, uint32_t groups_seen, spec_shape *out_shape,
						   gpu_vk_pipeline *vm_pipe);
/* drop the program's reference to its compiled entry (scan_prog_free) */
void	spec_release(struct scan_prog *prog);

/* PGSTROM_GPUSERV_SPECIALIZE=check: every dispatch also runs the VM kernel and compares (scan_exec.c) */
int		spec_check_enabled(struct scan_engine *eng);

#endif	/* GPUSERV_SPECIALIZE_H */
