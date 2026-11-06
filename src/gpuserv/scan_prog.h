/*
 * scan_prog.h
 *
 * Translates the planner's scan xpucode (kern_expression trees in a kern_session_info) into
 * the program of the scan VM (kernels_vm/scan.slang).
 */
#ifndef GPUSERV_SCAN_PROG_H
#define GPUSERV_SCAN_PROG_H
#include <stddef.h>
#include <stdint.h>

#include "scan_caps.h"
/* SCAN_MAX_REGS and SCAN_AGG_MAX are in scan_caps.h (the planner checks them too) */

/* where an output column of the scan comes from */
#define SCAN_PROJ_SRC		1		/* copy of the source column 'resno' (any type) */
#define SCAN_PROJ_COMP		2		/* value computed on the GPU, 'comp'-th result */
#define SCAN_PROJ_SYS		3		/* system column: ctid (-1) or tableoid (-6) */
#define SCAN_PROJ_INNER		4		/* a join: copy of the inner relation's column 'resno' */

typedef struct scan_proj_attr
{
	int			kind;
	int			resno;				/* SCAN_PROJ_SRC: attribute number; SCAN_PROJ_SYS: the system attribute */
	int			depth;				/* SCAN_PROJ_INNER: which inner relation (1 ...) */
	int			comp;				/* SCAN_PROJ_COMP: index in the per-row results */
	int			comp_bytes;			/* SCAN_PROJ_COMP: size of the datum (1, 2, 4 or 8) */
	int			comp_numeric;		/* SCAN_PROJ_COMP: a numeric { unscaled, scale }; the NULL word carries the scale */
} scan_proj_attr;

/*
 * The optional instruction groups of the interpreter (kernels_vm/scan_common.slang compiles each out with -DVM_NO_<group>).
 * A program that uses none of them runs on the "lite" join kernels, whose pipelines compile in a fraction of the time
 * (Metal needs minutes for the full join kernel; see the Makefile).
 */
#define SCAN_FEAT_NUM	0x1			/* numeric */
#define SCAN_FEAT_JSON	0x2			/* jsonb */
#define SCAN_FEAT_F8	0x4			/* float8 (software binary64) */
#define SCAN_FEAT_GEOM	0x8			/* geometry */
#define SCAN_FEAT_ALL	0xf

typedef struct scan_prog
{
	uint32_t	   *words;			/* program for the kernel */
	size_t			nwords;
	int				nattrs;			/* number of output columns */
	scan_proj_attr *attrs;
	uint32_t		ncomp;			/* number of computed values per matching row */
	int				nregs;			/* registers the program needs (selects the kernel variant) */
	uint32_t		feat;			/* SCAN_FEAT_* groups the instructions use (0: the lite kernels will do) */
	/*
	 * Join programs (A14): which parts of the program can answer "the backend must decide" (SOP_RECHECK_CODE,
	 * or an unreadable inner value). rck_outer: the scan quals of the outer row (the row has contributed
	 * nothing yet, so the aggregating kernel only flags it); rck_walk: the join walk or the projection (an
	 * accumulator cannot take back what the row's earlier matches added, so the service first runs a scout
	 * pass that names those rows and the aggregating pass skips them).
	 */
	int				rck_outer;
	int				rck_walk;
	int				max_src_resno;	/* highest source column copied as is */
	/* join (GpuJoin) */
	int				join_mode;
	int				ndepth;				/* inner relations */
	int				max_inner_resno[SCAN_JOIN_MAXDEPTH + 1];	/* highest column of inner relation d copied as is */
	/* the outer-join map of RIGHT/FULL joins: one word per inner row at the head of the output buffer */
	uint32_t		oj_words;			/* total words (0: no right/full join) */
	uint32_t		oj_off[SCAN_JOIN_MAXDEPTH + 1];		/* first word of relation d, or 0xffffffff */
	uint32_t		oj_nitems[SCAN_JOIN_MAXDEPTH + 1];	/* its rows */
	/* aggregation (GpuPreAgg without GROUP BY) */
	int				agg_mode;
	int				nagg;			/* number of aggregates (not counting key columns) */
	int				nkeys;			/* GROUP BY keys (0: aggregation without GROUP BY) */
	uint16_t		agg_action[SCAN_AGG_MAX];	/* KAGG_ACTION__* of each output column */
	int8_t			agg_of_col[SCAN_AGG_MAX];	/* output column -> aggregate index, or -1 */
	int8_t			key_of_col[SCAN_AGG_MAX];	/* output column -> key index, or -1 */
	/* numeric aggregates / keys: the scale the plan derived (every row must have it), or -1 */
	int16_t			agg_scale[SCAN_AGG_MAX];
	int16_t			key_scale[SCAN_GROUP_KEYS];
	/* A6 specialization (specialize.h): the compiled kernel of this program, if any */
	void		   *spec;				/* a spec_entry owned by the manager (we hold a reference) */
	void		   *spec_next;			/* GROUP BY: a bigger kernel being compiled (spec_pipe) */
	int				spec_grow_failed;
	int				spec_regens;
	void		   *spec_owner;
	int				spec_state;			/* 2: the generator refused the program or the compiler failed */
	int				spec_tasks;			/* tasks seen (the compilation starts at the N-th) */
} scan_prog;

struct kern_session_info;
struct kern_data_store;

/* 0 on success, -1 on failure with a message in err (the query cannot run on this service) */
int		scan_prog_build(const struct kern_session_info *session,
						const struct kern_data_store *kds_src,
						scan_prog *prog, char *err, size_t errlen);
/*
 * The same for a GpuJoin session (hash joins of up to SCAN_JOIN_MAXDEPTH inner relations). Each
 * inner relation is a hash-format store; the numbers describe where its hash slots and its end are
 * in the buffer the kernel sees.
 */
typedef struct scan_join_rel
{
	const struct kern_data_store *kds;
	uint32_t	kds_end;			/* byte offset of the end of the store in the inner buffer */
	uint32_t	hslot_word;			/* word offset of its hash slots */
	uint32_t	nslots;
	int			left_outer;			/* unmatched outer rows are kept (NULL inner columns) */
	int			semi;				/* semi join: the first match keeps the outer row, no inner columns */
	int			anti;				/* anti join: only an outer row without a match is kept */
	int			right_outer;		/* RIGHT or FULL: the kernel flags the inner rows that matched */
	/* A13: the rows are found through a GiST index (kds is the hash store of the rows, indexed by row id) */
	uint32_t	gist_page0;			/* byte offset of block 0 of the index pages in the inner buffer; 0: no index */
	uint32_t	gist_nblocks;
} scan_join_rel;

int		scan_prog_build_join(const struct kern_session_info *session,
							 const struct kern_data_store *kds_src,
							 const scan_join_rel *rels, int nrels,
							 scan_prog *prog, char *err, size_t errlen);
void	scan_prog_free(scan_prog *prog);
/* aggregation over a RIGHT/FULL join: point the kernels at the outer-join map behind their results (word `base` of the output) */
void	scan_prog_ojmap_place(const scan_prog *prog, uint32_t *words, size_t base);

#endif	/* GPUSERV_SCAN_PROG_H */
