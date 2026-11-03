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
#define SCAN_MAX_REGS		32		/* must equal VM_MAX_REGS of scan_common.slang */
#define SCAN_AGG_MAX		12		/* must equal AGG_MAX of agg.slang */

/* where an output column of the scan comes from */
#define SCAN_PROJ_SRC		1		/* copy of the source column 'resno' (any type) */
#define SCAN_PROJ_COMP		2		/* value computed on the GPU, 'comp'-th result */
#define SCAN_PROJ_SYS		3		/* system column: ctid (-1) or tableoid (-6) */
#define SCAN_PROJ_INNER		4		/* a join: copy of the inner relation's column 'resno' */

typedef struct scan_proj_attr
{
	int			kind;
	int			resno;				/* SCAN_PROJ_SRC: attribute number; SCAN_PROJ_SYS: the system attribute */
	int			comp;				/* SCAN_PROJ_COMP: index in the per-row results */
	int			comp_bytes;			/* SCAN_PROJ_COMP: size of the datum (1, 2, 4 or 8) */
} scan_proj_attr;

typedef struct scan_prog
{
	uint32_t	   *words;			/* program for the kernel */
	size_t			nwords;
	int				nattrs;			/* number of output columns */
	scan_proj_attr *attrs;
	uint32_t		ncomp;			/* number of computed values per matching row */
	int				nregs;			/* registers the program needs (selects the kernel variant) */
	int				max_src_resno;	/* highest source column copied as is */
	/* join (GpuJoin, one inner relation) */
	int				join_mode;
	int				max_inner_resno;	/* highest inner column copied as is */
	/* aggregation (GpuPreAgg without GROUP BY) */
	int				agg_mode;
	int				nagg;			/* number of aggregates (not counting key columns) */
	int				nkeys;			/* GROUP BY keys (0: aggregation without GROUP BY) */
	uint16_t		agg_action[SCAN_AGG_MAX];	/* KAGG_ACTION__* of each output column */
	int8_t			agg_of_col[SCAN_AGG_MAX];	/* output column -> aggregate index, or -1 */
	int8_t			key_of_col[SCAN_AGG_MAX];	/* output column -> key index, or -1 */
} scan_prog;

struct kern_session_info;
struct kern_data_store;

/* 0 on success, -1 on failure with a message in err (the query cannot run on this service) */
int		scan_prog_build(const struct kern_session_info *session,
						const struct kern_data_store *kds_src,
						scan_prog *prog, char *err, size_t errlen);
/*
 * The same for a GpuJoin session (one inner relation, inner hash join). kds_in is the inner
 * relation's hash-format store; the three numbers describe where its hash slots and its end are
 * in the buffer the kernel sees.
 */
int		scan_prog_build_join(const struct kern_session_info *session,
							 const struct kern_data_store *kds_src,
							 const struct kern_data_store *kds_in,
							 uint32_t kds_end, uint32_t hslot_word, uint32_t nslots,
							 scan_prog *prog, char *err, size_t errlen);
void	scan_prog_free(scan_prog *prog);

#endif	/* GPUSERV_SCAN_PROG_H */
