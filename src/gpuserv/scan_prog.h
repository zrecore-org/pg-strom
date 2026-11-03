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

#define SCAN_MAX_REGS		32		/* must equal VM_MAX_REGS of scan_common.slang */
#define SCAN_AGG_MAX		12		/* must equal AGG_MAX of agg.slang */

/* where an output column of the scan comes from */
#define SCAN_PROJ_SRC		1		/* copy of the source column 'resno' (any type) */
#define SCAN_PROJ_COMP		2		/* value computed on the GPU, 'comp'-th result */
#define SCAN_PROJ_SYS		3		/* system column: ctid (-1) or tableoid (-6) */

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
	/* aggregation (GpuPreAgg without GROUP BY) */
	int				agg_mode;
	int				nagg;
	uint16_t		agg_action[SCAN_AGG_MAX];	/* KAGG_ACTION__* of each output column */
} scan_prog;

struct kern_session_info;
struct kern_data_store;

/* 0 on success, -1 on failure with a message in err (the query cannot run on this service) */
int		scan_prog_build(const struct kern_session_info *session,
						const struct kern_data_store *kds_src,
						scan_prog *prog, char *err, size_t errlen);
void	scan_prog_free(scan_prog *prog);

#endif	/* GPUSERV_SCAN_PROG_H */
