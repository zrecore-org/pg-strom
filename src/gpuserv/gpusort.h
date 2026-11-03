/*
 * gpusort.h
 *
 * GPU-Sort of the rows of a GpuPreAgg result (the groups of a session), including the top-N cut
 * (LIMIT pushed down) and the window-rank filter (rank() / dense_rank() / row_number() < N per
 * PARTITION BY). The caller turns each row's sort keys into key words (gpusort_encode); the
 * order of the unsigned words is the sort order, so the same encoded keys serve the bitonic
 * network on the device (kernels_vm/gsort.slang) and the CPU merge sort used for small inputs,
 * for inputs that do not fit the device buffers and when the device fails.
 */
#ifndef GPUSERV_GPUSORT_H
#define GPUSERV_GPUSORT_H
#include <stddef.h>
#include <stdint.h>
#include "scan_exec.h"

#define GPUSORT_MAX_KEYS		8

typedef struct gpusort_spec
{
	int			nkeys;
	struct
	{
		int			asc;			/* ascending order */
		int			nulls_first;
		int			wide;			/* the value is a signed 128-bit integer (else 64-bit) */
		uint32_t	woff;			/* first key word of this key in a row */
	} key[GPUSORT_MAX_KEYS];
	uint32_t	nwords;				/* key words per row */
	/* window-rank filter: keep the rows whose rank (per partition) is below window_limit */
	char		window_func;		/* KSORT_WINDOW_FUNC__* or 0 */
	uint32_t	window_limit;
	int			partby_nkeys;		/* the first keys partition the rows, the next ones order them */
	int			orderby_nkeys;
	/* top-N: keep the first 'limit' rows when positive (not together with the window filter) */
	int64_t		limit;
} gpusort_spec;

/* lay out the key words after nkeys, asc, nulls_first and wide were set */
void	gpusort_spec_layout(gpusort_spec *spec);
/* write the words of key 'k' of a row (isnull: NULL; v: the signed value) */
void	gpusort_encode(const gpusort_spec *spec, int k, uint32_t *row, int isnull, __int128 v);

/*
 * Sort n rows of spec->nwords key words each. On success *p_order is a malloc'ed array of the row
 * numbers in sorted order, after the top-N cut / window filter, *p_nout of them. 0 on success, -1
 * with a message in err. *p_used_gpu tells which engine sorted.
 */
int		gpusort_run(scan_engine *eng, const gpusort_spec *spec, const uint32_t *keys, uint32_t n,
					uint32_t **p_order, uint32_t *p_nout, int *p_used_gpu, char *err, size_t errlen);
void	gpusort_engine_fini(scan_engine *eng);

#endif	/* GPUSERV_GPUSORT_H */
