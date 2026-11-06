/*
 * spec_verify.c
 *
 * Self-check of the specialized kernels (PGSTROM_GPUSERV_SPECIALIZE=check): scan_exec.c runs every
 * dispatch twice on the same input, once with the VM kernel and once with the specialized one, and this
 * compares the two output buffers.
 *
 * The outputs of the scan, aggregation and join-aggregation kernels are deterministic and are compared
 * word for word. A group table and the match records of a join are filled by atomic appends and slot
 * races, so their order differs from run to run: the slot records of each workgroup's table, and the
 * join's match records, are sorted before they are compared. When a group table overflows (rows that
 * found no slot go to the CPU or to an overflow list) the partition between the table and the
 * overflow depends on the race too, so such a dispatch is reported as "unchecked".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xpu_common.h"
#include "scan_exec.h"
#include "scan_prog.h"
#include "specialize.h"

#define V_GA_NK		2

static size_t	sort_words;

static int
cmp_rec(const void *pa, const void *pb)
{
	const uint32_t *a = pa, *b = pb;

	for (size_t i = 0; i < sort_words; i++)
		if (a[i] != b[i])
			return a[i] < b[i] ? -1 : 1;
	return 0;
}

/* the multiset of n records of rw words at p equals that at q (both are sorted in place on copies) */
static int
multiset_equal(const uint32_t *p, const uint32_t *q, size_t n, size_t rw)
{
	uint32_t   *cp = malloc(sizeof(uint32_t) * (n * rw + 1));
	uint32_t   *cq = malloc(sizeof(uint32_t) * (n * rw + 1));
	int			same;

	if (!cp || !cq)
	{
		free(cp);
		free(cq);
		return 1;				/* cannot check */
	}
	memcpy(cp, p, sizeof(uint32_t) * n * rw);
	memcpy(cq, q, sizeof(uint32_t) * n * rw);
	sort_words = rw;
	qsort(cp, n, sizeof(uint32_t) * rw, cmp_rec);
	qsort(cq, n, sizeof(uint32_t) * rw, cmp_rec);
	same = (memcmp(cp, cq, sizeof(uint32_t) * n * rw) == 0);
	free(cp);
	free(cq);
	return same;
}

/*
 * The match records of a join: { page, lp, item offsets..., (isnull, lo, hi) * ncomp }. The value words of a
 * computed value that is NULL are whatever the register held (a stale value of the lane's previous row, which
 * the two kernels need not agree on and the backend ignores), so they are zeroed before the records are sorted
 * and compared. On a difference msg tells the first record that is not in the other set.
 */
static int
join_records_equal(const uint32_t *p, const uint32_t *q, size_t n, size_t rw, size_t nd, size_t ncomp,
				   char *msg, size_t msglen)
{
	uint32_t   *cp = malloc(sizeof(uint32_t) * (n * rw + 1));
	uint32_t   *cq = malloc(sizeof(uint32_t) * (n * rw + 1));
	int			same = 1;

	if (!cp || !cq)
	{
		free(cp);
		free(cq);
		return 1;
	}
	memcpy(cp, p, sizeof(uint32_t) * n * rw);
	memcpy(cq, q, sizeof(uint32_t) * n * rw);
	for (size_t i = 0; i < n; i++)
		for (size_t k = 0; k < ncomp; k++)
		{
			size_t		o = i * rw + 2 + nd + 3 * k;

			if (cp[o] != 0)
				cp[o + 1] = cp[o + 2] = 0;
			if (cq[o] != 0)
				cq[o + 1] = cq[o + 2] = 0;
		}
	sort_words = rw;
	qsort(cp, n, sizeof(uint32_t) * rw, cmp_rec);
	qsort(cq, n, sizeof(uint32_t) * rw, cmp_rec);
	for (size_t i = 0; i < n && same; i++)
		if (memcmp(cp + i * rw, cq + i * rw, sizeof(uint32_t) * rw) != 0)
		{
			size_t		w = 0, len;

			while (cp[i * rw + w] == cq[i * rw + w])
				w++;
			len = snprintf(msg, msglen, "the %zu match records differ: sorted record %zu word %zu (page, lp, items, values): VM",
						   n, i, w);
			for (size_t j = 0; j < rw && len < msglen; j++)
				len += snprintf(msg + len, msglen - len, " %x", cp[i * rw + j]);
			if (len < msglen)
				len += snprintf(msg + len, msglen - len, "; specialized");
			for (size_t j = 0; j < rw && len < msglen; j++)
				len += snprintf(msg + len, msglen - len, " %x", cq[i * rw + j]);
			same = 0;
		}
	free(cp);
	free(cq);
	return same;
}

/*
 * 0: equal, 1: different (msg says where), 2: not comparable (a group table overflowed)
 * kind may carry SPEC_ARROW: the group kernels' overflow list then follows the tables at pc[3], as for a join
 * pc: the push constants of the dispatch; groups: its workgroup count
 */
static int
spec_verify_compare(int kind, const scan_prog *prog, int agg_max, int ga_c, const uint32_t *pc,
					uint32_t groups, const uint32_t *a, const uint32_t *b, size_t words,
					char *msg, size_t msglen)
{
	int			arrow = ((kind & SPEC_ARROW) != 0);

	kind &= ~SPEC_ARROW;
	if (arrow && kind == SPEC_GAGG)
		kind = SPEC_JGAGG;		/* the same layout: tables, then an overflow list at pc[3] */
	if (a[1] != 0 || b[1] != 0)
	{
		/* a kernel error: which row reports first depends on the race, and the rest of the output is
		 * whatever had been written by then; both kernels must report the same error */
		if (a[1] != b[1])
		{
			snprintf(msg, msglen, "first error %08x (VM) vs %08x (specialized)", a[1], b[1]);
			return 1;
		}
		return 0;
	}
	switch (kind)
	{
		case SPEC_SCAN:
		case SPEC_AGG:
		case SPEC_JAGG:
			for (size_t i = 0; i < words; i++)
				if (a[i] != b[i])
				{
					snprintf(msg, msglen, "word %zu: VM %08x, specialized %08x", i, a[i], b[i]);
					return 1;
				}
			return 0;

		case SPEC_GAGG:
		case SPEC_JGAGG:
		{
			size_t		rec = 2 + 2 * V_GA_NK + 6 * (size_t) agg_max;
			size_t		blk = 1 + (size_t) ga_c * rec;
			size_t		fb = pc[3];			/* GAGG: flags base; JGAGG: overflow base */

			if (kind == SPEC_GAGG)
			{
				for (size_t i = fb; i < words; i++)
					if ((a[i] & 1u) || (b[i] & 1u))
						return 2;
			}
			else if (a[fb] != 0 || b[fb] != 0)
				return 2;
			for (uint32_t g = 0; g < groups; g++)
			{
				size_t		base = 4 + ((size_t) pc[1] + g) * blk;
				const uint32_t *ta = a + base + 1, *tb = b + base + 1;
				uint32_t   *ra = malloc(sizeof(uint32_t) * (size_t) ga_c * rec + 4);
				uint32_t   *rb = malloc(sizeof(uint32_t) * (size_t) ga_c * rec + 4);
				size_t		na = 0, nb = 0;
				int			same;

				if (a[base] != b[base])
				{
					snprintf(msg, msglen, "workgroup %u: row count %u vs %u", pc[1] + g, a[base], b[base]);
					free(ra);
					free(rb);
					return 1;
				}
				if (!ra || !rb)
				{
					free(ra);
					free(rb);
					continue;
				}
				for (int s = 0; s < ga_c; s++)
				{
					if (ta[s * rec] != 0)
					{
						memcpy(ra + na * rec, ta + (size_t) s * rec, sizeof(uint32_t) * rec);
						na++;
					}
					if (tb[s * rec] != 0)
					{
						memcpy(rb + nb * rec, tb + (size_t) s * rec, sizeof(uint32_t) * rec);
						nb++;
					}
				}
				if (na != nb)
				{
					snprintf(msg, msglen, "workgroup %u: %zu groups vs %zu", pc[1] + g, na, nb);
					free(ra);
					free(rb);
					return 1;
				}
				same = multiset_equal(ra, rb, na, rec);
				free(ra);
				free(rb);
				if (!same)
				{
					snprintf(msg, msglen, "workgroup %u: the group tables differ", pc[1] + g);
					return 1;
				}
			}
			if (kind == SPEC_GAGG)
			{
				for (size_t i = fb; i < words; i++)
					if (a[i] != b[i])
					{
						snprintf(msg, msglen, "row flag %zu: %u vs %u", i - fb, a[i], b[i]);
						return 1;
					}
			}
			else if (arrow)
			{
				/* A12: one flag word per row of the pass (2 = the backend evaluates the row) follows the overflow list */
				size_t		flags0 = fb + 1 + (size_t) pc[4] * (1 + 2 * V_GA_NK + 3 * (size_t) agg_max);

				for (size_t i = flags0; i < words; i++)
					if (a[i] != b[i])
					{
						snprintf(msg, msglen, "row flag %zu: %u vs %u", i - flags0, a[i], b[i]);
						return 1;
					}
			}
			return 0;
		}

		case SPEC_JOIN:
		{
			size_t		rw = 2 + (size_t) prog->ndepth + 3 * (size_t) prog->ncomp;
			size_t		ojw = prog->oj_words;
			size_t		cap = pc[3];
			size_t		na = a[0], nb = b[0];

			if (na != nb)
			{
				snprintf(msg, msglen, "match count %zu vs %zu", na, nb);
				return 1;
			}
			for (size_t i = 1; i < 4; i++)
				if (a[i] != b[i])
				{
					snprintf(msg, msglen, "header word %zu: %u vs %u", i, a[i], b[i]);
					return 1;
				}
			for (size_t i = 4; i < 4 + ojw; i++)
				if (a[i] != b[i])
				{
					snprintf(msg, msglen, "outer-join map word %zu: %u vs %u", i - 4, a[i], b[i]);
					return 1;
				}
			if (na > cap)
				return 2;		/* more matches than room: which ones got a slot is a race; the task runs again with room */
			if (!join_records_equal(a + 4 + ojw, b + 4 + ojw, na, rw, (size_t) prog->ndepth, (size_t) prog->ncomp,
									msg, msglen))
				return 1;
			return 0;
		}
	}
	return 0;
}

int
spec_dispatch(scan_engine *eng, scan_prog *prog, int kind, const spec_shape *sh,
			  gpu_vk_pipeline *vm, gpu_vk_pipeline *sp, gpu_vk_buffer *const *bufs, const void *pc,
			  uint32_t groups)
{
	gpu_vk_ctx *vk = eng->vk;

	if (!sp || sp == vm || !spec_check_enabled(eng))
		return gpu_vk_dispatch(vk, sp ? sp : vm, bufs, pc, groups, 1, 1, NULL);
	{
		gpu_vk_buffer *ob = bufs[1];			/* the kernels write only their output buffer */
		size_t		words = ob->size / 4;
		uint32_t   *snap = malloc(ob->size), *res = malloc(ob->size);
		char		msg[500];
		int			rc, cmp;

		if (!snap || !res)
		{
			free(snap);
			free(res);
			return gpu_vk_dispatch(vk, sp, bufs, pc, groups, 1, 1, NULL);
		}
		memcpy(snap, ob->map, ob->size);
		rc = gpu_vk_dispatch(vk, vm, bufs, pc, groups, 1, 1, NULL);
		if (rc != 0)
			goto out;
		memcpy(res, ob->map, ob->size);
		memcpy(ob->map, snap, ob->size);
		rc = gpu_vk_dispatch(vk, sp, bufs, pc, groups, 1, 1, NULL);
		if (rc != 0)
			goto out;
		msg[0] = '\0';
		cmp = spec_verify_compare(kind, prog, sh->agg_max, sh->ga_c, (const uint32_t *) pc, groups,
								  res, (const uint32_t *) ob->map, words, msg, sizeof(msg));
		if (cmp == 1)
		{
			fprintf(stderr, "pgstrom-gpuserv: specialize: CHECK FAILED (kernel kind %d): %s\n", kind, msg);
			snprintf(vk->err, sizeof(vk->err), "the specialized kernel differs from the VM: %s", msg);
			rc = -1;
		}
		else
			fprintf(stderr, "pgstrom-gpuserv: specialize: check %s (kernel kind %d)\n",
					cmp == 0 ? "ok" : "skipped: the output overflowed (group table, match records)", kind);
out:
		free(snap);
		free(res);
		return rc;
	}
}
