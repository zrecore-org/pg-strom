/*
 * spec_check.c
 *
 * Offline check of the kernel generator (specialize.c): builds programs of the shapes the planner makes (by
 * hand, as scan_prog.c lays them out) for every kernel kind, generates the sources, and writes them as
 *   <outdir>/<name>/spec_defs.inc, spec_gen.inc, main.slang
 * so that `make spec_check` can compile each with slangc. No Vulkan device is needed.
 *
 *   spec_check <outdir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "xpu_common.h"
#include "scan_prog.h"
#include "scan_caps.h"
#include "specialize.h"

#define HDR 24
typedef struct
{
	uint32_t	w[512];
	uint32_t	n;
} wbuf;

static const char *kfile[] = {NULL, "scan.slang", "agg.slang", "gagg.slang", "join.slang", "jagg.slang", "jgagg.slang", "gtab.slang"};

static void
insn(wbuf *b, uint32_t op, uint32_t dst, uint32_t a, uint32_t rb, uint32_t k0, uint32_t k1)
{
	uint32_t   *p = b->w + b->n;

	p[0] = op; p[1] = dst; p[2] = a; p[3] = rb; p[4] = k0; p[5] = k1;
	b->n += 6;
}

/*
 * table: id int8, v int4, w int4 (v in register 0, w in register 1); quals: v + 1 > 3 AND w < 100.
 * kinds: 0 scan, 1 agg, 2 group, 3 join (the join kinds add a depth table)
 */
static void
build(int kind, scan_prog *prog)
{
	static wbuf	ins;
	uint32_t   *w;
	uint32_t	nq, nins, ncols = 3, nagg = 0, nkeys = 0, ncomp = 0;
	uint32_t	insnoff = HDR + ncols, compoff, aggoff, keyoff, nwords;
	uint32_t	inner_col = 0;

	memset(&ins, 0, sizeof(ins));
	insn(&ins, SOP_CONST, 2, 0, 0, 1, 0);									/* r2 = 1 */
	insn(&ins, SOP_ARITH, 3, 0, 2, SOP_ARITH_PL | (32 << 8), 7);			/* r3 = v + 1 */
	insn(&ins, SOP_CONST, 4, 0, 0, 3, 0);									/* r4 = 3 */
	insn(&ins, SOP_CMPI, 5, 3, 4, SOP_CMP_GT, 0);							/* r5 = r3 > r4 */
	insn(&ins, SOP_AND, 6, 5, 11, 0, 0);									/* r6 = r5 AND ... (jump to 11) */
	insn(&ins, SOP_CONST, 7, 0, 0, 100, 0);									/* r7 = 100 */
	insn(&ins, SOP_CMPI, 8, 1, 7, SOP_CMP_LT, 0);							/* r8 = w < 100 */
	insn(&ins, SOP_AND, 6, 8, 11, 0, 0);
	insn(&ins, SOP_MOV, 9, 6, 0, 0, 0);
	insn(&ins, SOP_MOV, 10, 9, 0, 0, 0);
	insn(&ins, SOP_MOV, 11, 10, 0, 0, 0);
	nq = ins.n / 6;
	/* the part after the quals: the projection / aggregate arguments */
	insn(&ins, SOP_ARITH, 12, 0, 1, SOP_ARITH_MUL | (32 << 8), 9);			/* r12 = v * w */
	nins = ins.n / 6;
	if (kind == 0)
		ncomp = 1;
	if (kind == 1)
		nagg = 3;
	if (kind == 2)
	{
		nagg = 3;
		nkeys = 1;
	}
	compoff = insnoff + ins.n;
	aggoff = compoff + ncomp;
	keyoff = aggoff + 4 * nagg;
	nwords = keyoff + 2 * nkeys + 4;
	if (kind == 3)
		inner_col = nwords + 14 * 1;		/* depth table at nwords: see below */
	w = calloc(nwords + 64, sizeof(uint32_t));
	w[0] = HDR; w[2] = insnoff; w[3] = ncols; w[5] = nq; w[6] = nins; w[7] = 11;
	w[8] = ncomp; w[9] = compoff; w[10] = nagg; w[11] = aggoff; w[12] = nkeys; w[13] = keyoff;
	w[20] = nwords;
	w[HDR + 0] = 8 | (8 << 8) | (SOP_VT_INT << 12) | (0xff << 16);
	w[HDR + 1] = 4 | (4 << 8) | (SOP_VT_INT << 12) | (0 << 16);
	w[HDR + 2] = 4 | (4 << 8) | (SOP_VT_INT << 12) | (1 << 16);
	memcpy(w + insnoff, ins.w, 4 * ins.n);
	if (kind == 0)
		w[compoff] = 12;
	if (nagg)
	{
		w[aggoff + 0] = 1; w[aggoff + 1] = 0xffffffffu; w[aggoff + 2] = 0xffffffffu;
		w[aggoff + 4] = 3; w[aggoff + 5] = 12; w[aggoff + 6] = 5;
		w[aggoff + 8] = 5; w[aggoff + 9] = 0; w[aggoff + 10] = 0xffffffffu;
	}
	if (nkeys)
		w[keyoff] = 1;
	(void) inner_col;
	prog->words = w;
	prog->nwords = nwords + 2;
	prog->nregs = 16;
}

/* a join: one inner relation, hash key v, join qual r13 = (inner column 0 reg 13) */
static void
build_join(int mode, scan_prog *prog)
{
	static wbuf	ins;
	uint32_t   *w;
	uint32_t	insnoff = HDR + 2, dtab, nwords, nq, h0, h1, h2, nins, pb;
	uint32_t	nagg = 0, nkeys = 0, ncomp = 0;

	memset(&ins, 0, sizeof(ins));
	insn(&ins, SOP_CONST, 2, 0, 0, 3, 0);									/* r2 = 3 */
	insn(&ins, SOP_CMPI, 3, 0, 2, SOP_CMP_GT, 0);							/* r3 = v > 3 */
	nq = ins.n / 6;
	h0 = nq;
	insn(&ins, SOP_HASHINIT, 4, 0, 0, 0, 0);
	insn(&ins, SOP_HASHMERGE, 4, 0, 0, 4, 0);
	insn(&ins, SOP_HASHFIN, 4, 0, 0, 0, 0);
	h1 = ins.n / 6;
	insn(&ins, SOP_CMPI, 5, 0, 6, SOP_CMP_EQ, 0);							/* join qual: v = inner col (r6) */
	h2 = ins.n / 6;
	pb = h2;
	insn(&ins, SOP_ARITH, 7, 0, 6, SOP_ARITH_PL | (64 << 8), 9);
	nins = ins.n / 6;
	if (mode == 0)
		ncomp = 1;
	if (mode == 1)
		nagg = 2;
	if (mode == 2)
	{
		nagg = 2;
		nkeys = 1;
	}
	dtab = insnoff + ins.n + ncomp + 4 * nagg + 2 * nkeys;
	nwords = dtab + 14 + 1 + 3;
	w = calloc(nwords + 8, sizeof(uint32_t));
	w[0] = HDR; w[2] = insnoff; w[3] = 2; w[5] = nq; w[6] = nins; w[7] = 3;
	w[8] = ncomp; w[9] = insnoff + ins.n;
	w[10] = nagg; w[11] = insnoff + ins.n + ncomp;
	w[12] = nkeys; w[13] = insnoff + ins.n + ncomp + 4 * nagg;
	w[14] = 1; w[15] = dtab; w[16] = pb;
	w[HDR + 0] = 8 | (8 << 8) | (SOP_VT_INT << 12) | (0xff << 16);
	w[HDR + 1] = 4 | (4 << 8) | (SOP_VT_INT << 12) | (0 << 16);
	memcpy(w + insnoff, ins.w, 4 * ins.n);
	if (ncomp)
		w[w[9]] = 7;
	if (nagg)
	{
		w[w[11] + 0] = 1; w[w[11] + 1] = 0xffffffffu; w[w[11] + 2] = 0xffffffffu;
		w[w[11] + 4] = 3; w[w[11] + 5] = 7; w[w[11] + 6] = 0xffffffffu;
	}
	if (nkeys)
		w[w[13]] = 6;
	/* depth table */
	w[dtab + 0] = h0; w[dtab + 1] = h1; w[dtab + 2] = h2; w[dtab + 3] = 4; w[dtab + 4] = 5;
	w[dtab + 5] = nwords; w[dtab + 6] = 1;				/* the inner column table: one column to register 6 */
	w[dtab + 10] = 0; w[dtab + 11] = 0; w[dtab + 12] = 0xffffffffu;
	w[nwords] = 4 | (4 << 8) | (SOP_VT_INT << 12) | (6 << 16);
	prog->words = w;
	prog->nwords = nwords + 1;
	prog->nregs = 16;
}

static int g_dm;			/* the next emit() makes a direct-mapped GROUP BY kernel */

static int
emit(const char *outdir, const char *name, int kind, int mode, int agg_max, int ga_c, int arrow)
{
	scan_prog	prog;
	spec_shape	sh;
	char	   *defs = NULL, *body = NULL, err[256];
	char		dir[512], path[600];
	FILE	   *f;

	memset(&prog, 0, sizeof(prog));
	if (kind >= SPEC_JOIN && kind != SPEC_GTAB)
		build_join(mode, &prog);
	else
		build(kind == SPEC_GTAB ? 2 : kind - 1, &prog);
	memset(&sh, 0, sizeof(sh));
	if (g_dm)
	{
		sh.dm = 1;
		sh.dm_nkeys = 1;
		sh.dm_base[0] = -5;
		sh.dm_w[0] = 20;
		ga_c = 21;
	}
	sh.kind = kind;
	sh.arrow = arrow;
	sh.agg_max = agg_max;
	sh.ga_c = ga_c;
	if (spec_generate(&sh, &prog, &defs, &body, err, sizeof(err)) != 0)
	{
		fprintf(stderr, "%s: generator refused: %s\n", name, err);
		return -1;
	}
	snprintf(dir, sizeof(dir), "%s/%s", outdir, name);
	mkdir(outdir, 0755);
	mkdir(dir, 0755);
	snprintf(path, sizeof(path), "%s/spec_defs.inc", dir);
	f = fopen(path, "w"); fputs(defs, f); fclose(f);
	snprintf(path, sizeof(path), "%s/spec_gen.inc", dir);
	f = fopen(path, "w"); fputs(body, f); fclose(f);
	snprintf(path, sizeof(path), "%s/main.slang", dir);
	f = fopen(path, "w");
	fprintf(f, "#include \"spec_defs.inc\"\n#include \"%s\"\n",
			(arrow && kind == SPEC_SCAN) ? "arrow.slang" :
			g_dm ? (kind == SPEC_GAGG ? "gagg_dm.slang" : "jgagg_dm.slang") : kfile[kind]);
	fclose(f);
	printf("%s\n", name);
	free(defs);
	free(body);
	free(prog.words);
	return 0;
}

int
main(int argc, char **argv)
{
	const char *out = argc > 1 ? argv[1] : "spec_out";
	int			rc = 0;

	rc |= emit(out, "scan", SPEC_SCAN, 0, 0, 0, 0);
	rc |= emit(out, "agg", SPEC_AGG, 0, 4, 0, 0);
	rc |= emit(out, "gagg", SPEC_GAGG, 0, 4, 128, 0);
	rc |= emit(out, "join", SPEC_JOIN, 0, 0, 0, 0);
	rc |= emit(out, "jagg", SPEC_JAGG, 1, 4, 0, 0);
	rc |= emit(out, "jgagg", SPEC_JGAGG, 2, 4, 128, 0);
	/* a different shape of the GROUP BY kernels: exact aggregates, a small table */
	rc |= emit(out, "gagg3", SPEC_GAGG, 0, 3, 16, 0);
	rc |= emit(out, "jgagg3", SPEC_JGAGG, 2, 2, 32, 0);
	/* direct-mapped GROUP BY */
	g_dm = 1;
	rc |= emit(out, "gaggdm", SPEC_GAGG, 0, 3, 0, 0);
	rc |= emit(out, "jgaggdm", SPEC_JGAGG, 2, 2, 0, 0);
	rc |= emit(out, "agaggdm", SPEC_GAGG, 0, 3, 0, 1);
	rc |= emit(out, "ajgaggdm", SPEC_JGAGG, 2, 2, 0, 1);
	g_dm = 0;
	/* an Arrow record batch as the outer relation */
	rc |= emit(out, "ascan", SPEC_SCAN, 0, 0, 0, 1);
	rc |= emit(out, "aagg", SPEC_AGG, 0, 4, 0, 1);
	rc |= emit(out, "agagg", SPEC_GAGG, 0, 4, 128, 1);
	rc |= emit(out, "ajoin", SPEC_JOIN, 0, 0, 0, 1);
	rc |= emit(out, "ajagg", SPEC_JAGG, 1, 4, 0, 1);
	rc |= emit(out, "ajgagg", SPEC_JGAGG, 2, 4, 128, 1);
	/* GROUP BY into the session-wide table */
	rc |= emit(out, "gtab", SPEC_GTAB, 0, 12, 0, 0);
	rc |= emit(out, "agtab", SPEC_GTAB, 0, 12, 0, 1);
	return rc ? 1 : 0;
}
