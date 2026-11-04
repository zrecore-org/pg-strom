/*
 * specialize.c
 *
 * A6 kernel specialization: generator (program -> Slang source) and manager (background compilation,
 * disk cache, pipelines). See specialize.h and .claude/hw-acceleration.md ("A6 status").
 *
 * The generated code is not a second implementation of the VM: every instruction other than AND / OR
 * (which are control flow) is executed by vm_step() of scan_common.slang, called with constant
 * operands, so the register file becomes constant-indexed and the dispatch disappears while the
 * semantics (and every helper: numeric, strings, overflow checks) stay the VM's. The column loads
 * call deform_col() the same way. What is generated are the pieces that index registers dynamically
 * from program words in the VM kernels (aggregate arguments, keys, projections, join sections).
 * The program words that depend on the query's data (inner buffer layout) are not baked in: the
 * kernels keep reading those from the program buffer.
 *
 * A program the generator does not understand (an unknown opcode, an unexpected layout) is refused
 * and keeps running on the VM.
 */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "xpu_common.h"
#include "scan_exec.h"
#include "scan_prog.h"
#include "scan_caps.h"
#include "specialize.h"

/* ================================================================================================
 * generator
 * ================================================================================================ */
typedef struct
{
	char	   *p;
	size_t		len, cap;
	int			oom;
} sbuf;

static void
sb_printf(sbuf *sb, const char *fmt, ...)
{
	va_list		ap;
	int			n;

	if (sb->oom)
		return;
	for (;;)
	{
		va_start(ap, fmt);
		n = vsnprintf(sb->p ? sb->p + sb->len : NULL, sb->cap - sb->len, fmt, ap);
		va_end(ap);
		if (n < 0)
		{
			sb->oom = 1;
			return;
		}
		if (sb->p && (size_t) n < sb->cap - sb->len)
		{
			sb->len += n;
			return;
		}
		{
			size_t	ncap = (sb->cap ? sb->cap * 2 : 4096);
			char   *np;

			while (ncap < sb->len + n + 1)
				ncap *= 2;
			np = realloc(sb->p, ncap);
			if (!np)
			{
				sb->oom = 1;
				return;
			}
			sb->p = np;
			sb->cap = ncap;
		}
	}
}

/* opcodes the generator knows to be correct as "vm_step with constant operands" (see the top of this file) */
static int
op_known(uint32_t op)
{
	switch (op)
	{
		case SOP_CONST: case SOP_AND: case SOP_OR: case SOP_NOT: case SOP_ISNULL: case SOP_ISNOTNULL:
		case SOP_MOV: case SOP_CAST: case SOP_CMPI: case SOP_CMPF: case SOP_ARITH: case SOP_FARITH:
		case SOP_I2F: case SOP_F2I: case SOP_HASHINIT: case SOP_HASHMERGE: case SOP_HASHFIN:
		case SOP_D2TS: case SOP_TS2D: case SOP_DATEOP:
		case SOP_CONSTSTR: case SOP_STRCMP: case SOP_STRLIKE: case SOP_STRLEN:
		case SOP_NUMCONST: case SOP_NUMCMP: case SOP_NUMARITH: case SOP_I2N: case SOP_N2I:
		case SOP_NUMDIV: case SOP_NUMRND:
		case SOP_CMPD: case SOP_DARITH: case SOP_DUNARY: case SOP_I2D: case SOP_D2I: case SOP_F2D: case SOP_D2F:
		case SOP_GEOMBOX: case SOP_GEOMPIP: case SOP_GEOMDWITHIN:
			return 1;
	}
	return 0;
}

typedef struct
{
	const uint32_t *w;
	size_t		nw;
	uint32_t	nregs;
	char	   *err;
	size_t		errlen;
	int			bad;
	int			kind;
	uint32_t	done[64][2];		/* instruction ranges already generated */
	int			ndone;
} gen_ctx;

static void
gen_fail(gen_ctx *g, const char *fmt, ...)
{
	va_list		ap;

	if (g->bad)
		return;
	g->bad = 1;
	va_start(ap, fmt);
	vsnprintf(g->err, g->errlen, fmt, ap);
	va_end(ap);
}

/*
 * Instructions [a, b) of the program as a function spec_r<a>_<b>(regs) returning the error code.
 * AND / OR jump forward: the instructions they skip are guarded by a flag per jump target (jumps to the same
 * target share it: a chain of n ANDed quals is n guards of one flag, not n guards of up to n flags, which made
 * the generated control flow quadratic and MoltenVK's SPIR-V translation take minutes).
 */
static int
jump_target_seen(const uint32_t *w, uint32_t insnoff, uint32_t a, uint32_t s)
{
	/* an earlier AND/OR of [a, s) with the same target as the one at s */
	for (uint32_t t = a; t < s; t++)
	{
		const uint32_t *tin = w + insnoff + 6 * t;

		if ((tin[0] == SOP_AND || tin[0] == SOP_OR) && tin[3] == w[insnoff + 6 * s + 3])
			return 1;
	}
	return 0;
}

static void
gen_range(gen_ctx *g, sbuf *o, uint32_t a, uint32_t b)
{
	const uint32_t *w = g->w;
	uint32_t	insnoff = w[2];
	uint32_t	nj = 0;

	for (int d = 0; d < g->ndone; d++)
		if (g->done[d][0] == a && g->done[d][1] == b)
			return;
	if (g->ndone >= 64)
	{
		gen_fail(g, "too many instruction sections");
		return;
	}
	g->done[g->ndone][0] = a;
	g->done[g->ndone][1] = b;
	g->ndone++;
	if (b > w[6] || a > b)
	{
		gen_fail(g, "instruction range [%u,%u) is outside the program", a, b);
		return;
	}
	if ((size_t) insnoff + (size_t) 6 * b > g->nw)
	{
		gen_fail(g, "the instruction stream is truncated");
		return;
	}
	sb_printf(o, "uint spec_r%u_%u(inout Reg regs[VM_MAX_REGS])\n{\n", a, b);
	for (uint32_t i = a; i < b; i++)
		if ((w[insnoff + 6 * i] == SOP_AND || w[insnoff + 6 * i] == SOP_OR) &&
			!jump_target_seen(w, insnoff, a, i))
		{
			sb_printf(o, "    bool j%u = false;\n", w[insnoff + 6 * i + 3]);
			nj++;
		}
	for (uint32_t i = a; i < b; i++)
	{
		const uint32_t *in = w + insnoff + 6 * i;
		uint32_t	op = in[0], dst = in[1], ra = in[2], rb = in[3], k0 = in[4], k1 = in[5];
		int			guarded = 0;

		if (!op_known(op))
		{
			gen_fail(g, "instruction %u: opcode %u is not known to the generator", i, op);
			return;
		}
		/* skipped by an earlier jump that targets beyond this instruction? */
		for (uint32_t s = a; s < i; s++)
		{
			const uint32_t *sin = w + insnoff + 6 * s;

			if ((sin[0] == SOP_AND || sin[0] == SOP_OR) && sin[3] > i && !jump_target_seen(w, insnoff, a, s))
			{
				sb_printf(o, guarded ? " && !j%u" : "    if (!j%u", sin[3]);
				guarded = 1;
			}
		}
		if (guarded)
			sb_printf(o, ")\n    {\n");
		else
			sb_printf(o, "    {\n");
		if (op == SOP_AND || op == SOP_OR)
		{
			if (dst >= g->nregs || ra >= g->nregs)
			{
				gen_fail(g, "instruction %u: register out of range", i);
				return;
			}
			if (rb <= i)
			{
				gen_fail(g, "instruction %u: backward jump", i);
				return;
			}
			if (op == SOP_AND)
				sb_printf(o,
						  "        if (regs[%u].isnull != 0)\n            regs[%u].isnull = 1;\n"
						  "        else if (regs[%u].v == 0)\n        {\n"
						  "            regs[%u].v = 0;\n            regs[%u].isnull = 0;\n            j%u = true;\n        }\n",
						  ra, dst, ra, dst, dst, rb);
			else
				sb_printf(o,
						  "        if (regs[%u].isnull != 0)\n            regs[%u].isnull = 1;\n"
						  "        else if (regs[%u].v != 0)\n        {\n"
						  "            regs[%u].v = 1;\n            regs[%u].isnull = 0;\n            j%u = true;\n        }\n",
						  ra, dst, ra, dst, dst, rb);
		}
		else
		{
			sb_printf(o,
					  "        uint ipx = 0;\n"
					  "        uint e = vm_step(%uu, %uu, %uu, %uu, %uu, %uu, ipx, regs);\n"
					  "        if (e != 0u)\n            return e;\n",
					  op, dst, ra, rb, k0, k1);
		}
		sb_printf(o, "    }\n");
	}
	sb_printf(o, "    return 0;\n}\n\n");
	(void) nj;
}

/*
 * The column loads of the table of ncols descriptors at word coloff: spec_<name>(tbase, page, lp, regs)
 * (the tuple at tbase of the current source; false after report_error()).
 */
static void
gen_deform(gen_ctx *g, sbuf *o, const char *name, uint32_t coloff, uint32_t ncols)
{
	const uint32_t *w = g->w;

	if ((size_t) coloff + ncols > g->nw)
	{
		gen_fail(g, "column table outside the program");
		return;
	}
	sb_printf(o,
			  "bool spec_deform_%s(uint tbase, uint page, uint lp, inout Reg regs[VM_MAX_REGS])\n{\n"
			  "    uint infomask2 = ld16(tbase + 18);\n"
			  "    uint infomask  = ld16(tbase + 20);\n"
			  "    uint natts     = infomask2 & 0x7ffu;\n"
			  "    bool hasnull   = (infomask & 1u) != 0;\n"
			  "    uint off       = ld8(tbase + 22);\n",
			  name);
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = w[coloff + c];
		uint32_t	slot = (desc >> 16) & 0xff;

		if (slot != 0xff && slot >= g->nregs)
		{
			gen_fail(g, "column %u loads register %u of %u", c, slot, g->nregs);
			return;
		}
		sb_printf(o, "    if (!deform_col(%uu, 0x%xu, tbase, natts, hasnull, page, lp, off, regs))\n        return false;\n",
				  c, desc);
	}
	sb_printf(o, "    return true;\n}\n\n");
}

/* spec_nulls_<name>(regs): the registers of a column table are NULL (before deforming the next inner tuple) */
static void
gen_nulls(gen_ctx *g, sbuf *o, const char *name, uint32_t coloff, uint32_t ncols)
{
	sb_printf(o, "void spec_nulls_%s(inout Reg regs[VM_MAX_REGS])\n{\n", name);
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	slot = (g->w[coloff + c] >> 16) & 0xff;

		if (slot != 0xff)
			sb_printf(o, "    regs[%u].v = 0;\n    regs[%u].isnull = 1;\n", slot, slot);
	}
	sb_printf(o, "}\n\n");
}

/* the outer row: spec_eval_row(pbase, page, lp, qonly, regs), eval_row() with the program baked in */
static void
gen_eval_row(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	nq = w[5], nins = w[6], qreg = w[7];

	sb_printf(o,
			  "bool spec_eval_row(uint pbase, uint page, uint lp, bool qonly, inout Reg regs[VM_MAX_REGS])\n{\n"
			  "    uint lpd      = ld32(pbase + 24 + lp * 4);\n"
			  "    uint lp_off   = lpd & 0x7fffu;\n"
			  "    uint lp_flags = (lpd >> 15) & 3u;\n"
			  "\n"
			  "    g_recheck = 0;\n"
			  "    if (lp_flags != 1)\n        return false;\n"
			  "\n"
			  "    [unroll]\n    for (uint i = 0; i < VM_MAX_REGS; i++)\n    {\n"
			  "        regs[i].v = 0;\n        regs[i].isnull = 1;\n        regs[i].sc = 0;\n    }\n"
			  "    if (!spec_deform_0(pbase + lp_off, page, lp, regs))\n        return false;\n"
			  "    if (g_recheck != 0)\n        return false;\n"
			  "\n"
			  "    bool hit = true;\n    uint err = 0;\n");
	if (qreg != 0xffffffffu)
	{
		if (qreg >= g->nregs)
		{
			gen_fail(g, "the qual register is out of range");
			return;
		}
		sb_printf(o, "    err = spec_r0_%u(regs);\n    if (err == 0)\n        hit = (regs[%u].isnull == 0 && regs[%u].v != 0);\n",
				  nq, qreg, qreg);
	}
	if (nins > nq && g->kind != SPEC_JOIN && g->kind != SPEC_JAGG && g->kind != SPEC_JGAGG)
		sb_printf(o, "    if (err == 0 && hit && !qonly)\n        err = spec_r%u_%u(regs);\n", nq, nins);
	sb_printf(o,
			  "    if (err == SOP_RECHECK_CODE)\n    {\n        g_recheck = 1;\n        return false;\n    }\n"
			  "    if (err != 0)\n    {\n        report_error(page, lp, err);\n        return false;\n    }\n"
			  "    return hit;\n}\n\n");
}

/* the outer row of an Arrow record batch: arrow_eval_row() of arrow_load.inc with the program baked in */
static void
gen_arrow_eval_row(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	nq = w[5], nins = w[6], qreg = w[7], ncols = w[3], coloff = w[0];

	if ((size_t) coloff + ncols > g->nw)
	{
		gen_fail(g, "column table outside the program");
		return;
	}
	sb_printf(o,
			  "bool spec_arrow_eval_row(uint row, uint epage, uint elp, bool qonly, inout Reg regs[VM_MAX_REGS])\n{\n"
			  /* GPU-Cache: a row whose bit in the row filter is 0 does not exist (free slot, or not visible) */
			  "    if (ptab[8u * %uu + 2u] != 0u && ((pagel[row >> 5] >> (row & 31u)) & 1u) == 0u)\n"
			  "        return false;\n"
			  "    g_recheck = 0;\n    g_src = 0;\n"
			  "    [unroll]\n    for (uint i = 0; i < VM_MAX_REGS; i++)\n    {\n"
			  "        regs[i].v = 0;\n        regs[i].isnull = 1;\n        regs[i].sc = 0;\n    }\n",
			  ncols);
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = w[coloff + c];
		uint32_t	slot = (desc >> 16) & 0xff;

		if (slot != 0xff && slot >= g->nregs)
		{
			gen_fail(g, "column %u loads register %u of %u", c, slot, g->nregs);
			return;
		}
		if (slot != 0xff)
			sb_printf(o, "    arrow_load_col(%uu, 0x%xu, row, regs);\n", c, desc);
	}
	sb_printf(o, "    bool hit = true;\n    uint err = 0;\n");
	if (qreg != 0xffffffffu)
	{
		if (qreg >= g->nregs)
		{
			gen_fail(g, "the qual register is out of range");
			return;
		}
		sb_printf(o, "    err = spec_r0_%u(regs);\n    if (err == 0)\n        hit = (regs[%u].isnull == 0 && regs[%u].v != 0);\n",
				  nq, qreg, qreg);
	}
	if (nins > nq && g->kind != SPEC_JOIN && g->kind != SPEC_JAGG && g->kind != SPEC_JGAGG)
		sb_printf(o, "    if (err == 0 && hit && !qonly)\n        err = spec_r%u_%u(regs);\n", nq, nins);
	sb_printf(o,
			  "    if (err != 0)\n    {\n        report_error(epage, elp, err);\n        return false;\n    }\n"
			  "    return hit;\n}\n\n");
}

static const char *
reg_ok(gen_ctx *g, uint32_t r)
{
	return (r < g->nregs) ? NULL : "register out of range";
}

/* agg_scales_ok() of scan_common.slang for the aggregate and key tables of the program */
static void
gen_scales_ok(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	naggs = w[10], aggoff = w[11], nkeys = w[12], keyoff = w[13];

	sb_printf(o, "bool spec_scales_ok(inout Reg regs[VM_MAX_REGS])\n{\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	sc1 = w[aggoff + 4 * k + 3], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		if (sc1 == 0)
			continue;
		if (reg_ok(g, ra) || (rf != 0xffffffffu && reg_ok(g, rf)))
		{
			gen_fail(g, "aggregate %u: register out of range", k);
			return;
		}
		if (rf != 0xffffffffu)
			sb_printf(o, "    if (!(regs[%u].isnull != 0 || regs[%u].v == 0))\n    ", rf, rf);
		sb_printf(o, "    if (regs[%u].isnull == 0 && regs[%u].sc != %uu)\n        return false;\n", ra, ra, sc1 - 1);
	}
	for (uint32_t k = 0; k < nkeys; k++)
	{
		uint32_t	sc1 = w[keyoff + nkeys + k], r = w[keyoff + k];

		if (sc1 == 0)
			continue;
		if (reg_ok(g, r))
		{
			gen_fail(g, "key %u: register out of range", k);
			return;
		}
		sb_printf(o, "    if (regs[%u].isnull == 0 && regs[%u].sc != %uu)\n        return false;\n", r, r, sc1 - 1);
	}
	sb_printf(o, "    return true;\n}\n\n");
}

/* the accumulation of one row into the thread's aggregates (agg.slang, join_walk.inc mode 1) */
static void
gen_acc(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	naggs = w[10], aggoff = w[11];

	/* the first error (an overflowing float8 sum), 0 if none */
	sb_printf(o, "uint spec_acc(inout Reg regs[VM_MAX_REGS], inout uint an[AGG_MAX], inout int64_t aa[AGG_MAX], inout int64_t ab[AGG_MAX])\n{\n    uint fe = 0u;\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	cls = w[aggoff + 4 * k], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		if (cls < 1 || cls > 8 || (cls != 1 && reg_ok(g, ra)) || (rf != 0xffffffffu && reg_ok(g, rf)))
		{
			gen_fail(g, "aggregate %u: unexpected class %u or register", k, cls);
			return;
		}
		sb_printf(o, "    {\n");
		if (rf != 0xffffffffu)
			sb_printf(o, "        if (!(regs[%u].isnull != 0 || regs[%u].v == 0))\n", rf, rf);
		sb_printf(o, "        {\n");
		if (cls == 1)
			sb_printf(o, "            an[%u] += 1;\n", k);
		else
		{
			sb_printf(o, "            if (regs[%u].isnull == 0)\n            {\n                int64_t v = regs[%u].v;\n", ra, ra);
			if (cls == 2)
				sb_printf(o, "                an[%u] += 1;\n", k);
			else if (cls >= 6)
				sb_printf(o, "                uint e%u = fagg_acc(%uu, uint64_t(v), an[%u], aa[%u]);\n"
						  "                if (fe == 0u)\n                    fe = e%u;\n", k, cls, k, k, k);
			else if (cls == 3)
				sb_printf(o,
						  "                an[%u] += 1;\n"
						  "                aa[%u] += int64_t(uint(v));\n"
						  "                ab[%u] += (v >> 32);\n", k, k, k);
			else
				sb_printf(o,
						  "                if (an[%u] == 0 || v %s aa[%u])\n                    aa[%u] = v;\n"
						  "                an[%u] = 1;\n", k, cls == 4 ? "<" : ">", k, k, k);
			sb_printf(o, "            }\n");
		}
		sb_printf(o, "        }\n    }\n");
	}
	sb_printf(o, "    return fe;\n}\n\n");
}

/* the staging of a row for the group table (gagg.slang A, join_walk.inc mode 2) */
static void
gen_stage(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	naggs = w[10], aggoff = w[11], nkeys = w[12], keyoff = w[13];

	if (nkeys > 2)
	{
		gen_fail(g, "more than two keys");
		return;
	}
	sb_printf(o, "void spec_stage(inout Reg regs[VM_MAX_REGS], uint tid)\n{\n"
				 "    int64_t k0 = 0, k1 = 0;\n    uint kn = 0;\n");
	for (uint32_t k = 0; k < nkeys; k++)
	{
		uint32_t	r = w[keyoff + k];

		if (reg_ok(g, r))
		{
			gen_fail(g, "key %u: register out of range", k);
			return;
		}
		sb_printf(o, "    if (regs[%u].isnull != 0) kn |= %uu; else k%u = regs[%u].v;\n", r, 1u << k, k, r);
	}
	sb_printf(o,
			  "    st_key[tid * GA_NK + 0] = k0;\n    st_key[tid * GA_NK + 1] = k1;\n"
			  "    st_hash[tid] = hash_keys(k0, k1, kn);\n    st_kn[tid] = kn | 0x80000000u;\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	cls = w[aggoff + 4 * k], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		if (cls < 1 || cls > 5 || (cls != 1 && reg_ok(g, ra)) || (rf != 0xffffffffu && reg_ok(g, rf)))
		{
			gen_fail(g, "aggregate %u: unexpected class %u or register", k, cls);
			return;
		}
		sb_printf(o, "    {\n        bool ok = true;\n");
		if (rf != 0xffffffffu)
			sb_printf(o, "        if (regs[%u].isnull != 0 || regs[%u].v == 0)\n            ok = false;\n", rf, rf);
		if (cls != 1)
			sb_printf(o, "        if (ok && regs[%u].isnull != 0)\n            ok = false;\n", ra);
		sb_printf(o, "        st_ok[tid * GA_NA + %u] = ok ? 1u : 0u;\n", k);
		if (cls == 1 || cls == 2)
			sb_printf(o, "        st_val[tid * GA_NA + %u] = 0;\n", k);
		else
			sb_printf(o, "        st_val[tid * GA_NA + %u] = regs[%u].v;\n", k, ra);
		sb_printf(o, "    }\n");
	}
	sb_printf(o, "}\n\n");
}

/*
 * Direct-mapped GROUP BY (gagg_dm.slang, jgagg_dm.slang): the slot of a row is computed from its key values,
 * the window of each key is a constant. spec_dm_row() adds a row to its slot with shared atomics (the
 * accumulation of phase C of gagg.slang) and returns false when the row cannot be aggregated here (a key
 * outside its window, a min/max argument beyond 32 bits): the caller hands it to the service.
 */
static void
gen_dm(gen_ctx *g, sbuf *o, const spec_shape *sh)
{
	const uint32_t *w = g->w;
	uint32_t	naggs = w[10], aggoff = w[11], nkeys = w[12], keyoff = w[13];
	long long	cells = 1;

	if (nkeys < 1 || nkeys > 2 || (int) nkeys != sh->dm_nkeys)
	{
		gen_fail(g, "the direct-mapped table needs the program's one or two keys");
		return;
	}
	for (uint32_t k = 0; k < nkeys; k++)
	{
		if (sh->dm_w[k] < 1 || sh->dm_w[k] > 4096 || sh->dm_base[k] > ((int64_t) 1 << 62) ||
			sh->dm_base[k] < -((int64_t) 1 << 62) || reg_ok(g, w[keyoff + k]))
		{
			gen_fail(g, "the window of key %u is not usable", k);
			return;
		}
		cells *= (long long) sh->dm_w[k] + 1;
	}
	if (cells != sh->ga_c)
	{
		gen_fail(g, "the table size does not match the windows");
		return;
	}
	sb_printf(o, "static const int64_t DM_B0 = int64_t(%lld);\nstatic const uint DM_W0 = %uu;\n",
			  (long long) sh->dm_base[0], (unsigned) sh->dm_w[0]);
	if (nkeys == 2)
		sb_printf(o, "static const int64_t DM_B1 = int64_t(%lld);\nstatic const uint DM_W1 = %uu;\n",
				  (long long) sh->dm_base[1], (unsigned) sh->dm_w[1]);
	sb_printf(o, "static const uint DM_W1P = %uu;\n\n", nkeys == 2 ? (unsigned) sh->dm_w[1] + 1 : 1u);

	/* spec_dm_key(slot, ...): the key of a slot */
	sb_printf(o, "void spec_dm_key(uint s, out uint kn, out int64_t k0, out int64_t k1)\n{\n"
				 "    kn = 0;\n    k0 = 0;\n    k1 = 0;\n"
				 "    uint i0 = s / DM_W1P;\n"
				 "    if (i0 == DM_W0)\n        kn |= 1u;\n    else\n        k0 = DM_B0 + int64_t(i0);\n");
	if (nkeys == 2)
		sb_printf(o, "    uint i1 = s %% DM_W1P;\n"
					 "    if (i1 == DM_W1)\n        kn |= 2u;\n    else\n        k1 = DM_B1 + int64_t(i1);\n");
	sb_printf(o, "}\n\n");

	/* spec_dm_ov(regs, rb): the staged row for the overflow list (gagg.slang's OV_REC) */
	sb_printf(o, "void spec_dm_ov(inout Reg regs[VM_MAX_REGS], uint rb)\n{\n"
				 "    int64_t k0 = 0, k1 = 0;\n    uint kn = 0;\n");
	for (uint32_t k = 0; k < nkeys; k++)
	{
		uint32_t	r = w[keyoff + k];

		sb_printf(o, "    if (regs[%u].isnull != 0) kn |= %uu; else k%u = regs[%u].v;\n", r, 1u << k, k, r);
	}
	sb_printf(o, "    outb[rb + 0] = kn;\n"
				 "    outb[rb + 1] = uint(uint64_t(k0) & 0xffffffffu);\n    outb[rb + 2] = uint(uint64_t(k0) >> 32);\n"
				 "    outb[rb + 3] = uint(uint64_t(k1) & 0xffffffffu);\n    outb[rb + 4] = uint(uint64_t(k1) >> 32);\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	cls = w[aggoff + 4 * k], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		if (cls < 1 || cls > 5 || (cls != 1 && reg_ok(g, ra)) || (rf != 0xffffffffu && reg_ok(g, rf)))
		{
			gen_fail(g, "aggregate %u: unexpected class %u or register", k, cls);
			return;
		}
		sb_printf(o, "    {\n        bool ok = true;\n");
		if (rf != 0xffffffffu)
			sb_printf(o, "        if (regs[%u].isnull != 0 || regs[%u].v == 0)\n            ok = false;\n", rf, rf);
		if (cls != 1)
			sb_printf(o, "        if (ok && regs[%u].isnull != 0)\n            ok = false;\n", ra);
		if (cls == 1 || cls == 2)
			sb_printf(o, "        int64_t v = 0;\n");
		else
			sb_printf(o, "        int64_t v = regs[%u].v;\n", ra);
		sb_printf(o, "        outb[rb + 1 + 2 * GA_NK + %u] = ok ? 1u : 0u;\n"
					 "        outb[rb + 1 + 2 * GA_NK + %u] = uint(uint64_t(v) & 0xffffffffu);\n"
					 "        outb[rb + 1 + 2 * GA_NK + %u] = uint(uint64_t(v) >> 32);\n    }\n",
				  3 * k, 3 * k + 1, 3 * k + 2);
	}
	sb_printf(o, "}\n\n");

	/* spec_dm_row(regs) */
	sb_printf(o, "bool spec_dm_row(inout Reg regs[VM_MAX_REGS])\n{\n    uint i0 = 0;\n    uint i1 = 0;\n");
	for (uint32_t k = 0; k < nkeys; k++)
	{
		uint32_t	r = w[keyoff + k];

		sb_printf(o, "    if (regs[%u].isnull != 0)\n        i%u = DM_W%u;\n    else\n    {\n"
					 "        if (regs[%u].v < DM_B%u || regs[%u].v >= DM_B%u + int64_t(DM_W%u))\n            return false;\n"
					 "        i%u = uint(regs[%u].v - DM_B%u);\n    }\n",
				  r, k, k, r, k, r, k, k, k, r, k);
	}
	sb_printf(o, "    uint slot = i0 * DM_W1P + i1;\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	cls = w[aggoff + 4 * k], ra = w[aggoff + 4 * k + 1], rf = w[aggoff + 4 * k + 2];

		sb_printf(o, "    bool ok%u = true;\n", k);
		if (rf != 0xffffffffu)
			sb_printf(o, "    if (regs[%u].isnull != 0 || regs[%u].v == 0)\n        ok%u = false;\n", rf, rf, k);
		if (cls != 1)
			sb_printf(o, "    if (ok%u && regs[%u].isnull != 0)\n        ok%u = false;\n", k, ra, k);
		if (cls == 3 || cls == 4 || cls == 5)
			sb_printf(o, "    int64_t v%u = regs[%u].v;\n", k, ra);
		if (cls == 4 || cls == 5)
			sb_printf(o, "    if (ok%u && (v%u < -2147483648 || v%u > 2147483647))\n        return false;\n", k, k, k);
	}
	sb_printf(o, "    sh_occ[slot] = 1u;\n    uint prev;\n");
	for (uint32_t k = 0; k < naggs; k++)
	{
		uint32_t	cls = w[aggoff + 4 * k];

		sb_printf(o, "    if (ok%u)\n    {\n        uint a = slot * GA_NA + %u;\n", k, k);
		if (cls == 1 || cls == 2)
			sb_printf(o, "        InterlockedAdd(sh_an[a], 1, prev);\n");
		else if (cls == 3)
			sb_printf(o, "        uint64_t u = uint64_t(v%u) ^ 0x8000000000000000ull;\n"
						 "        InterlockedAdd(sh_an[a], 1, prev);\n"
						 "        InterlockedAdd(sh_l0[a], uint(u & 0xffffu), prev);\n"
						 "        InterlockedAdd(sh_l1[a], uint((u >> 16) & 0xffffu), prev);\n"
						 "        InterlockedAdd(sh_l2[a], uint((u >> 32) & 0xffffu), prev);\n"
						 "        InterlockedAdd(sh_l3[a], uint(u >> 48), prev);\n", k);
		else
			sb_printf(o, "        uint b = uint(int(v%u)) ^ 0x80000000u;\n"
						 "        InterlockedOr(sh_an[a], 1, prev);\n"
						 "        Interlocked%s(sh_l0[a], b, prev);\n", k, cls == 4 ? "Min" : "Max");
		sb_printf(o, "    }\n");
	}
	sb_printf(o, "    return true;\n}\n\n");
}

/* the computed values of a scan row: outb[vb + 3k ...] (scan.slang) or of a join match (cb = first word) */
static void
gen_store_comp(gen_ctx *g, sbuf *o, const char *name, int scan_style)
{
	const uint32_t *w = g->w;
	uint32_t	ncomp = w[8], compoff = w[9];

	sb_printf(o, "void spec_%s(inout Reg regs[VM_MAX_REGS], uint vb)\n{\n", name);
	for (uint32_t k = 0; k < ncomp; k++)
	{
		uint32_t	r = w[compoff + k];

		if (reg_ok(g, r))
		{
			gen_fail(g, "computed value %u: register out of range", k);
			return;
		}
		sb_printf(o, "    {\n        Reg r = regs[%u];\n", r);
		if (scan_style)
			sb_printf(o, "        outb[vb + %u] = r.isnull | (r.sc << 8);\n", 3 * k);
		else
			sb_printf(o, "        outb[vb + %u] = r.isnull;\n", 3 * k);
		sb_printf(o,
				  "        outb[vb + %u] = uint(uint64_t(r.v) & 0xffffffffu);\n"
				  "        outb[vb + %u] = uint(uint64_t(r.v) >> 32);\n    }\n", 3 * k + 1, 3 * k + 2);
	}
	sb_printf(o, "}\n\n");
}

#define SPEC_JDT_WORDS	19

/* the join: per depth the hash section, the join qual section and the inner column loads */
static void
gen_join(gen_ctx *g, sbuf *o)
{
	const uint32_t *w = g->w;
	uint32_t	nd = w[14], dtab = w[15], proj_begin = w[16], nins = w[6];

	if (nd < 1 || nd > SCAN_JOIN_MAXDEPTH || (size_t) dtab + (size_t) SPEC_JDT_WORDS * nd > g->nw)
	{
		gen_fail(g, "unexpected join depth table");
		return;
	}
	for (uint32_t d = 1; d <= nd; d++)
	{
		const uint32_t *dt = w + dtab + SPEC_JDT_WORDS * (d - 1);
		char		nm[16];

		if (dt[14] != 0)
		{
			gen_fail(g, "a GiST-index join runs on the interpreter");
			return;
		}
		gen_range(g, o, dt[0], dt[1]);
		gen_range(g, o, dt[1], dt[2]);
		snprintf(nm, sizeof(nm), "in%u", d);
		gen_deform(g, o, nm, dt[5], dt[6]);
		gen_nulls(g, o, nm, dt[5], dt[6]);
		if (g->bad)
			return;
	}
	if (nins > proj_begin)
		gen_range(g, o, proj_begin, nins);
	/* spec_hash(d, regs, hv): the hash section of depth d and its value */
	sb_printf(o, "uint spec_hash(uint d, inout Reg regs[VM_MAX_REGS], out uint hv)\n{\n    hv = 0;\n    uint e = 0;\n    switch (d)\n    {\n");
	for (uint32_t d = 1; d <= nd; d++)
	{
		const uint32_t *dt = w + dtab + SPEC_JDT_WORDS * (d - 1);

		if (reg_ok(g, dt[3]))
		{
			gen_fail(g, "hash register out of range");
			return;
		}
		sb_printf(o, "    case %u:\n        e = spec_r%u_%u(regs);\n        hv = uint(regs[%u].v);\n        break;\n",
				  d, dt[0], dt[1], dt[3]);
	}
	sb_printf(o, "    default:\n        break;\n    }\n    return e;\n}\n\n");
	/* spec_jqual(d, regs, ok): the join qual section and its result */
	sb_printf(o, "uint spec_jqual(uint d, inout Reg regs[VM_MAX_REGS], out bool ok)\n{\n    ok = false;\n    uint e = 0;\n    switch (d)\n    {\n");
	for (uint32_t d = 1; d <= nd; d++)
	{
		const uint32_t *dt = w + dtab + SPEC_JDT_WORDS * (d - 1);
		uint32_t	jq = dt[4];

		if (jq != 0xffffffffu && reg_ok(g, jq))
		{
			gen_fail(g, "join qual register out of range");
			return;
		}
		sb_printf(o, "    case %u:\n        e = spec_r%u_%u(regs);\n", d, dt[1], dt[2]);
		if (jq == 0xffffffffu)
			sb_printf(o, "        ok = true;\n");
		else
			sb_printf(o, "        ok = (regs[%u].isnull == 0 && regs[%u].v != 0);\n", jq, jq);
		sb_printf(o, "        break;\n");
	}
	sb_printf(o, "    default:\n        break;\n    }\n    return e;\n}\n\n");
	/* spec_inner(d, it, page, lp, regs): null the registers of depth d, then deform the inner tuple at it */
	sb_printf(o, "bool spec_inner(uint d, uint it, uint page, uint lp, inout Reg regs[VM_MAX_REGS])\n{\n    switch (d)\n    {\n");
	for (uint32_t d = 1; d <= nd; d++)
		sb_printf(o, "    case %u:\n        spec_nulls_in%u(regs);\n        return spec_deform_in%u(it, page, lp, regs);\n", d, d, d);
	sb_printf(o, "    default:\n        break;\n    }\n    return false;\n}\n\n");
	sb_printf(o, "void spec_nulls(uint d, inout Reg regs[VM_MAX_REGS])\n{\n    switch (d)\n    {\n");
	for (uint32_t d = 1; d <= nd; d++)
		sb_printf(o, "    case %u:\n        spec_nulls_in%u(regs);\n        break;\n", d, d);
	sb_printf(o, "    default:\n        break;\n    }\n}\n\n");
	sb_printf(o, "uint spec_proj(inout Reg regs[VM_MAX_REGS])\n{\n");
	if (nins > proj_begin)
		sb_printf(o, "    return spec_r%u_%u(regs);\n", proj_begin, nins);
	else
		sb_printf(o, "    return 0;\n");
	sb_printf(o, "}\n\n");
}

/*
 * The register file the generated code needs: the planner's count, or more if an instruction names a higher
 * register (dst always, ra and rb unless the operand is an immediate or a jump target). The vm_step() calls
 * index the array with these constants, so it must be large enough.
 */
static uint32_t
instruction_registers(const uint32_t *w, size_t nw, uint32_t nregs)
{
	uint32_t	insnoff = w[2], nins = w[6], m = nregs;

	if ((size_t) insnoff + (size_t) 6 * nins > nw)
		return nregs;
	for (uint32_t i = 0; i < nins; i++)
	{
		const uint32_t *in = w + insnoff + 6 * i;
		uint32_t	op = in[0];

		if (in[1] + 1 > m)
			m = in[1] + 1;
		if (op != SOP_CONST && op != SOP_NUMCONST && in[2] + 1 > m)
			m = in[2] + 1;
		if (op != SOP_NUMCONST && op != SOP_AND && op != SOP_OR && in[3] + 1 > m)
			m = in[3] + 1;
	}
	return m;
}

int
spec_generate(const spec_shape *shape, const scan_prog *prog, char **p_defs, char **p_body,
			  char *err, size_t errlen)
{
	gen_ctx		g;
	sbuf		defs, body;
	const uint32_t *w = prog->words;
	uint32_t	nq, nins;
	int			kind = shape->kind;

	memset(&defs, 0, sizeof(defs));
	memset(&body, 0, sizeof(body));
	memset(&g, 0, sizeof(g));
	g.w = w;
	g.nw = prog->nwords;
	g.nregs = (prog->nregs > 0 ? prog->nregs : 1);
	g.err = err;
	g.errlen = errlen;
	err[0] = '\0';
	if (!w || prog->nwords < 24 || w[3] > prog->nwords)
	{
		gen_fail(&g, "no program");
		return -1;
	}
	g.nregs = instruction_registers(w, prog->nwords, g.nregs);
	if (g.nregs > SCAN_MAX_REGS)
	{
		gen_fail(&g, "the program names %u registers", g.nregs);
		return -1;
	}
	nq = w[5];
	nins = w[6];
	if (nq > nins || w[2] > prog->nwords)
	{
		gen_fail(&g, "unexpected program header");
		return -1;
	}
	sb_printf(&defs, "// generated by specialize.c\n#define SPEC_PROG 1\n#define VM_MAX_REGS %u\n", g.nregs);
	if (shape->arrow)
		sb_printf(&defs, "#define ARROW_SRC 1\n");
	if (kind == SPEC_AGG || kind == SPEC_JAGG)
		sb_printf(&defs, "#define AGG_MAX %d\n", shape->agg_max);
	if (kind == SPEC_GAGG || kind == SPEC_JGAGG)
		sb_printf(&defs, "#define GA_NA %d\n#define GA_C %d\n", shape->agg_max, shape->ga_c);
	if ((kind == SPEC_GAGG || kind == SPEC_JGAGG) && shape->dm)
		sb_printf(&defs, "#define SPEC_DM 1\n");
	if (kind == SPEC_AGG)
		sb_printf(&defs, "#define SCAN_WG 128\n");

	sb_printf(&body, "// generated by specialize.c: the program baked into the kernel\n\n");
	g.kind = kind;
	gen_range(&g, &body, 0, nq);
	if (nins > nq && kind != SPEC_JOIN && kind != SPEC_JAGG && kind != SPEC_JGAGG)
		gen_range(&g, &body, nq, nins);
	if (shape->arrow)
		gen_arrow_eval_row(&g, &body);
	else
	{
		gen_deform(&g, &body, "0", w[0], w[3]);
		gen_eval_row(&g, &body);
	}
	switch (kind)
	{
		case SPEC_SCAN:
			gen_store_comp(&g, &body, "store_comp", 1);
			break;
		case SPEC_AGG:
			gen_scales_ok(&g, &body);
			gen_acc(&g, &body);
			break;
		case SPEC_GAGG:
			gen_scales_ok(&g, &body);
			if (shape->dm)
				gen_dm(&g, &body, shape);
			else
				gen_stage(&g, &body);
			break;
		case SPEC_JOIN:
			gen_join(&g, &body);
			gen_store_comp(&g, &body, "store_comp_j", 0);
			break;
		case SPEC_JAGG:
			gen_join(&g, &body);
			gen_acc(&g, &body);
			break;
		case SPEC_JGAGG:
			gen_join(&g, &body);
			if (shape->dm)
				gen_dm(&g, &body, shape);
			else
				gen_stage(&g, &body);
			break;
		default:
			gen_fail(&g, "unknown kernel kind %d", kind);
	}
	if (g.bad || defs.oom || body.oom)
	{
		if (!g.bad)
			snprintf(err, errlen, "out of memory");
		free(defs.p);
		free(body.p);
		return -1;
	}
	*p_defs = defs.p;
	*p_body = body.p;
	return 0;
}
