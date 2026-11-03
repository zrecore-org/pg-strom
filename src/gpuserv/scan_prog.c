/*
 * scan_prog.c
 *
 * kern_expression trees (xpu_common.h) -> scan VM program. See scan_vm_ops.inc for the
 * instruction set and scan.slang for the program layout.
 *
 * Supported: bool, int1/2/4/8 and float4 values; comparison, +, -, *, /, % on them (with
 * PostgreSQL's overflow and zero-divide errors), casts between the integer types, unary
 * plus/minus, AND/OR/NOT with three-valued logic and IS [NOT] NULL. Anything else makes
 * scan_prog_build() fail, and the query then reports that the service cannot run it.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xpu_common.h"
#include "scan_prog.h"
#include "scan_caps.h"


#define PROG_HDR_WORDS		10
#define NO_REG				(-1)

/* ----------------------------------------------------------------
 * builder
 * ----------------------------------------------------------------
 */
typedef struct
{
	const kern_session_info *session;
	const kern_data_store *kds_src;
	uint32_t   *insn;				/* 6 words per instruction */
	uint32_t	ninsn;
	uint32_t	insn_cap;
	uint32_t	nslots;
	int			maxreg;						/* highest register in use + 1 */
	int			slot_resno[SCAN_MAX_REGS];	/* source attribute of a slot, 0 if none */
	int			slot_sysattr[SCAN_MAX_REGS];	/* system attribute (ctid, ...) of a slot, 0 if none */
	int			slot_nullonly[SCAN_MAX_REGS];/* only IS [NOT] NULL reads the slot */
	int			slot_used[SCAN_MAX_REGS];
	int			slot_vtype[SCAN_MAX_REGS];
	int			slot_bytes[SCAN_MAX_REGS];
	char	   *err;
	size_t		errlen;
} builder;

static int
fail(builder *b, const char *fmt, ...)
{
	va_list		ap;

	if (b->err && b->err[0] == '\0')
	{
		va_start(ap, fmt);
		vsnprintf(b->err, b->errlen, fmt, ap);
		va_end(ap);
	}
	return -1;
}

static int
emit(builder *b, uint32_t op, uint32_t dst, uint32_t a, uint32_t rb, uint32_t k0, uint32_t k1)
{
	uint32_t   *p;

	if (b->ninsn == b->insn_cap)
	{
		uint32_t	ncap = (b->insn_cap ? b->insn_cap * 2 : 64);
		uint32_t   *ni = realloc(b->insn, sizeof(uint32_t) * 6 * ncap);

		if (!ni)
			return fail(b, "out of memory");
		b->insn = ni;
		b->insn_cap = ncap;
	}
	p = b->insn + 6 * b->ninsn++;
	p[0] = op; p[1] = dst; p[2] = a; p[3] = rb; p[4] = k0; p[5] = k1;
	return (int)(b->ninsn - 1);
}

static int
new_reg(builder *b, int *next)
{
	if (*next >= SCAN_MAX_REGS)
		return fail(b, "the expression needs more than %d registers", SCAN_MAX_REGS);
	if (*next + 1 > b->maxreg)
		b->maxreg = *next + 1;
	return (*next)++;
}

/* value type of a TypeOpCode: 0 if the scan VM cannot hold it in a register */
static int
reg_vtype(uint32_t type_code, int *p_bytes)
{
	switch (type_code)
	{
		case TypeOpCode__bool:   *p_bytes = 1; return SOP_VT_BOOL;
		case TypeOpCode__int1:   *p_bytes = 1; return SOP_VT_INT;
		case TypeOpCode__int2:   *p_bytes = 2; return SOP_VT_INT;
		case TypeOpCode__int4:   *p_bytes = 4; return SOP_VT_INT;
		case TypeOpCode__int8:   *p_bytes = 8; return SOP_VT_INT;
		case TypeOpCode__float4: *p_bytes = 4; return SOP_VT_FLOAT4;
		default:
			break;
	}
	return 0;
}

static int expr_to_reg(builder *b, const kern_expression *kexp, int *next);

static int
emit_const(builder *b, const kern_expression *kexp, int *next)
{
	int			bytes, vt = reg_vtype(kexp->exptype, &bytes);
	int			dst;
	int64_t		v = 0;

	if (!vt)
		return fail(b, "constant of an unsupported type (opcode type %u)", kexp->exptype);
	if (!kexp->u.c.const_isnull)
	{
		const unsigned char *p = (const unsigned char *) kexp->u.c.const_value;

		switch (bytes)
		{
			case 1:
				v = (vt == SOP_VT_INT) ? (int64_t)(int8_t) p[0] : (int64_t) p[0];
				break;
			case 2:
			{
				int16_t		x;
				memcpy(&x, p, 2);
				v = x;
				break;
			}
			case 4:
			{
				int32_t		x;
				uint32_t	u;
				memcpy(&x, p, 4);
				memcpy(&u, p, 4);
				v = (vt == SOP_VT_FLOAT4) ? (int64_t) u : (int64_t) x;
				break;
			}
			default:
				memcpy(&v, p, 8);
				break;
		}
	}
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	if (emit(b, SOP_CONST, dst, kexp->u.c.const_isnull ? 1 : 0, 0,
			 (uint32_t)((uint64_t) v & 0xffffffffU), (uint32_t)((uint64_t) v >> 32)) < 0)
		return -1;
	return dst;
}

/* a query parameter is a constant for the session: its datum is stored in the session info */
static int
emit_param(builder *b, const kern_expression *kexp, int *next)
{
	const kern_session_info *session = b->session;
	uint32_t	id = kexp->u.p.param_id;
	int			bytes, vt = reg_vtype(kexp->exptype, &bytes);
	int			dst;
	int64_t		v = 0;
	uint32_t	off;

	if (!vt)
		return fail(b, "parameter of an unsupported type (opcode type %u)", kexp->exptype);
	if (id >= session->nparams)
		return fail(b, "parameter %u does not exist", id);
	off = session->poffset[id];
	if (off != 0)
	{
		const unsigned char *p = (const unsigned char *) session + off;

		switch (bytes)
		{
			case 1:
				v = (vt == SOP_VT_INT) ? (int64_t)(int8_t) p[0] : (int64_t) p[0];
				break;
			case 2:
			{
				int16_t		x;
				memcpy(&x, p, 2);
				v = x;
				break;
			}
			case 4:
			{
				int32_t		x;
				uint32_t	u;
				memcpy(&x, p, 4);
				memcpy(&u, p, 4);
				v = (vt == SOP_VT_FLOAT4) ? (int64_t) u : (int64_t) x;
				break;
			}
			default:
				memcpy(&v, p, 8);
				break;
		}
	}
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	if (emit(b, SOP_CONST, dst, off == 0 ? 1 : 0, 0,
			 (uint32_t)((uint64_t) v & 0xffffffffU), (uint32_t)((uint64_t) v >> 32)) < 0)
		return -1;
	return dst;
}

static int
emit_boolexpr(builder *b, const kern_expression *kexp, int *next)
{
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);
	int			mark = *next;
	int			acc = new_reg(b, next);
	int			first_step = -1;
	uint32_t	nsteps = 0;
	uint32_t	end;
	int			is_and = (kexp->opcode == FuncOpCode__BoolExpr_And);
	int			steps[64];

	if (acc < 0)
		return -1;
	if (kexp->nr_args > 64)
		return fail(b, "too many arguments in AND/OR");
	/* the identity of the accumulator: true for AND, false for OR */
	if (emit(b, SOP_CONST, acc, 0, 0, is_and ? 1 : 0, 0) < 0)
		return -1;
	for (uint32_t i = 0; i < kexp->nr_args; i++, karg = KEXP_NEXT_ARG(karg))
	{
		int		r;

		*next = acc + 1;
		r = expr_to_reg(b, karg, next);
		if (r < 0)
			return -1;
		steps[nsteps] = emit(b, is_and ? SOP_AND : SOP_OR, acc, r, 0, 0, 0);
		if (steps[nsteps] < 0)
			return -1;
		if (first_step < 0)
			first_step = steps[nsteps];
		nsteps++;
	}
	end = b->ninsn;
	for (uint32_t i = 0; i < nsteps; i++)
		b->insn[6 * steps[i] + 3] = end;
	*next = mark + 1;
	return acc;
}

static int
expr_to_reg(builder *b, const kern_expression *kexp, int *next)
{
	const kern_expression *karg;
	scan_func_class	fc;
	int			mark = *next;
	int			ra, rb, dst;

	switch (kexp->opcode)
	{
		case FuncOpCode__ConstExpr:
			return emit_const(b, kexp, next);

		case FuncOpCode__VarExpr:
		{
			uint32_t	slot = kexp->u.v.var_slot_id;
			int			bytes;

			if (slot >= b->nslots || slot >= SCAN_MAX_REGS)
				return fail(b, "variable slot %u is out of range", slot);
			if (!reg_vtype(kexp->exptype, &bytes))
				return fail(b, "variable of an unsupported type (opcode type %u)", kexp->exptype);
			if (b->slot_resno[slot] <= 0)
				return fail(b, "variable slot %u is not loaded from the scan relation", slot);
			b->slot_used[slot] = 1;
			return (int) slot;
		}

		case FuncOpCode__BoolExpr_And:
		case FuncOpCode__BoolExpr_Or:
			return emit_boolexpr(b, kexp, next);

		case FuncOpCode__BoolExpr_Not:
		case FuncOpCode__NullTestExpr_IsNull:
		case FuncOpCode__NullTestExpr_IsNotNull:
		{
			uint32_t	op = (kexp->opcode == FuncOpCode__BoolExpr_Not ? SOP_NOT :
							  kexp->opcode == FuncOpCode__NullTestExpr_IsNull ? SOP_ISNULL
																			  : SOP_ISNOTNULL);

			const kern_expression *kin;
			int			bytes;

			if (kexp->nr_args != 1)
				return fail(b, "malformed %s", scan_vm_opcode_name(kexp->opcode));
			kin = KEXP_FIRST_ARG(kexp);
			if (op != SOP_NOT && kin->opcode == FuncOpCode__VarExpr &&
				!reg_vtype(kin->exptype, &bytes))
			{
				/* IS [NOT] NULL of a column of any type needs only its NULL-ness */
				uint32_t	slot = kin->u.v.var_slot_id;

				if (slot >= b->nslots || slot >= SCAN_MAX_REGS || b->slot_resno[slot] <= 0)
					return fail(b, "variable slot %u is not loaded from the scan relation", slot);
				b->slot_used[slot] = 1;
				b->slot_nullonly[slot] = 1;
				ra = (int) slot;
			}
			else
			{
				ra = expr_to_reg(b, kin, next);
				if (ra < 0)
					return -1;
			}
			*next = mark;
			dst = new_reg(b, next);
			if (dst < 0 || emit(b, op, dst, ra, 0, 0, 0) < 0)
				return -1;
			return dst;
		}

		case FuncOpCode__ParamExpr:
			return emit_param(b, kexp, next);

		default:
			break;
	}

	fc = scan_vm_classify_func(kexp->opcode);
	if (fc.cls == 0)
		return fail(b, "%s is not supported by the Vulkan GpuScan yet", scan_vm_opcode_name(kexp->opcode));

	if (fc.cls == SOP_CAST || fc.cls == SOP_MOV || fc.cls == SOP_I2F || fc.cls == SOP_F2I ||
		fc.cls == -SOP_ARITH)
	{
		if (kexp->nr_args != 1)
			return fail(b, "malformed %s", scan_vm_opcode_name(kexp->opcode));
		ra = expr_to_reg(b, KEXP_FIRST_ARG(kexp), next);
		if (ra < 0)
			return -1;
		if (fc.cls == -SOP_ARITH)
		{
			/* unary minus: 0 - x */
			int		zero;

			*next = mark;
			zero = new_reg(b, next);
			dst = new_reg(b, next);
			if (zero < 0 || dst < 0 ||
				emit(b, SOP_CONST, zero, 0, 0, 0, 0) < 0 ||
				emit(b, SOP_ARITH, dst, zero, ra, SOP_ARITH_MI | (fc.width << 8), kexp->opcode) < 0)
				return -1;
			*next = dst + 1;
			return dst;
		}
		*next = mark;
		dst = new_reg(b, next);
		if (dst < 0 ||
			emit(b, fc.cls, dst, ra, 0, fc.width, kexp->opcode) < 0)
			return -1;
		return dst;
	}

	/* binary operators */
	if (kexp->nr_args != 2)
		return fail(b, "malformed %s", scan_vm_opcode_name(kexp->opcode));
	karg = KEXP_FIRST_ARG(kexp);
	ra = expr_to_reg(b, karg, next);
	if (ra < 0)
		return -1;
	rb = expr_to_reg(b, KEXP_NEXT_ARG(karg), next);
	if (rb < 0)
		return -1;
	*next = mark;
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	switch (fc.cls)
	{
		case SOP_CMPI:
		case SOP_CMPF:
			if (emit(b, fc.cls, dst, ra, rb, fc.kind, 0) < 0)
				return -1;
			break;
		case SOP_ARITH:
			if (emit(b, SOP_ARITH, dst, ra, rb, fc.kind | (fc.width << 8), kexp->opcode) < 0)
				return -1;
			break;
		case SOP_FARITH:
			if (emit(b, SOP_FARITH, dst, ra, rb, fc.kind, kexp->opcode) < 0)
				return -1;
			break;
		default:
			return fail(b, "%s is not supported by the Vulkan GpuScan yet", scan_vm_opcode_name(kexp->opcode));
	}
	return dst;
}

void
scan_prog_free(scan_prog *prog)
{
	free(prog->words);
	free(prog->attrs);
	memset(prog, 0, sizeof(*prog));
}

int
scan_prog_build(const kern_session_info *session, const kern_data_store *kds_src,
				scan_prog *prog, char *err, size_t errlen)
{
	builder		b;
	const kern_expression *kload, *kquals, *kproj;
	const kern_varslot_desc *vs = SESSION_KVARS_SLOT_DESC(session);
	const kern_data_store *kds_dst = SESSION_KDS_DST_HEAD(session);
	int			qreg = -1;
	uint32_t	nq;
	int			next;
	int			ncols = 0;
	uint32_t	nattrs = 0;
	uint32_t   *comp_regs = NULL;
	uint32_t	ncomp = 0;
	uint32_t	nwords;
	uint32_t   *w;
	int			max_src = 0;

	memset(prog, 0, sizeof(*prog));
	memset(&b, 0, sizeof(b));
	err[0] = '\0';
	b.session = session;
	b.kds_src = kds_src;
	b.err = err;
	b.errlen = errlen;
	b.nslots = session->kcxt_kvars_nslots;

	if ((session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__SCAN)
	{
		fail(&b, "only GpuScan is implemented by the Vulkan service so far (task flags %08x)",
			 session->xpu_task_flags);
		goto error;
	}
	if (session->xpucode_groupby_actions || session->xpucode_gpusort_keydesc ||
		session->xpucode_join_quals_packed)
	{
		fail(&b, "aggregation, join and sort are not implemented by the Vulkan service yet");
		goto error;
	}
	if (b.nslots > SCAN_MAX_REGS)
	{
		fail(&b, "too many kernel variables (%u)", b.nslots);
		goto error;
	}
	if (!kds_dst || !SESSION_KEXP_PROJECTION(session))
	{
		fail(&b, "the scan has no projection");
		goto error;
	}

	/* which source column feeds each kernel variable */
	kload = SESSION_KEXP_LOAD_VARS(session, 0);
	if (kload)
	{
		for (int i = 0; i < kload->u.load.nitems; i++)
		{
			int		resno = kload->u.load.desc[i].vl_resno;
			int		slot = kload->u.load.desc[i].vl_slot_id;

			if (slot < 0 || slot >= (int) b.nslots)
				continue;
			if (resno < 0)
			{
				b.slot_sysattr[slot] = resno;	/* ctid, tableoid: only projectable */
				continue;
			}
			if (resno == 0 || resno > (int) kds_src->ncols)
				continue;
			b.slot_resno[slot] = resno;
		}
	}

	/* the quals, then the computed projection values */
	next = b.nslots;
	kquals = SESSION_KEXP_SCAN_QUALS(session);
	if (kquals)
	{
		qreg = expr_to_reg(&b, kquals, &next);
		if (qreg < 0)
			goto error;
	}
	nq = b.ninsn;

	kproj = SESSION_KEXP_PROJECTION(session);
	nattrs = kproj->u.proj.nattrs;
	prog->attrs = calloc(nattrs ? nattrs : 1, sizeof(scan_proj_attr));
	comp_regs = calloc(nattrs ? nattrs : 1, sizeof(uint32_t));
	if (!prog->attrs || !comp_regs)
	{
		fail(&b, "out of memory");
		goto error;
	}
	prog->nattrs = nattrs;
	if (kds_dst->ncols < nattrs)
	{
		fail(&b, "projection has %u columns but the result store has %u", nattrs, kds_dst->ncols);
		goto error;
	}
	{
		const kern_expression *ksave = KEXP_FIRST_ARG(kproj);
		int		perm = b.nslots;		/* computed values live above this */

		if (kproj->nr_args != nattrs)
		{
			fail(&b, "projection arguments (%u) do not match its attributes (%u)",
				 kproj->nr_args, nattrs);
			goto error;
		}
		for (uint32_t j = 0; j < nattrs; j++, ksave = KEXP_NEXT_ARG(ksave))
		{
			const kern_expression *kin;
			scan_proj_attr *pa = &prog->attrs[j];

			if (ksave->opcode != FuncOpCode__SaveExpr || ksave->nr_args != 1)
			{
				fail(&b, "unexpected projection element %s", scan_vm_opcode_name(ksave->opcode));
				goto error;
			}
			kin = KEXP_FIRST_ARG(ksave);
			if (kin->opcode == FuncOpCode__VarExpr &&
				kin->u.v.var_slot_id < b.nslots &&
				(b.slot_sysattr[kin->u.v.var_slot_id] == -1 ||
				 b.slot_sysattr[kin->u.v.var_slot_id] == -6))
			{
				pa->kind = SCAN_PROJ_SYS;
				pa->resno = b.slot_sysattr[kin->u.v.var_slot_id];
			}
			else if (kin->opcode == FuncOpCode__VarExpr &&
				kin->u.v.var_slot_id < b.nslots &&
				b.slot_resno[kin->u.v.var_slot_id] > 0)
			{
				pa->kind = SCAN_PROJ_SRC;
				pa->resno = b.slot_resno[kin->u.v.var_slot_id];
				if (pa->resno > max_src)
					max_src = pa->resno;
			}
			else
			{
				int		bytes, r;
				int		nx = perm;

				if (!reg_vtype(ksave->exptype, &bytes))
				{
					fail(&b, "computed column of an unsupported type (opcode type %u)", ksave->exptype);
					goto error;
				}
				r = expr_to_reg(&b, kin, &nx);
				if (r < 0)
					goto error;
				if (r < (int) b.nslots)		/* a slot register: copy, results must be stable */
				{
					int		d = new_reg(&b, &nx);

					if (d < 0 || emit(&b, SOP_MOV, d, r, 0, 0, 0) < 0)
						goto error;
					r = d;
				}
				pa->kind = SCAN_PROJ_COMP;
				pa->comp = ncomp;
				pa->comp_bytes = bytes;
				comp_regs[ncomp++] = r;
				perm = r + 1;
			}
		}
	}

	/* the columns the kernel must walk: up to the highest slot the program reads */
	for (uint32_t s = 0; s < b.nslots; s++)
	{
		if (b.slot_used[s])
		{
			int		bytes;

			b.slot_vtype[s] = reg_vtype(vs[s].vs_type_code, &bytes);
			b.slot_bytes[s] = bytes;
			if (!b.slot_vtype[s] && b.slot_nullonly[s])
				b.slot_vtype[s] = SOP_VT_NULLONLY;
			else if (!b.slot_vtype[s] || kds_src->colmeta[b.slot_resno[s] - 1].attlen != bytes ||
				!kds_src->colmeta[b.slot_resno[s] - 1].attbyval)
			{
				fail(&b, "column %d does not have the type the expression expects", b.slot_resno[s]);
				goto error;
			}
			if (b.slot_resno[s] > ncols)
				ncols = b.slot_resno[s];
		}
	}

	nwords = PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp;
	w = calloc(nwords, sizeof(uint32_t));
	if (!w)
	{
		fail(&b, "out of memory");
		goto error;
	}
	w[0] = PROG_HDR_WORDS;						/* coloff */
	w[1] = PROG_HDR_WORDS + ncols;				/* loadoff (unused) */
	w[2] = PROG_HDR_WORDS + ncols;				/* insnoff */
	w[3] = ncols;
	w[4] = 0;
	w[5] = nq;
	w[6] = b.ninsn;
	w[7] = (qreg < 0) ? 0xffffffffU : (uint32_t) qreg;
	w[8] = ncomp;
	w[9] = PROG_HDR_WORDS + ncols + 6 * b.ninsn;	/* compoff */
	for (int c = 0; c < ncols; c++)
	{
		const kern_colmeta *cm = &kds_src->colmeta[c];
		uint32_t	alen = (cm->attlen == -1 ? 0xff : cm->attlen == -2 ? 0xfe : (uint32_t) cm->attlen);
		uint32_t	slot = 0xff, vt = 0;

		for (uint32_t s = 0; s < b.nslots; s++)
			if (b.slot_used[s] && b.slot_resno[s] == c + 1)
			{
				slot = s;
				vt = b.slot_vtype[s];
			}
		w[PROG_HDR_WORDS + c] = alen | ((uint32_t) cm->attalign << 8) | (vt << 12) | (slot << 16);
	}
	memcpy(w + PROG_HDR_WORDS + ncols, b.insn, sizeof(uint32_t) * 6 * b.ninsn);
	memcpy(w + PROG_HDR_WORDS + ncols + 6 * b.ninsn, comp_regs, sizeof(uint32_t) * ncomp);

	prog->words = w;
	prog->nwords = nwords;
	prog->ncomp = ncomp;
	prog->nregs = (b.maxreg > (int) b.nslots ? b.maxreg : (int) b.nslots);
	prog->max_src_resno = max_src;
	free(b.insn);
	free(comp_regs);
	return 0;

error:
	free(b.insn);
	free(comp_regs);
	scan_prog_free(prog);
	if (err[0] == '\0')
		snprintf(err, errlen, "cannot build the scan program");
	return -1;
}
