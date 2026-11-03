/*
 * scan_prog.c
 *
 * kern_expression trees (xpu_common.h) -> scan VM program. See scan_vm_ops.inc for the
 * instruction set and scan.slang for the program layout.
 *
 * Supported: bool, int1/2/4/8 and float4 values; comparison, +, -, *, /, % on them (with
 * PostgreSQL's overflow and zero-divide errors), casts between the integer types, unary
 * plus/minus, AND/OR/NOT with three-valued logic and IS [NOT] NULL; (A6) date, time, timestamp and
 * timestamptz as integers with date arithmetic, and text/varchar/bpchar as string registers (equality,
 * C-collation ordering, LIKE of a literal with % around it, length). Anything else makes
 * scan_prog_build() fail, and the query then reports that the service cannot run it.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xpu_common.h"
#include "scan_prog.h"
#include "scan_caps.h"
#include "specialize.h"


#define PROG_HDR_WORDS		24
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
	int			slot_iresno[SCAN_MAX_REGS];	/* attribute of an inner relation (a join), 0 if none */
	int			slot_idepth[SCAN_MAX_REGS];	/* ... and which inner relation (1 ...) */
	int			slot_iused[SCAN_MAX_REGS];
	int			slot_ivtype[SCAN_MAX_REGS];
	int			slot_sysattr[SCAN_MAX_REGS];	/* system attribute (ctid, ...) of a slot, 0 if none */
	int			slot_nullonly[SCAN_MAX_REGS];/* only IS [NOT] NULL reads the slot */
	int			slot_used[SCAN_MAX_REGS];
	int			slot_vtype[SCAN_MAX_REGS];
	int			slot_bytes[SCAN_MAX_REGS];
	int			slot_valread[SCAN_MAX_REGS];/* a column slot whose value is read (not only IS [NOT] NULL) */
	int			slot_strval[SCAN_MAX_REGS];	/* a string slot whose value (not only its NULL-ness) is read */
	unsigned char *pool;			/* constant pool: string literals (header word 20 = its word offset) */
	uint32_t	pool_len;
	uint32_t	pool_cap;
	int			no_strings;			/* a join program: strings are not supported */
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
		/* A6: dates and times are integers (days, microseconds) */
		case TypeOpCode__date:        *p_bytes = 4; return SOP_VT_INT;
		case TypeOpCode__time:
		case TypeOpCode__timestamp:
		case TypeOpCode__timestamptz: *p_bytes = 8; return SOP_VT_INT;
		/* A6: strings are registers that point into a buffer */
		case TypeOpCode__text:
		case TypeOpCode__bpchar:      *p_bytes = 0; return SOP_VT_STR;
		/* A6: numeric is { unscaled int64, scale } when the value fits (otherwise the row is rechecked) */
		case TypeOpCode__numeric:     *p_bytes = 0; return SOP_VT_NUM;
		default:
			break;
	}
	return 0;
}

/* where the payload of a plain (not compressed, not external) varlena is; 0 on failure */
static int
varlena_payload(const unsigned char *p, const unsigned char **p_data, uint32_t *p_len)
{
	if (p[0] & 1)
	{
		if (p[0] == 1)
			return 0;			/* external */
		*p_len = ((p[0] >> 1) & 0x7f) - 1;
		*p_data = p + 1;
		return 1;
	}
	if ((p[0] & 3) == 2)
		return 0;				/* compressed */
	{
		uint32_t	h;

		memcpy(&h, p, 4);
		*p_len = ((h >> 2) & 0x3fffffffU) - 4;
		*p_data = p + 4;
		return 1;
	}
}

/* add bytes to the constant pool; returns their offset, or -1 */
static int64_t
pool_add(builder *b, const unsigned char *data, uint32_t len)
{
	uint32_t	off = b->pool_len;

	if (off + len + 4 > b->pool_cap)
	{
		uint32_t	ncap = (b->pool_cap ? b->pool_cap * 2 : 256);
		unsigned char *np;

		while (ncap < off + len + 4)
			ncap *= 2;
		np = realloc(b->pool, ncap);
		if (!np)
			return fail(b, "out of memory");
		b->pool = np;
		b->pool_cap = ncap;
	}
	memcpy(b->pool + off, data, len);
	b->pool_len = (off + len + 3) & ~3U;		/* words */
	memset(b->pool + off + len, 0, b->pool_len - off - len);
	return off;
}

/* a numeric constant as a register: it must fit { int64, scale } (the planner checks, too) */
static int
emit_numconst(builder *b, const unsigned char *datum, int isnull, int *next)
{
	int			dst;
	int64_t		v = 0;
	uint32_t	sc = 0;

	if (b->no_strings)
		return fail(b, "numeric is not supported in a join");
	if (!isnull && !scan_vm_numeric_datum_decode(datum, &v, &sc))
		return fail(b, "a numeric constant that does not fit 64 bits (or NaN/Infinity)");
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	return emit(b, SOP_NUMCONST, dst, isnull ? 1 : 0, sc,
				(uint32_t)((uint64_t) v & 0xffffffffU), (uint32_t)((uint64_t) v >> 32)) < 0 ? -1 : dst;
}

/* a string constant (a text datum) as a register */
static int
emit_conststr(builder *b, const unsigned char *datum, int isnull, int *next)
{
	int			dst;
	const unsigned char *data = NULL;
	uint32_t	len = 0;
	int64_t		off = 0;

	if (b->no_strings)
		return fail(b, "strings are not supported in a join");
	if (!isnull)
	{
		if (!varlena_payload(datum, &data, &len))
			return fail(b, "a compressed or external string constant");
		if ((off = pool_add(b, data, len)) < 0)
			return -1;
	}
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	if (isnull)
		return emit(b, SOP_CONST, dst, 1, 0, 0, 0) < 0 ? -1 : dst;
	return emit(b, SOP_CONSTSTR, dst, 0, 0, (uint32_t) off, len) < 0 ? -1 : dst;
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
	if (vt == SOP_VT_STR)
		return emit_conststr(b, (const unsigned char *) kexp->u.c.const_value, kexp->u.c.const_isnull, next);
	if (vt == SOP_VT_NUM)
		return emit_numconst(b, (const unsigned char *) kexp->u.c.const_value, kexp->u.c.const_isnull, next);
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
	if (vt == SOP_VT_STR)
		return emit_conststr(b, (const unsigned char *) session + off, off == 0, next);
	if (vt == SOP_VT_NUM)
		return emit_numconst(b, (const unsigned char *) session + off, off == 0, next);
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
	int			is_and = (kexp->opcode != FuncOpCode__BoolExpr_Or);
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
			if (reg_vtype(kexp->exptype, &bytes) == SOP_VT_STR || reg_vtype(kexp->exptype, &bytes) == SOP_VT_NUM)
			{
				if (b->no_strings)
					return fail(b, "strings and numerics are not supported in a join");
				b->slot_strval[slot] = 1;
			}
			if (b->slot_resno[slot] > 0)
			{
				b->slot_used[slot] = 1;
				b->slot_valread[slot] = 1;
			}
			else if (b->slot_iresno[slot] > 0)
				b->slot_iused[slot] = 1;
			else
				return fail(b, "variable slot %u is not loaded from the scan relation", slot);
			return (int) slot;
		}

		case FuncOpCode__BoolExpr_And:
		case FuncOpCode__BoolExpr_Or:
		case FuncOpCode__JoinQuals:
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
				(!reg_vtype(kin->exptype, &bytes) || reg_vtype(kin->exptype, &bytes) == SOP_VT_STR ||
			  reg_vtype(kin->exptype, &bytes) == SOP_VT_NUM ||
			  kin->exptype == TypeOpCode__date || kin->exptype == TypeOpCode__time ||
			  kin->exptype == TypeOpCode__timestamp || kin->exptype == TypeOpCode__timestamptz))
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
		fc.cls == -SOP_ARITH || fc.cls == SOP_D2TS || fc.cls == SOP_TS2D || fc.cls == SOP_STRLEN ||
		fc.cls == SOP_I2N || fc.cls == SOP_N2I || (fc.cls == SOP_NUMARITH && fc.kind >= SOP_NUMOP_NEG))
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
		if (fc.cls == SOP_NUMARITH)		/* numeric_uminus, numeric_abs */
		{
			if (dst < 0 || emit(b, SOP_NUMARITH, dst, ra, 0, fc.kind, kexp->opcode) < 0)
				return -1;
			return dst;
		}
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
	if (fc.cls == SOP_STRLIKE)
	{
		/* the pattern is a literal with % around it: its literal goes to the constant pool */
		const kern_expression *kpat = KEXP_NEXT_ARG(karg);
		const unsigned char *pdatum;
		const unsigned char *pdata;
		uint32_t	plen;
		unsigned char *lit;
		int			litlen = 0, shape;

		if (kpat->opcode == FuncOpCode__ConstExpr && !kpat->u.c.const_isnull)
			pdatum = (const unsigned char *) kpat->u.c.const_value;
		else if (kpat->opcode == FuncOpCode__ParamExpr && kpat->u.p.param_id < b->session->nparams &&
				 b->session->poffset[kpat->u.p.param_id] != 0)
			pdatum = (const unsigned char *) b->session + b->session->poffset[kpat->u.p.param_id];
		else
			return fail(b, "the pattern of LIKE is not a constant");
		if (!varlena_payload(pdatum, &pdata, &plen))
			return fail(b, "a compressed or external LIKE pattern");
		lit = malloc(plen + 1);
		if (!lit)
			return fail(b, "out of memory");
		shape = scan_vm_like_shape(pdata, (int) plen, lit, &litlen);
		if (shape < 0)
		{
			free(lit);
			return fail(b, "the LIKE pattern has a wildcard the Vulkan service cannot match");
		}
		{
			int64_t		off = pool_add(b, lit, (uint32_t) litlen);

			free(lit);
			if (off < 0)
				return -1;
			rb = new_reg(b, next);
			if (rb < 0 || emit(b, SOP_CONSTSTR, rb, 0, 0, (uint32_t) off, (uint32_t) litlen) < 0)
				return -1;
		}
		*next = mark;
		dst = new_reg(b, next);
		if (dst < 0 ||
			emit(b, SOP_STRLIKE, dst, ra, rb,
				 shape | (fc.kind ? 0x100 : 0) | (fc.width ? 0x200 : 0), kexp->opcode) < 0)
			return -1;
		return dst;
	}
	rb = expr_to_reg(b, KEXP_NEXT_ARG(karg), next);
	if (rb < 0)
		return -1;
	if (fc.cls == SOP_CMPI && fc.width != 0)
	{
		/* date <op> timestamp: compare as timestamps; the date side is converted */
		int			t = new_reg(b, next);
		int			ds = (fc.width == 1 ? ra : rb);

		if (t < 0 || emit(b, SOP_D2TS, t, ds, 0, 0, kexp->opcode) < 0)
			return -1;
		if (fc.width == 1)
			ra = t;
		else
			rb = t;
	}
	if (fc.cls == SOP_DATEOP && fc.width == 1)
	{
		int			t = ra;			/* integer_pl_date(int4, date): date first */

		ra = rb;
		rb = t;
	}
	*next = mark;
	dst = new_reg(b, next);
	if (dst < 0)
		return -1;
	switch (fc.cls)
	{
		case SOP_DATEOP:
			if (emit(b, SOP_DATEOP, dst, ra, rb, fc.kind, kexp->opcode) < 0)
				return -1;
			break;
		case SOP_STRCMP:
			if (emit(b, SOP_STRCMP, dst, ra, rb, fc.kind, fc.width) < 0)
				return -1;
			break;
		case SOP_CMPI:
		case SOP_CMPF:
		case SOP_NUMCMP:
			if (emit(b, fc.cls, dst, ra, rb, fc.kind, 0) < 0)
				return -1;
			break;
		case SOP_NUMARITH:
			if (emit(b, SOP_NUMARITH, dst, ra, rb, fc.kind, kexp->opcode) < 0)
				return -1;
			break;
		case SOP_NUMDIV:
		case SOP_NUMRND:
			if (emit(b, fc.cls, dst, ra, rb, fc.kind, kexp->opcode) < 0)
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
	spec_release(prog);
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
	int			agg_mode;
	uint32_t	agg_tab[4 * SCAN_AGG_MAX];
	int			nagg = 0, nattrs_out = 0;
	int			nkeys = 0;
	uint32_t	key_reg_tab[2 * SCAN_GROUP_KEYS];	/* registers, then (numeric key) scale + 1 or 0 */
	int			key_slot[SCAN_GROUP_KEYS];
	int			key_isnum[SCAN_GROUP_KEYS];

	memset(prog, 0, sizeof(*prog));
	memset(&b, 0, sizeof(b));
	err[0] = '\0';
	b.session = session;
	b.kds_src = kds_src;
	b.err = err;
	b.errlen = errlen;
	b.nslots = session->kcxt_kvars_nslots;

	agg_mode = ((session->xpu_task_flags & DEVTASK__MASK) == DEVTASK__PREAGG);
	if (!agg_mode && (session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__SCAN)
	{
		fail(&b, "only GpuScan and GpuPreAgg are implemented by the Vulkan service so far (task flags %08x)",
			 session->xpu_task_flags);
		goto error;
	}
	/* GPU-Sort is applied to the groups of a GpuPreAgg by the service itself (gpusort.c), not by the scan */
	if ((session->xpucode_gpusort_keydesc && !agg_mode) || session->xpucode_join_quals_packed)
	{
		fail(&b, "join and sort are not implemented by the Vulkan service yet");
		goto error;
	}
	if (agg_mode)
	{
		if (!session->xpucode_groupby_actions)
		{
			fail(&b, "GpuPreAgg without aggregate actions");
			goto error;
		}
	}
	else if (session->xpucode_groupby_actions)
	{
		fail(&b, "unexpected aggregate actions in a scan");
		goto error;
	}
	if (b.nslots > SCAN_MAX_REGS)
	{
		fail(&b, "too many kernel variables (%u)", b.nslots);
		goto error;
	}
	if (!agg_mode && (!kds_dst || !SESSION_KEXP_PROJECTION(session)))
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

	if (agg_mode)
	{
		const kern_expression *kagg = SESSION_KEXP_GROUPBY_ACTIONS(session);
		const kern_expression *ksave = KEXP_FIRST_ARG(kagg);
		int		slot_reg[SCAN_MAX_REGS];
		int		perm = b.nslots;

		for (int i = 0; i < SCAN_MAX_REGS; i++)
			slot_reg[i] = -1;
		nattrs_out = kagg->u.pagg.nattrs;
		if (nattrs_out < 1 || nattrs_out > SCAN_AGG_MAX)
		{
			fail(&b, "%d output columns (up to %d are supported)", nattrs_out, SCAN_AGG_MAX);
			goto error;
		}
		prog->nattrs = nattrs_out;
		/* the GROUP BY keys: the SaveExpr arguments of the key hash */
		if (session->xpucode_groupby_keyhash)
		{
			const kern_expression *khash = (const kern_expression *)
				((const char *) session + session->xpucode_groupby_keyhash);
			const kern_expression *kk = KEXP_FIRST_ARG(khash);

			if (khash->nr_args < 1 || khash->nr_args > SCAN_GROUP_KEYS)
			{
				fail(&b, "%u GROUP BY keys (up to %d are supported)", khash->nr_args, SCAN_GROUP_KEYS);
				goto error;
			}
			for (uint32_t j = 0; j < khash->nr_args; j++, kk = KEXP_NEXT_ARG(kk))
			{
				const kern_expression *kin;
				int		bytes, vt, r, nx = perm;

				if (kk->opcode != FuncOpCode__SaveExpr || kk->nr_args != 1 || kk->u.save.sv_slot_id >= b.nslots)
				{
					fail(&b, "unexpected GROUP BY key element %s", scan_vm_opcode_name(kk->opcode));
					goto error;
				}
				vt = reg_vtype(kk->exptype, &bytes);
				if (vt != SOP_VT_INT && vt != SOP_VT_BOOL && vt != SOP_VT_NUM)
				{
					fail(&b, "GROUP BY key of an unsupported type (opcode type %u)", kk->exptype);
					goto error;
				}
				key_isnum[nkeys] = (vt == SOP_VT_NUM);
				prog->key_scale[nkeys] = -1;
				key_reg_tab[nkeys] = 0;				/* (the scale words follow the registers; filled below) */
				kin = KEXP_FIRST_ARG(kk);
				if (kin->opcode == FuncOpCode__VarExpr &&
					kin->u.v.var_slot_id < b.nslots && b.slot_resno[kin->u.v.var_slot_id] > 0)
				{
					b.slot_used[kin->u.v.var_slot_id] = 1;
					b.slot_valread[kin->u.v.var_slot_id] = 1;
					b.slot_strval[kin->u.v.var_slot_id] = 1;		/* a numeric key is read, not only its NULL-ness */
					r = (int) kin->u.v.var_slot_id;
				}
				else
				{
					r = expr_to_reg(&b, kin, &nx);
					if (r < 0)
						goto error;
					if (r < (int) b.nslots)
					{
						int		d = new_reg(&b, &nx);

						if (d < 0 || emit(&b, SOP_MOV, d, r, 0, 0, 0) < 0)
							goto error;
						r = d;
					}
					perm = r + 1;
				}
				slot_reg[kk->u.save.sv_slot_id] = r;
				key_slot[nkeys] = (int) kk->u.save.sv_slot_id;
				key_reg_tab[nkeys] = (uint32_t) r;
				nkeys++;
			}
		}
		/* the argument values: SaveExpr slots, computed after the quals */
		for (uint32_t j = 0; j < kagg->nr_args; j++, ksave = KEXP_NEXT_ARG(ksave))
		{
			const kern_expression *kin;
			uint32_t	slot;

			if (ksave->opcode != FuncOpCode__SaveExpr || ksave->nr_args != 1)
			{
				fail(&b, "unexpected aggregate argument element %s", scan_vm_opcode_name(ksave->opcode));
				goto error;
			}
			slot = ksave->u.save.sv_slot_id;
			kin = KEXP_FIRST_ARG(ksave);
			if (slot >= b.nslots)
			{
				fail(&b, "aggregate argument slot %u is out of range", slot);
				goto error;
			}
			if (kin->opcode == FuncOpCode__VarExpr &&
				kin->u.v.var_slot_id < b.nslots && b.slot_resno[kin->u.v.var_slot_id] > 0)
			{
				int		bytes;

				if (!reg_vtype(kin->exptype, &bytes))
				{
					fail(&b, "aggregate argument of an unsupported type (opcode type %u)", kin->exptype);
					goto error;
				}
				b.slot_used[kin->u.v.var_slot_id] = 1;
				b.slot_valread[kin->u.v.var_slot_id] = 1;
				b.slot_strval[kin->u.v.var_slot_id] = 1;
				slot_reg[slot] = (int) kin->u.v.var_slot_id;
			}
			else
			{
				int		nx = perm, r;

				r = expr_to_reg(&b, kin, &nx);
				if (r < 0)
					goto error;
				if (r < (int) b.nslots)
				{
					int		d = new_reg(&b, &nx);

					if (d < 0 || emit(&b, SOP_MOV, d, r, 0, 0, 0) < 0)
						goto error;
					r = d;
				}
				slot_reg[slot] = r;
				perm = r + 1;
			}
		}
		/* the aggregate table (the key columns of the output are not aggregates) */
		for (int i = 0; i < nattrs_out; i++)
		{
			const kern_aggregate_desc *d = &kagg->u.pagg.desc[i];

			prog->agg_action[i] = d->action;
			prog->agg_of_col[i] = -1;
			prog->key_of_col[i] = -1;
			if (d->action == KAGG_ACTION__VREF)
			{
				int		kidx = -1;

				for (int k = 0; k < nkeys; k++)
					if (key_slot[k] == d->arg0_slot_id)
						kidx = k;
				if (kidx < 0)
				{
					fail(&b, "output column %d refers to a slot that is not a GROUP BY key", i + 1);
					goto error;
				}
				prog->key_of_col[i] = (int8_t) kidx;
				prog->key_scale[kidx] = -1;
				key_reg_tab[nkeys + kidx] = 0;
				if (key_isnum[kidx])
				{
					/* a numeric key has the scale the plan derived (typmod of the output column) */
					if (d->typmod < (int32_t) 4)
					{
						fail(&b, "a numeric GROUP BY key without a known scale");
						goto error;
					}
					prog->key_scale[kidx] = (int) (((uint32_t)(d->typmod - 4)) & 0x7ffU);
					key_reg_tab[nkeys + kidx] = (uint32_t) prog->key_scale[kidx] + 1;
				}
				continue;
			}
			if (d->action == KAGG_ACTION__VREF_NOKEY)
			{
				fail(&b, "a column that depends on the GROUP BY key (VREF_NOKEY) is not supported yet");
				goto error;
			}
			prog->agg_of_col[i] = (int8_t) nagg;
			prog->agg_scale[nagg] = -1;
			int		sources[2] = {d->arg0_slot_id, d->filter_slot_id};
			int		regs_of[2] = {-1, -1};
			uint32_t	cls;

			switch (d->action)
			{
				case KAGG_ACTION__NROWS_ANY:	cls = 1; break;
				case KAGG_ACTION__NROWS_COND:	cls = 2; break;
				case KAGG_ACTION__PSUM_INT:
				case KAGG_ACTION__PSUM_INT64:
				case KAGG_ACTION__PAVG_INT:
				case KAGG_ACTION__PAVG_INT64:	cls = 3; break;
				case KAGG_ACTION__PSUM_NUMERIC:
				case KAGG_ACTION__PAVG_NUMERIC:
					/* a sum of numerics of one known scale is a sum of their unscaled int64 values */
					if (d->typmod < (int32_t) 4)
					{
						fail(&b, "a numeric aggregate without a known scale");
						goto error;
					}
					prog->agg_scale[nagg] = (int) (((uint32_t)(d->typmod - 4)) & 0x7ffU);
					cls = 3;
					break;
				case KAGG_ACTION__PMIN_INT32:
				case KAGG_ACTION__PMIN_INT64:	cls = 4; break;
				case KAGG_ACTION__PMAX_INT32:
				case KAGG_ACTION__PMAX_INT64:	cls = 5; break;
				default:
					fail(&b, "aggregate action %u is not implemented by the Vulkan service", d->action);
					goto error;
			}
			for (int k = 0; k < 2; k++)
			{
				int		sl = sources[k];

				if (sl < 0)
					continue;
				if (sl >= (int) b.nslots)
				{
					fail(&b, "aggregate slot %d is out of range", sl);
					goto error;
				}
				if (slot_reg[sl] >= 0)
					regs_of[k] = slot_reg[sl];
				else if (b.slot_resno[sl] > 0)
				{
					int		bytes;

					if (!reg_vtype(vs[sl].vs_type_code, &bytes))
					{
						fail(&b, "aggregate argument of an unsupported type");
						goto error;
					}
					b.slot_used[sl] = 1;
					b.slot_valread[sl] = 1;
					b.slot_strval[sl] = 1;
					regs_of[k] = sl;
				}
				else
				{
					fail(&b, "aggregate slot %d has no value", sl);
					goto error;
				}
			}
			if ((cls == 2 || cls >= 3) && regs_of[0] < 0)
			{
				fail(&b, "aggregate in column %d has no argument", i + 1);
				goto error;
			}
			agg_tab[4 * nagg + 0] = cls;
			agg_tab[4 * nagg + 1] = regs_of[0] < 0 ? 0xffffffffU : (uint32_t) regs_of[0];
			agg_tab[4 * nagg + 2] = regs_of[1] < 0 ? 0xffffffffU : (uint32_t) regs_of[1];
			agg_tab[4 * nagg + 3] = (prog->agg_scale[nagg] >= 0) ? (uint32_t) prog->agg_scale[nagg] + 1 : 0;
			nagg++;
		}
		for (int k = 0; k < nkeys; k++)
			if (key_isnum[k] && prog->key_scale[k] < 0)
			{
				fail(&b, "a numeric GROUP BY key %d is not an output column", k + 1);
				goto error;
			}
		prog->agg_mode = 1;
		prog->nagg = nagg;
		prog->nkeys = nkeys;
	}
	else
	{
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

					if (!reg_vtype(ksave->exptype, &bytes) || reg_vtype(ksave->exptype, &bytes) == SOP_VT_STR)
					{
						fail(&b, "computed column of an unsupported type (opcode type %u)", ksave->exptype);
						goto error;
					}
					int		sv = (int) ksave->u.save.sv_slot_id;

					r = expr_to_reg(&b, kin, &nx);
					if (r < 0)
						goto error;
					if (sv < (int) b.nslots && sv != r &&
						b.slot_resno[sv] <= 0 && b.slot_iresno[sv] <= 0 && b.slot_sysattr[sv] == 0)
					{
						/*
						 * The SaveExpr's own slot register is free (nothing reads it as a
						 * variable): keep the result there, so the computed columns do not
						 * add to the registers above the slots and the temporaries can be
						 * reused by the next column.
						 */
						if (emit(&b, SOP_MOV, sv, r, 0, 0, 0) < 0)
							goto error;
						r = sv;
					}
					else
					{
						if (r < (int) b.nslots)		/* a slot register: copy, results must be stable */
						{
							int		d = new_reg(&b, &nx);

							if (d < 0 || emit(&b, SOP_MOV, d, r, 0, 0, 0) < 0)
								goto error;
							r = d;
						}
						perm = r + 1;
					}
					pa->kind = SCAN_PROJ_COMP;
					pa->comp = ncomp;
					pa->comp_bytes = bytes;
					pa->comp_numeric = (reg_vtype(ksave->exptype, &bytes) == SOP_VT_NUM);
					comp_regs[ncomp++] = r;
				}
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
			if (b.slot_vtype[s] == SOP_VT_INT && b.slot_nullonly[s] && !b.slot_valread[s])
				b.slot_vtype[s] = SOP_VT_NULLONLY;		/* a date/time column that only IS [NOT] NULL reads */
			else if ((b.slot_vtype[s] == SOP_VT_STR || b.slot_vtype[s] == SOP_VT_NUM) && !b.slot_strval[s])
				b.slot_vtype[s] = SOP_VT_NULLONLY;		/* only IS [NOT] NULL reads it */
			else if (b.slot_vtype[s] == SOP_VT_STR || b.slot_vtype[s] == SOP_VT_NUM)
			{
				if (kds_src->colmeta[b.slot_resno[s] - 1].attlen != -1)
				{
					fail(&b, "column %d does not have the type the expression expects", b.slot_resno[s]);
					goto error;
				}
			}
			else if (!b.slot_vtype[s] && b.slot_nullonly[s])
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

	nwords = PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp + 4 * nagg + 2 * nkeys + b.pool_len / 4;
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
	w[10] = nagg;
	w[11] = PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp;	/* aggoff */
	w[12] = nkeys;
	w[13] = PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp + 4 * nagg;	/* keyoff */
	w[20] = PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp + 4 * nagg + 2 * nkeys;	/* constant pool (strings) */
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
	memcpy(w + PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp, agg_tab, sizeof(uint32_t) * 4 * nagg);
	memcpy(w + PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp + 4 * nagg, key_reg_tab, sizeof(uint32_t) * 2 * nkeys);
	if (b.pool_len > 0)
		memcpy(w + PROG_HDR_WORDS + ncols + 6 * b.ninsn + ncomp + 4 * nagg + 2 * nkeys, b.pool, b.pool_len);

	prog->words = w;
	prog->nwords = nwords;
	prog->ncomp = ncomp;
	prog->nregs = (b.maxreg > (int) b.nslots ? b.maxreg : (int) b.nslots);
	prog->max_src_resno = max_src;
	free(b.insn);
	free(b.pool);
	free(comp_regs);
	return 0;

error:
	free(b.insn);
	free(b.pool);
	free(comp_regs);
	scan_prog_free(prog);
	if (err[0] == '\0')
		snprintf(err, errlen, "cannot build the scan program");
	return -1;
}


/* ----------------------------------------------------------------
 * GpuJoin
 * ----------------------------------------------------------------
 */

/* the HashValue of the join keys: HASHINIT, one HASHMERGE per key, HASHFIN; returns its register */
static int
emit_hash_value(builder *b, const kern_expression *khash, int *next)
{
	const kern_expression *karg = KEXP_FIRST_ARG(khash);
	int			mark = *next;
	int			acc = new_reg(b, next);

	if (acc < 0 || emit(b, SOP_HASHINIT, acc, 0, 0, 0, 0) < 0)
		return -1;
	for (uint32_t i = 0; i < khash->nr_args; i++, karg = KEXP_NEXT_ARG(karg))
	{
		int		bytes, r;

		if (!reg_vtype(karg->exptype, &bytes) || karg->exptype == TypeOpCode__float4 ||
			reg_vtype(karg->exptype, &bytes) == SOP_VT_STR || reg_vtype(karg->exptype, &bytes) == SOP_VT_NUM)
			return fail(b, "join key of an unsupported type (opcode type %u)", karg->exptype);
		*next = acc + 1;
		r = expr_to_reg(b, karg, next);
		if (r < 0 || emit(b, SOP_HASHMERGE, acc, r, 0, bytes, 0) < 0)
			return -1;
	}
	if (emit(b, SOP_HASHFIN, acc, 0, 0, 0, 0) < 0)
		return -1;
	*next = mark + 1;
	return acc;
}

/*
 * Aggregation over a join (GpuPreAgg on top of GpuJoin): in place of the projection, the section
 * after the join computes the GROUP BY key expressions, the aggregate arguments and the FILTER
 * conditions into registers, evaluated at the point where a joined row is complete (every variable
 * of the outer relation and of each inner relation is in its register then). Fills the aggregate
 * table and the key register table of the program; the kernels read them as agg.slang / gagg.slang do.
 */
static uint32_t
join_agg_class(uint32_t action)
{
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:	return 1;
		case KAGG_ACTION__NROWS_COND:	return 2;
		case KAGG_ACTION__PSUM_INT:
		case KAGG_ACTION__PSUM_INT64:
		case KAGG_ACTION__PAVG_INT:
		case KAGG_ACTION__PAVG_INT64:	return 3;
		case KAGG_ACTION__PMIN_INT32:
		case KAGG_ACTION__PMIN_INT64:	return 4;
		case KAGG_ACTION__PMAX_INT32:
		case KAGG_ACTION__PMAX_INT64:	return 5;
		default:
			break;
	}
	return 0;
}

/* a kernel variable slot read directly (not through an expression): mark the column it comes from */
static int
join_use_slot(builder *b, uint32_t slot)
{
	int			bytes;
	const kern_varslot_desc *vs = SESSION_KVARS_SLOT_DESC(b->session);

	if (slot >= b->nslots || !reg_vtype(vs[slot].vs_type_code, &bytes) ||
		reg_vtype(vs[slot].vs_type_code, &bytes) == SOP_VT_STR)
		return fail(b, "aggregate argument of an unsupported type");
	if (b->slot_resno[slot] > 0)
	{
		b->slot_used[slot] = 1;
		b->slot_valread[slot] = 1;
	}
	else if (b->slot_iresno[slot] > 0)
		b->slot_iused[slot] = 1;
	else
		return fail(b, "aggregate slot %u has no value", slot);
	return (int) slot;
}

static int
join_build_agg(builder *b, scan_prog *prog, uint32_t *agg_tab, uint32_t *key_reg_tab,
			   int *p_nagg, int *p_nkeys)
{
	const kern_session_info *session = b->session;
	const kern_expression *kagg = SESSION_KEXP_GROUPBY_ACTIONS(session);
	const kern_expression *ksave = KEXP_FIRST_ARG(kagg);
	int			slot_reg[SCAN_MAX_REGS];
	int			key_slot[SCAN_GROUP_KEYS];
	int			perm = b->nslots;
	int			nagg = 0, nkeys = 0, nattrs_out = kagg->u.pagg.nattrs;

	for (int i = 0; i < SCAN_MAX_REGS; i++)
		slot_reg[i] = -1;
	if (nattrs_out < 1 || nattrs_out > SCAN_AGG_MAX)
		return fail(b, "%d output columns (up to %d are supported)", nattrs_out, SCAN_AGG_MAX);
	prog->nattrs = nattrs_out;
	/* the GROUP BY keys: the SaveExpr arguments of the key hash */
	if (session->xpucode_groupby_keyhash)
	{
		const kern_expression *khash = (const kern_expression *)
			((const char *) session + session->xpucode_groupby_keyhash);
		const kern_expression *kk = KEXP_FIRST_ARG(khash);

		if (khash->nr_args < 1 || khash->nr_args > SCAN_GROUP_KEYS)
			return fail(b, "%u GROUP BY keys (up to %d are supported)", khash->nr_args, SCAN_GROUP_KEYS);
		for (uint32_t j = 0; j < khash->nr_args; j++, kk = KEXP_NEXT_ARG(kk))
		{
			int		bytes, vt, r, nx = perm;

			if (kk->opcode != FuncOpCode__SaveExpr || kk->nr_args != 1 || kk->u.save.sv_slot_id >= b->nslots)
				return fail(b, "unexpected GROUP BY key element %s", scan_vm_opcode_name(kk->opcode));
			vt = reg_vtype(kk->exptype, &bytes);
			if (vt != SOP_VT_INT && vt != SOP_VT_BOOL)
				return fail(b, "GROUP BY key of an unsupported type (opcode type %u)", kk->exptype);
			r = expr_to_reg(b, KEXP_FIRST_ARG(kk), &nx);
			if (r < 0)
				return -1;
			if (r >= (int) b->nslots)
				perm = r + 1;
			slot_reg[kk->u.save.sv_slot_id] = r;
			key_slot[nkeys] = (int) kk->u.save.sv_slot_id;
			key_reg_tab[nkeys] = (uint32_t) r;
			nkeys++;
		}
	}
	/* the argument values: SaveExpr slots */
	for (uint32_t j = 0; j < kagg->nr_args; j++, ksave = KEXP_NEXT_ARG(ksave))
	{
		int		nx = perm, r;

		if (ksave->opcode != FuncOpCode__SaveExpr || ksave->nr_args != 1)
			return fail(b, "unexpected aggregate argument element %s", scan_vm_opcode_name(ksave->opcode));
		if (ksave->u.save.sv_slot_id >= b->nslots)
			return fail(b, "aggregate argument slot %u is out of range", ksave->u.save.sv_slot_id);
		r = expr_to_reg(b, KEXP_FIRST_ARG(ksave), &nx);
		if (r < 0)
			return -1;
		if (r >= (int) b->nslots)
			perm = r + 1;
		slot_reg[ksave->u.save.sv_slot_id] = r;
	}
	/* the aggregate table (the key columns of the output are not aggregates) */
	for (int i = 0; i < nattrs_out; i++)
	{
		const kern_aggregate_desc *d = &kagg->u.pagg.desc[i];
		int			sources[2] = {d->arg0_slot_id, d->filter_slot_id};
		int			regs_of[2] = {-1, -1};
		uint32_t	cls;

		prog->agg_action[i] = d->action;
		prog->agg_of_col[i] = -1;
		prog->key_of_col[i] = -1;
		if (d->action == KAGG_ACTION__VREF)
		{
			int		kidx = -1;

			for (int k = 0; k < nkeys; k++)
				if (key_slot[k] == d->arg0_slot_id)
					kidx = k;
			if (kidx < 0)
				return fail(b, "output column %d refers to a slot that is not a GROUP BY key", i + 1);
			prog->key_of_col[i] = (int8_t) kidx;
			continue;
		}
		if (d->action == KAGG_ACTION__VREF_NOKEY)
			return fail(b, "a column that depends on the GROUP BY key (VREF_NOKEY) is not supported yet");
		cls = join_agg_class(d->action);
		if (cls == 0)
			return fail(b, "aggregate action %u is not implemented by the Vulkan service", d->action);
		prog->agg_of_col[i] = (int8_t) nagg;
		for (int k = 0; k < 2; k++)
		{
			int		sl = sources[k];

			if (sl < 0)
				continue;
			if (sl >= (int) b->nslots)
				return fail(b, "aggregate slot %d is out of range", sl);
			if (slot_reg[sl] >= 0)
				regs_of[k] = slot_reg[sl];
			else if ((regs_of[k] = join_use_slot(b, (uint32_t) sl)) < 0)
				return -1;
		}
		if ((cls == 2 || cls >= 3) && regs_of[0] < 0)
			return fail(b, "aggregate in column %d has no argument", i + 1);
		agg_tab[4 * nagg + 0] = cls;
		agg_tab[4 * nagg + 1] = regs_of[0] < 0 ? 0xffffffffU : (uint32_t) regs_of[0];
		agg_tab[4 * nagg + 2] = regs_of[1] < 0 ? 0xffffffffU : (uint32_t) regs_of[1];
		agg_tab[4 * nagg + 3] = 0;
		nagg++;
	}
	prog->agg_mode = 1;
	prog->nagg = nagg;
	prog->nkeys = nkeys;
	*p_nagg = nagg;
	*p_nkeys = nkeys;
	return 0;
}

/*
 * Join program layout (words): header (PROG_HDR_WORDS), the outer column table, the column table
 * of each inner relation, the depth table (JDT_WORDS per inner relation), the instructions, the
 * registers of the computed values. Instructions: scan quals [0, nq), then for each depth its hash
 * value and its join quals, then the projection. Header: [0] outer column table, [2] instructions,
 * [3] outer columns, [5] end of the scan quals, [6] end of the program, [7] scan qual register,
 * [8] computed values, [9] their registers, [14] number of inner relations, [15] depth table,
 * [16] first projection instruction, [17] words of the outer-join map (RIGHT/FULL joins).
 */
#define JDT_WORDS			14

int
scan_prog_build_join(const kern_session_info *session, const kern_data_store *kds_src,
					 const scan_join_rel *rels, int nrels,
					 scan_prog *prog, char *err, size_t errlen)
{
	builder		b;
	const kern_expression *kquals, *kproj = NULL;
	const kern_varslot_desc *vs = SESSION_KVARS_SLOT_DESC(session);
	const kern_data_store *kds_dst = SESSION_KDS_DST_HEAD(session);
	int			qreg = -1;
	uint32_t	nq, proj_begin;
	int			next, ncols = 0, incols[SCAN_JOIN_MAXDEPTH + 1] = {0}, inc_total = 0;
	uint32_t	dsec[SCAN_JOIN_MAXDEPTH + 1][5];	/* hash begin, hash end, jq end, hash reg, jq reg */
	uint32_t	nattrs = 0, ncomp = 0, nwords;
	uint32_t   *comp_regs = NULL, *w;
	int			max_src = 0;
	int			agg_mode, nagg = 0, nkeys = 0;
	uint32_t	agg_tab[4 * SCAN_AGG_MAX];
	uint32_t	key_reg_tab[SCAN_GROUP_KEYS];

	memset(prog, 0, sizeof(*prog));
	memset(&b, 0, sizeof(b));
	err[0] = '\0';
	b.session = session;
	b.kds_src = kds_src;
	b.err = err;
	b.errlen = errlen;
	b.nslots = session->kcxt_kvars_nslots;
	b.no_strings = 1;

	agg_mode = ((session->xpu_task_flags & DEVTASK__MASK) == DEVTASK__PREAGG);
	if (!agg_mode && (session->xpu_task_flags & DEVTASK__MASK) != DEVTASK__JOIN)
	{
		fail(&b, "not a GpuJoin session");
		goto error;
	}
	if ((session->xpucode_gpusort_keydesc && !agg_mode) || (agg_mode != (session->xpucode_groupby_actions != 0)))
	{
		fail(&b, "sort on top of a join, or a join session with aggregate actions, is not implemented by the Vulkan service");
		goto error;
	}
	if (agg_mode)
	{
		for (int i = 0; i < nrels; i++)
			if (rels[i].right_outer)
			{
				fail(&b, "aggregation over a RIGHT or FULL join is not implemented by the Vulkan service");
				goto error;
			}
	}
	if (nrels < 1 || nrels > SCAN_JOIN_MAXDEPTH || session->kcxt_kvecs_ndims != nrels + 1)
	{
		fail(&b, "joins of %d inner relations are not implemented by the Vulkan service (up to %d)",
			 (int) session->kcxt_kvecs_ndims - 1, SCAN_JOIN_MAXDEPTH);
		goto error;
	}
	if (b.nslots > SCAN_MAX_REGS || !kds_dst || (!agg_mode && !SESSION_KEXP_PROJECTION(session)))
	{
		fail(&b, "unsupported join session (slots %u)", b.nslots);
		goto error;
	}
	/* the columns the kernel variables come from: depth 0 is the outer relation */
	for (int depth = 0; depth <= nrels; depth++)
	{
		const kern_expression *kload = SESSION_KEXP_LOAD_VARS(session, depth);
		const kern_data_store *kds = (depth == 0 ? kds_src : rels[depth - 1].kds);

		if (!kload)
			continue;
		for (int i = 0; i < kload->u.load.nitems; i++)
		{
			int		resno = kload->u.load.desc[i].vl_resno;
			int		slot = kload->u.load.desc[i].vl_slot_id;

			if (slot < 0 || slot >= (int) b.nslots)
				continue;
			if (depth == 0)
			{
				if (resno < 0)
					b.slot_sysattr[slot] = resno;
				else if (resno > 0 && resno <= (int) kds->ncols)
					b.slot_resno[slot] = resno;
			}
			else if (resno > 0 && resno <= (int) kds->ncols)
			{
				b.slot_iresno[slot] = resno;
				b.slot_idepth[slot] = depth;
			}
		}
	}
	/* sections: scan quals, then hash value and join quals of each depth, then the projection */
	next = b.nslots;
	kquals = SESSION_KEXP_SCAN_QUALS(session);
	if (kquals)
	{
		qreg = expr_to_reg(&b, kquals, &next);
		if (qreg < 0)
			goto error;
	}
	nq = b.ninsn;
	for (int depth = 1; depth <= nrels; depth++)
	{
		const kern_expression *khash = SESSION_KEXP_HASH_VALUE(session, depth);
		const kern_expression *kjq = SESSION_KEXP_JOIN_QUALS(session, depth);
		int		hash_reg, jq_reg = -1;

		if (!khash)
		{
			fail(&b, "join %d has no hash value (only hash joins are implemented)", depth);
			goto error;
		}
		dsec[depth][0] = b.ninsn;
		next = b.nslots;
		hash_reg = emit_hash_value(&b, khash, &next);
		if (hash_reg < 0)
			goto error;
		dsec[depth][1] = b.ninsn;
		next = b.nslots;
		if (kjq)
		{
			jq_reg = expr_to_reg(&b, kjq, &next);
			if (jq_reg < 0)
				goto error;
		}
		dsec[depth][2] = b.ninsn;
		dsec[depth][3] = (uint32_t) hash_reg;
		dsec[depth][4] = (jq_reg < 0) ? 0xffffffffU : (uint32_t) jq_reg;
	}
	proj_begin = b.ninsn;

	if (agg_mode)
	{
		/* aggregation: key expressions, aggregate arguments and FILTERs instead of a projection */
		if (join_build_agg(&b, prog, agg_tab, key_reg_tab, &nagg, &nkeys) != 0)
			goto error;
		nattrs = 0;
		comp_regs = calloc(1, sizeof(uint32_t));
		if (!comp_regs)
		{
			fail(&b, "out of memory");
			goto error;
		}
	}
	else
	{
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
		if (kds_dst->ncols < nattrs || kproj->nr_args != nattrs)
		{
			fail(&b, "projection has %u columns but the result store has %u", nattrs, kds_dst->ncols);
			goto error;
		}
	}
	if (!agg_mode)
	{
		const kern_expression *ksave = KEXP_FIRST_ARG(kproj);
		int		perm = b.nslots;

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
			if (kin->opcode == FuncOpCode__VarExpr && kin->u.v.var_slot_id < b.nslots &&
				b.slot_resno[kin->u.v.var_slot_id] > 0)
			{
				pa->kind = SCAN_PROJ_SRC;
				pa->resno = b.slot_resno[kin->u.v.var_slot_id];
				if (pa->resno > max_src)
					max_src = pa->resno;
			}
			else if (kin->opcode == FuncOpCode__VarExpr && kin->u.v.var_slot_id < b.nslots &&
					 b.slot_iresno[kin->u.v.var_slot_id] > 0)
			{
				pa->kind = SCAN_PROJ_INNER;
				pa->resno = b.slot_iresno[kin->u.v.var_slot_id];
				pa->depth = b.slot_idepth[kin->u.v.var_slot_id];
				if (pa->resno > prog->max_inner_resno[pa->depth])
					prog->max_inner_resno[pa->depth] = pa->resno;
			}
			else if (kin->opcode == FuncOpCode__VarExpr && kin->u.v.var_slot_id < b.nslots &&
					 (b.slot_sysattr[kin->u.v.var_slot_id] == -1 ||
					  b.slot_sysattr[kin->u.v.var_slot_id] == -6))
			{
				pa->kind = SCAN_PROJ_SYS;
				pa->resno = b.slot_sysattr[kin->u.v.var_slot_id];
			}
			else
			{
				int		bytes, r, nx = perm;

				if (!reg_vtype(ksave->exptype, &bytes) || reg_vtype(ksave->exptype, &bytes) == SOP_VT_STR ||
					reg_vtype(ksave->exptype, &bytes) == SOP_VT_NUM)
				{
					fail(&b, "computed column of an unsupported type (opcode type %u)", ksave->exptype);
					goto error;
				}
				r = expr_to_reg(&b, kin, &nx);
				if (r < 0)
					goto error;
				if (r < (int) b.nslots)
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
	/* column tables: up to the highest column the program reads on each side */
	for (uint32_t s = 0; s < b.nslots; s++)
	{
		int		bytes;

		if (b.slot_used[s])
		{
			b.slot_vtype[s] = reg_vtype(vs[s].vs_type_code, &bytes);
			if (b.slot_vtype[s] == SOP_VT_STR || b.slot_vtype[s] == SOP_VT_NUM)		/* join programs read the NULL-ness of strings only */
				b.slot_vtype[s] = SOP_VT_NULLONLY;
			else if (b.slot_vtype[s] == SOP_VT_INT && b.slot_nullonly[s] && !b.slot_valread[s])
				b.slot_vtype[s] = SOP_VT_NULLONLY;
			else if (!b.slot_vtype[s] && b.slot_nullonly[s])
				b.slot_vtype[s] = SOP_VT_NULLONLY;
			else if (!b.slot_vtype[s] || kds_src->colmeta[b.slot_resno[s] - 1].attlen != bytes ||
					 !kds_src->colmeta[b.slot_resno[s] - 1].attbyval)
			{
				fail(&b, "outer column %d does not have the type the expression expects", b.slot_resno[s]);
				goto error;
			}
			if (b.slot_resno[s] > ncols)
				ncols = b.slot_resno[s];
		}
		if (b.slot_iused[s])
		{
			const kern_data_store *kds_in = rels[b.slot_idepth[s] - 1].kds;

			b.slot_ivtype[s] = reg_vtype(vs[s].vs_type_code, &bytes);
			if (b.slot_ivtype[s] == SOP_VT_STR || b.slot_ivtype[s] == SOP_VT_NUM)
				b.slot_ivtype[s] = SOP_VT_NULLONLY;
			else if (!b.slot_ivtype[s] && b.slot_nullonly[s])
				b.slot_ivtype[s] = SOP_VT_NULLONLY;
			else if (!b.slot_ivtype[s] || kds_in->colmeta[b.slot_iresno[s] - 1].attlen != bytes ||
					 !kds_in->colmeta[b.slot_iresno[s] - 1].attbyval)
			{
				fail(&b, "inner column %d does not have the type the expression expects", b.slot_iresno[s]);
				goto error;
			}
			if (b.slot_iresno[s] > incols[b.slot_idepth[s]])
				incols[b.slot_idepth[s]] = b.slot_iresno[s];
		}
	}
	for (int depth = 1; depth <= nrels; depth++)
		inc_total += incols[depth];
	nwords = PROG_HDR_WORDS + ncols + inc_total + JDT_WORDS * nrels + 6 * b.ninsn + ncomp + 4 * nagg + nkeys;
	if (nwords > 16384)
	{
		fail(&b, "the join program is too large");
		goto error;
	}
	w = calloc(nwords, sizeof(uint32_t));
	if (!w)
	{
		fail(&b, "out of memory");
		goto error;
	}
	{
		uint32_t	dtab = PROG_HDR_WORDS + ncols + inc_total;
		uint32_t	insnoff = dtab + JDT_WORDS * nrels;
		uint32_t	ibase = PROG_HDR_WORDS + ncols;

		w[0] = PROG_HDR_WORDS;						/* outer column table */
		w[1] = insnoff;
		w[2] = insnoff;								/* instructions */
		w[3] = ncols;
		w[5] = nq;
		w[6] = b.ninsn;
		w[7] = (qreg < 0) ? 0xffffffffU : (uint32_t) qreg;
		w[8] = ncomp;
		w[9] = insnoff + 6 * b.ninsn;
		w[14] = nrels;
		w[15] = dtab;
		w[16] = proj_begin;
		for (int depth = 1; depth <= nrels; depth++)
		{
			uint32_t   *dt = &w[dtab + JDT_WORDS * (depth - 1)];

			dt[0] = dsec[depth][0];
			dt[1] = dsec[depth][1];
			dt[2] = dsec[depth][2];
			dt[3] = dsec[depth][3];
			dt[4] = dsec[depth][4];
			dt[5] = ibase;							/* this relation's column table */
			dt[6] = incols[depth];
			dt[7] = rels[depth - 1].kds_end;
			dt[8] = rels[depth - 1].hslot_word;
			dt[9] = rels[depth - 1].nslots;
			dt[10] = rels[depth - 1].left_outer ? 1 : 0;
			dt[11] = rels[depth - 1].semi ? 1 : rels[depth - 1].anti ? 2 : 0;
			dt[12] = 0xffffffffU;
			dt[13] = 0;
			prog->oj_off[depth] = 0xffffffffU;
			if (rels[depth - 1].right_outer)
			{
				dt[12] = prog->oj_words;
				dt[13] = rels[depth - 1].kds->nitems;
				prog->oj_off[depth] = prog->oj_words;
				prog->oj_nitems[depth] = rels[depth - 1].kds->nitems;
				prog->oj_words += rels[depth - 1].kds->nitems;
			}
			ibase += incols[depth];
		}
		for (int side = 0; side <= nrels; side++)
		{
			const kern_data_store *kds = (side == 0 ? kds_src : rels[side - 1].kds);
			int		n = (side == 0 ? ncols : incols[side]);
			uint32_t base = (side == 0 ? PROG_HDR_WORDS : w[dtab + JDT_WORDS * (side - 1) + 5]);

			for (int c = 0; c < n; c++)
			{
				const kern_colmeta *cm = &kds->colmeta[c];
				uint32_t	alen = (cm->attlen == -1 ? 0xff : cm->attlen == -2 ? 0xfe : (uint32_t) cm->attlen);
				uint32_t	slot = 0xff, vt = 0;

				for (uint32_t s = 0; s < b.nslots; s++)
				{
					if (side == 0 && b.slot_used[s] && b.slot_resno[s] == c + 1)
					{
						slot = s;
						vt = b.slot_vtype[s];
					}
					if (side > 0 && b.slot_iused[s] && b.slot_idepth[s] == side && b.slot_iresno[s] == c + 1)
					{
						slot = s;
						vt = b.slot_ivtype[s];
					}
				}
				w[base + c] = alen | ((uint32_t) cm->attalign << 8) | (vt << 12) | (slot << 16);
			}
		}
		w[17] = prog->oj_words;
		memcpy(w + insnoff, b.insn, sizeof(uint32_t) * 6 * b.ninsn);
		memcpy(w + insnoff + 6 * b.ninsn, comp_regs, sizeof(uint32_t) * ncomp);
		if (agg_mode)
		{
			/* the aggregate table and the key registers follow the computed values (none here) */
			uint32_t	aggoff = insnoff + 6 * b.ninsn + ncomp;

			w[10] = nagg;
			w[11] = aggoff;
			w[12] = nkeys;
			w[13] = aggoff + 4 * nagg;
			memcpy(w + aggoff, agg_tab, sizeof(uint32_t) * 4 * nagg);
			memcpy(w + aggoff + 4 * nagg, key_reg_tab, sizeof(uint32_t) * nkeys);
		}
	}
	prog->words = w;
	prog->nwords = nwords;
	prog->ncomp = ncomp;
	prog->nregs = (b.maxreg > (int) b.nslots ? b.maxreg : (int) b.nslots);
	prog->max_src_resno = max_src;
	prog->ndepth = nrels;
	prog->join_mode = 1;
	free(b.insn);
	free(comp_regs);
	return 0;

error:
	free(b.insn);
	free(comp_regs);
	scan_prog_free(prog);
	if (err[0] == '\0')
		snprintf(err, errlen, "cannot build the join program");
	return -1;
}
