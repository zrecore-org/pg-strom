/*
 * scan_caps.c
 *
 * What the scan VM can execute: classification of PostgreSQL function opcodes by name.
 * Shared by the planner side (codegen.c refuses to push down anything else) and by the
 * service's program builder (scan_prog.c), so the two cannot disagree.
 */
#include <string.h>
#include "xpu_common.h"
#include "scan_caps.h"

/* ----------------------------------------------------------------
 * opcode names (to classify the many intNNxx operators by name)
 * ----------------------------------------------------------------
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			[FuncOpCode__##NAME] = #NAME,
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	[FuncOpCode__##NAME] = #NAME,
static const char *const func_names[FuncOpCode__LoadVars] = {
#include "xpu_opcodes.h"
};
#undef FUNC_OPCODE
#undef DEVONLY_FUNC_OPCODE

const char *
scan_vm_opcode_name(uint32_t opcode)
{
	switch (opcode)
	{
		case FuncOpCode__ConstExpr:		return "ConstExpr";
		case FuncOpCode__ParamExpr:		return "ParamExpr";
		case FuncOpCode__VarExpr:		return "VarExpr";
		case FuncOpCode__BoolExpr_And:	return "BoolExpr_And";
		case FuncOpCode__BoolExpr_Or:	return "BoolExpr_Or";
		case FuncOpCode__BoolExpr_Not:	return "BoolExpr_Not";
		default:
			if (opcode < FuncOpCode__LoadVars && func_names[opcode])
				return func_names[opcode];
	}
	return "?";
}


int		scan_tier_full_only = 0;

/*
 * The feature group an instruction needs (SCAN_FEAT_*; 0: compiled into every kernel). Mirrors the guards of
 * scan_common.slang vm_step(). Instructions 130 up are the geometry group (feat_check.c verifies that every GEOM* opcode
 * of scan_vm_ops.inc is in it); NULLMRG sits among them but is always compiled in. A geometry instruction that is not
 * classified here would run on a kernel without geometry and fail as a malformed program.
 */
uint32_t
scan_vm_op_feat(uint32_t op)
{
	if (op >= SOP_NUMCONST && op <= SOP_NUMRND)
		return SCAN_FEAT_NUM;
	if (op >= SOP_JSONGET && op <= SOP_JSONCONT)
		return SCAN_FEAT_JSON;
	if (op >= SOP_CMPD && op <= SOP_D2F)
		return SCAN_FEAT_F8;
	if (op >= SOP_GEOMBOX && op < 200 && op != SOP_NULLMRG)
		return SCAN_FEAT_GEOM;
	return 0;
}

static int
int_width_of_digit(char c)
{
	switch (c)
	{
		case '1': return 8;
		case '2': return 16;
		case '4': return 32;
		case '8': return 64;
	}
	return 0;
}

static int
cmp_kind_of(const char *s)
{
	if (!strcmp(s, "eq")) return SOP_CMP_EQ;
	if (!strcmp(s, "ne")) return SOP_CMP_NE;
	if (!strcmp(s, "lt")) return SOP_CMP_LT;
	if (!strcmp(s, "le")) return SOP_CMP_LE;
	if (!strcmp(s, "gt")) return SOP_CMP_GT;
	if (!strcmp(s, "ge")) return SOP_CMP_GE;
	return -1;
}

static int
arith_kind_of(const char *s)
{
	if (!strcmp(s, "pl"))  return SOP_ARITH_PL;
	if (!strcmp(s, "mi"))  return SOP_ARITH_MI;
	if (!strcmp(s, "mul")) return SOP_ARITH_MUL;
	if (!strcmp(s, "div")) return SOP_ARITH_DIV;
	if (!strcmp(s, "mod")) return SOP_ARITH_MOD;
	return -1;
}

/* ----------------------------------------------------------------
 * A6: temporal types and strings
 * ----------------------------------------------------------------
 */

/*
 * date_eq, time_lt, timestamp_ge, timestamptz_ne: plain integer comparison of the stored values
 * (days / microseconds; infinity is INT_MAX/INT_MIN, so the order is the integers' order).
 * date_lt_timestamp and timestamp_lt_date compare after converting the date to a timestamp (width 1:
 * the left side is the date, 2: the right side), which is exact; the comparisons that involve a
 * timestamptz and a date or a timestamp depend on the session time zone and are not implemented.
 * date_pli, date_mii, integer_pl_date, date_mi: date arithmetic. date_to_timestamp, timestamp_date.
 */
static scan_func_class
classify_temporal(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	static const char *const tnames[] = {"timestamptz", "timestamp", "date", "time"};
	int			k;

	if (!strcmp(name, "date_pli"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_PLI; return fc; }
	if (!strcmp(name, "date_mii"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_MII; return fc; }
	if (!strcmp(name, "date_mi"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_MI; return fc; }
	if (!strcmp(name, "integer_pl_date"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_PLI; fc.width = 1; return fc; }	/* arguments swapped */
	if (!strcmp(name, "date_to_timestamp"))
		{ fc.cls = SOP_D2TS; fc.width = 1; return fc; }		/* the cast: out of range is an error */
	if (!strcmp(name, "timestamp_date"))
		{ fc.cls = SOP_TS2D; return fc; }
	for (size_t i = 0; i < sizeof(tnames) / sizeof(tnames[0]); i++)
	{
		size_t		n = strlen(tnames[i]);
		const char *rest;

		if (strncmp(name, tnames[i], n) != 0 || name[n] != '_')
			continue;
		rest = name + n + 1;
		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPI;
			fc.kind = k;
			return fc;
		}
		/* date <op> timestamp, timestamp <op> date */
		if (!strcmp(tnames[i], "date") || !strcmp(tnames[i], "timestamp"))
		{
			int			left_is_date = !strcmp(tnames[i], "date");
			const char *other = (left_is_date ? "timestamp" : "date");
			char		op[3];

			if (strlen(rest) > 3 && rest[2] == '_' && strcmp(rest + 3, other) == 0)
			{
				memcpy(op, rest, 2);
				op[2] = '\0';
				if ((k = cmp_kind_of(op)) >= 0)
				{
					fc.cls = SOP_CMPI;
					fc.kind = k;
					fc.width = (left_is_date ? 1 : 2);		/* which side is the date */
					return fc;
				}
			}
		}
		break;
	}
	return fc;
}

/* text_eq, bpcharne, text_lt (C collation only: the planner checks), textlen, textlike ... */
static scan_func_class
classify_string(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	int			k;

	if (!strncmp(name, "bpchar", 6))
	{
		const char *rest = name + 6;

		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_STRCMP; fc.kind = k; fc.width = 1;
		}
		else if (!strcmp(rest, "like") || !strcmp(rest, "nlike"))
		{
			/* PostgreSQL's bpcharlike does NOT strip the value's trailing blanks: they are matched as characters */
			fc.cls = SOP_STRLIKE; fc.kind = (rest[0] == 'n'); fc.width = 0;
		}
		return fc;
	}
	if (!strncmp(name, "text_", 5) && (k = cmp_kind_of(name + 5)) >= 0)
	{
		fc.cls = SOP_STRCMP; fc.kind = k;
	}
	else if (!strcmp(name, "like") || !strcmp(name, "textlike"))
	{
		fc.cls = SOP_STRLIKE; fc.kind = 0;
	}
	else if (!strcmp(name, "notlike") || !strcmp(name, "textnlike"))
	{
		fc.cls = SOP_STRLIKE; fc.kind = 1;
	}
	else if (!strcmp(name, "textlen"))
	{
		fc.cls = SOP_STRLEN;
		fc.width = 1;				/* characters of a UTF-8 string (the planner allows it for UTF8 only) */
	}
	return fc;
}

/*
 * The shape of a LIKE pattern the VM can match bytewise: a literal, optionally with % before and
 * after it (the backslash escapes a character). The unescaped literal is stored in lit (at least
 * plen bytes). Returns SOP_LIKE_*, or -1 if the pattern has '_' or an inner '%'.
 */
int
scan_vm_like_shape(const unsigned char *pat, int plen, unsigned char *lit, int *p_litlen)
{
	int			i = 0, n = 0, lead = 0, trail = 0;

	while (i < plen && pat[i] == '%')
	{
		lead = 1;
		i++;
	}
	while (i < plen)
	{
		unsigned char c = pat[i];

		if (c == '\\')
		{
			if (++i >= plen)
				return -1;			/* PostgreSQL raises an error */
			lit[n++] = pat[i++];
		}
		else if (c == '%')
			break;
		else if (c == '_')
			return -1;
		else
		{
			lit[n++] = c;
			i++;
		}
	}
	while (i < plen && pat[i] == '%')
	{
		trail = 1;
		i++;
	}
	if (i < plen)
		return -1;					/* a % between literals */
	*p_litlen = n;
	if (!lead)
		return trail ? SOP_LIKE_PREFIX : SOP_LIKE_EXACT;
	return trail ? SOP_LIKE_CONTAINS : SOP_LIKE_SUFFIX;
}

/* ----------------------------------------------------------------
 * A6: numeric. A numeric register is { unscaled int64, scale } (see scan_vm_ops.inc).
 * numeric_eq .. numeric_ge, numeric_add/sub/mul, numeric_uminus/uplus/abs (and the abs() alias),
 * intN_to_numeric, numeric_to_intN, numeric_div/mod/div_trunc, numeric_round/trunc, numeric_ceil/floor/sign, the
 * numeric(p,s) typmod coercion. Float casts etc. stay on the CPU.
 * ----------------------------------------------------------------
 */
static scan_func_class
classify_numeric(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	int			k;

	if (!strncmp(name, "numeric_", 8))
	{
		const char *rest = name + 8;

		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_NUMCMP; fc.kind = k;
		}
		else if (!strcmp(rest, "add")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_ARITH_PL; }
		else if (!strcmp(rest, "sub")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_ARITH_MI; }
		else if (!strcmp(rest, "mul")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_ARITH_MUL; }
		else if (!strcmp(rest, "uminus")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_NUMOP_NEG; }
		else if (!strcmp(rest, "abs")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_NUMOP_ABS; }
		else if (!strcmp(rest, "uplus")) { fc.cls = SOP_MOV; }
		else if (!strcmp(rest, "div")) { fc.cls = SOP_NUMDIV; fc.kind = SOP_NUMDIV_DIV; }
		else if (!strcmp(rest, "mod")) { fc.cls = SOP_NUMDIV; fc.kind = SOP_NUMDIV_MOD; }
		else if (!strcmp(rest, "div_trunc")) { fc.cls = SOP_NUMDIV; fc.kind = SOP_NUMDIV_TRUNC; }
		else if (!strcmp(rest, "round")) { fc.cls = SOP_NUMRND; fc.kind = SOP_NUMRND_ROUND; }
		else if (!strcmp(rest, "trunc")) { fc.cls = SOP_NUMRND; fc.kind = SOP_NUMRND_TRUNC; }
		else if (!strcmp(rest, "typmod")) { fc.cls = SOP_NUMRND; fc.kind = SOP_NUMRND_TYPMOD; }
		else if (!strcmp(rest, "ceil")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_NUMOP_CEIL; }
		else if (!strcmp(rest, "floor")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_NUMOP_FLOOR; }
		else if (!strcmp(rest, "sign")) { fc.cls = SOP_NUMARITH; fc.kind = SOP_NUMOP_SIGN; }
		else if (!strncmp(rest, "to_int", 6) && (k = int_width_of_digit(rest[6])) != 0 && rest[7] == '\0')
		{
			fc.cls = SOP_N2I; fc.width = k;
		}
	}
	else if (!strncmp(name, "int", 3) && int_width_of_digit(name[3]) != 0 && !strcmp(name + 4, "_to_numeric"))
	{
		fc.cls = SOP_I2N;
	}
	return fc;
}

/*
 * A8: jsonb. kind 0: the key form (object_field), 1: the array element form. cls SOP_JSONGET is "->", SOP_JSONTXT is
 * "->>" (a get and the text of the result), SOP_JSONNUM / SOP_JSONINT the "(->>)::numeric / intN" shortcut of the
 * planner (width = bits). Not on the VM: the float casts, int1, #>, path operators, jsonb_eq ...
 */
static scan_func_class
classify_jsonb(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	const char *rest;
	int			kind;

	if (strncmp(name, "jsonb_", 6) != 0)
		return fc;
	rest = name + 6;
	if (!strncmp(rest, "object_field", 12))
	{
		kind = 0;
		rest += 12;
	}
	else if (!strncmp(rest, "array_element", 13))
	{
		kind = 1;
		rest += 13;
	}
	else
	{
		if (!strcmp(rest, "exists"))
			fc.cls = SOP_JSONEXISTS;
		else if (!strcmp(rest, "typeof"))
			fc.cls = SOP_JSONTYPE;
		else if (!strcmp(rest, "contains"))
			fc.cls = SOP_JSONCONT;
		return fc;
	}
	fc.kind = kind;
	if (*rest == '\0')
		fc.cls = SOP_JSONGET;
	else if (!strcmp(rest, "_text"))
		fc.cls = SOP_JSONTXT;
	else if (!strcmp(rest, "_as_numeric"))
		fc.cls = SOP_JSONNUM;
	else if (!strcmp(rest, "_as_int2"))
	{
		fc.cls = SOP_JSONINT;
		fc.width = 16;
	}
	else if (!strcmp(rest, "_as_int4"))
	{
		fc.cls = SOP_JSONINT;
		fc.width = 32;
	}
	else if (!strcmp(rest, "_as_int8"))
	{
		fc.cls = SOP_JSONINT;
		fc.width = 64;
	}
	return fc;
}

/* value * 10^d without overflow of int64 (|value| <= INT64_MAX); 0 if it overflows */
int
scan_vm_num_up(int64_t *v, uint32_t d)
{
	static const int64_t p10[19] = {
		1LL, 10LL, 100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 100000000LL,
		1000000000LL, 10000000000LL, 100000000000LL, 1000000000000LL, 10000000000000LL,
		100000000000000LL, 1000000000000000LL, 10000000000000000LL, 100000000000000000LL,
		1000000000000000000LL};
	int64_t		lim;

	if (*v == 0)
		return 1;
	if (d > 18)
		return 0;
	lim = INT64_MAX / p10[d];
	if (*v > lim || *v < -lim)
		return 0;
	*v *= p10[d];
	return 1;
}

/* -1, 0, +1: exact comparison of two { unscaled, scale } values */
int
scan_vm_num_cmp(int64_t va, uint32_t sa, int64_t vb, uint32_t sb)
{
	if (sa < sb)
	{
		if (!scan_vm_num_up(&va, sb - sa))
			return va > 0 ? 1 : -1;
	}
	else if (sb < sa)
	{
		if (!scan_vm_num_up(&vb, sa - sb))
			return vb > 0 ? -1 : 1;
	}
	return va < vb ? -1 : (va > vb ? 1 : 0);
}

/* ----------------------------------------------------------------
 * A6 (part 2): numeric division, remainder, round/trunc, typmod coercion, ceil/floor/sign.
 * Every helper returns 1 with { *r, *rsc }, or 0 when the row must be evaluated by the backend (the result
 * does not fit { int64, scale <= 255 }, division by zero, ...). They reproduce numeric.c exactly; the GPU
 * kernel (scan_common.slang) has the same code in 32/64-bit arithmetic.
 * ----------------------------------------------------------------
 */
static uint64_t
num_p10u(uint32_t d)			/* 10^d, d <= 19 */
{
	uint64_t	p = 1;

	while (d-- > 0)
		p *= 10;
	return p;
}

static uint32_t
num_ndig(uint64_t m)			/* number of decimal digits of m > 0 */
{
	uint32_t	n = 1;
	uint64_t	p = 10;

	while (n < 19 && m >= p)
	{
		n++;
		p *= 10;
	}
	return n;
}

/*
 * The weight (in base 10000 digits) and the first digit of the normalized representation of
 * { v, sc }, as select_div_scale() of numeric.c sees them (zero: weight 0, first digit 0).
 */
static void
num_weight_firstdigit(int64_t v, uint32_t sc, int *w, uint32_t *fd)
{
	uint64_t	m;
	int			e10, k;

	if (v == 0)
	{
		*w = 0;
		*fd = 0;
		return;
	}
	m = (uint64_t)(v < 0 ? -v : v);
	e10 = (int) num_ndig(m) - 1 - (int) sc;
	*w = (e10 >= 0) ? e10 / 4 : -((-e10 + 3) / 4);
	k = (int) sc + 4 * *w;				/* -3 <= k <= 18 */
	*fd = (k >= 0) ? (uint32_t)(m / num_p10u((uint32_t) k)) : (uint32_t)(m * num_p10u((uint32_t) -k));
}

/* q = round_half_up(a * 10^E / b) for a >= 0, b > 0; 0 if q > INT64_MAX */
static int
num_divround(uint64_t a, uint64_t b, uint32_t E, uint64_t *q)
{
	uint64_t	Q = a / b, rem = a % b;

	while (E > 0)
	{
		uint32_t	k = E > 18 ? 18 : E;
		uint64_t	p = num_p10u(k), qc;
		unsigned __int128 t = (unsigned __int128) rem * p;

		E -= k;
		qc = (uint64_t)(t / b);
		rem = (uint64_t)(t % b);
		if (Q > ((uint64_t) INT64_MAX - qc) / p)
			return 0;
		Q = Q * p + qc;
	}
	if (rem >= b - rem)				/* 2 * rem >= b */
	{
		if (Q == (uint64_t) INT64_MAX)
			return 0;
		Q++;
	}
	*q = Q;
	return 1;
}

/* kind: 0 numeric_div, 1 numeric_mod, 2 numeric_div_trunc (div()) */
int
scan_vm_num_div(int kind, int64_t a, uint32_t sa, int64_t b, uint32_t sb, int64_t *r, uint32_t *rsc)
{
	if (b == 0)
		return 0;							/* the backend raises "division by zero" */
	if (kind == 0)
	{
		int			w1, w2, qw, rs;
		uint32_t	fd1, fd2, E;
		uint64_t	q;

		num_weight_firstdigit(a, sa, &w1, &fd1);
		num_weight_firstdigit(b, sb, &w2, &fd2);
		qw = w1 - w2;
		if (fd1 <= fd2)
			qw--;
		rs = 16 - qw * 4;					/* NUMERIC_MIN_SIG_DIGITS - qweight * DEC_DIGITS */
		if (rs < (int) sa) rs = (int) sa;
		if (rs < (int) sb) rs = (int) sb;
		if (rs < 0) rs = 0;
		if (rs > 1000) rs = 1000;
		if (rs > 255)
			return 0;
		E = sb + (uint32_t) rs - sa;		/* >= 0 because rs >= sa */
		if (!num_divround((uint64_t)(a < 0 ? -a : a), (uint64_t)(b < 0 ? -b : b), E, &q))
			return 0;
		*r = ((a < 0) != (b < 0)) ? -(int64_t) q : (int64_t) q;
		*rsc = (uint32_t) rs;
		return 1;
	}
	else
	{
		uint32_t	sc = sa > sb ? sa : sb;

		if (!scan_vm_num_up(&a, sc - sa) || !scan_vm_num_up(&b, sc - sb) || b == 0)
			return 0;
		if (kind == 1)
		{
			*r = a % b;
			*rsc = sc;
		}
		else
		{
			*r = a / b;
			*rsc = 0;
		}
		return 1;
	}
}

/* v / 10^d rounded half away from zero (round != 0) or truncated, for v >= 0 */
static uint64_t
num_down(uint64_t m, uint32_t d, int round)
{
	if (d == 0)
		return m;
	if (d >= 20)
		return 0;
	if (d == 19)
		return (round && m >= 5000000000000000000ULL) ? 1 : 0;
	{
		uint64_t	p = num_p10u(d), q = m / p;

		if (round && (m - q * p) >= p - (m - q * p))
			q++;
		return q;
	}
}

/*
 * kind 0: round(v, n), 1: trunc(v, n), 2: numeric(p, s) typmod coercion (n is the typmod; only s >= 0).
 * The scale grows to n (trailing zeros) or shrinks with rounding; a negative n rounds to tens, hundreds...
 * and the result has scale 0.
 */
int
scan_vm_num_round(int kind, int64_t v, uint32_t sc, int64_t n, int64_t *r, uint32_t *rsc)
{
	int64_t		prec = 0;
	uint64_t	m = (uint64_t)(v < 0 ? -v : v), q;
	int			neg = v < 0;

	if (kind == 2)
	{
		int64_t		t = n;
		int			s;

		if (t < 4)
		{
			*r = v;						/* not a valid typmod: the value is returned as is */
			*rsc = sc;
			return 1;
		}
		t -= 4;
		prec = (t >> 16) & 0xffff;
		s = (int)(((t & 0x7ff) ^ 0x400) - 0x400);
		if (s < 0 || prec < 1 || prec > 1000)
			return 0;
		n = s;
	}
	if (n > 255)
		return 0;
	if (n < -64)
		return 0;						/* (a huge negative scale: leave it to the backend) */
	if (n >= (int64_t) sc)
	{
		int64_t		x = v;

		if (!scan_vm_num_up(&x, (uint32_t)(n - (int64_t) sc)))
			return 0;
		*r = x;
		*rsc = (uint32_t) n;
		m = (uint64_t)(x < 0 ? -x : x);
	}
	else
	{
		uint32_t	d = (uint32_t)((int64_t) sc - n);

		q = num_down(m, d, kind != 1);
		if (n < 0)
		{
			int64_t		x = (int64_t) q;

			if (!scan_vm_num_up(&x, (uint32_t) -n))
				return 0;
			q = (uint64_t) x;
			*rsc = 0;
		}
		else
			*rsc = (uint32_t) n;
		*r = neg ? -(int64_t) q : (int64_t) q;
		m = q;
	}
	if (kind == 2 && prec <= 18 && m >= num_p10u((uint32_t) prec))
		return 0;						/* "numeric field overflow": the backend raises it */
	return 1;
}

/* kind: SOP_NUMOP_CEIL, SOP_NUMOP_FLOOR, SOP_NUMOP_SIGN; the result has scale 0 */
int
scan_vm_num_unary(int kind, int64_t v, uint32_t sc, int64_t *r, uint32_t *rsc)
{
	*rsc = 0;
	if (kind == SOP_NUMOP_SIGN)
	{
		*r = (v > 0) - (v < 0);
		return 1;
	}
	{
		uint64_t	m = (uint64_t)(v < 0 ? -v : v), q = num_down(m, sc, 0);
		int			inexact = (sc >= 20) ? (m != 0) : (m != q * num_p10u(sc));	/* a fraction was cut off */

		if (v >= 0)
			*r = (int64_t) q + ((kind == SOP_NUMOP_CEIL && inexact) ? 1 : 0);
		else
			*r = -(int64_t) q - ((kind == SOP_NUMOP_FLOOR && inexact) ? 1 : 0);
		return 1;
	}
}

/*
 * Decode the payload of a numeric varlena (plen bytes, after the varlena header; see numeric.c) into
 * { *v, *sc }. Returns 0 for NaN, +-Infinity and for a value that does not fit (|v| <= INT64_MAX,
 * scale <= 255): such rows are evaluated by the backend.
 */
int
scan_vm_numeric_payload_decode(const unsigned char *p, uint32_t plen, int64_t *v, uint32_t *sc)
{
	uint16_t	hdr;
	int			neg, weight;
	uint32_t	dscale, ndig, hl;
	int64_t		acc = 0;
	int			e;

	*v = 0;
	*sc = 0;
	if (plen < 2)
		return 0;
	memcpy(&hdr, p, 2);
	if ((hdr & 0xc000) == 0xc000)
		return 0;
	if (hdr & 0x8000)
	{
		neg = (hdr & 0x2000) != 0;
		dscale = (hdr >> 7) & 0x3f;
		weight = (int)(hdr & 0x3f) - ((hdr & 0x40) ? 64 : 0);
		hl = 2;
	}
	else
	{
		int16_t		w;

		if (plen < 4)
			return 0;
		neg = (hdr & 0x4000) != 0;
		dscale = hdr & 0x3fff;
		memcpy(&w, p + 2, 2);
		weight = w;
		hl = 4;
	}
	ndig = (plen - hl) / 2;
	if (dscale > 255 || ndig > 5)
		return 0;
	for (uint32_t i = 0; i < ndig; i++)
	{
		uint16_t	d;

		memcpy(&d, p + hl + 2 * i, 2);
		if (acc > (INT64_MAX - (int64_t) d) / 10000)
			return 0;
		acc = acc * 10000 + d;
	}
	e = 4 * (weight - (int) ndig + 1) + (int) dscale;
	if (acc != 0)
	{
		if (e >= 0)
		{
			if (!scan_vm_num_up(&acc, (uint32_t) e))
				return 0;
		}
		else
		{
			static const int64_t p10[19] = {
				1LL, 10LL, 100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 100000000LL,
				1000000000LL, 10000000000LL, 100000000000LL, 1000000000000LL, 10000000000000LL,
				100000000000000LL, 1000000000000000LL, 10000000000000000LL, 100000000000000000LL,
				1000000000000000000LL};

			if (-e > 18 || acc % p10[-e] != 0)
				return 0;
			acc /= p10[-e];
		}
	}
	*v = neg ? -acc : acc;
	*sc = dscale;
	return 1;
}

/* the same for a whole varlena datum (1-byte or 4-byte header; compressed and external ones fail) */
int
scan_vm_numeric_datum_decode(const unsigned char *d, int64_t *v, uint32_t *sc)
{
	if (d[0] & 1)
	{
		if (d[0] == 1)
			return 0;
		return scan_vm_numeric_payload_decode(d + 1, ((d[0] >> 1) & 0x7f) - 1, v, sc);
	}
	if ((d[0] & 3) == 2)
		return 0;
	{
		uint32_t	h;

		memcpy(&h, d, 4);
		return scan_vm_numeric_payload_decode(d + 4, ((h >> 2) & 0x3fffffffU) - 4, v, sc);
	}
}

/*
 * Form the numeric varlena (a 1-byte header, as heap_form_tuple would store it) of { v, sc } into buf
 * (at least SCAN_NUMERIC_MAXLEN bytes) exactly as numeric.c's make_result() does: digits in base 10000
 * aligned to the decimal point, leading and trailing zero digits stripped, dscale = sc. Returns its length.
 */
uint32_t
scan_vm_numeric_encode(int64_t v, uint32_t sc, unsigned char *buf)
{
	unsigned __int128 x;
	uint16_t	digs[8];			/* least significant first */
	int			nd = 0, f = (int)((sc + 3) / 4), lo = 0, weight;
	int			neg = (v < 0);
	uint32_t	hl, len;
	uint16_t	hdr;

	x = (unsigned __int128)(neg ? -(unsigned __int128)(int64_t) v : (unsigned __int128)(uint64_t) v);
	for (uint32_t i = 0; i < 4 * (uint32_t) f - sc; i++)
		x *= 10;
	while (x != 0)
	{
		digs[nd++] = (uint16_t)(x % 10000);
		x /= 10000;
	}
	/* digs[i] has the weight i - f */
	while (lo < nd && digs[lo] == 0)
		lo++;
	if (nd == 0)
	{
		weight = 0;
		neg = 0;
	}
	else
		weight = nd - 1 - f;
	if (sc <= 63 && weight <= 63 && weight >= -64)
	{
		hdr = 0x8000 | (neg ? 0x2000 : 0) | (uint16_t)(sc << 7) | (weight < 0 ? 0x40 : 0) | (uint16_t)(weight & 0x3f);
		hl = 2;
	}
	else
	{
		hdr = (neg ? 0x4000 : 0) | (uint16_t) sc;
		hl = 4;
	}
	len = 1 + hl + 2 * (uint32_t)(nd - lo);
	buf[0] = (unsigned char)((len << 1) | 1);
	memcpy(buf + 1, &hdr, 2);
	if (hl == 4)
	{
		int16_t		w = (int16_t) weight;

		memcpy(buf + 3, &w, 2);
	}
	for (int i = nd - 1, k = 0; i >= lo; i--, k++)
		memcpy(buf + 1 + hl + 2 * k, &digs[i], 2);
	return len;
}


/* ----------------------------------------------------------------
 * A8: jsonb. The binary form (utils/jsonb.h): a container is a header word (count | flags), the JEntry
 * words (keys then values for an object), then the data; an entry carries the type of the child and either
 * its length or, every 32nd entry, the end offset of the child (JB_OFFSET_STRIDE). A string is stored
 * as is, a numeric and a nested container are aligned to 4 bytes relative to the start of the data.
 * ----------------------------------------------------------------
 */
static uint32_t
jb_ld32(const unsigned char *b, uint32_t o)
{
	uint32_t	v;

	memcpy(&v, b + o, 4);
	return v;
}

int
scan_jb_open(const unsigned char *b, uint32_t cb, uint32_t clen, scan_jb_cont *c)
{
	uint32_t	h;

	if (clen < 4)
		return -1;
	h = jb_ld32(b, cb);
	c->count = h & 0x0fffffffU;
	c->isobj = (h & 0x20000000U) != 0;
	c->isarr = (h & 0x40000000U) != 0;
	c->isscalar = (h & 0x10000000U) != 0;
	if (c->isobj == c->isarr || c->count > clen / 4)
		return -1;
	c->nent = c->isobj ? 2 * c->count : c->count;
	if (4 + 4 * c->nent > clen)
		return -1;
	return 0;
}

int
scan_jb_child(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
			  uint32_t i, uint32_t *kind, uint32_t *voff, uint32_t *vlen)
{
	uint32_t	e = jb_ld32(b, cb + 4 + 4 * i);
	uint32_t	st = 0, f = e & 0x0fffffffU, len, pad = 0, dbase = 4 + 4 * c->nent;

	for (uint32_t j = i; j > 0; j--)
	{
		uint32_t	p = jb_ld32(b, cb + 4 + 4 * (j - 1));

		st += p & 0x0fffffffU;
		if (p & 0x80000000U)
			break;
	}
	if (e & 0x80000000U)
	{
		if (f < st)
			return -1;
		len = f - st;
	}
	else
		len = f;
	*kind = (e >> 28) & 7;
	if (*kind > SCAN_JB_CONT)
		return -1;
	if (*kind == SCAN_JB_NUM || *kind == SCAN_JB_CONT)
		pad = ((st + 3) & ~3U) - st;
	if (len < pad || (uint64_t) dbase + st + len > clen)
		return -1;
	if (*kind >= SCAN_JB_FALSE && *kind <= SCAN_JB_NULL)
	{
		*voff = 0;
		*vlen = 0;
		return 0;
	}
	*voff = cb + dbase + st + pad;
	*vlen = len - pad;
	return 0;
}

int
scan_jb_find_key(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
				 const unsigned char *key, uint32_t kl, uint32_t *idx)
{
	uint32_t	lo = 0, hi = c->count;

	while (lo < hi)
	{
		uint32_t	mid = lo + (hi - lo) / 2, kind, voff, vlen;
		int			cmp;

		if (scan_jb_child(b, cb, clen, c, mid, &kind, &voff, &vlen) < 0 || kind != SCAN_JB_STR)
			return -1;
		if (vlen == kl)
			cmp = memcmp(b + voff, key, kl);
		else
			cmp = (vlen > kl) ? 1 : -1;
		if (cmp < 0)
			lo = mid + 1;
		else if (cmp > 0)
			hi = mid;
		else
		{
			*idx = mid;
			return 1;
		}
	}
	return 0;
}

int
scan_jb_exists(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
			   const unsigned char *key, uint32_t kl)
{
	uint32_t	idx;

	if (c->isobj)
		return scan_jb_find_key(b, cb, clen, c, key, kl, &idx);
	for (uint32_t i = 0; i < c->count; i++)
	{
		uint32_t	kind, voff, vlen;

		if (scan_jb_child(b, cb, clen, c, i, &kind, &voff, &vlen) < 0)
			return -1;
		if (kind == SCAN_JB_STR && vlen == kl && memcmp(b + voff, key, kl) == 0)
			return 1;
	}
	return 0;
}

int
scan_jb_numeric(const unsigned char *b, uint32_t off, uint32_t len, int64_t *v, uint32_t *sc)
{
	uint32_t	hl, total;

	if (len < 1)
		return -1;
	if (b[off] & 1)
	{
		if (b[off] == 1)
			return -1;
		hl = 1;
		total = (b[off] >> 1) & 0x7f;
	}
	else
	{
		if (len < 4 || (b[off] & 3) == 2)
			return -1;
		hl = 4;
		total = (jb_ld32(b, off) >> 2) & 0x3fffffffU;
	}
	if (total < hl || total > len)
		return -1;
	return scan_vm_numeric_payload_decode(b + off + hl, total - hl, v, sc) ? 1 : 0;
}

/*
 * The pattern of "col @> const". The constant is turned into a conjunction of terms, each tested by walking
 * from the root of the column's value:
 *   EQ    the value at the path (keys of objects) is a scalar equal to the given one (same JSON type)
 *   ISOBJ the value at the path is an object (an empty {} in the constant)
 *   ISARR the value at the path is an array that is not a raw scalar (an empty [] in the constant)
 *   ELEM  the value at the path is such an array and one of its elements is the given scalar
 * Shapes this expresses exactly: objects whose values are scalars, objects, empty arrays or arrays of scalars
 * (to any depth up to JBP_DEPTH), and a root array of scalars. Everything else (arrays inside arrays, raw
 * scalars, an empty root array, numbers that do not fit { int64, scale }) is refused: the planner leaves
 * the qual on the CPU.
 * Layout (little-endian words): [0] number of terms; per term: [0] type | nsteps << 8 | kind << 16, nsteps
 * pairs { byte offset of the key in the pattern, length }, then the scalar: a string { offset, length } or
 * a numeric { lo, hi, scale }; the key and string bytes follow the words.
 */
#define JBP_DEPTH	6
#define JBP_TERMS	48
typedef struct
{
	uint32_t	type, nsteps, kind;
	const unsigned char *step[JBP_DEPTH];
	uint32_t	steplen[JBP_DEPTH];
	const unsigned char *sptr;
	uint32_t	slen;
	int64_t		nv;
	uint32_t	nsc;
} jbp_term;

typedef struct
{
	const unsigned char *b;
	uint32_t	clen_root;
	jbp_term	terms[JBP_TERMS];
	uint32_t	nterms;
} jbp_ctx;

static int
jbp_scalar(jbp_ctx *x, jbp_term *t, uint32_t kind, uint32_t voff, uint32_t vlen)
{
	t->kind = kind;
	switch (kind)
	{
		case SCAN_JB_STR:
			t->sptr = x->b + voff;
			t->slen = vlen;
			return 0;
		case SCAN_JB_NUM:
			return scan_jb_numeric(x->b, voff, vlen, &t->nv, &t->nsc) == 1 ? 0 : -1;
		case SCAN_JB_FALSE:
		case SCAN_JB_TRUE:
		case SCAN_JB_NULL:
			return 0;
	}
	return -1;
}

static int
jbp_walk(jbp_ctx *x, uint32_t cb, uint32_t clen, uint32_t depth,
		 const unsigned char **ksteps, const uint32_t *kstepl)
{
	scan_jb_cont c;
	jbp_term   *t;

	if (scan_jb_open(x->b, cb, clen, &c) < 0 || c.isscalar)
		return -1;
	if (x->nterms >= JBP_TERMS)
		return -1;
	if (c.isarr)
	{
		/* an array of scalars */
		if (c.count == 0)
		{
			if (depth == 0)
				return -1;				/* the empty root array */
			t = &x->terms[x->nterms++];
			memset(t, 0, sizeof(*t));
			t->type = SCAN_JBT_ISARR;
			t->nsteps = depth;
			for (uint32_t d = 0; d < depth; d++)
			{
				t->step[d] = ksteps[d];
				t->steplen[d] = kstepl[d];
			}
			return 0;
		}
		for (uint32_t i = 0; i < c.count; i++)
		{
			uint32_t	kind, voff, vlen;

			if (scan_jb_child(x->b, cb, clen, &c, i, &kind, &voff, &vlen) < 0 || kind == SCAN_JB_CONT)
				return -1;
			if (x->nterms >= JBP_TERMS)
				return -1;
			t = &x->terms[x->nterms++];
			memset(t, 0, sizeof(*t));
			t->type = SCAN_JBT_ELEM;
			t->nsteps = depth;
			for (uint32_t d = 0; d < depth; d++)
			{
				t->step[d] = ksteps[d];
				t->steplen[d] = kstepl[d];
			}
			if (jbp_scalar(x, t, kind, voff, vlen) < 0)
				return -1;
		}
		return 0;
	}
	/* an object */
	if (c.count == 0)
	{
		t = &x->terms[x->nterms++];
		memset(t, 0, sizeof(*t));
		t->type = SCAN_JBT_ISOBJ;
		t->nsteps = depth;
		for (uint32_t d = 0; d < depth; d++)
		{
			t->step[d] = ksteps[d];
			t->steplen[d] = kstepl[d];
		}
		return 0;
	}
	if (depth >= JBP_DEPTH)
		return -1;
	for (uint32_t i = 0; i < c.count; i++)
	{
		uint32_t	kk, koff, klen, kind, voff, vlen;
		const unsigned char *st[JBP_DEPTH];
		uint32_t	stl[JBP_DEPTH];

		if (scan_jb_child(x->b, cb, clen, &c, i, &kk, &koff, &klen) < 0 || kk != SCAN_JB_STR ||
			scan_jb_child(x->b, cb, clen, &c, c.count + i, &kind, &voff, &vlen) < 0)
			return -1;
		for (uint32_t d = 0; d < depth; d++)
		{
			st[d] = ksteps[d];
			stl[d] = kstepl[d];
		}
		st[depth] = x->b + koff;
		stl[depth] = klen;
		if (kind == SCAN_JB_CONT)
		{
			if (jbp_walk(x, voff, vlen, depth + 1, st, stl) < 0)
				return -1;
			continue;
		}
		if (x->nterms >= JBP_TERMS)
			return -1;
		t = &x->terms[x->nterms++];
		memset(t, 0, sizeof(*t));
		t->type = SCAN_JBT_EQ;
		t->nsteps = depth + 1;
		for (uint32_t d = 0; d <= depth; d++)
		{
			t->step[d] = st[d];
			t->steplen[d] = stl[d];
		}
		if (jbp_scalar(x, t, kind, voff, vlen) < 0)
			return -1;
	}
	return 0;
}

int
scan_vm_jsonb_pattern(const unsigned char *jb, uint32_t len, unsigned char *out, uint32_t outcap)
{
	static jbp_ctx x;				/* too big for the stack of a backend; not re-entrant (nor needed) */
	uint32_t	nwords = 1, nbytes = 0, wi, bpos;
	unsigned char *bytes;

	memset(&x, 0, sizeof(x));
	x.b = jb;
	x.clen_root = len;
	if (jbp_walk(&x, 0, len, 0, NULL, NULL) < 0 || x.nterms == 0)
		return -1;
	for (uint32_t i = 0; i < x.nterms; i++)
	{
		jbp_term   *t = &x.terms[i];

		nwords += 1 + 2 * t->nsteps;
		for (uint32_t d = 0; d < t->nsteps; d++)
			nbytes += (t->steplen[d] + 3) & ~3U;
		if (t->type == SCAN_JBT_EQ || t->type == SCAN_JBT_ELEM)
		{
			if (t->kind == SCAN_JB_STR)
			{
				nwords += 2;
				nbytes += (t->slen + 3) & ~3U;
			}
			else if (t->kind == SCAN_JB_NUM)
				nwords += 3;
		}
	}
	if ((uint64_t) 4 * nwords + nbytes > outcap)
		return -1;
	memset(out, 0, 4 * nwords + nbytes);
	bytes = out + 4 * nwords;
	bpos = 4 * nwords;
	wi = 0;
#define PUTW(v)	do { uint32_t __w = (uint32_t)(v); memcpy(out + 4 * wi, &__w, 4); wi++; } while (0)
	PUTW(x.nterms);
	for (uint32_t i = 0; i < x.nterms; i++)
	{
		jbp_term   *t = &x.terms[i];

		PUTW(t->type | (t->nsteps << 8) | (t->kind << 16));
		for (uint32_t d = 0; d < t->nsteps; d++)
		{
			PUTW(bpos);
			PUTW(t->steplen[d]);
			memcpy(out + bpos, t->step[d], t->steplen[d]);
			bpos += (t->steplen[d] + 3) & ~3U;
		}
		if (t->type == SCAN_JBT_EQ || t->type == SCAN_JBT_ELEM)
		{
			if (t->kind == SCAN_JB_STR)
			{
				PUTW(bpos);
				PUTW(t->slen);
				memcpy(out + bpos, t->sptr, t->slen);
				bpos += (t->slen + 3) & ~3U;
			}
			else if (t->kind == SCAN_JB_NUM)
			{
				PUTW((uint64_t) t->nv & 0xffffffffU);
				PUTW((uint64_t) t->nv >> 32);
				PUTW(t->nsc);
			}
		}
	}
#undef PUTW
	(void) bytes;
	return (int) bpos;
}

/* ----------------------------------------------------------------
 * A10: float8
 * ----------------------------------------------------------------
 */

/*
 * float8eq ... float8ge, float8pl / mi / mul / div, float8um, float8up, float8abs; the mixed forms float48xx (the first
 * argument is a float4) and float84xx (the second): width 1 / 2 says which argument is converted to float8 first;
 * float8_to_int2/4/8 (width: the target's bits), float8_to_float4, intN_to_float8, float4_to_float8. The unary
 * classes carry their kind in width: DUNARY 0 = negate, 1 = abs. float8 modulo does not exist in PostgreSQL.
 */
static scan_func_class
classify_float8(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	const char *rest = NULL;
	int			k, w;

	/* the math functions of float.c (A10): sqrt/round/trunc are the aliases dsqrt/dround/dtrunc, ceil/ceiling/floor/sign by name */
	if (!strcmp(name, "dsqrt")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_SQRT; return fc; }
	if (!strcmp(name, "dround")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_ROUND; return fc; }
	if (!strcmp(name, "ceil") || !strcmp(name, "ceiling")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_CEIL; return fc; }
	if (!strcmp(name, "floor")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_FLOOR; return fc; }
	if (!strcmp(name, "dtrunc")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_TRUNC; return fc; }
	if (!strcmp(name, "sign")) { fc.cls = SOP_DUNARY; fc.width = SOP_DUN_SIGN; return fc; }
	if (!strncmp(name, "float8", 6) && strncmp(name, "float84", 7) != 0 && strncmp(name, "float82", 7) != 0)
		rest = name + 6;
	else if (!strncmp(name, "float48", 7))
	{
		rest = name + 7;
		fc.width = 1;
	}
	else if (!strncmp(name, "float84", 7))
	{
		rest = name + 7;
		fc.width = 2;
	}
	else if (!strcmp(name, "float4_to_float8"))
	{
		fc.cls = SOP_F2D;
		return fc;
	}
	else if (!strncmp(name, "int", 3) && (w = int_width_of_digit(name[3])) != 0 && !strcmp(name + 4, "_to_float8"))
	{
		fc.cls = SOP_I2D;
		return fc;
	}
	if (!rest)
		return fc;
	if (fc.width == 0)
	{
		if (!strncmp(rest, "_to_int", 7) && (w = int_width_of_digit(rest[7])) != 0 && rest[8] == '\0')
		{
			fc.cls = SOP_D2I;
			fc.width = w;
			return fc;
		}
		if (!strcmp(rest, "_to_float4"))
		{
			fc.cls = SOP_D2F;
			return fc;
		}
		if (!strcmp(rest, "um")) { fc.cls = SOP_DUNARY; fc.width = 0; return fc; }
		if (!strcmp(rest, "abs")) { fc.cls = SOP_DUNARY; fc.width = 1; return fc; }
		if (!strcmp(rest, "up")) { fc.cls = SOP_MOV; return fc; }
	}
	if ((k = cmp_kind_of(rest)) >= 0)
	{
		fc.cls = SOP_CMPD;
		fc.kind = k;
	}
	else if ((k = arith_kind_of(rest)) >= 0 && k != SOP_ARITH_MOD)
	{
		fc.cls = SOP_DARITH;
		fc.kind = k;
	}
	else
		fc.width = 0;
	return fc;
}

/* ----------------------------------------------------------------
 * A11: PostGIS
 * ----------------------------------------------------------------
 */

/* geometry_overlaps (&&), geometry_contains (~), geometry_within (@): the 2D float boxes */
static scan_func_class
classify_geometry(const char *name)
{
	scan_func_class fc = {0, 0, 0};

	if (!strcmp(name, "geometry_overlaps"))
		{ fc.cls = SOP_GEOMBOX; fc.kind = SOP_GBOX_OVERLAPS; }
	else if (!strcmp(name, "geometry_contains"))
		{ fc.cls = SOP_GEOMBOX; fc.kind = SOP_GBOX_CONTAINS; }
	else if (!strcmp(name, "geometry_within"))
		{ fc.cls = SOP_GEOMBOX; fc.kind = SOP_GBOX_WITHIN; }
	/* a point in a (multi)polygon: the operand order is the kind, the planner checks the argument shapes (codegen.c) */
	else if (!strcmp(name, "st_contains") || !strcmp(name, "st_covers"))
		{ fc.cls = SOP_GEOMPIP; fc.kind = SOP_GPIP_POLY_POINT; }
	else if (!strcmp(name, "st_within") || !strcmp(name, "st_coveredby"))
		{ fc.cls = SOP_GEOMPIP; fc.kind = SOP_GPIP_POINT_POLY; }
	else if (!strcmp(name, "st_intersects"))
		{ fc.cls = SOP_GEOMPIP; fc.kind = SOP_GPIP_EITHER; }
	/* ST_DWithin(geometry, geometry, float8) of two points (A11 part 3): the planner checks the argument shapes */
	else if (!strcmp(name, "st_dwithin"))
		{ fc.cls = SOP_GEOMDWITHIN; fc.kind = 0; }
	/* st_makepoint(x, y), st_point(x, y): only as the point of the above (fused); st_setsrid(point, const) likewise */
	else if (!strcmp(name, "st_makepoint2") || !strcmp(name, "st_point") ||
			 !strcmp(name, "st_makepoint3") || !strcmp(name, "st_makepoint4"))
		{ fc.cls = SOP_GEOMMK; fc.kind = 0; }
	/* 3D (A11 part 4): ST_Z / ST_M of a point, ST_3DDWithin of two points */
	else if (!strcmp(name, "st_z"))
		{ fc.cls = SOP_GEOMZM; fc.kind = 0; }
	else if (!strcmp(name, "st_m"))
		{ fc.cls = SOP_GEOMZM; fc.kind = 1; }
	else if (!strcmp(name, "st_3ddwithin"))
		{ fc.cls = SOP_GEOM3DDW; fc.kind = 0; }
	/* ST_3DIntersects of two points is ST_3DDWithin with the tolerance 0 (probe: threed_probe2.sql); the n-D box operator &&& */
	else if (!strcmp(name, "st_distance"))
		{ fc.cls = SOP_GEOMDIST; fc.kind = 0; }
	else if (!strcmp(name, "st_3dmaxdistance"))
		{ fc.cls = SOP_GEOMDIST; fc.kind = 2; }
	else if (!strcmp(name, "st_3ddistance"))
		{ fc.cls = SOP_GEOMDIST; fc.kind = 1; }
	else if (!strcmp(name, "st_length") || !strcmp(name, "st_length2d"))
		{ fc.cls = SOP_GEOMLEN; fc.kind = 0; }
	else if (!strcmp(name, "st_3dlength"))
		{ fc.cls = SOP_GEOMLEN; fc.kind = 1; }
	else if (!strcmp(name, "st_perimeter") || !strcmp(name, "st_perimeter2d"))
		{ fc.cls = SOP_GEOMLEN; fc.kind = 2; }
	else if (!strcmp(name, "st_3dperimeter"))
		{ fc.cls = SOP_GEOMLEN; fc.kind = 3; }
	else if (!strcmp(name, "st_3dintersects"))
		{ fc.cls = SOP_GEOM3DDW; fc.kind = SOP_G3DDW_INTERSECTS; }
	else if (!strcmp(name, "geometry_overlaps_nd"))
		{ fc.cls = SOP_GEOMBOX; fc.kind = SOP_GBOX_OVERLAPS_ND; }
	/* the flag functions read the header of any geometry */
	else if (!strcmp(name, "st_hasz"))
		{ fc.cls = SOP_GEOMFLAG; fc.kind = SOP_GFLAG_HASZ; }
	else if (!strcmp(name, "st_hasm"))
		{ fc.cls = SOP_GEOMFLAG; fc.kind = SOP_GFLAG_HASM; }
	else if (!strcmp(name, "st_ndims"))
		{ fc.cls = SOP_GEOMFLAG; fc.kind = SOP_GFLAG_NDIMS; }
	else if (!strcmp(name, "st_zmflag"))
		{ fc.cls = SOP_GEOMFLAG; fc.kind = SOP_GFLAG_ZMFLAG; }
	else if (!strcmp(name, "st_coorddim"))
		{ fc.cls = SOP_GEOMFLAG; fc.kind = SOP_GFLAG_COORDDIM; }
	else if (!strcmp(name, "st_setsrid"))
		{ fc.cls = SOP_GEOMMK; fc.kind = 1; }
	return fc;
}

/* "int4lt", "int48pl", "float4mul", "int4_to_int8", "booleq", "int4um" ... */
scan_func_class
scan_vm_classify_func(uint32_t opcode)
{
	scan_func_class	fc = {0, 0, 0};
	const char *name = (opcode < FuncOpCode__LoadVars ? func_names[opcode] : NULL);
	int			w1, w2, k;

	if (!name)
		return fc;
	if ((fc = classify_numeric(name)).cls != 0)
		return fc;
	if ((fc = classify_jsonb(name)).cls != 0)
		return fc;
	if ((fc = classify_temporal(name)).cls != 0)
		return fc;
	if ((fc = classify_string(name)).cls != 0)
		return fc;
	if ((fc = classify_float8(name)).cls != 0)
		return fc;
	if ((fc = classify_geometry(name)).cls != 0)
		return fc;
	if (!strcmp(name, "booleq")) { fc.cls = SOP_CMPI; fc.kind = SOP_CMP_EQ; return fc; }
	if (!strcmp(name, "boolne")) { fc.cls = SOP_CMPI; fc.kind = SOP_CMP_NE; return fc; }
	if (!strcmp(name, "int4_to_bool")) { fc.cls = SOP_CAST; fc.width = 1; return fc; }
	if (!strncmp(name, "float4", 6))
	{
		const char *rest = name + 6;

		/* float4_to_intN */
		if (!strncmp(rest, "_to_int", 7) && (w1 = int_width_of_digit(rest[7])) != 0 && rest[8] == '\0')
		{
			fc.cls = SOP_F2I;
			fc.width = w1;
			return fc;
		}

		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPF;
			fc.kind = k;
		}
		else if ((k = arith_kind_of(rest)) >= 0 && k != SOP_ARITH_MOD)
		{
			fc.cls = SOP_FARITH;
			fc.kind = k;
		}
		else if (!strcmp(rest, "up"))
		{
			fc.cls = SOP_MOV;
		}
		return fc;
	}
	if (strncmp(name, "int", 3) != 0 || (w1 = int_width_of_digit(name[3])) == 0)
		return fc;
	{
		const char *rest = name + 4;

		/* intA_to_float4 */
		if (!strcmp(rest, "_to_float4"))
		{
			fc.cls = SOP_I2F;
			return fc;
		}
		/* intA_to_intB */
		if (!strncmp(rest, "_to_int", 7) && (w2 = int_width_of_digit(rest[7])) != 0 && rest[8] == '\0')
		{
			fc.cls = SOP_CAST;
			fc.width = w2;
			return fc;
		}
		/* intAB<op>: the result has the wider width */
		w2 = int_width_of_digit(rest[0]);
		if (w2 != 0)
			rest++;
		else
			w2 = w1;
		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPI;
			fc.kind = k;
		}
		else if ((k = arith_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_ARITH;
			fc.kind = k;
			fc.width = (w1 > w2 ? w1 : w2);
		}
		else if (!strcmp(rest, "up") && w2 == w1)
		{
			fc.cls = SOP_MOV;
		}
		else if (!strcmp(rest, "um") && w2 == w1)
		{
			fc.cls = SOP_ARITH;			/* 0 - x */
			fc.kind = SOP_ARITH_MI;
			fc.width = w1;
			fc.cls = -SOP_ARITH;		/* marks the unary minus */
		}
	}
	return fc;
}


bool
scan_vm_func_supported(uint32_t opcode)
{
	return scan_vm_classify_func(opcode).cls != 0;
}

bool
scan_vm_agg_action_supported(int action)
{
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:		/* count(*) */
		case KAGG_ACTION__NROWS_COND:		/* count(x) */
		case KAGG_ACTION__PMIN_INT32:		/* min(int1/2/4) */
		case KAGG_ACTION__PMIN_INT64:		/* min(int8) */
		case KAGG_ACTION__PMAX_INT32:
		case KAGG_ACTION__PMAX_INT64:
		case KAGG_ACTION__PSUM_INT:			/* sum(int1/2/4) */
		case KAGG_ACTION__PSUM_INT64:		/* sum(int8) */
		case KAGG_ACTION__PAVG_INT:			/* avg(int1/2/4) */
		case KAGG_ACTION__PAVG_INT64:		/* avg(int8) */
		case KAGG_ACTION__PSUM_NUMERIC:		/* sum(numeric) of a known scale */
		case KAGG_ACTION__PAVG_NUMERIC:
		/* A10: float8, integer software binary64 in the kernel; only without GROUP BY (scan_vm_agg_action_nogroup) */
		case KAGG_ACTION__PMIN_FP64:		/* min(float4/float8) */
		case KAGG_ACTION__PMAX_FP64:
		case KAGG_ACTION__PSUM_FP:			/* sum(float8), avg(float4/float8) */
		case KAGG_ACTION__PAVG_FP:
			return true;
		default:
			break;
	}
	return false;
}

/* the aggregate runs in the kernels without GROUP BY only (the GROUP BY kernels combine states with integer atomics) */
bool
scan_vm_agg_action_nogroup(int action)
{
	switch (action)
	{
		case KAGG_ACTION__PMIN_FP64:
		case KAGG_ACTION__PMAX_FP64:
		case KAGG_ACTION__PSUM_FP:
		case KAGG_ACTION__PAVG_FP:
			return true;
		default:
			break;
	}
	return false;
}

bool
scan_vm_groupkey_type_supported(int type_code)
{
	switch (type_code)
	{
		case TypeOpCode__bool:
		case TypeOpCode__int1:
		case TypeOpCode__int2:
		case TypeOpCode__int4:
		case TypeOpCode__int8:
		/* A6: dates and times are integers; their device hash and equality are those of int4 / int8 */
		case TypeOpCode__date:
		case TypeOpCode__time:
		case TypeOpCode__timestamp:
		case TypeOpCode__timestamptz:
			return true;
		default:
			break;
	}
	return false;
}

/*
 * A6: types of a hash-join key. The integer and date/time types of a GROUP BY key, and the strings
 * (text, varchar through its RelabelType, bpchar): the kernel hashes the payload bytes (bpchar without
 * the trailing blanks) with PostgreSQL's hash_bytes() and compares them bytewise, which is what
 * texteq / bpchareq do under every deterministic collation (the planner checks that).
 */
bool
scan_vm_joinkey_type_supported(int type_code)
{
	switch (type_code)
	{
		case TypeOpCode__text:
		case TypeOpCode__bpchar:
		case TypeOpCode__float4:		/* hashed with -0 -> +0 and one NaN (SOP_HASHMERGE k1 = 3, 4), compared by CMPF / CMPD */
		case TypeOpCode__float8:
			return true;
		default:
			break;
	}
	return scan_vm_groupkey_type_supported(type_code);
}

bool
scan_vm_gpusort_key_supported(int kind, int action, int type_code)
{
	switch (kind)
	{
		case KSORT_KEY_KIND__VREF:
			/* a GROUP BY key, or count(*) / count(x) */
			if (action == 0)
				return scan_vm_groupkey_type_supported(type_code);
			return (action == KAGG_ACTION__NROWS_ANY || action == KAGG_ACTION__NROWS_COND);
		case KSORT_KEY_KIND__PSUM_INT64:		/* sum(int1/2/4) */
		case KSORT_KEY_KIND__PSUM_NUMERIC:		/* sum(int8): a 128-bit state; sum(numeric) is another action */
			return (action == KAGG_ACTION__PSUM_INT || action == KAGG_ACTION__PSUM_INT64);
		case KSORT_KEY_KIND__PMINMAX_INT64:
			return (action == KAGG_ACTION__PMIN_INT32 || action == KAGG_ACTION__PMIN_INT64 ||
					action == KAGG_ACTION__PMAX_INT32 || action == KAGG_ACTION__PMAX_INT64);
		default:
			break;		/* averages, variances, float and numeric sums: the CPU sorts */
	}
	return false;
}
