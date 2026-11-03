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
			fc.cls = SOP_STRLIKE; fc.kind = (rest[0] == 'n'); fc.width = 1;
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
	if ((fc = classify_temporal(name)).cls != 0)
		return fc;
	if ((fc = classify_string(name)).cls != 0)
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
