/*
 * cpu_vm.c
 *
 * See cpu_vm.h. Keep this in step with kernels_vm/scan_common.slang: the two must give the
 * same results (the GROUP BY overflow rows are aggregated here, the others on the GPU).
 */
#include <math.h>
#include <string.h>
#include "xpu_common.h"
#include "scan_caps.h"
#define F64SOFT_C 1
#include "kernels_vm/f64_soft.inc"
#include "cpu_vm.h"

static int
out_of_range(int64_t r, uint32_t w)
{
	if (w == 8)  return r < -128 || r > 127;
	if (w == 16) return r < -32768 || r > 32767;
	if (w == 32) return r < -2147483648LL || r > 2147483647LL;
	return 0;
}

/* -1, 0, +1; NaN is equal to NaN and larger than everything else */
static int
f4cmp(float a, float b)
{
	int na = isnan(a), nb = isnan(b);

	if (na)
		return nb ? 0 : 1;
	if (nb)
		return -1;
	return a < b ? -1 : (a > b ? 1 : 0);
}

static int
cmp_result(int c, uint32_t kind)
{
	switch (kind)
	{
		case SOP_CMP_EQ: return c == 0;
		case SOP_CMP_NE: return c != 0;
		case SOP_CMP_LT: return c < 0;
		case SOP_CMP_LE: return c <= 0;
		case SOP_CMP_GT: return c > 0;
		default:         return c >= 0;
	}
}

static float
bits_to_f(uint32_t u)
{
	float	f;

	memcpy(&f, &u, 4);
	return f;
}

static uint32_t
f_to_bits(float f)
{
	uint32_t	u;

	memcpy(&u, &f, 4);
	return u;
}

/* A10: float8 (the registers hold the 64 bits of the double) */
static double
bits_to_d(uint64_t u)
{
	double		d;

	memcpy(&d, &u, 8);
	return d;
}

static uint64_t
d_to_bits(double d)
{
	uint64_t	u;

	memcpy(&u, &d, 8);
	return u;
}

/* -1, 0, +1; NaN is equal to NaN and larger than everything else (float8_cmp_internal) */
static int
f8cmp(double a, double b)
{
	int na = isnan(a), nb = isnan(b);

	if (na)
		return nb ? 0 : 1;
	if (nb)
		return -1;
	return a < b ? -1 : (a > b ? 1 : 0);
}

/*
 * A6: strings. A string register is { offset | length << 32 | source << 60 }; here the source is 0
 * (the heap tuple's page: offsets are relative to the tuple header) or 3 (the program's constant pool).
 */
typedef struct
{
	const unsigned char *htup;
	const unsigned char *pool;
} vm_env;

static const unsigned char *
str_ptr(const vm_env *env, int64_t v)
{
	uint64_t	u = (uint64_t) v;

	return ((u >> 60) & 3) == 3 ? env->pool + (u & 0xffffffffu) : env->htup + (u & 0xffffffffu);
}
static uint32_t
str_len(int64_t v)
{
	return (uint32_t)(((uint64_t) v >> 32) & 0x0fffffffu);
}
static uint32_t
str_truelen(const unsigned char *p, uint32_t len)
{
	while (len > 0 && p[len - 1] == 0x20)
		len--;
	return len;
}


/* ---- A8: jsonb. The same as the jsonb section of scan_common.slang (JSONGET ... JSONCONT) ---- */
static const unsigned char *
jb_base(const vm_env *env, int64_t v)
{
	return ((((uint64_t) v) >> 60) & 3) == 3 ? env->pool : env->htup;
}

static int64_t
jb_reg(uint32_t off, uint32_t len, uint32_t src)
{
	return (int64_t)((uint64_t) off | ((uint64_t) len << 32) | ((uint64_t) src << 60));
}

/* the child 'idx' of container (kind CONT) at b + cb: fills the result register; 0, or -1 malformed */
static int
jb_pick(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c, uint32_t idx,
		uint32_t src, vm_reg *r)
{
	uint32_t	kind, voff, vlen;

	if (scan_jb_child(b, cb, clen, c, idx, &kind, &voff, &vlen) < 0)
		return -1;
	r->v = jb_reg(voff, vlen, src);
	if (kind >= SCAN_JB_FALSE && kind <= SCAN_JB_NULL)
		r->v = 0;
	r->sc = kind;
	r->isnull = 0;
	return 0;
}

/* walk the keys of a pattern term from the root value x; 1: found (kind, off, len), 0: absent, -1: malformed */
static int
jb_walk_path(const unsigned char *b, vm_reg x, const unsigned char *pat, uint32_t nsteps, const uint32_t *steps,
			 uint32_t *kind, uint32_t *voff, uint32_t *vlen)
{
	uint32_t	cb = (uint32_t)((uint64_t) x.v & 0xffffffffU), clen = str_len(x.v);

	*kind = x.sc;
	*voff = cb;
	*vlen = clen;
	for (uint32_t d = 0; d < nsteps; d++)
	{
		scan_jb_cont c;
		uint32_t	idx;
		int			rc;

		if (*kind != SCAN_JB_CONT)
			return 0;
		if (scan_jb_open(b, *voff, *vlen, &c) < 0)
			return -1;
		if (!c.isobj)
			return 0;
		rc = scan_jb_find_key(b, *voff, *vlen, &c, pat + steps[2 * d], steps[2 * d + 1], &idx);
		if (rc <= 0)
			return rc;
		if (scan_jb_child(b, *voff, *vlen, &c, c.count + idx, kind, voff, vlen) < 0)
			return -1;
	}
	return 1;
}

/* does the scalar (kind, off, len) of the column equal the pattern's scalar? 1 / 0, or -1 (cannot tell) */
static int
jb_scalar_equal(const unsigned char *b, uint32_t kind, uint32_t off, uint32_t len,
				const unsigned char *pat, uint32_t tkind, const uint32_t *sw)
{
	if (kind != tkind)
		return 0;
	if (kind == SCAN_JB_STR)
		return len == sw[1] && memcmp(b + off, pat + sw[0], len) == 0;
	if (kind == SCAN_JB_NUM)
	{
		int64_t		av, pv = (int64_t)((uint64_t) sw[0] | ((uint64_t) sw[1] << 32));
		uint32_t	asc;
		int			rc = scan_jb_numeric(b, off, len, &av, &asc);

		if (rc <= 0)
			return -1;
		return scan_vm_num_cmp(av, asc, pv, sw[2]) == 0;
	}
	return 1;
}

/* 1: the column's value x contains the pattern, 0: no, -1: cannot tell exactly */
static int
jb_contains(const unsigned char *b, vm_reg x, const unsigned char *pat)
{
	uint32_t	nterms, w[64];
	const unsigned char *tp = pat + 4;

	memcpy(&nterms, pat, 4);
	for (uint32_t t = 0; t < nterms; t++)
	{
		uint32_t	w0, type, nsteps, tkind, kind, voff, vlen, nsw;
		int			rc;
		const uint32_t *sw;

		memcpy(&w0, tp, 4);
		type = w0 & 0xff;
		nsteps = (w0 >> 8) & 0xff;
		tkind = (w0 >> 16) & 0xff;
		if (nsteps > 8)
			return -1;
		memcpy(w, tp + 4, 8 * nsteps);
		nsw = (type == SCAN_JBT_EQ || type == SCAN_JBT_ELEM) ? (tkind == SCAN_JB_STR ? 2 : tkind == SCAN_JB_NUM ? 3 : 0) : 0;
		memcpy(w + 2 * nsteps, tp + 4 + 8 * nsteps, 4 * nsw);
		sw = w + 2 * nsteps;
		tp += 4 + 8 * nsteps + 4 * nsw;
		rc = jb_walk_path(b, x, pat, nsteps, w, &kind, &voff, &vlen);
		if (rc < 0)
			return -1;
		if (rc == 0)
			return 0;
		if (type == SCAN_JBT_EQ)
		{
			rc = jb_scalar_equal(b, kind, voff, vlen, pat, tkind, sw);
			if (rc <= 0)
				return rc;
		}
		else
		{
			scan_jb_cont c;

			if (kind != SCAN_JB_CONT)
				return 0;
			if (scan_jb_open(b, voff, vlen, &c) < 0)
				return -1;
			if (type == SCAN_JBT_ISOBJ)
			{
				if (!c.isobj)
					return 0;
			}
			else
			{
				int			found = 0;

				if (!c.isarr || c.isscalar)
					return 0;
				if (type == SCAN_JBT_ELEM)
				{
					for (uint32_t i = 0; i < c.count && !found; i++)
					{
						uint32_t	ek, eo, el;

						if (scan_jb_child(b, voff, vlen, &c, i, &ek, &eo, &el) < 0)
							return -1;
						rc = jb_scalar_equal(b, ek, eo, el, pat, tkind, sw);
						if (rc < 0)
							return -1;
						found = rc;
					}
					if (!found)
						return 0;
				}
			}
		}
	}
	return 1;
}

static uint32_t
vm_json(const vm_env *env, uint32_t op, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1,
		vm_reg *regs)
{
	vm_reg		x = regs[ra];

	(void) k1;
	switch (op)
	{
		case SOP_JSONGET:
		{
			vm_reg		y = regs[rb], r;
			const unsigned char *b = jb_base(env, x.v);
			uint32_t	src = (uint32_t)(((uint64_t) x.v >> 60) & 3);
			uint32_t	cb = (uint32_t)((uint64_t) x.v & 0xffffffffU), clen = str_len(x.v);
			int			found = 0;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			memset(&r, 0, sizeof(r));
			r.isnull = 1;
			if (x.sc == SCAN_JB_CONT)
			{
				scan_jb_cont c;

				if (scan_jb_open(b, cb, clen, &c) < 0)
					return SOP_RECHECK_CODE;
				if (k0 == 0 && c.isobj)
				{
					uint32_t	idx;
					const unsigned char *key = str_ptr(env, y.v);
					int			rc = scan_jb_find_key(b, cb, clen, &c, key, str_len(y.v), &idx);

					if (rc < 0 || (rc == 1 && jb_pick(b, cb, clen, &c, c.count + idx, src, &r) < 0))
						return SOP_RECHECK_CODE;
					found = (rc == 1);
				}
				else if (k0 == 1 && c.isarr)
				{
					int64_t		i = y.v;

					if (i < 0)
					{
						if (-i > (int64_t) c.count)
							i = -1;
						else
							i += c.count;
					}
					if (i >= 0 && i < (int64_t) c.count)
					{
						if (jb_pick(b, cb, clen, &c, (uint32_t) i, src, &r) < 0)
							return SOP_RECHECK_CODE;
						found = 1;
					}
				}
			}
			else if (k0 == 1 && (y.v == 0 || y.v == -1))
			{
				r = x;					/* a scalar behaves as an array of one element */
				found = 1;
			}
			if (found)
			{
				r.isnull = 0;
				regs[dst] = r;
			}
			else
				regs[dst].isnull = 1;
			return 0;
		}
		case SOP_JSONTXT:
		{
			if (x.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			regs[dst].isnull = 0;
			regs[dst].sc = 0;
			switch (x.sc)
			{
				case SCAN_JB_STR:
					regs[dst].v = x.v & ~(1LL << 62);
					break;
				case SCAN_JB_TRUE:
					regs[dst].v = jb_reg(k0, 4, 3);
					break;
				case SCAN_JB_FALSE:
					regs[dst].v = jb_reg(k0 + 4, 5, 3);
					break;
				case SCAN_JB_NULL:
					regs[dst].isnull = 1;
					break;
				default:
					regs[dst].v = x.v | (1LL << 62);		/* deferred */
					break;
			}
			return 0;
		}
		case SOP_JSONEXISTS:
		{
			vm_reg		y = regs[rb];
			const unsigned char *b = jb_base(env, x.v);
			uint32_t	cb = (uint32_t)((uint64_t) x.v & 0xffffffffU), clen = str_len(x.v);
			int			res;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (x.sc == SCAN_JB_CONT)
			{
				scan_jb_cont c;

				if (scan_jb_open(b, cb, clen, &c) < 0)
					return SOP_RECHECK_CODE;
				res = scan_jb_exists(b, cb, clen, &c, str_ptr(env, y.v), str_len(y.v));
				if (res < 0)
					return SOP_RECHECK_CODE;
			}
			else if (x.sc == SCAN_JB_STR)
				res = (clen == str_len(y.v) && memcmp(b + cb, str_ptr(env, y.v), clen) == 0);
			else
				res = 0;
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
		case SOP_JSONTYPE:
		{
			static const uint32_t toff[6] = {0, 6, 11, 17, 23, 30};
			static const uint32_t tlen[6] = {6, 5, 6, 6, 7, 4};
			uint32_t	t;

			if (x.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			switch (x.sc)
			{
				case SCAN_JB_STR:	t = 2; break;
				case SCAN_JB_NUM:	t = 3; break;
				case SCAN_JB_FALSE:
				case SCAN_JB_TRUE:	t = 4; break;
				case SCAN_JB_NULL:	t = 5; break;
				default:
				{
					const unsigned char *b = jb_base(env, x.v);
					uint32_t	cb = (uint32_t)((uint64_t) x.v & 0xffffffffU), clen = str_len(x.v);
					scan_jb_cont c;

					if (scan_jb_open(b, cb, clen, &c) < 0)
						return SOP_RECHECK_CODE;
					if (c.isobj)
						t = 0;
					else if (!c.isscalar)
						t = 1;
					else
					{
						uint32_t	kind, voff, vlen;

						if (c.count != 1 || scan_jb_child(b, cb, clen, &c, 0, &kind, &voff, &vlen) < 0 ||
							kind == SCAN_JB_CONT)
							return SOP_RECHECK_CODE;
						t = (kind == SCAN_JB_STR ? 2 : kind == SCAN_JB_NUM ? 3 : kind == SCAN_JB_NULL ? 5 : 4);
					}
					break;
				}
			}
			regs[dst].isnull = 0;
			regs[dst].sc = 0;
			regs[dst].v = jb_reg(k0 + toff[t], tlen[t], 3);
			return 0;
		}
		case SOP_JSONNUM:
		case SOP_JSONINT:
		{
			const unsigned char *b = jb_base(env, x.v);
			uint32_t	cb = (uint32_t)((uint64_t) x.v & 0xffffffffU), clen = str_len(x.v);
			int64_t		v;
			uint32_t	sc;

			if (x.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (x.sc == SCAN_JB_NULL)
			{
				regs[dst].isnull = 1;			/* ->> of a JSON null is SQL NULL */
				return 0;
			}
			if (x.sc != SCAN_JB_NUM || scan_jb_numeric(b, cb, clen, &v, &sc) != 1)
				return SOP_RECHECK_CODE;
			if (op == SOP_JSONINT && (sc != 0 || out_of_range(v, k0)))
				return SOP_RECHECK_CODE;		/* the text is not an integer or out of range: PostgreSQL's error */
			regs[dst].isnull = 0;
			regs[dst].v = v;
			regs[dst].sc = (op == SOP_JSONNUM) ? sc : 0;
			return 0;
		}
		case SOP_JSONCONT:
		{
			int			res;

			if (x.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (x.sc != SCAN_JB_CONT)
				res = 0;						/* a scalar is a raw-scalar array: it contains no object/array pattern */
			else
			{
				res = jb_contains(jb_base(env, x.v), x, env->pool + k0);
				if (res < 0)
					return SOP_RECHECK_CODE;
			}
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
	}
	return op | 0x80000000u;
}

/* ---- A11: PostGIS. The same as vm_geom() of scan_common.slang (the 2D float box of a geometry) ---- */
typedef struct
{
	uint32_t	st;			/* 0 ok, 1 empty point, 2 undecided (the backend decides) */
	uint32_t	b[4];		/* float4 bits of xmin, xmax, ymin, ymax */
} gbox;

static uint32_t
ld32u(const unsigned char *p)
{
	uint32_t	u;

	memcpy(&u, p, 4);
	return u;
}

/* +-infinity -> +-FLT_MAX (float4 bits) */
static uint32_t
gf_clamp(uint32_t f)
{
	if ((f & 0x7fffffffu) == 0x7f800000u)
		return (f & 0x80000000u) | 0x7f7fffffu;
	return f;
}

static gbox
geom_box2d(const unsigned char *p, uint32_t len)
{
	gbox		r;
	uint32_t	gflags, pos = 4;

	memset(&r, 0, sizeof(r));
	r.st = 2;
	if (len < 12)
		return r;
	gflags = p[3];
	if ((gflags & 0x40) && (gflags & 0x10))
		pos += 8;
	if (gflags & 0x08)
		return r;
	if (gflags & 0x04)
	{
		if (pos + 16 > len)
			return r;
		for (int i = 0; i < 4; i++)
		{
			r.b[i] = ld32u(p + pos + 4 * i);
			if ((r.b[i] & 0x7fffffffu) > 0x7f800000u)
				return r;
		}
		r.st = 0;
		return r;
	}
	if (pos + 8 > len || ld32u(p + pos) != 1)
		return r;
	if (ld32u(p + pos + 4) == 0)
	{
		r.st = 1;
		return r;
	}
	if (pos + 8 + 16 > len)
		return r;
	{
		uint64_t	xb, yb;

		memcpy(&xb, p + pos + 8, 8);
		memcpy(&yb, p + pos + 16, 8);
		if ((xb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull || (yb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull)
			return r;
		/* next_float_down / next_float_up of PostGIS never give infinity: beyond the float range it is +-FLT_MAX */
		r.b[0] = gf_clamp(f64_to_f32(xb, 1));
		r.b[1] = gf_clamp(f64_to_f32(xb, 2));
		r.b[2] = gf_clamp(f64_to_f32(yb, 1));
		r.b[3] = gf_clamp(f64_to_f32(yb, 2));
	}
	r.st = 0;
	return r;
}

/* a <= b on float4 bit patterns */
static int
gf_le(uint32_t a, uint32_t b)
{
	return f4cmp(bits_to_f(a), bits_to_f(b)) <= 0;
}

/* the instructions that A6 added; the same as vm_ext() of scan_common.slang */
static uint32_t
vm_ext(const vm_env *env, uint32_t op, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1,
	   vm_reg *regs)
{
	switch (op)
	{
		case SOP_GEOMBOX:
		{
			vm_reg		x = regs[ra], y = regs[rb];
			gbox		a, b;
			int			res = 0;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			a = geom_box2d(str_ptr(env, x.v), str_len(x.v));
			b = geom_box2d(str_ptr(env, y.v), str_len(y.v));
			if (a.st == 1 || b.st == 1)
				res = (k0 != SOP_GBOX_OVERLAPS);		/* an empty operand: && false, ~ and @ true */
			else if (a.st == 2 || b.st == 2)
				return SOP_RECHECK_CODE;
			else
			{
				if (k0 == SOP_GBOX_OVERLAPS)
					res = gf_le(a.b[0], b.b[1]) && gf_le(b.b[0], a.b[1]) && gf_le(a.b[2], b.b[3]) && gf_le(b.b[2], a.b[3]);
				else if (k0 == SOP_GBOX_CONTAINS)
					res = gf_le(a.b[0], b.b[0]) && gf_le(b.b[1], a.b[1]) && gf_le(a.b[2], b.b[2]) && gf_le(b.b[3], a.b[3]);
				else
					res = gf_le(b.b[0], a.b[0]) && gf_le(a.b[1], b.b[1]) && gf_le(b.b[2], a.b[2]) && gf_le(a.b[3], b.b[3]);
			}
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
		case SOP_D2TS:
		{
			vm_reg		x = regs[ra];

			regs[dst].isnull = x.isnull;
			if (!x.isnull)
			{
				int64_t		d = x.v, r;

				if (d == 2147483647)
					r = INT64_MAX;
				else if (d == -2147483648LL)
					r = INT64_MIN;
				else if (d >= 106751983)
				{
					if (k0 == 1)
						return k1 | (SOP_ERR_TSRANGE << 16);
					r = INT64_MAX - 1;
				}
				else
					r = d * 86400000000LL;
				regs[dst].v = r;
			}
			return 0;
		}
		case SOP_TS2D:
		{
			vm_reg		x = regs[ra];

			regs[dst].isnull = x.isnull;
			if (!x.isnull)
			{
				int64_t		t = x.v, r;

				if (t == INT64_MAX)
					r = 2147483647;
				else if (t == INT64_MIN)
					r = -2147483648LL;
				else
				{
					r = t / 86400000000LL;
					if (t % 86400000000LL < 0)
						r -= 1;
				}
				regs[dst].v = r;
			}
			return 0;
		}
		case SOP_DATEOP:
		{
			vm_reg		x = regs[ra], y = regs[rb];
			int			ainf;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			regs[dst].isnull = 0;
			ainf = (x.v == 2147483647 || x.v == -2147483648LL);
			if (k0 == SOP_DATEOP_MI)
			{
				if (ainf || y.v == 2147483647 || y.v == -2147483648LL)
					return k1 | (SOP_ERR_INFSUB << 16);
				regs[dst].v = x.v - y.v;
			}
			else if (ainf)
				regs[dst].v = x.v;
			else
			{
				int64_t		r = (k0 == SOP_DATEOP_PLI) ? x.v + y.v : x.v - y.v;

				if (r < -2451545 || r >= 2145031949)
					return k1 | (SOP_ERR_DATERANGE << 16);
				regs[dst].v = r;
			}
			return 0;
		}
		case SOP_CONSTSTR:
			regs[dst].v = (int64_t)((uint64_t) k0 | ((uint64_t) k1 << 32) | ((uint64_t) 3 << 60));
			regs[dst].isnull = 0;
			return 0;
		case SOP_STRCMP:
		case SOP_STRLIKE:
		{
			vm_reg		x = regs[ra], y = regs[rb];
			const unsigned char *pa, *pb;
			uint32_t	la, lb;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if ((((uint64_t) x.v | (uint64_t) y.v) >> 62) & 1)
				return SOP_RECHECK_CODE;		/* a deferred text of ->> (a numeric or a container) */
			pa = str_ptr(env, x.v);
			pb = str_ptr(env, y.v);
			la = str_len(x.v);
			lb = str_len(y.v);
			regs[dst].isnull = 0;
			if (op == SOP_STRCMP)
			{
				int			c;

				if (k1 == 1)
				{
					la = str_truelen(pa, la);
					lb = str_truelen(pb, lb);
				}
				if (k0 == SOP_CMP_EQ || k0 == SOP_CMP_NE)
					c = (la != lb) ? 1 : memcmp(pa, pb, la);
				else
				{
					c = memcmp(pa, pb, la < lb ? la : lb);
					if (c == 0 && la != lb)
						c = la < lb ? -1 : 1;
				}
				regs[dst].v = cmp_result(c, k0);
			}
			else
			{
				uint32_t	shape = k0 & 0xff;
				int			res = 0;

				if (k0 & 0x200)
					la = str_truelen(pa, la);
				if (shape == SOP_LIKE_EXACT)
					res = (la == lb) && memcmp(pa, pb, lb) == 0;
				else if (shape == SOP_LIKE_PREFIX)
					res = (la >= lb) && memcmp(pa, pb, lb) == 0;
				else if (shape == SOP_LIKE_SUFFIX)
					res = (la >= lb) && memcmp(pa + la - lb, pb, lb) == 0;
				else if (la >= lb)
				{
					if (lb == 0)
						res = 1;
					else
						for (uint32_t i = 0; i + lb <= la; i++)
							if (pa[i] == pb[0] && memcmp(pa + i, pb, lb) == 0)
							{
								res = 1;
								break;
							}
				}
				if (k0 & 0x100)
					res = !res;
				regs[dst].v = res;
			}
			return 0;
		}
		case SOP_STRLEN:
		{
			vm_reg		x = regs[ra];

			regs[dst].isnull = x.isnull;
			if (!x.isnull && (((uint64_t) x.v >> 62) & 1))
				return SOP_RECHECK_CODE;
			if (!x.isnull)
			{
				uint32_t	n = str_len(x.v);

				if (k0 == 1)
				{
					const unsigned char *p = str_ptr(env, x.v);
					uint32_t	cnt = 0;

					for (uint32_t i = 0; i < n; i++)
						if ((p[i] & 0xc0) != 0x80)
							cnt++;
					n = cnt;
				}
				regs[dst].v = n;
			}
			return 0;
		}
		case SOP_NUMCONST:
			regs[dst].v = (int64_t)((uint64_t) k0 | ((uint64_t) k1 << 32));
			regs[dst].sc = rb;
			regs[dst].isnull = ra;
			return 0;
		case SOP_NUMCMP:
		{
			vm_reg		x = regs[ra], y = regs[rb];

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			regs[dst].isnull = 0;
			regs[dst].v = cmp_result(scan_vm_num_cmp(x.v, x.sc, y.v, y.sc), k0);
			return 0;
		}
		case SOP_NUMARITH:
		{
			vm_reg		x = regs[ra], y;
			int64_t		a, b, r;
			uint32_t	sc;

			if (k0 == SOP_NUMOP_NEG || k0 == SOP_NUMOP_ABS)
			{
				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					regs[dst].v = (k0 == SOP_NUMOP_NEG || x.v < 0) ? -x.v : x.v;
					regs[dst].sc = x.sc;
				}
				return 0;
			}
			if (k0 >= SOP_NUMOP_CEIL)
			{
				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					int64_t		ur;
					uint32_t	usc;

					scan_vm_num_unary((int) k0, x.v, x.sc, &ur, &usc);
					regs[dst].v = ur;
					regs[dst].sc = usc;
				}
				return 0;
			}
			y = regs[rb];
			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			a = x.v;
			b = y.v;
			if (k0 == SOP_ARITH_MUL)
			{
				sc = x.sc + y.sc;
				if (__builtin_mul_overflow(a, b, &r))
					return SOP_RECHECK_CODE;
			}
			else
			{
				sc = x.sc > y.sc ? x.sc : y.sc;
				if (!scan_vm_num_up(&a, sc - x.sc) || !scan_vm_num_up(&b, sc - y.sc))
					return SOP_RECHECK_CODE;
				if (k0 == SOP_ARITH_PL ? __builtin_add_overflow(a, b, &r) : __builtin_sub_overflow(a, b, &r))
					return SOP_RECHECK_CODE;
			}
			if (r == INT64_MIN || sc > 255)
				return SOP_RECHECK_CODE;
			regs[dst].v = r;
			regs[dst].sc = sc;
			regs[dst].isnull = 0;
			return 0;
		}
		case SOP_NUMDIV:
		case SOP_NUMRND:
		{
			vm_reg		x = regs[ra], y = regs[rb];
			int64_t		qr;
			uint32_t	qsc;
			int			ok;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (op == SOP_NUMDIV)
				ok = scan_vm_num_div((int) k0, x.v, x.sc, y.v, y.sc, &qr, &qsc);
			else
				ok = scan_vm_num_round((int) k0, x.v, x.sc, y.v, &qr, &qsc);
			if (!ok)
				return SOP_RECHECK_CODE;
			regs[dst].v = qr;
			regs[dst].sc = qsc;
			regs[dst].isnull = 0;
			return 0;
		}
		case SOP_I2N:
		{
			vm_reg		x = regs[ra];

			regs[dst].isnull = x.isnull;
			if (!x.isnull)
			{
				if (x.v == INT64_MIN)
					return SOP_RECHECK_CODE;
				regs[dst].v = x.v;
				regs[dst].sc = 0;
			}
			return 0;
		}
		case SOP_N2I:
		{
			vm_reg		x = regs[ra];

			regs[dst].isnull = x.isnull;
			if (!x.isnull)
			{
				static const int64_t p10[19] = {
					1LL, 10LL, 100LL, 1000LL, 10000LL, 100000LL, 1000000LL, 10000000LL, 100000000LL,
					1000000000LL, 10000000000LL, 100000000000LL, 1000000000000LL, 10000000000000LL,
					100000000000000LL, 1000000000000000LL, 10000000000000000LL, 100000000000000000LL,
					1000000000000000000LL};
				int64_t		m = x.v < 0 ? -x.v : x.v, q, r;

				if (x.sc == 0)
					q = m;
				else if (x.sc >= 20)
					q = 0;
				else if (x.sc == 19)
					q = (m >= 5000000000000000000LL) ? 1 : 0;
				else
				{
					q = m / p10[x.sc];
					if (m - q * p10[x.sc] >= p10[x.sc] - (m - q * p10[x.sc]))
						q++;
				}
				r = x.v < 0 ? -q : q;
				if (out_of_range(r, k0))
					return SOP_RECHECK_CODE;
				regs[dst].v = r;
			}
			return 0;
		}
		default:
			if (op >= SOP_JSONGET && op <= SOP_JSONCONT)
				return vm_json(env, op, dst, ra, rb, k0, k1, regs);
			break;
	}
	return op | 0x80000000u;
}

/* instructions [ip, end); returns 0 or the error code */
static uint32_t
vm_run(const vm_env *env, const uint32_t *insn, uint32_t ip, uint32_t end, vm_reg *regs)
{
	while (ip < end)
	{
		const uint32_t *in = insn + 6 * (size_t) ip;
		uint32_t	op = in[0], dst = in[1], ra = in[2], rb = in[3], k0 = in[4], k1 = in[5];

		ip++;
		switch (op)
		{
			case SOP_CONST:
				regs[dst].v = (int64_t)((uint64_t) k0 | ((uint64_t) k1 << 32));
				regs[dst].isnull = ra;
				break;
			case SOP_AND:
				if (regs[ra].isnull)
					regs[dst].isnull = 1;
				else if (regs[ra].v == 0)
				{
					regs[dst].v = 0;
					regs[dst].isnull = 0;
					ip = rb;
				}
				break;
			case SOP_OR:
				if (regs[ra].isnull)
					regs[dst].isnull = 1;
				else if (regs[ra].v != 0)
				{
					regs[dst].v = 1;
					regs[dst].isnull = 0;
					ip = rb;
				}
				break;
			case SOP_NOT:
				regs[dst].isnull = regs[ra].isnull;
				regs[dst].v = (regs[ra].v == 0);
				break;
			case SOP_ISNULL:
				regs[dst].v = (regs[ra].isnull != 0);
				regs[dst].isnull = 0;
				break;
			case SOP_ISNOTNULL:
				regs[dst].v = (regs[ra].isnull == 0);
				regs[dst].isnull = 0;
				break;
			case SOP_MOV:
				regs[dst] = regs[ra];
				break;
			case SOP_CAST:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					if (k0 == 1)
						regs[dst].v = (x.v != 0);
					else
					{
						if (out_of_range(x.v, k0))
							return k1 | (SOP_ERR_INTRANGE << 16);
						regs[dst].v = x.v;
					}
				}
				break;
			}
			case SOP_CMPI:
			case SOP_CMPF:
			{
				vm_reg	x = regs[ra], y = regs[rb];

				if (x.isnull || y.isnull)
					regs[dst].isnull = 1;
				else
				{
					int		c;

					if (op == SOP_CMPI)
						c = (x.v < y.v ? -1 : (x.v > y.v ? 1 : 0));
					else
						c = f4cmp(bits_to_f((uint32_t) x.v), bits_to_f((uint32_t) y.v));
					regs[dst].isnull = 0;
					regs[dst].v = cmp_result(c, k0);
				}
				break;
			}
			case SOP_ARITH:
			{
				vm_reg	x = regs[ra], y = regs[rb];
				uint32_t aop = k0 & 0xffu, w = k0 >> 8, why = SOP_ERR_INTRANGE;
				int64_t	r = 0;
				int		bad = 0;

				if (x.isnull || y.isnull)
				{
					regs[dst].isnull = 1;
					break;
				}
				switch (aop)
				{
					case SOP_ARITH_PL:
						bad = (w == 64) ? __builtin_add_overflow(x.v, y.v, &r)
							: (r = x.v + y.v, out_of_range(r, w));
						break;
					case SOP_ARITH_MI:
						bad = (w == 64) ? __builtin_sub_overflow(x.v, y.v, &r)
							: (r = x.v - y.v, out_of_range(r, w));
						break;
					case SOP_ARITH_MUL:
						bad = (w == 64) ? __builtin_mul_overflow(x.v, y.v, &r)
							: (r = x.v * y.v, out_of_range(r, w));
						break;
					case SOP_ARITH_DIV:
						if (y.v == 0)
						{
							bad = 1;
							why = SOP_ERR_ZERODIV;
						}
						else if (y.v == -1)
						{
							if (w == 64)
							{
								bad = (x.v == INT64_MIN);
								r = bad ? 0 : -x.v;
							}
							else
							{
								r = -x.v;
								bad = out_of_range(r, w);
							}
						}
						else
							r = x.v / y.v;
						break;
					default:	/* MOD */
						if (y.v == 0)
						{
							bad = 1;
							why = SOP_ERR_ZERODIV;
						}
						else if (y.v == -1)
							r = 0;
						else
							r = x.v % y.v;
						break;
				}
				if (bad)
					return k1 | (why << 16);
				regs[dst].v = r;
				regs[dst].isnull = 0;
				break;
			}
			case SOP_FARITH:
			{
				vm_reg	x = regs[ra], y = regs[rb];
				float	a, b, r = 0;
				int		bad = 0, inf_in;
				uint32_t why = SOP_ERR_FLTOVER;

				if (x.isnull || y.isnull)
				{
					regs[dst].isnull = 1;
					break;
				}
				a = bits_to_f((uint32_t) x.v);
				b = bits_to_f((uint32_t) y.v);
				inf_in = isinf(a) || isinf(b);
				switch (k0)
				{
					case SOP_ARITH_PL: r = a + b; bad = isinf(r) && !inf_in; break;
					case SOP_ARITH_MI: r = a - b; bad = isinf(r) && !inf_in; break;
					case SOP_ARITH_MUL:
						r = a * b;
						bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && b != 0);
						if (!isinf(r))
							why = SOP_ERR_FLTUNDER;
						break;
					default:
						if (b == 0 && !isnan(a))		/* float4div: NaN / 0 is NaN */
						{
							bad = 1;
							why = SOP_ERR_ZERODIV;
						}
						else
						{
							r = a / b;
							bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && !isinf(b));
							if (!isinf(r))
								why = SOP_ERR_FLTUNDER;
						}
						break;
				}
				if (bad)
					return k1 | (why << 16);
				regs[dst].v = (int64_t) f_to_bits(r);
				regs[dst].isnull = 0;
				break;
			}
			case SOP_CMPD:
			{
				vm_reg	x = regs[ra], y = regs[rb];

				if (x.isnull || y.isnull)
					regs[dst].isnull = 1;
				else
				{
					regs[dst].isnull = 0;
					regs[dst].v = cmp_result(f8cmp(bits_to_d((uint64_t) x.v), bits_to_d((uint64_t) y.v)), k0);
				}
				break;
			}
			case SOP_DARITH:
			{
				vm_reg	x = regs[ra], y = regs[rb];
				volatile double a, b, r = 0;		/* volatile: separate operations, never fused */
				uint32_t why = 0;

				if (x.isnull || y.isnull)
				{
					regs[dst].isnull = 1;
					break;
				}
				a = bits_to_d((uint64_t) x.v);
				b = bits_to_d((uint64_t) y.v);
				if (k0 == SOP_ARITH_DIV && b == 0.0 && !isnan(a))		/* float8div: NaN / 0 is NaN */
					why = SOP_ERR_ZERODIV;
				else
				{
					switch (k0)
					{
						case SOP_ARITH_PL: r = a + b; break;
						case SOP_ARITH_MI: r = a - b; break;
						case SOP_ARITH_MUL: r = a * b; break;
						default: r = a / b; break;
					}
					if (isinf(r) && !isinf(a) && !isinf(b))
						why = SOP_ERR_FLTOVER;
					else if (k0 == SOP_ARITH_MUL && r == 0.0 && a != 0.0 && b != 0.0)
						why = SOP_ERR_FLTUNDER;
					else if (k0 == SOP_ARITH_DIV && r == 0.0 && a != 0.0 && !isinf(b))
						why = SOP_ERR_FLTUNDER;
				}
				if (why)
					return k1 | (why << 16);
				regs[dst].v = (int64_t) d_to_bits(r);
				regs[dst].isnull = 0;
				break;
			}
			case SOP_DUNARY:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
					regs[dst].v = (int64_t) (k0 == 0 ? ((uint64_t) x.v ^ 0x8000000000000000ull)
												 : ((uint64_t) x.v & 0x7fffffffffffffffull));
				break;
			}
			case SOP_I2D:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
					regs[dst].v = (int64_t) d_to_bits((double) x.v);
				break;
			}
			case SOP_D2I:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					double	d = rint(bits_to_d((uint64_t) x.v));
					int64_t	r;

					if (!(d >= -9223372036854775808.0 && d < 9223372036854775808.0))	/* false for NaN */
						return k1 | (SOP_ERR_INTRANGE << 16);
					r = (int64_t) d;
					if (out_of_range(r, k0))
						return k1 | (SOP_ERR_INTRANGE << 16);
					regs[dst].v = r;
				}
				break;
			}
			case SOP_F2D:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
					regs[dst].v = (int64_t) d_to_bits((double) bits_to_f((uint32_t) x.v));
				break;
			}
			case SOP_D2F:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					double	d = bits_to_d((uint64_t) x.v);
					float	f = (float) d;

					if (isinf(f) && !isinf(d))
						return k1 | (SOP_ERR_FLTOVER << 16);
					if (f == 0.0f && d != 0.0)
						return k1 | (SOP_ERR_FLTUNDER << 16);
					regs[dst].v = (int64_t) f_to_bits(f);
				}
				break;
			}
			case SOP_I2F:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
					regs[dst].v = (int64_t) f_to_bits((float) x.v);
				break;
			}
			case SOP_F2I:
			{
				vm_reg	x = regs[ra];

				regs[dst].isnull = x.isnull;
				if (!x.isnull)
				{
					float	f = bits_to_f((uint32_t) x.v);
					int64_t	r;

					if (isnan(f) || isinf(f) || f >= 9.2233720368547758e18f || f < -9.2233720368547758e18f)
						return k1 | (SOP_ERR_INTRANGE << 16);
					r = (int64_t) rintf(f);
					if (out_of_range(r, k0))
						return k1 | (SOP_ERR_INTRANGE << 16);
					regs[dst].v = r;
				}
				break;
			}
			default:
			{
				uint32_t	e = vm_ext(env, op, dst, ra, rb, k0, k1, regs);

				if (e != 0)
					return e;
				break;
			}
		}
	}
	return 0;
}

int
cpu_eval_row(const scan_prog *prog, const unsigned char *htup,
			 vm_reg regs[SCAN_MAX_REGS], uint32_t *err)
{
	const uint32_t *w = prog->words;
	uint32_t	coloff = w[0], insnoff = w[2], ncols = w[3], nq = w[5], nins = w[6], qreg = w[7];
	const uint32_t *insn = w + insnoff;
	uint16_t	infomask2, infomask;
	uint32_t	natts, off = htup[22];
	int			hasnull;
	int			hit = 1;
	uint32_t	e = 0;
	int			recheck = 0;
	vm_env		env;

	env.htup = htup;
	env.pool = (const unsigned char *) (w + w[20]);
	memcpy(&infomask2, htup + 18, 2);
	memcpy(&infomask, htup + 20, 2);
	natts = infomask2 & 0x7ff;
	hasnull = infomask & 1;
	for (int i = 0; i < SCAN_MAX_REGS; i++)
	{
		regs[i].v = 0;
		regs[i].isnull = 1;
		regs[i].sc = 0;
	}
	for (uint32_t c = 0; c < ncols; c++)
	{
		uint32_t	desc = w[coloff + c];
		uint32_t	alen = desc & 0xffu, align = (desc >> 8) & 0xfu, vt = (desc >> 12) & 0xfu;
		uint32_t	slot = (desc >> 16) & 0xffu;

		if (c >= natts || (hasnull && !((htup[23 + (c >> 3)] >> (c & 7)) & 1)))
			continue;
		if (alen == 0xff)
		{
			uint32_t	sz;

			if (htup[off] == 0)
				off = (off + align - 1) & ~(align - 1);
			if (htup[off] & 1)
				sz = (htup[off] == 1) ? ((htup[off + 1] == 18) ? 18 : 10) : ((htup[off] >> 1) & 0x7f);
			else
			{
				uint32_t	h;

				memcpy(&h, htup + off, 4);
				sz = (h >> 2) & 0x3fffffffu;
			}
			if (slot != 0xff)
			{
				regs[slot].v = 0;
				regs[slot].isnull = 0;
				if (vt == SOP_VT_NUM)
				{
					uint32_t	hl = 4, nsc;
					int64_t		nv;

					if (htup[off] & 1)
					{
						hl = 1;
						if (htup[off] == 1)
							recheck = 1;
					}
					else if ((htup[off] & 3) == 2)
						recheck = 1;
					if (!recheck)
					{
						if (scan_vm_numeric_payload_decode(htup + off + hl, sz - hl, &nv, &nsc))
						{
							regs[slot].v = nv;
							regs[slot].sc = nsc;
						}
						else
							recheck = 1;
					}
				}
				else if (vt == SOP_VT_STR || vt == SOP_VT_JSON)
				{
					uint32_t	hl = 4;

					if (htup[off] & 1)
					{
						hl = 1;
						if (htup[off] == 1)
							recheck = 1;
					}
					else if ((htup[off] & 3) == 2)
						recheck = 1;
					if (!recheck)
					{
						regs[slot].v = (int64_t)((uint64_t) (off + hl) | ((uint64_t) (sz - hl) << 32));
						regs[slot].sc = (vt == SOP_VT_JSON) ? SCAN_JB_CONT : 0;
					}
				}
			}
			off += sz;
		}
		else if (alen == 0xfe)
		{
			*err = 0x7fffffffu;
			return -1;
		}
		else
		{
			off = (off + align - 1) & ~(align - 1);
			if (slot != 0xff && vt == SOP_VT_NULLONLY)
			{
				regs[slot].v = 0;
				regs[slot].isnull = 0;
			}
			else if (slot != 0xff)
			{
				int64_t		v = 0;

				if (alen == 1)
					v = (vt == SOP_VT_INT) ? (int64_t)(int8_t) htup[off] : (int64_t) htup[off];
				else if (alen == 2)
				{
					int16_t		x;
					memcpy(&x, htup + off, 2);
					v = x;
				}
				else if (alen == 4)
				{
					int32_t		x;
					uint32_t	u;
					memcpy(&x, htup + off, 4);
					memcpy(&u, htup + off, 4);
					v = (vt == SOP_VT_FLOAT4) ? (int64_t) u : (int64_t) x;
				}
				else
					memcpy(&v, htup + off, 8);
				regs[slot].v = v;
				regs[slot].isnull = 0;
			}
			off += alen;
		}
	}
	if (recheck)
	{
		*err = 0x7ffffffeU;			/* a compressed or external string: the GPU would have flagged the row */
		return -1;
	}
	if (qreg != 0xffffffffu)
	{
		e = vm_run(&env, insn, 0, nq, regs);
		if (e == 0)
			hit = (!regs[qreg].isnull && regs[qreg].v != 0);
	}
	if (e == 0 && hit && nins > nq)
		e = vm_run(&env, insn, nq, nins, regs);
	if (e != 0)
	{
		*err = e;
		return -1;
	}
	return hit;
}
