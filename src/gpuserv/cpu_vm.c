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
	uint32_t	hz, z[2];	/* &&& only: has a Z; float4 bits of zmin, zmax */
	uint32_t	hm, m[2];	/* &&& only: has an M; float4 bits of mmin, mmax */
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
geom_box2d(const unsigned char *p, uint32_t len, int nd)
{
	gbox		r;
	uint32_t	gflags, pos = 4, hzf, hmf;

	memset(&r, 0, sizeof(r));
	r.st = 2;
	if (len < 12)
		return r;
	gflags = p[3];
	hzf = nd ? (gflags & 1) : 0;
	hmf = nd ? ((gflags >> 1) & 1) : 0;
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
		{
			uint32_t	zp = pos + 16;

			if (hzf)
			{
				if (zp + 8 > len)
					return r;
				r.hz = 1;
				r.z[0] = ld32u(p + zp);
				r.z[1] = ld32u(p + zp + 4);
				zp += 8;
				if ((r.z[0] & 0x7fffffffu) > 0x7f800000u || (r.z[1] & 0x7fffffffu) > 0x7f800000u)
					return r;
			}
			if (hmf)
			{
				if (zp + 8 > len)
					return r;
				r.hm = 1;
				r.m[0] = ld32u(p + zp);
				r.m[1] = ld32u(p + zp + 4);
				if ((r.m[0] & 0x7fffffffu) > 0x7f800000u || (r.m[1] & 0x7fffffffu) > 0x7f800000u)
					return r;
			}
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
	if (pos + 8 + 16 + 8 * (hzf + hmf) > len)
		return r;
	{
		uint64_t	xb, yb;

		memcpy(&xb, p + pos + 8, 8);
		memcpy(&yb, p + pos + 16, 8);
		if ((xb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull || (yb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull)
			return r;
		/* &&& (n-D box, gidx): a coordinate beyond the float4 range is the backend's (see scan_common.slang) */
		if (nd && ((xb & 0x7fffffffffffffffull) > 0x47EFFFFFE0000000ull || (yb & 0x7fffffffffffffffull) > 0x47EFFFFFE0000000ull))
			return r;
		if (hzf)
		{
			uint64_t	zb;

			memcpy(&zb, p + pos + 24, 8);
			if ((zb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull ||
				(nd && (zb & 0x7fffffffffffffffull) > 0x47EFFFFFE0000000ull))
				return r;
			r.hz = 1;
			r.z[0] = gf_clamp(f64_to_f32(zb, 1));
			r.z[1] = gf_clamp(f64_to_f32(zb, 2));
		}
		if (hmf)
		{
			uint64_t	mb;

			memcpy(&mb, p + pos + 24 + 8 * hzf, 8);
			if ((mb & 0x7fffffffffffffffull) > 0x7ff0000000000000ull ||
				(nd && (mb & 0x7fffffffffffffffull) > 0x47EFFFFFE0000000ull))
				return r;
			r.hm = 1;
			r.m[0] = gf_clamp(f64_to_f32(mb, 1));
			r.m[1] = gf_clamp(f64_to_f32(mb, 2));
		}
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

/*
 * ---- A11: a point in a (multi)polygon (GEOMPIP). The same as vm_geom_pip() of scan_common.slang. ----
 * Non-zero winding number; an edge's side of the point is decided on the software binary64 only when its sign is
 * beyond the error bound of the orientation test, anything doubtful (on an edge, on a vertex, too close, overflow,
 * non-finite) is PIP_UNSURE and the backend evaluates the row.
 */
#define PIP_OUT		0
#define PIP_IN		1
#define PIP_UNSURE	2

#define F64_NONFINITE(a)	(((a) & F64_MAG) >= F64_INF)
/* 2^-49: the bound of Shewchuk's orientation filter (3 + 16 eps) eps = 3.3e-16 with a safety factor of 5 */
#define PIP_ERRBOUND		0x3CE0000000000000ull

static uint64_t
ld64u(const unsigned char *p)
{
	uint64_t	u;

	memcpy(&u, p, 8);
	return u;
}

/* the winding number of a ring of n points at p (16 bytes each) around the point (px, py) */
static int
pip_ring(const unsigned char *p, uint32_t n, uint64_t px, uint64_t py, uint32_t stride)
{
	uint64_t	x1, y1;
	int			wn = 0;

	if (n < 4)
		return PIP_UNSURE;
	x1 = ld64u(p);
	y1 = ld64u(p + 8);
	if (F64_NONFINITE(x1) || F64_NONFINITE(y1))
		return PIP_UNSURE;
	for (uint32_t i = 1; i < n; i++)
	{
		uint64_t	x2 = ld64u(p + (size_t) stride * i);
		uint64_t	y2 = ld64u(p + (size_t) stride * i + 8);
		uint64_t	ymin, ymax;

		if (F64_NONFINITE(x2) || F64_NONFINITE(y2))
			return PIP_UNSURE;
		if (f64_eq(y1, y2))
		{
			/* horizontal (or a zero-length segment: ignored): only a point on it matters, and that is the boundary */
			if (f64_eq(x1, x2) || !f64_eq(py, y1))
				goto next;
			if (f64_le(x1, x2) ? (f64_le(x1, px) && f64_le(px, x2)) : (f64_le(x2, px) && f64_le(px, x1)))
				return PIP_UNSURE;
			goto next;
		}
		ymin = f64_lt(y1, y2) ? y1 : y2;
		ymax = f64_lt(y1, y2) ? y2 : y1;
		if (f64_lt(py, ymin) || f64_gt(py, ymax))
			goto next;
		{
			uint64_t	l = f64_mul(f64_sub(px, x1), f64_sub(y2, y1));
			uint64_t	r = f64_mul(f64_sub(x2, x1), f64_sub(py, y1));
			uint64_t	det = f64_sub(r, l);			/* cross(b - a, p - a): > 0 when p is left of a -> b */
			uint64_t	s = f64_add(l & F64_MAG, r & F64_MAG);
			uint32_t	e = (uint32_t) ((s >> 52) & 0x7ff);

			/* the error bound holds for finite, non-subnormal terms; s = 0 is a point on the line through a vertex */
			if (e < 0x080 || e > 0x7f0)
				return PIP_UNSURE;
			if (!f64_gt(det & F64_MAG, f64_mul(s, PIP_ERRBOUND)))
				return PIP_UNSURE;
			if (f64_le(y1, py) && f64_lt(py, y2))
			{
				if ((det & F64_SIGN) == 0)
					wn++;
			}
			else if (f64_le(y2, py) && f64_lt(py, y1))
			{
				if ((det & F64_SIGN) != 0)
					wn--;
			}
		}
	next:
		x1 = x2;
		y1 = y2;
	}
	return wn != 0 ? PIP_IN : PIP_OUT;
}

/* the polygon whose type word is at pos: the result, and *pnext = the offset after it (0: malformed) */
static int
pip_polygon(const unsigned char *p, uint32_t len, uint32_t pos, uint64_t px, uint64_t py, uint32_t *pnext)
{
	uint32_t	nrings, ptsoff, off;
	int			res = PIP_OUT;

	*pnext = 0;
	if (pos + 8 > len || ld32u(p + pos) != GEOM_POLYGONTYPE)
		return PIP_UNSURE;
	nrings = ld32u(p + pos + 4);
	if (nrings > (len - pos - 8) / 4)
		return PIP_UNSURE;
	ptsoff = pos + 8 + 4 * nrings + ((nrings & 1) ? 4 : 0);
	if (ptsoff > len)
		return PIP_UNSURE;
	off = ptsoff;
	for (uint32_t r = 0; r < nrings; r++)
	{
		uint32_t	n = ld32u(p + pos + 8 + 4 * r);
		int			rr;

		if (n > (len - off) / 16)
			return PIP_UNSURE;
		if (res != PIP_UNSURE && (r == 0 || res == PIP_IN))
		{
			rr = pip_ring(p + off, n, px, py, 16);
			if (rr == PIP_UNSURE)
				res = PIP_UNSURE;
			else if (r == 0)
				res = rr;
			else if (rr == PIP_IN)
				res = PIP_OUT;				/* in a hole */
		}
		off += 16 * n;
	}
	*pnext = off;
	return res;
}

/* the header of a geometry: 0 ok; *gtype, *srid (21 bits) and the offset of the type word. 1: not decidable here */
static int
geom_head(const unsigned char *p, uint32_t len, uint32_t *gtype, uint32_t *srid, uint32_t *pos, uint32_t *flags)
{
	uint32_t	gflags, o = 4;

	if (len < 12)
		return 1;
	gflags = p[3];
	if ((p[0] & 0xe0) != 0)
		return 1;
	*srid = ((uint32_t) (p[0] & 0x1f) << 16) | ((uint32_t) p[1] << 8) | p[2];
	if ((gflags & 0x40) && (gflags & 0x10))
		o += 8;
	*flags = gflags;
	if (o + 8 > len)
		return 1;
	if (gflags & 0x04)
	{
		/* a cached box: 4 floats when the geometry is 2D (a Z or M box has more, those are not handled) */
		if (gflags & 0x03)
			return 1;
		o += 16;
		if (o + 8 > len)
			return 1;
	}
	*pos = o;
	*gtype = ld32u(p + o);
	return 0;
}

/* 0: res set (0/1); else SOP_RECHECK_CODE. poly, point: string registers (point_x != NULL: the fused point) */
static uint32_t
geom_pip(const unsigned char *pa, uint32_t la, const unsigned char *pb, uint32_t lb, int have_xy,
		 uint64_t fx, uint64_t fy, uint32_t fsrid, uint32_t k0, int *res)
{
	uint32_t	ta, sa, pos_a, fl_a;
	uint32_t	tb = GEOM_POINTTYPE, sb = fsrid, pos_b = 0, fl_b = 0;
	uint64_t	px, py;
	int			rr = PIP_OUT;

	if (geom_head(pa, la, &ta, &sa, &pos_a, &fl_a))
		return SOP_RECHECK_CODE;
	if (!have_xy)
	{
		if (geom_head(pb, lb, &tb, &sb, &pos_b, &fl_b))
			return SOP_RECHECK_CODE;
		if ((k0 & SOP_GPIP_SYMM) && ta == GEOM_POINTTYPE && (tb == GEOM_POLYGONTYPE || tb == GEOM_MULTIPOLYGONTYPE))
		{
			const unsigned char *tp = pa; uint32_t tl = la;

			pa = pb; pb = tp; la = lb; lb = tl;
			{ uint32_t t;
			  t = ta; ta = tb; tb = t;
			  t = sa; sa = sb; sb = t;
			  t = pos_a; pos_a = pos_b; pos_b = t;
			  t = fl_a; fl_a = fl_b; fl_b = t; }
		}
	}
	if ((ta != GEOM_POLYGONTYPE && ta != GEOM_MULTIPOLYGONTYPE) || tb != GEOM_POINTTYPE)
		return SOP_RECHECK_CODE;
	if (sa != sb)
		return SOP_RECHECK_CODE;				/* PostGIS raises "mixed SRID": the backend does */
	if ((fl_a & 0x0b) != 0)
		return SOP_RECHECK_CODE;				/* Z, M or geodetic polygon */
	if (have_xy)
	{
		px = fx;
		py = fy;
	}
	else
	{
		uint32_t	npts;

		if ((fl_b & 0x0c) != 0 || pos_b + 8 > lb)	/* geodetic, or a cached box on a point */
			return SOP_RECHECK_CODE;
		npts = ld32u(pb + pos_b + 4);
		if (npts == 0)
		{
			*res = 0;							/* POINT EMPTY */
			return 0;
		}
		if (npts != 1 || pos_b + 8 + 16 > lb)
			return SOP_RECHECK_CODE;
		px = ld64u(pb + pos_b + 8);
		py = ld64u(pb + pos_b + 16);
	}
	if (F64_NONFINITE(px) || F64_NONFINITE(py))
		return SOP_RECHECK_CODE;
	/* the cached float box is rounded outward: a point beyond it is outside */
	if (fl_a & 0x04)
	{
		uint32_t	bo = pos_a - 16;
		uint32_t	bx0 = ld32u(pa + bo), bx1 = ld32u(pa + bo + 4), by0 = ld32u(pa + bo + 8), by1 = ld32u(pa + bo + 12);

		if ((bx0 & 0x7fffffffu) <= 0x7f800000u && (bx1 & 0x7fffffffu) <= 0x7f800000u &&
			(by0 & 0x7fffffffu) <= 0x7f800000u && (by1 & 0x7fffffffu) <= 0x7f800000u &&
			(f64_lt(px, f64_from_f32(bx0)) || f64_gt(px, f64_from_f32(bx1)) ||
			 f64_lt(py, f64_from_f32(by0)) || f64_gt(py, f64_from_f32(by1))))
		{
			*res = 0;
			return 0;
		}
	}
	if (ta == GEOM_POLYGONTYPE)
	{
		uint32_t	nx;

		rr = pip_polygon(pa, la, pos_a, px, py, &nx);
	}
	else
	{
		uint32_t	ng, o;

		if (pos_a + 8 > la)
			return SOP_RECHECK_CODE;
		ng = ld32u(pa + pos_a + 4);
		o = pos_a + 8;
		rr = PIP_OUT;
		/* PostGIS: the first polygon the point is not outside of decides (inside, or on its boundary: not "contained") */
		for (uint32_t i = 0; i < ng; i++)
		{
			uint32_t	nx;
			int			r1 = pip_polygon(pa, la, o, px, py, &nx);

			if (nx == 0)
				return SOP_RECHECK_CODE;
			if (r1 != PIP_OUT)
			{
				rr = r1;
				break;
			}
			o = nx;
		}
	}
	if (rr == PIP_UNSURE)
		return SOP_RECHECK_CODE;
	*res = (rr == PIP_IN);
	return 0;
}

/*
 * ---- A11: ST_DWithin of two points (GEOMDWITHIN). The same as vm_geom_dwithin() of scan_common.slang. ----
 */
#define GDW_LO		0x3FEFFFFFFFFFFFC0ull	/* 1 - 2^-47 */
#define GDW_HI		0x3FF0000000000020ull	/* 1 + 2^-47 */
#define GDW_TINY	0x07B0000000000000ull	/* 2^-900 */

static uint32_t
gdw_load(const unsigned char *p, uint32_t len, uint32_t *srid, uint64_t *px, uint64_t *py, int *empty)
{
	uint32_t	gtype, pos, flags, npts;

	if (geom_head(p, len, &gtype, srid, &pos, &flags) || gtype != GEOM_POINTTYPE || (flags & 0x0c) != 0 ||
		pos + 8 > len)
		return SOP_RECHECK_CODE;
	npts = ld32u(p + pos + 4);
	*empty = (npts == 0);
	if (npts != 0)
	{
		uint32_t	nd = 2 + (flags & 1) + ((flags >> 1) & 1);

		if (npts != 1 || pos + 8 + 8 * nd > len)
			return SOP_RECHECK_CODE;
		*px = ld64u(p + pos + 8);
		*py = ld64u(p + pos + 16);
	}
	return 0;
}

static uint32_t
geom_dwithin(const vm_env *env, const vm_reg *regs, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1, int *res)
{
	vm_reg		a = regs[ra], b = regs[rb];
	vm_reg		ya = regs[(k0 >> 8) & 0xff], yb = regs[(k0 >> 16) & 0xff], t = regs[k0 >> 24];
	int			fa = (k0 & SOP_GDW_FUSED_A) != 0, fb = (k0 & SOP_GDW_FUSED_B) != 0;
	uint32_t	sa = k1 & 0xfffff, sb = sa;
	uint64_t	ax = (uint64_t) a.v, ay = (uint64_t) ya.v, bx = (uint64_t) b.v, by = (uint64_t) yb.v, tol, dx, dy;
	int			ea = 0, eb = 0;

	if (!fa && gdw_load(str_ptr(env, a.v), str_len(a.v), &sa, &ax, &ay, &ea))
		return SOP_RECHECK_CODE;
	if (!fb && gdw_load(str_ptr(env, b.v), str_len(b.v), &sb, &bx, &by, &eb))
		return SOP_RECHECK_CODE;
	if (sa != sb)
		return SOP_RECHECK_CODE;
	tol = (uint64_t) t.v;
	if (F64_NONFINITE(tol) || ((tol >> 63) != 0 && (tol & F64_MAG) != 0))
		return SOP_RECHECK_CODE;
	*res = 0;
	if (ea || eb)
		return 0;
	if (F64_NONFINITE(ax) || F64_NONFINITE(ay) || F64_NONFINITE(bx) || F64_NONFINITE(by))
		return SOP_RECHECK_CODE;
	dx = f64_sub(ax, bx);
	dy = f64_sub(ay, by);
	if (F64_NONFINITE(dx) || F64_NONFINITE(dy))
		return SOP_RECHECK_CODE;
	if (((dx | dy) & F64_MAG) == 0)
		*res = 1;
	else if (k1 & SOP_GDW_EXACT)
	{
		uint64_t	q = f64_sqr_add(dx, f64_mul(dy, dy));

		if (F64_NONFINITE(q))
			return SOP_RECHECK_CODE;
		if (f64_lt(0x47EFFFFFE0000000ull, f64_sqrt(q)))
			return SOP_RECHECK_CODE;		/* beyond FLT_MAX, PostGIS' initial distance */
		*res = f64_le(f64_sqrt(q), tol);
	}
	else
	{
		uint64_t	d2 = f64_add(f64_mul(dx, dx), f64_mul(dy, dy));
		uint64_t	r2 = f64_mul(tol, tol);

		if (F64_NONFINITE(d2) || F64_NONFINITE(r2) || (d2 & F64_MAG) < GDW_TINY ||
			((r2 & F64_MAG) != 0 && (r2 & F64_MAG) < GDW_TINY))
			return SOP_RECHECK_CODE;
		if (f64_le(d2, f64_mul(r2, GDW_LO)))
			*res = 1;
		else if (f64_lt(f64_mul(r2, GDW_HI), d2))
			*res = 0;
		else
			return SOP_RECHECK_CODE;
	}
	return 0;
}

static uint32_t
gdw3_load(const unsigned char *p, uint32_t len, uint32_t *srid, uint64_t *px, uint64_t *py, uint64_t *pz, int *hz, int *empty)
{
	uint32_t	gtype, pos, flags, npts;

	if (geom_head(p, len, &gtype, srid, &pos, &flags) || gtype != GEOM_POINTTYPE || (flags & 0x0c) != 0 ||
		pos + 8 > len)
		return SOP_RECHECK_CODE;
	npts = ld32u(p + pos + 4);
	*empty = (npts == 0);
	*hz = ((flags & 1) != 0);
	if (npts != 0)
	{
		uint32_t	nd = 2 + (flags & 1) + ((flags >> 1) & 1);

		if (npts != 1 || pos + 8 + 8 * nd > len)
			return SOP_RECHECK_CODE;
		*px = ld64u(p + pos + 8);
		*py = ld64u(p + pos + 16);
		if (*hz)
			*pz = ld64u(p + pos + 24);
	}
	return 0;
}

/* ST_Z (k0 = 0) / ST_M (k0 = 1) / ST_X (2) / ST_Y (3) of a point: the same as vm_geom_zm() of scan_common.slang */
static uint32_t
geom_zm(const unsigned char *p, uint32_t len, uint32_t k0, int *isnull, uint64_t *val)
{
	uint32_t	gtype, srid, pos, flags, npts, hz, hm, idx;

	if (geom_head(p, len, &gtype, &srid, &pos, &flags) || gtype != GEOM_POINTTYPE || (flags & 0x0c) != 0 || pos + 8 > len)
		return SOP_RECHECK_CODE;
	npts = ld32u(p + pos + 4);
	hz = flags & 1;
	hm = (flags >> 1) & 1;
	idx = (k0 == 0 ? 2 : (k0 == 1 ? 2 + hz : k0 - 2));	/* ST_X (k0 2), ST_Y (3) */
	*isnull = 0;
	if (npts == 0 || (k0 == 0 && hz == 0) || (k0 == 1 && hm == 0))
	{
		*isnull = 1;
		return 0;
	}
	if (npts != 1 || pos + 8 + 8 * (2 + hz + hm) > len)
		return SOP_RECHECK_CODE;
	*val = ld64u(p + pos + 8 + 8 * idx);
	return 0;
}

/* the type word, SRID, dimension count and type-word offset of a geometry value; 0 when it is not readable here */
static uint32_t
gd_hdr(const unsigned char *s, uint32_t len, uint32_t *srid, uint32_t *nd, uint32_t *o)
{
	uint32_t	gflags;

	if (len < 12 || (s[0] & 0xe0) != 0)
		return 0;
	gflags = s[3];
	if (gflags & 0x08)
		return 0;
	*nd = 2 + (gflags & 1) + ((gflags >> 1) & 1);
	*srid = ((uint32_t)(s[0] & 0x1f) << 16) | ((uint32_t) s[1] << 8) | s[2];
	*o = 4;
	if ((gflags & 0x40) && (gflags & 0x10))
		*o += 8;
	if (gflags & 0x04)
		*o += 8 * *nd;
	if (*o + 8 > len)
		return 0;
	return ld32u(s + *o);
}

/* ---- ST_Distance of a point and a line, polygon, multi-geometry or collection: the same as gd_geom() of scan_common.slang (see the comment
 * there: one loop over the members, one containment test, one segment projection, one distance) ---- */
static int
gd_geom(const unsigned char *s, uint32_t len, uint64_t px, uint64_t py, uint64_t pz, int d3, uint64_t *best, int *found)
{
	uint32_t	srid = 0, nd = 2, o = 0, gtype = gd_hdr(s, len, &srid, &nd, &o), p = o, end = len, stride, rem0 = 1, rem1 = 0;

	if (gtype == 0 || gtype > 7)
		return 1;
	stride = 8 * nd;
	if (d3 && (s[3] & 1) == 0)
		return 1;
	if (gtype >= 4)
	{
		rem0 = ld32u(s + p + 4);
		p += 8;
		if (rem0 > (end - p) / 8)
			return 1;
	}
	for (uint32_t guard = 0; guard < 65536; guard++)
	{
		uint32_t	t, n, base, aoff = 0, an = 0;
		int			measure = 0;

		if (rem1 > 0)
			rem1--;
		else
		{
			if (rem0 == 0)
				break;
			rem0--;
			if (gtype == 7)
			{
				uint32_t	st;

				if (p + 8 > end)
					return 1;
				st = ld32u(s + p);
				if (st >= 4 && st <= 6)
				{
					rem1 = ld32u(s + p + 4);
					p += 8;
					if (rem1 > (end - p) / 8)
						return 1;
					continue;
				}
			}
		}
		if (p + 8 > end)
			return 1;
		t = ld32u(s + p);
		n = ld32u(s + p + 4);
		base = p + 8;
		if (t == 1)
		{
			if (n > 1 || (n == 1 && base + stride > end))
				return 1;
			aoff = base;
			an = n;
			measure = (n == 1);
			p = base + n * stride;
		}
		else if (t == 2)
		{
			if (n > (end - base) / stride)
				return 1;
			aoff = base;
			an = n;
			measure = (n > 0);
			p = base + n * stride;
		}
		else if (t == 3)
		{
			uint32_t	data = base + 4 * n + ((n & 1) ? 4 : 0), q = data;
			int			decided = 0;

			if (d3)
				return 1;
			if (n > (end - base) / 4 || data > end)
				return 1;
			for (uint32_t r = 0; r < n; r++)
			{
				uint32_t	np = ld32u(s + base + 4 * r);

				if (np > (end - q) / stride)
					return 1;
				if (!decided)
				{
					uint32_t	rr = pip_ring(s + q, np, px, py, stride);

					if (rr == PIP_UNSURE)
						return 1;
					if (r == 0 ? (rr == PIP_OUT) : (rr == PIP_IN))
					{
						aoff = q;
						an = np;
						measure = 1;
						decided = 1;
					}
				}
				q += np * stride;
			}
			if (n > 0)
			{
				*found = 1;
				if (!decided)
					*best = 0;
			}
			p = q;
		}
		else
			return 1;
		if (measure)
		{
			uint64_t	ax = ld64u(s + aoff), ay = ld64u(s + aoff + 8), az = d3 ? ld64u(s + aoff + 16) : 0;

			*found = 1;
			for (uint32_t i = 0; i < an; i++)
			{
				uint64_t	bx = ax, by = ay, bz = az, qx, qy, qz, rad = 0, scl = 0x3FF0000000000000ull, dd;
				int			inside = 0;

				if (i > 0)
				{
					bx = ld64u(s + aoff + i * stride);
					by = ld64u(s + aoff + i * stride + 8);
					if (d3)
						bz = ld64u(s + aoff + i * stride + 16);
				}
				if (F64_NONFINITE(bx) || F64_NONFINITE(by) || (d3 && F64_NONFINITE(bz)))
					return 1;
				qx = bx;
				qy = by;
				qz = bz;
				if (i > 0)
				{
					qx = ax;
					qy = ay;
					qz = az;
					if (!(f64_eq(ax, bx) && f64_eq(ay, by) && (!d3 || f64_eq(az, bz))))
					{
						uint64_t	bxa = f64_sub(bx, ax), bya = f64_sub(by, ay);
						uint64_t	pxa = f64_sub(px, ax), pya = f64_sub(py, ay);
						uint64_t	bza = d3 ? f64_sub(bz, az) : 0, pza = d3 ? f64_sub(pz, az) : 0;
						uint64_t	num = f64_fma(pxa, bxa, f64_mul(pya, bya));
						uint64_t	den = f64_fma(bxa, bxa, f64_mul(bya, bya));
						uint64_t	mag = f64_add(f64_mul(pxa, bxa) & F64_MAG, f64_mul(pya, bya) & F64_MAG);

						if (d3)
						{
							num = f64_fma(pza, bza, num);
							den = f64_fma(bza, bza, den);
						}
						if (F64_NONFINITE(bxa) || F64_NONFINITE(bya) || F64_NONFINITE(pxa) || F64_NONFINITE(pya) ||
							F64_NONFINITE(bza) || F64_NONFINITE(pza) || F64_NONFINITE(num) || F64_NONFINITE(den) ||
							(den & F64_MAG) == 0)
							return 1;
						if (d3)
						{
							uint64_t	rr = f64_div(num, den);

							if (F64_NONFINITE(rr))
								return 1;
							if (f64_gt(rr, 0x3FF0000000000000ull))
							{
								qx = bx;
								qy = by;
								qz = bz;
							}
							else if (!f64_lt(rr, 0))
							{
								qx = f64_fma(rr, bxa, ax);
								qy = f64_fma(rr, bya, ay);
								qz = f64_fma(rr, bza, az);
							}
						}
						else
						{
							if (f64_le(num & F64_MAG, f64_mul(mag, 0x3D70000000000000ull)) ||
								f64_le(f64_sub(num, den) & F64_MAG, f64_mul(f64_add(mag, den), 0x3D70000000000000ull)))
								return 1;
							if (f64_gt(num, den))
							{
								qx = bx;
								qy = by;
							}
							else if (!f64_lt(num, 0))
							{
								/* a vertical or horizontal segment: |px - ax| / |py - ay| exactly (see the kernel) */
								uint64_t	sv;

								if (f64_eq(ax, bx))
									sv = f64_sub(px, ax) & F64_MAG;
								else if (f64_eq(ay, by))
									sv = f64_sub(py, ay) & F64_MAG;
								else
								{
									uint64_t	cross = f64_fma(f64_sub(ay, py), bxa, f64_mul(f64_sub(ax, px), bya) ^ F64_SIGN);

									sv = f64_div(cross, den);
									rad = den;
								}
								if (F64_NONFINITE(sv))
									return 1;
								scl = sv & F64_MAG;
								if (f64_eq(ax, bx) || f64_eq(ay, by))
									rad = 0x3FF0000000000000ull;
								inside = 1;
							}
						}
					}
				}
				if (!inside)
				{
					uint64_t	h = f64_sub(qx, px), v = f64_sub(qy, py);

					rad = f64_sqr_add(h, f64_mul(v, v));
					if (d3)
					{
						uint64_t	w = f64_sub(qz, pz);

						if (F64_NONFINITE(w))
							return 1;
						rad = f64_sqr_add(w, rad);
					}
					if (F64_NONFINITE(h) || F64_NONFINITE(v))
						return 1;
				}
				if (F64_NONFINITE(rad))
					return 1;
				dd = f64_mul(scl, f64_sqrt(rad));
				if (F64_NONFINITE(dd))
					return 1;
				if (f64_lt(dd, *best))
					*best = dd;
				ax = bx;
				ay = by;
				az = bz;
			}
		}
	}
	return (rem0 > 0 || rem1 > 0) ? 1 : 0;
}

/* ST_Distance / ST_3DDistance / ST_3DMaxDistance of two points: the same as vm_geom_dist() of scan_common.slang. *isnull: an empty operand */
static uint32_t
geom_dist(const vm_env *env, const vm_reg *regs, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1, int *isnull, uint64_t *val)
{
	vm_reg		a = regs[ra], b = regs[rb];
	vm_reg		ya = regs[(k0 >> 8) & 0xff], yb = regs[(k0 >> 16) & 0xff];
	vm_reg		za = regs[(k1 >> 20) & 0x1f], zb = regs[(k1 >> 25) & 0x1f];
	int			fa = (k0 & SOP_GDW_FUSED_A) != 0, fb = (k0 & SOP_GDW_FUSED_B) != 0;
	int			d3 = (k1 & SOP_GDIST_3D) != 0;
	uint32_t	sa = k1 & 0xfffff, sb = sa;
	uint64_t	ax = (uint64_t) a.v, ay = (uint64_t) ya.v, az = (uint64_t) za.v;
	uint64_t	bx = (uint64_t) b.v, by = (uint64_t) yb.v, bz = (uint64_t) zb.v;
	uint64_t	dx, dy, q;
	int			ea = 0, eb = 0, hza = 1, hzb = 1;

	*isnull = 0;
	if (!fa && gdw3_load(str_ptr(env, a.v), str_len(a.v), &sa, &ax, &ay, &az, &hza, &ea))
		return SOP_RECHECK_CODE;
	if (!fb && gdw3_load(str_ptr(env, b.v), str_len(b.v), &sb, &bx, &by, &bz, &hzb, &eb))
		return SOP_RECHECK_CODE;
	if (sa != sb)
		return SOP_RECHECK_CODE;
	if (ea || eb)
	{
		*isnull = 1;
		return 0;
	}
	if (d3 && (!hza || !hzb))
		return SOP_RECHECK_CODE;
	if (F64_NONFINITE(ax) || F64_NONFINITE(ay) || F64_NONFINITE(bx) || F64_NONFINITE(by) ||
		(d3 && (F64_NONFINITE(az) || F64_NONFINITE(bz))))
		return SOP_RECHECK_CODE;
	dx = f64_sub(bx, ax);
	dy = f64_sub(by, ay);
	q = f64_sqr_add(dx, f64_mul(dy, dy));
	if (d3)
	{
		uint64_t	dz = f64_sub(bz, az);

		if (F64_NONFINITE(dz))
			return SOP_RECHECK_CODE;
		q = f64_sqr_add(dz, q);
	}
	if (F64_NONFINITE(dx) || F64_NONFINITE(dy) || F64_NONFINITE(q))
		return SOP_RECHECK_CODE;
	*val = f64_sqrt(q);
	if (f64_lt(0x47EFFFFFE0000000ull, *val))
		return SOP_RECHECK_CODE;
	return 0;
}

/* ST_Distance / ST_3DDistance (op SOP_GEOMDIST), ST_DWithin (SOP_GEOMDWITHIN) and ST_3DDWithin (SOP_GEOM3DDW) with an operand that is not a point:
 * the same as the non-point part of vm_geom_dwdist() of scan_common.slang. *np = 0: both operands are points (the caller calls geom_dist /
 * geom_dwithin / geom_3ddwithin); else the result is in *isnull and *val (ST_DWithin: the boolean) or the return value is SOP_RECHECK_CODE.
 * The null checks are the caller's. */
static uint32_t
geom_dwdist_np(const vm_env *env, const vm_reg *regs, uint32_t op, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1,
			   int *np, int *isnull, uint64_t *val)
{
	int			isdw = (op == SOP_GEOMDWITHIN || op == SOP_GEOM3DDW);
	int			d3 = (op == SOP_GEOM3DDW) || (op == SOP_GEOMDIST && (k1 & SOP_GDIST_3D) != 0);
	vm_reg		a = regs[ra], b = regs[rb];
	vm_reg		ya = regs[(k0 >> 8) & 0xff], yb = regs[(k0 >> 16) & 0xff], t = regs[k0 >> 24];
	vm_reg		za = regs[(k1 >> 20) & 0x1f], zb = regs[(k1 >> 25) & 0x1f];
	int			fa = (k0 & SOP_GDW_FUSED_A) != 0, fb = (k0 & SOP_GDW_FUSED_B) != 0;
	uint32_t	hsa = 0, hsb = 0, hna = 0, hnb = 0, hoa = 0, hob = 0;
	uint32_t	ta = fa ? 1 : gd_hdr(str_ptr(env, a.v), str_len(a.v), &hsa, &hna, &hoa);
	uint32_t	tb = fb ? 1 : gd_hdr(str_ptr(env, b.v), str_len(b.v), &hsb, &hnb, &hob);
	int			pa = (ta == 1), pe = 0, phz = 1, found = 0;
	uint64_t	px, py, pz, best = 0x7FEFFFFFFFFFFFFFull, tol = (uint64_t) t.v;
	uint32_t	psrid = k1 & 0xfffff;
	vm_reg		g = pa ? b : a;

	*np = 0;
	if (ta == 0 || tb == 0)
		return SOP_RECHECK_CODE;
	if (ta == 1 && tb == 1)
		return 0;
	*np = 1;
	*isnull = 0;
	if (ta != 1 && tb != 1)
		return SOP_RECHECK_CODE;
	if (isdw ? ((k0 & SOP_GDW_NP) == 0 || (k1 & SOP_GDW_EXACT) == 0) : (k0 & SOP_GDW_NP) == 0)
		return SOP_RECHECK_CODE;
	px = pa ? (uint64_t) a.v : (uint64_t) b.v;
	py = pa ? (uint64_t) ya.v : (uint64_t) yb.v;
	pz = pa ? (uint64_t) za.v : (uint64_t) zb.v;
	if (pa ? !fa : !fb)
	{
		vm_reg		pr = pa ? a : b;

		if (gdw3_load(str_ptr(env, pr.v), str_len(pr.v), &psrid, &px, &py, &pz, &phz, &pe))
			return SOP_RECHECK_CODE;
	}
	if ((pa ? hsb : hsa) != psrid)
		return SOP_RECHECK_CODE;
	if (d3 && !phz)
		return SOP_RECHECK_CODE;
	if (isdw && (F64_NONFINITE(tol) || ((tol >> 63) != 0 && (tol & F64_MAG) != 0)))
		return SOP_RECHECK_CODE;
	*val = 0;
	if (pe)
	{
		*isnull = !isdw;
		return 0;
	}
	if (F64_NONFINITE(px) || F64_NONFINITE(py) || (d3 && F64_NONFINITE(pz)))
		return SOP_RECHECK_CODE;
	if (gd_geom(str_ptr(env, g.v), str_len(g.v), px, py, pz, d3, &best, &found))
		return SOP_RECHECK_CODE;
	if (!found)
	{
		*isnull = !isdw;
		return 0;
	}
	if (isdw)
	{
		if (f64_lt(0x47EFFFFFE0000000ull, best))
			return SOP_RECHECK_CODE;
		*val = f64_le(best, tol) ? 1 : 0;
	}
	else
	{
		if (f64_lt(0x47EFFFFFE0000000ull, best))
			return SOP_RECHECK_CODE;
		*val = best;
	}
	return 0;
}

/* PostGIS 3.7's ptarray_signed_area, the one text the kernel compiles too (kernels_vm/area_ring.inc) */
#define AREA_FN static inline
#define AR_F2 struct ArF2s
#define AR_SRC_T const unsigned char *
#define AR_LD(src, off) ld64u((src) + (off))
#include "kernels_vm/area_ring.inc"

/* ST_Length / ST_Length2D (k0 0), ST_3DLength (1), ST_Perimeter / ST_Perimeter2D (2), ST_3DPerimeter (3), ST_Area (4): the same as vm_geom_len() of
 * scan_common.slang (see the comment there for the summation order, and for why it is written as single loops) */
static uint32_t
geom_len(const unsigned char *s, uint32_t len, uint32_t k0, uint64_t *val)
{
	uint32_t	gflags, hz, hm, nd, o = 4, p, end = len, gtype, stride, rem0 = 1, rem1 = 0;
	int			per = (k0 >= 2), area = (k0 == 4), d3, inner = 0;
	uint64_t	tot = 0, sub = 0;

	if (len < 12 || (s[0] & 0xe0) != 0)
		return SOP_RECHECK_CODE;
	gflags = s[3];
	if (gflags & 0x08)
		return SOP_RECHECK_CODE;
	hz = gflags & 1;
	hm = (gflags >> 1) & 1;
	nd = 2 + hz + hm;
	if ((gflags & 0x40) && (gflags & 0x10))
		o += 8;
	if (gflags & 0x04)
		o += 8 * nd;
	if (o + 8 > len)
		return SOP_RECHECK_CODE;
	d3 = ((k0 & 1) != 0) && hz;
	p = o;
	gtype = ld32u(s + p);
	if (gtype == 0 || gtype > 7)
		return SOP_RECHECK_CODE;
	stride = 8 * nd;
	if (d3 && (s[3] & 1) == 0)
		return 1;
	if (gtype >= 4)
	{
		rem0 = ld32u(s + p + 4);
		p += 8;
		if (rem0 > (end - p) / 8)
			return SOP_RECHECK_CODE;
	}
	for (uint32_t guard = 0; guard < 65536; guard++)
	{
		uint32_t	t, n, base, narr = 0, rbase = 0, data = 0, q;
		uint64_t	c = 0;

		if (inner)
		{
			if (rem1 == 0)
			{
				tot = f64_add(tot, sub);
				inner = 0;
				if (F64_NONFINITE(tot))
					return SOP_RECHECK_CODE;
			}
			else
				rem1--;
		}
		if (!inner)
		{
			if (rem0 == 0)
				break;
			rem0--;
			if (gtype == 7)
			{
				uint32_t	st;

				if (p + 8 > end)
					return SOP_RECHECK_CODE;
				st = ld32u(s + p);
				if (st >= 4 && st <= 6)
				{
					rem1 = ld32u(s + p + 4);
					p += 8;
					if (rem1 > (end - p) / 8)
						return SOP_RECHECK_CODE;
					sub = 0;
					inner = 1;
					continue;
				}
			}
		}
		if (p + 8 > end)
			return SOP_RECHECK_CODE;
		t = ld32u(s + p);
		n = ld32u(s + p + 4);
		base = p + 8;
		if (t == 1)
		{
			if (n > 1 || (n == 1 && base + stride > end))
				return SOP_RECHECK_CODE;
			p = base + n * stride;
		}
		else if (t == 2)
		{
			if (n > (end - base) / stride)
				return SOP_RECHECK_CODE;
			narr = per ? 0 : 1;
			data = base;
		}
		else if (t == 3)
		{
			data = base + 4 * n + ((n & 1) ? 4 : 0);
			if (n > (end - base) / 4 || data > end)
				return SOP_RECHECK_CODE;
			narr = per ? n : 0;
			rbase = base;
		}
		else
			return SOP_RECHECK_CODE;
		q = data;
		for (uint32_t ar = 0; ar < narr; ar++)
		{
			uint32_t	np = (t == 2) ? n : ld32u(s + rbase + 4 * ar);
			uint64_t	asum = 0;

			if (np > (end - q) / stride)
				return SOP_RECHECK_CODE;
			if (area)
			{
				if (np >= 3)
				{
					uint64_t	sa = area_ring_signed(s, q, stride, np);

					if (sa == AR_NONE)
						return SOP_RECHECK_CODE;
					sa &= F64_MAG;
					c = (ar == 0) ? f64_add(c, sa) : f64_sub(c, sa);
				}
				q += np * stride;
				continue;
			}
			for (uint32_t i = 0; i + 1 < np; i++)
			{
				uint32_t	p0 = q + i * stride, p1 = p0 + stride;
				uint64_t	dx = f64_sub(ld64u(s + p1), ld64u(s + p0));
				uint64_t	dy = f64_sub(ld64u(s + p1 + 8), ld64u(s + p0 + 8));
				uint64_t	qq = f64_sqr_add(dx, f64_mul(dy, dy));

				if (d3)
					qq = f64_sqr_add(f64_sub(ld64u(s + p1 + 16), ld64u(s + p0 + 16)), qq);
				if (F64_NONFINITE(dx) || F64_NONFINITE(dy) || F64_NONFINITE(qq))
					return SOP_RECHECK_CODE;
				asum = f64_add(asum, f64_sqrt(qq));
				if (F64_NONFINITE(asum))
					return SOP_RECHECK_CODE;
			}
			c = f64_add(c, asum);
			q += np * stride;
		}
		if (t == 2)
			p = base + n * stride;
		else if (t == 3)
		{
			uint32_t	qe = data;

			for (uint32_t ar = 0; ar < n; ar++)
			{
				uint32_t	np = ld32u(s + rbase + 4 * ar);

				if (np > (end - qe) / stride)
					return SOP_RECHECK_CODE;
				qe += np * stride;
			}
			p = qe;
		}
		if (inner)
			sub = f64_add(sub, c);
		else
			tot = f64_add(tot, c);
		if (F64_NONFINITE(sub) || F64_NONFINITE(tot))
			return SOP_RECHECK_CODE;
	}
	if (inner)
		tot = f64_add(tot, sub);
	if (rem0 > 0 || rem1 > 0 || F64_NONFINITE(tot))
		return SOP_RECHECK_CODE;
	*val = tot;
	return 0;
}

/* ST_NPoints (k0 6), ST_NumPoints (7), ST_NumGeometries (8), ST_NRings (9), ST_NumInteriorRings (10), ST_IsEmpty (11): the same as
 * vm_geom_cnt() of scan_common.slang; *isnull is set for the NULL answers of ST_NumPoints / ST_NumInteriorRings */
static uint32_t
geom_cnt(const unsigned char *s, uint32_t len, uint32_t k0, int *isnull, int64_t *val)
{
	uint32_t	gflags, nd, o = 4, p, end = len, gtype, stride, n0, rem0 = 1, rem1 = 0, top, npts = 0, nrings = 0, v;
	int			inner = 0, nonempty = 0;

	*isnull = 0;
	if (len < 12 || (s[0] & 0xe0) != 0)
		return SOP_RECHECK_CODE;
	gflags = s[3];
	if (gflags & 0x08)
		return SOP_RECHECK_CODE;
	nd = 2 + (gflags & 1) + ((gflags >> 1) & 1);
	if ((gflags & 0x40) && (gflags & 0x10))
		o += 8;
	if (gflags & 0x04)
		o += 8 * nd;
	if (o + 8 > len)
		return SOP_RECHECK_CODE;
	p = o;
	gtype = ld32u(s + p);
	if (gtype == 0 || gtype > 7)
		return SOP_RECHECK_CODE;
	stride = 8 * nd;
	n0 = ld32u(s + p + 4);
	if (k0 == SOP_GFLAG_NUMPOINTS || k0 == SOP_GFLAG_NUMINTRINGS)
	{
		if (k0 == SOP_GFLAG_NUMPOINTS ? gtype != 2 : gtype != 3)
			*isnull = 1;
		else
			*val = (k0 == SOP_GFLAG_NUMPOINTS) ? (int64_t) n0 : (int64_t) (n0 == 0 ? 0 : n0 - 1);
		return 0;
	}
	if (gtype >= 4)
	{
		rem0 = n0;
		p += 8;
		if (rem0 > (end - p) / 8)
			return SOP_RECHECK_CODE;
	}
	top = rem0;
	for (uint32_t guard = 0; guard < 65536; guard++)
	{
		uint32_t	t, n, base;

		if (inner)
		{
			if (rem1 == 0)
				inner = 0;
			else
				rem1--;
		}
		if (!inner)
		{
			if (rem0 == 0)
				break;
			rem0--;
			if (gtype == 7)
			{
				uint32_t	st;

				if (p + 8 > end)
					return SOP_RECHECK_CODE;
				st = ld32u(s + p);
				if (st >= 4 && st <= 6)
				{
					rem1 = ld32u(s + p + 4);
					p += 8;
					if (rem1 > (end - p) / 8)
						return SOP_RECHECK_CODE;
					inner = 1;
					continue;
				}
			}
		}
		if (p + 8 > end)
			return SOP_RECHECK_CODE;
		t = ld32u(s + p);
		n = ld32u(s + p + 4);
		base = p + 8;
		if (t == 1)
		{
			if (n > 1 || (n == 1 && base + stride > end))
				return SOP_RECHECK_CODE;
			npts += n;
			p = base + n * stride;
		}
		else if (t == 2)
		{
			if (n > (end - base) / stride)
				return SOP_RECHECK_CODE;
			npts += n;
			p = base + n * stride;
		}
		else if (t == 3)
		{
			uint32_t	data = base + 4 * n + ((n & 1) ? 4 : 0), qe = data;

			if (n > (end - base) / 4 || data > end)
				return SOP_RECHECK_CODE;
			for (uint32_t ar = 0; ar < n; ar++)
			{
				uint32_t	np = ld32u(s + base + 4 * ar);

				if (np > (end - qe) / stride)
					return SOP_RECHECK_CODE;
				if (ar == 0 && np == 0)
					return SOP_RECHECK_CODE;
				npts += np;
				qe += np * stride;
			}
			nrings += n;
			p = qe;
		}
		else
			return SOP_RECHECK_CODE;
		if (n > 0)
			nonempty = 1;
	}
	if (rem0 > 0 || rem1 > 0)
		return SOP_RECHECK_CODE;
	v = npts;
	if (k0 == SOP_GFLAG_NUMGEOMS)
		v = nonempty ? (gtype >= 4 ? top : 1) : 0;
	else if (k0 == SOP_GFLAG_NRINGS)
		v = nrings;
	else if (k0 == SOP_GFLAG_ISEMPTY)
		v = nonempty ? 0 : 1;
	*val = (int64_t) v;
	return 0;
}

/* ST_HasZ / ST_HasM / ST_NDims / ST_Zmflag / ST_CoordDim / ST_SRID: the same as vm_geom_flag() of scan_common.slang */
static uint32_t
geom_flag(const unsigned char *p, uint32_t len, uint32_t k0, int64_t *val)
{
	uint32_t	hz, hm, v;

	if (len < 8 || (p[0] & 0xe0) != 0)
		return SOP_RECHECK_CODE;
	if (k0 == SOP_GFLAG_SRID)
	{
		if (p[0] & 0x10)
			return SOP_RECHECK_CODE;
		*val = (int64_t) (((uint32_t) p[0] << 16) | ((uint32_t) p[1] << 8) | p[2]);
		return 0;
	}
	hz = p[3] & 1;
	hm = (p[3] >> 1) & 1;
	v = 2 + hz + hm;
	if (k0 == SOP_GFLAG_HASZ)
		v = hz;
	else if (k0 == SOP_GFLAG_HASM)
		v = hm;
	else if (k0 == SOP_GFLAG_ZMFLAG)
		v = 2 * hz + hm;
	*val = (int64_t) v;
	return 0;
}

/* the same as vm_geom_3ddwithin() of scan_common.slang */
static uint32_t
geom_3ddwithin(const vm_env *env, const vm_reg *regs, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1, int *res)
{
	vm_reg		a = regs[ra], b = regs[rb];
	vm_reg		ya = regs[(k0 >> 8) & 0xff], yb = regs[(k0 >> 16) & 0xff], t = regs[k0 >> 24];
	vm_reg		za = regs[(k1 >> 20) & 0x1f], zb = regs[(k1 >> 25) & 0x1f];
	int			fa = (k0 & SOP_GDW_FUSED_A) != 0, fb = (k0 & SOP_GDW_FUSED_B) != 0;
	uint32_t	sa = k1 & 0xfffff, sb = sa;
	uint64_t	ax = (uint64_t) a.v, ay = (uint64_t) ya.v, az = (uint64_t) za.v;
	uint64_t	bx = (uint64_t) b.v, by = (uint64_t) yb.v, bz = (uint64_t) zb.v;
	uint64_t	tol, dx, dy, dz;
	int			ea = 0, eb = 0, hza = 1, hzb = 1;

	if (!fa && gdw3_load(str_ptr(env, a.v), str_len(a.v), &sa, &ax, &ay, &az, &hza, &ea))
		return SOP_RECHECK_CODE;
	if (!fb && gdw3_load(str_ptr(env, b.v), str_len(b.v), &sb, &bx, &by, &bz, &hzb, &eb))
		return SOP_RECHECK_CODE;
	if (sa != sb)
		return SOP_RECHECK_CODE;
	if (ea || eb || !hza || !hzb)
		return SOP_RECHECK_CODE;
	tol = (uint64_t) t.v;
	if (F64_NONFINITE(tol) || ((tol >> 63) != 0 && (tol & F64_MAG) != 0))
		return SOP_RECHECK_CODE;
	if (F64_NONFINITE(ax) || F64_NONFINITE(ay) || F64_NONFINITE(az) ||
		F64_NONFINITE(bx) || F64_NONFINITE(by) || F64_NONFINITE(bz))
		return SOP_RECHECK_CODE;
	dx = f64_sub(ax, bx);
	dy = f64_sub(ay, by);
	dz = f64_sub(az, bz);
	if (F64_NONFINITE(dx) || F64_NONFINITE(dy) || F64_NONFINITE(dz))
		return SOP_RECHECK_CODE;
	*res = 0;
	if (((dx | dy | dz) & F64_MAG) == 0)
		*res = 1;
	else if (k1 & SOP_GDW_EXACT)
	{
		uint64_t	q = f64_sqr_add(dz, f64_sqr_add(dx, f64_mul(dy, dy)));

		if (F64_NONFINITE(q))
			return SOP_RECHECK_CODE;
		if (f64_lt(0x47EFFFFFE0000000ull, f64_sqrt(q)))
			return SOP_RECHECK_CODE;		/* beyond FLT_MAX, PostGIS' initial distance */
		*res = f64_le(f64_sqrt(q), tol);
	}
	else
	{
		uint64_t	d2 = f64_add(f64_add(f64_mul(dx, dx), f64_mul(dy, dy)), f64_mul(dz, dz));
		uint64_t	r2 = f64_mul(tol, tol);

		if (F64_NONFINITE(d2) || F64_NONFINITE(r2) || (d2 & F64_MAG) < GDW_TINY ||
			((r2 & F64_MAG) != 0 && (r2 & F64_MAG) < GDW_TINY))
			return SOP_RECHECK_CODE;
		if (f64_le(d2, f64_mul(r2, GDW_LO)))
			*res = 1;
		else if (f64_lt(f64_mul(r2, GDW_HI), d2))
			*res = 0;
		else
			return SOP_RECHECK_CODE;
	}
	return 0;
}

/* the instructions that A6 added; the same as vm_ext() of scan_common.slang */
static uint32_t
vm_ext(const vm_env *env, uint32_t op, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1,
	   vm_reg *regs)
{
	switch (op)
	{
		case SOP_GEOMPIP:
		{
			vm_reg		x = regs[ra], y = regs[rb];
			int			res = 0;
			uint32_t	rc;

			if (x.isnull || y.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (k0 & SOP_GPIP_FUSED)
			{
				vm_reg		yy = regs[k1 & 0xff];

				if (yy.isnull)
				{
					regs[dst].isnull = 1;
					return 0;
				}
				rc = geom_pip(str_ptr(env, x.v), str_len(x.v), NULL, 0, 1, (uint64_t) y.v, (uint64_t) yy.v, k1 >> 8,
							  k0, &res);
			}
			else
				rc = geom_pip(str_ptr(env, x.v), str_len(x.v), str_ptr(env, y.v), str_len(y.v), 0, 0, 0, 0, k0, &res);
			if (rc != 0)
				return rc;
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
		case SOP_GEOMZM:
		{
			vm_reg		a = regs[ra];
			int			isnull = 0;
			uint64_t	val = 0;
			uint32_t	rc;

			if (a.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			rc = geom_zm(str_ptr(env, a.v), str_len(a.v), k0, &isnull, &val);
			if (rc != 0)
				return rc;
			regs[dst].isnull = isnull;
			if (!isnull)
				regs[dst].v = (int64_t) val;
			return 0;
		}
		case SOP_GEOMLEN:
		{
			vm_reg		a = regs[ra];
			uint64_t	val = 0;
			uint32_t	rc;

			if (a.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			rc = geom_len(str_ptr(env, a.v), str_len(a.v), k0, &val);
			if (rc != 0)
				return rc;
			regs[dst].isnull = 0;
			regs[dst].v = (int64_t) val;
			return 0;
		}
		case SOP_GEOMDIST:
		{
			vm_reg		a = regs[ra], b = regs[rb];
			int			isnull = 0, np = 0, d3 = (k1 & SOP_GDIST_3D) != 0;
			uint64_t	val = 0;
			uint32_t	rc;

			if (a.isnull || b.isnull ||
				((k0 & SOP_GDW_FUSED_A) && (regs[(k0 >> 8) & 0xff].isnull || (d3 && regs[(k1 >> 20) & 0x1f].isnull))) ||
				((k0 & SOP_GDW_FUSED_B) && (regs[(k0 >> 16) & 0xff].isnull || (d3 && regs[(k1 >> 25) & 0x1f].isnull))))
			{
				regs[dst].isnull = 1;
				return 0;
			}
			rc = geom_dwdist_np(env, regs, SOP_GEOMDIST, ra, rb, k0, k1, &np, &isnull, &val);
			if (rc == 0 && !np)
				rc = geom_dist(env, regs, ra, rb, k0, k1, &isnull, &val);
			if (rc != 0)
				return rc;
			regs[dst].isnull = isnull;
			if (!isnull)
				regs[dst].v = (int64_t) val;
			return 0;
		}
		case SOP_GEOMFLAG:
		{
			vm_reg		a = regs[ra];
			int64_t		val = 0;
			uint32_t	rc;

			if (a.isnull)
			{
				regs[dst].isnull = 1;
				return 0;
			}
			if (k0 >= SOP_GFLAG_NPOINTS)
			{
				int			isnull = 0;

				rc = geom_cnt(str_ptr(env, a.v), str_len(a.v), k0, &isnull, &val);
				if (rc != 0)
					return rc;
				regs[dst].isnull = isnull;
				if (!isnull)
					regs[dst].v = val;
				return 0;
			}
			rc = geom_flag(str_ptr(env, a.v), str_len(a.v), k0, &val);
			if (rc != 0)
				return rc;
			regs[dst].isnull = 0;
			regs[dst].v = val;
			return 0;
		}
		case SOP_GEOM3DDW:
		{
			vm_reg		a = regs[ra], b = regs[rb];
			int			res = 0, np = 0, isnull = 0;
			uint64_t	val = 0;
			uint32_t	rc;

			if (a.isnull || b.isnull || regs[k0 >> 24].isnull ||
				((k0 & SOP_GDW_FUSED_A) && (regs[(k0 >> 8) & 0xff].isnull || regs[(k1 >> 20) & 0x1f].isnull)) ||
				((k0 & SOP_GDW_FUSED_B) && (regs[(k0 >> 16) & 0xff].isnull || regs[(k1 >> 25) & 0x1f].isnull)))
			{
				regs[dst].isnull = 1;
				return 0;
			}
			rc = geom_dwdist_np(env, regs, SOP_GEOM3DDW, ra, rb, k0, k1, &np, &isnull, &val);
			if (rc == 0 && np)
				res = (int) val;
			else if (rc == 0)
				rc = geom_3ddwithin(env, regs, ra, rb, k0, k1, &res);
			if (rc != 0)
				return rc;
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
		case SOP_NULLMRG:
			regs[dst].v = regs[ra].v;
			regs[dst].isnull = (regs[ra].isnull || regs[rb].isnull);
			return 0;
		case SOP_GEOMDWITHIN:
		{
			vm_reg		a = regs[ra], b = regs[rb];
			int			res = 0, np = 0, isnull = 0;
			uint64_t	val = 0;
			uint32_t	rc;

			if (a.isnull || b.isnull || regs[k0 >> 24].isnull ||
				((k0 & SOP_GDW_FUSED_A) && regs[(k0 >> 8) & 0xff].isnull) ||
				((k0 & SOP_GDW_FUSED_B) && regs[(k0 >> 16) & 0xff].isnull))
			{
				regs[dst].isnull = 1;
				return 0;
			}
			rc = geom_dwdist_np(env, regs, SOP_GEOMDWITHIN, ra, rb, k0, k1, &np, &isnull, &val);
			if (rc == 0 && np)
				res = (int) val;
			else if (rc == 0)
				rc = geom_dwithin(env, regs, ra, rb, k0, k1, &res);
			if (rc != 0)
				return rc;
			regs[dst].isnull = 0;
			regs[dst].v = res;
			return 0;
		}
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
			a = geom_box2d(str_ptr(env, x.v), str_len(x.v), k0 == SOP_GBOX_OVERLAPS_ND);
			b = geom_box2d(str_ptr(env, y.v), str_len(y.v), k0 == SOP_GBOX_OVERLAPS_ND);
			if (a.st == 1 || b.st == 1)
				res = (k0 != SOP_GBOX_OVERLAPS && k0 != SOP_GBOX_OVERLAPS_ND);	/* an empty operand: && and &&& false, ~ and @ true */
			else if (a.st == 2 || b.st == 2)
				return SOP_RECHECK_CODE;
			else
			{
				if (k0 == SOP_GBOX_OVERLAPS || k0 == SOP_GBOX_OVERLAPS_ND)
				{
					res = gf_le(a.b[0], b.b[1]) && gf_le(b.b[0], a.b[1]) && gf_le(a.b[2], b.b[3]) && gf_le(b.b[2], a.b[3]);
					/* &&& compares a dimension only when both operands have it */
					if (k0 == SOP_GBOX_OVERLAPS_ND && res && a.hz && b.hz)
						res = gf_le(a.z[0], b.z[1]) && gf_le(b.z[0], a.z[1]);
					if (k0 == SOP_GBOX_OVERLAPS_ND && res && a.hm && b.hm)
						res = gf_le(a.m[0], b.m[1]) && gf_le(b.m[0], a.m[1]);
				}
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
				{
					uint32_t	e = f64_math_err(k0, (uint64_t) x.v);

					if (e != 0)
						return k1 | (e << 16);
					regs[dst].v = (int64_t) f64_math(k0, (uint64_t) x.v);
				}
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
