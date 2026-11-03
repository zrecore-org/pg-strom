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

/* the instructions that A6 added; the same as vm_ext() of scan_common.slang */
static uint32_t
vm_ext(const vm_env *env, uint32_t op, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t k0, uint32_t k1,
	   vm_reg *regs)
{
	switch (op)
	{
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
						if (b == 0)
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
				else if (vt == SOP_VT_STR)
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
						regs[slot].v = (int64_t)((uint64_t) (off + hl) | ((uint64_t) (sz - hl) << 32));
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
