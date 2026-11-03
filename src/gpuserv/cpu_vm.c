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

/* instructions [ip, end); returns 0 or the error code */
static uint32_t
vm_run(const uint32_t *insn, uint32_t ip, uint32_t end, vm_reg *regs)
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
				return op | 0x80000000u;
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

	memcpy(&infomask2, htup + 18, 2);
	memcpy(&infomask, htup + 20, 2);
	natts = infomask2 & 0x7ff;
	hasnull = infomask & 1;
	for (int i = 0; i < SCAN_MAX_REGS; i++)
	{
		regs[i].v = 0;
		regs[i].isnull = 1;
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
			off += sz;
			if (slot != 0xff)
			{
				regs[slot].v = 0;
				regs[slot].isnull = 0;
			}
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
	if (qreg != 0xffffffffu)
	{
		e = vm_run(insn, 0, nq, regs);
		if (e == 0)
			hit = (!regs[qreg].isnull && regs[qreg].v != 0);
	}
	if (e == 0 && hit && nins > nq)
		e = vm_run(insn, nq, nins, regs);
	if (e != 0)
	{
		*err = e;
		return -1;
	}
	return hit;
}
