// df64: a double-float pair (two float32, value hi + lo, |lo| <= ulp(hi)/2) built from the error-free transformations
// TwoSum, FastTwoSum and TwoProduct (through fma). It is NOT binary64: about 48 bits of mantissa, the float32 exponent
// range, no IEEE rounding of the pair. It can only DECIDE a case early (a result with a proven error bound), never produce
// a float8 result. S0 of A10 part 6 (test/gpuserv/df64_probe.sh) measures it; S1 will use it as a prefilter.
//
// Compiled as C by the probe's host half (DF64_C: struct, fmaf, -ffp-contract=off) and as Slang by the kernels; the two
// must agree bit for bit (the probe checks that). Every intermediate is `precise` (DF_P) in Slang so that no compiler
// may reassociate, contract (a*b+c into fma) or drop the error terms.
#ifdef DF64_C
#include <math.h>
#define DF_FN static inline
#define DF_P
#define DF_FMA(a, b, c) fmaf((a), (b), (c))
#define DF_SQRT(a) sqrtf(a)
#define DF_ABS(a) fabsf(a)
typedef struct { float hi, lo; } DF;
#else
#define DF_FN
#ifdef DF_USE_PRECISE
#define DF_P precise
#else
#define DF_P
#endif
#define DF_FMA(a, b, c) fma((a), (b), (c))
#define DF_SQRT(a) sqrt(a)
#define DF_ABS(a) abs(a)
struct DF { float hi; float lo; };
#endif

DF_FN DF df_mk(float h, float l) { DF r; r.hi = h; r.lo = l; return r; }

// a + b = s + e exactly (Knuth)
DF_FN DF df_two_sum(float a, float b)
{
	DF_P float s = a + b;
	DF_P float bb = s - a;
	DF_P float e = (a - (s - bb)) + (b - bb);
	return df_mk(s, e);
}
// a + b = s + e exactly when |a| >= |b| or a == 0 (Dekker)
DF_FN DF df_quick_two_sum(float a, float b)
{
	DF_P float s = a + b;
	DF_P float e = b - (s - a);
	return df_mk(s, e);
}
// a * b = p + e exactly (barring underflow)
DF_FN DF df_two_prod(float a, float b)
{
	DF_P float p = a * b;
	DF_P float e = DF_FMA(a, b, -p);
	return df_mk(p, e);
}

DF_FN DF df_add(DF a, DF b)		// DWPlusDW (Joldes, Muller, Popescu 2017, alg. 6): relative error <= 3u^2 / (1 - 4u), u = 2^-24
{
	DF s = df_two_sum(a.hi, b.hi);
	DF t = df_two_sum(a.lo, b.lo);
	DF_P float c = s.lo + t.hi;
	DF v = df_quick_two_sum(s.hi, c);
	DF_P float w = t.lo + v.lo;
	return df_quick_two_sum(v.hi, w);
}
DF_FN DF df_neg(DF a) { return df_mk(-a.hi, -a.lo); }
DF_FN DF df_sub(DF a, DF b) { return df_add(a, df_neg(b)); }
DF_FN DF df_mul(DF a, DF b)		// DWTimesDW3 (alg. 13): relative error <= 5u^2 (7u^2 used as the bound)
{
	DF p = df_two_prod(a.hi, b.hi);
	DF_P float t0 = a.lo * b.lo;
	DF_P float t1 = DF_FMA(a.hi, b.lo, t0);
	DF_P float c2 = DF_FMA(a.lo, b.hi, t1);
	DF_P float c3 = p.lo + c2;
	return df_quick_two_sum(p.hi, c3);
}
DF_FN DF df_mul_f(DF a, float b)
{
	DF p = df_two_prod(a.hi, b);
	DF_P float c = DF_FMA(a.lo, b, p.lo);
	return df_quick_two_sum(p.hi, c);
}
DF_FN DF df_div(DF a, DF b)		// DWDivDW-style (three quotient digits)
{
	DF_P float q1 = a.hi / b.hi;
	DF r = df_sub(a, df_mul_f(b, q1));
	DF_P float q2 = r.hi / b.hi;
	r = df_sub(r, df_mul_f(b, q2));
	DF_P float q3 = r.hi / b.hi;
	DF q = df_quick_two_sum(q1, q2);
	return df_add(q, df_mk(q3, 0.0f));
}
DF_FN DF df_sqrt(DF a)			// one Newton step from the float square root (a > 0)
{
	DF_P float s = DF_SQRT(a.hi);
	DF p = df_two_prod(s, s);
	DF_P float r = ((a.hi - p.hi) - p.lo) + a.lo;
	DF_P float c = r / (s + s);
	return df_quick_two_sum(s, c);
}
// 1: a > b surely, -1: a < b surely, 0: undecided (|a - b| <= bound, bound >= 0)
DF_FN int df_cmp_bound(DF a, DF b, float bound)
{
	DF_P float d = (a.hi - b.hi) + (a.lo - b.lo);
	if (d > bound) return 1;
	if (d < -bound) return -1;
	return 0;
}

#ifndef DF64_C
// binary64 (as two words) -> df64 with integer arithmetic only: hi = the value rounded to 24 bits, lo = the rest (29 bits,
// rounded to float: relative error 2^-49 of the value). ok = false outside |x| in [2^-60, 2^60] (zero is ok) or for NaN/Inf.
DF df_from_f64bits(uint lo32, uint hi32, out bool ok)
{
	uint sgn = hi32 & 0x80000000u;
	int e = int((hi32 >> 20) & 0x7FFu);
	uint mh = (hi32 & 0xFFFFFu) | 0x100000u;			// bits 52..32 of the 53-bit mantissa
	uint top = (mh << 3) | (lo32 >> 29);				// the upper 24 bits
	int rem = int(lo32 & 0x1FFFFFFFu);					// the lower 29 bits
	ok = (e >= 1023 - 60 && e <= 1023 + 60);
	if ((hi32 & 0x7FFFFFFFu) == 0u && lo32 == 0u)
	{
		ok = true;
		return df_mk(asfloat(sgn), 0.0f);
	}
	if (rem > 0x10000000 || (rem == 0x10000000 && (top & 1u) != 0u))
	{
		top++;
		rem -= 0x20000000;
	}
	float scale = asfloat(uint(e - 1023 - 23 + 127) << 23);		// 2^(e-1023-23)
	float hi = float(top) * scale;
	float lo = (float(rem) * scale) * 1.862645149230957e-09f;		// 2^-29
	return df_mk(asfloat(asuint(hi) | sgn), asfloat(asuint(lo) ^ sgn));
}
#endif
