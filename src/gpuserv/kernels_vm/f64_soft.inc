// f64_soft.inc: IEEE-754 binary64 in 64-bit integer arithmetic, for devices without fp64 (Apple GPUs).
//
// A double is a uint64_t holding its bit pattern (the VM keeps it in Reg.v). Every function takes and returns bit
// patterns, rounds to nearest with ties to even and is bit-identical to a hardware double operation, except that a NaN
// result is always the canonical quiet NaN (the payload of a NaN is not defined by PostgreSQL). No float instruction is
// used: nothing here depends on the compiler's fast-math, flush-to-zero or contraction (FMA) settings.
//
//   f64_add f64_sub f64_mul f64_div f64_sqrt
//   f64_key                         order-preserving integer key: PostgreSQL's float8 order (NaN = NaN, the largest;
//                                   -0 = +0), so the comparisons below, min/max, sort and group keys need no arithmetic
//   f64_eq f64_ne f64_lt f64_le f64_gt f64_ge     (float8_eq ... float8_ge of PostgreSQL)
//   f64_rint f64_fits_i64 f64_to_i64_rint f64_from_i64
//   f64_from_f32 f64_to_f32(mode)   mode 0: nearest even, 1: toward -inf, 2: toward +inf (the last two for PostGIS bounding boxes)
//   f64_arith f64_arith_err f64_to_f32_err      PostgreSQL's float8pl/mi/mul/div and float8 -> float4: the result and the
//                                   error (overflow, underflow, division by zero) the CPU would raise for it
//
// The source is the common subset of C and Slang, so the same text is checked against the host's hardware double in
// bulk (src/gpuserv/f64soft_check.c, which defines F64SOFT_C) and compiled into the kernels. Shift counts are always
// below the operand's width (a larger count is undefined in C and in SPIR-V).
// Internal format: sig has its leading bit at bit 62 (bits 9..0 are guard/round/sticky), the value is
// sig * 2^(e - 1023 - 62) with e the biased exponent of the leading bit.

#ifdef F64SOFT_C
#include <stdint.h>
#include <stdbool.h>
typedef uint32_t uint;
#define F64S_FN static inline
#else
#define F64S_FN
#endif

#define F64_SIGN	0x8000000000000000ull
#define F64_MAG		0x7FFFFFFFFFFFFFFFull
#define F64_INF		0x7FF0000000000000ull
#define F64_QNAN	0x7FF8000000000000ull
#define F64_FRAC	0x000FFFFFFFFFFFFFull

// number of leading zero bits of a non-zero 32-bit value
F64S_FN uint f64s_clz32(uint x)
{
	uint n = 0u;
	if ((x & 0xFFFF0000u) == 0u) { n += 16u; x <<= 16; }
	if ((x & 0xFF000000u) == 0u) { n += 8u; x <<= 8; }
	if ((x & 0xF0000000u) == 0u) { n += 4u; x <<= 4; }
	if ((x & 0xC0000000u) == 0u) { n += 2u; x <<= 2; }
	if ((x & 0x80000000u) == 0u) { n += 1u; }
	return n;
}

// number of leading zero bits of a 64-bit value (64 for zero)
F64S_FN uint f64s_clz64(uint64_t x)
{
	uint hi = (uint)(x >> 32);
	uint lo = (uint)(x & 0xFFFFFFFFull);

	if (hi != 0u)
		return f64s_clz32(hi);
	if (lo != 0u)
		return 32u + f64s_clz32(lo);
	return 64u;
}

// x >> n with the shifted-out bits ORed into bit 0 (a sticky bit), for any n >= 0
F64S_FN uint64_t f64s_shr_jam(uint64_t x, uint n)
{
	if (n == 0u)
		return x;
	if (n < 64u)
		return (x >> n) | (((x << (64u - n)) != 0ull) ? 1ull : 0ull);
	return (x != 0ull) ? 1ull : 0ull;
}

// high 64 bits of the 128-bit product (the low half is a * b)
F64S_FN uint64_t f64s_mulhi(uint64_t a, uint64_t b)
{
	uint64_t a0 = a & 0xFFFFFFFFull, a1 = a >> 32;
	uint64_t b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
	uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
	uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFull) + (p10 & 0xFFFFFFFFull);

	return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Round sig (leading bit at 62, or any non-zero value when e == 1) to a double: nearest, ties to even; subnormal
// results (e <= 0) and overflow to infinity included.
F64S_FN uint64_t f64s_round_pack(uint sign, int e, uint64_t sig)
{
	uint64_t sgn = ((uint64_t) sign) << 63;
	uint64_t r, bits;

	if (e >= 2047)
		return sgn | F64_INF;
	if (e <= 0)
	{
		sig = f64s_shr_jam(sig, (uint)(1 - e));
		e = 1;
	}
	r = (sig + 0x200ull) >> 10;
	if ((sig & 0x3FFull) == 0x200ull)
		r &= ~1ull;
	bits = (((uint64_t)(e - 1)) << 52) + r;
	if (bits >= F64_INF)
		return sgn | F64_INF;
	return sgn | bits;
}

F64S_FN uint64_t f64_add(uint64_t a, uint64_t b)
{
	uint64_t ma = a & F64_MAG, mb = b & F64_MAG, t;
	uint sa, sb;
	int ea, eb, e;
	uint64_t xa, xb, sig;

	if (ma > F64_INF || mb > F64_INF)
		return F64_QNAN;
	if (ma == F64_INF || mb == F64_INF)
	{
		if (ma == F64_INF && mb == F64_INF && ((a ^ b) & F64_SIGN) != 0ull)
			return F64_QNAN;				// inf - inf
		return (ma == F64_INF) ? a : b;
	}
	if (ma < mb)							// |a| >= |b| from here on
	{
		t = a; a = b; b = t;
		t = ma; ma = mb; mb = t;
	}
	sa = (uint)(a >> 63);
	sb = (uint)(b >> 63);
	if (mb == 0ull)
	{
		if (ma == 0ull)
			return (sa != 0u && sb != 0u) ? F64_SIGN : 0ull;		// (-0) + (-0) = -0, otherwise +0
		return a;
	}
	ea = (int)((a >> 52) & 0x7FFull);
	eb = (int)((b >> 52) & 0x7FFull);
	xa = (a & F64_FRAC) << 10;
	xb = (b & F64_FRAC) << 10;
	if (ea == 0)
		ea = 1;
	else
		xa |= 1ull << 62;
	if (eb == 0)
		eb = 1;
	else
		xb |= 1ull << 62;
	xb = f64s_shr_jam(xb, (uint)(ea - eb));
	e = ea;
	if (sa == sb)
	{
		sig = xa + xb;
		if (sig >= (1ull << 63))
		{
			sig = f64s_shr_jam(sig, 1u);
			e += 1;
		}
		return f64s_round_pack(sa, e, sig);
	}
	sig = xa - xb;
	if (sig == 0ull)
		return 0ull;						// exact cancellation is +0
	{
		uint lz = f64s_clz64(sig) - 1u;		// move the leading bit to bit 62, but keep e >= 1

		if ((int) lz >= e)
			lz = (uint)(e - 1);
		sig <<= lz;
		e -= (int) lz;
	}
	return f64s_round_pack(sa, e, sig);
}

F64S_FN uint64_t f64_sub(uint64_t a, uint64_t b)
{
	return f64_add(a, b ^ F64_SIGN);
}

// the exponent (may be <= 0 for a subnormal) of a non-zero finite value; its significand has bit 52 set
F64S_FN int f64s_unpack_exp(uint64_t a)
{
	int e = (int)((a >> 52) & 0x7FFull);

	if (e == 0)
		return 1 - (int)(f64s_clz64(a & F64_FRAC) - 11u);
	return e;
}

F64S_FN uint64_t f64s_unpack_sig(uint64_t a)
{
	uint64_t f = a & F64_FRAC;

	if (((a >> 52) & 0x7FFull) == 0ull)
		return f << (f64s_clz64(f) - 11u);
	return f | (1ull << 52);
}

F64S_FN uint64_t f64_mul(uint64_t a, uint64_t b)
{
	uint sign = (uint)((a ^ b) >> 63);
	uint64_t ma = a & F64_MAG, mb = b & F64_MAG;
	uint64_t xa, xb, hi, lo, sig;
	int e;

	if (ma > F64_INF || mb > F64_INF)
		return F64_QNAN;
	if (ma == F64_INF || mb == F64_INF)
	{
		if (ma == 0ull || mb == 0ull)
			return F64_QNAN;				// 0 * inf
		return (((uint64_t) sign) << 63) | F64_INF;
	}
	if (ma == 0ull || mb == 0ull)
		return ((uint64_t) sign) << 63;
	e = f64s_unpack_exp(a) + f64s_unpack_exp(b) - 1023;
	xa = f64s_unpack_sig(a);
	xb = f64s_unpack_sig(b);
	hi = f64s_mulhi(xa, xb);
	lo = xa * xb;							// the product is in [2^104, 2^106)
	if ((hi & (1ull << 41)) != 0ull)
	{
		sig = (hi << 21) | (lo >> 43);
		if ((lo & ((1ull << 43) - 1ull)) != 0ull)
			sig |= 1ull;
		e += 1;
	}
	else
	{
		sig = (hi << 22) | (lo >> 42);
		if ((lo & ((1ull << 42) - 1ull)) != 0ull)
			sig |= 1ull;
	}
	return f64s_round_pack(sign, e, sig);
}

F64S_FN uint64_t f64_div(uint64_t a, uint64_t b)
{
	uint sign = (uint)((a ^ b) >> 63);
	uint64_t ma = a & F64_MAG, mb = b & F64_MAG;
	uint64_t xa, xb, rem, q;
	int e;

	if (ma > F64_INF || mb > F64_INF)
		return F64_QNAN;
	if (ma == F64_INF)
		return (mb == F64_INF) ? F64_QNAN : ((((uint64_t) sign) << 63) | F64_INF);
	if (mb == F64_INF)
		return ((uint64_t) sign) << 63;
	if (mb == 0ull)
		return (ma == 0ull) ? F64_QNAN : ((((uint64_t) sign) << 63) | F64_INF);
	if (ma == 0ull)
		return ((uint64_t) sign) << 63;
	e = f64s_unpack_exp(a) - f64s_unpack_exp(b) + 1023;
	xa = f64s_unpack_sig(a);
	xb = f64s_unpack_sig(b);
	rem = xa;
	if (xa < xb)							// make the quotient of the significands [1, 2)
	{
		rem = xa << 1;
		e -= 1;
	}
	q = 0ull;
	for (int i = 0; i < 63; i++)			// restoring division: 63 quotient bits, the first is 1
	{
		q <<= 1;
		if (rem >= xb)
		{
			rem -= xb;
			q |= 1ull;
		}
		rem <<= 1;
	}
	if (rem != 0ull)
		q |= 1ull;							// sticky
	return f64s_round_pack(sign, e, q);
}

F64S_FN uint64_t f64_sqrt(uint64_t a)
{
	uint64_t ma = a & F64_MAG;
	uint64_t x, hi, q, rh, rl;
	int u, e;

	if (ma > F64_INF)
		return F64_QNAN;
	if (ma == 0ull)
		return a;							// sqrt(-0) = -0
	if ((a & F64_SIGN) != 0ull)
		return F64_QNAN;
	if (ma == F64_INF)
		return a;
	u = f64s_unpack_exp(a) - 1023;			// value = x * 2^(u - 52), x in [2^52, 2^53)
	x = f64s_unpack_sig(a);
	if ((u & 1) != 0)
	{
		x <<= 1;
		u -= 1;
	}
	e = (u >> 1) + 1023;
	hi = x << 8;							// the radicand is hi * 2^64: bits 125..64 hold x, the rest is zero
	q = 0ull;
	rh = 0ull;
	rl = 0ull;
	for (int i = 62; i >= 0; i--)			// digit by digit; the remainder needs 128 bits
	{
		uint psh = (i >= 32) ? (uint)(2 * (i - 32)) : 0u;
		uint pair = (i >= 32) ? (uint)((hi >> psh) & 3ull) : 0u;
		uint64_t t = (q << 2) | 1ull;

		rh = (rh << 2) | (rl >> 62);
		rl = (rl << 2) | (uint64_t) pair;
		if (rh != 0ull || rl >= t)
		{
			uint64_t nl = rl - t;

			if (rl < t)
				rh -= 1ull;
			rl = nl;
			q = (q << 1) | 1ull;
		}
		else
			q <<= 1;
	}
	if (rh != 0ull || rl != 0ull)
		q |= 1ull;							// sticky
	return f64s_round_pack(0u, e, q);
}

// ---- order: PostgreSQL's float8 comparison as integer comparison of keys ----

F64S_FN uint64_t f64_key(uint64_t a)
{
	uint64_t mag = a & F64_MAG;

	if (mag > F64_INF)
		return 0xFFF0000000000001ull;		// every NaN: one value above +inf
	if (mag == 0ull)
		return F64_SIGN;					// -0 = +0
	return ((a & F64_SIGN) != 0ull) ? ~a : (a | F64_SIGN);
}

F64S_FN bool f64_eq(uint64_t a, uint64_t b) { return f64_key(a) == f64_key(b); }
F64S_FN bool f64_ne(uint64_t a, uint64_t b) { return f64_key(a) != f64_key(b); }
F64S_FN bool f64_lt(uint64_t a, uint64_t b) { return f64_key(a) < f64_key(b); }
F64S_FN bool f64_le(uint64_t a, uint64_t b) { return f64_key(a) <= f64_key(b); }
F64S_FN bool f64_gt(uint64_t a, uint64_t b) { return f64_key(a) > f64_key(b); }
F64S_FN bool f64_ge(uint64_t a, uint64_t b) { return f64_key(a) >= f64_key(b); }

// ---- integer conversions ----

// rint(): round to an integral value, ties to even (the cast float8 -> int of PostgreSQL rounds this way)
F64S_FN uint64_t f64_rint(uint64_t a)
{
	uint64_t mag = a & F64_MAG, sgn = a & F64_SIGN, m, rem, half, one;
	int u;
	uint sh;

	if (mag >= 0x4330000000000000ull || mag == 0ull)
		return a;							// |x| >= 2^52 (infinity and NaN included) or zero
	u = (int)(mag >> 52) - 1023;
	if (u < -1)
		return sgn;							// |x| < 0.5
	if (u == -1)							// [0.5, 1): 0.5 rounds to 0 (even), the rest to 1
		return ((mag & F64_FRAC) == 0ull) ? sgn : (sgn | 0x3FF0000000000000ull);
	sh = (uint)(52 - u);					// 1 .. 52 fraction bits
	one = 1ull << sh;
	half = one >> 1;
	m = mag;
	rem = m & (one - 1ull);
	m -= rem;
	if (rem > half || (rem == half && (m & one) != 0ull))
		m += one;							// a carry into the exponent is correct on the bit pattern
	return sgn | m;
}

// the value rint(a) is in [-2^63, 2^63) (false for NaN and infinity)
F64S_FN bool f64_fits_i64(uint64_t a)
{
	uint64_t r = f64_rint(a);

	return f64_ge(r, 0xC3E0000000000000ull) && f64_lt(r, 0x43E0000000000000ull);	// NaN and infinity fail the upper bound
}

// rint(a) as int64 (only when f64_fits_i64(a))
F64S_FN int64_t f64_to_i64_rint(uint64_t a)
{
	uint64_t r = f64_rint(a), mag = r & F64_MAG, m, v;
	int u;

	if (mag == 0ull)
		return 0;
	u = (int)(mag >> 52) - 1023;
	if (u < 0)
		return 0;
	m = (mag & F64_FRAC) | (1ull << 52);
	v = (u >= 52) ? (m << (uint)(u - 52)) : (m >> (uint)(52 - u));
	if ((r & F64_SIGN) != 0ull)
		v = 0ull - v;
	return (int64_t) v;
}

F64S_FN uint64_t f64_from_i64(int64_t x)
{
	uint sign = (x < 0) ? 1u : 0u;
	uint64_t mag = (sign != 0u) ? (0ull - (uint64_t) x) : (uint64_t) x;
	uint lz;

	if (mag == 0ull)
		return 0ull;
	lz = f64s_clz64(mag);
	if (lz == 0u)							// only -2^63
		return f64s_round_pack(sign, 1023 + 63, f64s_shr_jam(mag, 1u));
	return f64s_round_pack(sign, 1023 + 63 - (int) lz, mag << (lz - 1u));
}

// ---- float4 <-> float8 (float4 is a 32-bit pattern in a uint) ----

F64S_FN uint64_t f64_from_f32(uint f)
{
	uint64_t sgn = ((uint64_t)(f >> 31)) << 63;
	uint e = (f >> 23) & 0xFFu;
	uint frac = f & 0x7FFFFFu;
	int ei;

	if (e == 255u)
		return (frac != 0u) ? F64_QNAN : (sgn | F64_INF);
	if (e == 0u)
	{
		uint lz;

		if (frac == 0u)
			return sgn;
		lz = f64s_clz32(frac) - 8u;			// normalize the subnormal
		frac = (frac << lz) & 0x7FFFFFu;
		ei = 1 - (int) lz;
	}
	else
		ei = (int) e;
	return sgn | (((uint64_t)(ei + 896)) << 52) | (((uint64_t) frac) << 29);
}

// mode: 0 nearest even, 1 toward -infinity, 2 toward +infinity. Overflow gives infinity, or the largest finite
// float4 for a directed rounding toward zero.
F64S_FN uint f64_to_f32(uint64_t a, uint mode)
{
	uint sign = (uint)(a >> 63);
	uint sgn = sign << 31;
	uint64_t mag = a & F64_MAG, sig, keep, rem, half;
	int e;
	bool up;

	if (mag > F64_INF)
		return 0x7FC00000u;
	if (mag == F64_INF)
		return sgn | 0x7F800000u;
	if (mag == 0ull)
		return sgn;
	sig = f64s_unpack_sig(a) << 10;
	e = f64s_unpack_exp(a) - 1023 + 127;	// float4 biased exponent of the leading bit
	if (e < 255)
	{
		uint64_t bits;

		if (e <= 0)
		{
			sig = f64s_shr_jam(sig, (uint)(1 - e));
			e = 1;
		}
		keep = sig >> 39;
		rem = sig & ((1ull << 39) - 1ull);
		half = 1ull << 38;
		if (mode == 0u)
			up = (rem > half) || (rem == half && (keep & 1ull) != 0ull);
		else if (mode == 1u)
			up = (sign != 0u) && (rem != 0ull);
		else
			up = (sign == 0u) && (rem != 0ull);
		bits = (((uint64_t)(e - 1)) << 23) + keep + (up ? 1ull : 0ull);
		if (bits < 0x7F800000ull)
			return sgn | (uint) bits;
	}
	if ((mode == 1u && sign == 0u) || (mode == 2u && sign != 0u))
		return sgn | 0x7F7FFFFFu;
	return sgn | 0x7F800000u;
}

// ---- PostgreSQL's float8 operators: result and error rules (float.c, PostgreSQL 15 - 19) ----

// error reasons returned by f64_arith_err (the same numbers as SCANVM_ERR_ZERODIV, _FLTOVER, _FLTUNDER of scan_vm_ops.inc)
#define F64_E_ZERODIV	1u
#define F64_E_OVER		3u
#define F64_E_UNDER		4u

F64S_FN bool f64_isinf(uint64_t a) { return (a & F64_MAG) == F64_INF; }
F64S_FN bool f64_iszero(uint64_t a) { return (a & F64_MAG) == 0ull; }

// op: 0 float8pl, 1 float8mi, 2 float8mul, 3 float8div (SCANVM_ARITH_PL ... SCANVM_ARITH_DIV)
F64S_FN uint64_t f64_arith(uint op, uint64_t a, uint64_t b)
{
	if (op == 0u)
		return f64_add(a, b);
	if (op == 1u)
		return f64_sub(a, b);
	if (op == 2u)
		return f64_mul(a, b);
	return f64_div(a, b);
}

// 0, or the error PostgreSQL raises for a op b = r: division by zero (not for a NaN dividend: NaN / 0 is NaN), then
// overflow (an infinite result from finite operands), then underflow (mul, div: a zero result from non-zero operands)
F64S_FN uint f64_arith_err(uint op, uint64_t a, uint64_t b, uint64_t r)
{
	if (op == 3u && f64_iszero(b) && (a & F64_MAG) <= F64_INF)
		return F64_E_ZERODIV;
	if (f64_isinf(r) && !f64_isinf(a) && !f64_isinf(b))
		return F64_E_OVER;
	if (op == 2u && f64_iszero(r) && !f64_iszero(a) && !f64_iszero(b))
		return F64_E_UNDER;
	if (op == 3u && f64_iszero(r) && !f64_iszero(a) && !f64_isinf(b))
		return F64_E_UNDER;
	return 0u;
}

// dtof(): r is f64_to_f32(a, 0)
F64S_FN uint f64_to_f32_err(uint64_t a, uint r)
{
	if ((r & 0x7FFFFFFFu) == 0x7F800000u && !f64_isinf(a))
		return F64_E_OVER;
	if ((r & 0x7FFFFFFFu) == 0u && !f64_iszero(a))
		return F64_E_UNDER;
	return 0u;
}
