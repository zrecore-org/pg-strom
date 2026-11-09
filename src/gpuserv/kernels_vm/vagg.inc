// vagg.inc: the transition and combine functions of PostgreSQL's float8 variance / stddev / covariance / regression aggregates
// (float8_accum, float8_combine, float8_regr_accum, float8_regr_combine of src/backend/utils/adt/float.c) on binary64 bit patterns,
// in the soft binary64 of f64_soft.inc (so a kernel and the service compute exactly what the CPU does for the same sequence of
// calls). Common subset of C and Slang, like f64_soft.inc; checked against the installed PostgreSQL by
// test/gpuserv/vagg_check.sh (src/gpuserv/vagg_check.c, which defines F64SOFT_C).
//
// The state is PostgreSQL's own, one double per field (N is kept as an integer count here):
//   VAcc  (float8_accum, 3 doubles)       n, sx = Sx, sxx = Sxx
//   RAcc  (float8_regr_accum, 6 doubles)  n, sx, sxx, sy, syy, sxy; the arguments are (Y, X) as in regr_slope(Y, X)
// *.err is set when PostgreSQL would raise "value out of range: overflow" (a non-finite result of finite inputs).
//
// FUSED MULTIPLY-ADD. PostgreSQL's source is `tmp = newval * N - Sx`, `Sxx += tmpX * tmpX * scale` ...; the installed binary
// (Apple clang, arm64, contraction on by default; checked with a disassembly of float8_accum / float8_regr_accum in
// PostgreSQL 17.x from Homebrew) fuses exactly these into fnmsub / fmadd: tmp = fma(newval, N, -Sx), Sxx = fma(tmp * tmp, scale,
// Sxx) (the products tmp * tmp, tmpX * tmpY are rounded first), and fdiv for `tmp * tmp / (N * old N)` (not fused). float8_combine
// and float8_regr_combine have no fused operation. This code follows that binary through f64_fma. A PostgreSQL built without
// contraction (another compiler or -ffp-contract=off) rounds tmp once more and differs in the last bit: vagg_check.sh detects it.

#ifndef VAGG_INC
#define VAGG_INC

#define VAGG_ONE		0x3FF0000000000000ull

#define VACC_FIELDS		uint64_t n; uint64_t sx; uint64_t sxx; uint err;
#define RACC_FIELDS		uint64_t n; uint64_t sx; uint64_t sxx; uint64_t sy; uint64_t syy; uint64_t sxy; uint err;
#ifdef F64SOFT_C
typedef struct { VACC_FIELDS } VAcc;
typedef struct { RACC_FIELDS } RAcc;
#else
struct VAcc { VACC_FIELDS };
struct RAcc { RACC_FIELDS };
#endif

// The kernels call the expensive operations (the 63-step division, the 128-bit fused multiply-add) through ONE real function each
// ([noinline]): a dozen inlined copies made the join kernels 1 MB bigger (Metal compiles those for minutes, see the plan's A13).
#ifdef F64SOFT_C
#define VG_DIV(a, b)		f64_div(a, b)
#define VG_FMA(a, b, c)		f64_fma(a, b, c)
#define VG_MUL(a, b)		f64_mul(a, b)
#define VG_ADD(a, b)		f64_add(a, b)
#else
[noinline] uint64_t vg_div(uint64_t a, uint64_t b) { return f64_div(a, b); }
[noinline] uint64_t vg_fma(uint64_t a, uint64_t b, uint64_t c) { return f64_fma(a, b, c); }
[noinline] uint64_t vg_mul(uint64_t a, uint64_t b) { return f64_mul(a, b); }
[noinline] uint64_t vg_add(uint64_t a, uint64_t b) { return f64_add(a, b); }
#define VG_DIV(a, b)		vg_div(a, b)
#define VG_FMA(a, b, c)		vg_fma(a, b, c)
#define VG_MUL(a, b)		vg_mul(a, b)
#define VG_ADD(a, b)		vg_add(a, b)
#endif

F64S_FN bool vagg_isnan(uint64_t a) { return (a & F64_MAG) > F64_INF; }
F64S_FN bool vagg_nonfinite(uint64_t a) { return (a & F64_MAG) >= F64_INF; }
F64S_FN uint64_t vagg_dbl(uint64_t n) { return f64_from_i64((int64_t) n); }

F64S_FN VAcc vacc_init()
{
	VAcc r;

	r.n = 0ull;
	r.sx = 0ull;
	r.sxx = 0ull;
	r.err = 0u;
	return r;
}

F64S_FN RAcc racc_init()
{
	RAcc r;

	r.n = 0ull;
	r.sx = 0ull;
	r.sxx = 0ull;
	r.sy = 0ull;
	r.syy = 0ull;
	r.sxy = 0ull;
	r.err = 0u;
	return r;
}

// float8_accum(state, x)
F64S_FN VAcc vacc_add(VAcc s, uint64_t x)
{
	VAcc r = s;
	uint64_t nn = vagg_dbl(s.n + 1ull);

	r.n = s.n + 1ull;
	r.sx = VG_ADD(s.sx, x);
	if (s.n > 0ull)
	{
		uint64_t tmp = VG_FMA(x, nn, r.sx ^ F64_SIGN);			// newval * N - Sx, fused
		uint64_t q = VG_DIV(VG_MUL(tmp, tmp), VG_MUL(nn, vagg_dbl(s.n)));

		r.sxx = VG_ADD(s.sxx, q);
		if (f64_isinf(r.sx) || f64_isinf(r.sxx))
		{
			if (!f64_isinf(s.sx) && !f64_isinf(x))
				r.err = 1u;
			r.sxx = F64_QNAN;
		}
	}
	else if (vagg_nonfinite(x))
		r.sxx = F64_QNAN;
	return r;
}

// float8_combine(a, b)
F64S_FN VAcc vacc_merge(VAcc a, VAcc b)
{
	VAcc r;
	uint64_t n1, n2, tmp, t;

	if (a.n == 0ull)
		return b;
	if (b.n == 0ull)
		return a;
	r.err = 0u;
	r.n = a.n + b.n;
	r.sx = VG_ADD(a.sx, b.sx);
	if (f64_arith_err(0u, a.sx, b.sx, r.sx) != 0u)
		r.err = 1u;
	n1 = vagg_dbl(a.n);
	n2 = vagg_dbl(b.n);
	tmp = VG_ADD(VG_DIV(a.sx, n1), VG_DIV(b.sx, n2) ^ F64_SIGN);
	t = VG_MUL(VG_MUL(n1, n2), tmp);
	t = VG_DIV(VG_MUL(tmp, t), vagg_dbl(r.n));
	r.sxx = VG_ADD(VG_ADD(a.sxx, b.sxx), t);
	if (f64_isinf(r.sxx) && !f64_isinf(a.sxx) && !f64_isinf(b.sxx))
		r.err = 1u;
	return r;
}

// float8_regr_accum(state, y, x)
F64S_FN RAcc racc_add(RAcc s, uint64_t y, uint64_t x)
{
	RAcc r = s;
	uint64_t nn = vagg_dbl(s.n + 1ull);

	r.n = s.n + 1ull;
	r.sx = VG_ADD(s.sx, x);
	r.sy = VG_ADD(s.sy, y);
	if (s.n > 0ull)
	{
		uint64_t tx = VG_FMA(x, nn, r.sx ^ F64_SIGN);
		uint64_t ty = VG_FMA(y, nn, r.sy ^ F64_SIGN);
		uint64_t scale = VG_DIV(VAGG_ONE, VG_MUL(nn, vagg_dbl(s.n)));

		r.sxx = VG_FMA(VG_MUL(tx, tx), scale, s.sxx);
		r.syy = VG_FMA(VG_MUL(ty, ty), scale, s.syy);
		r.sxy = VG_FMA(VG_MUL(tx, ty), scale, s.sxy);
		if (f64_isinf(r.sx) || f64_isinf(r.sxx) || f64_isinf(r.sy) || f64_isinf(r.syy) || f64_isinf(r.sxy))
		{
			if (((f64_isinf(r.sx) || f64_isinf(r.sxx)) && !f64_isinf(s.sx) && !f64_isinf(x)) ||
				((f64_isinf(r.sy) || f64_isinf(r.syy)) && !f64_isinf(s.sy) && !f64_isinf(y)) ||
				(f64_isinf(r.sxy) && !f64_isinf(s.sx) && !f64_isinf(x) && !f64_isinf(s.sy) && !f64_isinf(y)))
				r.err = 1u;
			if (f64_isinf(r.sxx))
				r.sxx = F64_QNAN;
			if (f64_isinf(r.syy))
				r.syy = F64_QNAN;
			if (f64_isinf(r.sxy))
				r.sxy = F64_QNAN;
		}
	}
	else
	{
		if (vagg_nonfinite(x))
		{
			r.sxx = F64_QNAN;
			r.sxy = F64_QNAN;
		}
		if (vagg_nonfinite(y))
		{
			r.syy = F64_QNAN;
			r.sxy = F64_QNAN;
		}
	}
	return r;
}

// float8_regr_combine(a, b)
F64S_FN RAcc racc_merge(RAcc a, RAcc b)
{
	RAcc r;
	uint64_t n1, n2, n12, nn, t1, t2, t;

	if (a.n == 0ull)
		return b;
	if (b.n == 0ull)
		return a;
	r.err = 0u;
	r.n = a.n + b.n;
	n1 = vagg_dbl(a.n);
	n2 = vagg_dbl(b.n);
	nn = vagg_dbl(r.n);
	n12 = VG_MUL(n1, n2);
	r.sx = VG_ADD(a.sx, b.sx);
	if (f64_arith_err(0u, a.sx, b.sx, r.sx) != 0u)
		r.err = 1u;
	t1 = VG_ADD(VG_DIV(a.sx, n1), VG_DIV(b.sx, n2) ^ F64_SIGN);
	t = VG_MUL(n12, t1);
	r.sxx = VG_ADD(VG_ADD(a.sxx, b.sxx), VG_DIV(VG_MUL(t1, t), nn));
	if (f64_isinf(r.sxx) && !f64_isinf(a.sxx) && !f64_isinf(b.sxx))
		r.err = 1u;
	r.sy = VG_ADD(a.sy, b.sy);
	if (f64_arith_err(0u, a.sy, b.sy, r.sy) != 0u)
		r.err = 1u;
	t2 = VG_ADD(VG_DIV(a.sy, n1), VG_DIV(b.sy, n2) ^ F64_SIGN);
	r.syy = VG_ADD(VG_ADD(a.syy, b.syy), VG_DIV(VG_MUL(t2, VG_MUL(n12, t2)), nn));
	if (f64_isinf(r.syy) && !f64_isinf(a.syy) && !f64_isinf(b.syy))
		r.err = 1u;
	r.sxy = VG_ADD(VG_ADD(a.sxy, b.sxy), VG_DIV(VG_MUL(t, t2), nn));
	if (f64_isinf(r.sxy) && !f64_isinf(a.sxy) && !f64_isinf(b.sxy))
		r.err = 1u;
	return r;
}

#endif
