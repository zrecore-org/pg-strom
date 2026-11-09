// area_ring.inc: PostGIS 3.7's ptarray_signed_area (liblwgeom/ptarray.c) in the software binary64 of f64_soft.inc, the common subset of
// C and Slang: cpu_vm.c and scan_common.slang include this one text, so the CPU twin and the kernel cannot drift apart.
//
// PostGIS 3.7 does not sum x * (y2 - y1) any more. It sums the determinants of the fan triangles (P0, P(i-1), P(i)):
// det = ax * by - ay * bx with a = P(i-1) - P0 and b = P(i) - P0 (the differences kept with their exact rounding errors axerr ...),
// adds the determinants with Kahan compensation, keeps the rounding error of every determinant (subtract tail, product tails by fma, the
// cross terms of the difference errors as a small exact expansion, summed with Kahan) in a second Kahan stream, adds the second stream to
// the first and returns -sum / 2. When the sum is small against the sum of the absolute terms (1e-12) it computes the area again with an
// exact expansion of up to 4096 terms and returns that. Terms that overflow are rescaled by a power of two.
//
// This function reproduces the common path and gives up (returns AR_NONE: the caller hands the row to the backend) for everything
// else: a non-finite coordinate, difference, product or sum, a triangle whose product exponent is beyond the point where PostGIS starts
// to rescale, and a ring whose sum is within twice the 1e-12 band of the cancellation test (PostGIS then returns the exact area; the
// extra factor only moves rows to the backend). Rows it returns are PostGIS' bits (test/gpuserv/postgis/area_probe.c).
//
// The includer defines AR_F2 (the struct type), AR_SRC_T (the type of the ring data), AR_LD(src, off) (the float8 bits at a byte offset), AREA_FN.
// area_ring_signed(src, q, stride, np) is the area (np >= 3 points from byte q, `stride` bytes apart, X then Y).

#define AR_NONE 0xFFFFFFFFFFFFFFFFull
#define AR_HALF 0x3FE0000000000000ull
#define AR_TOL  0x3D819799812DEA11ull		// 2e-12

struct ArF2s
{
	uint64_t	hi;
	uint64_t	lo;
};

AREA_FN AR_F2 ar_two_diff(uint64_t a, uint64_t b)
{
	AR_F2		r;
	uint64_t	bvirt, avirt, bround, around;

	r.hi = f64_sub(a, b);
	bvirt = f64_sub(a, r.hi);
	avirt = f64_add(r.hi, bvirt);
	bround = f64_sub(bvirt, b);
	around = f64_sub(a, avirt);
	r.lo = f64_add(around, bround);
	return r;
}

AREA_FN bool ar_finite(uint64_t a)
{
	return (a & F64_INF) != F64_INF;
}

// the biased exponent of a nonzero finite value, or 0 for zero (a subnormal counts as its frexp exponent would not matter at this size)
AREA_FN int ar_exp(uint64_t a)
{
	return (int) ((a >> 52) & 0x7FFull);
}

// Kahan step of ptarray_signed_area_add for finite operands; gives up when the sum overflows
#define AR_KAHAN(sum, comp, value) \
	{ \
		uint64_t y_ = f64_sub(value, comp); \
		uint64_t t_ = f64_add(sum, y_); \
		comp = f64_sub(f64_sub(t_, sum), y_); \
		sum = t_; \
		if (!ar_finite(t_) || !ar_finite(comp)) \
			return AR_NONE; \
	}

AREA_FN uint64_t area_ring_signed(AR_SRC_T src, uint q, uint stride, uint np)
{
	uint64_t	sum = 0ull, comp = 0ull, tsum = 0ull, tcomp = 0ull, detabs = 0ull, tailabs = 0ull;
	uint64_t	x0 = AR_LD(src, q), y0 = AR_LD(src, q + 8u);
	uint64_t	x2 = AR_LD(src, q + stride), y2 = AR_LD(src, q + stride + 8u);

	if (!ar_finite(x0) || !ar_finite(y0) || !ar_finite(x2) || !ar_finite(y2))
		return AR_NONE;
	for (uint i = 2u; i < np; i++)
	{
		uint64_t	x3 = AR_LD(src, q + i * stride), y3 = AR_LD(src, q + i * stride + 8u);
		if (!ar_finite(x3) || !ar_finite(y3))
			return AR_NONE;
		AR_F2		ax = ar_two_diff(x2, x0), ay = ar_two_diff(y2, y0);
		AR_F2		bx = ar_two_diff(x3, x0), by = ar_two_diff(y3, y0);
		if (!ar_finite(ax.hi) || !ar_finite(ay.hi) || !ar_finite(bx.hi) || !ar_finite(by.hi))
			return AR_NONE;
		// PostGIS rescales when the larger product exponent (frexp: biased - 1022) of ax*by, ay*bx exceeds 1000: give up beyond 990
		if (ar_exp(ax.hi) + ar_exp(by.hi) > 3024 || ar_exp(ay.hi) + ar_exp(bx.hi) > 3024)
			return AR_NONE;

		uint64_t	detleft = f64_mul(ax.hi, by.hi), detright = f64_mul(ay.hi, bx.hi);
		uint64_t	det = f64_sub(detleft, detright);
		if (!ar_finite(detleft) || !ar_finite(detright) || !ar_finite(det))
			return AR_NONE;

		uint64_t	b = f64_sub(det, detleft);
		uint64_t	subr = f64_sub(f64_sub(detleft, f64_sub(det, b)), f64_add(detright, b));
		uint64_t	leftr = f64_fma(ax.hi, by.hi, detleft ^ F64_SIGN);
		uint64_t	rightr = f64_fma(ay.hi, bx.hi, detright ^ F64_SIGN);
		// the products of the difference errors; the first three add, the others subtract (ptarray_signed_area_expansion_add_product)
		uint64_t	pa[6], pb[6];
		pa[0] = ax.hi; pb[0] = by.lo;
		pa[1] = ax.lo; pb[1] = by.hi;
		pa[2] = ax.lo; pb[2] = by.lo;
		pa[3] = ay.hi; pb[3] = bx.lo;
		pa[4] = ay.lo; pb[4] = bx.hi;
		pa[5] = ay.lo; pb[5] = bx.lo;

		uint64_t	terms[16];
		uint		nt = 0u;
		uint64_t	tail = 0ull;
		// the expansion residual gets: subr, leftr, -rightr, then for each product p: sign * p, sign * fma(a, b, -p)
		uint64_t	vals[15];
		vals[0] = subr;
		vals[1] = leftr;
		vals[2] = rightr ^ F64_SIGN;
		tail = f64_add(tail, subr & F64_MAG);
		tail = f64_add(tail, leftr & F64_MAG);
		tail = f64_add(tail, rightr & F64_MAG);
		for (uint k = 0u; k < 6u; k++)
		{
			uint64_t	p = f64_mul(pa[k], pb[k]);
			uint64_t	sg = (k < 3u) ? 0ull : F64_SIGN;
			uint64_t	e = f64_fma(pa[k], pb[k], p ^ F64_SIGN);
			if (!ar_finite(p) || !ar_finite(e))
				return AR_NONE;
			vals[3u + 2u * k] = p ^ sg;
			vals[4u + 2u * k] = e ^ sg;
			tail = f64_add(tail, p & F64_MAG);
		}
		// ptarray_signed_area_expansion_add, one value after the other (zeros are skipped)
		for (uint k = 0u; k < 15u; k++)
		{
			uint64_t	value = vals[k];
			uint		n = 0u;
			if ((value & F64_MAG) == 0ull)
				continue;
			if (!ar_finite(value))
				return AR_NONE;
			for (uint i2 = 0u; i2 < nt; i2++)
			{
				uint64_t	t = terms[i2];
				uint64_t	s = f64_add(value, t);
				uint64_t	bvirt = f64_sub(s, value);
				uint64_t	avirt = f64_sub(s, bvirt);
				uint64_t	err = f64_add(f64_sub(value, avirt), f64_sub(t, bvirt));
				if ((err & F64_MAG) != 0ull)
					terms[n++] = err;
				value = s;
			}
			if ((value & F64_MAG) != 0ull)
				terms[n++] = value;
			nt = n;
		}
		uint64_t	err = 0ull, ec = 0ull;
		for (uint k = 0u; k < nt; k++)
			AR_KAHAN(err, ec, terms[k])
		tailabs = f64_add(tailabs, tail);

		AR_KAHAN(sum, comp, det)
		detabs = f64_add(detabs, det & F64_MAG);
		AR_KAHAN(tsum, tcomp, err)
		x2 = x3;
		y2 = y3;
	}
	if ((tsum & F64_MAG) != 0ull)
		AR_KAHAN(sum, comp, tsum)
	if (f64_gt(tailabs, 0ull))
	{
		uint64_t	mx = f64_gt(detabs, tailabs) ? detabs : tailabs;
		if (f64_lt(sum & F64_MAG, f64_mul(mx, AR_TOL)))
			return AR_NONE;
	}
	if (!ar_finite(sum))
		return AR_NONE;
	return f64_mul(sum ^ F64_SIGN, AR_HALF);
}
