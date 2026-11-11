// geom_pred.inc: R5, ST_Intersects / ST_Contains / ST_Within / ST_Covers / ST_CoveredBy of two geometries that are lines, polygons or multi-lines /
// multi-polygons (2D, Z, M or ZM: the predicates look at X and Y only), in the software binary64 of f64_soft.inc. The common subset of C and
// Slang: cpu_vm.c and scan_common.slang include this one text, so the CPU twin and the kernel cannot drift apart.
//
// PostGIS answers these with GEOS (exact for valid input). This code never tries to be exact: it decides a pair only where a sign is beyond
// the error bound of the orientation test (Shewchuk's (3 + 16 eps) eps with a safety factor of 5 = 2^-49; the same bound as pip_ring) and
// hands everything near a boundary (a touching or shared vertex or edge, a point on a line, collinear overlap) to the backend (GP_RECHECK),
// so a decided answer is the answer of the exact predicates:
//   * the segments of the boundaries (the lines, the rings) are compared pairwise: GP_CROSS (all four orientations certain and opposite),
//     none (the bounding boxes are apart, or both endpoints of one segment are certainly on one side of the other), else unsure;
//   * without a crossing and without an unsure pair the boundaries are certainly apart, so every line and every polygon shell is wholly
//     inside or wholly outside the other geometry's area, which one vertex of it (its first) tells by a certain point-in-polygon test.
// Intersects: boxes apart -> false; a crossing -> true; unsure -> recheck; else any root inside the other's area -> true, else false.
// Contains (Covers: the same answers when no boundary touches): the container must be an area; B's box not inside A's -> false; a crossing
// -> false; a root of B outside A -> false; unsure -> recheck; if B is an area, a hole of A with a vertex inside B -> false; else true.
// A polygon ring is decided by the non-zero winding number AND the crossing parity (GEOS uses the parity): where they differ the ring is not
// a simple one and the row is the backend's. Work is bounded: more than GP_MAXPAIRS segment pairs is the backend's.
//
// The includer defines GP_FN (function qualifier), GP_T(name) (the struct type: `struct name` in C), GP_SRC_T (the type of the data handle),
// GP_LD32(src, off) and GP_LD64(src, off) (a 4 / 8 byte value at an absolute byte offset). Offsets are absolute: a geometry is described by its
// first byte `base` and its `end`.

#define GP_FALSE     0u
#define GP_TRUE      1u
#define GP_RECHECK   2u
#define GP_NA        3u             // not a case of this code (the caller goes on with the point-in-polygon path)
#define GP_MAXPAIRS  16384u
#define GP_ERRBOUND  0x3CE0000000000000ull      // 2^-49

#define GPP_OUT      0u
#define GPP_IN       1u
#define GPP_UNSURE   2u

struct GpGeo
{
	uint		ok;
	uint		gtype;
	uint		srid;
	uint		flags;
	uint		stride;		// bytes per point
	uint		first;		// the offset of the first component's type word
	uint		ncomp;
	uint		end;
};

struct GpComp
{
	uint		ok;
	uint		type;		// 2 line, 3 polygon
	uint		n;			// points of a line, rings of a polygon
	uint		ringarr;	// polygon: the offset of the ring point counts
	uint		pts;		// the offset of the first point
	uint		next;		// the offset after the component
};

struct GpInfo
{
	uint		ok;
	uint		nsegs;
	uint64_t	minx;
	uint64_t	miny;
	uint64_t	maxx;
	uint64_t	maxy;
};

// the header of a serialized geometry that is a line, polygon, multilinestring or multipolygon of 2D / Z / M / ZM; ok = 0 otherwise
GP_FN GP_T(GpGeo) gp_head(GP_SRC_T src, uint base, uint len)
{
	GP_T(GpGeo)	g;
	uint		gflags, o, nd;

	g.ok = 0u;
	g.gtype = 0u;
	g.srid = 0u;
	g.flags = 0u;
	g.stride = 16u;
	g.first = 0u;
	g.ncomp = 0u;
	g.end = base + len;
	if (len < 12u)
		return g;
	gflags = GP_LD32(src, base) >> 24;
	if ((GP_LD32(src, base) & 0xe0u) != 0u)
		return g;
	g.srid = ((GP_LD32(src, base) & 0x1fu) << 16) | (((GP_LD32(src, base) >> 8) & 0xffu) << 8) | ((GP_LD32(src, base) >> 16) & 0xffu);
	g.flags = gflags;
	if ((gflags & 0x08u) != 0u)
		return g;							// geodetic
	nd = 2u + (gflags & 1u) + ((gflags >> 1) & 1u);
	g.stride = 8u * nd;
	o = 4u;
	if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
		o += 8u;
	if ((gflags & 0x04u) != 0u)
		o += 8u * nd;						// the cached float box: 2 floats per dimension
	if (o + 8u > len)
		return g;
	g.gtype = GP_LD32(src, base + o);
	if (g.gtype != 2u && g.gtype != 3u && g.gtype != 5u && g.gtype != 6u)
		return g;
	if (g.gtype == 2u || g.gtype == 3u)
	{
		g.first = base + o;
		g.ncomp = 1u;
	}
	else
	{
		g.first = base + o + 8u;
		g.ncomp = GP_LD32(src, base + o + 4u);
		if (g.ncomp == 0u || g.ncomp > (len - o - 8u) / 8u)
			return g;
	}
	g.ok = 1u;
	return g;
}

// the component whose type word is at o
GP_FN GP_T(GpComp) gp_comp(GP_SRC_T src, uint end, uint stride, uint o)
{
	GP_T(GpComp) c;

	c.ok = 0u;
	c.type = 0u;
	c.n = 0u;
	c.ringarr = 0u;
	c.pts = 0u;
	c.next = 0u;
	if (o + 8u > end)
		return c;
	c.type = GP_LD32(src, o);
	c.n = GP_LD32(src, o + 4u);
	if (c.type == 2u)
	{
		if (c.n > (end - o - 8u) / stride)
			return c;
		c.pts = o + 8u;
		c.next = c.pts + stride * c.n;
		c.ok = 1u;
	}
	else if (c.type == 3u)
	{
		uint		total = 0u;

		if (c.n > (end - o - 8u) / 4u)
			return c;
		c.ringarr = o + 8u;
		c.pts = o + 8u + 4u * c.n + ((c.n & 1u) != 0u ? 4u : 0u);
		if (c.pts > end)
			return c;
		for (uint r = 0u; r < c.n; r++)
		{
			uint		nr = GP_LD32(src, c.ringarr + 4u * r);

			if (nr > (end - c.pts) / stride - total)
				return c;
			total += nr;
		}
		c.next = c.pts + stride * total;
		c.ok = 1u;
	}
	return c;
}

// validates the geometry (components, counts, offsets, closed rings of at least 4 points, lines of at least 2, finite X and Y) and measures it
GP_FN GP_T(GpInfo) gp_prep(GP_SRC_T src, GP_T(GpGeo) g)
{
	GP_T(GpInfo) f;
	uint		o = g.first;
	uint		want = (g.gtype == 5u ? 2u : g.gtype == 6u ? 3u : g.gtype);
	bool		have = false;

	f.ok = 0u;
	f.nsegs = 0u;
	f.minx = 0ull;
	f.miny = 0ull;
	f.maxx = 0ull;
	f.maxy = 0ull;
	for (uint ci = 0u; ci < g.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(src, g.end, g.stride, o);
		uint		nr, po;

		if (c.ok == 0u || c.type != want)
			return f;
		nr = (c.type == 2u ? 1u : c.n);
		if (nr == 0u)
			return f;
		po = c.pts;
		for (uint r = 0u; r < nr; r++)
		{
			uint		n = (c.type == 2u ? c.n : GP_LD32(src, c.ringarr + 4u * r));
			uint64_t	fx = 0ull, fy = 0ull;

			if (n < (c.type == 2u ? 2u : 4u))
				return f;
			for (uint i = 0u; i < n; i++)
			{
				uint64_t	x = GP_LD64(src, po + g.stride * i);
				uint64_t	y = GP_LD64(src, po + g.stride * i + 8u);

				if ((x & F64_MAG) >= F64_INF || (y & F64_MAG) >= F64_INF)
					return f;
				if (i == 0u)
				{
					fx = x;
					fy = y;
				}
				if (!have)
				{
					f.minx = x; f.maxx = x; f.miny = y; f.maxy = y;
					have = true;
				}
				else
				{
					if (f64_lt(x, f.minx)) f.minx = x;
					if (f64_gt(x, f.maxx)) f.maxx = x;
					if (f64_lt(y, f.miny)) f.miny = y;
					if (f64_gt(y, f.maxy)) f.maxy = y;
				}
				if (c.type == 3u && i == n - 1u && (!f64_eq(x, fx) || !f64_eq(y, fy)))
					return f;				// an open ring
			}
			f.nsegs += n - 1u;
			po += g.stride * n;
		}
		o = c.next;
	}
	f.ok = 1u;
	return f;
}

// the side of p of the line a -> b: 1 left, 2 right, 0 not certain
GP_FN uint gp_side(uint64_t x1, uint64_t y1, uint64_t x2, uint64_t y2, uint64_t px, uint64_t py)
{
	uint64_t	l = f64_mul(f64_sub(px, x1), f64_sub(y2, y1));
	uint64_t	r = f64_mul(f64_sub(x2, x1), f64_sub(py, y1));
	uint64_t	det = f64_sub(r, l);
	uint64_t	s = f64_add(l & F64_MAG, r & F64_MAG);
	uint		e = (uint) ((s >> 52) & 0x7ffull);

	if (e < 0x080u || e > 0x7f0u)
		return 0u;
	if (!f64_gt(det & F64_MAG, f64_mul(s, GP_ERRBOUND)))
		return 0u;
	return (det & F64_SIGN) == 0ull ? 1u : 2u;
}

// two segments: 0 certainly apart, 1 certainly crossing (properly), 2 not certain
GP_FN uint gp_segpair(uint64_t ax1, uint64_t ay1, uint64_t ax2, uint64_t ay2, uint64_t bx1, uint64_t by1, uint64_t bx2, uint64_t by2)
{
	uint64_t	aminx = f64_lt(ax1, ax2) ? ax1 : ax2, amaxx = f64_lt(ax1, ax2) ? ax2 : ax1;
	uint64_t	aminy = f64_lt(ay1, ay2) ? ay1 : ay2, amaxy = f64_lt(ay1, ay2) ? ay2 : ay1;
	uint64_t	bminx = f64_lt(bx1, bx2) ? bx1 : bx2, bmaxx = f64_lt(bx1, bx2) ? bx2 : bx1;
	uint64_t	bminy = f64_lt(by1, by2) ? by1 : by2, bmaxy = f64_lt(by1, by2) ? by2 : by1;
	uint		o1, o2, o3, o4;

	if (f64_lt(amaxx, bminx) || f64_lt(bmaxx, aminx) || f64_lt(amaxy, bminy) || f64_lt(bmaxy, aminy))
		return 0u;
	o1 = gp_side(ax1, ay1, ax2, ay2, bx1, by1);
	o2 = gp_side(ax1, ay1, ax2, ay2, bx2, by2);
	if (o1 != 0u && o1 == o2)
		return 0u;
	o3 = gp_side(bx1, by1, bx2, by2, ax1, ay1);
	o4 = gp_side(bx1, by1, bx2, by2, ax2, ay2);
	if (o3 != 0u && o3 == o4)
		return 0u;
	if (o1 != 0u && o2 != 0u && o3 != 0u && o4 != 0u)
		return 1u;
	return 2u;
}

// the point-in-ring test of pip_ring (kernel scan_common.slang) with the crossing parity checked against the winding number
GP_FN uint gp_pip_ring(GP_SRC_T src, uint off, uint n, uint stride, uint64_t px, uint64_t py)
{
	int			wn = 0;
	uint		cnt = 0u;
	uint64_t	x1 = GP_LD64(src, off);
	uint64_t	y1 = GP_LD64(src, off + 8u);

	if (n < 4u)
		return GPP_UNSURE;
	for (uint i = 1u; i < n; i++)
	{
		uint64_t	x2 = GP_LD64(src, off + stride * i);
		uint64_t	y2 = GP_LD64(src, off + stride * i + 8u);

		if (f64_eq(y1, y2))
		{
			// horizontal (or zero-length): only a point on it matters, and that is the boundary
			if (!f64_eq(x1, x2) && f64_eq(py, y1))
			{
				bool		inside = f64_le(x1, x2) ? (f64_le(x1, px) && f64_le(px, x2)) : (f64_le(x2, px) && f64_le(px, x1));

				if (inside)
					return GPP_UNSURE;
			}
		}
		else
		{
			uint64_t	ymin = f64_lt(y1, y2) ? y1 : y2;
			uint64_t	ymax = f64_lt(y1, y2) ? y2 : y1;

			if (!(f64_lt(py, ymin) || f64_gt(py, ymax)))
			{
				uint64_t	l = f64_mul(f64_sub(px, x1), f64_sub(y2, y1));
				uint64_t	r = f64_mul(f64_sub(x2, x1), f64_sub(py, y1));
				uint64_t	det = f64_sub(r, l);
				uint64_t	s = f64_add(l & F64_MAG, r & F64_MAG);
				uint		e = (uint) ((s >> 52) & 0x7ffull);

				if (e < 0x080u || e > 0x7f0u)
					return GPP_UNSURE;
				if (!f64_gt(det & F64_MAG, f64_mul(s, GP_ERRBOUND)))
					return GPP_UNSURE;
				if (f64_le(y1, py) && f64_lt(py, y2))
				{
					if ((det & F64_SIGN) == 0ull)
					{
						wn++;
						cnt++;
					}
				}
				else if (f64_le(y2, py) && f64_lt(py, y1))
				{
					if ((det & F64_SIGN) != 0ull)
					{
						wn--;
						cnt++;
					}
				}
			}
		}
		x1 = x2;
		y1 = y2;
	}
	if ((wn != 0) != ((cnt & 1u) != 0u))
		return GPP_UNSURE;				// a self-overlapping ring: the winding rule and GEOS' parity rule disagree
	return wn != 0 ? GPP_IN : GPP_OUT;
}

// the point against one polygon component (shell, then the holes the point is inside of)
GP_FN uint gp_pip_poly(GP_SRC_T src, uint stride, GP_T(GpComp) c, uint64_t px, uint64_t py)
{
	uint		res = GPP_OUT;
	uint		po = c.pts;

	for (uint r = 0u; r < c.n; r++)
	{
		uint		n = GP_LD32(src, c.ringarr + 4u * r);

		if (res != GPP_UNSURE && (r == 0u || res == GPP_IN))
		{
			uint		rr = gp_pip_ring(src, po, n, stride, px, py);

			if (rr == GPP_UNSURE)
				res = GPP_UNSURE;
			else if (r == 0u)
				res = rr;
			else if (rr == GPP_IN)
				res = GPP_OUT;			// in a hole
		}
		po += stride * n;
	}
	return res;
}

// the point against a polygon or multipolygon: the first polygon the point is not outside of decides
GP_FN uint gp_pip_area(GP_SRC_T src, GP_T(GpGeo) g, uint64_t px, uint64_t py)
{
	uint		o = g.first;

	for (uint ci = 0u; ci < g.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(src, g.end, g.stride, o);
		uint		rr = gp_pip_poly(src, g.stride, c, px, py);

		if (rr != GPP_OUT)
			return rr;
		o = c.next;
	}
	return GPP_OUT;
}

// the boundaries against each other: 0 certainly apart, 1 a certain crossing, 2 not certain
GP_FN uint gp_boundary(GP_SRC_T sa, GP_T(GpGeo) A, GP_SRC_T sb, GP_T(GpGeo) B)
{
	uint		res = 0u;
	uint		oa = A.first;

	for (uint ca = 0u; ca < A.ncomp; ca++)
	{
		GP_T(GpComp) a = gp_comp(sa, A.end, A.stride, oa);
		uint		nra = (a.type == 2u ? 1u : a.n);
		uint		pa = a.pts;

		for (uint ra = 0u; ra < nra; ra++)
		{
			uint		na = (a.type == 2u ? a.n : GP_LD32(sa, a.ringarr + 4u * ra));

			for (uint i = 0u; i + 1u < na; i++)
			{
				uint64_t	ax1 = GP_LD64(sa, pa + A.stride * i);
				uint64_t	ay1 = GP_LD64(sa, pa + A.stride * i + 8u);
				uint64_t	ax2 = GP_LD64(sa, pa + A.stride * (i + 1u));
				uint64_t	ay2 = GP_LD64(sa, pa + A.stride * (i + 1u) + 8u);
				uint		ob = B.first;

				for (uint cb = 0u; cb < B.ncomp; cb++)
				{
					GP_T(GpComp) b = gp_comp(sb, B.end, B.stride, ob);
					uint		nrb = (b.type == 2u ? 1u : b.n);
					uint		pb = b.pts;

					for (uint rb = 0u; rb < nrb; rb++)
					{
						uint		nb = (b.type == 2u ? b.n : GP_LD32(sb, b.ringarr + 4u * rb));

						for (uint j = 0u; j + 1u < nb; j++)
						{
							uint		code = gp_segpair(ax1, ay1, ax2, ay2,
														  GP_LD64(sb, pb + B.stride * j),
														  GP_LD64(sb, pb + B.stride * j + 8u),
														  GP_LD64(sb, pb + B.stride * (j + 1u)),
														  GP_LD64(sb, pb + B.stride * (j + 1u) + 8u));

							if (code == 1u)
								return 1u;
							if (code == 2u)
								res = 2u;
						}
						pb += B.stride * nb;
					}
					ob = b.next;
				}
			}
			pa += A.stride * na;
		}
		oa = a.next;
	}
	return res;
}

// the first vertex of every component of `from` against the area `area`: GP_ROOT_IN when one is inside, GP_ROOT_OUT when one is outside and
// none inside would matter to the caller (it asks `want_out`), 2 when one is not certain
#define GPR_NONE   0u
#define GPR_IN     1u
#define GPR_OUT    2u
#define GPR_UNSURE 4u
// bit set of what was seen over the roots of `from`
GP_FN uint gp_roots(GP_SRC_T sf, GP_T(GpGeo) F, GP_SRC_T sa, GP_T(GpGeo) A)
{
	uint		seen = 0u;
	uint		o = F.first;

	for (uint ci = 0u; ci < F.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(sf, F.end, F.stride, o);
		uint		rr = gp_pip_area(sa, A, GP_LD64(sf, c.pts), GP_LD64(sf, c.pts + 8u));

		seen |= (rr == GPP_IN ? GPR_IN : rr == GPP_OUT ? GPR_OUT : GPR_UNSURE);
		o = c.next;
	}
	return seen;
}

// a hole of the area A with its first vertex inside the area B: bit set like gp_roots
GP_FN uint gp_holes(GP_SRC_T sa, GP_T(GpGeo) A, GP_SRC_T sb, GP_T(GpGeo) B)
{
	uint		seen = 0u;
	uint		o = A.first;

	for (uint ci = 0u; ci < A.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(sa, A.end, A.stride, o);
		uint		po = c.pts;

		for (uint r = 0u; r < c.n; r++)
		{
			uint		n = GP_LD32(sa, c.ringarr + 4u * r);

			if (r > 0u)
			{
				uint		rr = gp_pip_area(sb, B, GP_LD64(sa, po), GP_LD64(sa, po + 8u));

				seen |= (rr == GPP_IN ? GPR_IN : rr == GPP_OUT ? GPR_OUT : GPR_UNSURE);
			}
			po += A.stride * n;
		}
		o = c.next;
	}
	return seen;
}

// symm: Intersects(a, b); else Contains(a, b). GP_FALSE / GP_TRUE / GP_RECHECK, or GP_NA when an operand is not a line / polygon / multi-line /
// multi-polygon (the caller has other paths for points)
GP_FN uint gp_try(GP_SRC_T sa, uint ba, uint la, GP_SRC_T sb, uint bb, uint lb, uint symm)
{
	GP_T(GpGeo)	A = gp_head(sa, ba, la);
	GP_T(GpGeo)	B = gp_head(sb, bb, lb);
	GP_T(GpInfo) fa, fb;
	bool		area_a, area_b;
	uint		bc, seen;

	if (A.ok == 0u || B.ok == 0u)
	{
		// a header that was not accepted: NA when one of them is a point or a type this code does not take
		return GP_NA;
	}
	if (A.srid != B.srid)
		return GP_RECHECK;					// PostGIS raises "mixed SRID": the backend does
	fa = gp_prep(sa, A);
	fb = gp_prep(sb, B);
	if (fa.ok == 0u || fb.ok == 0u)
		return GP_RECHECK;
	area_a = (A.gtype == 3u || A.gtype == 6u);
	area_b = (B.gtype == 3u || B.gtype == 6u);
	if (symm != 0u)
	{
		if (f64_lt(fa.maxx, fb.minx) || f64_lt(fb.maxx, fa.minx) || f64_lt(fa.maxy, fb.miny) || f64_lt(fb.maxy, fa.miny))
			return GP_FALSE;
	}
	else
	{
		if (f64_lt(fb.minx, fa.minx) || f64_gt(fb.maxx, fa.maxx) || f64_lt(fb.miny, fa.miny) || f64_gt(fb.maxy, fa.maxy))
			return GP_FALSE;
		if (!area_a)
			return GP_RECHECK;				// a line containing something: collinear overlap, the backend's
	}
	if (fa.nsegs > GP_MAXPAIRS / fb.nsegs)
		return GP_RECHECK;
	bc = gp_boundary(sa, A, sb, B);
	if (symm != 0u)
	{
		if (bc == 1u)
			return GP_TRUE;
		if (bc == 2u)
			return GP_RECHECK;
		seen = 0u;
		if (area_b)
			seen |= gp_roots(sa, A, sb, B);
		if (area_a)
			seen |= gp_roots(sb, B, sa, A);
		if ((seen & GPR_IN) != 0u)
			return GP_TRUE;
		return (seen & GPR_UNSURE) != 0u ? GP_RECHECK : GP_FALSE;
	}
	if (bc == 1u)
		return GP_FALSE;
	seen = gp_roots(sb, B, sa, A);
	if ((seen & GPR_OUT) != 0u)
		return GP_FALSE;
	if (bc == 2u || (seen & GPR_UNSURE) != 0u)
		return GP_RECHECK;
	if (area_b)
	{
		seen = gp_holes(sa, A, sb, B);
		if ((seen & GPR_IN) != 0u)
			return GP_FALSE;
		if ((seen & GPR_UNSURE) != 0u)
			return GP_RECHECK;
	}
	return GP_TRUE;
}
