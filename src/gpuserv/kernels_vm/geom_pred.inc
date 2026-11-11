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
#define GPM_CONTAINS   0u       // the relation asked: Contains(a, b), Intersects, and the DE-9IM relations of R6
#define GPM_INTERSECTS 1u
#define GPM_TOUCHES    2u
#define GPM_CROSSES    3u
#define GPM_OVERLAPS   4u
#define GPM_EQUALS     5u
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
GP_FN GP_T(GpGeo) gp_head(GP_SRC_T src, uint base, uint len, uint pts)
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
	if (pts != 0u)
	{
		// a point or multipoint: first = the first element (type word, point count, coordinates), ncomp = the number of elements
		if (g.gtype != 1u && g.gtype != 4u)
			return g;
	}
	else if (g.gtype != 2u && g.gtype != 3u && g.gtype != 5u && g.gtype != 6u)
		return g;
	if (g.gtype == 1u || g.gtype == 2u || g.gtype == 3u)
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

// validates the geometry (components, counts, offsets, closed rings of at least 4 points, lines of at least 2, finite X and Y, not all points equal) and measures it
GP_FN GP_T(GpInfo) gp_prep(GP_SRC_T src, GP_T(GpGeo) g)
{
	GP_T(GpInfo) f;
	uint		o = g.first;
	uint		want = g.gtype;
	bool		have = false;

	if (g.gtype == 5u)
		want = 2u;
	if (g.gtype == 6u)
		want = 3u;
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
			bool		moved = false;

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
				if (!f64_eq(x, fx) || !f64_eq(y, fy))
					moved = true;
				if (c.type == 3u && i == n - 1u && (!f64_eq(x, fx) || !f64_eq(y, fy)))
					return f;				// an open ring
			}
			if (!moved)
				return f;					// a zero-length line or ring: the CPU decides
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

// the boundaries against each other: 0 certainly apart, 1 a certain crossing, 2 not certain. full = 0: the first certain crossing ends the scan
// (answer 1, an unsure pair seen before it is not reported); full = 1: the first unsure pair ends it (answer 2), a crossing is answered 1 only
// when the whole scan found no unsure pair
GP_NOINL GP_FN uint gp_boundary(GP_SRC_T sa, GP_T(GpGeo) A, GP_SRC_T sb, GP_T(GpGeo) B, uint full)
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
							{
								if (full == 0u)
									return 1u;
								res = 1u;
							}
							if (code == 2u)
							{
								if (full != 0u)
									return 2u;
								res = 2u;
							}
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

// bit set of what was seen over the roots of an operand
#define GPR_NONE   0u
#define GPR_IN     1u
#define GPR_OUT    2u
#define GPR_UNSURE 4u

// the bit of gp_roots for a point-in-area answer
GP_FN uint gp_seenbit(uint rr)
{
	if (rr == GPP_IN)
		return GPR_IN;
	if (rr == GPP_OUT)
		return GPR_OUT;
	return GPR_UNSURE;
}

// the first vertex of every component of `from` against the area `area`: GP_ROOT_IN when one is inside, GP_ROOT_OUT when one is outside and
// none inside would matter to the caller (it asks `want_out`), 2 when one is not certain
GP_NOINL GP_FN uint gp_roots(GP_SRC_T sf, GP_T(GpGeo) F, GP_SRC_T sa, GP_T(GpGeo) A)
{
	uint		seen = 0u;
	uint		o = F.first;

	for (uint ci = 0u; ci < F.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(sf, F.end, F.stride, o);
		uint		rr = gp_pip_area(sa, A, GP_LD64(sf, c.pts), GP_LD64(sf, c.pts + 8u));

		seen |= gp_seenbit(rr);
		o = c.next;
	}
	return seen;
}

// a hole of the area A with its first vertex inside the area B: bit set like gp_roots
GP_NOINL GP_FN uint gp_holes(GP_SRC_T sa, GP_T(GpGeo) A, GP_SRC_T sb, GP_T(GpGeo) B)
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

				seen |= gp_seenbit(rr);
			}
			po += A.stride * n;
		}
		o = c.next;
	}
	return seen;
}

// ---- a point or multipoint against a line / polygon / multi-line / multi-polygon ----

// the point p against the line / polygon G: GPR_IN (inside the area), GPR_OUT (certainly not on it), GPR_UNSURE (on it, or too close to tell)
GP_FN uint gp_pt_status(GP_SRC_T sg, GP_T(GpGeo) G, uint area, uint64_t px, uint64_t py)
{
	uint		o, ci, r, i;

	if (area != 0u)
		return gp_seenbit(gp_pip_area(sg, G, px, py));
	o = G.first;
	for (ci = 0u; ci < G.ncomp; ci++)
	{
		GP_T(GpComp) c = gp_comp(sg, G.end, G.stride, o);

		for (i = 0u; i + 1u < c.n; i++)
		{
			uint64_t	x1 = GP_LD64(sg, c.pts + G.stride * i);
			uint64_t	y1 = GP_LD64(sg, c.pts + G.stride * i + 8u);
			uint64_t	x2 = GP_LD64(sg, c.pts + G.stride * (i + 1u));
			uint64_t	y2 = GP_LD64(sg, c.pts + G.stride * (i + 1u) + 8u);
			uint64_t	xlo = x1, xhi = x2, ylo = y1, yhi = y2;

			if (f64_lt(x2, x1))
			{
				xlo = x2;
				xhi = x1;
			}
			if (f64_lt(y2, y1))
			{
				ylo = y2;
				yhi = y1;
			}
			if (f64_lt(px, xlo) || f64_gt(px, xhi) || f64_lt(py, ylo) || f64_gt(py, yhi))
				continue;
			if (gp_side(x1, y1, x2, y2, px, py) == 0u)
				return GPR_UNSURE;
		}
		o = c.next;
	}
	return GPR_OUT;
}

// the elements of the point / multipoint P against G (bits of GPR_*; GPR_UNSURE | 8 = not decidable here: an empty or non-finite point, the
// work cap)
GP_FN uint gp_pt_seen(GP_SRC_T sp, GP_T(GpGeo) P, GP_SRC_T sg, GP_T(GpGeo) G, uint area, uint nsegs)
{
	uint		seen = 0u;
	uint		o = P.first;
	uint		budget = GP_MAXPAIRS;

	for (uint k = 0u; k < P.ncomp; k++)
	{
		uint		t, n;
		uint64_t	x, y;

		if (o + 8u > P.end)
			return 0x0cu;
		t = GP_LD32(sp, o);
		n = GP_LD32(sp, o + 4u);
		if (t != 1u || n != 1u || o + 8u + P.stride > P.end)
			return 0x0cu;					// an empty point or something else
		x = GP_LD64(sp, o + 8u);
		y = GP_LD64(sp, o + 16u);
		if ((x & F64_MAG) >= F64_INF || (y & F64_MAG) >= F64_INF)
			return 0x0cu;
		if (nsegs > budget)
			return 0x0cu;
		budget -= nsegs;
		seen |= gp_pt_status(sg, G, area, x, y);
		o += 8u + P.stride;
	}
	return seen;
}

// P is the (multi)point (p_is_a: the first operand), G a line / area. mode GPM_CONTAINS: Contains(a, b); GPM_INTERSECTS; other modes are the
// backend's
GP_NOINL GP_FN uint gp_pt(GP_SRC_T sp, GP_T(GpGeo) P, GP_SRC_T sg, GP_T(GpGeo) G, uint p_is_a, uint mode)
{
	GP_T(GpInfo) fg;
	uint		area, seen;

	if (mode > GPM_INTERSECTS)
		return GP_NA;
	area = (G.gtype == 3u || G.gtype == 6u) ? 1u : 0u;
	if (area != 0u && P.gtype == 1u && (G.flags & 3u) == 0u)
		return GP_NA;						// the point-in-polygon path of the VM (2D polygons, with the cached box)
	if (P.srid != G.srid)
		return GP_RECHECK;
	fg = gp_prep(sg, G);
	if (fg.ok == 0u)
		return GP_RECHECK;
	if (mode == GPM_CONTAINS && p_is_a != 0u)
	{
		// a finite set of points contains nothing that has an extent
		if (f64_eq(fg.minx, fg.maxx) && f64_eq(fg.miny, fg.maxy))
			return GP_RECHECK;
		return GP_FALSE;
	}
	seen = gp_pt_seen(sp, P, sg, G, area, fg.nsegs);
	if ((seen & 8u) != 0u)
		return GP_RECHECK;
	if (mode == GPM_INTERSECTS)
	{
		if ((seen & GPR_IN) != 0u)
			return GP_TRUE;
		return (seen & GPR_UNSURE) != 0u ? GP_RECHECK : GP_FALSE;
	}
	// Contains(G, P): every point inside or on G, one inside
	if ((seen & GPR_OUT) != 0u)
		return GP_FALSE;
	return (seen & GPR_UNSURE) != 0u ? GP_RECHECK : GP_TRUE;
}

// mode: GPM_*. GP_FALSE / GP_TRUE / GP_RECHECK, or GP_NA when an operand is not a line / polygon / multi-line / multi-polygon / point / multipoint
// the caller has a path for (the point-in-polygon path of the VM)
GP_FN uint gp_try(GP_SRC_T sa, uint ba, uint la, GP_SRC_T sb, uint bb, uint lb, uint mode)
{
	GP_T(GpGeo)	A = gp_head(sa, ba, la, 0u);
	GP_T(GpGeo)	B = gp_head(sb, bb, lb, 0u);
	GP_T(GpInfo) fa, fb;
	bool		area_a, area_b;
	uint		bc, seen;

	if (A.ok == 0u || B.ok == 0u)
	{
		if (A.ok == 0u && B.ok == 0u)
			return GP_NA;
		if (A.ok == 0u)
		{
			GP_T(GpGeo)	P = gp_head(sa, ba, la, 1u);

			if (P.ok == 0u)
				return GP_NA;
			return gp_pt(sa, P, sb, B, 1u, mode);
		}
		else
		{
			GP_T(GpGeo)	P = gp_head(sb, bb, lb, 1u);

			if (P.ok == 0u)
				return GP_NA;
			return gp_pt(sb, P, sa, A, 0u, mode);
		}
	}
	if (A.srid != B.srid)
		return GP_RECHECK;					// PostGIS raises "mixed SRID": the backend does
	fa = gp_prep(sa, A);
	fb = gp_prep(sb, B);
	if (fa.ok == 0u || fb.ok == 0u)
		return GP_RECHECK;
	area_a = (A.gtype == 3u || A.gtype == 6u);
	area_b = (B.gtype == 3u || B.gtype == 6u);
	if (mode != GPM_CONTAINS)
	{
		if (f64_lt(fa.maxx, fb.minx) || f64_lt(fb.maxx, fa.minx) || f64_lt(fa.maxy, fb.miny) || f64_lt(fb.maxy, fa.miny))
			return GP_FALSE;				// boxes apart: Intersects, and every relation that needs a common point, is false
	}
	else
	{
		if (f64_lt(fb.minx, fa.minx) || f64_gt(fb.maxx, fa.maxx) || f64_lt(fb.miny, fa.miny) || f64_gt(fb.maxy, fa.maxy))
			return GP_FALSE;
		if (!area_a)
			return GP_RECHECK;				// a line containing something: collinear overlap, the backend's
	}
	if (mode == GPM_EQUALS)
	{
		if (area_a != area_b || !f64_eq(fa.minx, fb.minx) || !f64_eq(fa.maxx, fb.maxx) ||
			!f64_eq(fa.miny, fb.miny) || !f64_eq(fa.maxy, fb.maxy))
			return GP_FALSE;				// other dimension or other envelope: not the same point set
	}
	if (mode == GPM_CROSSES && area_a && area_b)
		return GP_FALSE;					// area / area has no "crosses" (GEOS answers false)
	if (mode == GPM_OVERLAPS && area_a != area_b)
		return GP_FALSE;					// the dimensions differ
	if (fa.nsegs > GP_MAXPAIRS / fb.nsegs)
		return GP_RECHECK;
	if (mode == GPM_TOUCHES)
	{
		// a certain crossing: the interiors meet; boundaries certainly apart: disjoint or nested; either way not "touches"; the rest touches
		// or is too close to tell
		bc = gp_boundary(sa, A, sb, B, 0u);
		return bc == 2u ? GP_RECHECK : GP_FALSE;
	}
	if (mode == GPM_CROSSES)
	{
		if (area_a || area_b)
		{
			// line / area: a proper crossing of the boundary puts the line inside and outside; none and no doubt: wholly inside or outside
			bc = gp_boundary(sa, A, sb, B, 0u);
			if (bc == 1u)
				return GP_TRUE;
			if (bc == 2u || A.gtype == 5u || B.gtype == 5u || A.gtype == 6u || B.gtype == 6u)
				return GP_RECHECK;		// parts: one inside, one outside, without any crossing
			return GP_FALSE;
		}
		// line / line: crossings only at points: true; a collinear overlap anywhere makes the intersection a line
		bc = gp_boundary(sa, A, sb, B, 1u);
		if (bc == 2u)
			return GP_RECHECK;
		return bc == 1u ? GP_TRUE : GP_FALSE;
	}
	if (mode == GPM_OVERLAPS)
	{
		if (area_a)
		{
			// area / area: a proper crossing of the boundaries makes the interiors share an area that neither covers
			bc = gp_boundary(sa, A, sb, B, 0u);
			if (bc == 1u)
				return GP_TRUE;
			if (bc == 2u || A.gtype == 6u || B.gtype == 6u)
				return GP_RECHECK;		// parts or holes: the interiors can meet without a crossing
			{
				GP_T(GpComp) ca = gp_comp(sa, A.end, A.stride, A.first);
				GP_T(GpComp) cb = gp_comp(sb, B.end, B.stride, B.first);

				if (ca.ok == 0u || cb.ok == 0u || ca.n != 1u || cb.n != 1u)
					return GP_RECHECK;
			}
			return GP_FALSE;
		}
		// line / line: only an overlap of lines (collinear) counts, which is an unsure pair
		bc = gp_boundary(sa, A, sb, B, 1u);
		return bc == 2u ? GP_RECHECK : GP_FALSE;
	}
	if (mode == GPM_EQUALS)
	{
		// boundaries certainly apart, or two areas whose boundaries cross: not equal; equal geometries touch everywhere (unsure)
		bc = gp_boundary(sa, A, sb, B, 0u);
		if (bc == 0u)
			return GP_FALSE;
		if (bc == 1u && area_a)
			return GP_FALSE;
		return GP_RECHECK;
	}
	bc = gp_boundary(sa, A, sb, B, 0u);
	if (mode == GPM_INTERSECTS)
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
