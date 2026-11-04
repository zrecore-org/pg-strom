/*
 * FMA probe: does PostGIS's ST_Contains(polygon, point) agree with a plain double evaluation of lw_segment_side,
 * or with the fused evaluation its machine code uses?  gen: writes points near the edge AB of a triangle.
 * cmp: reads "id|result" lines from the real PostGIS and counts, per model of the orientation test, the points
 * where the model's ptarray_contains_point_partial disagrees; and checks a floating-point filter.
 * Build: cc -O0 -ffp-contract=off -o probe probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct { double x, y; } P2;
static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double urand(void) { return (double)(rnd() >> 11) / 9007199254740992.0; }

/* the triangle: A, B random with full mantissas, C far away on one side */
static P2 TRI[4];
static void mktri(uint64_t seed)
{
	rng ^= seed * 0x9E3779B97F4A7C15ULL;
	for (int i = 0; i < 8; i++) rnd();
	TRI[0].x = urand() * 2000 - 1000; TRI[0].y = urand() * 2000 - 1000;
	TRI[1].x = urand() * 2000 - 1000; TRI[1].y = urand() * 2000 - 1000;
	TRI[2].x = (TRI[0].x + TRI[1].x) / 2 + (TRI[1].y - TRI[0].y) * 400.0;	/* far, on the left of A->B */
	TRI[2].y = (TRI[0].y + TRI[1].y) / 2 - (TRI[1].x - TRI[0].x) * 400.0;
	TRI[3] = TRI[0];
}

typedef double (*sidefn)(const P2 *p1, const P2 *p2, const P2 *q);
/* PostGIS source: (q->x - p1->x) * (p2->y - p1->y) - (p2->x - p1->x) * (q->y - p1->y) */
static double side_plain(const P2 *p1, const P2 *p2, const P2 *q)
{
	volatile double dx1 = q->x - p1->x, dy2 = p2->y - p1->y, dx2 = p2->x - p1->x, dy1 = q->y - p1->y;
	volatile double a = dx1 * dy2, b = dx2 * dy1;
	return a - b;
}
/* the machine code of the installed PostGIS: fmadd(dx1, dy2, -(dx2 * (p1.y - q.y)))  */
static double side_fma1(const P2 *p1, const P2 *p2, const P2 *q)
{
	volatile double dx1 = q->x - p1->x, dy2 = p2->y - p1->y, dx2 = p2->x - p1->x, ndy1 = p1->y - q->y;
	volatile double b = dx2 * ndy1;
	return fma(dx1, dy2, b);
}
static double side_fma2(const P2 *p1, const P2 *p2, const P2 *q)
{
	volatile double dx1 = q->x - p1->x, dy2 = p2->y - p1->y, dx2 = p2->x - p1->x, dy1 = q->y - p1->y;
	volatile double a = dx1 * dy2;
	return fma(-dx2, dy1, a);
}

static int pt_in_seg(const P2 *P, const P2 *A1, const P2 *A2)
{
	return ((A1->x <= P->x && P->x < A2->x) || (A1->x >= P->x && P->x > A2->x)) ||
		   ((A1->y <= P->y && P->y < A2->y) || (A1->y >= P->y && P->y > A2->y));
}
static int sgn(double v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

/* ptarray_contains_point_partial: 0 outside, 1 inside, -1 boundary */
static int contains_partial(sidefn side, const P2 *ring, int n, const P2 *pt)
{
	int wn = 0;
	const P2 *seg1 = &ring[0], *seg2;
	for (int i = 0; i < n - 1; i++)
	{
		seg2 = &ring[i + 1];
		if (seg1->x == seg2->x && seg1->y == seg2->y) { seg1 = seg2; continue; }
		double ymin = fmin(seg1->y, seg2->y), ymax = fmax(seg1->y, seg2->y);
		if (pt->y > ymax || pt->y < ymin) { seg1 = seg2; continue; }
		int s = sgn(side(seg1, seg2, pt));
		if (s == 0 && pt_in_seg(pt, seg1, seg2)) return -1;
		if (s < 0 && seg1->y <= pt->y && pt->y < seg2->y) wn++;
		else if (s > 0 && seg2->y <= pt->y && pt->y < seg1->y) wn--;
		seg1 = seg2;
	}
	return wn == 0 ? 0 : 1;
}

/* a Shewchuk-style filter: the sign is certain if |det| >= 2 * errbound (the doubled bound covers the fused variants) */
static int certain_side(const P2 *p1, const P2 *p2, const P2 *q)
{
	volatile double dx1 = q->x - p1->x, dy2 = p2->y - p1->y, dx2 = p2->x - p1->x, dy1 = q->y - p1->y;
	volatile double a = dx1 * dy2, b = dx2 * dy1;
	volatile double det = a - b;
	const double eps = 1.1102230246251565e-16;	/* 2^-53 */
	double bound = (3.0 + 16.0 * eps) * eps * (fabs(a) + fabs(b));
	return fabs(det) >= 2.0 * bound;
}

int main(int argc, char **argv)
{
	if (argc >= 4 && !strcmp(argv[1], "gen"))
	{
		long n = atol(argv[2]);
		mktri(atoll(argv[3]));
		printf("# A=%.17g,%.17g B=%.17g,%.17g C=%.17g,%.17g\n", TRI[0].x, TRI[0].y, TRI[1].x, TRI[1].y, TRI[2].x, TRI[2].y);
		for (long i = 1; i <= n; i++)
		{
			double t = 0.02 + 0.96 * urand();
			P2 q = { TRI[0].x + t * (TRI[1].x - TRI[0].x), TRI[0].y + t * (TRI[1].y - TRI[0].y) };
			int kx = (int)(rnd() % 13) - 6, ky = (int)(rnd() % 13) - 6;
			for (int k = 0; k < abs(kx); k++) q.x = nextafter(q.x, kx > 0 ? INFINITY : -INFINITY);
			for (int k = 0; k < abs(ky); k++) q.y = nextafter(q.y, ky > 0 ? INFINITY : -INFINITY);
			printf("%ld,%.17g,%.17g\n", i, q.x, q.y);
		}
		return 0;
	}
	if (argc >= 5 && !strcmp(argv[1], "cmp"))
	{
		/* cmp <seed> <points.csv> <postgis result file> */
		mktri(atoll(argv[2]));
		FILE *fp = fopen(argv[3], "r"), *fr = fopen(argv[4], "r");
		char line[256], rl[64];
		long n = 0, bad[3] = {0, 0, 0}, uncertain = 0, certain_bad[3] = {0, 0, 0}, differ01 = 0, differ02 = 0, contained = 0;
		sidefn fns[3] = { side_plain, side_fma1, side_fma2 };
		while (fgets(line, sizeof(line), fp))
		{
			long id; P2 q; int obs;
			if (line[0] == '#') continue;
			if (sscanf(line, "%ld,%lf,%lf", &id, &q.x, &q.y) != 3) continue;
			if (!fgets(rl, sizeof(rl), fr)) break;
			{ long rid; if (sscanf(rl, "%ld|%d", &rid, &obs) != 2 || rid != id) { fprintf(stderr, "result file out of step at %ld\n", id); return 2; } }
			int pred[3], cert = 1;
			for (int k = 0; k < 3; k++) pred[k] = (contains_partial(fns[k], TRI, 4, &q) == 1);
			for (int e = 0; e < 3; e++) if (!(q.y > fmax(TRI[e].y, TRI[e+1].y) || q.y < fmin(TRI[e].y, TRI[e+1].y)) && !certain_side(&TRI[e], &TRI[e+1], &q)) cert = 0;
			if (pred[0] != pred[1]) differ01++;
			if (pred[0] != pred[2]) differ02++;
			if (!cert) uncertain++;
			contained += obs;
			for (int k = 0; k < 3; k++) { if (pred[k] != obs) { bad[k]++; if (cert) certain_bad[k]++; } }
			n++;
		}
		printf("points %ld, PostGIS said contained for %ld\n", n, contained);
		printf("model plain (separate roundings):  %ld mismatches with PostGIS (%ld among filter-certain points)\n", bad[0], certain_bad[0]);
		printf("model fma1 (the installed machine code): %ld mismatches (%ld among certain)\n", bad[1], certain_bad[1]);
		printf("model fma2 (the other fusion):       %ld mismatches (%ld among certain)\n", bad[2], certain_bad[2]);
		printf("plain vs fma1 disagree on %ld points, plain vs fma2 on %ld; the filter calls %ld points uncertain (%.4f%%)\n",
			   differ01, differ02, uncertain, n ? 100.0 * uncertain / n : 0.0);
		return 0;
	}
	if (argc >= 4 && !strcmp(argv[1], "unc"))
	{
		/* the share of random points the filter cannot decide: uniform in the box of A and B */
		long n = atol(argv[2]), unc = 0, inrange = 0;
		mktri(atoll(argv[3]));
		double x0 = fmin(TRI[0].x, TRI[1].x), x1 = fmax(TRI[0].x, TRI[1].x), y0 = fmin(TRI[0].y, TRI[1].y), y1 = fmax(TRI[0].y, TRI[1].y);
		for (long i = 0; i < n; i++)
		{
			P2 q = { x0 + (x1 - x0) * urand(), y0 + (y1 - y0) * urand() };
			int cert = 1, any = 0;
			for (int e = 0; e < 3; e++)
				if (!(q.y > fmax(TRI[e].y, TRI[e+1].y) || q.y < fmin(TRI[e].y, TRI[e+1].y)))
				{
					any = 1;
					if (!certain_side(&TRI[e], &TRI[e+1], &q)) cert = 0;
				}
			inrange += any;
			if (!cert) unc++;
		}
		printf("%ld random points, %ld uncertain (%.8f%%)\n", n, unc, 100.0 * unc / n);
		return 0;
	}
	fprintf(stderr, "usage: probe gen N seed | probe cmp seed points.csv results | probe unc N seed\n");
	return 1;
}
