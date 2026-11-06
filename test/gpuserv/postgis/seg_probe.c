/*
 * Reads the "S|i|px|py|ax|ay|bx|by|dist" lines of seg_probe.sql. Per region of the projection (before A, after B, inside) it counts the rows where a
 * candidate formula differs from PostGIS' ST_Distance(point, segment). The vertex regions: sqrt(fma(h,h,v*v)) of the point and the end.
 * The inside: candidates built from the cross product (B-A) x (P-A) divided by |B-A|, or the foot of the perpendicular, with the different fusings.
 * Build: cc -O0 -ffp-contract=off -o probe seg_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0; }
static double vdist(double px, double py, double qx, double qy) { volatile double v2 = (qy - py) * (qy - py); double h = qx - px; return sqrt(fma(h, h, v2)); }
#define NC 128
int main(void)
{
	char line[1024];
	long n[3] = {0, 0, 0}, bad[3][NC];
	char names[NC][48];
	int nc = 0;
	memset(bad, 0, sizeof(bad));
	while (fgets(line, sizeof(line), stdin))
	{
		double px, py, ax, ay, bx, by, d;
		char *p;
		int region;
		long double bxa, bya, pxa, pya, num, den, r;
		if (strncmp(line, "S|", 2)) continue;
		p = line + 2; strtol(p, &p, 10); p++;
		px = strtod(p, &p); p++; py = strtod(p, &p); p++; ax = strtod(p, &p); p++; ay = strtod(p, &p); p++;
		bx = strtod(p, &p); p++; by = strtod(p, &p); p++; d = strtod(p, &p);
		bxa = (long double) bx - ax; bya = (long double) by - ay; pxa = (long double) px - ax; pya = (long double) py - ay;
		num = pxa * bxa + pya * bya; den = bxa * bxa + bya * bya; r = num / den;
		if (r < -1e-6L) region = 0; else if (r > 1 + 1e-6L) region = 1; else if (r > 1e-6L && r < 1 - 1e-6L) region = 2; else continue;
		n[region]++;
		{
			double c[NC];
			int k = 0;
			double Bxa = bx - ax, Bya = by - ay, Pxa = px - ax, Pya = py - ay;
			if (region == 0) { c[k] = vdist(px, py, ax, ay); strcpy(names[k], "vertex A"); k++; }
			else if (region == 1) { c[k] = vdist(px, py, bx, by); strcpy(names[k], "vertex B"); k++; }
			else
			{
				for (int cv = 0; cv < 5; cv++) for (int dv = 0; dv < 3; dv++) for (int sv = 0; sv < 3; sv++)
				{
					volatile double t, a1, a2; double cross, den2, dd;
					if (cv == 0) { t = Bya * Pxa; cross = fma(Bxa, Pya, -t); }
					else if (cv == 1) { t = Bxa * Pya; cross = fma(-Bya, Pxa, t); }
					else if (cv == 2) { a1 = Bxa * Pya; a2 = Bya * Pxa; cross = a1 - a2; }
					else if (cv == 3) { t = Pxa * Bya; cross = fma(Pya, Bxa, -t); }
					else { a1 = Pya * Bxa; a2 = Pxa * Bya; cross = a1 - a2; }
					if (dv == 0) { t = Bya * Bya; den2 = fma(Bxa, Bxa, t); } else if (dv == 1) { t = Bxa * Bxa; den2 = fma(Bya, Bya, t); }
					else { a1 = Bxa * Bxa; a2 = Bya * Bya; den2 = a1 + a2; }
					if (sv == 0) dd = fabs(cross) / sqrt(den2); else if (sv == 1) dd = sqrt((cross * cross) / den2); else dd = fabs(cross / sqrt(den2));
					if (k < NC) { c[k] = dd; snprintf(names[k], 48, "cross%d den%d form%d", cv, dv, sv); k++; }
				}
			}
			if (region == 2)
			{
				/* PostGIS' older distance2d_pt_seg: s = ((Ay - py) * (Bx - Ax) - (Ax - px) * (By - Ay)) / L2; d = |s| * sqrt(L2) */
				for (int cv = 0; cv < 3; cv++) for (int dv = 0; dv < 3; dv++) for (int ev = 0; ev < 3; ev++)
				{
					volatile double t, a1, a2; double cross, den2, den3, sv2, dd;
					double u1a = -Pya, u1b = Bxa, u2a = -Pxa, u2b = Bya;
					if (cv == 0) { t = u2a * u2b; cross = fma(u1a, u1b, -t); }
					else if (cv == 1) { t = u1a * u1b; cross = fma(-u2a, u2b, t); }
					else { a1 = u1a * u1b; a2 = u2a * u2b; cross = a1 - a2; }
					if (dv == 0) { t = Bya * Bya; den2 = fma(Bxa, Bxa, t); } else if (dv == 1) { t = Bxa * Bxa; den2 = fma(Bya, Bya, t); }
					else { a1 = Bxa * Bxa; a2 = Bya * Bya; den2 = a1 + a2; }
					if (ev == 0) { t = Bya * Bya; den3 = fma(Bxa, Bxa, t); } else if (ev == 1) { t = Bxa * Bxa; den3 = fma(Bya, Bya, t); }
					else { a1 = Bxa * Bxa; a2 = Bya * Bya; den3 = a1 + a2; }
					sv2 = cross / den2;
					dd = fabs(sv2) * sqrt(den3);
					if (k < NC) { c[k] = dd; snprintf(names[k], 48, "s: cross%d den%d sqrtden%d", cv, dv, ev); k++; }
				}
			}
			nc = k > nc ? k : nc;
			for (int q = 0; q < k; q++) if (!same(c[q], d)) bad[region][q]++;
		}
	}
	printf("rows: before A %ld, after B %ld, inside %ld\n", n[0], n[1], n[2]);
	for (int q = 0; q < 1; q++) printf("vertex A candidate: %ld mismatches; vertex B: %ld mismatches\n", bad[0][q], bad[1][q]);
	for (int q = 0; q < 72; q++) printf("inside %-24s %ld mismatches\n", names[q], bad[2][q]);
	return 0;
}
