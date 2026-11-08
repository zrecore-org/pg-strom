/*
 * Reads the "S|i|px|py|pz|ax|ay|az|bx|by|bz|dist" lines of seg3d_probe.sql. Per region of the projection (before A, after B, inside) it counts the
 * rows where a candidate formula differs from PostGIS' ST_3DDistance(point, segment). Vertex regions: sqrt(fma(dz,dz,fma(dx,dx,dy*dy))). Inside:
 * r = num / den, c = A + r (B - A) with the fusings a compiler may pick, then the same vertex distance from p to c.
 * Build: cc -O0 -ffp-contract=off -o probe seg3d_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0; }
static double vdist(double px, double py, double pz, double qx, double qy, double qz)
{
	volatile double v2 = (qy - py) * (qy - py);
	double h = qx - px, z = qz - pz;
	return sqrt(fma(z, z, fma(h, h, v2)));
}
#define NC 512
int main(void)
{
	char line[2048];
	long n[3] = {0, 0, 0}, bad[3][NC];
	char names[3][NC][64];
	int nc = 0;
	memset(bad, 0, sizeof(bad));
	while (fgets(line, sizeof(line), stdin))
	{
		double p[3], a[3], b[3], d;
		char *s;
		int region;
		long double num, den, r;
		if (!strncmp(line, "T|", 2))
		{
			/* the whole model: A for r < 0, B for r > 1, else the foot; counted apart */
			double tp[3], ta[3], tb[3], td, TB[3], TP[3], nn, dd, rr, mod;
			char *u = line + 2;
			static long tn = 0, tbad = 0, tcls[6] = {0}, tclsn[6] = {0}; long ti;
			ti = strtol(u, &u, 10); u++;
			for (int k = 0; k < 3; k++) { tp[k] = strtod(u, &u); u++; }
			for (int k = 0; k < 3; k++) { ta[k] = strtod(u, &u); u++; }
			for (int k = 0; k < 3; k++) { tb[k] = strtod(u, &u); u++; }
			td = strtod(u, &u);
			for (int q = 0; q < 3; q++) { TB[q] = tb[q] - ta[q]; TP[q] = tp[q] - ta[q]; }
			{
				volatile double t1 = TP[1] * TB[1], t2 = TB[1] * TB[1];
				nn = fma(TP[2], TB[2], fma(TP[0], TB[0], t1));
				dd = fma(TB[2], TB[2], fma(TB[0], TB[0], t2));
			}
			rr = nn / dd;
			if (TB[0] == 0 && TB[1] == 0 && TB[2] == 0) mod = vdist(tp[0], tp[1], tp[2], ta[0], ta[1], ta[2]);	/* A == B */
			else if (rr < 0) mod = vdist(tp[0], tp[1], tp[2], ta[0], ta[1], ta[2]);
			else if (rr > 1) mod = vdist(tp[0], tp[1], tp[2], tb[0], tb[1], tb[2]);
			else { double cc[3]; for (int q = 0; q < 3; q++) cc[q] = fma(rr, TB[q], ta[q]); mod = vdist(tp[0], tp[1], tp[2], cc[0], cc[1], cc[2]); }
			tn++; tclsn[ti % 6]++; if (!same(mod, td)) { tbad++; tcls[ti % 6]++; }
			if (tn == 100000 || tn == 120000) { printf("edge rows %ld, mismatches %ld; by i%%6:", tn, tbad); for (int q = 0; q < 6; q++) printf(" %ld/%ld", tcls[q], tclsn[q]); printf("\n"); }
			continue;
		}
		if (strncmp(line, "S|", 2)) continue;
		s = line + 2; strtol(s, &s, 10); s++;
		for (int k = 0; k < 3; k++) { p[k] = strtod(s, &s); s++; }
		for (int k = 0; k < 3; k++) { a[k] = strtod(s, &s); s++; }
		for (int k = 0; k < 3; k++) { b[k] = strtod(s, &s); s++; }
		d = strtod(s, &s);
		num = 0; den = 0;
		for (int k = 0; k < 3; k++) { num += ((long double) p[k] - a[k]) * ((long double) b[k] - a[k]); den += ((long double) b[k] - a[k]) * ((long double) b[k] - a[k]); }
		r = num / den;
		if (r < -1e-6L) region = 0; else if (r > 1 + 1e-6L) region = 1; else if (r > 1e-6L && r < 1 - 1e-6L) region = 2; else continue;
		n[region]++;
		{
			double c[NC];
			int k = 0;
			double B[3], P[3];
			for (int q = 0; q < 3; q++) { B[q] = b[q] - a[q]; P[q] = p[q] - a[q]; }
			if (region == 0) { c[k] = vdist(p[0], p[1], p[2], a[0], a[1], a[2]); strcpy(names[region][k], "vertex A"); k++; }
			else if (region == 1) { c[k] = vdist(p[0], p[1], p[2], b[0], b[1], b[2]); strcpy(names[region][k], "vertex B"); k++; }
			else
			{
				for (int nv = 0; nv < 6; nv++) for (int dv = 0; dv < 6; dv++) for (int cv = 0; cv < 3; cv++)
				{
					volatile double t1, t2, t3; double nn, dd, rr, cc[3], tt;
					/* a*b + c*d + e*f evaluated left to right: the first sum fused one way or the other or not at all, then the third product fused or not */
					t1 = P[0] * B[0]; t2 = P[1] * B[1]; t3 = P[2] * B[2];
					tt = (nv % 3 == 0) ? fma(P[0], B[0], t2) : (nv % 3 == 1) ? fma(P[1], B[1], t1) : t1 + t2;
					nn = (nv < 3) ? fma(P[2], B[2], tt) : tt + t3;
					t1 = B[0] * B[0]; t2 = B[1] * B[1]; t3 = B[2] * B[2];
					tt = (dv % 3 == 0) ? fma(B[0], B[0], t2) : (dv % 3 == 1) ? fma(B[1], B[1], t1) : t1 + t2;
					dd = (dv < 3) ? fma(B[2], B[2], tt) : tt + t3;
					rr = nn / dd;
					if (cv == 0) for (int q = 0; q < 3; q++) cc[q] = fma(rr, B[q], a[q]);
					else if (cv == 1) for (int q = 0; q < 3; q++) { volatile double t = rr * B[q]; cc[q] = a[q] + t; }
					else for (int q = 0; q < 3; q++) cc[q] = fma(rr, b[q] - a[q], a[q]);
					if (k < NC) { c[k] = vdist(p[0], p[1], p[2], cc[0], cc[1], cc[2]); snprintf(names[region][k], 64, "num%d den%d c%d", nv, dv, cv); k++; }
				}
			}
			if (k > nc) nc = k;
			for (int i = 0; i < k; i++) if (!same(c[i], d)) bad[region][i]++;
		}
	}
	for (int region = 0; region < 3; region++)
	{
		long best = -1;
		int bi = 0;
		printf("region %d (%s): %ld rows\n", region, region == 0 ? "before A" : region == 1 ? "after B" : "inside", n[region]);
		if (!n[region]) continue;
		for (int i = 0; i < (region < 2 ? 1 : nc); i++)
			if (best < 0 || bad[region][i] < best) { best = bad[region][i]; bi = i; }
		printf("   best: %s: %ld mismatches\n", names[region][bi], best);
		if (region == 2)
			for (int i = 0; i < nc; i++)
				if (bad[region][i] * 20 < n[region]) printf("   %-24s mismatches %ld\n", names[region][i], bad[region][i]);
	}
	return 0;
}
