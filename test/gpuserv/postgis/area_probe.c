/*
 * Reads the "A|i|n0|n1|<n0 x,y,z>|<n1 x,y,z>|area0|area1|area01|area10" lines of area_probe.sql and counts, per candidate, the rows where it
 * differs from PostGIS: ptarray_signed_area = sum of x * (y2 - y1) over the triples, / 2, with and without a fused multiply-add; the polygon
 * area = |exterior| - |hole| ...
 * Build: cc -O0 -ffp-contract=off -o probe area_probe.c -lm; ./probe < lines.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0 || (a != a && b != b); }
static double ring(const double (*p)[3], int n, int fused)
{
	double sum = 0.0;
	if (n < 3) return 0.0;
	double x0 = p[0][0];
	for (int i = 2; i < n; i++)
	{
		volatile double x = p[i - 1][0] - x0;
		volatile double d = p[0 + (i - 2)][1] - p[i][1];		/* y2 - y1 with P1 = p[i-2], P3 = p[i] */
		if (fused) sum = fma(x, d, sum);
		else { volatile double t = x * d; sum += t; }
	}
	return sum / 2.0;
}
int main(void)
{
	static char line[1 << 20];
	long rows = 0, bad[8] = {0};
	while (fgets(line, sizeof(line), stdin))
	{
		static double a[64][3], b[64][3];
		double v[4];
		long n0, n1;
		char *p;
		if (strncmp(line, "A|", 2)) continue;
		p = line + 2;
		strtol(p, &p, 10); p++;
		n0 = strtol(p, &p, 10); p++; n1 = strtol(p, &p, 10);
		for (long i = 0; i < n0; i++) for (int c = 0; c < 3; c++) { p++; a[i][c] = strtod(p, &p); }
		for (long i = 0; i < n1; i++) for (int c = 0; c < 3; c++) { p++; b[i][c] = strtod(p, &p); }
		for (int i = 0; i < 4; i++) { p++; v[i] = strtod(p, &p); }
		rows++;
		for (int f = 0; f < 2; f++)
		{
			double r0 = ring(a, n0, f), r1 = ring(b, n1, f);
			double e0 = fabs(r0), e1 = fabs(r1);
			double p01 = 0.0, p10 = 0.0;
			p01 += e0; p01 -= e1; p10 += e1; p10 -= e0;
			if (!same(e0, v[0])) { bad[f * 4 + 0]++; if (bad[f * 4] < 4) printf("f%d row %ld n0 %ld mine %.17g pg %.17g\n", f, rows, n0, e0, v[0]); }
			if (!same(e1, v[1])) bad[f * 4 + 1]++;
			if (!same(p01, v[2])) bad[f * 4 + 2]++;
			if (!same(p10, v[3])) bad[f * 4 + 3]++;
		}
	}
	printf("rows %ld\nplain: ext %ld ring1 %ld poly(0,h1) %ld poly(1,h0) %ld\nfused: ext %ld ring1 %ld poly(0,h1) %ld poly(1,h0) %ld\n", rows,
		   bad[0], bad[1], bad[2], bad[3], bad[4], bad[5], bad[6], bad[7]);
	return 0;
}
