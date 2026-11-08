/*
 * Reads the "A|i|kind|px|py|ax|ay|bx|by|dist" lines of seg_axis_probe.sql (a segment A -> B that is vertical, horizontal or has a tiny slope) and counts, for
 * the rows whose foot is inside the segment, the mismatches of: the generic model (|s| * sqrt(L2), seg_probe.c), the axis-aligned distance |h| (used when
 * bxa == 0 for a vertical segment or bya == 0 for a horizontal one), and the combination "|h| when exactly axis-aligned, else the generic model".
 * Build: cc -O0 -ffp-contract=off -o probe seg_axis_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0; }
int main(void)
{
	char line[1024];
	long n[2][3] = {{0}}, gen[2][3] = {{0}}, axis[2][3] = {{0}}, comb[2][3] = {{0}};
	while (fgets(line, sizeof(line), stdin))
	{
		double px, py, ax, ay, bx, by, d, bxa, bya, pxa, pya, den, num, r, g, h, c;
		char *s, kind;
		int cls, k;
		if (strncmp(line, "A|", 2)) continue;
		s = line + 2; strtol(s, &s, 10); s++;
		kind = *s; s += 2;
		px = strtod(s, &s); s++; py = strtod(s, &s); s++; ax = strtod(s, &s); s++; ay = strtod(s, &s); s++;
		bx = strtod(s, &s); s++; by = strtod(s, &s); s++; d = strtod(s, &s);
		bxa = bx - ax; bya = by - ay; pxa = px - ax; pya = py - ay;
		{
			volatile double t1 = pya * bya, t2 = bya * bya, t3 = (ax - px) * bya;
			double nn = fma(pxa, bxa, t1), dd = fma(bxa, bxa, t2);
			num = nn; den = dd;
			if (den == 0) continue;
			r = num / den;
			if (r < 1e-6 || r > 1 - 1e-6) continue;
			g = fabs(fma(ay - py, bxa, -t3) / den) * sqrt(den);
		}
		h = kind == 'V' ? fabs(px - ax) : fabs(py - ay);
		k = kind == 'V' ? (bxa == 0 ? 0 : fabs(bxa) < 1e-9 * fabs(bya) ? 1 : 2) : (bya == 0 ? 0 : fabs(bya) < 1e-9 * fabs(bxa) ? 1 : 2);
		cls = (kind == 'V') ? 0 : 1;
		c = (bxa == 0 && kind == 'V') || (bya == 0 && kind == 'H') ? h : g;
		n[cls][k]++;
		if (!same(g, d)) gen[cls][k]++;
		if (!same(h, d)) axis[cls][k]++;
		if (!same(c, d)) comb[cls][k]++;
	}
	for (int cls = 0; cls < 2; cls++)
		for (int k = 0; k < 3; k++)
			printf("%s %s: %ld rows, generic model off %ld, |h| off %ld, combined off %ld\n", cls ? "horizontal" : "vertical",
				   k == 0 ? "exact" : k == 1 ? "tiny slope" : "sloped", n[cls][k], gen[cls][k], axis[cls][k], comb[cls][k]);
	return 0;
}
