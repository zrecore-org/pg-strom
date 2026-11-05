/*
 * Reads the "L|i|n|x|y|z|...|len3d|len2d|len2d_of_2dline|len3d_of_2dline" lines of len_probe.sql and counts, per candidate, the rows where
 * it differs from PostGIS: the segment length sqrt(fma(z,z,fma(h,h,v*v))) / sqrt(fma(h,h,v*v)), summed left to right.
 * Build: cc -O0 -ffp-contract=off -o probe len_probe.c -lm; ./probe < lines.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0; }
int main(void)
{
	static char line[1 << 16];
	long rows = 0, bad3 = 0, bad2 = 0, bad2d = 0, bad3of2d = 0, badplain3 = 0, badplain2 = 0;
	while (fgets(line, sizeof(line), stdin))
	{
		double x[16], y[16], z[16], l3, l2, l2of2d, l3of2d, s3 = 0, s2 = 0, p3 = 0, p2 = 0, s2d = 0;
		long n;
		char *p;
		if (strncmp(line, "L|", 2)) continue;
		p = line + 2;
		strtol(p, &p, 10); p++;
		n = strtol(p, &p, 10);
		if (n > 16) continue;
		for (long i = 0; i < n; i++)
		{
			p++; x[i] = strtod(p, &p); p++; y[i] = strtod(p, &p); p++; z[i] = strtod(p, &p);
		}
		p++; l3 = strtod(p, &p); p++; l2 = strtod(p, &p); p++; l2of2d = strtod(p, &p); p++; l3of2d = strtod(p, &p);
		for (long i = 0; i + 1 < n; i++)
		{
			volatile double h = x[i + 1] - x[i], v = y[i + 1] - y[i], w = z[i + 1] - z[i];
			volatile double vv = v * v;
			s3 += sqrt(fma(w, w, fma(h, h, vv)));
			s2 += sqrt(fma(h, h, vv));
			p3 += sqrt(h * h + v * v + w * w);
			p2 += sqrt(h * h + v * v);
		}
		s2d = s2;
		rows++;
		if (!same(s3, l3)) bad3++;
		if (!same(s2, l2)) bad2++;
		if (!same(s2d, l2of2d)) bad2d++;
		if (!same(s2d, l3of2d)) bad3of2d++;
		if (!same(p3, l3)) badplain3++;
		if (!same(p2, l2)) badplain2++;
	}
	printf("rows %ld\nST_3DLength  fma order: %ld mismatches (plain: %ld)\nST_Length    fma order: %ld mismatches (plain: %ld)\n"
		   "ST_Length of a 2D line: %ld mismatches; ST_3DLength of a 2D line vs the 2D formula: %ld mismatches\n",
		   rows, bad3, badplain3, bad2, badplain2, bad2d, bad3of2d);
	return 0;
}
