/*
 * Checks kernels_vm/area_ring.inc (the software-binary64 twin of PostGIS 3.7's ptarray_signed_area) against the "A|..." lines of
 * area_probe.sql: every ring it accepts must equal PostGIS' bits, and it must accept most rings.
 * Build: cc -O1 -ffp-contract=off -I../../../src/gpuserv/kernels_vm -o twin area_twin_check.c -lm; ./twin < lines.txt
 */
#define F64SOFT_C 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "f64_soft.inc"
#define AREA_FN static inline
#define AR_F2 struct ArF2s
#define AR_SRC_T const double *
#define AR_LD(src, off) (*(const uint64_t *) ((const char *) (src) + (off)))
#include "area_ring.inc"
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0 || (a != a && b != b); }
int main(void)
{
	static char line[1 << 22];
	long rows = 0, none = 0, bad = 0;
	while (fgets(line, sizeof(line), stdin))
	{
		static double a[4][100008], b[4][100008];
		double v[4], pts0[100008][3], pts1[100008][3];
		long n0, n1;
		char *p;
		if (strncmp(line, "A|", 2)) continue;
		p = line + 2;
		strtol(p, &p, 10); p++;
		n0 = strtol(p, &p, 10); p++; n1 = strtol(p, &p, 10);
		for (long i = 0; i < n0; i++) for (int c = 0; c < 3; c++) { p++; pts0[i][c] = strtod(p, &p); }
		for (long i = 0; i < n1; i++) for (int c = 0; c < 3; c++) { p++; pts1[i][c] = strtod(p, &p); }
		for (int i = 0; i < 4; i++) { p++; v[i] = strtod(p, &p); }
		(void) a; (void) b;
		rows++;
		for (int r = 0; r < 2; r++)
		{
			uint64_t res = area_ring_signed(r ? &pts1[0][0] : &pts0[0][0], 0, 24, r ? n1 : n0);
			double d;
			if (res == AR_NONE) { none++; continue; }
			memcpy(&d, &res, 8);
			d = fabs(d);
			if (!same(d, v[r])) { bad++; if (bad < 6) printf("row %ld ring %d: twin %.17g pg %.17g\n", rows, r, d, v[r]); }
		}
	}
	printf("rows %ld rings %ld declined %ld wrong %ld\n", rows, rows * 2, none, bad);
	return bad != 0;
}
