/*
 * Reads the "D|i|ax|ay|az|bx|by|bz|dist3d|dist2d|maxdist3d" lines of dist_probe.sql and counts, per candidate evaluation order, the
 * rows where it differs from PostGIS' answer. Build: cc -O0 -ffp-contract=off -o probe dist_probe.c -lm; ./probe < lines.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
static int same(double a, double b) { return memcmp(&a, &b, 8) == 0; }
int main(void)
{
	char line[1024];
	long n = 0, m3[8] = {0}, m2[4] = {0}, mm[1] = {0};
	while (fgets(line, sizeof(line), stdin))
	{
		double ax, ay, az, bx, by, bz, d3, d2, dm;
		long i;
		char *p;
		if (strncmp(line, "D|", 2)) continue;
		p = line + 2;
		i = strtol(p, &p, 10);
		(void) i;
		if (*p == '|') p++; ax = strtod(p, &p); if (*p == '|') p++;
		ay = strtod(p, &p); if (*p == '|') p++;
		az = strtod(p, &p); if (*p == '|') p++;
		bx = strtod(p, &p); if (*p == '|') p++;
		by = strtod(p, &p); if (*p == '|') p++;
		bz = strtod(p, &p); if (*p == '|') p++;
		d3 = strtod(p, &p); if (*p == '|') p++;
		d2 = strtod(p, &p); if (*p == '|') p++;
		dm = strtod(p, &p);
		{
			volatile double h = bx - ax, v = by - ay, z = bz - az;
			double c[8];
			c[0] = sqrt(h * h + v * v + z * z);						/* plain */
			c[1] = sqrt(fma(z, z, fma(v, v, h * h)));				/* ((h*h + v*v) + z*z) fused left to right */
			c[2] = sqrt(fma(z, z, fma(h, h, v * v)));
			c[3] = sqrt(fma(v, v, fma(h, h, z * z)));
			c[4] = sqrt(fma(h, h, fma(v, v, z * z)));
			c[5] = sqrt(fma(h, h, fma(z, z, v * v)));
			c[6] = sqrt(fma(v, v, fma(z, z, h * h)));
			c[7] = sqrt(fma(z, z, h * h + v * v));					/* only the last add fused */
			for (int k = 0; k < 8; k++) if (!same(c[k], d3)) m3[k]++;
			if (!same(c[2], dm)) mm[0]++;
			{
				double e[4];
				e[0] = sqrt(h * h + v * v);
				e[1] = sqrt(fma(v, v, h * h));
				e[2] = sqrt(fma(h, h, v * v));
				e[3] = hypot(h, v);
				for (int k = 0; k < 4; k++) if (!same(e[k], d2)) m2[k]++;
			}
		}
		n++;
	}
	printf("rows %ld\n3D: plain %ld  fma(z,z,fma(v,v,h*h)) %ld  fma(z,z,fma(h,h,v*v)) %ld  fma(v,v,fma(h,h,z*z)) %ld\n    fma(h,h,fma(v,v,z*z)) %ld  fma(h,h,fma(z,z,v*v)) %ld  fma(v,v,fma(z,z,h*h)) %ld  fma(z,z,h*h+v*v) %ld\n",
		   n, m3[0], m3[1], m3[2], m3[3], m3[4], m3[5], m3[6], m3[7]);
	printf("3DMaxDistance vs fma(z,z,fma(h,h,v*v)): %ld mismatches\n2D: plain %ld  fma(v,v,h*h) %ld  fma(h,h,v*v) %ld  hypot %ld\n", mm[0], m2[0], m2[1], m2[2], m2[3]);
	return 0;
}
