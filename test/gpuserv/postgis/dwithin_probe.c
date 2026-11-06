/*
 * How does the installed PostGIS decide ST_DWithin / ST_3DDWithin of two points? With the exact distance d (dist_probe.c: the
 * sqrt(fma(...)) value), is the decision "d <= t"? gen: reads the "D|..." lines of dist_probe.sql and writes SQL that evaluates
 * ST_DWithin / ST_3DDWithin at t = d, nextafter(d, 0) and nextafter(d, +inf) for each pair. cmp: reads the "W|..." result lines of that SQL
 * and counts the rows where the answer differs from the rule "d <= t".
 * Build: cc -O0 -ffp-contract=off -o probe dwithin_probe.c -lm
 *   ./probe gen < dump.txt > data.sql ; (run data.sql through postgis_dev.sh ADHOC, keep the W| lines) ; ./probe cmp < results.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double d3of(double ax, double ay, double az, double bx, double by, double bz)
{
	volatile double h = bx - ax, v = by - ay, z = bz - az, vv = v * v;
	return sqrt(fma(z, z, fma(h, h, vv)));
}
static double d2of(double ax, double ay, double bx, double by)
{
	volatile double h = bx - ax, v = by - ay, vv = v * v;
	return sqrt(fma(h, h, vv));
}
int main(int argc, char **argv)
{
	char line[1024];
	if (argc < 2) return 2;
	if (!strcmp(argv[1], "gen"))
	{
		long n = 0;
		printf("create extension if not exists postgis;\nset extra_float_digits = 3;\ncreate temp table w (i int, ax float8, ay float8, az float8, bx float8, by float8, bz float8, t2 float8[], t3 float8[]);\ninsert into w values\n");
		while (fgets(line, sizeof(line), stdin) && n < 40000)
		{
			double ax, ay, az, bx, by, bz, a, b, c, d;
			long i;
			char *p;
			if (strncmp(line, "D|", 2)) continue;
			p = line + 2; i = strtol(p, &p, 10); p++;
			ax = strtod(p, &p); p++; ay = strtod(p, &p); p++; az = strtod(p, &p); p++;
			bx = strtod(p, &p); p++; by = strtod(p, &p); p++; bz = strtod(p, &p);
			a = d2of(ax, ay, bx, by); b = d3of(ax, ay, az, bx, by, bz);
			c = a; d = b;
			printf("%s(%ld, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g, '{%.17g,%.17g,%.17g}', '{%.17g,%.17g,%.17g}')\n", n ? "," : "", i,
				   ax, ay, az, bx, by, bz, a, nextafter(a, 0.0), nextafter(a, INFINITY), b, nextafter(b, 0.0), nextafter(b, INFINITY));
			(void) c; (void) d;
			n++;
		}
		printf(";\nselect 'W', i,\n"
			   " ST_DWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t2[1]), ST_DWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t2[2]), ST_DWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t2[3]),\n"
			   " ST_3DDWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t3[1]), ST_3DDWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t3[2]), ST_3DDWithin(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz), t3[3]),\n"
			   " t2[1] = t2[2], t3[1] = t3[2]\nfrom w order by i;\n");
		return 0;
	}
	{
		long n = 0, bad2[3] = {0, 0, 0}, bad3[3] = {0, 0, 0};
		while (fgets(line, sizeof(line), stdin))
		{
			char f[8][16];
			long i;
			if (strncmp(line, "W|", 2)) continue;
			if (sscanf(line + 2, "%ld|%15[^|]|%15[^|]|%15[^|]|%15[^|]|%15[^|]|%15[^|]|%15[^|]|%15[^|]", &i, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]) < 9) continue;
			n++;
			/* expected for "d <= t": t = d -> true, t = nextafter(d, 0) -> false (unless d is 0 or the neighbours coincide), t = nextafter(d, inf) -> true */
			if (f[0][0] != 't') bad2[0]++;
			if (f[6][0] != 't' && f[1][0] != 'f') bad2[1]++;
			if (f[2][0] != 't') bad2[2]++;
			if (f[3][0] != 't') bad3[0]++;
			if (f[7][0] != 't' && f[4][0] != 'f') bad3[1]++;
			if (f[5][0] != 't') bad3[2]++;
		}
		printf("rows %ld\nST_DWithin   at d: %ld  below: %ld  above: %ld mismatches of the rule d <= t\nST_3DDWithin at d: %ld  below: %ld  above: %ld\n",
			   n, bad2[0], bad2[1], bad2[2], bad3[0], bad3[1], bad3[2]);
	}
	return 0;
}
