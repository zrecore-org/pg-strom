/*
 * vagg_check.c: kernels_vm/vagg.inc (PostgreSQL's float8_accum / float8_combine / float8_regr_accum / float8_regr_combine in the
 * soft binary64) against the installed PostgreSQL. Reads the cases that test/gpuserv/vagg_check.sh generated with PostgreSQL's
 * own functions (lines "V|xs|state|stateA|stateB|combined" and "R|ys|xs|state|stateA|stateB|combined": every double as the 16 hex
 * digits of its bits, lists separated by commas, "ERR" where PostgreSQL raised an error) and compares bit for bit (any NaN
 * equals any NaN). Exit status 0: everything matched.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define F64SOFT_C 1
#include "kernels_vm/f64_soft.inc"
#include "kernels_vm/vagg.inc"

#define MAXV 256

static int
parse_list(const char *s, uint64_t *out)
{
	int			n = 0;

	if (!*s)
		return 0;
	while (*s)
	{
		out[n++] = strtoull(s, NULL, 16);
		s = strchr(s, ',');
		if (!s)
			break;
		s++;
	}
	return n;
}

static int
same(uint64_t a, uint64_t b)
{
	if ((a & F64_MAG) > F64_INF && (b & F64_MAG) > F64_INF)
		return 1;
	return a == b;
}

static double
bits2d(uint64_t b)
{
	double		d;

	memcpy(&d, &b, 8);
	return d;
}

/* the state array {N, Sx, ...} against the accumulator */
static int
vsame(const VAcc *a, const uint64_t *st)
{
	return (double) a->n == bits2d(st[0]) && same(a->sx, st[1]) && same(a->sxx, st[2]);
}

static int
rsame(const RAcc *a, const uint64_t *st)
{
	return (double) a->n == bits2d(st[0]) && same(a->sx, st[1]) && same(a->sxx, st[2]) &&
		same(a->sy, st[3]) && same(a->syy, st[4]) && same(a->sxy, st[5]);
}

static VAcc
vfrom(const uint64_t *st)
{
	VAcc		a;

	a.n = (uint64_t) bits2d(st[0]);
	a.sx = st[1];
	a.sxx = st[2];
	a.err = 0;
	return a;
}

static RAcc
rfrom(const uint64_t *st)
{
	RAcc		a;

	a.n = (uint64_t) bits2d(st[0]);
	a.sx = st[1];
	a.sxx = st[2];
	a.sy = st[3];
	a.syy = st[4];
	a.sxy = st[5];
	a.err = 0;
	return a;
}

int
main(int argc, char **argv)
{
	FILE	   *fp = argc > 1 ? fopen(argv[1], "r") : stdin;
	static char line[1 << 20];
	long		ncase = 0, nbad = 0, nerr = 0, ncomb = 0;

	if (!fp)
	{
		perror("vagg_check: open");
		return 2;
	}
	while (fgets(line, sizeof(line), fp))
	{
		char	   *f[8];
		int			nf = 0;
		char	   *p = line;
		uint64_t	xs[MAXV], ys[MAXV], st[8], sa[8], sb[8], sc[8];
		int			nx = 0, ny = 0, isr, bad = 0, half;
		int			err_all, err_a, err_b, err_c;

		line[strcspn(line, "\r\n")] = 0;
		while (nf < 8)
		{
			f[nf++] = p;
			p = strchr(p, '|');
			if (!p)
				break;
			*p++ = 0;
		}
		isr = (f[0][0] == 'R');
		if (nf < (isr ? 7 : 6))
			continue;
		{
			int			o = 1;

			nx = parse_list(f[o++], xs);
			if (isr)
				ny = parse_list(f[o++], ys);
			err_all = !strcmp(f[o], "ERR");
			if (!err_all)
				parse_list(f[o], st);
			o++;
			err_a = !strcmp(f[o], "ERR");
			if (!err_a)
				parse_list(f[o], sa);
			o++;
			err_b = !strcmp(f[o], "ERR");
			if (!err_b)
				parse_list(f[o], sb);
			o++;
			err_c = !strcmp(f[o], "ERR");
			if (!err_c)
				parse_list(f[o], sc);
		}
		half = nx / 2;
		ncase++;
		if (!isr)
		{
			VAcc		a = vacc_init(), b = vacc_init(), c = vacc_init();

			for (int i = 0; i < nx; i++)
			{
				c = vacc_add(c, xs[i]);
				if (i < half)
					a = vacc_add(a, xs[i]);
				else
					b = vacc_add(b, xs[i]);
			}
			if (err_all != (int) c.err || (!err_all && !vsame(&c, st)))
				bad |= 1;
			if (err_a != (int) a.err || (!err_a && !vsame(&a, sa)))
				bad |= 2;
			if (err_b != (int) b.err || (!err_b && !vsame(&b, sb)))
				bad |= 4;
			if (!err_a && !err_b)
			{
				VAcc		m = vacc_merge(vfrom(sa), vfrom(sb));

				ncomb++;
				if (err_c != (int) m.err || (!err_c && !vsame(&m, sc)))
				{
					bad |= 8;
					if (getenv("VAGG_VERBOSE"))
						fprintf(stderr, "  merge: mine %016llx %016llx ref %016llx %016llx\n", (unsigned long long) m.sx, (unsigned long long) m.sxx, (unsigned long long) sc[1], (unsigned long long) sc[2]);
				}
			}
			nerr += err_all;
		}
		else
		{
			RAcc		a = racc_init(), b = racc_init(), c = racc_init();

			for (int i = 0; i < nx; i++)
			{
				c = racc_add(c, ys[i], xs[i]);
				if (i < half)
					a = racc_add(a, ys[i], xs[i]);
				else
					b = racc_add(b, ys[i], xs[i]);
			}
			if (err_all != (int) c.err || (!err_all && !rsame(&c, st)))
				bad |= 1;
			if (err_a != (int) a.err || (!err_a && !rsame(&a, sa)))
				bad |= 2;
			if (err_b != (int) b.err || (!err_b && !rsame(&b, sb)))
				bad |= 4;
			if (!err_a && !err_b)
			{
				RAcc		m = racc_merge(rfrom(sa), rfrom(sb));

				ncomb++;
				if (err_c != (int) m.err || (!err_c && !rsame(&m, sc)))
					bad |= 8;
			}
			nerr += err_all;
		}
		if (bad)
		{
			nbad++;
			if (nbad <= 5)
			{
				fprintf(stderr, "MISMATCH case %ld (%s, %d rows) parts %d\n", ncase, isr ? "regr" : "var", nx, bad);
				if (getenv("VAGG_VERBOSE"))
				{
					fprintf(stderr, "  all  ref %s: %016llx %016llx %016llx\n", err_all ? "ERR" : "ok ", (unsigned long long) st[0], (unsigned long long) st[1], (unsigned long long) st[2]);
				}
			}
		}
	}
	printf("vagg_check: %ld cases (%ld merges, %ld with an overflow error), %ld mismatches\n", ncase, ncomb, nerr, nbad);
	return nbad ? 1 : 0;
}
