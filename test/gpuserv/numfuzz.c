/*
 * numfuzz.c: differential check of the numeric helpers of scan_caps.c (the C copy of the VM's numeric
 * division, remainder, round/trunc, typmod coercion, ceil/floor/sign) against PostgreSQL's own results.
 * Input (stdin): rows "a|b|n|p|s|div|mod|divtrunc|round|trunc|ceil|floor|sign|typmod" as produced by
 * numfuzz.sql (a result is "ERR" if PostgreSQL raised an error). Output: mismatches and totals; exit 1
 * on a mismatch. A result the helper hands to the backend ("recheck") is not a mismatch (counted).
 *
 *   cc -O2 -I src -I src/gpuserv -o /tmp/numfuzz test/gpuserv/numfuzz.c src/gpuserv/scan_caps.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "xpu_common.h"
#include "scan_caps.h"

static int
parse(const char *s, int64_t *v, uint32_t *sc)
{
	int			neg = 0;
	unsigned __int128 acc = 0;
	int			dot = -1, nd = 0;

	if (*s == '-') { neg = 1; s++; }
	for (; *s; s++)
	{
		if (*s == '.') { dot = nd; continue; }
		if (*s < '0' || *s > '9')
			return 0;
		acc = acc * 10 + (*s - '0');
		nd++;
		if (acc > (unsigned __int128) INT64_MAX)
			return 0;
	}
	*sc = (dot < 0) ? 0 : (uint32_t)(nd - dot);
	*v = neg ? -(int64_t) acc : (int64_t) acc;
	return 1;
}

static void
fmt(char *buf, int64_t v, uint32_t sc)
{
	char		d[600];
	int			neg = v < 0, n;
	uint64_t	m = neg ? (uint64_t) -v : (uint64_t) v;

	n = snprintf(d, sizeof(d), "%llu", (unsigned long long) m);
	if ((uint32_t) n <= sc)
	{
		memmove(d + (sc + 1 - n), d, n + 1);
		memset(d, '0', sc + 1 - n);
		n = sc + 1;
	}
	if (sc == 0)
		sprintf(buf, "%s%s", neg ? "-" : "", d);
	else
		sprintf(buf, "%s%.*s.%s", neg ? "-" : "", n - (int) sc, d, d + n - sc);
}

static long nchecked, nrecheck, nbad;

static void
check(const char *what, const char *line, int ok, int64_t r, uint32_t rsc, const char *expect)
{
	char		got[700];

	if (!ok)
	{
		nrecheck++;
		return;
	}
	fmt(got, r, rsc);
	nchecked++;
	if (strcmp(expect, "ERR") == 0)
	{
		nbad++;
		printf("MISMATCH %s: helper gives %s, PostgreSQL raises an error: %s\n", what, got, line);
	}
	else if (strcmp(got, expect) != 0)
	{
		nbad++;
		printf("MISMATCH %s: helper %s, PostgreSQL %s: %s\n", what, got, expect, line);
	}
}

int
main(void)
{
	char		line[4096];

	while (fgets(line, sizeof(line), stdin))
	{
		char	   *f[16], *p = line;
		int			nf = 0;
		int64_t		a, b, r;
		uint32_t	sa, sb, rsc;
		long		n, prec, s;
		char		orig[4096];

		line[strcspn(line, "\n")] = 0;
		strcpy(orig, line);
		while (nf < 14)
		{
			f[nf++] = p;
			p = strchr(p, '|');
			if (!p)
				break;
			*p++ = 0;
		}
		if (nf < 14)
			continue;
		if (!parse(f[0], &a, &sa) || !parse(f[1], &b, &sb))
			continue;
		n = atol(f[2]);
		prec = atol(f[3]);
		s = atol(f[4]);
		check("div", orig, scan_vm_num_div(SOP_NUMDIV_DIV, a, sa, b, sb, &r, &rsc), r, rsc, f[5]);
		check("mod", orig, scan_vm_num_div(SOP_NUMDIV_MOD, a, sa, b, sb, &r, &rsc), r, rsc, f[6]);
		check("div()", orig, scan_vm_num_div(SOP_NUMDIV_TRUNC, a, sa, b, sb, &r, &rsc), r, rsc, f[7]);
		check("round", orig, scan_vm_num_round(SOP_NUMRND_ROUND, a, sa, n, &r, &rsc), r, rsc, f[8]);
		check("trunc", orig, scan_vm_num_round(SOP_NUMRND_TRUNC, a, sa, n, &r, &rsc), r, rsc, f[9]);
		check("ceil", orig, scan_vm_num_unary(SOP_NUMOP_CEIL, a, sa, &r, &rsc), r, rsc, f[10]);
		check("floor", orig, scan_vm_num_unary(SOP_NUMOP_FLOOR, a, sa, &r, &rsc), r, rsc, f[11]);
		check("sign", orig, scan_vm_num_unary(SOP_NUMOP_SIGN, a, sa, &r, &rsc), r, rsc, f[12]);
		check("typmod", orig, scan_vm_num_round(SOP_NUMRND_TYPMOD, a, sa, (prec << 16) + s + 4, &r, &rsc), r, rsc, f[13]);
	}
	printf("checked %ld results, %ld handed to the backend, %ld mismatches\n", nchecked, nrecheck, nbad);
	return nbad ? 1 : 0;
}
