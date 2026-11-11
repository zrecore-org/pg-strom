/*
 * df64_cover.inc -- part (c) of df64_probe: for realistic float8 qual predicates, the fraction of rows a df64 evaluation with a
 * proven error bound decides, against the rows that need the exact (soft binary64) path. Host only: the df64 operations are the
 * C compilation of kernels_vm/df64.inc, which part (a) shows bit-identical to the device. Included by df64_probe.c.
 *
 * Error model (running bounds, absolute, in double; the device would carry them in float32 with a (1 + 2^-20) margin):
 *   conversion binary64 -> df64: 2^-48 |x|                    (measured: 2^-49 |x| at most)
 *   df_add: e1 + e2 + 4 u^2 (|a| + |b|)                       (DWPlusDW: 3u^2 / (1 - 4u) |a + b|, u = 2^-24, Joldes/Muller/Popescu 2017)
 *   df_mul: |a| e2 + |b| e1 + e1 e2 + 7 u^2 |a b|              (DWTimesDW3: 5u^2)
 *   a floor of 2^-110 for the lo parts that underflow, and an operand range guard 2^-60 <= |x| <= 2^60 (outside: undecided)
 *   the CPU's own double roundings: 2^-52 per operation on the operands of that operation (the CPU computes round(round(x*a)+b))
 * A row is DECIDED when |df64 value - threshold| exceeds the sum; the decision is then the CPU's answer, which is checked
 * against a double oracle (WRONG must stay 0).
 */
typedef struct { DF v; double e; } DE;

#define U2 (1.0 / 281474976710656.0 / 2048.0 * 2048.0)	/* u^2 = 2^-48 */
static double dabs(double x) { return x < 0 ? -x : x; }
static double mag(DF a) { return dabs((double) a.hi) + dabs((double) a.lo); }

static int in_range(double x) { x = dabs(x); return x == 0 || (x >= 8.673617379884035e-19 && x <= 1.152921504606847e18); }

static DE de_conv(double x)
{
	DE r;
	double hi = (double)(float) x;
	r.v = df_mk((float) x, (float)(x - hi));
	r.e = dabs(x) * 3.552713678800501e-15;	/* 2^-48 */
	return r;
}
static DE de_add(DE a, DE b)
{
	DE r;
	r.v = df_add(a.v, b.v);
	r.e = a.e + b.e + 4.0 * U2 * (mag(a.v) + mag(b.v)) + 1e-33;
	return r;
}
static DE de_neg(DE a) { a.v = df_neg(a.v); return a; }
static DE de_mul(DE a, DE b)
{
	DE r;
	r.v = df_mul(a.v, b.v);
	r.e = mag(a.v) * b.e + mag(b.v) * a.e + a.e * b.e + 7.0 * U2 * mag(a.v) * mag(b.v) + 1e-33;
	return r;
}
/* the sign of -1 / 0 / +1 of (a - c) when certain, 2 when undecided; extra: the CPU's own rounding errors */
static int de_vs(DE a, double c, double extra)
{
	DE cc = de_conv(c);
	DF d = df_sub(a.v, cc.v);
	double dd = (double) d.hi + (double) d.lo;
	double err = a.e + cc.e + extra + 4.0 * U2 * (mag(a.v) + mag(cc.v)) + 1e-33;
	if (dabs(dd) * (1.0 - 4.76837158203125e-7) > err)
		return dd > 0 ? 1 : -1;
	return 2;
}

typedef struct { uint64_t rows, dec, wrong; } cov;

static void note(cov *c, int sure, int truth_lt)
{
	c->rows++;
	if (sure == 2) return;
	c->dec++;
	if ((sure < 0) != (truth_lt != 0)) c->wrong++;		/* sure < 0: the value is below the threshold: "<" is true */
}

/* x*a + b < c */
static void pred_A(cov *c, double x, double a, double b, double cth)
{
	volatile double p = x * a, s;
	int sure;
	s = p + b;
	if (!in_range(x) || !in_range(a) || !in_range(b)) { c->rows++; return; }
	{
		DE v = de_add(de_mul(de_conv(x), de_conv(a)), de_conv(b));
		sure = de_vs(v, cth, 4.440892098500626e-16 * (dabs(p) + dabs(s)));
	}
	note(c, sure, s < cth);
}
/* abs(x - y) < d */
static void pred_B(cov *c, double x, double y, double d)
{
	volatile double s = x - y;
	double ab = dabs(s);
	int sure;
	if (!in_range(x) || !in_range(y)) { c->rows++; return; }
	{
		DE v = de_add(de_conv(x), de_neg(de_conv(y)));
		if ((double) v.v.hi < 0 || ((double) v.v.hi == 0 && (double) v.v.lo < 0)) v = de_neg(v);
		sure = de_vs(v, d, 2.220446049250313e-16 * ab);
	}
	note(c, sure, ab < d);
}
/* x*x + y*y < r2 */
static void pred_D(cov *c, double x, double y, double r2)
{
	volatile double p = x * x, q = y * y, s;
	int sure;
	s = p + q;
	if (!in_range(x) || !in_range(y)) { c->rows++; return; }
	{
		DE X = de_conv(x), Y = de_conv(y);
		DE v = de_add(de_mul(X, X), de_mul(Y, Y));
		sure = de_vs(v, r2, 4.440892098500626e-16 * (p + q + s));
	}
	note(c, sure, s < r2);
}
/* x < c: the conversion alone */
static void pred_C(cov *c, double x, double cth)
{
	DE X = de_conv(x);
	int sure;
	if (!in_range(x) || !in_range(cth)) { c->rows++; return; }
	sure = de_vs(X, cth, 0.0);
	note(c, sure, x < cth);
}

static int cmd_cover(uint64_t n)
{
	static const struct { const char *name; double rel; int dec; } dist[] = {
		{ "uniform random", -1, 0 }, { "within 1e-3 of the threshold", 1e-3, 0 }, { "within 1e-6", 1e-6, 0 },
		{ "within 1e-9", 1e-9, 0 }, { "within 1e-12", 1e-12, 0 }, { "within 1e-14", 1e-14, 0 }, { "within 1e-16 (a few ulps)", 1e-16, 0 },
		{ "decimals k/100 (exact ties exist)", -2, 1 },
	};
	const double a = 1.7, b = -3.3, cth = 850.0, d = 50.0, r2 = 250000.0;
	int wrong = 0;

	printf("COVERAGE of a df64 prefilter with a proven error bound (rows decided by df64 / rows), %llu rows per cell\n", (unsigned long long) n);
	printf("%-36s %14s %14s %14s %14s\n", "data", "x < c", "x*a+b < c", "abs(x-y) < d", "x*x+y*y < r2");
	for (unsigned k = 0; k < sizeof(dist) / sizeof(dist[0]); k++)
	{
		cov cA = {0}, cB = {0}, cC = {0}, cD = {0};
		for (uint64_t i = 0; i < n; i++)
		{
			double x, y, rel = dist[k].rel, dl = (rndu() * 2 - 1) * (rel > 0 ? rel : 0);
			/* A */
			if (rel == -1) x = rndu() * 1000;
			else if (rel == -2) x = (double)(rnd() % 100000) / 100.0;
			else x = (cth * (1 + dl) - b) / a;
			if (rel == -2) pred_A(&cA, x, 2.0, 0.5, 100.5 + (double)(rnd() % 3) * 0.5); else pred_A(&cA, x, a, b, cth);
			/* C */
			if (rel > 0) x = cth * (1 + dl); else if (rel == -2) x = (double)(rnd() % 100000) / 100.0; else x = rndu() * 1000;
			pred_C(&cC, x, rel == -2 ? (double)(rnd() % 100000) / 100.0 : cth);
			/* B */
			y = rel == -2 ? (double)(rnd() % 100000) / 100.0 : rndu() * 1000;
			if (rel == -1) x = rndu() * 1000;
			else if (rel == -2) x = (double)(rnd() % 100000) / 100.0;
			else x = y + (rnd() & 1 ? d : -d) * (1 + dl);
			pred_B(&cB, x, y, rel == -2 ? 50.0 : d);
			/* D */
			if (rel == -1) { x = rndu() * 700; y = rndu() * 700; }
			else if (rel == -2) { x = (double)(rnd() % 50000) / 100.0; y = (double)(rnd() % 50000) / 100.0; }
			else { double th = rndu() * 6.283185307179586, r = sqrt(r2) * (1 + dl / 2); x = r * cos(th); y = r * sin(th); }
			pred_D(&cD, x, y, rel == -2 ? 250000.0 : r2);
		}
		printf("%-36s %13.4f%% %13.4f%% %13.4f%% %13.4f%%\n", dist[k].name, 100.0 * cC.dec / cC.rows, 100.0 * cA.dec / cA.rows,
			   100.0 * cB.dec / cB.rows, 100.0 * cD.dec / cD.rows);
		wrong += (int)(cA.wrong + cB.wrong + cC.wrong + cD.wrong);
		if (cA.wrong + cB.wrong + cC.wrong + cD.wrong)
			printf("   WRONG decisions: A %llu B %llu C %llu D %llu\n", (unsigned long long) cA.wrong, (unsigned long long) cB.wrong, (unsigned long long) cC.wrong, (unsigned long long) cD.wrong);
	}
	printf("COVER: %s (wrong decisions: %d)\n", wrong ? "FAILED" : "every decided row agrees with the CPU's double evaluation", wrong);
	return wrong ? 1 : 0;
}
