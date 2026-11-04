/*
 * f64soft_check.c
 *
 * Self-check of the software binary64 of kernels_vm/f64_soft.inc (make -C src/gpuserv f64soft_check &&
 * src/gpuserv/f64soft_check [N [device index|-1 [spv]]]).
 *
 * Part 1 (host): the library, compiled here as C (F64SOFT_C), is compared bit for bit with the host's hardware double
 * on N operand pairs per operation (default 4,000,000): random bit patterns, close exponents (cancellation, alignment),
 * sparse mantissas (exact results and ties), subnormals, results around the subnormal and overflow boundaries,
 * specials, and the ranges where a double rounds to a float4 subnormal or overflows. The comparison functions are
 * compared with PostgreSQL's float8_eq ... float8_ge, rint and the int64 cast with rint() and the range check.
 *
 * Part 2 (device, unless the index is -1): the same library compiled by slangc (build/f64soft_check.spv) runs on
 * min(N, 16M) pairs per operation on the Vulkan device and is compared with the hardware double of the host.
 *
 * A NaN result matches any NaN. Exit status 0: everything matched.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "gpu_vk.h"
#include "scan_caps.h"

#define F64SOFT_C 1
#include "kernels_vm/f64_soft.inc"

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t
rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return rng;
}

static uint64_t d2u(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static double u2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- operand generators ---- */
static const uint64_t specials[] = {
	0x0000000000000000ull, 0x8000000000000000ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull,
	0x7FF8000000000000ull, 0xFFF8000000000001ull, 0x7FF0000000000001ull,
	0x3FF0000000000000ull, 0xBFF0000000000000ull, 0x3FE0000000000000ull, 0x4000000000000000ull,
	0x7FEFFFFFFFFFFFFFull, 0xFFEFFFFFFFFFFFFFull, 0x0010000000000000ull, 0x000FFFFFFFFFFFFFull,
	0x0000000000000001ull, 0x8000000000000001ull, 0x3FF0000000000001ull, 0x3FEFFFFFFFFFFFFFull,
	0x4330000000000000ull, 0x4340000000000000ull, 0x43E0000000000000ull, 0xC3E0000000000000ull,
	0x3CA0000000000000ull, 0x3FF8000000000000ull, 0x4008000000000000ull, 0x36A0000000000000ull,
};
#define NSPECIAL (sizeof(specials) / sizeof(specials[0]))

static uint64_t
mkd(uint64_t sign, uint64_t exp, uint64_t frac)
{
	return (sign << 63) | ((exp & 0x7FF) << 52) | (frac & F64_FRAC);
}

static uint64_t
sparse_frac(void)
{
	uint64_t f = 0;
	int n = rnd() % 4;

	for (int i = 0; i < n; i++)
		f |= 1ull << (rnd() % 52);
	if (rnd() % 3 == 0)
		f |= 1ull << 51;
	return f;
}

/* one operand pair; category by index so that every run covers all of them */
static void
gen_pair(uint64_t i, uint64_t *pa, uint64_t *pb)
{
	uint64_t a, b, ea, eb;

	switch (i % 12)
	{
		case 0:		/* uniform bits */
			a = rnd();
			b = rnd();
			break;
		case 1:		/* close exponents, random mantissas */
			ea = 1 + rnd() % 2045;
			eb = ea + (int64_t)(rnd() % 7) - 3;
			if (eb < 1 || eb > 2046)
				eb = ea;
			a = mkd(rnd() & 1, ea, rnd());
			b = mkd(rnd() & 1, eb, rnd());
			break;
		case 2:		/* sparse mantissas, close exponents: exact results and ties */
			ea = 900 + rnd() % 200;
			eb = ea + (int64_t)(rnd() % 120) - 60;
			a = mkd(rnd() & 1, ea, sparse_frac());
			b = mkd(rnd() & 1, eb, sparse_frac());
			break;
		case 3:		/* subnormal operands */
			a = mkd(rnd() & 1, 0, rnd() >> (rnd() % 52));
			b = (rnd() & 1) ? mkd(rnd() & 1, 0, rnd() >> (rnd() % 52)) : rnd();
			break;
		case 4:		/* specials */
			a = specials[rnd() % NSPECIAL];
			b = specials[rnd() % NSPECIAL];
			break;
		case 5:		/* product / quotient around the subnormal boundary */
			ea = 1 + rnd() % 2045;
			eb = (rnd() & 1) ? (uint64_t)(int64_t)(1023 - (int64_t) ea + (int64_t)(rnd() % 70) - 62)
							 : (uint64_t)(int64_t)(ea + 1023 - (int64_t)(rnd() % 70) + 5);
			if ((int64_t) eb < 1 || eb > 2046)
				eb = 1 + rnd() % 2045;
			a = mkd(rnd() & 1, ea, rnd());
			b = mkd(rnd() & 1, eb, rnd());
			break;
		case 6:		/* around overflow */
			ea = 1000 + rnd() % 47;
			eb = 2046 - (ea - 1023) + (int64_t)(rnd() % 5) - 2;
			if ((int64_t) eb < 1 || eb > 2046)
				eb = 1023;
			a = mkd(rnd() & 1, ea, rnd());
			b = mkd(rnd() & 1, eb, rnd());
			break;
		case 7:		/* powers of two and neighbours */
			a = mkd(rnd() & 1, 1 + rnd() % 2045, (rnd() & 1) ? 0 : F64_FRAC);
			b = mkd(rnd() & 1, 1 + rnd() % 2045, (rnd() & 1) ? 0 : (rnd() & 1) ? F64_FRAC : 1);
			break;
		case 8:		/* ranges where a double becomes a float4 subnormal, or overflows */
			a = mkd(rnd() & 1, (rnd() & 1) ? 860 + rnd() % 60 : 1135 + rnd() % 30, rnd());
			b = mkd(rnd() & 1, 860 + rnd() % 320, rnd());
			break;
		case 9:		/* small integers and halves (rint, exact sums) */
			a = d2u((double)((int64_t)(rnd() % 4001) - 2000) / (double)(1 << (rnd() % 4)));
			b = d2u((double)((int64_t)(rnd() % 4001) - 2000) / (double)(1 << (rnd() % 4)));
			break;
		case 10:	/* around 2^52 .. 2^64 (rint, int64 range) */
			a = mkd(rnd() & 1, 1023 + 50 + rnd() % 15, (rnd() & 1) ? rnd() : sparse_frac());
			b = mkd(rnd() & 1, 1023 + 50 + rnd() % 15, rnd());
			break;
		default:	/* x, and x*x of a short mantissa (perfect squares for sqrt) */
			a = mkd(0, 1023 + (int64_t)(rnd() % 80) - 40, rnd() & 0xFFFFFFF0000000ull);
			b = d2u(u2d(a) * u2d(a));
			if (rnd() & 1)
			{
				uint64_t t = a;

				a = b;
				b = t;
			}
			break;
	}
	*pa = a;
	*pb = b;
}

/* ---- the operations: library vs. hardware ---- */
typedef struct
{
	const char *name;
	uint64_t  (*soft)(uint64_t a, uint64_t b, uint32_t mode);
	uint64_t  (*hw)(uint64_t a, uint64_t b, uint32_t mode);
	int			kind;		/* 0: 64-bit result (any NaN matches), 1: compare only when the guard holds, 2: 32-bit */
	uint32_t	mode;
} f64op;

static uint64_t s_add(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_add(a, b); }
static uint64_t s_sub(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_sub(a, b); }
static uint64_t s_mul(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_mul(a, b); }
static uint64_t s_div(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_div(a, b); }
static uint64_t s_sqrt(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_sqrt(a); }
static uint64_t h_add(uint64_t a, uint64_t b, uint32_t m) { (void) m; return d2u(u2d(a) + u2d(b)); }
static uint64_t h_sub(uint64_t a, uint64_t b, uint32_t m) { (void) m; return d2u(u2d(a) - u2d(b)); }
static uint64_t h_mul(uint64_t a, uint64_t b, uint32_t m) { (void) m; return d2u(u2d(a) * u2d(b)); }
static uint64_t h_div(uint64_t a, uint64_t b, uint32_t m) { (void) m; return d2u(u2d(a) / u2d(b)); }
static uint64_t h_sqrt(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return d2u(sqrt(u2d(a))); }

/* PostgreSQL's float8 comparison functions (float.h) */
static int pg_nan(double x) { return isnan(x); }
static int pg_eq(double a, double b) { return pg_nan(a) ? pg_nan(b) : (!pg_nan(b) && a == b); }
static int pg_ne(double a, double b) { return pg_nan(a) ? !pg_nan(b) : (pg_nan(b) || a != b); }
static int pg_lt(double a, double b) { return !pg_nan(a) && (pg_nan(b) || a < b); }
static int pg_le(double a, double b) { return pg_nan(b) || (!pg_nan(a) && a <= b); }
static int pg_gt(double a, double b) { return !pg_nan(b) && (pg_nan(a) || a > b); }
static int pg_ge(double a, double b) { return pg_nan(a) || (!pg_nan(b) && a >= b); }
static uint64_t s_eq(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_eq(a, b); }
static uint64_t s_ne(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_ne(a, b); }
static uint64_t s_lt(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_lt(a, b); }
static uint64_t s_le(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_le(a, b); }
static uint64_t s_gt(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_gt(a, b); }
static uint64_t s_ge(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_ge(a, b); }
static uint64_t h_eq(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_eq(u2d(a), u2d(b)); }
static uint64_t h_ne(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_ne(u2d(a), u2d(b)); }
static uint64_t h_lt(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_lt(u2d(a), u2d(b)); }
static uint64_t h_le(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_le(u2d(a), u2d(b)); }
static uint64_t h_gt(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_gt(u2d(a), u2d(b)); }
static uint64_t h_ge(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_ge(u2d(a), u2d(b)); }

static uint64_t s_rint(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_rint(a); }
static uint64_t h_rint(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return d2u(rint(u2d(a))); }
static uint64_t s_fits(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_fits_i64(a); }
static uint64_t h_fits(uint64_t a, uint64_t b, uint32_t m)
{
	double r = rint(u2d(a));

	(void) b; (void) m;
	return r >= -9223372036854775808.0 && r < 9223372036854775808.0;	/* false for NaN */
}
/* the value only where it fits (kind 1: the guard is h_fits) */
static uint64_t s_toi64(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return (uint64_t) f64_to_i64_rint(a); }
static uint64_t h_toi64(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return (uint64_t)(int64_t) rint(u2d(a)); }
static uint64_t s_fromi64(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_from_i64((int64_t) a); }
static uint64_t h_fromi64(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return d2u((double)(int64_t) a); }

static uint64_t s_fromf32(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_from_f32((uint32_t) a); }
static uint64_t h_fromf32(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return d2u((double) u2f((uint32_t) a)); }
static uint64_t s_tof32(uint64_t a, uint64_t b, uint32_t m) { (void) b; return f64_to_f32(a, m); }
static uint64_t h_tof32(uint64_t a, uint64_t b, uint32_t m)
{
	double d = u2d(a);
	float r = (float) d;				/* nearest even */

	(void) b;
	if (!isnan(d) && m == 1 && (double) r > d)
		r = nextafterf(r, -INFINITY);
	else if (!isnan(d) && m == 2 && (double) r < d)
		r = nextafterf(r, INFINITY);
	return f2u(r);
}

/* the error PostgreSQL raises (float.c): written from its code, on the hardware double */
static uint64_t pg_err_arith(int op, double a, double b)
{
	double r;

	if (op == 3 && b == 0.0 && !isnan(a))
		return 1;							/* division by zero (float8div: not for a NaN dividend) */
	r = op == 0 ? a + b : op == 1 ? a - b : op == 2 ? a * b : a / b;
	if (isinf(r) && !isinf(a) && !isinf(b))
		return 3;							/* overflow */
	if (op == 2 && r == 0.0 && a != 0.0 && b != 0.0)
		return 4;							/* underflow */
	if (op == 3 && r == 0.0 && a != 0.0 && !isinf(b))
		return 4;
	return 0;
}
static uint64_t s_err_add(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_arith_err(0, a, b, f64_arith(0, a, b)); }
static uint64_t s_err_sub(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_arith_err(1, a, b, f64_arith(1, a, b)); }
static uint64_t s_err_mul(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_arith_err(2, a, b, f64_arith(2, a, b)); }
static uint64_t s_err_div(uint64_t a, uint64_t b, uint32_t m) { (void) m; return f64_arith_err(3, a, b, f64_arith(3, a, b)); }
static uint64_t h_err_add(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_err_arith(0, u2d(a), u2d(b)); }
static uint64_t h_err_sub(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_err_arith(1, u2d(a), u2d(b)); }
static uint64_t h_err_mul(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_err_arith(2, u2d(a), u2d(b)); }
static uint64_t h_err_div(uint64_t a, uint64_t b, uint32_t m) { (void) m; return pg_err_arith(3, u2d(a), u2d(b)); }
static uint64_t s_err_f32(uint64_t a, uint64_t b, uint32_t m) { (void) b; (void) m; return f64_to_f32_err(a, f64_to_f32(a, 0)); }
static uint64_t h_err_f32(uint64_t a, uint64_t b, uint32_t m)
{
	double num = u2d(a);
	float result = (float) num;

	(void) b; (void) m;
	if (isinf(result) && !isinf(num))
		return 3;
	if (result == 0.0f && num != 0.0)
		return 4;
	return 0;
}

/* kind: 0 plain 64-bit, 1 value compared only where the guard (the previous op) is true, 2 32-bit result */
static const f64op ops[] = {
	{ "add",    s_add,    h_add,    0, 0 },
	{ "sub",    s_sub,    h_sub,    0, 0 },
	{ "mul",    s_mul,    h_mul,    0, 0 },
	{ "div",    s_div,    h_div,    0, 0 },
	{ "sqrt",   s_sqrt,   h_sqrt,   0, 0 },
	{ "eq",     s_eq,     h_eq,     0, 0 },
	{ "ne",     s_ne,     h_ne,     0, 0 },
	{ "lt",     s_lt,     h_lt,     0, 0 },
	{ "le",     s_le,     h_le,     0, 0 },
	{ "gt",     s_gt,     h_gt,     0, 0 },
	{ "ge",     s_ge,     h_ge,     0, 0 },
	{ "rint",   s_rint,   h_rint,   0, 0 },
	{ "fits_i64", s_fits, h_fits,   0, 0 },
	{ "to_i64_rint", s_toi64, h_toi64, 1, 0 },
	{ "from_i64", s_fromi64, h_fromi64, 0, 0 },
	{ "from_f32", s_fromf32, h_fromf32, 0, 0 },
	{ "to_f32_rn", s_tof32, h_tof32, 2, 0 },
	{ "to_f32_rd", s_tof32, h_tof32, 2, 1 },
	{ "to_f32_ru", s_tof32, h_tof32, 2, 2 },
	{ "err_add", s_err_add, h_err_add, 0, 0 },
	{ "err_sub", s_err_sub, h_err_sub, 0, 0 },
	{ "err_mul", s_err_mul, h_err_mul, 0, 0 },
	{ "err_div", s_err_div, h_err_div, 0, 0 },
	{ "err_to_f32", s_err_f32, h_err_f32, 0, 0 },
};

/* the error numbers of the library are those of the VM */
_Static_assert(F64_E_ZERODIV == SOP_ERR_ZERODIV && F64_E_OVER == SOP_ERR_FLTOVER && F64_E_UNDER == SOP_ERR_FLTUNDER,
			   "f64_soft.inc error numbers differ from scan_vm_ops.inc");
#define NOPS ((int)(sizeof(ops) / sizeof(ops[0])))

/* the operands an operation is given (from_i64 and from_f32 take an integer pattern) */
static void
prep(int op, uint64_t *a, uint64_t *b)
{
	const char *nm = ops[op].name;

	if (strcmp(nm, "from_i64") == 0)
	{
		switch (rnd() % 6)
		{
			case 0: break;
			case 1: *a = (uint64_t)((int64_t) *a >> (rnd() % 64)); break;
			case 2: *a = (uint64_t)((int64_t)(*a >> (rnd() % 64)) * ((rnd() & 1) ? 1 : -1)); break;
			case 3: *a = (rnd() & 1 ? 0x20000000000000ull : 0x40000000000000ull) + (rnd() % 9) - 4; break;
			case 4: *a = (1ull << (rnd() % 64)) + (rnd() % 5) - 2; break;
			default: *a = (rnd() & 1) ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull; *a += rnd() % 3; break;
		}
	}
	else if (strcmp(nm, "from_f32") == 0)
	{
		uint32_t f = (uint32_t) *a;

		if (rnd() % 4 == 0)
			f = (f & 0x807FFFFFu);			/* subnormals and zero */
		else if (rnd() % 8 == 0)
			f = (uint32_t)(specials[rnd() % NSPECIAL] >> 32);
		*a = f;
	}
	(void) b;
}

static int
same64(int kind, uint64_t s, uint64_t h)
{
	if (kind == 2)
		return (s == h) || ((u2f((uint32_t) s) != u2f((uint32_t) s)) && (u2f((uint32_t) h) != u2f((uint32_t) h)));
	if (s == h)
		return 1;
	return isnan(u2d(s)) && isnan(u2d(h));
}

static int total_fail;

static int
host_part(uint64_t n)
{
	int bad = 0;

	for (int op = 0; op < NOPS; op++)
	{
		uint64_t nbad = 0;

		for (uint64_t i = 0; i < n; i++)
		{
			uint64_t a, b, s, h;

			gen_pair(i, &a, &b);
			prep(op, &a, &b);
			if (ops[op].kind == 1 && !h_fits(a, b, 0))
				continue;
			s = ops[op].soft(a, b, ops[op].mode);
			h = ops[op].hw(a, b, ops[op].mode);
			if (!same64(ops[op].kind, s, h))
			{
				if (nbad < 4)
					printf("  FAIL host %s(%016llx, %016llx): soft %016llx hardware %016llx\n", ops[op].name,
						   (unsigned long long) a, (unsigned long long) b, (unsigned long long) s, (unsigned long long) h);
				nbad++;
			}
		}
		printf("  %-12s %s (%llu cases%s)\n", ops[op].name, nbad ? "FAILED" : "ok", (unsigned long long) n,
			   nbad ? ", mismatches counted" : "");
		if (nbad)
		{
			printf("               %llu mismatches\n", (unsigned long long) nbad);
			bad++;
		}
	}
	return bad;
}

/* key order: sorting by f64_key agrees with the comparison functions on random pairs and the specials */
static int
order_part(uint64_t n)
{
	uint64_t nbad = 0;

	for (uint64_t i = 0; i < n; i++)
	{
		uint64_t a, b;
		int c1, c2;

		gen_pair(i, &a, &b);
		c1 = f64_key(a) < f64_key(b) ? -1 : (f64_key(a) > f64_key(b) ? 1 : 0);
		c2 = pg_lt(u2d(a), u2d(b)) ? -1 : (pg_gt(u2d(a), u2d(b)) ? 1 : 0);
		if (c1 != c2)
			nbad++;
	}
	printf("  %-12s %s\n", "key order", nbad ? "FAILED" : "ok");
	return nbad ? 1 : 0;
}

/* ---- the device ---- */
#define GPU_MAX (1u << 24)

static int
gpu_part(int devindex, const char *spvpath, uint64_t n)
{
	gpu_vk_ctx	vk;
	gpu_vk_buffer ba, bb, br;
	gpu_vk_pipeline pl;
	gpu_vk_buffer *bufs[3];
	FILE	   *fp;
	void	   *spv;
	long		spvlen;
	uint64_t   *va, *vb, *vr;
	int			bad = 0;

	if (n > GPU_MAX)
		n = GPU_MAX;
	if (!(fp = fopen(spvpath, "rb")))
	{
		printf("  cannot open %s (make build/f64soft_check.spv)\n", spvpath);
		return 1;
	}
	fseek(fp, 0, SEEK_END);
	spvlen = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	spv = malloc(spvlen);
	if (fread(spv, 1, spvlen, fp) != (size_t) spvlen)
	{
		printf("  cannot read %s\n", spvpath);
		return 1;
	}
	fclose(fp);
	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, devindex) != 0)
	{
		printf("  cannot open Vulkan device %d: %s\n", devindex, vk.err);
		return 1;
	}
	if (gpu_vk_buffer_create(&vk, &ba, n * 8, 0) || gpu_vk_buffer_create(&vk, &bb, n * 8, 0) ||
		gpu_vk_buffer_create(&vk, &br, n * 8, 0))
	{
		printf("  cannot allocate buffers: %s\n", vk.err);
		return 1;
	}
	if (gpu_vk_pipeline_create(&vk, &pl, spv, spvlen, 3, 12) != 0)
	{
		printf("  the device's shader compiler rejected the kernel: %s\n", vk.err);
		return 1;
	}
	{
		char desc[256];

		gpu_vk_describe(&vk, desc, sizeof(desc));
		printf("  %s\n", desc);
	}
	va = ba.map;
	vb = bb.map;
	vr = br.map;
	bufs[0] = &ba;
	bufs[1] = &bb;
	bufs[2] = &br;
	for (int op = 0; op < NOPS; op++)
	{
		uint32_t pc[3];
		uint64_t nbad = 0;
		double ms = 0.0;

		for (uint64_t i = 0; i < n; i++)
		{
			gen_pair(i, &va[i], &vb[i]);
			prep(op, &va[i], &vb[i]);
		}
		pc[0] = (uint32_t) n;
		pc[1] = (uint32_t) op;
		pc[2] = ops[op].mode;
		memset(vr, 0xA5, n * 8);
		if (gpu_vk_dispatch(&vk, &pl, bufs, pc, (uint32_t)((n + 63) / 64), 1, 1, &ms) != 0)
		{
			printf("  %-12s dispatch failed: %s\n", ops[op].name, vk.err);
			bad++;
			continue;
		}
		for (uint64_t i = 0; i < n; i++)
		{
			uint64_t h, s = vr[i];

			if (ops[op].kind == 1 && !h_fits(va[i], vb[i], 0))
				continue;
			h = ops[op].hw(va[i], vb[i], ops[op].mode);
			if (ops[op].kind == 2)
				s &= 0xFFFFFFFFull;
			if (!same64(ops[op].kind, s, h))
			{
				if (nbad < 4)
					printf("  FAIL device %s(%016llx, %016llx): device %016llx hardware %016llx\n", ops[op].name,
						   (unsigned long long) va[i], (unsigned long long) vb[i], (unsigned long long) s,
						   (unsigned long long) h);
				nbad++;
			}
		}
		printf("  %-12s %s (%llu cases, %.2f ms, %.0f Mop/s)\n", ops[op].name, nbad ? "FAILED" : "ok",
			   (unsigned long long) n, ms, ms > 0 ? (double) n / ms / 1000.0 : 0.0);
		if (nbad)
		{
			printf("               %llu mismatches\n", (unsigned long long) nbad);
			bad++;
		}
	}
	gpu_vk_pipeline_destroy(&vk, &pl);
	gpu_vk_buffer_destroy(&vk, &ba);
	gpu_vk_buffer_destroy(&vk, &bb);
	gpu_vk_buffer_destroy(&vk, &br);
	gpu_vk_close(&vk);
	return bad;
}

int
main(int argc, char **argv)
{
	uint64_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : 4000000ull;
	int dev = argc > 2 ? atoi(argv[2]) : 0;
	const char *spv = argc > 3 ? argv[3] : "build/f64soft_check.spv";

	printf("host: library (C) vs hardware double\n");
	total_fail += host_part(n);
	total_fail += order_part(n);
	if (dev >= 0)
	{
		printf("device %d: library (Slang, %s) vs hardware double\n", dev, spv);
		total_fail += gpu_part(dev, spv, n);
	}
	printf(total_fail ? "F64SOFT CHECK: %d FAILED\n" : "F64SOFT CHECK: ALL PASSED\n", total_fail);
	return total_fail ? 1 : 0;
}
