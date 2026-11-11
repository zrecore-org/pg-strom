/*
 * df64_probe.c -- S0 of A10 part 6 (test/gpuserv/df64_probe.sh): does a double-float pair (kernels_vm/df64.inc) pay as an
 * exact-answer prefilter for float8 quals on this device?
 *
 *   df64_probe corr  <spv> [N]               error-free transformations and df64 operations on the device, bit for bit
 *                                            against the host (the same source compiled as C, -ffp-contract=off), plus the
 *                                            exact identities s + e == a + b and p + e == a * b checked in double
 *   df64_probe bench <spv> <op> <var>        ALU throughput of one operation (best of 5), var 0 float32, 1 df64, 2 soft binary64
 *   df64_probe cover                         host only: the fraction of rows a df64 evaluation with a proven error bound decides
 *
 * No cluster, no module install; it needs only slangc output and the Vulkan loader.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "gpu_vk.h"

#define DF64_C 1
#include "kernels_vm/df64.inc"

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double rndu(void) { return (double)(rnd() >> 11) / 9007199254740992.0; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint64_t d2u(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static double u2d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

static int load_spv(const char *path, void **spv, long *len)
{
	FILE *fp = fopen(path, "rb");
	if (!fp) { printf("cannot open %s\n", path); return -1; }
	fseek(fp, 0, SEEK_END); *len = ftell(fp); fseek(fp, 0, SEEK_SET);
	*spv = malloc(*len);
	if (fread(*spv, 1, *len, fp) != (size_t) *len) { printf("cannot read %s\n", path); return -1; }
	fclose(fp);
	return 0;
}


/* ---- operand generation for the correctness run ---- */
static float rnd_float(int emin, int emax)
{
	uint32_t m = (uint32_t) rnd() & 0x7FFFFF;
	int e = emin + (int)(rnd() % (uint64_t)(emax - emin + 1));
	return u2f(((uint32_t)(rnd() & 1) << 31) | ((uint32_t)(e + 127) << 23) | m);
}
static float sparse_float(int e)
{
	uint32_t m = 0;
	for (int k = rnd() % 4; k > 0; k--) m |= 1u << (rnd() % 23);
	return u2f(((uint32_t)(rnd() & 1) << 31) | ((uint32_t)(e + 127) << 23) | m);
}
static float next_up_n(float a, int n) { return u2f(f2u(a) + (uint32_t) n); }

static void gen_case(uint64_t i, float *a, float *b, float *c, float *al, float *bl, double *x)
{
	switch (i % 10)
	{
		case 0: *a = rnd_float(-60, 60); *b = rnd_float(-60, 60); break;
		case 1: *a = rnd_float(-20, 20); *b = rnd_float(-20, 20); break;
		case 2: *a = rnd_float(-10, 10); *b = -next_up_n(*a, (int)(rnd() % 9) - 4); break;			/* cancellation */
		case 3: *a = rnd_float(-60, 60); *b = rnd_float(-100, -40); break;							/* big exponent gap */
		case 4: { int e = (int)(rnd() % 60) - 30; *a = sparse_float(e); *b = sparse_float(e + (int)(rnd() % 40) - 20); break; }	/* ties */
		case 5: *a = rnd_float(-126, -110); *b = rnd_float(-126, -100); break;						/* tiny, products underflow */
		case 6: *a = u2f((uint32_t)(rnd() & 0x807FFFFF)); *b = rnd_float(-30, 30); break;			/* subnormal operand */
		case 7: *a = rnd_float(55, 63); *b = rnd_float(55, 63); break;								/* products overflow to inf? no: 2^126 */
		case 8: *a = (float)((int64_t)(rnd() % 20001) - 10000) / (float)(1 << (rnd() % 8)); *b = (float)((int64_t)(rnd() % 20001) - 10000); break;
		default: *a = rnd_float(-3, 3); *b = rnd_float(-3, 3); break;
	}
	*c = (i % 3 == 0) ? -fmaf(*a, *b, 0.0f) * (1.0f + (float)(((int)(rnd() % 5)) - 2) * 1.2e-7f) : rnd_float(-60, 60);
	/* low parts: |lo| <= ulp(hi)/2 */
	*al = *a * (float)((rndu() - 0.5) * 1.1920929e-7);
	*bl = *b * (float)((rndu() - 0.5) * 1.1920929e-7);
	if (i % 7 == 0) *al = 0.0f;
	*x = u2d(((rnd() & 1) << 63) | ((uint64_t)(1023 - 62 + rnd() % 125) << 52) | (rnd() & 0xFFFFFFFFFFFFFull));
}

/* "small": below 2^-70, where an intermediate of the df64 operations (a lo part is 2^-24 of its hi, a product of two lo parts 2^-48
 * of the product) can reach the float32 subnormal range, which the device flushes. The coverage run's operand guard keeps clear of it. */
static int is_sub(float f) { return f != 0.0f && fabsf(f) < 8.470329472543003e-22f; }

static const char *oname[18] = { "two_sum.s", "two_sum.e", "fast2sum.s", "fast2sum.e", "two_prod.p", "two_prod.e", "fma", "df_add.hi",
	"df_add.lo", "df_mul.hi", "df_mul.lo", "df_div.hi", "df_div.lo", "df_sqrt.hi", "df_sqrt.lo", "a*b+c(naive)", "cvt.hi", "cvt.lo" };

static int cmd_corr(const char *spvpath, uint64_t n)
{
	gpu_vk_ctx vk; gpu_vk_buffer bi, bo; gpu_vk_pipeline pl; gpu_vk_buffer *bufs[2];
	void *spv; long spvlen; uint32_t *in, *out;
	uint64_t mism[18] = {0}, msub[18] = {0}, idviol = 0, idchk = 0, ncases_sub = 0;
	uint64_t shown = 0;
	double ms = 0;

	if (load_spv(spvpath, &spv, &spvlen)) return 2;
	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, 0)) { printf("cannot open device: %s\n", vk.err); return 2; }
	if (gpu_vk_buffer_create(&vk, &bi, n * 32, 0) || gpu_vk_buffer_create(&vk, &bo, n * 72, 0)) { printf("alloc: %s\n", vk.err); return 2; }
	if (gpu_vk_pipeline_create(&vk, &pl, spv, spvlen, 2, 4)) { printf("pipeline: %s\n", vk.err); return 2; }
	in = bi.map; out = bo.map;
	for (uint64_t i = 0; i < n; i++)
	{
		float a, b, c, al, bl; double x;
		gen_case(i, &a, &b, &c, &al, &bl, &x);
		in[i * 8 + 0] = f2u(a); in[i * 8 + 1] = f2u(b); in[i * 8 + 2] = f2u(c); in[i * 8 + 3] = f2u(al); in[i * 8 + 4] = f2u(bl);
		in[i * 8 + 6] = (uint32_t) d2u(x); in[i * 8 + 7] = (uint32_t)(d2u(x) >> 32);
	}
	memset(out, 0xA5, n * 72);
	bufs[0] = &bi; bufs[1] = &bo;
	{ uint32_t pc = (uint32_t) n;
	  if (gpu_vk_dispatch(&vk, &pl, bufs, &pc, (uint32_t)((n + 63) / 64), 1, 1, &ms)) { printf("dispatch: %s\n", vk.err); return 2; } }
	for (uint64_t i = 0; i < n; i++)
	{
		float a = u2f(in[i * 8]), b = u2f(in[i * 8 + 1]), c = u2f(in[i * 8 + 2]), al = u2f(in[i * 8 + 3]), bl = u2f(in[i * 8 + 4]);
		double x = u2d((uint64_t) in[i * 8 + 6] | ((uint64_t) in[i * 8 + 7] << 32));
		DF A = df_mk(a, al), B = df_mk(b, bl);
		DF ts = df_two_sum(a, b), qs = df_quick_two_sum(a, b), tp = df_two_prod(a, b), ad = df_add(A, B), mu = df_mul(A, B);
		DF dv = df_div(A, B), sq = df_sqrt(df_mk(fabsf(a), fabsf(al)));
		volatile float pr = a * b;
		float fm = fmaf(a, b, c), naive = pr + c;
		DF cv;
		float ref[18];
		int sub;
		uint32_t *o = out + i * 18;
		{ double hi = (double)(float) x; cv = df_mk((float) x, (float)(x - hi)); }
		ref[0] = ts.hi; ref[1] = ts.lo; ref[2] = qs.hi; ref[3] = qs.lo; ref[4] = tp.hi; ref[5] = tp.lo; ref[6] = fm;
		ref[7] = ad.hi; ref[8] = ad.lo; ref[9] = mu.hi; ref[10] = mu.lo; ref[11] = dv.hi; ref[12] = dv.lo; ref[13] = sq.hi; ref[14] = sq.lo;
		ref[15] = naive; ref[16] = cv.hi; ref[17] = cv.lo;
		sub = is_sub(a) || is_sub(b) || is_sub(c) || is_sub(al) || is_sub(bl);
		for (int k = 0; k < 18; k++) sub |= is_sub(ref[k]);
		ncases_sub += sub;
		for (int k = 0; k < 18; k++)
		{
			float d = u2f(o[k]);
			if (o[k] == f2u(ref[k]) || (isnan(d) && isnan(ref[k])) || ((o[k] & 0x7FFFFFFFu) == 0 && (f2u(ref[k]) & 0x7FFFFFFFu) == 0))
				continue;		/* the device does not keep the sign of a zero (+0 for -0): equal as values */
			if (k == 16 || k == 17) { if (fabs(x) < 1e-18 || fabs(x) > 1e18) continue; }
			mism[k]++; msub[k] += sub;
			if (!sub && shown < 6)
			{
				shown++;
				printf("  mismatch %s case %llu: a=%a b=%a c=%a al=%a bl=%a device %08x host %08x\n", oname[k], (unsigned long long) i, a, b, c, al, bl, o[k], f2u(ref[k]));
			}
		}
		/* identities on the device's own outputs, exact in double */
		{
			float s = u2f(o[0]), e = u2f(o[1]), p = u2f(o[4]), pe = u2f(o[5]);
			if (!sub && fabs((double) a) / fabs((double) b + 1e-300) < 1e7 && fabs((double) b) / fabs((double) a + 1e-300) < 1e7 && fabsf(s) < 1e30f)
			{ idchk++; if ((double) s + (double) e != (double) a + (double) b) idviol++; }
			if (!sub && fabsf(p) > 1e-20f && fabsf(p) < 1e30f)
			{ idchk++; if ((double) p + (double) pe != (double) a * (double) b) idviol++; }
		}
	}
	printf("  device ran %llu cases in %.2f ms; %llu of them involve values below 2^-70 (flushed subnormal intermediates possible)\n", (unsigned long long) n, ms, (unsigned long long) ncases_sub);
	{
		int bad = 0;
		for (int k = 0; k < 18; k++)
			if (mism[k] != msub[k])
			{
				printf("  %-14s %llu mismatches against the host outside tiny values (%llu more with a value below 2^-70)\n", oname[k], (unsigned long long)(mism[k] - msub[k]), (unsigned long long) msub[k]);
				if (k != 15) bad++;
			}
		printf("  exact identities (s+e==a+b, p+e==a*b in double) on the device outputs: %llu checked, %llu violated\n", (unsigned long long) idchk, (unsigned long long) idviol);
		printf("DF64 CORR: %s\n", (bad == 0 && idviol == 0) ? "EXACT (bit for bit with the host outside tiny values)" : "BROKEN");
		if (bad == 0 && idviol == 0 && ncases_sub)
		{
			uint64_t t = 0; for (int k = 0; k < 18; k++) t += mism[k];
			if (t) printf("  (%llu mismatches in all, every one with a value below 2^-70: the device flushes float32 subnormals)\n", (unsigned long long) t);
		}
		gpu_vk_pipeline_destroy(&vk, &pl); gpu_vk_close(&vk);
		return (bad == 0 && idviol == 0) ? 0 : 1;
	}
}

/* ---- throughput ---- */
static void to_df(double x, float *h, float *l) { *h = (float) x; *l = (float)(x - (double) *h); }

static int cmd_bench(const char *spvpath, int op, int var)
{
	gpu_vk_ctx vk; gpu_vk_buffer bi, bo; gpu_vk_pipeline pl; gpu_vk_buffer *bufs[2];
	void *spv; long spvlen; uint32_t *in;
	const uint32_t N = 65536;
	uint32_t pc[2];
	double best = 1e30, ms = 0;
	uint32_t iters = 32;
	static const double opsper[7] = { 32, 32, 32, 32, 32, 16, 16 };

	if (load_spv(spvpath, &spv, &spvlen)) return 2;
	memset(&vk, 0, sizeof(vk));
	if (gpu_vk_open(&vk, 0)) { printf("cannot open device: %s\n", vk.err); return 2; }
	if (gpu_vk_buffer_create(&vk, &bi, N * 48, 0) || gpu_vk_buffer_create(&vk, &bo, N * 8, 0)) { printf("alloc: %s\n", vk.err); return 2; }
	if (gpu_vk_pipeline_create(&vk, &pl, spv, spvlen, 2, 8)) { printf("pipeline: %s\n", vk.err); return 2; }
	
	in = bi.map;
	for (uint32_t i = 0; i < N; i++)
	{
		double x = 1.0 + rndu(), y = (op == 4) ? 0.45 + 0.1 * rndu() : 0.9999 + 0.0002 * rndu();
		float xh, xl, yh, yl;
		if (op == 5) { x = 100.0 + rndu() * 100; y = x + (rndu() - 0.5) * 1e-6; }
		to_df(x, &xh, &xl); to_df(y, &yh, &yl);
		in[i * 12 + 0] = f2u((float) x); in[i * 12 + 1] = f2u((float) y);
		in[i * 12 + 2] = f2u(xh); in[i * 12 + 3] = f2u(xl); in[i * 12 + 4] = f2u(yh); in[i * 12 + 5] = f2u(yl);
		in[i * 12 + 6] = (uint32_t) d2u(x); in[i * 12 + 7] = (uint32_t)(d2u(x) >> 32);
		in[i * 12 + 8] = (uint32_t) d2u(y); in[i * 12 + 9] = (uint32_t)(d2u(y) >> 32);
	}
	bufs[0] = &bi; bufs[1] = &bo;
	pc[0] = N;
	/* calibrate to about 150 ms */
	for (;;)
	{
		pc[1] = iters;
		if (gpu_vk_dispatch(&vk, &pl, bufs, pc, N / 64, 1, 1, &ms)) { printf("dispatch: %s\n", vk.err); return 2; }
		if (ms >= 120.0 || iters >= (1u << 22)) break;
		{ double f = 150.0 / (ms > 0.05 ? ms : 0.05); if (f > 16) f = 16; if (f < 1.5) f = 1.5; iters = (uint32_t)(iters * f); }
	}
	for (int r = 0; r < 5; r++)
	{
		pc[1] = iters;
		if (gpu_vk_dispatch(&vk, &pl, bufs, pc, N / 64, 1, 1, &ms)) { printf("dispatch: %s\n", vk.err); return 2; }
		if (ms < best) best = ms;
	}
	printf("BENCH op=%d var=%d iters=%u best=%.2f ms  %.3f Gop/s\n", op, var, iters, best, (double) N * iters * opsper[op] / (best / 1e3) / 1e9);
	gpu_vk_pipeline_destroy(&vk, &pl); gpu_vk_close(&vk);
	return 0;
}

/* ---- coverage (host) ---- */
#include "df64_cover.inc"

int main(int argc, char **argv)
{
	if (argc >= 3 && !strcmp(argv[1], "corr")) return cmd_corr(argv[2], argc > 3 ? strtoull(argv[3], NULL, 10) : 1000000ull);
	if (argc >= 5 && !strcmp(argv[1], "bench")) return cmd_bench(argv[2], atoi(argv[3]), atoi(argv[4]));
	if (argc >= 2 && !strcmp(argv[1], "cover")) return cmd_cover(argc > 2 ? strtoull(argv[2], NULL, 10) : 2000000ull);
	fprintf(stderr, "usage: df64_probe corr <spv> [N] | bench <spv> <op> <var> | cover [N]\n");
	return 2;
}
