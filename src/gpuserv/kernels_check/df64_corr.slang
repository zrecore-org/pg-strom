// Correctness half of df64_probe: the error-free transformations and the df64 operations, one operand set per thread,
// results as float bit patterns for a bit-for-bit comparison with the host (kernels_vm/df64.inc compiled as C).
// Built in several variants (slangc -fp-mode, -DDF_USE_PRECISE) by test/gpuserv/df64_probe.sh.
#include "df64.inc"
#ifndef CORR_LEVEL
#define CORR_LEVEL 3
#endif

struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint> inW;      // 8 words per thread: a b c al bl (float bits), x64 lo, x64 hi
[[vk::binding(1,0)]] RWStructuredBuffer<uint> outW;   // 18 words per thread

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	float a = asfloat(inW[i * 8 + 0]), b = asfloat(inW[i * 8 + 1]), c = asfloat(inW[i * 8 + 2]);
	float al = asfloat(inW[i * 8 + 3]), bl = asfloat(inW[i * 8 + 4]);
	DF A = df_mk(a, al), B = df_mk(b, bl);
	DF ts = df_two_sum(a, b);
	DF qs = df_quick_two_sum(a, b);
	DF tp = df_two_prod(a, b);
	DF ad = df_mk(0.0f, 0.0f), mu = ad, dv = ad, sq = ad;
#if CORR_LEVEL >= 1
	ad = df_add(A, B);
	mu = df_mul(A, B);
#endif
#if CORR_LEVEL >= 2
	dv = df_div(A, B);
	sq = df_sqrt(df_mk(abs(a), abs(al)));
#endif
	float fm = fma(a, b, c);
	float naive = a * b + c;			// no `precise`: whether the toolchain contracts this is the question
	uint o = i * 18;
	outW[o + 0] = asuint(ts.hi);  outW[o + 1] = asuint(ts.lo);
	outW[o + 2] = asuint(qs.hi);  outW[o + 3] = asuint(qs.lo);
	outW[o + 4] = asuint(tp.hi);  outW[o + 5] = asuint(tp.lo);
	outW[o + 6] = asuint(fm);
	outW[o + 7] = asuint(ad.hi);  outW[o + 8] = asuint(ad.lo);
	outW[o + 9] = asuint(mu.hi);  outW[o + 10] = asuint(mu.lo);
	outW[o + 11] = asuint(dv.hi); outW[o + 12] = asuint(dv.lo);
	outW[o + 13] = asuint(sq.hi); outW[o + 14] = asuint(sq.lo);
	outW[o + 15] = asuint(naive);
#if CORR_LEVEL >= 3
	bool ok;
	DF cv = df_from_f64bits(inW[i * 8 + 6], inW[i * 8 + 7], ok);
	outW[o + 16] = asuint(cv.hi);  outW[o + 17] = ok ? asuint(cv.lo) : 0x7FC00000u;
#endif
}
