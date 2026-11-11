// probe: do the error-free transformations survive this device's compiler? (A10 part 6; devcaps.c gpuserv_df64_selfcheck compares
// every output bit for bit with kernels_vm/df64.inc compiled as C). in: 6 words per thread: a b al bl (float bits), the low and
// high word of a binary64; out: 12 words per thread.
#include "../kernels_vm/df64.inc"

struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint> inW;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> outW;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	float a = asfloat(inW[i * 6 + 0]), b = asfloat(inW[i * 6 + 1]);
	float al = asfloat(inW[i * 6 + 2]), bl = asfloat(inW[i * 6 + 3]);
	DF A = df_mk(a, al), B = df_mk(b, bl);
	DF ts = df_two_sum(a, b);
	DF tp = df_two_prod(a, b);
	DF ad = df_add(A, B);
	DF mu = df_mul(A, B);
	bool ok;
	DF cv = df_from_f64bits(inW[i * 6 + 4], inW[i * 6 + 5], ok);
	uint o = i * 12;
	outW[o + 0] = asuint(ts.hi);  outW[o + 1] = asuint(ts.lo);
	outW[o + 2] = asuint(tp.hi);  outW[o + 3] = asuint(tp.lo);
	outW[o + 4] = asuint(ad.hi);  outW[o + 5] = asuint(ad.lo);
	outW[o + 6] = asuint(mu.hi);  outW[o + 7] = asuint(mu.lo);
	outW[o + 8] = asuint(cv.hi);  outW[o + 9] = asuint(cv.lo);
	outW[o + 10] = ok ? 1u : 0u;
	outW[o + 11] = 0u;
}
