// Throughput half of df64_probe: the same loop structure for three number systems, selected by -DVAR (0 float32, 1 df64,
// 2 software binary64 of f64_soft.inc) and -DOP (0 add, 1 mul, 2 div, 3 sqrt(x+y), 4 chain of 16 fma, 5 compare with a bound).
// Two independent dependent chains per thread, pc.iters iterations; the result is stored so nothing is eliminated.
#include "df64.inc"
#include "f64_soft.inc"

struct PC { uint n; uint iters; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint> inW;      // 12 words per thread: x,y as float, x,y as hi/lo, x,y as binary64 (lo, hi), z
[[vk::binding(1,0)]] RWStructuredBuffer<uint> outW;   // 2 words per thread

#if VAR == 0
typedef float NUM;
float n_rd(uint i, uint k) { return asfloat(inW[i * 12 + k]); }
float n_add(float a, float b) { return a + b; }
float n_mul(float a, float b) { return a * b; }
float n_div(float a, float b) { return a / b; }
float n_sqrt(float a) { return sqrt(a); }
float n_fma(float a, float b, float c) { return fma(a, b, c); }
uint n_cmp(float a, float c, float bound) { return (a < c) ? 1u : 0u; }
uint n_bits(float a) { return asuint(a); }
float n_bump(float a) { return asfloat(asuint(a) + 1u); }
#elif VAR == 1
typedef DF NUM;
DF n_rd(uint i, uint k) { return df_mk(asfloat(inW[i * 12 + 2 + 2 * k]), asfloat(inW[i * 12 + 3 + 2 * k])); }
DF n_add(DF a, DF b) { return df_add(a, b); }
DF n_mul(DF a, DF b) { return df_mul(a, b); }
DF n_div(DF a, DF b) { return df_div(a, b); }
DF n_sqrt(DF a) { return df_sqrt(a); }
DF n_fma(DF a, DF b, DF c) { return df_add(df_mul(a, b), c); }
uint n_cmp(DF a, DF c, float bound) { return uint(df_cmp_bound(a, c, bound) + 1); }
uint n_bits(DF a) { return asuint(a.hi) ^ asuint(a.lo); }
DF n_bump(DF a) { return df_mk(a.hi, asfloat(asuint(a.lo) + 1u)); }
#else
typedef uint64_t NUM;
uint64_t n_rd(uint i, uint k) { return uint64_t(inW[i * 12 + 6 + 2 * k]) | (uint64_t(inW[i * 12 + 7 + 2 * k]) << 32); }
uint64_t n_add(uint64_t a, uint64_t b) { return f64_add(a, b); }
uint64_t n_mul(uint64_t a, uint64_t b) { return f64_mul(a, b); }
uint64_t n_div(uint64_t a, uint64_t b) { return f64_div(a, b); }
uint64_t n_sqrt(uint64_t a) { return f64_sqrt(a); }
uint64_t n_fma(uint64_t a, uint64_t b, uint64_t c) { return f64_fma(a, b, c); }
uint n_cmp(uint64_t a, uint64_t c, float bound) { return f64_lt(a, c) ? 1u : 0u; }
uint n_bits(uint64_t a) { return uint(a) ^ uint(a >> 32); }
uint64_t n_bump(uint64_t a) { return a + 1ull; }
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	NUM x0 = n_rd(i, 0), y = n_rd(i, 1);
	NUM x1 = n_add(x0, y);
	uint acc = 0;
	uint lo32 = inW[i * 12 + 6], hi32 = inW[i * 12 + 7];
	[loop] for (uint it = 0; it < pc.iters; it++)
	{
		[unroll] for (int k = 0; k < 16; k++)
		{
#if OP == 0
			x0 = n_add(x0, y); x1 = n_add(x1, y);
#elif OP == 1
			x0 = n_mul(x0, y); x1 = n_mul(x1, y);
#elif OP == 2
			x0 = n_div(x0, y); x1 = n_div(x1, y);
#elif OP == 3
			x0 = n_sqrt(n_add(x0, y)); x1 = n_sqrt(n_add(x1, y));
#elif OP == 4
			x0 = n_fma(x0, y, y); x1 = n_fma(x1, y, y);
#elif OP == 5
			acc += n_cmp(x0, x1, 1e-9f); x1 = n_bump(x1);
#else
			{ bool ok; lo32 += 0x9E3779B9u; DF c = df_from_f64bits(lo32, hi32 ^ (lo32 >> 29), ok);
			  acc += ok ? asuint(c.hi) ^ asuint(c.lo) : 1u; }
#endif
		}
	}
	outW[i * 2 + 0] = n_bits(x0) ^ acc;
	outW[i * 2 + 1] = n_bits(x1);
}
