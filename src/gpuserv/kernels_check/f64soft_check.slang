// Device half of f64soft_check: one library operation per dispatch (pc.op, the order of ops[] in f64soft_check.c),
// one operand pair per thread. The result goes back as a 64-bit pattern (float4 results in the low 32 bits).
#include "f64_soft.inc"

struct PC { uint n; uint op; uint mode; };

// c for the fma check, the same as fma_c() of f64soft_check.c
uint64_t fma_c(uint64_t a, uint64_t b)
{
	uint64_t m = (a * 0x9E3779B97F4A7C15ull) ^ (b >> 5) ^ (b << 11);
	int ea = int((a >> 52) & 0x7FFull), eb = int((b >> 52) & 0x7FFull), ec = ea + eb - 1023 + int((a >> 7) % 9ull) - 4;
	uint sel = uint((a >> 20) & 15ull);

	if (sel < 4u)
	{
		uint64_t p = f64_mul(a, b) ^ 0x8000000000000000ull;
		return p + uint64_t(int64_t(int(b & 3ull) - 1));
	}
	if (sel == 4u) return 0ull;
	if (sel == 5u) return 0x8000000000000000ull;
	if (sel == 6u) return 0x7FF0000000000000ull | (m & 0x8000000000000000ull);
	if (ec < 0) ec = 0;
	if (ec > 2046) ec = 2046;
	return (m & 0x800FFFFFFFFFFFFFull) | (uint64_t(ec) << 52);
}
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint64_t> inA;
[[vk::binding(1,0)]] StructuredBuffer<uint64_t> inB;
[[vk::binding(2,0)]] RWStructuredBuffer<uint64_t> outR;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	uint64_t a = inA[i];
	uint64_t b = inB[i];
	uint64_t r = 0ull;

	if (pc.op == 0u)       r = f64_add(a, b);
	else if (pc.op == 1u)  r = f64_sub(a, b);
	else if (pc.op == 2u)  r = f64_mul(a, b);
	else if (pc.op == 3u)  r = f64_div(a, b);
	else if (pc.op == 4u)  r = f64_sqrt(a);
	else if (pc.op == 5u)  r = f64_eq(a, b) ? 1ull : 0ull;
	else if (pc.op == 6u)  r = f64_ne(a, b) ? 1ull : 0ull;
	else if (pc.op == 7u)  r = f64_lt(a, b) ? 1ull : 0ull;
	else if (pc.op == 8u)  r = f64_le(a, b) ? 1ull : 0ull;
	else if (pc.op == 9u)  r = f64_gt(a, b) ? 1ull : 0ull;
	else if (pc.op == 10u) r = f64_ge(a, b) ? 1ull : 0ull;
	else if (pc.op == 11u) r = f64_rint(a);
	else if (pc.op == 12u) r = f64_fits_i64(a) ? 1ull : 0ull;
	else if (pc.op == 13u) r = (uint64_t) f64_to_i64_rint(a);
	else if (pc.op == 14u) r = f64_from_i64((int64_t) a);
	else if (pc.op == 15u) r = f64_from_f32((uint)(a & 0xFFFFFFFFull));
	else if (pc.op == 16u || pc.op == 17u || pc.op == 18u) r = (uint64_t) f64_to_f32(a, pc.mode);
	else if (pc.op == 19u) r = (uint64_t) f64_arith_err(0u, a, b, f64_arith(0u, a, b));
	else if (pc.op == 20u) r = (uint64_t) f64_arith_err(1u, a, b, f64_arith(1u, a, b));
	else if (pc.op == 21u) r = (uint64_t) f64_arith_err(2u, a, b, f64_arith(2u, a, b));
	else if (pc.op == 22u) r = (uint64_t) f64_arith_err(3u, a, b, f64_arith(3u, a, b));
	else if (pc.op == 24u) r = f64_math(4u, a);
	else if (pc.op == 25u) r = f64_math(5u, a);
	else if (pc.op == 26u) r = f64_math(6u, a);
	else if (pc.op == 27u) r = f64_math(7u, a);
	else if (pc.op == 28u) r = (uint64_t) f64_math_err(2u, a);
	else if (pc.op == 29u) r = f64_math(2u, a);
	else if (pc.op == 30u) r = f64_math(3u, a);
	else if (pc.op == 23u) r = (uint64_t) f64_to_f32_err(a, f64_to_f32(a, 0u));
	else if (pc.op == 31u) r = f64_sqr_add(a, b);
	else if (pc.op == 32u) r = f64_fma(a, b, fma_c(a, b));
	else if (pc.op == 33u) r = f64_addf4(a, b);
	else if (pc.op == 34u) r = uint64_t(f64_arith_err(0u, a, b, f64_addf4(a, b)));
	else                   r = ~0ull;
	outR[i] = r;
}
