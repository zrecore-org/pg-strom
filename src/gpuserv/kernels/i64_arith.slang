// probe: 64-bit integer arithmetic (shifts, multiply beyond 2^32, divide, compare)
// 256 threads, 8 uint64 results per thread. Verified exactly on the host.
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<uint64_t> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	uint64_t u = (uint64_t)i;
	int64_t  s = (int64_t)i - 128;
	uint64_t v0 = (u << 33) | 5;
	uint64_t v1 = (u + 1) * 4294967311ull;              // (2^32 + 15) * (i+1), wraps
	int64_t  v2 = s * 3000000000ll;                     // signed, crosses 2^32
	int64_t  v3 = (-(int64_t)(i + 1) * 1000003ll) >> 3; // arithmetic shift of negative
	uint64_t big = 0xFFFFFFFFFFFFFFF0ull - u * 1000003ull;
	uint64_t v4 = big / 7ull;
	uint64_t v5 = big % 7ull;
	uint64_t v6 = ((s << 32) > ((int64_t)10 << 32)) ? 1ull : 0ull;       // signed 64-bit compare
	uint64_t v7 = (big > 0xFFFFFFFF00000000ull + (u << 20)) ? 1ull : 0ull; // unsigned 64-bit compare
	out[i * 8 + 0] = v0;
	out[i * 8 + 1] = v1;
	out[i * 8 + 2] = (uint64_t)v2;
	out[i * 8 + 3] = (uint64_t)v3;
	out[i * 8 + 4] = v4;
	out[i * 8 + 5] = v5;
	out[i * 8 + 6] = v6;
	out[i * 8 + 7] = v7;
}
