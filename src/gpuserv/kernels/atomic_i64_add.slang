// probe: 64-bit atomic add. out[0] += (1 + 2^32) from every thread; out[1+(i%4)] += 1
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<int64_t> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	int64_t o;
	InterlockedAdd(out[0], 4294967297ll, o);
	InterlockedAdd(out[1 + (i % 4)], (int64_t)1, o);
}
