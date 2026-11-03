// probe: 64-bit atomic min / max with values beyond 32 bits (signed)
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<int64_t> out;   // [0]=min (init INT64_MAX), [1]=max (init INT64_MIN)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	int64_t v = ((int64_t)((i * 37 + 11) % 4001) - 2000) * 4294967297ll;
	int64_t o;
	InterlockedMin(out[0], v, o);
	InterlockedMax(out[1], v, o);
}
