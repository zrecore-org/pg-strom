// probe: int64 subgroup sum.  out[i] = WaveActiveSum(v) (v needs > 32 bits); lane[i*2], lane[i*2+1]
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<int64_t> out;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> lane;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	int64_t v = ((int64_t)(i + 1) * 4294967297ll) - 1000000000000ll;
	out[i] = WaveActiveSum(v);
	lane[i * 2 + 0] = WaveGetLaneIndex();
	lane[i * 2 + 1] = WaveGetLaneCount();
}
