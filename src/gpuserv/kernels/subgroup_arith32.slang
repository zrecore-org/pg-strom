// probe: 32-bit subgroup sum / min / max.  out[i*5 + 0..2] sum,min,max; 3 lane; 4 lane count
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	uint v = ((i * 2654435761u) >> 7) % 1000 + 1;
	out[i * 5 + 0] = WaveActiveSum(v);
	out[i * 5 + 1] = WaveActiveMin(v);
	out[i * 5 + 2] = WaveActiveMax(v);
	out[i * 5 + 3] = WaveGetLaneIndex();
	out[i * 5 + 4] = WaveGetLaneCount();
}
