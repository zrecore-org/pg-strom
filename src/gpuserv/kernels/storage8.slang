// probe: 8-bit loads at unaligned byte offsets from a StructuredBuffer<uint8_t>
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint8_t> src;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	out[i] = (uint)src[3 * i + 1]
		   | ((uint)src[3 * i + 2] << 8)
		   | ((uint)src[3 * i + 5] << 16)
		   | ((uint)src[7 * i + 3] << 24);
}
