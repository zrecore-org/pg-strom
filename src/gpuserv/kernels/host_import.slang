// probe: read an imported host allocation: thread t sums words t, t+64, t+128, ... into out[t]
struct PC { uint nwords; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint> src;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint t = tid.x;
	uint sum = 0;
	for (uint k = t; k < pc.nwords; k += 64)
		sum += src[k] ^ (k * 31u);
	out[t] = sum;
}
