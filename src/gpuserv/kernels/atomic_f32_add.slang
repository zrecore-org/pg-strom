// probe: float32 atomic add, N threads add 1.0f to out[0]
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<float> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= pc.n)
		return;
	float o;
	InterlockedAdd(out[0], 1.0f, o);
}
