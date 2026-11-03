// probe: a kernel writing a host-mapped buffer: out[i] = i * 2654435761 + 7
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x < pc.n)
		out[tid.x] = tid.x * 2654435761u + 7u;
}
