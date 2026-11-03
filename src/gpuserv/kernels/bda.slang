// probe: raw pointer from a push-constant device address; read-modify-write p[i] = p[i]*3 + 1
struct PC { uint64_t addr; uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	uint *p = (uint *)pc.addr;
	p[i] = p[i] * 3 + 1;
}
