// probe: 64-bit compare-exchange loop adding (i+1) * 2^32 + 1 to out[0]
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
	int64_t inc = ((int64_t)(i + 1) << 32) + 1;
	int64_t cmp = 0;
	for (int tries = 0; tries < 100000; tries++)
	{
		int64_t orig;
		InterlockedCompareExchange(out[0], cmp, cmp + inc, orig);
		if (orig == cmp)
			break;
		cmp = orig;
	}
}
