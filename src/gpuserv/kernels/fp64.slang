// probe: IEEE double arithmetic. Inputs come from the host so nothing is constant folded.
// in: 0.1 0.2 1.1 1.0 3.0 8.0 4.0 10.0 1e15 ; out per thread: 8 doubles
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<double> src;
[[vk::binding(1,0)]] RWStructuredBuffer<double> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	if (i >= pc.n)
		return;
	double a = src[0], b = src[1], c = src[2], d = src[3], e = src[4], f = src[5], g = src[6], h = src[7], k15 = src[8];
	double acc = 0.0;
	for (int k = 0; k < 10; k++)
		acc += a;
	out[i * 8 + 0] = a + b;                  // 0.1 + 0.2
	out[i * 8 + 1] = c * c;                  // 1.1 * 1.1
	out[i * 8 + 2] = d / e;                  // 1/3 (may be off by one ulp on some drivers)
	out[i * 8 + 3] = d / f;                  // 1/8 exact
	out[i * 8 + 4] = h / g;                  // 10/4 exact
	out[i * 8 + 5] = acc;                    // ten additions of 0.1
	out[i * 8 + 6] = (double)(int64_t)(9007199254740993ll + (int64_t)i); // int64 -> double rounding
	out[i * 8 + 7] = (double)(int64_t)(c * k15) - (double)i;           // double -> int64 -> double
}
