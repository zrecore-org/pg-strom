// Device half of hash_check: hash_bytes_src() of kernels_vm/hash_bytes.inc for the case c = len * 4 + align
// (len 0..40 bytes placed at byte offset align of a 64-byte slot). One thread per case.
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] StructuredBuffer<uint> inA;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> outR;

uint ldb_s(uint src, uint o) { return (inA[o >> 2] >> ((o & 3u) * 8u)) & 0xffu; }
uint ldw_s(uint src, uint o)
{
	uint sh = (o & 3u) * 8u;
	if (sh == 0)
		return inA[o >> 2];
	return (inA[o >> 2] >> sh) | (inA[(o >> 2) + 1] << (32u - sh));
}
#include "hash_bytes.inc"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint c = tid.x;
	if (c >= pc.n)
		return;
	outR[c] = hash_bytes_src(0, c * 64u + (c & 3u), c >> 2);
}
