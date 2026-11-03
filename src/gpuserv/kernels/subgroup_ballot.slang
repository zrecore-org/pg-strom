// probe: subgroup ballot / count / prefix / elect / broadcast compaction.
// info[i*10 + 0..3] ballot.xyzw, 4 count, 5 prefix, 6 isFirst, 7 readFirst(i),
//               8 lane index, 9 lane count.   comp[(i - lane) + prefix] = i if pred.
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<uint> info;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> comp;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	uint i = tid.x;
	uint lane = WaveGetLaneIndex();
	bool pred = (i % 3 == 0) || (i % 7 == 2);
	uint4 b = WaveActiveBallot(pred);
	uint cnt = WaveActiveCountBits(pred);
	uint pfx = WavePrefixCountBits(pred);
	uint first = WaveIsFirstLane() ? 1u : 0u;
	uint rf = WaveReadLaneFirst(i);
	if (pred)
		comp[(i - lane) + pfx] = i;
	info[i * 10 + 0] = b.x;
	info[i * 10 + 1] = b.y;
	info[i * 10 + 2] = b.z;
	info[i * 10 + 3] = b.w;
	info[i * 10 + 4] = cnt;
	info[i * 10 + 5] = pfx;
	info[i * 10 + 6] = first;
	info[i * 10 + 7] = rf;
	info[i * 10 + 8] = lane;
	info[i * 10 + 9] = WaveGetLaneCount();
}
