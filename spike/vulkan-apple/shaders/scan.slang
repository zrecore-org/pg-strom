// Slang port of scan.comp. Same descriptor layout and push-constant layout,
// so the same host harness can drive either shader.
//
// Mimics a GpuScan kernel: unaligned byte-wise heap-row deform, int64 qual,
// subgroup ballot compaction, one 32-bit global atomic per subgroup.

[[vk::binding(0, 0)]] StructuredBuffer<uint8_t>   rows;
// out[0] = match counter, out[1 + i] = matching row index
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>    outb;

struct PC
{
    uint    nrows;
    uint    stride;
    int64_t thresh;
};
[[vk::push_constant]] ConstantBuffer<PC> pc;

uint ld32(uint o)
{
    return uint(rows[o]) | (uint(rows[o + 1]) << 8) |
           (uint(rows[o + 2]) << 16) | (uint(rows[o + 3]) << 24);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    bool hit = false;

    if (gid < pc.nrows)
    {
        uint base = gid * pc.stride;
        int  a    = int(ld32(base));
        uint blo  = ld32(base + 4);
        uint bhi  = ld32(base + 8);
        int64_t b = int64_t(uint64_t(blo) | (uint64_t(bhi) << 32));
        uint flag = uint(rows[base + 12]);

        hit = (a % 7 == 0) && (b + int64_t(a) * 3 > pc.thresh) && (flag != 0u);
    }
    // every invocation takes part in the wave ops (uniform control flow)
    uint cnt  = WaveActiveCountBits(hit);
    uint excl = WavePrefixCountBits(hit);
    uint pos  = 0;
    if (WaveIsFirstLane() && cnt > 0)
        InterlockedAdd(outb[0], cnt, pos);
    pos = WaveReadLaneFirst(pos);
    if (hit)
        outb[1 + pos + excl] = gid;
}
