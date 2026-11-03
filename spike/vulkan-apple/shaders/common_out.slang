// Output side shared by every kernel in the spike: push constants, the result
// buffer, first-error reporting and subgroup-ballot compaction of matches.

// out[0] = match count, out[1] = first error code, out[2] = row/tuple id of that
// error, out[3 + i] = id of the i-th match
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  outb;

struct PC
{
    uint nrows;     // rows, or heap pages in the heap kernels
    uint stride;    // row stride in bytes, or number of workgroups in the heap kernels
};
[[vk::push_constant]] ConstantBuffer<PC> kparam;

// remember the first error (code + id); later errors are dropped
void report_error(uint gid, uint err)
{
    uint prev;
    InterlockedCompareExchange(outb[1], 0, err, prev);
    if (prev == 0)
        outb[2] = gid;
}

// subgroup ballot compaction of matching ids into outb (as in scan.slang)
void emit_matches(uint gid, bool hit)
{
    uint cnt  = WaveActiveCountBits(hit);
    uint excl = WavePrefixCountBits(hit);
    uint pos  = 0;
    if (WaveIsFirstLane() && cnt > 0)
        InterlockedAdd(outb[0], cnt, pos);
    pos = WaveReadLaneFirst(pos);
    if (hit)
        outb[3 + pos + excl] = gid;
}
