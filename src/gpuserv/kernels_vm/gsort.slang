// GPU-Sort: bitonic sort of row indexes by multi-word keys (GpuPreAgg result rows; see gpusort.c).
//
// skeys is n rows of nw uint32 words; the words of a row, compared as unsigned integers from the
// first to the last, give the sort order (the host encodes signed values with the sign bit flipped,
// DESC by complementing, NULL placement by a leading word). sperm holds npad row indexes (npad is a
// power of two >= GS_TILE); the indexes >= n are padding that sorts after every real row. The row
// index is the last tie-breaker, so the order is total and the network's result is deterministic.
//
// No atomics, nothing shared between workgroups within a pass, no subgroup operations: a pass is one
// dispatch and the host puts a memory barrier between passes (gpu_vk_dispatch_seq).
//   mode 0: every workgroup sorts one tile of GS_TILE indexes in shared memory (bitonic sort with the
//           alternating directions that the larger merge steps need)
//   mode 1: one global compare-exchange step (k, j) with j >= GS_TILE  (one thread per pair)
//   mode 2: the steps j = GS_TILE/2 .. 1 of merge size k inside every tile, in shared memory
#ifndef GS_TILE
#define GS_TILE 512
#endif
static const uint GS_WG = GS_TILE / 2;

[[vk::binding(0, 0)]] StructuredBuffer<uint>   skeys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> sperm;

struct GSPC
{
    uint n;         // rows
    uint nw;        // key words per row
    uint k;         // merge size (mode 1, 2)
    uint j;         // distance (mode 1)
    uint mode;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<GSPC> gp;

groupshared uint sh[GS_TILE];

int gs_cmp(uint a, uint b)
{
    uint n = gp.n;
    if (a >= n)
    {
        if (b >= n)
            return (a < b) ? -1 : ((a > b) ? 1 : 0);
        return 1;
    }
    if (b >= n)
        return -1;
    uint nw = gp.nw;
    uint ra = a * nw;
    uint rb = b * nw;
    for (uint w = 0; w < nw; w++)
    {
        uint x = skeys[ra + w];
        uint y = skeys[rb + w];
        if (x != y)
            return (x < y) ? -1 : 1;
    }
    return (a < b) ? -1 : ((a > b) ? 1 : 0);
}

[shader("compute")]
[numthreads(GS_WG, 1, 1)]
void gsort_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint mode = gp.mode;

    if (mode == 1)
    {
        uint j = gp.j;
        uint k = gp.k;
        uint p = gid.x * GS_WG + t;
        uint i = ((p & ~(j - 1)) << 1) | (p & (j - 1));
        uint l = i + j;
        bool asc = ((i & k) == 0);
        uint a = sperm[i];
        uint b = sperm[l];
        int c = gs_cmp(a, b);
        if ((c > 0) == asc)
        {
            sperm[i] = b;
            sperm[l] = a;
        }
        return;
    }

    uint base = gid.x * GS_TILE;
    sh[t] = sperm[base + t];
    sh[t + GS_WG] = sperm[base + t + GS_WG];
    GroupMemoryBarrierWithGroupSync();

    uint kfirst = 2;
    uint klast = GS_TILE;
    if (mode == 2)
    {
        kfirst = gp.k;
        klast = gp.k;
    }
    for (uint k = kfirst; k <= klast; k <<= 1)
    {
        uint jstart = (mode == 2) ? GS_WG : (k >> 1);
        for (uint j = jstart; j > 0; j >>= 1)
        {
            uint i = ((t & ~(j - 1)) << 1) | (t & (j - 1));
            uint l = i + j;
            bool asc = (((base + i) & k) == 0);
            uint a = sh[i];
            uint b = sh[l];
            int c = gs_cmp(a, b);
            if ((c > 0) == asc)
            {
                sh[i] = b;
                sh[l] = a;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    sperm[base + t] = sh[t];
    sperm[base + t + GS_WG] = sh[t + GS_WG];
}
