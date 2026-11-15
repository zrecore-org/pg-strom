// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 zrecore-org and exactmesh contributors
//
// Radix sort of row indexes by multi-word keys, variant with a gathered key word (experiment of A18 "tuning for wide keys"; the
// counting scheme is that of rsort.slang, ported from the mongo fork, poc/gpu/shaders/sort_*.comp at commit 1f70e65a552).
// rsortw.slang reads the digit of every pass through the row index (a random gather per pass, 3 per pass). Here the key word of a
// word-group is gathered ONCE into the pair (key word, row index) (rsortk_gather), and the up to 8 digit passes of that word then
// read and write the pairs sequentially (rsortk_count / rsortk_scan / rsortk_scatter, a stable counting sort by one 4-bit digit of
// the pair's key). The pairs ping-pong between two buffers; the identity (index in y) is the initial order. No atomics, no subgroup
// operations, 16 KiB of shared memory: the base tier. Bindings (shared by the four entry points): 0 key rows (nw words per row),
// 1 histogram, 2 pairs in, 3 pairs out.

[[vk::binding(0, 0)]] StructuredBuffer<uint>    rk_keys;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  rk_bh;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint2> rk_in;     // x = key word of the current group, y = row index
[[vk::binding(3, 0)]] RWStructuredBuffer<uint2> rk_out;

struct RKPC
{
    uint n;         // rows
    uint B;         // blocks = ceil(n / 4096)
    uint nw;        // key words per row
    uint word;      // gather: the key word
    uint shift;     // 0, 4, .. 28: the digit
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<RKPC> rk;

groupshared uint rk_cnt[16 * 256];
groupshared uint rk_part[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortk_gather(uint3 tid : SV_DispatchThreadID)
{
    uint e = tid.x;
    if (e < rk.n)
    {
        uint2 p = rk_in[e];
        p.x = rk_keys[p.y * rk.nw + rk.word];
        rk_in[e] = p;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortk_count(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rk_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rk.n)
            rk_cnt[((rk_in[e].x >> rk.shift) & 15) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint tot = 0;
        for (uint j = 0; j < 256; j++)
            tot += rk_cnt[t * 256 + j];
        rk_bh[t * rk.B + blk] = tot;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortk_scan(uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint M = 16 * rk.B;
    uint per = (M + 255) / 256;
    uint lo = min(t * per, M);
    uint hi = min(lo + per, M);
    uint s = 0;
    for (uint i = lo; i < hi; i++)
        s += rk_bh[i];
    rk_part[t] = s;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rk_part[j];
            rk_part[j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint run = rk_part[t];
    for (uint i = lo; i < hi; i++)
    {
        uint v = rk_bh[i];
        rk_bh[i] = run;
        run += v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortk_scatter(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rk_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rk.n)
            rk_cnt[((rk_in[e].x >> rk.shift) & 15) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rk_cnt[t * 256 + j];
            rk_cnt[t * 256 + j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rk.n)
        {
            uint2 p = rk_in[e];
            uint d = (p.x >> rk.shift) & 15;
            uint pos = rk_bh[d * rk.B + blk] + rk_cnt[d * 256 + t];
            rk_cnt[d * 256 + t]++;
            rk_out[pos] = p;
        }
    }
}
