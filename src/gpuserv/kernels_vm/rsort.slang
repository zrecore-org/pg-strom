// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 zrecore-org and exactmesh contributors
//
// Origin: the counting radix sort of the mongo fork (zrecore-org), poc/gpu/shaders/sort_count.comp, sort_scan.comp and
// sort_scatter.comp at commit 1f70e65a552, ported from GLSL to Slang for PG-Strom (shared between the two projects with
// the approval of the user; see doc/gpu-sharing.md).
//
// LSD radix sort of 64-bit keys, 4 bits per pass (16 passes), carrying a row index with every key; stable. Each pass is three
// dispatches over a pair of ping-pong buffers (the host swaps them between passes and puts a barrier between dispatches):
//   rsort_count:   per-block histogram of the digit (block = 256 threads x 16 consecutive keys)
//   rsort_scan:    exclusive scan of the 16*B digit-major histogram, one workgroup
//   rsort_scatter: stable scatter to (global offset of (digit, block)) + (same-digit keys of earlier threads of the block)
// No atomics, no subgroup operations, 16 KiB of shared memory: runs on the base tier (MoltenVK, lavapipe) and the desktop tier.
// A key is the uint2 (x = low word, y = high word); on pass 0 the row index is the input position.

[[vk::binding(0, 0)]] StructuredBuffer<uint2>   rs_kin;    // count, scatter: keys of this pass
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  rs_bh;     // count, scan: histogram [digit * B + block]
[[vk::binding(2, 0)]] StructuredBuffer<uint>    rs_iin;    // scatter: row indexes of this pass (unused on pass 0)
[[vk::binding(3, 0)]] RWStructuredBuffer<uint2> rs_kout;   // scatter: keys sorted by the digit
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>  rs_iout;   // scatter: row indexes sorted by the digit

struct RSPC
{
    uint n;         // keys
    uint B;         // blocks = ceil(n / 4096)
    uint pass;      // 0..15
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<RSPC> rp;

groupshared uint rs_cnt[16 * 256];      // [digit][thread]; every thread owns its column
groupshared uint rs_part[256];

uint rs_digit(uint2 k, uint pass)
{
    uint s = pass * 4;
    return (s < 32) ? ((k.x >> s) & 15) : ((k.y >> (s - 32)) & 15);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsort_count(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rs_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rp.n)
            rs_cnt[rs_digit(rs_kin[e], rp.pass) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint tot = 0;
        for (uint j = 0; j < 256; j++)
            tot += rs_cnt[t * 256 + j];
        rs_bh[t * rp.B + blk] = tot;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsort_scan(uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint M = 16 * rp.B;
    uint per = (M + 255) / 256;
    uint lo = min(t * per, M);
    uint hi = min(lo + per, M);
    uint s = 0;
    for (uint i = lo; i < hi; i++)
        s += rs_bh[i];
    rs_part[t] = s;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rs_part[j];
            rs_part[j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint run = rs_part[t];
    for (uint i = lo; i < hi; i++)
    {
        uint v = rs_bh[i];
        rs_bh[i] = run;
        run += v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsort_scatter(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rs_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rp.n)
            rs_cnt[rs_digit(rs_kin[e], rp.pass) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rs_cnt[t * 256 + j];
            rs_cnt[t * 256 + j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rp.n)
        {
            uint2 k = rs_kin[e];
            uint d = rs_digit(k, rp.pass);
            uint pos = rs_bh[d * rp.B + blk] + rs_cnt[d * 256 + t];
            rs_cnt[d * 256 + t]++;
            rs_kout[pos] = k;
            rs_iout[pos] = (rp.pass == 0) ? e : rs_iin[e];
        }
    }
}
