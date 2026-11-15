// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 zrecore-org and exactmesh contributors
//
// Radix sort of row indexes by multi-word keys for GPU-Sort (gpusort.c): the counting scheme of kernels_vm/rsort.slang (ported
// from the mongo fork, poc/gpu/shaders/sort_*.comp at commit 1f70e65a552), with the digit read through the row index from a key
// buffer of nw uint32 words per row. One pass sorts the current order of the indexes by one 4-bit digit of one key word (stable);
// the host runs the words from the last to the first, the digits of a word from the lowest, and skips a digit that is equal in
// every row. The initial order is the identity, so the result equals the bitonic network's order (the row index is the last
// tie-breaker). No atomics, no subgroup operations, 16 KiB of shared memory: the base tier.
//   rsortw_count:   per-block histogram of the digit (block = 256 threads x 16 consecutive positions)
//   rsortw_scan:    exclusive scan of the 16*B digit-major histogram, one workgroup
//   rsortw_scatter: stable scatter of the row indexes

[[vk::binding(0, 0)]] StructuredBuffer<uint>    rw_keys;   // n rows of nw words
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  rw_bh;     // histogram [digit * B + block]
[[vk::binding(2, 0)]] StructuredBuffer<uint>    rw_in;     // row indexes in the current order
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>  rw_out;    // row indexes in the order of this pass

struct RWPC
{
    uint n;         // rows
    uint B;         // blocks = ceil(n / 4096)
    uint nw;        // key words per row
    uint word;      // the key word of this pass
    uint shift;     // 0, 4, .. 28: the digit of the word
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<RWPC> rq;

groupshared uint rw_cnt[16 * 256];
groupshared uint rw_part[256];

uint rw_digit(uint row)
{
    return (rw_keys[row * rq.nw + rq.word] >> rq.shift) & 15;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortw_count(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rw_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rq.n)
            rw_cnt[rw_digit(rw_in[e]) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint tot = 0;
        for (uint j = 0; j < 256; j++)
            tot += rw_cnt[t * 256 + j];
        rw_bh[t * rq.B + blk] = tot;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortw_scan(uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint M = 16 * rq.B;
    uint per = (M + 255) / 256;
    uint lo = min(t * per, M);
    uint hi = min(lo + per, M);
    uint s = 0;
    for (uint i = lo; i < hi; i++)
        s += rw_bh[i];
    rw_part[t] = s;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rw_part[j];
            rw_part[j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint run = rw_part[t];
    for (uint i = lo; i < hi; i++)
    {
        uint v = rw_bh[i];
        rw_bh[i] = run;
        run += v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rsortw_scatter(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint t = lid.x;
    uint blk = gid.x;
    for (uint d = 0; d < 16; d++)
        rw_cnt[d * 256 + t] = 0;
    uint base = blk * 4096 + t * 16;
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rq.n)
            rw_cnt[rw_digit(rw_in[e]) * 256 + t]++;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t < 16)
    {
        uint run = 0;
        for (uint j = 0; j < 256; j++)
        {
            uint v = rw_cnt[t * 256 + j];
            rw_cnt[t * 256 + j] = run;
            run += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < 16; i++)
    {
        uint e = base + i;
        if (e < rq.n)
        {
            uint row = rw_in[e];
            uint d = rw_digit(row);
            uint pos = rw_bh[d * rq.B + blk] + rw_cnt[d * 256 + t];
            rw_cnt[d * 256 + t]++;
            rw_out[pos] = row;
        }
    }
}
