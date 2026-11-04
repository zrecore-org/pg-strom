// GpuPreAgg over a GpuJoin, with GROUP BY on up to two integer keys: the join kernel of join.slang
// (the walk is join_walk.inc) combined with the group table of gagg.slang. One workgroup runs over
// a run of outer pages and keeps a hash table of its groups in shared memory; the service merges the
// workgroups' tables. The rounds of gagg.slang (A stage, B find the slot, C accumulate, separated
// by barriers; nobody ever waits for another thread) are the iterations of the join's walk loop:
// each lane produces at most one joined row per iteration, and the loop runs until no lane of the
// whole workgroup has anything left (a shared flag), so every barrier is reached by all threads.
//
// The program is a join program (see join.slang) whose "projection" section computes the key
// expressions, the aggregate arguments and the FILTER conditions; header words as in gagg.slang:
// [10] aggregates, [11] aggregate table offset, [12] keys, [13] offset of the key registers.
//
// Output (outb): [1] first error, [2],[3] page and line pointer; from word 4, per workgroup a block
// of 1 + GA_C * GA_REC words, exactly as gagg.slang writes it. A joined row whose key finds no slot
// within a few probes (or whose min/max argument does not fit 32 bits) is not lost: its staged
// values are appended to an overflow list at word kparam.total_lps (the "overflow base"): { count,
// then records of OV_REC words { null mask, key0 lo, hi, key1 lo, hi, (ok, value lo, hi) * GA_NA } },
// at most kparam.extra records (the count may exceed it: the host runs the task again with room).
#define WITH_INNER 1
#define JW_MODE 2
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif
#include "join_common.inc"

#ifndef GA_NA
#define GA_NA 4
#endif
#define GA_NK 2
#ifndef GA_C
#define GA_C 128
#endif
#define GA_REC (2 + 2 * GA_NK + 6 * GA_NA)
#define OV_REC (1 + 2 * GA_NK + 3 * GA_NA)
#define GA_MAXPROBE 8

static const uint AC_COUNT_ANY  = 1;
static const uint AC_COUNT_COND = 2;
static const uint AC_SUM        = 3;
static const uint AC_MIN        = 4;
static const uint AC_MAX        = 5;
static const uint NONE_REG      = 0xffffffffu;
static const uint COMMITTED     = 0x8000u;
static const uint NO_SLOT       = 0xffffffffu;

groupshared uint    sh_owner[GA_C];
groupshared uint    sh_knull[GA_C];
groupshared int64_t sh_key[GA_C * GA_NK];
groupshared uint    sh_an[GA_C * GA_NA];
groupshared uint    sh_l0[GA_C * GA_NA];
groupshared uint    sh_l1[GA_C * GA_NA];
groupshared uint    sh_l2[GA_C * GA_NA];
groupshared uint    sh_l3[GA_C * GA_NA];
groupshared uint    sh_hits;
groupshared uint    sh_flags[3];                    // per round: bit 0 some lane walks, bit 1 some lane has a row

groupshared uint    st_kn[SCAN_WG];                 // null mask | 0x80000000 if the lane has a row
groupshared uint    st_hash[SCAN_WG];
groupshared uint    st_slot[SCAN_WG];
groupshared int64_t st_key[SCAN_WG * GA_NK];
groupshared uint    st_ok[SCAN_WG * GA_NA];
groupshared int64_t st_val[SCAN_WG * GA_NA];

uint hash_keys(int64_t k0, int64_t k1, uint kn)
{
    uint h = 0x811c9dc5u;
    h = (h ^ (uint(k0) ^ (uint(k0 >> 32) * 0x9e3779b1u) ^ ((kn & 1u) * 0x85ebca6bu))) * 0x01000193u;
    h = (h ^ (uint(k1) ^ (uint(k1 >> 32) * 0x9e3779b1u) ^ ((kn & 2u) * 0x85ebca6bu))) * 0x01000193u;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    return h;
}

#ifdef SPEC_PROG
#include "spec_gen.inc"           // A6: the program baked in (specialize.c)
#endif

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void jgagg_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint wg = kparam.page_base + gid.x;             // page_base counts workgroups here
    uint ppw = kparam.blcksz;                       // pages per workgroup
    uint first_page = wg * ppw;
    if (first_page >= kparam.npages)
        return;                                     // uniform: the whole group leaves
    uint last_page = min(first_page + ppw, kparam.npages);
    uint tid = lid.x;

    uint coloff   = prog[0];
    uint insnoff  = prog[2];
    uint ncols    = prog[3];
    uint nq       = prog[5];
    uint nins     = prog[6];
    uint qreg     = prog[7];
    uint nd       = prog[14];
    uint dtab     = prog[15];
    uint proj_begin = prog[16];
    uint naggs    = prog[10];
    uint aggoff   = prog[11];
    uint nkeys    = prog[12];
    uint keyoff   = prog[13];
    uint ovf_base = kparam.total_lps;
    uint ovf_cap  = kparam.extra;
    uint a_cls[GA_NA];
    for (uint k = 0; k < GA_NA; k++)
        a_cls[k] = (k < naggs) ? prog[aggoff + 4 * k] : 0;

    // empty table
    for (uint s = tid; s < GA_C; s += SCAN_WG)
    {
        sh_owner[s] = 0;
        sh_knull[s] = 0;
        for (uint i = 0; i < GA_NK; i++)
            sh_key[s * GA_NK + i] = 0;
        for (uint k = 0; k < GA_NA; k++)
        {
            sh_an[s * GA_NA + k] = 0;
            sh_l0[s * GA_NA + k] = (a_cls[k] == AC_MIN) ? 0xffffffffu : 0u;
            sh_l1[s * GA_NA + k] = 0;
            sh_l2[s * GA_NA + k] = 0;
            sh_l3[s * GA_NA + k] = 0;
        }
    }
    if (tid == 0)
    {
        sh_hits = 0;
        sh_flags[0] = 0; sh_flags[1] = 0; sh_flags[2] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint ri = 0;                                    // round counter (selects the slot of sh_flags)
    for (uint page = first_page; page < last_page; page++)
    {
#include "join_walk.inc"
    }

    // write the table out
    uint blk = 4 + wg * (1 + GA_C * GA_REC);
    if (tid == 0)
        outb[blk] = sh_hits;
    for (uint s = tid; s < GA_C; s += SCAN_WG)
    {
        uint rb = blk + 1 + s * GA_REC;
        uint o = sh_owner[s];
        outb[rb] = (o != 0) ? 1u : 0u;
        if (o == 0)
            continue;
        outb[rb + 1] = sh_knull[s];
        for (uint i = 0; i < GA_NK; i++)
        {
            uint64_t kv = uint64_t(sh_key[s * GA_NK + i]);
            outb[rb + 2 + 2 * i] = uint(kv & 0xffffffffu);
            outb[rb + 3 + 2 * i] = uint(kv >> 32);
        }
        for (uint k = 0; k < GA_NA; k++)
        {
            uint a = s * GA_NA + k;
            uint ab = rb + 2 + 2 * GA_NK + 6 * k;
            outb[ab + 0] = sh_an[a];
            outb[ab + 1] = 0;
            outb[ab + 2] = sh_l0[a];
            outb[ab + 3] = sh_l1[a];
            outb[ab + 4] = sh_l2[a];
            outb[ab + 5] = sh_l3[a];
        }
    }
}
