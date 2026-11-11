// GROUP BY over a join with direct-mapped group slots: the join kernel of join.slang (the walk is
// join_walk.inc, JW_MODE 3) combined with the table of gagg_dm.slang. Only built as a specialized kernel
// (SPEC_PROG and SPEC_DM, see specialize.c). A joined row's key maps to its slot arithmetically, so there is
// no staging, no slot claiming and no barrier: a lane that completes a joined row adds it to the workgroup's
// table with shared atomics at once (the lanes of a wave walk until all are done, as in jagg.slang). A row
// whose key is outside the window, or whose min/max argument does not fit 32 bits, is appended to the
// overflow list at word kparam.total_lps, exactly as in jgagg.slang, which also describes the output.
#define WITH_INNER 1
#define JW_MODE 3
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif
#include "join_common.inc"

#ifndef GA_NA
#define GA_NA 4
#endif
#define GA_NK 4                                // the record layout of jgagg.slang; this kernel groups on at most two keys
#ifndef GA_C
#define GA_C 128
#endif
#define GA_REC (2 + 2 * GA_NK + 6 * GA_NA)
#define OV_REC (1 + 2 * GA_NK + 3 * GA_NA)

static const uint AC_COUNT_ANY  = 1;
static const uint AC_COUNT_COND = 2;
static const uint AC_SUM        = 3;
static const uint AC_MIN        = 4;
static const uint AC_MAX        = 5;

groupshared uint    sh_occ[GA_C];
groupshared uint    sh_an[GA_C * GA_NA];
groupshared uint    sh_l0[GA_C * GA_NA];
groupshared uint    sh_l1[GA_C * GA_NA];
groupshared uint    sh_l2[GA_C * GA_NA];
groupshared uint    sh_l3[GA_C * GA_NA];
groupshared uint    sh_hits;

#include "spec_gen.inc"

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
    uint ovf_base = kparam.total_lps;
    uint ovf_cap  = kparam.extra;
    uint a_cls[GA_NA];
    for (uint k = 0; k < GA_NA; k++)
        a_cls[k] = (k < naggs) ? prog[aggoff + 4 * k] : 0;

    // empty table
    for (uint s = tid; s < GA_C; s += SCAN_WG)
    {
        sh_occ[s] = 0;
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
        sh_hits = 0;
    GroupMemoryBarrierWithGroupSync();

    uint my_hits = 0;
    for (uint page = first_page; page < last_page; page++)
    {
#include "join_walk.inc"
    }
    uint prev;
    InterlockedAdd(sh_hits, my_hits, prev);
    GroupMemoryBarrierWithGroupSync();

    // write the table out
    uint blk = 4 + wg * (1 + GA_C * GA_REC);
    if (tid == 0)
        outb[blk] = sh_hits;
    for (uint s = tid; s < GA_C; s += SCAN_WG)
    {
        uint rb = blk + 1 + s * GA_REC;
        uint o = sh_occ[s];
        outb[rb] = (o != 0) ? 1u : 0u;
        if (o == 0)
            continue;
        uint kn;
        int64_t k0, k1;
        spec_dm_key(s, kn, k0, k1);
        outb[rb + 1] = kn;
        outb[rb + 2] = uint(uint64_t(k0) & 0xffffffffu);
        outb[rb + 3] = uint(uint64_t(k0) >> 32);
        outb[rb + 4] = uint(uint64_t(k1) & 0xffffffffu);
        outb[rb + 5] = uint(uint64_t(k1) >> 32);
        for (uint i = 2; i < GA_NK; i++)
        {
            outb[rb + 2 + 2 * i] = 0;
            outb[rb + 3 + 2 * i] = 0;
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
