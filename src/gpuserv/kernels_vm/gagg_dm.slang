// GpuPreAgg kernel with GROUP BY on small-window integer keys, direct-mapped: only built as a specialized
// kernel (SPEC_PROG and SPEC_DM, see specialize.c), for the keys of a session whose groups the service has
// seen. The program's key values map to table slots arithmetically (spec_dm_row: key - window base, NULL in
// the extra index of its key), so unlike gagg.slang nothing is staged, no slot is claimed and no barrier
// separates rounds: each thread evaluates its rows and adds them to their slots with shared atomics, in the
// same limb format as gagg.slang. A row whose key is outside the window, or whose min/max argument does not
// fit 32 bits, is not aggregated here: heap rows are flagged for the service (as gagg.slang does), Arrow rows
// go to an overflow list. The output has the layout of gagg.slang's with GA_C slots per workgroup, so the
// service merges it with the same code; see gagg.slang for the program header, the output and the
// overflow handling.
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif

#ifndef GA_NA
#define GA_NA 4
#endif
#define GA_NK 2
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
void gagg_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint wg = kparam.page_base + gid.x;             // page_base counts workgroups here
    uint ppw = kparam.blcksz;                       // pages per workgroup
    uint first_page = wg * ppw;
    if (first_page >= kparam.npages)
        return;                                     // uniform: the whole group leaves
    uint last_page = min(first_page + ppw, kparam.npages);
    uint tid = lid.x;

    uint ncols   = prog[3];
    uint naggs   = prog[10];
    uint aggoff  = prog[11];
    uint flags_base = kparam.total_lps;             // ARROW_SRC: the overflow list { count, records } instead
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
#ifdef ARROW_SRC
        uint nlp      = arrow_page_rows(page, ncols);
        uint row0     = arrow_page_row0(page, ncols);
#else
        uint pte      = ptab[2 * page];
        uint lp_base  = ptab[2 * page + 1];
        g_src         = (pte & 0x80000000u) != 0 ? 1u : 0u;
        uint pbase    = pte & 0x7fffffffu;
        uint pd_lower = ld16(pbase + 12);
        uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);
#endif

        for (uint base = 0; base < nlp; base += SCAN_WG)
        {
            uint lp = base + tid;
            Reg  regs[VM_MAX_REGS];
            bool hit = false;

            if (lp < nlp)
            {
#ifdef ARROW_SRC
                hit = spec_arrow_eval_row(row0 + lp, page, lp, false, regs);
#else
                hit = spec_eval_row(pbase, page, lp, false, regs);
#endif
                if (hit && !spec_scales_ok(regs))
                {
                    hit = false;                    // a numeric of another scale than the plan's:
                    g_recheck = 1;                  // the backend evaluates the row
                }
                bool overflow = false;
                if (hit)
                {
                    my_hits += 1;
                    overflow = !spec_dm_row(regs);
                }
#ifdef ARROW_SRC
                if (overflow)
                {
                    // the key is outside the window: hand the row to the service (as agagg does)
                    uint idx;
                    InterlockedAdd(outb[flags_base], 1, idx);
                    if (idx < kparam.extra)
                        spec_dm_ov(regs, flags_base + 1 + idx * OV_REC);
                }
#else
                outb[flags_base + lp_base + lp] = overflow ? 1u : (g_recheck != 0 ? 2u : 0u);   // 2: the backend evaluates the row
#endif
            }
        }
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
