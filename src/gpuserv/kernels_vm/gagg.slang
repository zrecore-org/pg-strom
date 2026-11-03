// GpuPreAgg kernel with GROUP BY: one workgroup runs over a run of pages and keeps a hash table
// of its groups in shared memory; the service merges the workgroups' tables.
//
// No thread ever waits for another (the Vulkan model gives no forward progress guarantee, and
// the base tier has no memory model): the work is done in rounds of WG rows, separated by
// barriers.
//   A  every thread evaluates its row and stages the key and the aggregate arguments in shared
//      memory
//   B  every thread finds or claims the slot of its key in the table: a slot is claimed with
//      one compare-and-swap that stores the claiming thread's index, and a thread that meets a
//      claimed slot compares its key with the claimer's staged key (published before the
//      barrier), so nobody has to wait for a key to be written
//   C  every row's own thread adds its values to the slot's accumulators with 32-bit shared
//      atomics (counts; sums as four 16-bit limbs of the value biased by 2^63, so a limb sum
//      cannot overflow 32 bits within a workgroup; min/max on the value biased by 2^31, for
//      values that fit 32 bits, others overflow to the CPU), while every slot's owning thread
//      turns a fresh slot into a committed one by copying the claimer's staged key into it
// A row whose key finds no slot within a few probes is an overflow row: it is flagged and the
// service aggregates it on the CPU.
//
// Program header: [10] aggregates, [11] aggregate table offset (4 words each, as in agg.slang),
// [12] keys, [13] offset of the key registers. Output (outb): [1] first error, [2],[3] page and
// line pointer; from word 4, per workgroup a block of 1 + GA_C * REC words:
//   { hits, then GA_C records { occupied, null mask, GA_NK * (lo, hi), GA_NA * 6 words } }
// An aggregate's 6 words: { n, 0, L0, L1, L2, L3 }; for sums n is the number of values and the
// sum is (L0 + L1 << 16 + L2 << 32 + L3 << 48) - n * 2^63; for min/max n is 1 if there is a value
// and L0 is the value biased by 2^31.
// followed (at kparam.total_lps's pc word: flags_base) by one word per line pointer: 1 if the row
// overflowed. The aggregate words are those of agg.slang.
#include "scan_common.slang"

#ifndef GA_NA
#define GA_NA 4
#endif
#define GA_NK 2
#ifndef GA_C
#define GA_C 128
#endif
#define GA_REC (2 + 2 * GA_NK + 6 * GA_NA)
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

groupshared uint    st_kn[SCAN_WG];                 // null mask | 0x80000000 if the row qualifies
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

    uint coloff  = prog[0];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint naggs   = prog[10];
    uint aggoff  = prog[11];
    uint nkeys   = prog[12];
    uint keyoff  = prog[13];
    uint flags_base = kparam.total_lps;
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
        sh_hits = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint page = first_page; page < last_page; page++)
    {
        uint pte      = ptab[2 * page];
        uint lp_base  = ptab[2 * page + 1];
        g_src         = (pte & 0x80000000u) != 0 ? 1u : 0u;
        uint pbase    = pte & 0x7fffffffu;
        uint pd_lower = ld16(pbase + 12);
        uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);

        for (uint base = 0; base < nlp; base += SCAN_WG)
        {
            uint lp = base + tid;
            Reg  regs[VM_MAX_REGS];
            bool hit = false;

            // A: evaluate and stage
            if (lp < nlp)
                hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
            st_slot[tid] = NO_SLOT;
            if (hit && !agg_scales_ok(aggoff, naggs, keyoff, nkeys, regs))
            {
                hit = false;                        // a numeric of another scale than the plan's:
                g_recheck = 1;                      // the backend evaluates the row
            }
            if (hit)
            {
                int64_t k0 = 0, k1 = 0;
                uint kn = 0;
                if (nkeys > 0)
                {
                    Reg r = regs[prog[keyoff]];
                    if (r.isnull != 0) kn |= 1u; else k0 = r.v;
                }
                if (nkeys > 1)
                {
                    Reg r = regs[prog[keyoff + 1]];
                    if (r.isnull != 0) kn |= 2u; else k1 = r.v;
                }
                st_key[tid * GA_NK + 0] = k0;
                st_key[tid * GA_NK + 1] = k1;
                st_hash[tid] = hash_keys(k0, k1, kn);
                st_kn[tid] = kn | 0x80000000u;
                for (uint k = 0; k < naggs; k++)
                {
                    uint cls = prog[aggoff + 4 * k];
                    uint ra  = prog[aggoff + 4 * k + 1];
                    uint rf  = prog[aggoff + 4 * k + 2];
                    bool ok  = true;
                    if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                        ok = false;                         // FILTER (WHERE ...) is not true
                    if (ok && cls != AC_COUNT_ANY && regs[ra].isnull != 0)
                        ok = false;
                    st_ok[tid * GA_NA + k] = ok ? 1u : 0u;
                    st_val[tid * GA_NA + k] = (cls == AC_COUNT_ANY || cls == AC_COUNT_COND) ? 0 : regs[ra].v;
                }
                uint prev;
                InterlockedAdd(sh_hits, 1, prev);
            }
            else
                st_kn[tid] = 0;
            GroupMemoryBarrierWithGroupSync();

            // B: find or claim the slot of the row's key
            uint slot = NO_SLOT;
            bool fits = true;                               // min/max arguments must fit 32 bits
            if (hit)
            {
                for (uint k = 0; k < naggs; k++)
                {
                    int64_t v = st_val[tid * GA_NA + k];
                    if ((a_cls[k] == AC_MIN || a_cls[k] == AC_MAX) && st_ok[tid * GA_NA + k] != 0 &&
                        (v < -2147483648 || v > 2147483647))
                        fits = false;
                }
            }
            if (hit && fits)
            {
                uint kn = st_kn[tid] & 3u;
                int64_t k0 = st_key[tid * GA_NK + 0];
                int64_t k1 = st_key[tid * GA_NK + 1];
                uint s = st_hash[tid] % GA_C;
                for (uint p = 0; p < GA_MAXPROBE; p++)
                {
                    uint o = sh_owner[s];
                    if (o == 0)
                    {
                        uint prev;
                        InterlockedCompareExchange(sh_owner[s], 0, tid + 1, prev);
                        if (prev == 0)
                        {
                            slot = s;
                            break;
                        }
                        o = prev;
                    }
                    // somebody owns the slot: is it our key?
                    uint skn;
                    int64_t s0, s1;
                    if (o == COMMITTED)
                    {
                        skn = sh_knull[s];
                        s0 = sh_key[s * GA_NK + 0];
                        s1 = sh_key[s * GA_NK + 1];
                    }
                    else
                    {
                        uint c = o - 1;                     // a thread of this round
                        skn = st_kn[c] & 3u;
                        s0 = st_key[c * GA_NK + 0];
                        s1 = st_key[c * GA_NK + 1];
                    }
                    if (skn == kn && s0 == k0 && s1 == k1)
                    {
                        slot = s;
                        break;
                    }
                    s = (s + 1) % GA_C;
                }
            }
            st_slot[tid] = slot;
            if (lp < nlp)
                outb[flags_base + lp_base + lp] = (hit && slot == NO_SLOT) ? 1u : (g_recheck != 0 ? 2u : 0u);   // 2: the backend evaluates the row
            GroupMemoryBarrierWithGroupSync();

            // C: this row's values into its slot, and fresh slots turn committed
            if (slot != NO_SLOT)
            {
                for (uint k = 0; k < naggs; k++)
                {
                    if (st_ok[tid * GA_NA + k] == 0)
                        continue;
                    uint cls = a_cls[k];
                    uint a = slot * GA_NA + k;
                    int64_t v = st_val[tid * GA_NA + k];
                    uint prev;
                    if (cls == AC_COUNT_ANY || cls == AC_COUNT_COND)
                        InterlockedAdd(sh_an[a], 1, prev);
                    else if (cls == AC_SUM)
                    {
                        uint64_t u = uint64_t(v) ^ 0x8000000000000000ull;
                        InterlockedAdd(sh_an[a], 1, prev);
                        InterlockedAdd(sh_l0[a], uint(u & 0xffffu), prev);
                        InterlockedAdd(sh_l1[a], uint((u >> 16) & 0xffffu), prev);
                        InterlockedAdd(sh_l2[a], uint((u >> 32) & 0xffffu), prev);
                        InterlockedAdd(sh_l3[a], uint(u >> 48), prev);
                    }
                    else
                    {
                        uint b = uint(int(v)) ^ 0x80000000u;
                        InterlockedOr(sh_an[a], 1, prev);
                        if (cls == AC_MIN)
                            InterlockedMin(sh_l0[a], b, prev);
                        else
                            InterlockedMax(sh_l0[a], b, prev);
                    }
                }
            }
            for (uint s = tid; s < GA_C; s += SCAN_WG)
            {
                uint o = sh_owner[s];
                if (o != 0 && o != COMMITTED)
                {
                    uint c = o - 1;
                    sh_knull[s] = st_kn[c] & 3u;
                    sh_key[s * GA_NK + 0] = st_key[c * GA_NK + 0];
                    sh_key[s * GA_NK + 1] = st_key[c * GA_NK + 1];
                    sh_owner[s] = COMMITTED;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
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
