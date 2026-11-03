// GpuJoin kernel (inner hash join, one inner relation): one workgroup per heap page, one thread per
// outer line pointer. A thread evaluates the outer scan quals, computes the join key hash, walks
// the hash chain of the inner relation (the backend's KDS_FORMAT_HASH buffer, imported as innerb),
// compares the stored hash, deforms the inner tuple, runs the join quals and, for a match, the
// projection part of the program; matches are written to the output with one atomic add per
// subgroup.
//
// Program header (words): [0] outer column table, [2] instructions, [3] outer columns, [5] end of the
// scan quals, [6] end of the program, [7] scan qual register, [8] computed values, [9] their
// registers, [14] end of the hash section, [15] end of the join qual section, [16] hash register,
// [17] join qual register, [18] inner column table, [19] inner columns, [20] byte offset of the
// end of the inner KDS in innerb, [21] word offset of its hash slots, [22] number of hash slots.
// Instructions: [5, 14) hash, [14, 15) join quals, [15, 6) projection.
//
// Output (outb): [0] number of matches (may exceed the capacity: the host retries), [1] first
// error, [2],[3] page and line pointer; from word 4, per match { page, lp, byte offset of the
// inner item in innerb, (isnull, lo, hi) * computed values }.
#define WITH_INNER 1
#include "scan_common.slang"

static const uint NO_ITEM = 0xffffffffu;

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void join_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint page = kparam.page_base + gid.x;
    if (page >= kparam.npages)
        return;                                     // uniform: the whole group leaves

    uint coloff   = prog[0];
    uint insnoff  = prog[2];
    uint ncols    = prog[3];
    uint nq       = prog[5];
    uint nins     = prog[6];
    uint qreg     = prog[7];
    uint ncomp    = prog[8];
    uint compoff  = prog[9];
    uint hash_end = prog[14];
    uint join_end = prog[15];
    uint hash_reg = prog[16];
    uint jq_reg   = prog[17];
    uint icoloff  = prog[18];
    uint incols   = prog[19];
    uint kds_end  = prog[20];
    uint hslot_w  = prog[21];
    uint nslots   = prog[22];
    uint cap      = kparam.total_lps;               // capacity of the match records
    uint rec_words = 3 + 3 * ncomp;

    uint pte      = ptab[2 * page];
    uint outer_src = (pte & 0x80000000u) != 0 ? 1u : 0u;
    uint pbase    = pte & 0x7fffffffu;
    g_src         = outer_src;
    uint pd_lower = ld16(pbase + 12);
    uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);

    for (uint base = 0; base < nlp; base += SCAN_WG)
    {
        uint lp = base + lid.x;
        Reg  regs[VM_MAX_REGS];
        bool hit = false;
        uint h = 0;
        uint item = NO_ITEM;

        g_src = outer_src;
        // the outer row: deform and scan quals only (nins = nq)
        if (lp < nlp)
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
        if (hit)
        {
            uint err = vm_run(insnoff, nq, hash_end, regs);
            if (err != 0)
            {
                report_error(page, lp, err);
                hit = false;
            }
            else
            {
                h = uint(regs[hash_reg].v);
                g_src = 2;
                uint wi = hslot_w + (h % nslots) * 2;
                uint64_t off = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
                if (off != 0 && off != 0xffffffffffffffffull)
                    item = kds_end - uint(off);
            }
        }

        // walk the chains: every lane of the subgroup goes on until all chains are done
        [loop]
        while (true)
        {
            bool active = (item != NO_ITEM);
            if (!WaveActiveAnyTrue(active))
                break;
            bool match = false;
            if (active)
            {
                g_src = 2;
                if (ld32(item + 12) == h)               // the stored hash value
                {
                    null_slots(icoloff, incols, regs);
                    if (deform_tuple(item, page, lp, icoloff, incols, regs))
                    {
                        uint err = vm_run(insnoff, hash_end, join_end, regs);
                        if (err == 0)
                        {
                            match = (jq_reg == 0xffffffffu) ||
                                    (regs[jq_reg].isnull == 0 && regs[jq_reg].v != 0);
                            if (match && nins > join_end)
                            {
                                err = vm_run(insnoff, join_end, nins, regs);
                                if (err != 0)
                                    match = false;
                            }
                        }
                        if (err != 0)
                            report_error(page, lp, err);
                    }
                }
            }
            uint cnt  = WaveActiveCountBits(match);
            uint excl = WavePrefixCountBits(match);
            uint pos  = 0;
            if (WaveIsFirstLane() && cnt > 0)
                InterlockedAdd(outb[0], cnt, pos);
            pos = WaveReadLaneFirst(pos);
            if (match)
            {
                uint idx = pos + excl;
                if (idx < cap)
                {
                    uint rb = 4 + idx * rec_words;
                    outb[rb + 0] = page;
                    outb[rb + 1] = lp;
                    outb[rb + 2] = item;
                    for (uint k = 0; k < ncomp; k++)
                    {
                        Reg r = regs[prog[compoff + k]];
                        outb[rb + 3 + 3 * k + 0] = r.isnull;
                        outb[rb + 3 + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                        outb[rb + 3 + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                    }
                }
            }
            if (active)
            {
                uint wi = item / 4;                     // the chain pointer is at the item start
                g_src = 2;
                uint64_t nx = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
                item = (nx == 0 || nx == 0xffffffffffffffffull) ? NO_ITEM : kds_end - uint(nx);
            }
        }
    }
}
