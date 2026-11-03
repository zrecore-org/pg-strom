// GpuJoin kernel (hash joins of up to JOIN_MAXD inner relations): one workgroup per heap page, one
// thread per outer line pointer. A thread evaluates the outer scan quals and then walks the join
// tree depth first as a small state machine: at depth d it looks at the next item of the hash
// chain of relation d (the backend's KDS_FORMAT_HASH buffer, imported as innerb), compares the
// stored hash, deforms the inner tuple and runs the join quals; a match goes one depth deeper (the
// hash value of depth d+1 is computed from the columns loaded so far), at the last depth it runs
// the projection and appends a record. An exhausted chain goes back one depth. For a LEFT OUTER
// join an exhausted chain without a match continues with NULL inner columns. Records are appended
// with one atomic add per subgroup.
//
// Program header (words): [0] outer column table, [2] instructions, [3] outer columns, [5] end of the
// scan quals, [6] end of the program, [7] scan qual register, [8] computed values, [9] their
// registers, [14] number of inner relations, [15] depth table, [16] first projection instruction.
// The depth table has 12 words per relation: [0] hash begin, [1] hash end = join qual begin,
// [2] join qual end, [3] hash register, [4] join qual register, [5] its column table, [6] its
// columns, [7] byte offset of the end of its KDS in innerb, [8] word offset of its hash slots,
// [9] number of hash slots, [10] LEFT OUTER, [11] 1: SEMI, 2: ANTI.
//
// Output (outb): [0] number of matches (may exceed the capacity: the host retries), [1] first
// error, [2],[3] page and line pointer; from word 4, per match { page, lp, byte offset of the inner
// item in innerb for each relation (0xffffffff: none, LEFT OUTER), (isnull, lo, hi) * computed
// values }.
#define WITH_INNER 1
#include "scan_common.slang"

#define JOIN_MAXD 4
#define JDT_WORDS 12
static const uint NO_ITEM = 0xffffffffu;

// the first item of the chain a hash value selects in relation d (dt: its depth table row)
uint chain_first(uint dt, uint h)
{
    uint kds_end = prog[dt + 7];
    uint wi = prog[dt + 8] + (h % prog[dt + 9]) * 2;
    g_src = 2;
    uint64_t off = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
    return (off != 0 && off != 0xffffffffffffffffull) ? kds_end - uint(off) : NO_ITEM;
}

// the item after `item` in its chain (the chain pointer is at the start of the item)
uint chain_next(uint dt, uint item)
{
    uint kds_end = prog[dt + 7];
    uint wi = item / 4;
    g_src = 2;
    uint64_t nx = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
    return (nx == 0 || nx == 0xffffffffffffffffull) ? NO_ITEM : kds_end - uint(nx);
}

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
    uint nd       = prog[14];
    uint dtab     = prog[15];
    uint proj_begin = prog[16];
    uint cap      = kparam.total_lps;               // capacity of the match records
    uint rec_words = 2 + nd + 3 * ncomp;

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
        uint nxt[JOIN_MAXD + 1];                    // the next chain item to look at, per depth
        uint cur[JOIN_MAXD + 1];                    // the item matched at each depth
        uint hv[JOIN_MAXD + 1];                     // the hash value at each depth
        uint d = 0;                                 // current depth; 0: this lane is done
        uint mflag = 0;                             // bit d: depth d has found a match
        for (uint i = 0; i <= JOIN_MAXD; i++)
        {
            nxt[i] = NO_ITEM;
            cur[i] = NO_ITEM;
            hv[i] = 0;
        }

        g_src = outer_src;
        // the outer row: deform and scan quals only (nins = nq)
        if (lp < nlp)
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
        if (hit)
        {
            uint dt = dtab;
            uint err = vm_run(insnoff, prog[dt], prog[dt + 1], regs);
            if (err != 0)
                report_error(page, lp, err);
            else
            {
                hv[1] = uint(regs[prog[dt + 3]].v);
                nxt[1] = chain_first(dt, hv[1]);
                d = 1;
            }
        }

        // every lane of the subgroup goes on until all lanes are done
        [loop]
        while (true)
        {
            if (!WaveActiveAnyTrue(d != 0))
                break;
            bool emit = false;
            if (d != 0)
            {
                uint dt = dtab + JDT_WORDS * (d - 1);
                uint it = nxt[d];
                bool ok = false;
                if (it != NO_ITEM)
                {
                    nxt[d] = chain_next(dt, it);
                    g_src = 2;
                    if (ld32(it + 12) == hv[d])                 // the stored hash value
                    {
                        null_slots(prog[dt + 5], prog[dt + 6], regs);
                        if (deform_tuple(it, page, lp, prog[dt + 5], prog[dt + 6], regs))
                        {
                            uint err = vm_run(insnoff, prog[dt + 1], prog[dt + 2], regs);
                            if (err == 0)
                            {
                                uint jq = prog[dt + 4];
                                ok = (jq == 0xffffffffu) || (regs[jq].isnull == 0 && regs[jq].v != 0);
                            }
                            else
                            {
                                report_error(page, lp, err);
                                d = 0;
                            }
                        }
                    }
                    if (ok)
                    {
                        mflag |= (1u << d);
                        cur[d] = it;
                        if (prog[dt + 11] != 0)
                        {
                            // semi and anti joins: the first match decides, stop walking the chain
                            nxt[d] = NO_ITEM;
                            if (prog[dt + 11] == 2)
                                ok = false;                     // anti: the outer row is dropped
                        }
                    }
                }
                else if ((prog[dt + 10] != 0 || prog[dt + 11] == 2) && ((mflag >> d) & 1u) == 0)
                {
                    // LEFT OUTER or ANTI: the chain is done and nothing matched: NULL inner columns
                    mflag |= (1u << d);
                    cur[d] = NO_ITEM;
                    null_slots(prog[dt + 5], prog[dt + 6], regs);
                    ok = true;
                }
                else
                    d = d - 1;                                  // back to the parent (0: done)

                if (ok && d != 0)
                {
                    if (d == nd)
                    {
                        emit = true;
                        if (nins > proj_begin)
                        {
                            uint err = vm_run(insnoff, proj_begin, nins, regs);
                            if (err != 0)
                            {
                                report_error(page, lp, err);
                                emit = false;
                                d = 0;
                            }
                        }
                    }
                    else
                    {
                        d = d + 1;
                        mflag &= ~(1u << d);
                        uint dt2 = dtab + JDT_WORDS * (d - 1);
                        uint err = vm_run(insnoff, prog[dt2], prog[dt2 + 1], regs);
                        if (err != 0)
                        {
                            report_error(page, lp, err);
                            d = 0;
                        }
                        else
                        {
                            hv[d] = uint(regs[prog[dt2 + 3]].v);
                            nxt[d] = chain_first(dt2, hv[d]);
                        }
                    }
                }
            }

            uint cnt  = WaveActiveCountBits(emit);
            if (cnt > 0)
            {
                uint excl = WavePrefixCountBits(emit);
                uint pos  = 0;
                if (WaveIsFirstLane())
                    InterlockedAdd(outb[0], cnt, pos);
                pos = WaveReadLaneFirst(pos);
                if (emit)
                {
                    uint idx = pos + excl;
                    if (idx < cap)
                    {
                        uint rb = 4 + idx * rec_words;
                        outb[rb + 0] = page;
                        outb[rb + 1] = lp;
                        for (uint k = 0; k < nd; k++)
                            outb[rb + 2 + k] = cur[k + 1];
                        for (uint k = 0; k < ncomp; k++)
                        {
                            Reg r = regs[prog[compoff + k]];
                            outb[rb + 2 + nd + 3 * k + 0] = r.isnull;
                            outb[rb + 2 + nd + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                            outb[rb + 2 + nd + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                        }
                    }
                }
            }
        }
    }
}
