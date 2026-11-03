// GpuScan kernel for Apache Arrow record batches (Arrow_Fdw): one thread per row. The row
// evaluation is the scan VM of scan_common.slang; only the way a row's columns are loaded differs
// from the heap kernel (scan.slang): columns are columnar buffers with validity bitmaps.
//
// Bindings are those of scan_common.slang with a different meaning:
//   pagef  the Arrow data (a copy of the record batch's column buffers, or a mapped file window)
//   ptab   the column table, 8 words per column of the program's column list:
//            [0] validity bitmap byte offset (0xffffffff: no bitmap, no NULLs)  [1] its length in bytes
//            [2] values byte offset   [3] width of a value in bytes (0: bool, a bit per value)
//            [4] mode: 0 normal, 1 all NULL, 2 constant (a virtual column)  [5],[6] the constant (lo, hi)
//            [7] length of the values in bytes (a row beyond it reads as NULL)
//   pagel  unused (a placeholder)
// Push constants (kparam): npages = rows of this pass, page_base = first workgroup of this
// dispatch, blcksz = absolute index of the pass's first row, total_lps = rows of this pass.
//
// Output (outb): [1] first error (FuncOpCode, 0 = none), [2] row (absolute) of it; from word 4,
// one word per row of the pass: 1 if the row qualifies; then per row { isnull, lo, hi } * ncomp,
// the computed projection values (written for qualifying rows).
#include "scan_common.slang"

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void arrow_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint rel = (kparam.page_base + gid.x) * SCAN_WG + lid.x;
    if (rel >= kparam.npages)
        return;
    uint row = kparam.blcksz + rel;                 // the row of the record batch

    uint coloff  = prog[0];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint ncomp   = prog[8];
    uint compoff = prog[9];

    Reg regs[VM_MAX_REGS];
    [unroll]
    for (uint i = 0; i < VM_MAX_REGS; i++)
    {
        regs[i].v = 0;
        regs[i].isnull = 1;
    }
    g_src = 0;

    for (uint c = 0; c < ncols; c++)
    {
        uint desc = prog[coloff + c];
        uint slot = (desc >> 16) & 0xffu;
        if (slot == 0xffu)
            continue;
        uint vt = (desc >> 12) & 0xfu;
        uint t  = c * 8;
        uint mode = ptab[t + 4];

        if (mode == 1)
            continue;                               // NULL in every row
        if (mode == 2)
        {
            regs[slot].v = int64_t(uint64_t(ptab[t + 5]) | (uint64_t(ptab[t + 6]) << 32));
            regs[slot].isnull = 0;
            continue;
        }
        uint noff = ptab[t + 0];
        if (noff != 0xffffffffu)
        {
            uint nbyte = row >> 3;
            if (nbyte >= ptab[t + 1] || ((ld8(noff + nbyte) >> (row & 7u)) & 1u) == 0)
                continue;                           // NULL
        }
        if (vt == SOP_VT_NULLONLY)
        {
            regs[slot].v = 0;
            regs[slot].isnull = 0;
            continue;
        }
        uint voff = ptab[t + 2];
        uint w    = ptab[t + 3];
        uint vlen = ptab[t + 7];
        int64_t v = 0;
        if (w == 0 ? ((row >> 3) >= vlen) : ((row + 1) * w > vlen))
            continue;                               // short buffer: NULL
        if (vt == SOP_VT_BOOL)
            v = int64_t((ld8(voff + (row >> 3)) >> (row & 7u)) & 1u);
        else if (w == 1)
        {
            uint b = ld8(voff + row);
            v = int64_t(int(b ^ 0x80u) - 128);
        }
        else if (w == 2)
        {
            uint h = ld16(voff + row * 2);
            v = int64_t(int(h ^ 0x8000u) - 32768);
        }
        else if (w == 4)
        {
            uint u = ld32(voff + row * 4);
            v = (vt == SOP_VT_FLOAT4) ? int64_t(u) : int64_t(int(u));
        }
        else
            v = ld64(voff + row * 8);
        regs[slot].v = v;
        regs[slot].isnull = 0;
    }

    bool hit = true;
    uint err = 0;
    if (qreg != 0xffffffffu)
    {
        err = vm_run(insnoff, 0, nq, regs);
        if (err == 0)
            hit = (regs[qreg].isnull == 0 && regs[qreg].v != 0);
    }
    if (err == 0 && hit && nins > nq)
        err = vm_run(insnoff, nq, nins, regs);
    if (err != 0)
    {
        report_error(row, 0, err);
        hit = false;
    }
    outb[4 + rel] = hit ? 1u : 0u;
    if (hit)
    {
        uint vb = 4 + kparam.total_lps + rel * (3 * ncomp);
        for (uint k = 0; k < ncomp; k++)
        {
            Reg r = regs[prog[compoff + k]];
            outb[vb + 3 * k + 0] = r.isnull;
            outb[vb + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
            outb[vb + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
        }
    }
}
