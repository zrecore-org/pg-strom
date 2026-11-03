// GpuScan kernel: one workgroup per heap page, one thread per line pointer. The row
// evaluation is in scan_common.slang; this kernel records which line pointers qualify and
// the computed projection values.
#include "scan_common.slang"

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void scan_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint page = kparam.page_base + gid.x;
    if (page >= kparam.npages)
        return;                                     // uniform: the whole group leaves

    uint coloff  = prog[0];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint ncomp   = prog[8];
    uint compoff = prog[9];

    uint pte      = ptab[2 * page];
    uint lp_base  = ptab[2 * page + 1];
    g_src         = (pte & 0x80000000u) != 0 ? 1u : 0u;
    uint pbase    = pte & 0x7fffffffu;
    uint pd_lower = ld16(pbase + 12);
    uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);
    uint val_base = 4 + kparam.total_lps;

    for (uint base = 0; base < nlp; base += SCAN_WG)
    {
        uint lp  = base + lid.x;
        bool hit = false;
        Reg  regs[VM_MAX_REGS];

        if (lp < nlp)
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
        if (lp < nlp)
        {
            outb[4 + lp_base + lp] = hit ? 1u : (g_recheck != 0 ? 2u : 0u);   // 2: the backend must evaluate the row
            if (hit)
            {
                uint vb = val_base + (lp_base + lp) * (3 * ncomp);
                for (uint k = 0; k < ncomp; k++)
                {
                    Reg r = regs[prog[compoff + k]];
                    outb[vb + 3 * k + 0] = r.isnull | (r.sc << 8);  // numeric: the scale rides above the NULL flag
                    outb[vb + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                    outb[vb + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                }
            }
        }
    }
}
