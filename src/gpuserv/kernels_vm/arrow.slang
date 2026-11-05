// GpuScan kernel for Apache Arrow record batches (Arrow_Fdw): one thread per row. The row
// evaluation is the scan VM of scan_common.slang; only the way a row's columns are loaded differs
// from the heap kernel (scan.slang): columns are columnar buffers with validity bitmaps.
//
// The bindings and the column table are described in arrow_load.inc.
// Push constants (kparam): npages = rows of this pass, page_base = first workgroup of this
// dispatch, blcksz = absolute index of the pass's first row, total_lps = rows of this pass.
//
// Output (outb): [1] first error (FuncOpCode, 0 = none), [2] row (absolute) of it; from word 4,
// one word per row of the pass: 1 if the row qualifies; then per row { isnull, lo, hi } * ncomp,
// the computed projection values (written for qualifying rows).
#define ARROW_SCAN 1               // four push-constant words, see scan_common.slang
#include "scan_common.slang"
#include "arrow_load.inc"
#ifdef SPEC_PROG
#include "spec_gen.inc"           // A6: the program baked in (specialize.c)
#endif

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
#ifdef SPEC_PROG
    bool hit = spec_arrow_eval_row(row, row, 0, false, regs);
#else
    bool hit = arrow_eval_row(row, row, 0, coloff, ncols, insnoff, nq, nins, qreg, regs);
#endif
    outb[4 + rel] = hit ? 1u : 0u;
    if (hit)
    {
        uint vb = 4 + kparam.total_lps + rel * (3 * ncomp);
#ifdef SPEC_PROG
        spec_store_comp(regs, vb);
#else
        for (uint k = 0; k < ncomp; k++)
        {
            Reg r = regs[prog[compoff + k]];
            outb[vb + 3 * k + 0] = r.isnull | (r.sc << 8);  // numeric: the scale rides above the NULL flag
            outb[vb + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
            outb[vb + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
        }
#endif
    }
}
