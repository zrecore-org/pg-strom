// GpuPreAgg kernel without GROUP BY: one workgroup per run of kparam.blcksz (here: pages per
// workgroup) heap pages; each thread aggregates the qualifying tuples of its line pointers over
// the whole run, then the workgroup reduces and writes one partial record per aggregate. No atomics and nothing is shared between workgroups:
// the service adds the per-page records up (in 128-bit where needed).
//
// Program header words 10 and 11 give the number of aggregates and the offset of their
// table, 4 words each: { class, argument register, filter register, 0 } (0xffffffff: none).
//
// Output (outb): [1] first error, [2],[3] page and line pointer of it; from word 4, per workgroup
// and aggregate 6 words { n, hits, A_lo, A_hi, B_lo, B_hi } (hits, only in the first aggregate
// of a workgroup: the number of qualifying tuples):
//   count(*), count(x): n = count
//   sum, avg of integers: n = number of non-NULL values, A = sum of the low 32 bits of the
//                         values, B = sum of the high 32 bits (sign-extended): the sum is
//                         B * 2^32 + A, exact in 128 bits
//   min, max: n = 1 if any value, A = the value
// After the records, at word kparam.total_lps, one word per line pointer (ptab[2 * page + 1] is the number
// of line pointers before the page): 2 if the row needs a string the kernel cannot read (compressed or
// stored out of line), which the service hands to the backend; 0 otherwise.
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif

#ifndef AGG_MAX
#define AGG_MAX 12
#endif
// the row-flag word of line pointer `lp` of `page`; an Arrow record batch (virtual pages of AR_VP rows) has no line
// pointer table: its flags are one word per row of the pass
#ifdef ARROW_SRC
#define FLAG_AT(page, lp) (kparam.total_lps + (page) * AR_VP + (lp))
#else
#define FLAG_AT(page, lp) (kparam.total_lps + ptab[2 * (page) + 1] + (lp))
#endif
static const uint AC_COUNT_ANY  = 1;
static const uint AC_COUNT_COND = 2;
static const uint AC_SUM        = 3;
static const uint AC_MIN        = 4;
static const uint AC_MAX        = 5;
static const uint AC_FSUM       = 6;            // float8 sum / avg, min, max (A10): see fagg_acc()
static const uint AC_FMIN       = 7;
static const uint AC_FMAX       = 8;
static const uint AC_VAR        = 10;           // variance / stddev (n, Sx, Sxx), regr head (11) + continuation (12): see vs_acc()
static const uint AC_REG        = 11;
static const uint AC_VCONT      = 12;
static const uint NONE_REG      = 0xffffffffu;

groupshared uint    sh_h[SCAN_WG];
groupshared uint    sh_n[AGG_MAX * SCAN_WG];
groupshared int64_t sh_a[AGG_MAX * SCAN_WG];
groupshared int64_t sh_b[AGG_MAX * SCAN_WG];

#ifdef SPEC_PROG
#include "spec_gen.inc"           // A6: the program baked in (specialize.c)
#endif

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void agg_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint wg = kparam.page_base + gid.x;             // page_base counts workgroups here
    uint ppw = kparam.blcksz;                       // pages per workgroup
    uint first_page = wg * ppw;
    if (first_page >= kparam.npages)
        return;                                     // uniform: the whole group leaves
    uint last_page = min(first_page + ppw, kparam.npages);

    uint coloff  = prog[0];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint naggs   = prog[10];
    uint aggoff  = prog[11];

    uint    hits = 0;
    uint    an[AGG_MAX];
    int64_t aa[AGG_MAX];
    int64_t ab[AGG_MAX];
    for (uint k = 0; k < AGG_MAX; k++)
    {
        an[k] = 0;
        aa[k] = 0;
        ab[k] = 0;
    }

    for (uint page = first_page; page < last_page; page++)
    {
#ifdef ARROW_SRC
        uint nlp      = arrow_page_rows(page, ncols);
        uint row0     = arrow_page_row0(page, ncols);
#else
        uint pte      = ptab[2 * page];
        g_src         = (pte & 0x80000000u) != 0 ? 1u : 0u;
        uint pbase    = pte & 0x7fffffffu;
        uint pd_lower = ld16(pbase + 12);
        uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);
#endif

        for (uint base = 0; base < nlp; base += SCAN_WG)
        {
            uint lp = base + lid.x;
            Reg  regs[VM_MAX_REGS];
            bool hit = false;

            if (lp < nlp)
            {
#if defined(ARROW_SRC) && defined(SPEC_PROG)
                hit = spec_arrow_eval_row(row0 + lp, page, lp, false, regs);
#elif defined(ARROW_SRC)
                hit = arrow_eval_row(row0 + lp, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
#elif defined(SPEC_PROG)
                hit = spec_eval_row(pbase, page, lp, false, regs);
#else
                hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
#endif
                if (g_recheck != 0)     // flag the row for the backend (kparam.total_lps: base of the flags)
                    outb[FLAG_AT(page, lp)] = 2u;
            }
            if (!hit)
                continue;
#ifdef SPEC_PROG
            if (!spec_scales_ok(regs))
#else
            if (!agg_scales_ok(aggoff, naggs, 0, 0, regs))
#endif
            {
                // a numeric of another scale than the plan's: the backend evaluates the row
                outb[FLAG_AT(page, lp)] = 2u;
                continue;
            }
            hits += 1;
#ifdef SPEC_PROG
            {
                uint ae = spec_acc(regs, an, aa, ab);
                if (ae != 0u)
                    report_error(page, lp, ae);
            }
#else
            for (uint k = 0; k < naggs; k++)
            {
                uint cls = prog[aggoff + 4 * k];
                uint ra  = prog[aggoff + 4 * k + 1];
                uint rf  = prog[aggoff + 4 * k + 2];

                if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                    continue;                           // FILTER (WHERE ...) is not true
#ifndef VM_NO_F8
                if (cls >= AC_VAR)
                {
                    if (cls == AC_VCONT)
                        continue;
                    bool isreg = (cls >= AC_REG);
                    uint rb = (cls == AC_REG) ? prog[aggoff + 4 * (k + 1) + 1] : ra;         // the Y register of regr(Y, X)
                    if (regs[ra].isnull != 0 || (isreg && regs[rb].isnull != 0))
                        continue;
                    VS s;
                    s.n = an[k]; s.a0 = aa[k]; s.b0 = ab[k];
                    s.a1 = 0; s.b1 = 0; s.a2 = 0;
                    if (isreg)
                    {
                        s.a1 = aa[k + 1]; s.b1 = ab[k + 1]; s.a2 = aa[k + 2];
                    }
                    uint ae = 0u;
                    s = vs_acc(cls, s, uint64_t(regs[ra].v), uint64_t(regs[rb].v), ae);
                    an[k] = s.n; aa[k] = s.a0; ab[k] = s.b0;
                    if (isreg)
                    {
                        an[k + 1] = s.n; aa[k + 1] = s.a1; ab[k + 1] = s.b1;
                        an[k + 2] = s.n; aa[k + 2] = s.a2;
                    }
                    if (ae != 0u)
                        report_error(page, lp, ae);
                    continue;
                }
#endif
                if (cls == AC_COUNT_ANY)
                    an[k] += 1;
                else if (regs[ra].isnull == 0)
                {
                    int64_t v = regs[ra].v;
                    if (cls == AC_COUNT_COND)
                        an[k] += 1;
                    else if (cls == AC_SUM)
                    {
                        an[k] += 1;
                        aa[k] += int64_t(uint(v));      // low 32 bits, zero-extended
                        ab[k] += (v >> 32);             // high 32 bits, sign-extended
                    }
                    else if (cls >= AC_FSUM)
                    {
                        uint ae = fagg_acc(cls, uint64_t(v), an[k], aa[k]);
                        if (ae != 0u)
                            report_error(page, lp, ae);
                    }
                    else if (cls == AC_MIN)
                    {
                        if (an[k] == 0 || v < aa[k])
                            aa[k] = v;
                        an[k] = 1;
                    }
                    else
                    {
                        if (an[k] == 0 || v > aa[k])
                            aa[k] = v;
                        an[k] = 1;
                    }
                }
            }
#endif
        }
    }

    // reduce the workgroup
    sh_h[lid.x] = hits;
    for (uint k = 0; k < naggs; k++)
    {
        sh_n[k * SCAN_WG + lid.x] = an[k];
        sh_a[k * SCAN_WG + lid.x] = aa[k];
        sh_b[k * SCAN_WG + lid.x] = ab[k];
    }
    for (uint stride = SCAN_WG / 2; stride > 0; stride >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (lid.x < stride)
        {
            sh_h[lid.x] += sh_h[lid.x + stride];
            for (uint k = 0; k < naggs; k++)
            {
                uint cls = prog[aggoff + 4 * k];
                uint x = k * SCAN_WG + lid.x;
                uint y = x + stride;

#ifndef VM_NO_F8
                if (cls >= AC_VAR)
                {
                    if (cls == AC_VCONT)
                        continue;
                    bool isreg = (cls >= AC_REG);
                    uint x1 = x + SCAN_WG, x2 = x + 2 * SCAN_WG;
                    VS sa, sb;
                    sa.n = sh_n[x]; sa.a0 = sh_a[x]; sa.b0 = sh_b[x]; sa.a1 = 0; sa.b1 = 0; sa.a2 = 0;
                    sb.n = sh_n[y]; sb.a0 = sh_a[y]; sb.b0 = sh_b[y]; sb.a1 = 0; sb.b1 = 0; sb.a2 = 0;
                    if (isreg)
                    {
                        sa.a1 = sh_a[x1]; sa.b1 = sh_b[x1]; sa.a2 = sh_a[x2];
                        sb.a1 = sh_a[y + SCAN_WG]; sb.b1 = sh_b[y + SCAN_WG]; sb.a2 = sh_a[y + 2 * SCAN_WG];
                    }
                    uint ae = 0u;
                    sa = vs_merge(cls, sa, sb, ae);
                    sh_n[x] = sa.n; sh_a[x] = sa.a0; sh_b[x] = sa.b0;
                    if (isreg)
                    {
                        sh_n[x1] = sa.n; sh_a[x1] = sa.a1; sh_b[x1] = sa.b1;
                        sh_n[x2] = sa.n; sh_a[x2] = sa.a2;
                    }
                    if (ae != 0u)
                        report_error(0u, 0u, ae);
                    continue;
                }
#endif
                if (cls >= AC_FSUM)
                {
                    uint nx = sh_n[x];
                    int64_t ax = sh_a[x];
                    uint ae = fagg_merge(cls, sh_n[y], sh_a[y], nx, ax);
                    sh_n[x] = nx;
                    sh_a[x] = ax;
                    if (ae != 0u)
                        report_error(0u, 0u, ae);
                }
                else if (cls == AC_MIN || cls == AC_MAX)
                {
                    if (sh_n[y] != 0)
                    {
                        if (sh_n[x] == 0 ||
                            (cls == AC_MIN ? sh_a[y] < sh_a[x] : sh_a[y] > sh_a[x]))
                            sh_a[x] = sh_a[y];
                        sh_n[x] = 1;
                    }
                }
                else
                {
                    sh_n[x] += sh_n[y];
                    sh_a[x] += sh_a[y];
                    sh_b[x] += sh_b[y];
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0)
    {
        for (uint k = 0; k < naggs; k++)
        {
            uint rb = 4 + (wg * naggs + k) * 6;
            uint64_t a = uint64_t(sh_a[k * SCAN_WG]);
            uint64_t b = uint64_t(sh_b[k * SCAN_WG]);
            outb[rb + 0] = sh_n[k * SCAN_WG];
            outb[rb + 1] = (k == 0) ? sh_h[0] : 0;
            outb[rb + 2] = uint(a & 0xffffffffu);
            outb[rb + 3] = uint(a >> 32);
            outb[rb + 4] = uint(b & 0xffffffffu);
            outb[rb + 5] = uint(b >> 32);
        }
    }
}
