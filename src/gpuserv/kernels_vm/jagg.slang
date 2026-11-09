// GpuPreAgg over a GpuJoin, without GROUP BY: the join kernel of join.slang (the walk is
// join_walk.inc) whose emit adds the joined row to the thread's accumulators, as agg.slang does for
// the rows of a scan, so the joined rows are never written out. One workgroup runs over a run of
// kparam.blcksz outer pages; at the end the workgroup reduces in shared memory (no atomics, nothing
// shared between workgroups) and writes one partial record per aggregate, in the format of agg.slang.
//
// The program is a join program (see join.slang) whose "projection" section computes the aggregate
// arguments and FILTER conditions; header words 10 and 11 give the number of aggregates and the
// offset of their table, 4 words each: { class, argument register, filter register, 0 }.
//
// A GiST-index join (program header word 18 nonzero, interpreter only) skips the outer rows the
// scout pass named (join_walk.inc: poison, bitmap behind the page table); a poison in this kernel
// is an error.
//
// Output (outb): [1] first error, [2],[3] page and line pointer of it; from word 4, per workgroup
// and aggregate 6 words { n, hits, A_lo, A_hi, B_lo, B_hi } as in agg.slang (hits only in the first
// aggregate of a workgroup: the number of joined rows).
#define WITH_INNER 1
#define JW_MODE 1
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif
#include "join_common.inc"

#ifndef AGG_MAX
#define AGG_MAX 12
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
void jagg_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint wg = kparam.page_base + gid.x;             // page_base counts workgroups here
    uint ppw = kparam.blcksz;                       // pages per workgroup
    uint first_page = wg * ppw;
    if (first_page >= kparam.npages)
        return;                                     // uniform: the whole group leaves
    uint last_page = min(first_page + ppw, kparam.npages);

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
#include "join_walk.inc"
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
                    bool isreg = (cls == AC_REG);
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
