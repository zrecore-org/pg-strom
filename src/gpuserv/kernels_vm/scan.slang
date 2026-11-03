// GpuScan kernel: one workgroup per heap page, one thread per line pointer.
//
// The input is the raw pages the backend sent (already visibility-checked on the CPU) or the
// service read from the file. Each thread deforms the columns the program needs from its
// tuple, runs the qual part of the program and, for a matching tuple, the projection part
// whose results are written next to the tuple's address.
//
// Program buffer (uint words); all offsets are word indexes:
//   [0] coloff  [1] loadoff  [2] insnoff  [3] ncols     deform: walk columns 0..ncols-1
//   [4] nload   [5] nq       [6] nins     [7] qreg      nq insns of quals, then projection
//   [8] ncomp   [9] compoff (ncomp result registers)
// column descriptor: attlen (0xff = varlena, 0xfe = cstring) | align << 8 | vtype << 12
//                    | slot register << 16 (0xff: column is only skipped)
//
// Output (outb): [1] first error (FuncOpCode, 0 = none), [2],[3] page and line pointer of it;
// from word 4, one word per line pointer in page order (ptab[2 * page + 1] is the number of
// line pointers before the page): 1 if the tuple qualifies, else 0. Then, per line pointer,
// the computed projection values { isnull, lo, hi } * ncomp (written for qualifying tuples).

#define SCANVM_OP(n, v) static const uint SOP_##n = v;
#include "scan_vm_ops.inc"

#ifndef VM_MAX_REGS
#define VM_MAX_REGS 32
#endif
static const uint MAX_REGS   = VM_MAX_REGS;
static const uint INSN_WORDS = 6;
#ifndef SCAN_WG
#define SCAN_WG 64
#endif

[[vk::binding(0, 0)]] StructuredBuffer<uint>    pagef;      // pages of the relation file (or all pages, copied)
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>  outb;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    prog;
[[vk::binding(3, 0)]] StructuredBuffer<uint>    ptab;       // per page: { byte offset (bit 31: in pagel), line pointers before it }
[[vk::binding(4, 0)]] StructuredBuffer<uint>    pagel;      // the pages the backend loaded

struct PC
{
    uint npages;
    uint page_base;
    uint blcksz;        // (unused: pages are located through ptab)
    uint total_lps;     // line pointers of all pages (start of the value area is 4 + total_lps)
};
[[vk::push_constant]] ConstantBuffer<PC> kparam;

struct Reg
{
    int64_t v;
    uint    isnull;
};

// Loads from the page buffers. Tuples start on 8-byte boundaries and a column is aligned
// as its type says, so almost every access is a single aligned word. The source of a page
// is the same for the whole workgroup.
static bool g_loaded;
uint pw(uint wi) { return g_loaded ? pagel[wi] : pagef[wi]; }
uint ld8(uint o)  { return (pw(o >> 2) >> ((o & 3u) * 8u)) & 0xffu; }
uint ld16(uint o)
{
    if ((o & 3u) == 3u)
        return ld8(o) | (ld8(o + 1) << 8);
    return (pw(o >> 2) >> ((o & 2u) * 8u)) & 0xffffu;
}
uint ld32(uint o)
{
    uint sh = (o & 3u) * 8u;
    if (sh == 0)
        return pw(o >> 2);
    return (pw(o >> 2) >> sh) | (pw((o >> 2) + 1) << (32u - sh));
}
int64_t ld64(uint o) { return int64_t(uint64_t(ld32(o)) | (uint64_t(ld32(o + 4)) << 32)); }

int64_t b2l(bool c) { return c ? 1 : 0; }

bool out_of_range(int64_t r, uint w)
{
    if (w == 8)  return r < -128 || r > 127;
    if (w == 16) return r < -32768 || r > 32767;
    if (w == 32) return r < -2147483648 || r > 2147483647;
    return false;
}

// -1, 0, +1; NaN is equal to NaN and larger than everything else
int f4cmp(float a, float b)
{
    bool na = isnan(a);
    bool nb = isnan(b);
    if (na)
        return nb ? 0 : 1;
    if (nb)
        return -1;
    return a < b ? -1 : (a > b ? 1 : 0);
}

bool cmp_result(int c, uint kind)
{
    switch (kind)
    {
    case SOP_CMP_EQ: return c == 0;
    case SOP_CMP_NE: return c != 0;
    case SOP_CMP_LT: return c < 0;
    case SOP_CMP_LE: return c <= 0;
    case SOP_CMP_GT: return c > 0;
    default:         return c >= 0;
    }
}

// Runs instructions [ip, end). Returns the FuncOpCode of a failing function, 0 on success.
uint vm_run(uint insnoff, uint ip, uint end, inout Reg regs[VM_MAX_REGS])
{
    while (ip < end)
    {
        uint ib  = insnoff + ip * INSN_WORDS;
        uint op  = prog[ib + 0];
        uint dst = prog[ib + 1];
        uint ra  = prog[ib + 2];
        uint rb  = prog[ib + 3];
        uint k0  = prog[ib + 4];
        uint k1  = prog[ib + 5];

        ip++;
        switch (op)
        {
        case SOP_CONST:
            regs[dst].v = int64_t(uint64_t(k0) | (uint64_t(k1) << 32));
            regs[dst].isnull = ra;
            break;

        // three-valued AND / OR as accumulate-and-jump
        case SOP_AND:
            if (regs[ra].isnull != 0)
                regs[dst].isnull = 1;
            else if (regs[ra].v == 0)
            {
                regs[dst].v = 0;
                regs[dst].isnull = 0;
                ip = rb;
            }
            break;
        case SOP_OR:
            if (regs[ra].isnull != 0)
                regs[dst].isnull = 1;
            else if (regs[ra].v != 0)
            {
                regs[dst].v = 1;
                regs[dst].isnull = 0;
                ip = rb;
            }
            break;
        case SOP_NOT:
            regs[dst].isnull = regs[ra].isnull;
            regs[dst].v = b2l(regs[ra].v == 0);
            break;
        case SOP_ISNULL:
            regs[dst].v = b2l(regs[ra].isnull != 0);
            regs[dst].isnull = 0;
            break;
        case SOP_ISNOTNULL:
            regs[dst].v = b2l(regs[ra].isnull == 0);
            regs[dst].isnull = 0;
            break;
        case SOP_MOV:
            regs[dst] = regs[ra];
            break;

        case SOP_CAST:
        {
            Reg x = regs[ra];
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
            {
                if (k0 == 1)
                    regs[dst].v = b2l(x.v != 0);
                else
                {
                    if (out_of_range(x.v, k0))
                        return k1;
                    regs[dst].v = x.v;
                }
            }
            break;
        }

        case SOP_CMPI:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
                regs[dst].isnull = 1;
            else
            {
                regs[dst].isnull = 0;
                regs[dst].v = b2l(cmp_result(x.v < y.v ? -1 : (x.v > y.v ? 1 : 0), k0));
            }
            break;
        }

        case SOP_CMPF:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
                regs[dst].isnull = 1;
            else
            {
                regs[dst].isnull = 0;
                regs[dst].v = b2l(cmp_result(f4cmp(asfloat(uint(x.v)), asfloat(uint(y.v))), k0));
            }
            break;
        }

        case SOP_ARITH:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
            {
                regs[dst].isnull = 1;
                break;
            }
            uint    aop = k0 & 0xffu;
            uint    w   = k0 >> 8;
            int64_t r   = 0;
            bool    bad = false;

            switch (aop)
            {
            case SOP_ARITH_PL:
                r = x.v + y.v;
                bad = (w == 64) ? (((x.v ^ r) & (y.v ^ r)) < 0) : out_of_range(r, w);
                break;
            case SOP_ARITH_MI:
                r = x.v - y.v;
                bad = (w == 64) ? (((x.v ^ y.v) & (x.v ^ r)) < 0) : out_of_range(r, w);
                break;
            case SOP_ARITH_MUL:
                r = x.v * y.v;
                if (w == 64)
                {
                    if (x.v == -1)
                        bad = (y.v == -9223372036854775807 - 1);
                    else if (x.v != 0)
                        bad = (r / x.v != y.v);
                }
                else
                    bad = out_of_range(r, w);
                break;
            case SOP_ARITH_DIV:
                if (y.v == 0)
                    bad = true;
                else if (y.v == -1)
                {
                    r = -x.v;
                    bad = (w == 64) ? (x.v == -9223372036854775807 - 1) : out_of_range(r, w);
                }
                else
                    r = x.v / y.v;
                break;
            default:    // SOP_ARITH_MOD
                if (y.v == 0)
                    bad = true;
                else if (y.v == -1)
                    r = 0;
                else
                    r = x.v % y.v;
                break;
            }
            if (bad)
                return k1;
            regs[dst].v = r;
            regs[dst].isnull = 0;
            break;
        }

        case SOP_FARITH:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
            {
                regs[dst].isnull = 1;
                break;
            }
            float a = asfloat(uint(x.v));
            float b = asfloat(uint(y.v));
            float r = 0;
            bool  bad = false;
            bool  inf_in = isinf(a) || isinf(b);

            switch (k0)
            {
            case SOP_ARITH_PL: r = a + b; bad = isinf(r) && !inf_in; break;
            case SOP_ARITH_MI: r = a - b; bad = isinf(r) && !inf_in; break;
            case SOP_ARITH_MUL:
                r = a * b;
                bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && b != 0);
                break;
            default:    // SOP_ARITH_DIV
                if (b == 0)
                    bad = true;
                else
                {
                    r = a / b;
                    bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && !isinf(b));
                }
                break;
            }
            if (bad)
                return k1;
            regs[dst].v = int64_t(asuint(r));
            regs[dst].isnull = 0;
            break;
        }

        default:
            return op | 0x80000000u;    // malformed program
        }
    }
    return 0;
}

void report_error(uint page, uint lp, uint err)
{
    uint prev;
    InterlockedCompareExchange(outb[1], 0, err, prev);
    if (prev == 0)
    {
        outb[2] = page;
        outb[3] = lp;
    }
}

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void scan_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint page = kparam.page_base + gid.x;
    if (page >= kparam.npages)
        return;                                     // uniform: the whole group leaves

    uint coloff  = prog[0];
    uint loadoff = prog[1];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nload   = prog[4];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint ncomp   = prog[8];
    uint compoff = prog[9];

    uint pte      = ptab[2 * page];
    uint lp_base  = ptab[2 * page + 1];
    g_loaded      = (pte & 0x80000000u) != 0;
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
        {
            uint lpd = ld32(pbase + 24 + lp * 4);
            uint lp_off   = lpd & 0x7fffu;
            uint lp_flags = (lpd >> 15) & 3u;

            if (lp_flags == 1)      // LP_NORMAL
            {
                uint tbase = pbase + lp_off;
                uint infomask2 = ld16(tbase + 18);
                uint infomask  = ld16(tbase + 20);
                uint natts     = infomask2 & 0x7ffu;
                bool hasnull   = (infomask & 1u) != 0;
                uint off       = ld8(tbase + 22);       // t_hoff
                bool bad       = false;

                [unroll]
                for (uint i = 0; i < VM_MAX_REGS; i++)
                {
                    regs[i].v = 0;
                    regs[i].isnull = 1;
                }
                for (uint c = 0; c < ncols; c++)
                {
                    uint desc  = prog[coloff + c];
                    uint alen  = desc & 0xffu;
                    uint align = (desc >> 8) & 0xfu;
                    uint vt    = (desc >> 12) & 0xfu;
                    uint slot  = (desc >> 16) & 0xffu;
                    bool isnull = (c >= natts) ||
                                  (hasnull && (((ld8(tbase + 23 + (c >> 3)) >> (c & 7u)) & 1u) == 0));
                    if (isnull)
                        continue;       // register stays NULL, no space is used

                    if (alen == 0xffu)
                    {
                        // varlena: a zero byte is padding, otherwise a 1-byte header needs no alignment
                        uint b0 = ld8(tbase + off);
                        if (b0 == 0)
                        {
                            off = (off + align - 1) & ~(align - 1);
                            b0 = ld8(tbase + off);
                        }
                        uint sz;
                        if ((b0 & 1u) != 0)
                            sz = (b0 == 1u) ? ((ld8(tbase + off + 1) == 18u) ? 18u : 10u) : ((b0 >> 1) & 0x7fu);
                        else
                            sz = (ld32(tbase + off) >> 2) & 0x3fffffffu;
                        off += sz;
                        if (slot != 0xffu)      // only NULL-ness of a varlena is available
                        {
                            regs[slot].v = 0;
                            regs[slot].isnull = 0;
                        }
                    }
                    else if (alen == 0xfeu)
                    {
                        bad = true;     // cstring attributes do not occur in heap tables
                        break;
                    }
                    else
                    {
                        off = (off + align - 1) & ~(align - 1);
                        if (slot != 0xffu && vt == SOP_VT_NULLONLY)
                        {
                            regs[slot].v = 0;
                            regs[slot].isnull = 0;
                        }
                        else if (slot != 0xffu)
                        {
                            int64_t v = 0;
                            if (alen == 1)
                            {
                                uint b = ld8(tbase + off);
                                v = (vt == SOP_VT_INT) ? int64_t(int(b ^ 0x80u) - 128) : int64_t(b);
                            }
                            else if (alen == 2)
                            {
                                uint h = ld16(tbase + off);
                                v = int64_t(int(h ^ 0x8000u) - 32768);
                            }
                            else if (alen == 4)
                            {
                                uint u = ld32(tbase + off);
                                v = (vt == SOP_VT_FLOAT4) ? int64_t(u) : int64_t(int(u));
                            }
                            else
                                v = ld64(tbase + off);
                            regs[slot].v = v;
                            regs[slot].isnull = 0;
                        }
                        off += alen;
                    }
                }
                if (bad)
                    report_error(page, lp, 0x7fffffffu);
                else
                {
                    uint err = 0;

                    if (qreg == 0xffffffffu)
                        hit = true;
                    else
                    {
                        err = vm_run(insnoff, 0, nq, regs);
                        if (err == 0)
                            hit = (regs[qreg].isnull == 0 && regs[qreg].v != 0);
                    }
                    if (err == 0 && hit && nins > nq)
                        err = vm_run(insnoff, nq, nins, regs);
                    if (err != 0)
                    {
                        report_error(page, lp, err);
                        hit = false;
                    }
                }
            }
        }

        if (lp < nlp)
        {
            outb[4 + lp_base + lp] = hit ? 1u : 0u;
            if (hit)
            {
                uint vb = val_base + (lp_base + lp) * (3 * ncomp);
                for (uint k = 0; k < ncomp; k++)
                {
                    Reg r = regs[prog[compoff + k]];
                    outb[vb + 3 * k + 0] = r.isnull;
                    outb[vb + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                    outb[vb + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                }
            }
        }
    }
}
