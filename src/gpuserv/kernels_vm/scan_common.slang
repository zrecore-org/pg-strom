// Shared by the scan and aggregation kernels (scan.slang, agg.slang): the page buffers, the VM,
// and the evaluation of one line pointer.
//
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
#ifdef WITH_INNER
[[vk::binding(5, 0)]] StructuredBuffer<uint>    innerb;     // the join inner buffer (hash table of the inner relation)
#endif

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
static uint g_src;          // 0: pagef, 1: pagel, 2: the join inner buffer
uint pw(uint wi)
{
#ifdef WITH_INNER
    if (g_src == 2)
        return innerb[wi];
#endif
    return g_src == 1 ? pagel[wi] : pagef[wi];
}
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

// Jenkins lookup3 hash_any() of the little-endian bytes of a value of len 1, 2, 4 or 8 bytes
// (lo, hi: its low and high 32 bits): the hash PostgreSQL computes for hash joins.
uint rot32(uint x, uint k) { return (x << k) | (x >> (32u - k)); }
uint hash_any_value(uint lo, uint hi, uint len)
{
    uint a = 0x9e3779b9u + len + 3923095u;
    uint b = a;
    uint c = a;
    if (len == 1)
        a += lo & 0xffu;
    else if (len == 2)
        a += lo & 0xffffu;
    else if (len == 4)
        a += lo;
    else
    {
        a += lo;
        b += hi;
    }
    c ^= b; c -= rot32(b, 14);
    a ^= c; a -= rot32(c, 11);
    b ^= a; b -= rot32(a, 25);
    c ^= b; c -= rot32(b, 16);
    a ^= c; a -= rot32(c, 4);
    b ^= a; b -= rot32(a, 14);
    c ^= b; c -= rot32(b, 24);
    return c;
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
                        return k1 | (SOP_ERR_INTRANGE << 16);
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
            uint    why = SOP_ERR_INTRANGE;

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
                {
                    bad = true;
                    why = SOP_ERR_ZERODIV;
                }
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
                {
                    bad = true;
                    why = SOP_ERR_ZERODIV;
                }
                else if (y.v == -1)
                    r = 0;
                else
                    r = x.v % y.v;
                break;
            }
            if (bad)
                return k1 | (why << 16);
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
            uint  why = SOP_ERR_FLTOVER;
            bool  inf_in = isinf(a) || isinf(b);

            switch (k0)
            {
            case SOP_ARITH_PL: r = a + b; bad = isinf(r) && !inf_in; break;
            case SOP_ARITH_MI: r = a - b; bad = isinf(r) && !inf_in; break;
            case SOP_ARITH_MUL:
                r = a * b;
                bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && b != 0);
                if (!isinf(r))
                    why = SOP_ERR_FLTUNDER;
                break;
            default:    // SOP_ARITH_DIV
                if (b == 0)
                {
                    bad = true;
                    why = SOP_ERR_ZERODIV;
                }
                else
                {
                    r = a / b;
                    bad = (isinf(r) && !inf_in) || (r == 0 && a != 0 && !isinf(b));
                    if (!isinf(r))
                        why = SOP_ERR_FLTUNDER;
                }
                break;
            }
            if (bad)
                return k1 | (why << 16);
            regs[dst].v = int64_t(asuint(r));
            regs[dst].isnull = 0;
            break;
        }

        case SOP_HASHINIT:
            regs[dst].v = int64_t(0xffffffffu);
            regs[dst].isnull = 0;
            break;
        case SOP_HASHMERGE:
            if (regs[ra].isnull == 0)
            {
                uint64_t uv = uint64_t(regs[ra].v);
                uint h = hash_any_value(uint(uv & 0xffffffffu), uint(uv >> 32), k0);
                uint acc = uint(regs[dst].v);
                regs[dst].v = int64_t(((acc >> 3) | (acc << 29)) ^ h);
            }
            break;
        case SOP_HASHFIN:
            regs[dst].v = int64_t(uint(regs[dst].v) ^ 0xffffffffu);
            break;

        case SOP_I2F:
        {
            Reg x = regs[ra];
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
                regs[dst].v = int64_t(asuint(float(x.v)));
            break;
        }

        case SOP_F2I:
        {
            Reg x = regs[ra];
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
            {
                float f = asfloat(uint(x.v));
                if (isnan(f) || isinf(f) || f >= 9.2233720368547758e18 || f < -9.2233720368547758e18)
                    return k1 | (SOP_ERR_INTRANGE << 16);
                int64_t r = int64_t(round(f));      // round to nearest even, as rint()
                if (out_of_range(r, k0))
                    return k1 | (SOP_ERR_INTRANGE << 16);
                regs[dst].v = r;
            }
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

// Deform the columns a program needs from the heap-format tuple at byte offset tbase of the
// current source (g_src) into the registers named by the column descriptors at coloff. A join
// inner tuple (kern_hashitem) has the layout of a heap tuple whose first 8 bytes are the chain
// pointer, so the same code reads it. Returns false after report_error().
bool deform_tuple(uint tbase, uint page, uint lp, uint coloff, uint ncols,
                  inout Reg regs[VM_MAX_REGS])
{
    uint infomask2 = ld16(tbase + 18);
    uint infomask  = ld16(tbase + 20);
    uint natts     = infomask2 & 0x7ffu;
    bool hasnull   = (infomask & 1u) != 0;
    uint off       = ld8(tbase + 22);       // t_hoff

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
            report_error(page, lp, 0x7fffffffu);    // cstring attributes do not occur in heap tables
            return false;
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
    return true;
}

// Set the registers a column table loads to NULL (before deforming the next inner tuple).
void null_slots(uint coloff, uint ncols, inout Reg regs[VM_MAX_REGS])
{
    for (uint c = 0; c < ncols; c++)
    {
        uint slot = (prog[coloff + c] >> 16) & 0xffu;
        if (slot != 0xffu)
        {
            regs[slot].v = 0;
            regs[slot].isnull = 1;
        }
    }
}

// Evaluate line pointer `lp` of the page at pbase: deform the columns the program needs, run
// the qual part of the program and, for a qualifying tuple, the rest of it (projection or
// aggregate arguments). Returns true if the tuple qualifies; the registers then hold the
// computed values. A failure is recorded with report_error() and counts as "not qualifying".
bool eval_row(uint pbase, uint page, uint lp,
              uint coloff, uint ncols, uint insnoff, uint nq, uint nins, uint qreg,
              inout Reg regs[VM_MAX_REGS])
{
    uint lpd      = ld32(pbase + 24 + lp * 4);
    uint lp_off   = lpd & 0x7fffu;
    uint lp_flags = (lpd >> 15) & 3u;

    if (lp_flags != 1)      // not LP_NORMAL
        return false;

    [unroll]
    for (uint i = 0; i < VM_MAX_REGS; i++)
    {
        regs[i].v = 0;
        regs[i].isnull = 1;
    }
    if (!deform_tuple(pbase + lp_off, page, lp, coloff, ncols, regs))
        return false;

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
        report_error(page, lp, err);
        return false;
    }
    return hit;
}
