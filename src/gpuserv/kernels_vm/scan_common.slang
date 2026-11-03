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
#ifdef WITH_INNER
    uint extra;         // join kernels: a fifth parameter (jgagg.slang: capacity of the overflow list)
#endif
};
[[vk::push_constant]] ConstantBuffer<PC> kparam;

struct Reg
{
    int64_t v;
    uint    isnull;
    uint    sc;         // numeric: the scale of v (value = v / 10^sc); unused by the other types
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

// the same on the bit patterns: Metal flushes denormal floats to zero in a float comparison (1.4e-45 >= 0 was true),
// so a float4 is ordered through its bits (sign-magnitude as a signed integer; -0 and +0 are equal)
int f4cmpbits(uint a, uint b)
{
    uint ma = a & 0x7fffffffu;
    uint mb = b & 0x7fffffffu;
    bool na = ma > 0x7f800000u;
    bool nb = mb > 0x7f800000u;
    if (na)
        return nb ? 0 : 1;
    if (nb)
        return -1;
    int ka = ((a & 0x80000000u) != 0u) ? -int(ma) : int(ma);
    int kb = ((b & 0x80000000u) != 0u) ? -int(mb) : int(mb);
    return ka < kb ? -1 : (ka > kb ? 1 : 0);
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

// ---------------------------------------------------------------------------------------------
// A6: temporal types and strings
//
// Strings: a register holds { byte offset (bits 0-31) | length (32-59) | source (60-61) } of the
// payload of a varlena (heap page buffers, the join inner buffer, or the constant pool of the
// program). A value stored compressed or out of line cannot be read here: deform_tuple() sets
// g_recheck and the kernel reports the row for re-evaluation by the backend.
// ---------------------------------------------------------------------------------------------
static uint g_recheck;

uint pw_s(uint src, uint wi)
{
    if (src == 3)
        return prog[wi];
#ifdef WITH_INNER
    if (src == 2)
        return innerb[wi];
#endif
    return src == 1 ? pagel[wi] : pagef[wi];
}
uint ldb_s(uint src, uint o) { return (pw_s(src, o >> 2) >> ((o & 3u) * 8u)) & 0xffu; }
uint ldw_s(uint src, uint o)    // 4 bytes at any alignment
{
    uint sh = (o & 3u) * 8u;
    if (sh == 0)
        return pw_s(src, o >> 2);
    return (pw_s(src, o >> 2) >> sh) | (pw_s(src, (o >> 2) + 1) << (32u - sh));
}
uint str_off(Reg r) { return uint(uint64_t(r.v) & 0xffffffffu); }
uint str_len(Reg r) { return uint((uint64_t(r.v) >> 32) & 0x0fffffffu); }
uint str_src(Reg r) { return uint((uint64_t(r.v) >> 60) & 3u); }

// length without trailing blanks (bpchar)
uint str_truelen(uint src, uint off, uint len)
{
    while (len > 0 && ldb_s(src, off + len - 1) == 0x20u)
        len--;
    return len;
}

// bytewise compare of n bytes: -1, 0, +1
int str_memcmp(uint sa, uint oa, uint sb, uint ob, uint n)
{
    uint i = 0;
    for (; i + 4 <= n; i += 4)
    {
        uint wa = ldw_s(sa, oa + i);
        uint wb = ldw_s(sb, ob + i);
        if (wa != wb)
        {
            for (uint j = 0; j < 4; j++)
            {
                uint ca = (wa >> (8 * j)) & 0xffu;
                uint cb = (wb >> (8 * j)) & 0xffu;
                if (ca != cb)
                    return ca < cb ? -1 : 1;
            }
        }
    }
    for (; i < n; i++)
    {
        uint ca = ldb_s(sa, oa + i);
        uint cb = ldb_s(sb, ob + i);
        if (ca != cb)
            return ca < cb ? -1 : 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// A6: numeric. A register holds { unscaled value v, scale sc } with |v| <= INT64_MAX and sc <= 255.
// Everything that does not fit (NaN, Infinity, more than 18-19 digits, an overflowing intermediate
// result) makes the instruction return SOP_RECHECK_CODE: the backend evaluates the row.
// ---------------------------------------------------------------------------------------------
// 10^d for d <= 18 (a computed loop: a constant int64 array does not survive MoltenVK/Metal)
int64_t num_p10(uint d)
{
    int64_t p = 1;
    for (uint i = 0; i < d; i++)
        p = p * 10;
    return p;
}
static const int64_t NUM_MAX = 9223372036854775807;

// v = v * 10^d; false on overflow
bool num_up(inout int64_t v, uint d)
{
    if (v == 0)
        return true;
    if (d > 18u)
        return false;
    int64_t p = num_p10(d);
    int64_t lim = NUM_MAX / p;
    if (v > lim || v < -lim)
        return false;
    v = v * p;
    return true;
}

// 16 bits at any byte offset (the payload of a 1-byte-header varlena is at an odd offset; ld16 needs 2-byte alignment)
uint ld16u(uint o) { return ld8(o) | (ld8(o + 1) << 8); }

// Decode the numeric varlena payload (plen bytes at byte offset o of the current source): false if it is
// NaN / Infinity or does not fit in { int64, scale }. Layout: see numeric.c (short and long header).
bool num_decode(uint o, uint plen, out int64_t v, out uint sc)
{
    v = 0;
    sc = 0;
    if (plen < 2)
        return false;
    uint hdr = ld16u(o);
    bool neg;
    int  weight;
    uint dscale, doff, ndig;
    if ((hdr & 0xc000u) == 0xc000u)
        return false;                               // NaN or Infinity
    if ((hdr & 0x8000u) != 0)
    {
        neg = (hdr & 0x2000u) != 0;
        dscale = (hdr >> 7) & 0x3fu;
        weight = int(hdr & 0x3fu) - (((hdr & 0x40u) != 0) ? 64 : 0);
        doff = o + 2;
        ndig = (plen - 2) / 2;
    }
    else
    {
        if (plen < 4)
            return false;
        neg = (hdr & 0x4000u) != 0;
        dscale = hdr & 0x3fffu;
        weight = int(ld16u(o + 2) ^ 0x8000u) - 32768;
        doff = o + 4;
        ndig = (plen - 4) / 2;
    }
    if (dscale > 255u || ndig > 5u)
        return false;
    int64_t acc = 0;
    for (uint i = 0; i < ndig; i++)
    {
        int64_t d = int64_t(ld16u(doff + 2u * i));
        if (acc > (NUM_MAX - d) / 10000)
            return false;
        acc = acc * 10000 + d;
    }
    int e = 4 * (weight - int(ndig) + 1) + int(dscale);
    if (acc != 0)
    {
        if (e >= 0)
        {
            if (!num_up(acc, uint(e)))
                return false;
        }
        else
        {
            uint m = uint(-e);
            if (m > 18u)
                return false;
            int64_t p = num_p10(m);
            if (acc % p != 0)
                return false;
            acc = acc / p;
        }
    }
    v = neg ? -acc : acc;
    sc = dscale;
    return true;
}

int num_cmp(int64_t va, uint sa, int64_t vb, uint sb)
{
    // the one with the smaller scale is scaled up; if that overflows its magnitude is larger
    if (sa < sb)
    {
        if (!num_up(va, sb - sa))
            return va > 0 ? 1 : -1;
    }
    else if (sb < sa)
    {
        if (!num_up(vb, sa - sb))
            return vb > 0 ? -1 : 1;
    }
    return va < vb ? -1 : (va > vb ? 1 : 0);
}

// The numeric instructions. Returns 0, or SOP_RECHECK_CODE.
uint vm_num(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    switch (op)
    {
    case SOP_NUMCONST:
        regs[dst].v = int64_t(uint64_t(k0) | (uint64_t(k1) << 32));
        regs[dst].sc = rb;
        regs[dst].isnull = ra;
        break;
    case SOP_NUMCMP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        regs[dst].isnull = 0;
        regs[dst].v = b2l(cmp_result(num_cmp(x.v, x.sc, y.v, y.sc), k0));
        break;
    }
    case SOP_NUMARITH:
    {
        Reg x = regs[ra];
        if (k0 == SOP_NUMOP_NEG || k0 == SOP_NUMOP_ABS)
        {
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
            {
                regs[dst].v = (k0 == SOP_NUMOP_NEG || x.v < 0) ? -x.v : x.v;
                regs[dst].sc = x.sc;
            }
            break;
        }
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        int64_t a = x.v;
        int64_t b = y.v;
        int64_t r;
        uint    sc;
        if (k0 == SOP_ARITH_MUL)
        {
            sc = x.sc + y.sc;
            // |a| * |b| must fit in INT64_MAX. The check is made before multiplying: a wrapped signed product
            // followed by "r / a != b" is undefined behaviour that the Metal compiler folds away.
            int64_t ma = a < 0 ? -a : a;                // |a| <= INT64_MAX (v is never INT64_MIN)
            int64_t mb = b < 0 ? -b : b;
            if (ma != 0 && mb > 9223372036854775807 / ma)
                return SOP_RECHECK_CODE;
            r = a * b;
        }
        else
        {
            sc = max(x.sc, y.sc);
            if (!num_up(a, sc - x.sc) || !num_up(b, sc - y.sc))
                return SOP_RECHECK_CODE;
            // overflow is excluded before the operation (signed wrap-around is undefined behaviour)
            if (k0 == SOP_ARITH_PL)
            {
                if ((b > 0 && a > 9223372036854775807 - b) || (b < 0 && a < -9223372036854775807 - b))
                    return SOP_RECHECK_CODE;
                r = a + b;
            }
            else
            {
                if ((b < 0 && a > 9223372036854775807 + b) || (b > 0 && a < -9223372036854775807 + b))
                    return SOP_RECHECK_CODE;
                r = a - b;
            }
        }
        if (r == (-9223372036854775807 - 1) || sc > 255u)
            return SOP_RECHECK_CODE;
        regs[dst].v = r;
        regs[dst].sc = sc;
        regs[dst].isnull = 0;
        break;
    }
    case SOP_I2N:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            if (x.v == (-9223372036854775807 - 1))
                return SOP_RECHECK_CODE;
            regs[dst].v = x.v;
            regs[dst].sc = 0;
        }
        break;
    }
    case SOP_N2I:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            int64_t m = x.v < 0 ? -x.v : x.v;
            int64_t q;
            if (x.sc == 0)
                q = m;
            else if (x.sc >= 20u)
                q = 0;
            else if (x.sc == 19u)
                q = (m >= 5000000000000000000) ? 1 : 0;
            else
            {
                int64_t p = num_p10(x.sc);
                q = m / p;
                if (m - q * p >= p - (m - q * p))
                    q += 1;                         // round half away from zero
            }
            int64_t r = x.v < 0 ? -q : q;
            if (out_of_range(r, k0))
                return SOP_RECHECK_CODE;            // PostgreSQL's range error: the backend raises it
            regs[dst].v = r;
        }
        break;
    }
    default:
        return op | 0x80000000u;
    }
    return 0;
}

// Instructions of A6 (not in the original VM). Returns 0 or the error code.
uint vm_ext(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    switch (op)
    {
    case SOP_D2TS:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            int64_t d = x.v;
            int64_t r;
            if (d == 2147483647)
                r = int64_t(9223372036854775807);
            else if (d == -2147483648)
                r = -9223372036854775807 - 1;
            else if (d >= 106751983)        // beyond the largest timestamp
            {
                if (k0 == 1)
                    return k1 | (SOP_ERR_TSRANGE << 16);
                r = int64_t(9223372036854775806);   // above every finite timestamp, below infinity
            }
            else
                r = d * 86400000000;
            regs[dst].v = r;
        }
        break;
    }
    case SOP_TS2D:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            int64_t t = x.v;
            int64_t r;
            if (t == int64_t(9223372036854775807))
                r = 2147483647;
            else if (t == -9223372036854775807 - 1)
                r = -2147483648;
            else
            {
                r = t / 86400000000;
                if (t % 86400000000 < 0)
                    r -= 1;                 // floor
            }
            regs[dst].v = r;
        }
        break;
    }
    case SOP_DATEOP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        regs[dst].isnull = 0;
        bool ainf = (x.v == 2147483647 || x.v == -2147483648);
        if (k0 == SOP_DATEOP_MI)
        {
            if (ainf || y.v == 2147483647 || y.v == -2147483648)
                return k1 | (SOP_ERR_INFSUB << 16);
            regs[dst].v = x.v - y.v;
        }
        else if (ainf)
            regs[dst].v = x.v;              // infinity does not change
        else
        {
            int64_t r = (k0 == SOP_DATEOP_PLI) ? x.v + y.v : x.v - y.v;
            if (r < -2451545 || r >= 2145031949)
                return k1 | (SOP_ERR_DATERANGE << 16);
            regs[dst].v = r;
        }
        break;
    }

    case SOP_CONSTSTR:
        regs[dst].v = int64_t(uint64_t(prog[20] * 4u + k0) | (uint64_t(k1) << 32) | (uint64_t(3u) << 60));
        regs[dst].isnull = 0;
        break;
    case SOP_STRCMP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        uint sa = str_src(x), sb = str_src(y);
        uint oa = str_off(x), ob = str_off(y);
        uint la = str_len(x), lb = str_len(y);
        int  c;
        if (k1 == 1)
        {
            la = str_truelen(sa, oa, la);
            lb = str_truelen(sb, ob, lb);
        }
        if (k0 == SOP_CMP_EQ || k0 == SOP_CMP_NE)
            c = (la != lb) ? 1 : str_memcmp(sa, oa, sb, ob, la);
        else
        {
            c = str_memcmp(sa, oa, sb, ob, min(la, lb));
            if (c == 0 && la != lb)
                c = la < lb ? -1 : 1;
        }
        regs[dst].isnull = 0;
        regs[dst].v = b2l(cmp_result(c, k0));
        break;
    }
    case SOP_STRLIKE:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        uint sa = str_src(x), sb = str_src(y);
        uint oa = str_off(x), ob = str_off(y);
        uint la = str_len(x), lb = str_len(y);
        uint shape = k0 & 0xffu;
        bool res = false;
        if ((k0 & 0x200u) != 0)
            la = str_truelen(sa, oa, la);
        if (shape == SOP_LIKE_EXACT)
            res = (la == lb) && str_memcmp(sa, oa, sb, ob, lb) == 0;
        else if (shape == SOP_LIKE_PREFIX)
            res = (la >= lb) && str_memcmp(sa, oa, sb, ob, lb) == 0;
        else if (shape == SOP_LIKE_SUFFIX)
            res = (la >= lb) && str_memcmp(sa, oa + la - lb, sb, ob, lb) == 0;
        else if (la >= lb)
        {
            if (lb == 0)
                res = true;
            else
            {
                uint first = ldb_s(sb, ob);
                for (uint i = 0; i + lb <= la; i++)
                {
                    if (ldb_s(sa, oa + i) == first && str_memcmp(sa, oa + i, sb, ob, lb) == 0)
                    {
                        res = true;
                        break;
                    }
                }
            }
        }
        if ((k0 & 0x100u) != 0)
            res = !res;
        regs[dst].isnull = 0;
        regs[dst].v = b2l(res);
        break;
    }
    case SOP_STRLEN:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            uint n = str_len(x);
            if (k0 == 1)
            {
                uint sa = str_src(x), oa = str_off(x), cnt = 0;
                for (uint i = 0; i < n; i++)
                    if ((ldb_s(sa, oa + i) & 0xc0u) != 0x80u)
                        cnt++;
                n = cnt;
            }
            regs[dst].v = int64_t(n);
        }
        break;
    }
    default:
        if (op >= SOP_NUMCONST && op <= SOP_N2I)
            return vm_num(op, dst, ra, rb, k0, k1, regs);
        return op | 0x80000000u;    // malformed program
    }
    return 0;
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
                regs[dst].v = b2l(cmp_result(f4cmpbits(uint(x.v), uint(y.v)), k0));
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
            // 64-bit overflow is excluded before the operation: a wrapped signed result checked afterwards is
            // undefined behaviour that the Metal compiler may fold away
            case SOP_ARITH_PL:
                if (w == 64)
                    bad = (y.v > 0 && x.v > 9223372036854775807 - y.v) ||
                          (y.v < 0 && x.v < (-9223372036854775807 - 1) - y.v);
                if (!bad)
                {
                    r = x.v + y.v;
                    if (w != 64)
                        bad = out_of_range(r, w);
                }
                break;
            case SOP_ARITH_MI:
                if (w == 64)
                    bad = (y.v < 0 && x.v > 9223372036854775807 + y.v) ||
                          (y.v > 0 && x.v < (-9223372036854775807 - 1) + y.v);
                if (!bad)
                {
                    r = x.v - y.v;
                    if (w != 64)
                        bad = out_of_range(r, w);
                }
                break;
            case SOP_ARITH_MUL:
                if (w == 64)
                {
                    if (x.v != 0 && y.v != 0)
                    {
                        // magnitudes as unsigned; a negative product may reach 2^63, a positive one 2^63 - 1
                        uint64_t ux = x.v < 0 ? uint64_t(-(x.v + 1)) + 1ul : uint64_t(x.v);
                        uint64_t uy = y.v < 0 ? uint64_t(-(y.v + 1)) + 1ul : uint64_t(y.v);
                        uint64_t lim = ((x.v < 0) == (y.v < 0)) ? 0x7fffffffffffffffull : 0x8000000000000000ull;
                        bad = (ux > lim / uy);
                    }
                    if (!bad)
                        r = x.v * y.v;
                }
                else
                {
                    r = x.v * y.v;
                    bad = out_of_range(r, w);
                }
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
        {
            uint e = vm_ext(op, dst, ra, rb, k0, k1, regs);
            if (e != 0)
                return e;
            break;
        }
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
            if (slot != 0xffu)
            {
                regs[slot].v = 0;
                regs[slot].isnull = 0;
                if (vt == SOP_VT_NUM)
                {
                    // numeric: decode to { unscaled, scale }; compressed, external, NaN/Infinity or too
                    // large values are evaluated by the backend
                    uint hl = 4;
                    if ((b0 & 1u) != 0)
                    {
                        hl = 1;
                        if (b0 == 1u)
                            g_recheck = 1;
                    }
                    else if ((b0 & 3u) == 2u)
                        g_recheck = 1;
                    if (g_recheck == 0)
                    {
                        int64_t nv;
                        uint    nsc;
                        if (num_decode(tbase + off + hl, sz - hl, nv, nsc))
                        {
                            regs[slot].v = nv;
                            regs[slot].sc = nsc;
                        }
                        else
                            g_recheck = 1;
                    }
                }
                else if (vt == SOP_VT_STR)
                {
                    // a string register: where the payload is (see above); a value stored out of line
                    // (b0 == 1) or compressed (4-byte header with the compression flag) is not readable
                    uint hl = 4;
                    if ((b0 & 1u) != 0)
                    {
                        hl = 1;
                        if (b0 == 1u)
                            g_recheck = 1;
                    }
                    else if ((b0 & 3u) == 2u)
                        g_recheck = 1;
                    if (g_recheck == 0)
                        regs[slot].v = int64_t(uint64_t(tbase + off + hl) | (uint64_t(sz - hl) << 32) |
                                               (uint64_t(g_src) << 60));
                }
            }
            off += sz;
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

    g_recheck = 0;
    if (lp_flags != 1)      // not LP_NORMAL
        return false;

    [unroll]
    for (uint i = 0; i < VM_MAX_REGS; i++)
    {
        regs[i].v = 0;
        regs[i].isnull = 1;
        regs[i].sc = 0;
    }
    if (!deform_tuple(pbase + lp_off, page, lp, coloff, ncols, regs))
        return false;
    if (g_recheck != 0)
        return false;           // the caller looks at g_recheck: the backend must evaluate this row

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
    if (err == SOP_RECHECK_CODE)
    {
        g_recheck = 1;          // an instruction could not evaluate this row (numeric): the backend does
        return false;
    }
    if (err != 0)
    {
        report_error(page, lp, err);
        return false;
    }
    return hit;
}
