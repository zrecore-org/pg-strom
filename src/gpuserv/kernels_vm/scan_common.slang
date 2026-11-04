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
#include "f64_soft.inc"         // A10: float8 in 64-bit integer arithmetic (the device has no fp64)

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
#if defined(WITH_INNER) || defined(ARROW_SRC)
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


// ---- A6 part 2: division, rounding. The same algorithms as scan_caps.c (scan_vm_num_div & co), see there.
uint64_t num_p10u(uint d)               // 10^d, d <= 19
{
    uint64_t p = 1ul;
    for (uint i = 0; i < d; i++)
        p = p * 10ul;
    return p;
}

uint num_ndig(uint64_t m)               // decimal digits of m > 0
{
    uint n = 1;
    uint64_t p = 10ul;
    while (n < 19u && m >= p)
    {
        n++;
        p = p * 10ul;
    }
    return n;
}

// weight (base 10000) and first digit of the normalized value, as select_div_scale() sees them
void num_weight_firstdigit(int64_t v, uint sc, out int w, out uint fd)
{
    w = 0;
    fd = 0;
    if (v == 0)
        return;
    uint64_t m = uint64_t(v < 0 ? -v : v);
    int e10 = int(num_ndig(m)) - 1 - int(sc);
    w = (e10 >= 0) ? (e10 / 4) : -((-e10 + 3) / 4);
    int k = int(sc) + 4 * w;
    fd = (k >= 0) ? uint(m / num_p10u(uint(k))) : uint(m * num_p10u(uint(-k)));
}

// q = round_half_up(a * 10^E / b), a >= 0, b > 0; false if q > INT64_MAX.
// Long division in chunks of 18 decimal digits; (rem * 10^k) / b is a 128-by-64-bit division done bit by bit
// with 32-bit limb arithmetic only (there is no 128-bit integer type).
bool num_divround(uint64_t a, uint64_t b, uint E, out uint64_t q)
{
    uint64_t Q = a / b;
    uint64_t rem = a % b;
    q = 0ul;
    while (E > 0u)
    {
        uint k = min(E, 18u);
        E -= k;
        uint64_t p = num_p10u(k);
        uint64_t qc = 0ul;
        if (rem != 0ul)
        {
            // t = rem * p as { hi, lo }
            uint64_t a0 = rem & 0xfffffffful, a1 = rem >> 32;
            uint64_t b0 = p & 0xfffffffful, b1 = p >> 32;
            uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
            uint64_t mid = (p00 >> 32) + (p01 & 0xfffffffful) + (p10 & 0xfffffffful);
            uint64_t lo = (p00 & 0xfffffffful) | (mid << 32);
            uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
            // hi < b because rem < b and p < 2^64: the quotient fits 64 bits
            uint64_t r = hi;
            for (uint i = 0; i < 64u; i++)
            {
                r = (r << 1) | (lo >> 63);          // r < b < 2^63: no overflow
                lo = lo << 1;
                qc = qc << 1;
                if (r >= b)
                {
                    r -= b;
                    qc |= 1ul;
                }
            }
            rem = r;
        }
        if (Q > (9223372036854775807ul - qc) / p)
            return false;
        Q = Q * p + qc;
    }
    if (rem >= b - rem)                             // 2 * rem >= b: round half up
    {
        if (Q == 9223372036854775807ul)
            return false;
        Q += 1ul;
    }
    q = Q;
    return true;
}

// m / 10^d rounded half up (round) or truncated, for m >= 0
uint64_t num_down(uint64_t m, uint d, bool rnd)
{
    if (d == 0u)
        return m;
    if (d >= 20u)
        return 0ul;
    if (d == 19u)
        return (rnd && m >= 5000000000000000000ul) ? 1ul : 0ul;
    uint64_t p = num_p10u(d);
    uint64_t q = m / p;
    uint64_t f = m - q * p;
    if (rnd && f >= p - f)
        q += 1ul;
    return q;
}

// numeric_div (k0 = 0), numeric_mod (1), div() (2). Returns 0 or SOP_RECHECK_CODE.
uint num_divide(uint k0, Reg x, Reg y, out int64_t r, out uint rsc)
{
    r = 0;
    rsc = 0;
    if (y.v == 0)
        return SOP_RECHECK_CODE;                    // division by zero: the backend raises the error
    if (k0 == SOP_NUMDIV_DIV)
    {
        int w1, w2;
        uint fd1, fd2;
        num_weight_firstdigit(x.v, x.sc, w1, fd1);
        num_weight_firstdigit(y.v, y.sc, w2, fd2);
        int qw = w1 - w2;
        if (fd1 <= fd2)
            qw--;
        int rs = 16 - qw * 4;
        rs = max(rs, int(x.sc));
        rs = max(rs, int(y.sc));
        rs = max(rs, 0);
        rs = min(rs, 1000);
        if (rs > 255)
            return SOP_RECHECK_CODE;
        uint E = y.sc + uint(rs) - x.sc;
        uint64_t q;
        if (!num_divround(uint64_t(x.v < 0 ? -x.v : x.v), uint64_t(y.v < 0 ? -y.v : y.v), E, q))
            return SOP_RECHECK_CODE;
        r = ((x.v < 0) != (y.v < 0)) ? -int64_t(q) : int64_t(q);
        rsc = uint(rs);
        return 0;
    }
    uint sc = max(x.sc, y.sc);
    int64_t a = x.v;
    int64_t b = y.v;
    if (!num_up(a, sc - x.sc) || !num_up(b, sc - y.sc) || b == 0)
        return SOP_RECHECK_CODE;
    if (k0 == SOP_NUMDIV_MOD)
    {
        r = (b == -1) ? 0 : (a % b);
        rsc = sc;
    }
    else
    {
        // a / b truncated; a is never INT64_MIN, so b == -1 is safe
        r = a / b;
        rsc = 0;
    }
    return 0;
}

// round (k0 = 0), trunc (1), numeric(p,s) coercion (2) of x to the scale / typmod n
uint num_round(uint k0, Reg x, int64_t n, out int64_t r, out uint rsc)
{
    int64_t v = x.v;
    uint sc = x.sc;
    int64_t prec = 0;
    r = v;
    rsc = sc;
    if (k0 == SOP_NUMRND_TYPMOD)
    {
        if (n < 4)
            return 0;                               // not a valid typmod: the value as it is
        int64_t t = n - 4;
        prec = (t >> 16) & 0xffff;
        int s = int(((t & 0x7ff) ^ 0x400) - 0x400);
        if (s < 0 || prec < 1 || prec > 1000)
            return SOP_RECHECK_CODE;
        n = int64_t(s);
    }
    if (n > 255 || n < -64)
        return SOP_RECHECK_CODE;
    uint64_t m;
    if (n >= int64_t(sc))
    {
        int64_t xx = v;
        if (!num_up(xx, uint(n - int64_t(sc))))
            return SOP_RECHECK_CODE;
        r = xx;
        rsc = uint(n);
        m = uint64_t(xx < 0 ? -xx : xx);
    }
    else
    {
        uint d = uint(int64_t(sc) - n);
        uint64_t q = num_down(uint64_t(v < 0 ? -v : v), d, k0 != SOP_NUMRND_TRUNC);
        if (n < 0)
        {
            int64_t xx = int64_t(q);
            if (!num_up(xx, uint(-n)))
                return SOP_RECHECK_CODE;
            q = uint64_t(xx);
            rsc = 0;
        }
        else
            rsc = uint(n);
        r = (v < 0) ? -int64_t(q) : int64_t(q);
        m = q;
    }
    if (k0 == SOP_NUMRND_TYPMOD && prec <= 18 && m >= num_p10u(uint(prec)))
        return SOP_RECHECK_CODE;                    // "numeric field overflow": the backend raises it
    return 0;
}

// ceil, floor, sign (scale 0)
void num_unary(uint k0, Reg x, out int64_t r)
{
    if (k0 == SOP_NUMOP_SIGN)
    {
        r = (x.v > 0) ? 1 : ((x.v < 0) ? -1 : 0);
        return;
    }
    uint64_t m = uint64_t(x.v < 0 ? -x.v : x.v);
    uint64_t q = num_down(m, x.sc, false);
    bool inexact = (x.sc >= 20u) ? (m != 0ul) : (m != q * num_p10u(x.sc));
    if (x.v >= 0)
        r = int64_t(q) + ((k0 == SOP_NUMOP_CEIL && inexact) ? 1 : 0);
    else
        r = -int64_t(q) - ((k0 == SOP_NUMOP_FLOOR && inexact) ? 1 : 0);
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
        return 0;
    case SOP_NUMCMP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        regs[dst].isnull = 0;
        regs[dst].v = b2l(cmp_result(num_cmp(x.v, x.sc, y.v, y.sc), k0));
        return 0;
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
            return 0;
        }
        if (k0 >= SOP_NUMOP_CEIL)
        {
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
            {
                int64_t ur;
                num_unary(k0, x, ur);
                regs[dst].v = ur;
                regs[dst].sc = 0;
            }
            return 0;
        }
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
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
        return 0;
    }
    case SOP_NUMDIV:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        int64_t qr;
        uint qsc;
        uint rc = num_divide(k0, x, y, qr, qsc);
        if (rc != 0u)
            return rc;
        regs[dst].v = qr;
        regs[dst].sc = qsc;
        regs[dst].isnull = 0;
        return 0;
    }
    case SOP_NUMRND:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        int64_t qr;
        uint qsc;
        uint rc = num_round(k0, x, y.v, qr, qsc);
        if (rc != 0u)
            return rc;
        regs[dst].v = qr;
        regs[dst].sc = qsc;
        regs[dst].isnull = 0;
        return 0;
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
        return 0;
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
        return 0;
    }
    default:
        return op | 0x80000000u;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// A8: jsonb. A jsonb register is { byte offset | length << 32 | source << 60 } of a JsonbContainer (Reg.sc =
// JB_CONT) or of a scalar inside one (Reg.sc = the JEntry type: string payload, numeric varlena, false, true,
// null). The binary form is that of utils/jsonb.h: a header word (count | flags), the JEntry words (keys, then values,
// of an object), the data. An entry has the type of the child and its length, or at every 32nd entry the end
// offset. Strings are stored as is, numerics and containers are aligned to 4 bytes from the start of the data.
// Anything the kernel cannot read exactly (malformed data, a numeric that does not fit) is rechecked by the backend.
// scan_caps.c and cpu_vm.c have the same code in C.
// ---------------------------------------------------------------------------------------------
static const uint JB_STR = 0u;
static const uint JB_NUM = 1u;
static const uint JB_FALSE = 2u;
static const uint JB_TRUE = 3u;
static const uint JB_NULL = 4u;
static const uint JB_CONT = 5u;

int64_t jb_mkreg(uint off, uint len, uint src)
{
    return int64_t(uint64_t(off) | (uint64_t(len) << 32) | (uint64_t(src) << 60));
}

// Open the container at cb of clen bytes. false if malformed. isobj: object; isscalar: a raw scalar.
bool jb_open(uint s, uint cb, uint clen, out uint count, out uint nent, out bool isobj, out bool isarr, out bool isscalar)
{
    count = 0;
    nent = 0;
    isobj = false;
    isarr = false;
    isscalar = false;
    if (clen < 4u)
        return false;
    uint h = ldw_s(s, cb);
    count = h & 0x0fffffffu;
    isobj = (h & 0x20000000u) != 0u;
    isarr = (h & 0x40000000u) != 0u;
    isscalar = (h & 0x10000000u) != 0u;
    if (isobj == isarr || count > clen / 4u)
        return false;
    nent = isobj ? 2u * count : count;
    return 4u + 4u * nent <= clen;
}

// Child entry i (a key and the value of an object are entries i and count + i): its JEntry type and where its
// bytes are. false if the entry runs outside the container.
bool jb_child(uint s, uint cb, uint clen, uint nent, uint i, out uint kind, out uint voff, out uint vlen)
{
    kind = 0;
    voff = 0;
    vlen = 0;
    uint e = ldw_s(s, cb + 4u + 4u * i);
    uint st = 0;
    for (uint j = i; j > 0u; j--)
    {
        uint p = ldw_s(s, cb + 4u + 4u * (j - 1u));
        st += p & 0x0fffffffu;
        if ((p & 0x80000000u) != 0u)
            break;
    }
    uint f = e & 0x0fffffffu;
    uint len = f;
    if ((e & 0x80000000u) != 0u)
    {
        if (f < st)
            return false;
        len = f - st;
    }
    kind = (e >> 28) & 7u;
    if (kind > JB_CONT)
        return false;
    uint pad = 0;
    if (kind == JB_NUM || kind == JB_CONT)
        pad = ((st + 3u) & ~3u) - st;
    uint dbase = 4u + 4u * nent;
    if (len < pad || dbase + st + len > clen)
        return false;
    if (kind >= JB_FALSE && kind <= JB_NULL)
        return true;
    voff = cb + dbase + st + pad;
    vlen = len - pad;
    return true;
}

// Binary search of an object's keys (ordered by length, then bytes). 1 found (idx), 0 absent, 2 malformed.
uint jb_find_key(uint s, uint cb, uint clen, uint count, uint nent, uint ks, uint ko, uint kl, out uint idx)
{
    idx = 0;
    uint lo = 0;
    uint hi = count;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        uint kind, voff, vlen;
        if (!jb_child(s, cb, clen, nent, mid, kind, voff, vlen) || kind != JB_STR)
            return 2u;
        int c;
        if (vlen == kl)
            c = str_memcmp(s, voff, ks, ko, kl);
        else
            c = (vlen > kl) ? 1 : -1;
        if (c < 0)
            lo = mid + 1u;
        else if (c > 0)
            hi = mid;
        else
        {
            idx = mid;
            return 1u;
        }
    }
    return 0u;
}

// jsonb_exists on an object or an array. 1 / 0, or 2 malformed.
uint jb_exists(uint s, uint cb, uint clen, uint count, uint nent, bool isobj, uint ks, uint ko, uint kl)
{
    if (isobj)
    {
        uint idx;
        return jb_find_key(s, cb, clen, count, nent, ks, ko, kl, idx);
    }
    for (uint i = 0; i < count; i++)
    {
        uint kind, voff, vlen;
        if (!jb_child(s, cb, clen, nent, i, kind, voff, vlen))
            return 2u;
        if (kind == JB_STR && vlen == kl && str_memcmp(s, voff, ks, ko, kl) == 0)
            return 1u;
    }
    return 0u;
}

// The numeric varlena at [off, off + len) of source s as { v, sc }: 1, 0 = does not fit, 2 = malformed.
uint jb_numeric(uint s, uint off, uint len, out int64_t v, out uint sc)
{
    v = 0;
    sc = 0;
    if (len < 1u)
        return 2u;
    uint b0 = ldb_s(s, off);
    uint hl = 4u;
    uint total;
    if ((b0 & 1u) != 0u)
    {
        if (b0 == 1u)
            return 2u;
        hl = 1u;
        total = (b0 >> 1) & 0x7fu;
    }
    else
    {
        if (len < 4u || (b0 & 3u) == 2u)
            return 2u;
        total = (ldw_s(s, off) >> 2) & 0x3fffffffu;
    }
    if (total < hl || total > len)
        return 2u;
    // num_decode() reads through the page source g_src: the jsonb always lives there
    uint saved = g_src;
    g_src = s;
    bool ok = num_decode(off + hl, total - hl, v, sc);
    g_src = saved;
    return ok ? 1u : 0u;
}

// The child idx of an opened container as a register. false if malformed.
bool jb_pick(uint s, uint cb, uint clen, uint nent, uint idx, inout Reg r)
{
    uint kind, voff, vlen;
    if (!jb_child(s, cb, clen, nent, idx, kind, voff, vlen))
        return false;
    r.v = (kind >= JB_FALSE && kind <= JB_NULL) ? 0 : jb_mkreg(voff, vlen, s);
    r.sc = kind;
    r.isnull = 0;
    return true;
}

// Follow the keys of a pattern term (words at wp of the program) from the register x; 1 found, 0 absent, 2 malformed.
uint jb_walk_path(Reg x, uint nsteps, uint wp, uint patb, out uint kind, out uint voff, out uint vlen)
{
    uint s = str_src(x);
    kind = x.sc;
    voff = str_off(x);
    vlen = str_len(x);
    for (uint d = 0; d < nsteps; d++)
    {
        if (kind != JB_CONT)
            return 0u;
        uint count, nent;
        bool isobj, isarr, isscalar;
        if (!jb_open(s, voff, vlen, count, nent, isobj, isarr, isscalar))
            return 2u;
        if (!isobj)
            return 0u;
        uint idx;
        uint rc = jb_find_key(s, voff, vlen, count, nent, 3u, patb + prog[wp + 2u * d], prog[wp + 2u * d + 1u], idx);
        if (rc != 1u)
            return rc;
        uint k2, o2, l2;
        if (!jb_child(s, voff, vlen, nent, count + idx, k2, o2, l2))
            return 2u;
        kind = k2;
        voff = o2;
        vlen = l2;
    }
    return 1u;
}

// Is the column's scalar (kind, off, len) the pattern's scalar? 1 / 0, or 2: cannot tell. sw: word index of the scalar.
uint jb_scalar_equal(uint s, uint kind, uint off, uint len, uint tkind, uint sw, uint patb)
{
    if (kind != tkind)
        return 0u;
    if (kind == JB_STR)
    {
        if (len != prog[sw + 1u])
            return 0u;
        return str_memcmp(s, off, 3u, patb + prog[sw], len) == 0 ? 1u : 0u;
    }
    if (kind == JB_NUM)
    {
        int64_t av;
        uint asc;
        if (jb_numeric(s, off, len, av, asc) != 1u)
            return 2u;
        int64_t pv = int64_t(uint64_t(prog[sw]) | (uint64_t(prog[sw + 1u]) << 32));
        return num_cmp(av, asc, pv, prog[sw + 2u]) == 0 ? 1u : 0u;
    }
    return 1u;
}

// x @> the pattern at byte offset pat of the constant pool. 1 / 0, or 2: cannot tell.
uint jb_contains(Reg x, uint pat)
{
    uint s = str_src(x);
    uint patw = prog[20] + (pat >> 2);          // word index of the pattern
    uint patb = prog[20] * 4u + pat;            // byte offset of the pattern in the program buffer
    uint nterms = prog[patw];
    uint tw = patw + 1u;
    for (uint t = 0; t < nterms; t++)
    {
        uint w0 = prog[tw];
        uint type = w0 & 0xffu;
        uint nsteps = (w0 >> 8) & 0xffu;
        uint tkind = (w0 >> 16) & 0xffu;
        uint wp = tw + 1u;
        uint sw = wp + 2u * nsteps;
        uint nsw = 0;
        if (type == 1u || type == 4u)
            nsw = (tkind == JB_STR) ? 2u : ((tkind == JB_NUM) ? 3u : 0u);
        tw = sw + nsw;
        uint kind, voff, vlen;
        uint rc = jb_walk_path(x, nsteps, wp, patb, kind, voff, vlen);
        if (rc == 2u)
            return 2u;
        if (rc == 0u)
            return 0u;
        if (type == 1u)                         // EQ
        {
            rc = jb_scalar_equal(s, kind, voff, vlen, tkind, sw, patb);
            if (rc != 1u)
                return rc;
            continue;
        }
        if (kind != JB_CONT)
            return 0u;
        uint count, nent;
        bool isobj, isarr, isscalar;
        if (!jb_open(s, voff, vlen, count, nent, isobj, isarr, isscalar))
            return 2u;
        if (type == 2u)                         // ISOBJ
        {
            if (!isobj)
                return 0u;
            continue;
        }
        if (!isarr || isscalar)                 // ISARR, ELEM
            return 0u;
        if (type == 4u)
        {
            bool found = false;
            for (uint i = 0; i < count && !found; i++)
            {
                uint ek, eo, el;
                if (!jb_child(s, voff, vlen, nent, i, ek, eo, el))
                    return 2u;
                uint r2 = jb_scalar_equal(s, ek, eo, el, tkind, sw, patb);
                if (r2 == 2u)
                    return 2u;
                found = (r2 == 1u);
            }
            if (!found)
                return 0u;
        }
    }
    return 1u;
}

// The jsonb instructions. Returns 0, or SOP_RECHECK_CODE.
uint vm_json(uint op, uint dst, uint ra, uint rb, uint k0, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    uint xs = str_src(x);
    uint xo = str_off(x);
    uint xl = str_len(x);
    switch (op)
    {
    case SOP_JSONGET:
    {
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        Reg r;
        r.v = 0;
        r.isnull = 1;
        r.sc = 0;
        if (x.sc == JB_CONT)
        {
            uint count, nent;
            bool isobj, isarr, isscalar;
            if (!jb_open(xs, xo, xl, count, nent, isobj, isarr, isscalar))
                return SOP_RECHECK_CODE;
            if (k0 == 0u && isobj)
            {
                uint idx;
                uint rc = jb_find_key(xs, xo, xl, count, nent, str_src(y), str_off(y), str_len(y), idx);
                if (rc == 2u)
                    return SOP_RECHECK_CODE;
                if (rc == 1u && !jb_pick(xs, xo, xl, nent, count + idx, r))
                    return SOP_RECHECK_CODE;
            }
            else if (k0 == 1u && isarr)
            {
                int64_t i = y.v;
                if (i < 0)
                {
                    if (-i > int64_t(count))
                        i = -1;
                    else
                        i += int64_t(count);
                }
                if (i >= 0 && i < int64_t(count))
                {
                    if (!jb_pick(xs, xo, xl, nent, uint(i), r))
                        return SOP_RECHECK_CODE;
                }
            }
        }
        else if (k0 == 1u && (y.v == 0 || y.v == -1))
            r = x;                              // a scalar behaves as an array of one element
        regs[dst] = r;
        break;
    }
    case SOP_JSONTXT:
    {
        if (x.isnull != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        regs[dst].isnull = 0;
        regs[dst].sc = 0;
        if (x.sc == JB_STR)
            regs[dst].v = int64_t(uint64_t(x.v) & ~(uint64_t(0x40000000u) << 32));
        else if (x.sc == JB_TRUE)
            regs[dst].v = jb_mkreg(prog[20] * 4u + k0, 4u, 3u);
        else if (x.sc == JB_FALSE)
            regs[dst].v = jb_mkreg(prog[20] * 4u + k0 + 4u, 5u, 3u);
        else if (x.sc == JB_NULL)
            regs[dst].isnull = 1;
        else
            regs[dst].v = int64_t(uint64_t(x.v) | (uint64_t(0x40000000u) << 32));     // deferred: a numeric or a container
        break;
    }
    case SOP_JSONEXISTS:
    {
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        uint res = 0;
        if (x.sc == JB_CONT)
        {
            uint count, nent;
            bool isobj, isarr, isscalar;
            if (!jb_open(xs, xo, xl, count, nent, isobj, isarr, isscalar))
                return SOP_RECHECK_CODE;
            res = jb_exists(xs, xo, xl, count, nent, isobj, str_src(y), str_off(y), str_len(y));
            if (res == 2u)
                return SOP_RECHECK_CODE;
        }
        else if (x.sc == JB_STR && xl == str_len(y))
            res = (str_memcmp(xs, xo, str_src(y), str_off(y), xl) == 0) ? 1u : 0u;
        regs[dst].isnull = 0;
        regs[dst].v = int64_t(res);
        break;
    }
    case SOP_JSONTYPE:
    {
        if (x.isnull != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        uint t = 0;
        if (x.sc == JB_STR)
            t = 2u;
        else if (x.sc == JB_NUM)
            t = 3u;
        else if (x.sc == JB_FALSE || x.sc == JB_TRUE)
            t = 4u;
        else if (x.sc == JB_NULL)
            t = 5u;
        else
        {
            uint count, nent;
            bool isobj, isarr, isscalar;
            if (!jb_open(xs, xo, xl, count, nent, isobj, isarr, isscalar))
                return SOP_RECHECK_CODE;
            if (isobj)
                t = 0u;
            else if (!isscalar)
                t = 1u;
            else
            {
                uint kind, voff, vlen;
                if (count != 1u || !jb_child(xs, xo, xl, nent, 0u, kind, voff, vlen) || kind == JB_CONT)
                    return SOP_RECHECK_CODE;
                t = (kind == JB_STR) ? 2u : ((kind == JB_NUM) ? 3u : ((kind == JB_NULL) ? 5u : 4u));
            }
        }
        uint toff = 0;
        uint tlen = 6;
        if (t == 1u) { toff = 6u; tlen = 5u; }
        else if (t == 2u) { toff = 11u; tlen = 6u; }
        else if (t == 3u) { toff = 17u; tlen = 6u; }
        else if (t == 4u) { toff = 23u; tlen = 7u; }
        else if (t == 5u) { toff = 30u; tlen = 4u; }
        regs[dst].isnull = 0;
        regs[dst].sc = 0;
        regs[dst].v = jb_mkreg(prog[20] * 4u + k0 + toff, tlen, 3u);
        break;
    }
    case SOP_JSONNUM:
    case SOP_JSONINT:
    {
        if (x.isnull != 0 || x.sc == JB_NULL)   // ->> of a JSON null is SQL NULL
        {
            regs[dst].isnull = 1;
            break;
        }
        int64_t v;
        uint sc;
        if (x.sc != JB_NUM || jb_numeric(xs, xo, xl, v, sc) != 1u)
            return SOP_RECHECK_CODE;
        if (op == SOP_JSONINT && (sc != 0u || out_of_range(v, k0)))
            return SOP_RECHECK_CODE;            // not an integer text, or out of range: PostgreSQL's own error
        regs[dst].isnull = 0;
        regs[dst].v = v;
        regs[dst].sc = (op == SOP_JSONNUM) ? sc : 0u;
        break;
    }
    case SOP_JSONCONT:
    {
        if (x.isnull != 0)
        {
            regs[dst].isnull = 1;
            break;
        }
        uint res = 0;
        if (x.sc == JB_CONT)
        {
            res = jb_contains(x, k0);
            if (res == 2u)
                return SOP_RECHECK_CODE;
        }
        regs[dst].isnull = 0;
        regs[dst].v = int64_t(res);
        break;
    }
    default:
        return op | 0x80000000u;
    }
    return 0;
}

// Instructions of A6 (not in the original VM). Returns 0 or the error code.
// The float8 instructions (A10): a register holds the 64 bits of the double, every operation is f64_soft.inc's.
// Returns 0, or k1 | reason << 16 of the error PostgreSQL raises.
uint vm_f8(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    Reg y = regs[rb];

    switch (op)
    {
    case SOP_CMPD:
    {
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        uint64_t kx = f64_key(uint64_t(x.v));
        uint64_t ky = f64_key(uint64_t(y.v));
        regs[dst].isnull = 0;
        regs[dst].v = b2l(cmp_result(kx < ky ? -1 : (kx > ky ? 1 : 0), k0));
        return 0;
    }
    case SOP_DARITH:
    {
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        uint64_t a = uint64_t(x.v);
        uint64_t b = uint64_t(y.v);
        uint64_t r = f64_arith(k0, a, b);
        uint e = f64_arith_err(k0, a, b, r);
        if (e != 0u)
            return k1 | (e << 16);
        regs[dst].v = int64_t(r);
        regs[dst].isnull = 0;
        return 0;
    }
    case SOP_DUNARY:
    {
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
            regs[dst].v = int64_t((k0 == 0u) ? (uint64_t(x.v) ^ F64_SIGN) : (uint64_t(x.v) & F64_MAG));
        return 0;
    }
    case SOP_I2D:
    {
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
            regs[dst].v = int64_t(f64_from_i64(x.v));
        return 0;
    }
    case SOP_D2I:
    {
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            uint64_t a = uint64_t(x.v);
            if (!f64_fits_i64(a))
                return k1 | (SOP_ERR_INTRANGE << 16);
            int64_t r = f64_to_i64_rint(a);
            if (out_of_range(r, k0))
                return k1 | (SOP_ERR_INTRANGE << 16);
            regs[dst].v = r;
        }
        return 0;
    }
    case SOP_F2D:
    {
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
            regs[dst].v = int64_t(f64_from_f32(uint(x.v)));
        return 0;
    }
    default:    // SOP_D2F
    {
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0)
        {
            uint64_t a = uint64_t(x.v);
            uint r = f64_to_f32(a, 0u);
            uint e = f64_to_f32_err(a, r);
            if (e != 0u)
                return k1 | (e << 16);
            regs[dst].v = int64_t(r);
        }
        return 0;
    }
    }
    return 0;
}

// float8 aggregates without GROUP BY (A10). Classes 6 (sum, avg), 7 (min) and 8 (max); a partial is (n, a): a the bits of the
// double, n the number of values (sum) or 1 (min, max), n == 0 when there is no value yet. The order of additions is fixed
// (per thread, then the tree of the workgroup, then the service), so the result is deterministic, but not the order of a
// serial scan: the sum can differ from PostgreSQL's in the last bits. Returns 0, or SOP_ERR_FLTOVER << 16 when a sum of
// finite values overflows (float8pl raises an error for it).
uint fagg_acc(uint cls, uint64_t v, inout uint n, inout int64_t a)
{
    if (cls == 6u)
    {
        if (n == 0u)
            a = int64_t(v);
        else
        {
            uint64_t s = f64_add(uint64_t(a), v);
            if (f64_arith_err(0u, uint64_t(a), v, s) != 0u)
                return SOP_ERR_FLTOVER << 16;
            a = int64_t(s);
        }
        n += 1u;
        return 0u;
    }
    uint64_t kv = f64_key(v);
    uint64_t ka = f64_key(uint64_t(a));
    if (n == 0u || (cls == 7u ? (kv < ka) : (kv > ka)))
        a = int64_t(v);
    n = 1u;
    return 0u;
}

// add the partial (n2, a2) into (n, a)
uint fagg_merge(uint cls, uint n2, int64_t a2, inout uint n, inout int64_t a)
{
    if (n2 == 0u)
        return 0u;
    if (n == 0u)
    {
        n = n2;
        a = a2;
        return 0u;
    }
    if (cls == 6u)
    {
        uint64_t s = f64_add(uint64_t(a), uint64_t(a2));
        if (f64_arith_err(0u, uint64_t(a), uint64_t(a2), s) != 0u)
            return SOP_ERR_FLTOVER << 16;
        a = int64_t(s);
        n += n2;
        return 0u;
    }
    uint64_t kv = f64_key(uint64_t(a2));
    uint64_t ka = f64_key(uint64_t(a));
    if (cls == 7u ? (kv < ka) : (kv > ka))
        a = a2;
    n = 1u;
    return 0u;
}

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
        return 0;
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
        return 0;
    }
    case SOP_DATEOP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
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
        return 0;
    }

    case SOP_CONSTSTR:
        regs[dst].v = int64_t(uint64_t(prog[20] * 4u + k0) | (uint64_t(k1) << 32) | (uint64_t(3u) << 60));
        regs[dst].isnull = 0;
        return 0;
    case SOP_STRCMP:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        if (((uint64_t(x.v) | uint64_t(y.v)) >> 62) != 0ul)
            return SOP_RECHECK_CODE;            // a deferred text of ->> (a numeric or a container)
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
        return 0;
    }
    case SOP_STRLIKE:
    {
        Reg x = regs[ra];
        Reg y = regs[rb];
        if ((x.isnull | y.isnull) != 0)
        {
            regs[dst].isnull = 1;
            return 0;
        }
        if (((uint64_t(x.v) | uint64_t(y.v)) >> 62) != 0ul)
            return SOP_RECHECK_CODE;
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
        return 0;
    }
    case SOP_STRLEN:
    {
        Reg x = regs[ra];
        regs[dst].isnull = x.isnull;
        if (x.isnull == 0 && (uint64_t(x.v) >> 62) != 0ul)
            return SOP_RECHECK_CODE;
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
        return 0;
    }
    default:
        if (op >= SOP_NUMCONST && op <= SOP_NUMRND)
            return vm_num(op, dst, ra, rb, k0, k1, regs);
        if (op >= SOP_JSONGET && op <= SOP_JSONCONT)
            return vm_json(op, dst, ra, rb, k0, regs);
        if (op >= SOP_CMPD && op <= SOP_D2F)
            return vm_f8(op, dst, ra, rb, k0, k1, regs);
        return op | 0x80000000u;    // malformed program
    }
    return 0;
}

// Numeric aggregates and GROUP BY keys: the plan fixed the scale of the value (word 3 of the aggregate entry, the
// word after the key registers: scale + 1, 0 = not numeric), so that the sum of unscaled int64 values and the key
// equality are exact. A qualifying row whose value has another scale is evaluated by the backend instead.
bool agg_scales_ok(uint aggoff, uint naggs, uint keyoff, uint nkeys, inout Reg regs[VM_MAX_REGS])
{
    for (uint k = 0; k < naggs; k++)
    {
        uint sc1 = prog[aggoff + 4 * k + 3];
        if (sc1 == 0u)
            continue;
        uint ra = prog[aggoff + 4 * k + 1];
        uint rf = prog[aggoff + 4 * k + 2];
        if (rf != 0xffffffffu && (regs[rf].isnull != 0 || regs[rf].v == 0))
            continue;                               // this aggregate skips the row
        if (regs[ra].isnull == 0 && regs[ra].sc != sc1 - 1u)
            return false;
    }
    for (uint k = 0; k < nkeys; k++)
    {
        uint sc1 = prog[keyoff + nkeys + k];
        if (sc1 == 0u)
            continue;
        uint r = prog[keyoff + k];
        if (regs[r].isnull == 0 && regs[r].sc != sc1 - 1u)
            return false;
    }
    return true;
}

// NOTE for vm_num(), vm_ext() and vm_step(): a case ends with "return 0;", not "break;". With constant
// operands (the specialized kernels) Slang folds the switch, and a "break" in the middle of a case that sits
// in a nested "if" came out of its SPIR-V structurizer as a fall-through (the code after the "if" ran for
// the NULL branch too: `n1 = 12.34 or n1 = 0` counted the rows with a NULL n1). A "break" that leaves an
// inner loop or switch (ARITH's switch on the operator, the loop of LIKE '%x%') is not affected and stays.
// Executes one instruction { op, dst, ra, rb, k0, k1 }; AND / OR set ip to their jump target. Returns the
// FuncOpCode of a failing function, 0 on success. vm_run() below interprets a program with it; the
// specialized kernels (specialize.c) call it with constant operands, so the generated code runs
// exactly the semantics of the VM. (The loop is only there to keep the body's nesting unchanged.)
uint vm_step(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout uint ip,
             inout Reg regs[VM_MAX_REGS])
{
    [loop]
    while (true)
    {
        switch (op)
        {
        case SOP_CONST:
            regs[dst].v = int64_t(uint64_t(k0) | (uint64_t(k1) << 32));
            regs[dst].isnull = ra;
            return 0;

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
            return 0;
        case SOP_OR:
            if (regs[ra].isnull != 0)
                regs[dst].isnull = 1;
            else if (regs[ra].v != 0)
            {
                regs[dst].v = 1;
                regs[dst].isnull = 0;
                ip = rb;
            }
            return 0;
        case SOP_NOT:
            regs[dst].isnull = regs[ra].isnull;
            regs[dst].v = b2l(regs[ra].v == 0);
            return 0;
        case SOP_ISNULL:
            regs[dst].v = b2l(regs[ra].isnull != 0);
            regs[dst].isnull = 0;
            return 0;
        case SOP_ISNOTNULL:
            regs[dst].v = b2l(regs[ra].isnull == 0);
            regs[dst].isnull = 0;
            return 0;
        case SOP_MOV:
            regs[dst] = regs[ra];
            return 0;

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
            return 0;
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
            return 0;
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
            return 0;
        }

        case SOP_ARITH:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
            {
                regs[dst].isnull = 1;
                return 0;
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
            return 0;
        }

        case SOP_FARITH:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            if ((x.isnull | y.isnull) != 0)
            {
                regs[dst].isnull = 1;
                return 0;
            }
            // Metal flushes denormal floats to zero (operands and results), PostgreSQL does not: a row with a denormal
            // operand, or whose result may be denormal or underflow, is evaluated by the backend (exact, including
            // PostgreSQL's underflow error)
            uint  ua = uint(x.v), ub = uint(y.v);
            if ((((ua & 0x7f800000u) == 0u) && ((ua & 0x7fffffu) != 0u)) ||
                (((ub & 0x7f800000u) == 0u) && ((ub & 0x7fffffu) != 0u)))
                return SOP_RECHECK_CODE;
            float a = asfloat(ua);
            float b = asfloat(ub);
            float r = 0;
            bool  bad = false;
            uint  why = SOP_ERR_FLTOVER;
            bool  inf_in = isinf(a) || isinf(b);
            bool  tiny = false;

            switch (k0)
            {
            case SOP_ARITH_PL:
                r = a + b; bad = isinf(r) && !inf_in;
                tiny = (abs(r) < 1.17549435e-38) && !(r == 0 && a == -b);
                break;
            case SOP_ARITH_MI:
                r = a - b; bad = isinf(r) && !inf_in;
                tiny = (abs(r) < 1.17549435e-38) && !(r == 0 && a == b);
                break;
            case SOP_ARITH_MUL:
                r = a * b;
                bad = (isinf(r) && !inf_in);
                tiny = (abs(r) < 1.17549435e-38) && a != 0 && b != 0;
                break;
            default:    // SOP_ARITH_DIV
                if (b == 0 && !isnan(a))        // float4div: NaN / 0 is NaN
                {
                    bad = true;
                    why = SOP_ERR_ZERODIV;
                }
                else
                {
                    r = a / b;
                    bad = (isinf(r) && !inf_in);
                    tiny = (abs(r) < 1.17549435e-38) && a != 0 && !isinf(b);
                }
                break;
            }
            if (!bad && tiny && !isnan(r))
                return SOP_RECHECK_CODE;
            if (bad)
                return k1 | (why << 16);
            regs[dst].v = int64_t(asuint(r));
            regs[dst].isnull = 0;
            return 0;
        }

        case SOP_HASHINIT:
            regs[dst].v = int64_t(0xffffffffu);
            regs[dst].isnull = 0;
            return 0;
        case SOP_HASHMERGE:
            if (regs[ra].isnull == 0)
            {
                uint64_t uv = uint64_t(regs[ra].v);
                uint h = hash_any_value(uint(uv & 0xffffffffu), uint(uv >> 32), k0);
                uint acc = uint(regs[dst].v);
                regs[dst].v = int64_t(((acc >> 3) | (acc << 29)) ^ h);
            }
            return 0;
        case SOP_HASHFIN:
            regs[dst].v = int64_t(uint(regs[dst].v) ^ 0xffffffffu);
            return 0;

        case SOP_I2F:
        {
            Reg x = regs[ra];
            regs[dst].isnull = x.isnull;
            if (x.isnull == 0)
                regs[dst].v = int64_t(asuint(float(x.v)));
            return 0;
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
                // round to nearest, ties to even (PostgreSQL's rint()). Not round(): Metal's round() is half away
                // from zero (2.5 -> 3), which the SPIR-V translation keeps.
                float fl = floor(f);
                float fd = f - fl;
                int64_t r = int64_t(fl);
                if (fd > 0.5 || (fd == 0.5 && (r & 1) != 0))
                    r += 1;
                if (out_of_range(r, k0))
                    return k1 | (SOP_ERR_INTRANGE << 16);
                regs[dst].v = r;
            }
            return 0;
        }

        default:
        {
            uint e = vm_ext(op, dst, ra, rb, k0, k1, regs);
            if (e != 0)
                return e;
            return 0;
        }
        }
        return 0;
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
        uint e = vm_step(op, dst, ra, rb, k0, k1, ip, regs);
        if (e != 0)
            return e;
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

// Deform column c (descriptor desc) of the heap-format tuple at byte offset tbase of the current source
// (g_src) into its register; off is the running byte offset in the tuple. Returns false after
// report_error(). deform_tuple() calls it for every column of a table; the specialized kernels call it
// with constant c and desc (the body keeps the nesting it had inside the column loop).
bool deform_col(uint c, uint desc, uint tbase, uint natts, bool hasnull, uint page, uint lp,
                inout uint off, inout Reg regs[VM_MAX_REGS])
{
    {
        uint alen  = desc & 0xffu;
        uint align = (desc >> 8) & 0xfu;
        uint vt    = (desc >> 12) & 0xfu;
        uint slot  = (desc >> 16) & 0xffu;
        bool isnull = (c >= natts) ||
                      (hasnull && (((ld8(tbase + 23 + (c >> 3)) >> (c & 7u)) & 1u) == 0));
        if (isnull)
            return true;    // register stays NULL, no space is used

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
                else if (vt == SOP_VT_STR || vt == SOP_VT_JSON)
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
                    {
                        regs[slot].v = int64_t(uint64_t(tbase + off + hl) | (uint64_t(sz - hl) << 32) |
                                               (uint64_t(g_src) << 60));
                        regs[slot].sc = (vt == SOP_VT_JSON) ? 5u : 0u;      // jsonb: the register holds a container
                    }
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
        if (!deform_col(c, prog[coloff + c], tbase, natts, hasnull, page, lp, off, regs))
            return false;
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
