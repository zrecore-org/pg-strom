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
// Geometry groups (R15): VM_GEOM_MASK says which geometry instructions this kernel interprets (the host picks, per program, the
// smallest kernel tier that has them: scan_caps.h SCAN_FEAT_G_*). VM_NO_GEOM is the empty mask, the default is all of them.
#define VM_G_BOX   1
#define VM_G_PIP   2
#define VM_G_DW    4        // ST_DWithin of two points
#define VM_G_WALK  8        // ST_DWithin / ST_3DDWithin / ST_Distance with a line, polygon or multi-geometry operand (all three of them, points too)
#define VM_G_ZM    16
#define VM_G_FLAG  32
#define VM_G_LEN   64
#define VM_G_DW3   128      // ST_3DDWithin of two points
#define VM_G_DIST  256      // ST_Distance / ST_3DDistance of two points
#ifdef VM_NO_GEOM
#undef VM_GEOM_MASK
#define VM_GEOM_MASK 0
#endif
#ifndef VM_GEOM_MASK
#ifdef WITH_INNER
// the default (full tier) join kernel: what the join programs used before the geometry tiers (a join program has no GEOMBOX, GEOMDIST,
// GEOMZM, GEOMFLAG or GEOMLEN, and its non-point operand is a poison record)
#define VM_GEOM_MASK (VM_G_BOX | VM_G_PIP | VM_G_DW | VM_G_DW3)
#else
#define VM_GEOM_MASK 511
#endif
#endif
// PostGIS' initial minimum distance is FLT_MAX: the point-to-point decisions check it in every kernel but the default join kernel (whose
// SPIR-V stays as it was: Metal compiles each distinct kernel for minutes); the walker tier has it, and a join program that needs a
// distance beyond FLT_MAX is not a case of the full tier (such a row is the backend's)
#if !defined(WITH_INNER) || (VM_GEOM_MASK & VM_G_WALK) != 0
#define VM_G_FLTMAX_CHECK 1
#endif
#ifndef VM_NO_F8
#include "vagg.inc"             // A10: PostgreSQL's float8_accum / float8_regr_accum / ..._combine on it (variance, stddev, covar, corr, regr_*)
#endif

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
// (the Arrow scan kernel, arrow.slang, has four words even when a specialized one is generated with ARROW_SRC: its pipeline layout
// is 16 bytes, spec_mgr.c)
#if defined(WITH_INNER) || (defined(ARROW_SRC) && !defined(ARROW_SCAN))
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

#include "hash_bytes.inc"      // A6: hash_bytes() of a string register (text / bpchar join keys)

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
        {
            uint e = f64_math_err(k0, uint64_t(x.v));
            if (e != 0u)
                return k1 | (e << 16);
            regs[dst].v = int64_t(f64_math(k0, uint64_t(x.v)));
        }
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

// float8 aggregates (A10). Classes 6 (sum, avg), 7 (min), 8 (max) and 9 (sum of float4: every addition is rounded to binary32, as
// float4pl does); a partial is (n, a): a the bits of the double, n the number of values (sum) or 1 (min, max), n == 0 when there
// is no value yet. Without GROUP BY the order of additions is fixed (per thread, then the tree of the workgroup, then the
// service); with GROUP BY it is the row order inside a workgroup (gagg.slang, jgagg.slang: the owner of a slot adds the staged
// rows in thread order), then the workgroups in order at the service. Either way the result is deterministic, but not the order
// of a serial scan: the sum can differ from PostgreSQL's in the last bits. Returns 0, or SOP_ERR_FLTOVER << 16 when a sum of
// finite values overflows (float8pl / float4pl raise an error for it).
uint fagg_acc(uint cls, uint64_t v, inout uint n, inout int64_t a)
{
    if (cls == 6u || cls == 9u)
    {
        if (n == 0u)
            a = int64_t(v);
        else
        {
            uint64_t s = (cls == 9u) ? f64_addf4(uint64_t(a), v) : f64_add(uint64_t(a), v);
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
    if (cls == 6u || cls == 9u)
    {
        uint64_t s = (cls == 9u) ? f64_addf4(uint64_t(a), uint64_t(a2)) : f64_add(uint64_t(a), uint64_t(a2));
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

#ifndef VM_NO_F8
// variance / stddev / covar / corr / regr_* (A10): classes 10 (var: one aggregate entry holds n, Sx = a0, Sxx = b0 of float8_accum),
// 11 (regr head: the same for X, plus the entries k + 1 = (n, Sy, Syy) and k + 2 = (n, Sxy) of float8_regr_accum) and 12 (the
// continuation entries, which the loops skip; the head updates them). The state of an entry is (n, a, b) as in every class: the record
// format of the kernels is unchanged. A row adds with vs_acc (the row's value order is the serial order of a thread, or of the
// slot's owner), partials merge with vs_merge (float8_combine / float8_regr_combine). err: SOP_ERR_FLTOVER << 16, as float8pl.
struct VS
{
    uint    n;
    int64_t a0;
    int64_t b0;
    int64_t a1;
    int64_t b1;
    int64_t a2;
};


// exact variance / stddev of int2, int4 and int8 (class 13: the head of three entries, the other two are class 12; A10). The state of
// PostgreSQL's int4_accum / int8_accum, exactly: N, Sx as a signed 128-bit sum (b0:a0, low word first) and Sx2, the sum of the squares,
// as an unsigned 192-bit sum (a2:b1:a1). Integer addition is exact and order independent, so any order of rows and of partials gives
// PostgreSQL's numbers; the final function runs on the backend (aggfuncs.c).
VS ivs_acc(VS s, uint64_t x)
{
    VS r = s;
    uint64_t xhi = (int64_t(x) < 0) ? 0xffffffffffffffffull : 0ull;
    uint64_t lo = uint64_t(s.a0) + x;
    uint64_t hi = uint64_t(s.b0) + xhi + ((lo < x) ? 1ull : 0ull);
    // x * x as 128 bits: |x| = h * 2^32 + l
    uint64_t ax = (int64_t(x) < 0) ? (0ull - x) : x;
    uint64_t l = ax & 0xffffffffull;
    uint64_t h = ax >> 32;
    uint64_t ll = l * l;
    uint64_t lh = l * h;
    uint64_t hh = h * h;
    uint64_t plo = ll + (lh << 33);
    uint64_t phi = hh + (lh >> 31) + ((plo < ll) ? 1ull : 0ull);
    uint64_t t0 = uint64_t(s.a1) + plo;
    uint64_t c0 = (t0 < plo) ? 1ull : 0ull;
    uint64_t t1 = uint64_t(s.b1) + phi;
    uint64_t c1 = (t1 < phi) ? 1ull : 0ull;
    uint64_t t2 = t1 + c0;
    if (t2 < t1)
        c1 += 1ull;
    r.n = s.n + 1u;
    r.a0 = int64_t(lo);
    r.b0 = int64_t(hi);
    r.a1 = int64_t(t0);
    r.b1 = int64_t(t2);
    r.a2 = int64_t(uint64_t(s.a2) + c1);
    return r;
}

VS ivs_merge(VS a, VS b)
{
    VS r = a;
    uint64_t lo = uint64_t(a.a0) + uint64_t(b.a0);
    uint64_t hi = uint64_t(a.b0) + uint64_t(b.b0) + ((lo < uint64_t(b.a0)) ? 1ull : 0ull);
    uint64_t t0 = uint64_t(a.a1) + uint64_t(b.a1);
    uint64_t c0 = (t0 < uint64_t(b.a1)) ? 1ull : 0ull;
    uint64_t t1 = uint64_t(a.b1) + uint64_t(b.b1);
    uint64_t c1 = (t1 < uint64_t(b.b1)) ? 1ull : 0ull;
    uint64_t t2 = t1 + c0;
    if (t2 < t1)
        c1 += 1ull;
    r.n = a.n + b.n;
    r.a0 = int64_t(lo);
    r.b0 = int64_t(hi);
    r.a1 = int64_t(t0);
    r.b1 = int64_t(t2);
    r.a2 = int64_t(uint64_t(a.a2) + uint64_t(b.a2) + c1);
    return r;
}

[noinline] VS vs_acc(uint cls, VS s, uint64_t x, uint64_t y, inout uint err)
{
    VS r = s;
    if (cls == 13u)
        return ivs_acc(s, x);
    if (cls == 10u)
    {
        VAcc t;
        t.n = uint64_t(s.n);
        t.sx = uint64_t(s.a0);
        t.sxx = uint64_t(s.b0);
        t.err = 0u;
        t = vacc_add(t, x);
        r.n = uint(t.n);
        r.a0 = int64_t(t.sx);
        r.b0 = int64_t(t.sxx);
        if (t.err != 0u)
            err = SOP_ERR_FLTOVER << 16;
    }
    else
    {
        RAcc t;
        t.n = uint64_t(s.n);
        t.sx = uint64_t(s.a0);
        t.sxx = uint64_t(s.b0);
        t.sy = uint64_t(s.a1);
        t.syy = uint64_t(s.b1);
        t.sxy = uint64_t(s.a2);
        t.err = 0u;
        t = racc_add(t, y, x);
        r.n = uint(t.n);
        r.a0 = int64_t(t.sx);
        r.b0 = int64_t(t.sxx);
        r.a1 = int64_t(t.sy);
        r.b1 = int64_t(t.syy);
        r.a2 = int64_t(t.sxy);
        if (t.err != 0u)
            err = SOP_ERR_FLTOVER << 16;
    }
    return r;
}

[noinline] VS vs_merge(uint cls, VS a, VS b, inout uint err)
{
    VS r = a;
    if (cls == 13u)
        return ivs_merge(a, b);
    if (cls == 10u)
    {
        VAcc ta, tb;
        ta.n = uint64_t(a.n); ta.sx = uint64_t(a.a0); ta.sxx = uint64_t(a.b0); ta.err = 0u;
        tb.n = uint64_t(b.n); tb.sx = uint64_t(b.a0); tb.sxx = uint64_t(b.b0); tb.err = 0u;
        ta = vacc_merge(ta, tb);
        r.n = uint(ta.n);
        r.a0 = int64_t(ta.sx);
        r.b0 = int64_t(ta.sxx);
        if (ta.err != 0u)
            err = SOP_ERR_FLTOVER << 16;
    }
    else
    {
        RAcc ta, tb;
        ta.n = uint64_t(a.n); ta.sx = uint64_t(a.a0); ta.sxx = uint64_t(a.b0);
        ta.sy = uint64_t(a.a1); ta.syy = uint64_t(a.b1); ta.sxy = uint64_t(a.a2); ta.err = 0u;
        tb.n = uint64_t(b.n); tb.sx = uint64_t(b.a0); tb.sxx = uint64_t(b.b0);
        tb.sy = uint64_t(b.a1); tb.syy = uint64_t(b.b1); tb.sxy = uint64_t(b.a2); tb.err = 0u;
        ta = racc_merge(ta, tb);
        r.n = uint(ta.n);
        r.a0 = int64_t(ta.sx);
        r.b0 = int64_t(ta.sxx);
        r.a1 = int64_t(ta.sy);
        r.b1 = int64_t(ta.syy);
        r.a2 = int64_t(ta.sxy);
        if (ta.err != 0u)
            err = SOP_ERR_FLTOVER << 16;
    }
    return r;
}
#endif

// PostGIS (A11): a geometry register is a string register over a GSERIALIZED payload (v1 and v2, xpu_postgis.h).
// The 2D float box of a geometry as gserialized_datum_get_box2df_p() makes it: the cached box if there is one, the
// coordinates of a point rounded outward to float4 (next_float_down / next_float_up = f64_to_f32 mode 1 / 2);
// st 1: an empty point (no box: the operators are false), st 2: the VM cannot tell (geodetic, NaN, a small line or
// polygon without a cached box, a malformed value): the backend decides.
struct GBox
{
    uint st;
    uint xmin;
    uint xmax;
    uint ymin;
    uint ymax;
    // &&& (nd): the Z and M ranges when the geometry has them (float4 bits)
    uint hz;
    uint zmin;
    uint zmax;
    uint hm;
    uint mmin;
    uint mmax;
};

uint64_t ld64s(uint src, uint o)
{
    return uint64_t(ldw_s(src, o)) | (uint64_t(ldw_s(src, o + 4u)) << 32);
}

// +-infinity -> +-FLT_MAX (as float4 bits)
uint gf_clamp(uint f)
{
    if ((f & 0x7fffffffu) == 0x7f800000u)
        return (f & 0x80000000u) | 0x7f7fffffu;
    return f;
}

GBox geom_box2d(uint src, uint off, uint len, bool nd)
{
    GBox r;
    r.st = 2u;
    r.xmin = 0u;
    r.xmax = 0u;
    r.ymin = 0u;
    r.ymax = 0u;
    r.hz = 0u;
    r.zmin = 0u;
    r.zmax = 0u;
    r.hm = 0u;
    r.mmin = 0u;
    r.mmax = 0u;
    if (len < 12u)
        return r;
    uint gflags = ldb_s(src, off + 3u);
    uint hzf = nd ? (gflags & 1u) : 0u;
    uint hmf = nd ? ((gflags >> 1u) & 1u) : 0u;
    uint pos = 4u;                                      // after srid[3] and gflags
    if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
        pos += 8u;                                      // v2: the extended flags
    if ((gflags & 0x08u) != 0u)
        return r;                                       // geodetic
    if ((gflags & 0x04u) != 0u)
    {
        if (pos + 16u > len)
            return r;
        r.xmin = ldw_s(src, off + pos);
        r.xmax = ldw_s(src, off + pos + 4u);
        r.ymin = ldw_s(src, off + pos + 8u);
        r.ymax = ldw_s(src, off + pos + 12u);
        if ((r.xmin & 0x7fffffffu) > 0x7f800000u || (r.xmax & 0x7fffffffu) > 0x7f800000u ||
            (r.ymin & 0x7fffffffu) > 0x7f800000u || (r.ymax & 0x7fffffffu) > 0x7f800000u)
        {
            r.st = 2u;
            return r;
        }
        uint zp = pos + 16u;
        if (hzf != 0u)
        {
            if (zp + 8u > len)
                return r;
            r.hz = 1u;
            r.zmin = ldw_s(src, off + zp);
            r.zmax = ldw_s(src, off + zp + 4u);
            zp += 8u;
            if ((r.zmin & 0x7fffffffu) > 0x7f800000u || (r.zmax & 0x7fffffffu) > 0x7f800000u)
                return r;
        }
        if (hmf != 0u)
        {
            if (zp + 8u > len)
                return r;
            r.hm = 1u;
            r.mmin = ldw_s(src, off + zp);
            r.mmax = ldw_s(src, off + zp + 4u);
            if ((r.mmin & 0x7fffffffu) > 0x7f800000u || (r.mmax & 0x7fffffffu) > 0x7f800000u)
                return r;
        }
        r.st = 0u;
        return r;
    }
    if (pos + 8u > len || ldw_s(src, off + pos) != 1u)
        return r;                                       // only a point has no cached box we can make
    if (ldw_s(src, off + pos + 4u) == 0u)
    {
        r.st = 1u;                                      // POINT EMPTY
        return r;
    }
    if (pos + 8u + 16u + 8u * (hzf + hmf) > len)
        return r;
    uint64_t xb = uint64_t(ldw_s(src, off + pos + 8u)) | (uint64_t(ldw_s(src, off + pos + 12u)) << 32);
    uint64_t yb = uint64_t(ldw_s(src, off + pos + 16u)) | (uint64_t(ldw_s(src, off + pos + 20u)) << 32);
    if ((xb & F64_MAG) > F64_INF || (yb & F64_MAG) > F64_INF)
        return r;
    // &&& goes through PostGIS' n-D box (gidx), which treats a coordinate beyond the float4 range differently from the 2D box
    // (a Z of 1e300 overlapped a Z range of [7, 7]): such a point is the backend's. 0x47EFFFFFE0000000 is FLT_MAX as a double.
    if (nd && ((xb & F64_MAG) > 0x47EFFFFFE0000000ull || (yb & F64_MAG) > 0x47EFFFFFE0000000ull))
        return r;
    if (hzf != 0u)
    {
        uint64_t zb = ld64s(src, off + pos + 24u);
        if ((zb & F64_MAG) > F64_INF || (nd && (zb & F64_MAG) > 0x47EFFFFFE0000000ull))
            return r;
        r.hz = 1u;
        r.zmin = gf_clamp(f64_to_f32(zb, 1u));
        r.zmax = gf_clamp(f64_to_f32(zb, 2u));
    }
    if (hmf != 0u)
    {
        uint64_t mb = ld64s(src, off + pos + 24u + 8u * hzf);
        if ((mb & F64_MAG) > F64_INF || (nd && (mb & F64_MAG) > 0x47EFFFFFE0000000ull))
            return r;
        r.hm = 1u;
        r.mmin = gf_clamp(f64_to_f32(mb, 1u));
        r.mmax = gf_clamp(f64_to_f32(mb, 2u));
    }
    // next_float_down / next_float_up of PostGIS never give infinity: a value beyond the float range becomes +-FLT_MAX
    r.xmin = gf_clamp(f64_to_f32(xb, 1u));
    r.xmax = gf_clamp(f64_to_f32(xb, 2u));
    r.ymin = gf_clamp(f64_to_f32(yb, 1u));
    r.ymax = gf_clamp(f64_to_f32(yb, 2u));
    r.st = 0u;
    return r;
}

// a <= b on float4 bit patterns (no NaN here; Metal's float compare flushes denormals, so the bits are ordered)
bool gf_le(uint a, uint b) { return f4cmpbits(a, b) <= 0; }

// The PostGIS instructions. Returns 0, or SOP_RECHECK_CODE.
uint vm_geom(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    Reg y = regs[rb];

    if ((x.isnull | y.isnull) != 0u)
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    bool nd = (k0 == SOP_GBOX_OVERLAPS_ND);
    GBox a = geom_box2d(str_src(x), str_off(x), str_len(x), nd);
    GBox b = geom_box2d(str_src(y), str_off(y), str_len(y), nd);
    bool res = false;
    if (a.st == 1u || b.st == 1u)
    {
        // an empty operand decides: && and &&& are false, ~ and @ are true (PostGIS 3.7: box2df_contains of an empty box)
        res = (k0 != SOP_GBOX_OVERLAPS && !nd);
    }
    else if (a.st == 2u || b.st == 2u)
        return SOP_RECHECK_CODE;
    else
    {
        if (k0 == SOP_GBOX_OVERLAPS || nd)
        {
            res = gf_le(a.xmin, b.xmax) && gf_le(b.xmin, a.xmax) && gf_le(a.ymin, b.ymax) && gf_le(b.ymin, a.ymax);
            // &&& compares a dimension only when both operands have it (a Z against an M is not compared)
            if (nd && res && a.hz != 0u && b.hz != 0u)
                res = gf_le(a.zmin, b.zmax) && gf_le(b.zmin, a.zmax);
            if (nd && res && a.hm != 0u && b.hm != 0u)
                res = gf_le(a.mmin, b.mmax) && gf_le(b.mmin, a.mmax);
        }
        else if (k0 == SOP_GBOX_CONTAINS)
            res = gf_le(a.xmin, b.xmin) && gf_le(b.xmax, a.xmax) && gf_le(a.ymin, b.ymin) && gf_le(b.ymax, a.ymax);
        else
            res = gf_le(b.xmin, a.xmin) && gf_le(a.xmax, b.xmax) && gf_le(b.ymin, a.ymin) && gf_le(a.ymax, b.ymax);
    }
    regs[dst].isnull = 0u;
    regs[dst].v = b2l(res);
    return 0u;
}

// A point in a (multi)polygon (GEOMPIP), the same algorithm as geom_pip() of cpu_vm.c: the non-zero winding number,
// every edge's side of the point decided on the software binary64 only when its sign is beyond the error bound of the
// orientation test (Shewchuk's (3 + 16 eps) eps with a safety factor of 5 = 2^-49); the doubtful (a point on an edge or
// a vertex, too close, overflow, non-finite) is PIP_UNSURE and the backend evaluates the row.
#define PIP_OUT     0u
#define PIP_IN      1u
#define PIP_UNSURE  2u
#define PIP_ERRBOUND 0x3CE0000000000000ull

bool f64_nonfinite(uint64_t a) { return (a & F64_MAG) >= F64_INF; }

// the winding number of a ring of n points at off (16 bytes each) around (px, py)
uint pip_ring(uint src, uint off, uint n, uint64_t px, uint64_t py, uint stride)
{
    if (n < 4u)
        return PIP_UNSURE;
    int wn = 0;
    uint64_t x1 = ld64s(src, off);
    uint64_t y1 = ld64s(src, off + 8u);
    if (f64_nonfinite(x1) || f64_nonfinite(y1))
        return PIP_UNSURE;
    for (uint i = 1u; i < n; i++)
    {
        uint64_t x2 = ld64s(src, off + stride * i);
        uint64_t y2 = ld64s(src, off + stride * i + 8u);
        if (f64_nonfinite(x2) || f64_nonfinite(y2))
            return PIP_UNSURE;
        if (f64_eq(y1, y2))
        {
            // horizontal (or zero-length, ignored): only a point on it matters, and that is the boundary
            if (!f64_eq(x1, x2) && f64_eq(py, y1))
            {
                bool inside = f64_le(x1, x2) ? (f64_le(x1, px) && f64_le(px, x2)) : (f64_le(x2, px) && f64_le(px, x1));
                if (inside)
                    return PIP_UNSURE;
            }
        }
        else
        {
            uint64_t ymin = f64_lt(y1, y2) ? y1 : y2;
            uint64_t ymax = f64_lt(y1, y2) ? y2 : y1;
            if (!(f64_lt(py, ymin) || f64_gt(py, ymax)))
            {
                uint64_t l = f64_mul(f64_sub(px, x1), f64_sub(y2, y1));
                uint64_t r = f64_mul(f64_sub(x2, x1), f64_sub(py, y1));
                uint64_t det = f64_sub(r, l);           // cross(b - a, p - a): > 0 when p is left of a -> b
                uint64_t s = f64_add(l & F64_MAG, r & F64_MAG);
                uint e = uint((s >> 52) & 0x7ffu);
                // the bound holds for finite, non-subnormal terms; s = 0: the point is on the line through a vertex
                if (e < 0x080u || e > 0x7f0u)
                    return PIP_UNSURE;
                if (!f64_gt(det & F64_MAG, f64_mul(s, PIP_ERRBOUND)))
                    return PIP_UNSURE;
                if (f64_le(y1, py) && f64_lt(py, y2))
                {
                    if ((det & F64_SIGN) == 0ull)
                        wn++;
                }
                else if (f64_le(y2, py) && f64_lt(py, y1))
                {
                    if ((det & F64_SIGN) != 0ull)
                        wn--;
                }
            }
        }
        x1 = x2;
        y1 = y2;
    }
    return wn != 0 ? PIP_IN : PIP_OUT;
}

// the polygon whose type word is at pos: the result; pnext = the offset after it (0: malformed)
uint pip_polygon(uint src, uint base, uint len, uint pos, uint64_t px, uint64_t py, out uint pnext)
{
    pnext = 0u;
    if (pos + 8u > len || ldw_s(src, base + pos) != 3u)
        return PIP_UNSURE;
    uint nrings = ldw_s(src, base + pos + 4u);
    if (nrings > (len - pos - 8u) / 4u)
        return PIP_UNSURE;
    uint off = pos + 8u + 4u * nrings + ((nrings & 1u) != 0u ? 4u : 0u);
    if (off > len)
        return PIP_UNSURE;
    uint res = PIP_OUT;
    for (uint r = 0u; r < nrings; r++)
    {
        uint n = ldw_s(src, base + pos + 8u + 4u * r);
        if (n > (len - off) / 16u)
            return PIP_UNSURE;
        if (res != PIP_UNSURE && (r == 0u || res == PIP_IN))
        {
            uint rr = pip_ring(src, base + off, n, px, py, 16u);
            if (rr == PIP_UNSURE)
                res = PIP_UNSURE;
            else if (r == 0u)
                res = rr;
            else if (rr == PIP_IN)
                res = PIP_OUT;                          // in a hole
        }
        off += 16u * n;
    }
    pnext = off;
    return res;
}

struct GHead
{
    uint bad;       // 1: not decidable here
    uint gtype;
    uint srid;
    uint pos;       // the offset of the type word
    uint flags;
};

GHead geom_head(uint src, uint base, uint len)
{
    GHead h;
    h.bad = 1u;
    h.gtype = 0u;
    h.srid = 0u;
    h.pos = 0u;
    h.flags = 0u;
    if (len < 12u)
        return h;
    uint b0 = ldb_s(src, base);
    uint gflags = ldb_s(src, base + 3u);
    if ((b0 & 0xe0u) != 0u)
        return h;
    h.srid = ((b0 & 0x1fu) << 16) | (ldb_s(src, base + 1u) << 8) | ldb_s(src, base + 2u);
    uint o = 4u;
    if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
        o += 8u;
    h.flags = gflags;
    if (o + 8u > len)
        return h;
    if ((gflags & 0x04u) != 0u)
    {
        if ((gflags & 0x03u) != 0u)
            return h;                                   // a Z or M box has more floats: not handled
        o += 16u;
        if (o + 8u > len)
            return h;
    }
    h.pos = o;
    h.gtype = ldw_s(src, base + o);
    h.bad = 0u;
    return h;
}

// ST_DWithin of two points (GEOMDWITHIN), the same as geom_dwithin() of cpu_vm.c. PostGIS compares its distance (which
// differs from plain binary64 in 8% of random pairs: it fuses in places) with the tolerance, so the squares are compared
// with a relative margin of 2^-47 (the distance is within 2 ulp, the squares within 2^-50) and the boundary is rechecked.
#define GDW_LO    0x3FEFFFFFFFFFFFC0ull    // 1 - 2^-47
#define GDW_HI    0x3FF0000000000020ull    // 1 + 2^-47
#define GDW_TINY  0x07B0000000000000ull    // 2^-900: below it a square may have lost precision

// a point from its string register: 0, or SOP_RECHECK_CODE (not a plain 2D-or-Z/M POINT)
uint gdw_load(Reg r, inout uint srid, inout uint64_t px, inout uint64_t py, inout bool empty)
{
    uint s = str_src(r);
    uint b = str_off(r);
    uint l = str_len(r);
    GHead h = geom_head(s, b, l);
    if (h.bad != 0u || h.gtype != 1u || (h.flags & 0x0cu) != 0u || h.pos + 8u > l)
        return SOP_RECHECK_CODE;
    srid = h.srid;
    uint npts = ldw_s(s, b + h.pos + 4u);
    empty = (npts == 0u);
    if (!empty)
    {
        uint nd = 2u + (h.flags & 1u) + ((h.flags >> 1) & 1u);
        if (npts != 1u || h.pos + 8u + 8u * nd > l)
            return SOP_RECHECK_CODE;
        px = ld64s(s, b + h.pos + 8u);
        py = ld64s(s, b + h.pos + 16u);
    }
    return 0u;
}

uint vm_geom_dwithin(uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg a = regs[ra];
    Reg b = regs[rb];
    bool fa = (k0 & SOP_GDW_FUSED_A) != 0u;
    bool fb = (k0 & SOP_GDW_FUSED_B) != 0u;
    Reg ya = regs[(k0 >> 8) & 0xffu];
    Reg yb = regs[(k0 >> 16) & 0xffu];
    Reg t = regs[k0 >> 24];
    if ((a.isnull | b.isnull | t.isnull) != 0u || (fa && ya.isnull != 0u) || (fb && yb.isnull != 0u))
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint sa = k1 & 0xfffffu;
    uint sb = sa;
    uint64_t ax = uint64_t(a.v);
    uint64_t ay = uint64_t(ya.v);
    uint64_t bx = uint64_t(b.v);
    uint64_t by = uint64_t(yb.v);
    bool ea = false;
    bool eb = false;
    if (!fa && gdw_load(a, sa, ax, ay, ea) != 0u)
        return SOP_RECHECK_CODE;
    if (!fb && gdw_load(b, sb, bx, by, eb) != 0u)
        return SOP_RECHECK_CODE;
    if (sa != sb)
        return SOP_RECHECK_CODE;                        // PostGIS raises "mixed SRID": the backend does
    uint64_t tol = uint64_t(t.v);
    if (f64_nonfinite(tol) || ((tol >> 63) != 0ull && (tol & F64_MAG) != 0ull))
        return SOP_RECHECK_CODE;                        // negative (PostGIS raises), NaN or infinite
    bool res = false;
    if (!(ea || eb))
    {
        if (f64_nonfinite(ax) || f64_nonfinite(ay) || f64_nonfinite(bx) || f64_nonfinite(by))
            return SOP_RECHECK_CODE;
        uint64_t dx = f64_sub(ax, bx);
        uint64_t dy = f64_sub(ay, by);
        if (f64_nonfinite(dx) || f64_nonfinite(dy))
            return SOP_RECHECK_CODE;
        if (((dx | dy) & F64_MAG) == 0ull)
            res = true;                                 // the same point: distance 0 <= tolerance
        else if ((k1 & SOP_GDW_EXACT) != 0u)
        {
            // the installed PostGIS decides "exact distance <= tolerance" (calibrated by the planner): PostGIS' own double, no band
            uint64_t q = f64_sqr_add(dx, f64_mul(dy, dy));
            if (f64_nonfinite(q))
                return SOP_RECHECK_CODE;
#ifdef VM_G_FLTMAX_CHECK
            // PostGIS starts its minimum at FLT_MAX: a distance beyond it is never seen (ST_DWithin is true for a tolerance >= FLT_MAX, ST_Distance NULL)
            if (f64_lt(0x47EFFFFFE0000000ull, f64_sqrt(q)))
                return SOP_RECHECK_CODE;
#endif
            res = f64_le(f64_sqrt(q), tol);
        }
        else
        {
            uint64_t d2 = f64_add(f64_mul(dx, dx), f64_mul(dy, dy));
            uint64_t r2 = f64_mul(tol, tol);
            if (f64_nonfinite(d2) || f64_nonfinite(r2) || (d2 & F64_MAG) < GDW_TINY ||
                ((r2 & F64_MAG) != 0ull && (r2 & F64_MAG) < GDW_TINY))
                return SOP_RECHECK_CODE;
            if (f64_le(d2, f64_mul(r2, GDW_LO)))
                res = true;
            else if (f64_lt(f64_mul(r2, GDW_HI), d2))
                res = false;
            else
                return SOP_RECHECK_CODE;                // within 2^-47 of the boundary
        }
    }
    regs[dst].isnull = 0u;
    regs[dst].v = res ? 1 : 0;
    return 0u;
}

// A point with its coordinates for the 3D functions: like gdw_load() plus the Z (hz false: the point has no Z value)
uint gdw3_load(Reg r, inout uint srid, inout uint64_t px, inout uint64_t py, inout uint64_t pz, inout bool hz, inout bool empty)
{
    uint s = str_src(r);
    uint b = str_off(r);
    uint l = str_len(r);
    GHead h = geom_head(s, b, l);
    if (h.bad != 0u || h.gtype != 1u || (h.flags & 0x0cu) != 0u || h.pos + 8u > l)
        return SOP_RECHECK_CODE;
    srid = h.srid;
    uint npts = ldw_s(s, b + h.pos + 4u);
    empty = (npts == 0u);
    hz = ((h.flags & 1u) != 0u);
    if (!empty)
    {
        uint nd = 2u + (h.flags & 1u) + ((h.flags >> 1) & 1u);
        if (npts != 1u || h.pos + 8u + 8u * nd > l)
            return SOP_RECHECK_CODE;
        px = ld64s(s, b + h.pos + 8u);
        py = ld64s(s, b + h.pos + 16u);
        if (hz)
            pz = ld64s(s, b + h.pos + 24u);
    }
    return 0u;
}

// ST_Z (k0 = 0), ST_M (k0 = 1), ST_X (2) and ST_Y (3) of a point
uint vm_geom_zm(uint dst, uint ra, uint k0, inout Reg regs[VM_MAX_REGS])
{
    Reg a = regs[ra];
    if (a.isnull != 0u)
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint s = str_src(a);
    uint b = str_off(a);
    uint l = str_len(a);
    GHead h = geom_head(s, b, l);
    if (h.bad != 0u || h.gtype != 1u || (h.flags & 0x0cu) != 0u || h.pos + 8u > l)
        return SOP_RECHECK_CODE;                        // not a POINT (PostGIS raises), geodetic, malformed
    uint npts = ldw_s(s, b + h.pos + 4u);
    uint hz = h.flags & 1u;
    uint hm = (h.flags >> 1) & 1u;
    uint nd = 2u + hz + hm;
    uint idx = (k0 == 0u) ? 2u : (k0 == 1u ? 2u + hz : k0 - 2u);     // ST_X (k0 2) and ST_Y (k0 3) read the first two doubles
    if (npts == 0u || (k0 == 0u && hz == 0u) || (k0 == 1u && hm == 0u))
    {
        regs[dst].isnull = 1u;                          // an empty point, or no such dimension
        return 0u;
    }
    if (npts != 1u || h.pos + 8u + 8u * nd > l)
        return SOP_RECHECK_CODE;
    regs[dst].isnull = 0u;
    regs[dst].v = int64_t(ld64s(s, b + h.pos + 8u + 8u * idx));
    return 0u;
}

// ---- ST_Distance of a point and a line, a polygon or a multi-geometry or collection of those (2D; GEOMDIST), the same as gd_* of cpu_vm.c ----
// PostGIS (lw_dist2d_*, mode DIST_MIN): the minimum over the members of the distance to the first vertex of every line or ring and to each
// segment, a segment's being that to the foot of the perpendicular (r = ((p-A).(B-A)) / |B-A|^2 with the fused operations a compiler
// contracts it to, 0 <= r <= 1) or to the nearer end; a point in a polygon's exterior ring but not in a hole is at distance 0, one in a hole
// is at the distance to that hole's ring, one outside the exterior ring is at the distance to it. A point near an edge (the winding test
// cannot tell), non-finite values and anything unusual are the backend's.
// The type word of a geometry string register, its SRID, dimension count and the offset of the type word; 0 when it is not readable here.
uint gd_hdr(Reg r, inout uint srid, inout uint nd, inout uint o)
{
    uint s = str_src(r);
    uint b = str_off(r);
    uint l = str_len(r);
    if (l < 12u || (ldb_s(s, b) & 0xe0u) != 0u)
        return 0u;
    uint gflags = ldb_s(s, b + 3u);
    if ((gflags & 0x08u) != 0u)
        return 0u;
    nd = 2u + (gflags & 1u) + ((gflags >> 1u) & 1u);
    srid = ((ldb_s(s, b) & 0x1fu) << 16) | (ldb_s(s, b + 1u) << 8) | ldb_s(s, b + 2u);
    o = 4u;
    if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
        o += 8u;
    if ((gflags & 0x04u) != 0u)
        o += 8u * nd;
    if (o + 8u > l)
        return 0u;
    return ldw_s(s, b + o);
}

// ST_Distance of the point (px, py) and the geometry in register g (any type but a point); the result in best, found = false: nothing to
// measure. One loop over the members, with exactly one copy of each heavy step (the containment test of a polygon's rings, the segment
// projection, the distance): the Slang compiler inlines every call, so a step written once but called from several places multiplies the
// kernel (the first version of this was 22 MB of SPIR-V per kernel).
[noinline] uint gd_geom(Reg g, uint64_t px, uint64_t py, uint64_t pz, bool d3in, inout uint64_t best, inout bool found)
{
#ifdef VM_NO_GEOM3D
    const bool d3 = false;                              // (a variant without the 3D distance: the wrapper hands those to the backend)
#else
    bool d3 = d3in;
#endif
    uint s = str_src(g);
    uint b = str_off(g);
    uint end = b + str_len(g);
    uint srid = 0u;
    uint nd = 2u;
    uint o = 0u;
    uint gtype = gd_hdr(g, srid, nd, o);
    if (gtype == 0u || gtype > 7u)
        return 1u;
    uint stride = 8u * nd;
    if (d3 && (ldb_s(s, b + 3u) & 1u) == 0u)
        return 1u;                                      // the 3D distance needs a Z in the geometry
    uint p = b + o;
    uint rem0 = 1u;                                     // members still to read at the top level
    uint rem1 = 0u;                                     // members left in a multi-geometry inside a collection
    if (gtype >= 4u)
    {
        rem0 = ldw_s(s, p + 4u);
        p += 8u;
        if (rem0 > (end - p) / 8u)
            return 1u;
    }
    for (uint guard = 0u; guard < 65536u; guard++)
    {
        if (rem1 > 0u)
            rem1--;
        else
        {
            if (rem0 == 0u)
                break;
            rem0--;
            if (gtype == 7u)
            {
                if (p + 8u > end)
                    return 1u;
                uint st = ldw_s(s, p);
                if (st >= 4u && st <= 6u)
                {
                    rem1 = ldw_s(s, p + 4u);
                    p += 8u;
                    if (rem1 > (end - p) / 8u)
                        return 1u;
                    continue;
                }
            }
        }
        // the member at p: a point, a line or a polygon; (aoff, an) is the array of points to measure, if any
        if (p + 8u > end)
            return 1u;
        uint t = ldw_s(s, p);
        uint n = ldw_s(s, p + 4u);
        uint base = p + 8u;
        uint aoff = 0u;
        uint an = 0u;
        bool measure = false;
        if (t == 1u)
        {
            if (n > 1u || (n == 1u && base + stride > end))
                return 1u;
            aoff = base;
            an = n;
            measure = (n == 1u);
            p = base + n * stride;
        }
        else if (t == 2u)
        {
            if (n > (end - base) / stride)
                return 1u;
            aoff = base;
            an = n;
            measure = (n > 0u);
            p = base + n * stride;
        }
        else if (t == 3u)
        {
#ifdef VM_NO_GEOMPOLY
            return 1u;                                  // (a variant without polygons)
#endif
            if (d3)
                return 1u;                              // the 3D distance of a polygon is the backend's (PostGIS projects onto its plane)
            uint data = base + 4u * n + ((n & 1u) != 0u ? 4u : 0u);
            if (n > (end - base) / 4u || data > end)
                return 1u;
            uint q = data;
            bool decided = false;
            for (uint r = 0u; r < n; r++)
            {
                uint np = ldw_s(s, base + 4u * r);
                if (np > (end - q) / stride)
                    return 1u;
                if (!decided)
                {
                    uint rr = pip_ring(s, q, np, px, py, stride);
                    if (rr == PIP_UNSURE)
                        return 1u;
                    // outside the exterior ring: the distance to it; inside a hole: the distance to that hole's ring
                    if (r == 0u ? (rr == PIP_OUT) : (rr == PIP_IN))
                    {
                        aoff = q;
                        an = np;
                        measure = true;
                        decided = true;
                    }
                }
                q += np * stride;
            }
            if (n > 0u)
            {
                found = true;
                if (!decided)
                    best = 0ull;                        // inside the polygon (and in no hole)
            }
            p = q;
        }
        else
            return 1u;
        if (measure)
        {
            found = true;
            uint64_t ax = ld64s(s, aoff);
            uint64_t ay = ld64s(s, aoff + 8u);
            uint64_t az = d3 ? ld64s(s, aoff + 16u) : 0ull;
            for (uint i = 0u; i < an; i++)
            {
                uint64_t bx = ax;
                uint64_t by = ay;
                uint64_t bz = az;
                if (i > 0u)
                {
                    bx = ld64s(s, aoff + i * stride);
                    by = ld64s(s, aoff + i * stride + 8u);
                    if (d3)
                        bz = ld64s(s, aoff + i * stride + 16u);
                }
                if (f64_nonfinite(bx) || f64_nonfinite(by) || (d3 && f64_nonfinite(bz)))
                    return 1u;
                // the distance of this step is scl * sqrt(rad). 2D: to a vertex (the first step; a segment's end A or B when the point projects
                // before or after it): rad = h^2 + v^2, scl = 1; to the inside of a segment A -> B: PostGIS' distance2d_pt_seg,
                // |s| * sqrt(L2) with s = ((Ay - py) (Bx - Ax) - (Ax - px) (By - Ay)) / L2 and L2 = |B - A|^2 (the fused operations, found
                // by postgis/seg_probe.c: 0 differences in 89,523 inside rows; the foot-of-the-perpendicular model was wrong in the last bits).
                // 3D: PostGIS' lw_dist3d_pt_seg: r = num / den with num = fma(pz', bz', fma(px', bx', py' * by')) and den likewise (primes: relative
                // to A), A for r < 0, B for r > 1, else the foot c = fma(r, B - A, A) per coordinate, then the distance to it, rad =
                // fma(dz, dz, fma(dx, dx, dy * dy)) (postgis/seg3d_probe.c: 0 differences in 113,586 inside rows)
                uint64_t qx = bx;
                uint64_t qy = by;
                uint64_t qz = bz;
                uint64_t rad = 0ull;
                uint64_t scl = 0x3FF0000000000000ull;
                bool inside = false;
                if (i > 0u)
                {
                    qx = ax;
                    qy = ay;
                    qz = az;
                    if (!(f64_eq(ax, bx) && f64_eq(ay, by) && (!d3 || f64_eq(az, bz))))
                    {
                        uint64_t bxa = f64_sub(bx, ax);
                        uint64_t bya = f64_sub(by, ay);
                        uint64_t pxa = f64_sub(px, ax);
                        uint64_t pya = f64_sub(py, ay);
                        uint64_t bza = d3 ? f64_sub(bz, az) : 0ull;
                        uint64_t pza = d3 ? f64_sub(pz, az) : 0ull;
                        uint64_t num = f64_fma(pxa, bxa, f64_mul(pya, bya));
                        uint64_t den = f64_fma(bxa, bxa, f64_mul(bya, bya));
                        uint64_t mag = f64_add(f64_mul(pxa, bxa) & F64_MAG, f64_mul(pya, bya) & F64_MAG);
                        if (d3)
                        {
                            num = f64_fma(pza, bza, num);
                            den = f64_fma(bza, bza, den);
                        }
                        if (f64_nonfinite(bxa) || f64_nonfinite(bya) || f64_nonfinite(pxa) || f64_nonfinite(pya) ||
                            f64_nonfinite(bza) || f64_nonfinite(pza) || f64_nonfinite(num) || f64_nonfinite(den) || (den & F64_MAG) == 0ull)
                            return 1u;
                        if (d3)
                        {
                            uint64_t rr = f64_div(num, den);
                            if (f64_nonfinite(rr))
                                return 1u;
                            if (f64_gt(rr, 0x3FF0000000000000ull))
                            {
                                qx = bx;
                                qy = by;
                                qz = bz;
                            }
                            else if (!f64_lt(rr, 0ull))
                            {
                                for (uint kk = 0u; kk < 3u; kk++)
                                {
                                    uint64_t cb = (kk == 0u) ? bxa : ((kk == 1u) ? bya : bza);
                                    uint64_t ca = (kk == 0u) ? ax : ((kk == 1u) ? ay : az);
                                    uint64_t cc = f64_fma(rr, cb, ca);
                                    if (kk == 0u)
                                        qx = cc;
                                    else if (kk == 1u)
                                        qy = cc;
                                    else
                                        qz = cc;
                                }
                            }
                        }
                        else
                        {
                            // which side of A and B the point projects to is decided by r = num / den; how PostGIS' build fuses num and den is not
                            // known to the last bit, so a point whose r is within 2^-40 (of the terms' size) of 0 or 1 is the backend's
                            if (f64_le(num & F64_MAG, f64_mul(mag, 0x3D70000000000000ull)) ||
                                f64_le(f64_sub(num, den) & F64_MAG, f64_mul(f64_add(mag, den), 0x3D70000000000000ull)))
                                return 1u;
                            if (f64_gt(num, den))
                            {
                                qx = bx;
                                qy = by;
                            }
                            else if (!f64_lt(num, 0ull))
                            {
                                // a vertical or horizontal segment: PostGIS answers |px - ax| / |py - ay| exactly (postgis/seg_axis_probe.c: the generic
                                // model is off by 1-2 ulp in 35% of those rows, this one in none)
                                uint64_t sv;
                                if (f64_eq(ax, bx))
                                    sv = f64_sub(px, ax) & F64_MAG;
                                else if (f64_eq(ay, by))
                                    sv = f64_sub(py, ay) & F64_MAG;
                                else
                                {
                                    uint64_t cross = f64_fma(f64_sub(ay, py), bxa, f64_mul(f64_sub(ax, px), bya) ^ F64_SIGN);
                                    sv = f64_div(cross, den);
                                    rad = den;
                                }
                                if (f64_nonfinite(sv))
                                    return 1u;
                                scl = sv & F64_MAG;
                                if (f64_eq(ax, bx) || f64_eq(ay, by))
                                    rad = 0x3FF0000000000000ull;
                                inside = true;
                            }
                        }
                    }
                }
                if (!inside)
                {
                    uint64_t h = f64_sub(qx, px);
                    uint64_t v = f64_sub(qy, py);
                    rad = f64_sqr_add(h, f64_mul(v, v));
                    if (d3)
                    {
                        uint64_t w = f64_sub(qz, pz);
                        if (f64_nonfinite(w))
                            return 1u;
                        rad = f64_sqr_add(w, rad);
                    }
                    if (f64_nonfinite(h) || f64_nonfinite(v))
                        return 1u;
                }
                if (f64_nonfinite(rad))
                    return 1u;
                uint64_t dd = f64_mul(scl, f64_sqrt(rad));
                if (f64_nonfinite(dd))
                    return 1u;
                if (f64_lt(dd, best))
                    best = dd;
                ax = bx;
                ay = by;
                az = bz;
            }
        }
    }
    return (rem0 > 0u || rem1 > 0u) ? 1u : 0u;
}

// ST_Distance / ST_3DDistance / ST_3DMaxDistance of two points (GEOMDIST), the same as geom_dist() of cpu_vm.c. PostGIS evaluates
// sqrt(fma(dz, dz, fma(dx, dx, dy * dy))) (3D) / sqrt(fma(dx, dx, dy * dy)) (2D); f64_sqr_add is the fused step, so the value is
// PostGIS' own, bit for bit (postgis/dist_probe.c), not an approximation to be rechecked at a boundary.
uint vm_geom_dist(uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg a = regs[ra];
    Reg b = regs[rb];
    bool fa = (k0 & SOP_GDW_FUSED_A) != 0u;
    bool fb = (k0 & SOP_GDW_FUSED_B) != 0u;
    bool d3 = (k1 & SOP_GDIST_3D) != 0u;
    Reg ya = regs[(k0 >> 8) & 0xffu];
    Reg yb = regs[(k0 >> 16) & 0xffu];
    Reg za = regs[(k1 >> 20) & 0x1fu];
    Reg zb = regs[(k1 >> 25) & 0x1fu];
    if ((a.isnull | b.isnull) != 0u || (fa && (ya.isnull | (d3 ? za.isnull : 0u)) != 0u) ||
        (fb && (yb.isnull | (d3 ? zb.isnull : 0u)) != 0u))
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint sa = k1 & 0xfffffu;
    uint sb = sa;
    uint64_t ax = uint64_t(a.v);
    uint64_t ay = uint64_t(ya.v);
    uint64_t az = uint64_t(za.v);
    uint64_t bx = uint64_t(b.v);
    uint64_t by = uint64_t(yb.v);
    uint64_t bz = uint64_t(zb.v);
    bool ea = false;
    bool eb = false;
    bool hza = true;
    bool hzb = true;
    if (!fa && gdw3_load(a, sa, ax, ay, az, hza, ea) != 0u)
        return SOP_RECHECK_CODE;
    if (!fb && gdw3_load(b, sb, bx, by, bz, hzb, eb) != 0u)
        return SOP_RECHECK_CODE;
    if (sa != sb)
        return SOP_RECHECK_CODE;                        // PostGIS raises "mixed SRID": the backend does
    if (ea || eb)
    {
        regs[dst].isnull = 1u;                          // an empty point has no distance (PostGIS: NULL)
        return 0u;
    }
    if (d3 && (!hza || !hzb))
        return SOP_RECHECK_CODE;                        // an operand without Z: PostGIS measures in 2D (and may say so)
    if (f64_nonfinite(ax) || f64_nonfinite(ay) || f64_nonfinite(bx) || f64_nonfinite(by) ||
        (d3 && (f64_nonfinite(az) || f64_nonfinite(bz))))
        return SOP_RECHECK_CODE;
    uint64_t dx = f64_sub(bx, ax);
    uint64_t dy = f64_sub(by, ay);
    uint64_t q = f64_sqr_add(dx, f64_mul(dy, dy));
    if (d3)
    {
        uint64_t dz = f64_sub(bz, az);
        if (f64_nonfinite(dz))
            return SOP_RECHECK_CODE;
        q = f64_sqr_add(dz, q);
    }
    if (f64_nonfinite(dx) || f64_nonfinite(dy) || f64_nonfinite(q))
        return SOP_RECHECK_CODE;                        // an overflow: PostGIS' answer (NULL or Infinity) is the backend's
    uint64_t dist = f64_sqrt(q);
    if (f64_lt(0x47EFFFFFE0000000ull, dist))
        return SOP_RECHECK_CODE;                        // beyond FLT_MAX PostGIS answers NULL (its minimum starts there)
    regs[dst].isnull = 0u;
    regs[dst].v = int64_t(dist);
    return 0u;
}

// ST_Distance / ST_3DDistance (GEOMDIST), ST_DWithin (GEOMDWITHIN) and ST_3DDWithin (GEOM3DDW) with an operand that is not a point (a line, a polygon,
// a multi-geometry or a collection; the other operand a point), the same as geom_dwdist_np() of cpu_vm.c. The entry of all three instructions: a
// point and a point go to vm_geom_dist / vm_geom_dwithin / vm_geom_3ddwithin, everything else is measured here, so the walker gd_geom is called from
// one place (and is a real function, [noinline]: inlined it made the join kernels 4 MB bigger, which Metal's compiler could not take). The
// ST_DWithin forms need SOP_GDW_NP (the planner's calibration) and SOP_GDW_EXACT; PostGIS decides them as "distance <= tolerance" with the distance of
// ST_Distance / ST_3DDistance (lwgeom_mindistance2d_tolerance, initial value FLT_MAX), an empty geometry is false. The 3D forms measure lines, points
// and multi-geometries of those that have a Z; a polygon is the backend's.
uint vm_geom_dwdist(uint op, uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    bool isdw = (op == SOP_GEOMDWITHIN || op == SOP_GEOM3DDW);
    bool d3 = (op == SOP_GEOM3DDW) || (op == SOP_GEOMDIST && (k1 & SOP_GDIST_3D) != 0u);
    Reg a = regs[ra];
    Reg b = regs[rb];
    bool fa = (k0 & SOP_GDW_FUSED_A) != 0u;
    bool fb = (k0 & SOP_GDW_FUSED_B) != 0u;
    Reg ya = regs[(k0 >> 8) & 0xffu];
    Reg yb = regs[(k0 >> 16) & 0xffu];
    Reg za = regs[(k1 >> 20) & 0x1fu];
    Reg zb = regs[(k1 >> 25) & 0x1fu];
    Reg t = regs[k0 >> 24];
    if ((a.isnull | b.isnull) != 0u || (fa && ya.isnull != 0u) || (fb && yb.isnull != 0u) || (isdw && t.isnull != 0u) ||
        (d3 && ((fa && za.isnull != 0u) || (fb && zb.isnull != 0u))))
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint hsa = 0u;
    uint hsb = 0u;
    uint hna = 0u;
    uint hnb = 0u;
    uint hoa = 0u;
    uint hob = 0u;
    uint ta = fa ? 1u : gd_hdr(a, hsa, hna, hoa);
    uint tb = fb ? 1u : gd_hdr(b, hsb, hnb, hob);
    if (ta == 0u || tb == 0u)
        return SOP_RECHECK_CODE;
    if (ta == 1u && tb == 1u)
    {
        if (op == SOP_GEOM3DDW)
            return vm_geom_3ddwithin(dst, ra, rb, k0, k1, regs);
        if (!isdw)
            return vm_geom_dist(dst, ra, rb, k0, k1, regs);
        return vm_geom_dwithin(dst, ra, rb, k0, k1, regs);
    }
    if (ta != 1u && tb != 1u)
        return SOP_RECHECK_CODE;                        // two non-points
#ifdef VM_NO_GEOM3D
    if (d3)
        return SOP_RECHECK_CODE;
#endif
    if (isdw ? ((k0 & SOP_GDW_NP) == 0u || (k1 & SOP_GDW_EXACT) == 0u) : (k0 & SOP_GDW_NP) == 0u)
        return SOP_RECHECK_CODE;                        // not calibrated; ST_3DMaxDistance (no flag) is not the minimum
    bool pa = (ta == 1u);                               // the point is operand a
    uint64_t px = pa ? uint64_t(a.v) : uint64_t(b.v);
    uint64_t py = pa ? uint64_t(ya.v) : uint64_t(yb.v);
    uint64_t pz = pa ? uint64_t(za.v) : uint64_t(zb.v);
    uint psrid = k1 & 0xfffffu;
    bool pe = false;
    bool phz = true;
    if (pa ? !fa : !fb)
    {
        if (gdw3_load(pa ? a : b, psrid, px, py, pz, phz, pe) != 0u)
            return SOP_RECHECK_CODE;
    }
    if ((pa ? hsb : hsa) != psrid)
        return SOP_RECHECK_CODE;                        // PostGIS raises "mixed SRID"
    if (d3 && !phz)
        return SOP_RECHECK_CODE;                        // a point without Z: PostGIS measures in 2D (and says so)
    uint64_t tol = uint64_t(t.v);
    if (isdw && (f64_nonfinite(tol) || ((tol >> 63) != 0ull && (tol & F64_MAG) != 0ull)))
        return SOP_RECHECK_CODE;                        // negative (PostGIS raises), NaN or infinite
    if (pe)
    {
        regs[dst].isnull = isdw ? 0u : 1u;              // an empty point: ST_Distance NULL, ST_DWithin false
        regs[dst].v = 0;
        return 0u;
    }
    if (f64_nonfinite(px) || f64_nonfinite(py) || (d3 && f64_nonfinite(pz)))
        return SOP_RECHECK_CODE;
    uint64_t best = 0x7FEFFFFFFFFFFFFFull;
    bool found = false;
    if (gd_geom(pa ? b : a, px, py, pz, d3, best, found) != 0u)
        return SOP_RECHECK_CODE;
    if (!found)
    {
        regs[dst].isnull = isdw ? 0u : 1u;              // the other geometry is empty
        regs[dst].v = 0;
        return 0u;
    }
    regs[dst].isnull = 0u;
    if (isdw)
    {
        if (f64_lt(0x47EFFFFFE0000000ull, best))
            return SOP_RECHECK_CODE;                    // beyond FLT_MAX, PostGIS' initial distance
        regs[dst].v = f64_le(best, tol) ? 1 : 0;
    }
    else
    {
        if (f64_lt(0x47EFFFFFE0000000ull, best))
            return SOP_RECHECK_CODE;                    // beyond FLT_MAX PostGIS answers NULL
        regs[dst].v = int64_t(best);
    }
    return 0u;
}

// ST_Length / ST_Length2D / ST_3DLength (k0 0 / 1) and ST_Perimeter / ST_Perimeter2D / ST_3DPerimeter (k0 2 / 3) of a geometry (GEOMLEN),
// the same as geom_len() of cpu_vm.c. PostGIS adds up per geometry: a line's length is the sum of its segment lengths
// sqrt(fma(dz, dz, fma(dx, dx, dy * dy))) (the Z term for the 3D forms of a geometry with Z) added left to right from 0; a polygon's
// perimeter is the sum of its rings' lengths; a multi-geometry or collection adds up what its members return (each member's own total
// first, then added to the container's running total), and a member that does not count (a point for both, a line for the perimeter, a
// polygon for the length) adds nothing. Points, lines, polygons, their multi-geometries and a collection of those are read here; a
// collection inside a collection, curves and the like go back to the backend. A cached box of any dimension is skipped.
// One loop over the members (a point, a line or a polygon; the members of a multi-geometry, or of a collection, which may itself hold a
// multi-geometry), one loop over a member's arrays (a line; a polygon's rings), one segment computation: each heavy step is written, and
// so inlined, once (the Slang compiler copies a function at every call site).
#define AREA_FN
#define AR_F2 ArF2s
#define AR_SRC_T uint
#define AR_LD(src, off) ld64s(src, off)
#include "area_ring.inc"

uint vm_geom_len(uint dst, uint ra, uint k0, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    if (x.isnull != 0u)
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint s = str_src(x);
    uint b = str_off(x);
    uint l = str_len(x);
    if (l < 12u || (ldb_s(s, b) & 0xe0u) != 0u)
        return SOP_RECHECK_CODE;
    uint gflags = ldb_s(s, b + 3u);
    if ((gflags & 0x08u) != 0u)
        return SOP_RECHECK_CODE;                        // geodetic
    uint hz = gflags & 1u;
    uint hm = (gflags >> 1u) & 1u;
    uint nd = 2u + hz + hm;
    uint o = 4u;
    if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
        o += 8u;
    if ((gflags & 0x04u) != 0u)
        o += 8u * nd;                                   // the cached float box: two floats per dimension
    if (o + 8u > l)
        return SOP_RECHECK_CODE;
    bool per = (k0 >= 2u);
    bool area = (k0 == 4u);                             // ST_Area: the polygons' rings by PostGIS' ptarray_signed_area (area_ring.inc)
    bool d3 = ((k0 & 1u) != 0u) && (hz != 0u);
    uint end = b + l;
    uint p = b + o;
    uint gtype = ldw_s(s, p);
    if (gtype == 0u || gtype > 7u)
        return SOP_RECHECK_CODE;
    uint stride = 8u * nd;
    uint rem0 = 1u;                                     // members still to read at the top level
    uint rem1 = 0u;                                     // members left in a multi-geometry inside a collection
    bool inner = false;
    if (gtype >= 4u)
    {
        rem0 = ldw_s(s, p + 4u);
        p += 8u;
        if (rem0 > (end - p) / 8u)
            return SOP_RECHECK_CODE;
    }
    uint64_t tot = 0ull;                                // the container's running total
    uint64_t sub = 0ull;                                // a multi-geometry inside a collection
    for (uint guard = 0u; guard < 65536u; guard++)
    {
        if (inner)
        {
            if (rem1 == 0u)
            {
                tot = f64_add(tot, sub);                // the inner multi-geometry is done
                inner = false;
                if (f64_nonfinite(tot))
                    return SOP_RECHECK_CODE;
            }
            else
                rem1--;
        }
        if (!inner)
        {
            if (rem0 == 0u)
                break;
            rem0--;
            if (gtype == 7u)
            {
                if (p + 8u > end)
                    return SOP_RECHECK_CODE;
                uint st = ldw_s(s, p);
                if (st >= 4u && st <= 6u)
                {
                    rem1 = ldw_s(s, p + 4u);
                    p += 8u;
                    if (rem1 > (end - p) / 8u)
                        return SOP_RECHECK_CODE;
                    sub = 0ull;
                    inner = true;
                    continue;
                }
            }
        }
        // the member at p; c is what it adds to its container, (aoff, an) the arrays it measures
        if (p + 8u > end)
            return SOP_RECHECK_CODE;
        uint t = ldw_s(s, p);
        uint n = ldw_s(s, p + 4u);
        uint base = p + 8u;
        uint64_t c = 0ull;
        uint narr = 0u;                                 // arrays to measure: a line has one, a polygon n rings, a point none
        uint rbase = 0u;                                // where a polygon's ring counts are
        uint data = 0u;
        if (t == 1u)
        {
            if (n > 1u || (n == 1u && base + stride > end))
                return SOP_RECHECK_CODE;
            p = base + n * stride;
        }
        else if (t == 2u)
        {
            if (n > (end - base) / stride)
                return SOP_RECHECK_CODE;
            narr = per ? 0u : 1u;
            data = base;
        }
        else if (t == 3u)
        {
            data = base + 4u * n + ((n & 1u) != 0u ? 4u : 0u);
            if (n > (end - base) / 4u || data > end)
                return SOP_RECHECK_CODE;
            narr = per ? n : 0u;
            rbase = base;
        }
        else
            return SOP_RECHECK_CODE;
        uint q = data;                                  // the next array (the points of a line are one array of n points)
        for (uint ar = 0u; ar < narr; ar++)
        {
            uint np = (t == 2u) ? n : ldw_s(s, rbase + 4u * ar);
            if (np > (end - q) / stride)
                return SOP_RECHECK_CODE;
            uint64_t asum = 0ull;
            if (area)
            {
                // lwpoly_area: a ring of fewer than 3 points is skipped; the exterior ring's |area| is added, a hole's subtracted
                if (np >= 3u)
                {
                    uint64_t sa = area_ring_signed(s, q, stride, np);
                    if (sa == AR_NONE)
                        return SOP_RECHECK_CODE;
                    sa = sa & F64_MAG;
                    c = (ar == 0u) ? f64_add(c, sa) : f64_sub(c, sa);
                }
                q += np * stride;
                continue;
            }
            for (uint i = 0u; i + 1u < np; i++)
            {
                uint p0 = q + i * stride;
                uint p1 = p0 + stride;
                uint64_t dx = f64_sub(ld64s(s, p1), ld64s(s, p0));
                uint64_t dy = f64_sub(ld64s(s, p1 + 8u), ld64s(s, p0 + 8u));
                uint64_t qq = f64_sqr_add(dx, f64_mul(dy, dy));
                if (d3)
                    qq = f64_sqr_add(f64_sub(ld64s(s, p1 + 16u), ld64s(s, p0 + 16u)), qq);
                if (f64_nonfinite(dx) || f64_nonfinite(dy) || f64_nonfinite(qq))
                    return SOP_RECHECK_CODE;
                asum = f64_add(asum, f64_sqrt(qq));
                if (f64_nonfinite(asum))
                    return SOP_RECHECK_CODE;
            }
            c = f64_add(c, asum);                       // a line: 0 + its length; a polygon: its rings added in order
            q += np * stride;
        }
        if (t == 2u)
            p = base + n * stride;
        else if (t == 3u)
        {
            // past all the rings (also when the perimeter is not measured)
            uint qe = data;
            for (uint ar = 0u; ar < n; ar++)
            {
                uint np = ldw_s(s, rbase + 4u * ar);
                if (np > (end - qe) / stride)
                    return SOP_RECHECK_CODE;
                qe += np * stride;
            }
            p = qe;
        }
        if (inner)
            sub = f64_add(sub, c);
        else
            tot = f64_add(tot, c);
        if (f64_nonfinite(sub) || f64_nonfinite(tot))
            return SOP_RECHECK_CODE;
    }
    if (inner)
        tot = f64_add(tot, sub);                        // the last member was in an inner multi-geometry
    if (rem0 > 0u || rem1 > 0u || f64_nonfinite(tot))
        return SOP_RECHECK_CODE;
    regs[dst].isnull = 0u;
    regs[dst].v = int64_t(tot);
    return 0u;
}

// ST_HasZ / ST_HasM / ST_NDims / ST_Zmflag / ST_CoordDim (GEOMFLAG): the flags byte of any geometry, the same as geom_flag()
// of cpu_vm.c
// ST_NPoints (k0 6), ST_NumPoints (7), ST_NumGeometries (8), ST_NRings (9), ST_NumInteriorRings (10), ST_IsEmpty (11): the same as geom_cnt()
// of cpu_vm.c. One walk over the members (a multi-geometry inside a collection too; a collection inside a collection, curves and the like
// go back to the backend): PostGIS counts the vertices of every member (an empty member adds 0) and the rings of every polygon; a
// geometry is empty when no member has an element (a polygon: a ring); ST_NumGeometries is 0 for an empty geometry, the member count of a
// collection or 1; ST_NumPoints is the point count of a LINESTRING only and ST_NumInteriorRings that of a POLYGON (rings - 1, 0 when empty),
// NULL for every other type (probe: postgis/measure_probe.sql).
uint vm_geom_cnt(uint dst, uint ra, uint k0, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    if (x.isnull != 0u)
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint s = str_src(x);
    uint b = str_off(x);
    uint l = str_len(x);
    if (l < 12u || (ldb_s(s, b) & 0xe0u) != 0u)
        return SOP_RECHECK_CODE;
    uint gflags = ldb_s(s, b + 3u);
    if ((gflags & 0x08u) != 0u)
        return SOP_RECHECK_CODE;
    uint nd = 2u + (gflags & 1u) + ((gflags >> 1u) & 1u);
    uint o = 4u;
    if ((gflags & 0x40u) != 0u && (gflags & 0x10u) != 0u)
        o += 8u;
    if ((gflags & 0x04u) != 0u)
        o += 8u * nd;
    if (o + 8u > l)
        return SOP_RECHECK_CODE;
    uint end = b + l;
    uint p = b + o;
    uint gtype = ldw_s(s, p);
    if (gtype == 0u || gtype > 7u)
        return SOP_RECHECK_CODE;
    uint stride = 8u * nd;
    uint n0 = ldw_s(s, p + 4u);
    if (k0 == 7u || k0 == 10u)
    {
        // a line's point count / a polygon's interior rings; NULL for every other type
        regs[dst].isnull = 0u;
        if (k0 == 7u ? gtype != 2u : gtype != 3u)
            regs[dst].isnull = 1u;
        else if (k0 == 7u)
            regs[dst].v = int64_t(n0);
        else
            regs[dst].v = int64_t(n0 == 0u ? 0u : n0 - 1u);
        return 0u;
    }
    uint rem0 = 1u;
    uint rem1 = 0u;
    bool inner = false;
    if (gtype >= 4u)
    {
        rem0 = n0;
        p += 8u;
        if (rem0 > (end - p) / 8u)
            return SOP_RECHECK_CODE;
    }
    uint top = rem0;
    uint npts = 0u;
    uint nrings = 0u;
    bool nonempty = false;
    for (uint guard = 0u; guard < 65536u; guard++)
    {
        if (inner)
        {
            if (rem1 == 0u)
                inner = false;
            else
                rem1--;
        }
        if (!inner)
        {
            if (rem0 == 0u)
                break;
            rem0--;
            if (gtype == 7u)
            {
                if (p + 8u > end)
                    return SOP_RECHECK_CODE;
                uint st = ldw_s(s, p);
                if (st >= 4u && st <= 6u)
                {
                    rem1 = ldw_s(s, p + 4u);
                    p += 8u;
                    if (rem1 > (end - p) / 8u)
                        return SOP_RECHECK_CODE;
                    inner = true;
                    continue;
                }
            }
        }
        if (p + 8u > end)
            return SOP_RECHECK_CODE;
        uint t = ldw_s(s, p);
        uint n = ldw_s(s, p + 4u);
        uint base = p + 8u;
        if (t == 1u)
        {
            if (n > 1u || (n == 1u && base + stride > end))
                return SOP_RECHECK_CODE;
            npts += n;
            p = base + n * stride;
        }
        else if (t == 2u)
        {
            if (n > (end - base) / stride)
                return SOP_RECHECK_CODE;
            npts += n;
            p = base + n * stride;
        }
        else if (t == 3u)
        {
            uint data = base + 4u * n + ((n & 1u) != 0u ? 4u : 0u);
            if (n > (end - base) / 4u || data > end)
                return SOP_RECHECK_CODE;
            uint qe = data;
            for (uint ar = 0u; ar < n; ar++)
            {
                uint np = ldw_s(s, base + 4u * ar);
                if (np > (end - qe) / stride)
                    return SOP_RECHECK_CODE;
                if (ar == 0u && np == 0u)
                    return SOP_RECHECK_CODE;            // a polygon whose first ring is empty: PostGIS' emptiness rule differs
                npts += np;
                qe += np * stride;
            }
            nrings += n;
            p = qe;
        }
        else
            return SOP_RECHECK_CODE;
        if (n > 0u)
            nonempty = true;
    }
    if (rem0 > 0u || rem1 > 0u)
        return SOP_RECHECK_CODE;
    uint v = npts;
    if (k0 == 8u)
        v = nonempty ? (gtype >= 4u ? top : 1u) : 0u;
    else if (k0 == 9u)
        v = nrings;
    else if (k0 == 11u)
        v = nonempty ? 0u : 1u;
    regs[dst].isnull = 0u;
    regs[dst].v = int64_t(v);
    return 0u;
}

uint vm_geom_flag(uint dst, uint ra, uint k0, inout Reg regs[VM_MAX_REGS])
{
    if (k0 >= 6u)
        return vm_geom_cnt(dst, ra, k0, regs);
    Reg x = regs[ra];
    if (x.isnull != 0u)
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    if (str_len(x) < 8u || (ldb_s(str_src(x), str_off(x)) & 0xe0u) != 0u)
        return SOP_RECHECK_CODE;                        // not a header we know
    uint gflags = ldb_s(str_src(x), str_off(x) + 3u);
    uint hz = gflags & 1u;
    uint hm = (gflags >> 1u) & 1u;
    uint v = 2u + hz + hm;                              // NDims, CoordDim
    if (k0 == SOP_GFLAG_SRID)
    {
        // the 21-bit SRID (a value with bit 20 set would be sign-extended by PostGIS: the backend's)
        uint sb0 = ldb_s(str_src(x), str_off(x));
        if ((sb0 & 0x10u) != 0u)
            return SOP_RECHECK_CODE;
        regs[dst].isnull = 0u;
        regs[dst].v = int64_t((sb0 << 16) | (ldb_s(str_src(x), str_off(x) + 1u) << 8) | ldb_s(str_src(x), str_off(x) + 2u));
        return 0u;
    }
    if (k0 == SOP_GFLAG_HASZ)
        v = hz;
    else if (k0 == SOP_GFLAG_HASM)
        v = hm;
    else if (k0 == SOP_GFLAG_ZMFLAG)
        v = 2u * hz + hm;
    regs[dst].isnull = 0u;
    regs[dst].v = int64_t(v);
    return 0u;
}

// ST_3DDWithin of two points (GEOM3DDW), the same as geom_3ddwithin() of cpu_vm.c
uint vm_geom_3ddwithin(uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg a = regs[ra];
    Reg b = regs[rb];
    bool fa = (k0 & SOP_GDW_FUSED_A) != 0u;
    bool fb = (k0 & SOP_GDW_FUSED_B) != 0u;
    Reg ya = regs[(k0 >> 8) & 0xffu];
    Reg yb = regs[(k0 >> 16) & 0xffu];
    Reg t = regs[k0 >> 24];
    Reg za = regs[(k1 >> 20) & 0x1fu];
    Reg zb = regs[(k1 >> 25) & 0x1fu];
    if ((a.isnull | b.isnull | t.isnull) != 0u || (fa && (ya.isnull | za.isnull) != 0u) || (fb && (yb.isnull | zb.isnull) != 0u))
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint sa = k1 & 0xfffffu;
    uint sb = sa;
    uint64_t ax = uint64_t(a.v);
    uint64_t ay = uint64_t(ya.v);
    uint64_t az = uint64_t(za.v);
    uint64_t bx = uint64_t(b.v);
    uint64_t by = uint64_t(yb.v);
    uint64_t bz = uint64_t(zb.v);
    bool ea = false;
    bool eb = false;
    bool hza = true;
    bool hzb = true;
    if (!fa && gdw3_load(a, sa, ax, ay, az, hza, ea) != 0u)
        return SOP_RECHECK_CODE;
    if (!fb && gdw3_load(b, sb, bx, by, bz, hzb, eb) != 0u)
        return SOP_RECHECK_CODE;
    if (sa != sb)
        return SOP_RECHECK_CODE;                        // PostGIS raises "mixed SRID": the backend does
    if (ea || eb || !hza || !hzb)
        return SOP_RECHECK_CODE;                        // empty; or no Z: PostGIS raises a NOTICE and measures in 2D
    uint64_t tol = uint64_t(t.v);
    if (f64_nonfinite(tol) || ((tol >> 63) != 0ull && (tol & F64_MAG) != 0ull))
        return SOP_RECHECK_CODE;                        // negative (PostGIS raises), NaN or infinite
    if (f64_nonfinite(ax) || f64_nonfinite(ay) || f64_nonfinite(az) || f64_nonfinite(bx) || f64_nonfinite(by) || f64_nonfinite(bz))
        return SOP_RECHECK_CODE;
    uint64_t dx = f64_sub(ax, bx);
    uint64_t dy = f64_sub(ay, by);
    uint64_t dz = f64_sub(az, bz);
    if (f64_nonfinite(dx) || f64_nonfinite(dy) || f64_nonfinite(dz))
        return SOP_RECHECK_CODE;
    bool res = false;
    if (((dx | dy | dz) & F64_MAG) == 0ull)
        res = true;                                     // the same point: distance 0 <= tolerance
    else if ((k1 & SOP_GDW_EXACT) != 0u)
    {
        uint64_t q = f64_sqr_add(dz, f64_sqr_add(dx, f64_mul(dy, dy)));
        if (f64_nonfinite(q))
            return SOP_RECHECK_CODE;
#ifdef VM_G_FLTMAX_CHECK
        if (f64_lt(0x47EFFFFFE0000000ull, f64_sqrt(q)))
            return SOP_RECHECK_CODE;                    // beyond FLT_MAX, PostGIS' initial distance (see vm_geom_dwithin)
#endif
        res = f64_le(f64_sqrt(q), tol);                 // PostGIS' own double against the tolerance (calibrated by the planner)
    }
    else
    {
        uint64_t d2 = f64_add(f64_add(f64_mul(dx, dx), f64_mul(dy, dy)), f64_mul(dz, dz));
        uint64_t r2 = f64_mul(tol, tol);
        if (f64_nonfinite(d2) || f64_nonfinite(r2) || (d2 & F64_MAG) < GDW_TINY ||
            ((r2 & F64_MAG) != 0ull && (r2 & F64_MAG) < GDW_TINY))
            return SOP_RECHECK_CODE;
        if (f64_le(d2, f64_mul(r2, GDW_LO)))
            res = true;
        else if (f64_lt(f64_mul(r2, GDW_HI), d2))
            res = false;
        else
            return SOP_RECHECK_CODE;                    // within 2^-47 of the boundary
    }
    regs[dst].isnull = 0u;
    regs[dst].v = res ? 1 : 0;
    return 0u;
}

// Returns 0 (res set), or SOP_RECHECK_CODE. a: the polygon register; b: the point register, or with SOP_GPIP_FUSED
// the float8 x (xv) and y (yv) of a point with SRID fsrid.
uint vm_geom_pip(uint dst, uint ra, uint rb, uint k0, uint k1, inout Reg regs[VM_MAX_REGS])
{
    Reg x = regs[ra];
    Reg y = regs[rb];
    bool fused = (k0 & SOP_GPIP_FUSED) != 0u;
    uint yreg = k1 & 0xffu;
    if ((x.isnull | y.isnull) != 0u || (fused && regs[yreg].isnull != 0u))
    {
        regs[dst].isnull = 1u;
        return 0u;
    }
    uint sa = str_src(x);
    uint ba = str_off(x);
    uint la = str_len(x);
    uint sb = str_src(y);
    uint bb = str_off(y);
    uint lb = str_len(y);
    GHead ha = geom_head(sa, ba, la);
    GHead hb;
    uint64_t px = uint64_t(y.v);
    uint64_t py = uint64_t(regs[yreg].v);
    if (ha.bad != 0u)
        return SOP_RECHECK_CODE;
    if (fused)
    {
        hb.bad = 0u;
        hb.gtype = 1u;
        hb.srid = k1 >> 8;
        hb.pos = 0u;
        hb.flags = 0u;
    }
    else
    {
        hb = geom_head(sb, bb, lb);
        if (hb.bad != 0u)
            return SOP_RECHECK_CODE;
        if ((k0 & SOP_GPIP_SYMM) != 0u && ha.gtype == 1u && (hb.gtype == 3u || hb.gtype == 6u))
        {
            GHead th = ha; ha = hb; hb = th;
            uint t;
            t = sa; sa = sb; sb = t;
            t = ba; ba = bb; bb = t;
            t = la; la = lb; lb = t;
        }
    }
    if ((ha.gtype != 3u && ha.gtype != 6u) || hb.gtype != 1u)
        return SOP_RECHECK_CODE;
    if (ha.srid != hb.srid)
        return SOP_RECHECK_CODE;                        // PostGIS raises "mixed SRID": the backend does
    if ((ha.flags & 0x0bu) != 0u)
        return SOP_RECHECK_CODE;                        // Z, M or geodetic polygon
    bool empty = false;
    if (!fused)
    {
        if ((hb.flags & 0x0cu) != 0u || hb.pos + 8u > lb)
            return SOP_RECHECK_CODE;
        uint npts = ldw_s(sb, bb + hb.pos + 4u);
        if (npts == 0u)
            empty = true;
        else
        {
            if (npts != 1u || hb.pos + 8u + 16u > lb)
                return SOP_RECHECK_CODE;
            px = ld64s(sb, bb + hb.pos + 8u);
            py = ld64s(sb, bb + hb.pos + 16u);
        }
    }
    bool res = false;
    if (!empty)
    {
        if (f64_nonfinite(px) || f64_nonfinite(py))
            return SOP_RECHECK_CODE;
        bool outside_box = false;
        if ((ha.flags & 0x04u) != 0u)
        {
            // the cached float box is rounded outward: a point beyond it is outside
            uint bo = ba + ha.pos - 16u;
            uint bx0 = ldw_s(sa, bo);
            uint bx1 = ldw_s(sa, bo + 4u);
            uint by0 = ldw_s(sa, bo + 8u);
            uint by1 = ldw_s(sa, bo + 12u);
            if ((bx0 & 0x7fffffffu) <= 0x7f800000u && (bx1 & 0x7fffffffu) <= 0x7f800000u &&
                (by0 & 0x7fffffffu) <= 0x7f800000u && (by1 & 0x7fffffffu) <= 0x7f800000u &&
                (f64_lt(px, f64_from_f32(bx0)) || f64_gt(px, f64_from_f32(bx1)) ||
                 f64_lt(py, f64_from_f32(by0)) || f64_gt(py, f64_from_f32(by1))))
                outside_box = true;
        }
        if (!outside_box)
        {
            uint rr = PIP_OUT;
            uint nx;
            if (ha.gtype == 3u)
                rr = pip_polygon(sa, ba, la, ha.pos, px, py, nx);
            else
            {
                if (ha.pos + 8u > la)
                    return SOP_RECHECK_CODE;
                uint ng = ldw_s(sa, ba + ha.pos + 4u);
                uint o = ha.pos + 8u;
                // PostGIS: the first polygon the point is not outside of decides (inside, or on its boundary)
                for (uint i = 0u; i < ng; i++)
                {
                    uint r1 = pip_polygon(sa, ba, la, o, px, py, nx);
                    if (nx == 0u)
                        return SOP_RECHECK_CODE;
                    if (r1 != PIP_OUT)
                    {
                        rr = r1;
                        break;
                    }
                    o = nx;
                }
            }
            if (rr == PIP_UNSURE)
                return SOP_RECHECK_CODE;
            res = (rr == PIP_IN);
        }
    }
    regs[dst].isnull = 0u;
    regs[dst].v = b2l(res);
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
#ifndef VM_NO_NUM
        if (op >= SOP_NUMCONST && op <= SOP_NUMRND)
            return vm_num(op, dst, ra, rb, k0, k1, regs);
#endif
#ifndef VM_NO_JSON
        if (op >= SOP_JSONGET && op <= SOP_JSONCONT)
            return vm_json(op, dst, ra, rb, k0, regs);
#endif
#ifndef VM_NO_F8
        if (op >= SOP_CMPD && op <= SOP_D2F)
            return vm_f8(op, dst, ra, rb, k0, k1, regs);
#endif
#if (VM_GEOM_MASK & VM_G_BOX) != 0
        if (op == SOP_GEOMBOX)
            return vm_geom(op, dst, ra, rb, k0, k1, regs);
#endif
#if (VM_GEOM_MASK & VM_G_PIP) != 0
        if (op == SOP_GEOMPIP)
            return vm_geom_pip(dst, ra, rb, k0, k1, regs);
#endif
#if (VM_GEOM_MASK & VM_G_WALK) != 0
        // ONE call site for the three: the walker inside is a real function ([noinline]) and is called from vm_geom_dwdist only
        if (op == SOP_GEOMDWITHIN || op == SOP_GEOM3DDW || op == SOP_GEOMDIST)
            return vm_geom_dwdist(op, dst, ra, rb, k0, k1, regs);
#else
        // point and point only (a non-point operand is SOP_RECHECK_CODE / a poison record): each instruction is a tier of its own,
        // their soft-float code is 1-2 MB of SPIR-V apiece (plan R15)
#if (VM_GEOM_MASK & VM_G_DW) != 0
        if (op == SOP_GEOMDWITHIN)
            return vm_geom_dwithin(dst, ra, rb, k0, k1, regs);
#endif
#if (VM_GEOM_MASK & VM_G_DW3) != 0
        if (op == SOP_GEOM3DDW)
            return vm_geom_3ddwithin(dst, ra, rb, k0, k1, regs);
#endif
#if (VM_GEOM_MASK & VM_G_DIST) != 0
        if (op == SOP_GEOMDIST)
            return vm_geom_dist(dst, ra, rb, k0, k1, regs);
#endif
#endif
#if (VM_GEOM_MASK & VM_G_ZM) != 0
        if (op == SOP_GEOMZM)
            return vm_geom_zm(dst, ra, k0, regs);
#endif
#if (VM_GEOM_MASK & VM_G_FLAG) != 0
        if (op == SOP_GEOMFLAG)
            return vm_geom_flag(dst, ra, k0, regs);
#endif
#if (VM_GEOM_MASK & VM_G_LEN) != 0
        if (op == SOP_GEOMLEN)
            return vm_geom_len(dst, ra, k0, regs);
#endif
        if (op == SOP_NULLMRG)
        {
            Reg x = regs[ra];
            Reg y = regs[rb];
            regs[dst].v = x.v;
            regs[dst].isnull = x.isnull | y.isnull;
            return 0u;
        }
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
            if (regs[ra].isnull == 0 && (k1 == 3 || k1 == 4))
            {
                // a float4 (k1 = 3) / float8 (k1 = 4) key: -0 hashes as +0 and every NaN as the canonical one (hashfloat4/8)
                uint64_t uv = uint64_t(regs[ra].v);
                uint lo = uint(uv & 0xffffffffu);
                uint hi = uint(uv >> 32);
                if (k1 == 3)
                {
                    hi = 0;
                    if ((lo & 0x7fffffffu) == 0)
                        lo = 0;
                    else if ((lo & 0x7fffffffu) > 0x7f800000u)
                        lo = 0x7fc00000u;
                }
                else
                {
                    uint ah = hi & 0x7fffffffu;
                    if ((ah | lo) == 0)
                    {
                        lo = 0;
                        hi = 0;
                    }
                    else if (ah > 0x7ff00000u || (ah == 0x7ff00000u && lo != 0))
                    {
                        lo = 0;
                        hi = 0x7ff80000u;
                    }
                }
                uint h = hash_any_value(lo, hi, k1 == 3 ? 4u : 8u);
                uint acc = uint(regs[dst].v);
                regs[dst].v = int64_t(((acc >> 3) | (acc << 29)) ^ h);
            }
            else if (regs[ra].isnull == 0 && k1 == 5)
            {
                // a numeric key { unscaled int64, scale }: equal values hash alike whatever the scale (1.5, 1.50): the
                // trailing zeros are stripped, then the value and the scale are hashed (devtype_numeric_hash)
                int64_t nv = regs[ra].v;
                uint nsc = regs[ra].sc;
                if (nv == 0)
                    nsc = 0;
                else
                {
                    while (nsc > 0 && (nv % 10) == 0)
                    {
                        nv = nv / 10;
                        nsc = nsc - 1;
                    }
                }
                uint64_t uv = uint64_t(nv);
                uint h = hash_any_value(uint(uv & 0xffffffffu), uint(uv >> 32), 8u) ^ hash_any_value(nsc, 0u, 4u);
                uint acc = uint(regs[dst].v);
                regs[dst].v = int64_t(((acc >> 3) | (acc << 29)) ^ h);
            }
            else if (regs[ra].isnull == 0 && k1 != 0)
            {
                // a string key (k1 = 1 text / varchar, 2 bpchar: trailing blanks do not count)
                Reg x = regs[ra];
                uint ss = str_src(x);
                uint so = str_off(x);
                uint sl = str_len(x);
                if (k1 == 2)
                    sl = str_truelen(ss, so, sl);
                uint h = hash_bytes_src(ss, so, sl);
                uint acc = uint(regs[dst].v);
                regs[dst].v = int64_t(((acc >> 3) | (acc << 29)) ^ h);
            }
            else if (regs[ra].isnull == 0)
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
            else if (slot != 0xffu && vt == SOP_VT_STR)
            {
                // a fixed-length by-reference value (a 16 byte uuid): a string register over its bytes
                regs[slot].v = int64_t(uint64_t(tbase + off) | (uint64_t(alen) << 32) | (uint64_t(g_src) << 60));
                regs[slot].isnull = 0;
                regs[slot].sc = 0;
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
