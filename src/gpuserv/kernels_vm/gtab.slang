// GpuPreAgg with GROUP BY over many groups (A3): a hash table in global memory that lives for the whole session.
//
// gagg.slang keeps a table of 64-128 slots per workgroup in shared memory and sends the rows that find no slot to the
// CPU, which is right for tens of groups and hopeless for thousands. Here one table serves every workgroup and every
// task of the session; the service reads it once, at the end (or when it has to grow it).
//
// No thread ever waits for another (the Vulkan model gives no forward progress guarantee, and the base tier has no
// memory model and no 64-bit atomics), and a key written by one thread is read by another only after a dispatch has
// ended (the dispatch boundary is the one synchronization point with defined visibility). So a task runs in up to
// three dispatches over the same pages (the mode is in the top 4 bits of kparam.page_base):
//   0  every row is evaluated, its key looked up in the table; a found row adds its values to the slot's accumulators with
//      32-bit atomics, a row whose key is not in the table is flagged 4 (its values are not added)
//   1  (only if rows were flagged 4) the flagged rows insert their keys: a slot is claimed with one compare-and-swap of its
//      tag (the 32-bit hash of the key); the thread that wins writes the key words, a thread that finds the tag
//      equal does nothing (the same key, or a different key with the same hash, which then never gets a slot and is
//      aggregated on the CPU)
//   2  the flagged rows look up again and add their values; a row that still finds no slot (probe limit, table
//      full, hash collision) is flagged 1: the service aggregates it on the CPU (exact)
// A row is flagged 2 if the backend must evaluate it (a string or numeric the kernel cannot read, as in gagg.slang).
//
// Table buffer (gtab): header { 0: slots claimed, 1: rows that qualified, 2: slot mask, 3: words per slot }, from word 8
// the slots, each { tag, null mask, (lo, hi) of each of the GA_NK = 4 keys (unused keys 0), accumulators of the aggregates in order }.
//   count:   { n }
//   sum:     { n, Lpos, Cpos, Lneg, Cneg, nbig, Lxl, Cxl, Lxh, Cxh }   v in [0, 2^32): Lpos += v;  v in (-2^32, 0): Lneg += -v;
//            other v: x = v + 2^63 (unsigned), Lxl += low word of x, Lxh += high word, nbig++. L is a 32-bit sum and C counts
//            its wraps (a wrap is seen by the add that causes it: the add returns the old value), so the total of a pair is
//            L + C * 2^32. sum = (Pos) - (Neg) + (Lxl + 2^32 Lxh - nbig * 2^63).
//   min/max: { flag, ~b (min) or b (max) } with b = value + 2^31 as unsigned (values outside 32 bits overflow to the CPU)
// Flags (as gagg.slang): outb[flags_base + ...] one word per line pointer (ARROW_SRC: one per row of the pass, at page * AR_VP + lp),
// 0 done, 1 CPU, 2 backend, 4 insert. ARROW_SRC has no heap tuple to aggregate on the CPU, so a mode 3 dispatch writes
// the rows flagged 1 (staged key and argument values) to an overflow list { count, records of OV_REC words } at kparam.total_lps - ...:
// its position is outb[ovf_base] with ovf_base = kparam.extra (mode 3 only; the flags start at 4).
// Program header as gagg.slang: [10] aggregates, [11] aggregate table, [12] keys, [13] key registers.
// String keys (A6): header word 21 has bit i set when key i is a string (text, varchar) and bit 4 + i when it is a bpchar. The
// staged key of such a key is the string register { offset | length << 32 | source << 60 } of the row's value, which stays valid
// for the task (the pages are the task's): the tag hashes the bytes (hash_bytes_src, bpchar without the trailing blanks), two
// keys are equal if their bytes are, and a slot's key words point at the first row that inserted it. The service therefore reads
// the table and clears it at the end of every task (the pointers do not outlive the pages), and resolves the bytes then.
#define GTAB 1
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif

#ifndef GA_NA
#define GA_NA 12
#endif
#define GA_NK 4
#define GT_MAXPROBE 64
#define GT_HDR 8
#define GT_KEYW (2 + 2 * GA_NK)
#define OV_REC (1 + 2 * GA_NK + 3 * GA_NA)

[[vk::binding(5, 0)]] RWStructuredBuffer<uint> gtab;

static const uint AC_COUNT_ANY  = 1;
static const uint AC_COUNT_COND = 2;
static const uint AC_SUM        = 3;
static const uint AC_MIN        = 4;
static const uint AC_MAX        = 5;
static const uint AC_SMIN       = 14;           // min / max of text (14, 15) and bpchar (16, 17), A6b
static const uint NONE_REG      = 0xffffffffu;
static const uint NO_SLOT       = 0xffffffffu;

static uint g_kstr;                                 // header word 21: the string keys (see above)
static uint g_side;                                 // header word 22: first word of the per-row side area of the text min / max (outb)
static uint g_rw;                                   // header word 23: words per row of the side area: 1 + 2 * (text aggregates)

// bytewise order of two string registers (the C collation: memcmp, then the length); bpchar ignores the trailing blanks (as agg.slang)
int str_reg_cmp(int64_t x, int64_t y, bool bp)
{
    uint sa = uint((uint64_t(x) >> 60) & 3u), sb = uint((uint64_t(y) >> 60) & 3u);
    uint oa = uint(uint64_t(x) & 0xffffffffu), ob = uint(uint64_t(y) & 0xffffffffu);
    uint la = uint((uint64_t(x) >> 32) & 0x0fffffffu), lb = uint((uint64_t(y) >> 32) & 0x0fffffffu);
    if (bp)
    {
        la = str_truelen(sa, oa, la);
        lb = str_truelen(sb, ob, lb);
    }
    int c = str_memcmp(sa, oa, sb, ob, min(la, lb));
    if (c == 0 && la != lb)
        c = (la < lb) ? -1 : 1;
    return c;
}

groupshared uint    sh_hits;
groupshared uint    st_kn[SCAN_WG];                 // null mask | 0x80000000 if the row qualifies
groupshared uint    st_hash[SCAN_WG];
groupshared int64_t st_key[SCAN_WG * GA_NK];
groupshared uint    st_ok[SCAN_WG * GA_NA];
groupshared int64_t st_val[SCAN_WG * GA_NA];

uint hash_key_words(int64_t k, uint bit, uint kn, uint h)
{
    return (h ^ (uint(k) ^ (uint(k >> 32) * 0x9e3779b1u) ^ (((kn >> bit) & 1u) * 0x85ebca6bu))) * 0x01000193u;
}

// the bytes of a string key: its source, offset and length (bpchar: without the trailing blanks)
uint gt_str_len(int64_t k, bool bp)
{
    uint len = uint((uint64_t(k) >> 32) & 0x0fffffffu);
    if (bp)
        len = str_truelen(uint((uint64_t(k) >> 60) & 3u), uint(uint64_t(k) & 0xffffffffu), len);
    return len;
}

uint hash_keys(int64_t k[GA_NK], uint kn, uint kstr)
{
    uint h = 0x811c9dc5u;
    for (uint i = 0; i < GA_NK; i++)
    {
        if (((kstr >> i) & 1u) != 0 && ((kn >> i) & 1u) == 0)
        {
            uint hv = hash_bytes_src(uint((uint64_t(k[i]) >> 60) & 3u), uint(uint64_t(k[i]) & 0xffffffffu),
                                     gt_str_len(k[i], ((kstr >> (4 + i)) & 1u) != 0));
            h = (h ^ hv) * 0x01000193u;
        }
        else
            h = hash_key_words(k[i], i, kn, h);
    }
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    h *= 0x297a2d39u;
    h ^= h >> 15;
    return h;
}

#ifdef SPEC_PROG
#include "spec_gen.inc"
#endif

uint acc_width(uint cls)
{
    if (cls == AC_COUNT_ANY || cls == AC_COUNT_COND)
        return 1;
    if (cls == AC_SUM)
        return 10;
    return 2;
}

// word i += x, and the wrap into word c
void add_wrap(uint i, uint c, uint x)
{
    uint prev;
    InterlockedAdd(gtab[i], x, prev);
    if (prev + x < prev)
    {
        uint d;
        InterlockedAdd(gtab[c], 1, d);
    }
}

// the first word of the slot of a key, NO_SLOT if the table has no such slot (yet)
uint gt_find(uint tag, uint kn, uint tid, uint mask, uint stride, uint nkeys)
{
    uint s = tag & mask;
    for (uint p = 0; p < GT_MAXPROBE; p++)
    {
        uint base = GT_HDR + s * stride;
        uint t = gtab[base];
        if (t == 0)
            return NO_SLOT;
        if (t == tag && gtab[base + 1] == kn)
        {
            bool same = true;
            for (uint i = 0; i < nkeys; i++)          // (the words of the unused keys are 0 on both sides)
            {
                uint64_t kv = uint64_t(st_key[tid * GA_NK + i]);
                if (((g_kstr >> i) & 1u) != 0 && ((kn >> i) & 1u) == 0)
                {
                    // a string key: the same bytes (the slot's words point at the first row's value in the task's pages)
                    bool bp = ((g_kstr >> (4 + i)) & 1u) != 0;
                    uint64_t sv = uint64_t(gtab[base + 2 + 2 * i]) | (uint64_t(gtab[base + 3 + 2 * i]) << 32);
                    uint la = gt_str_len(int64_t(sv), bp);
                    uint lb = gt_str_len(int64_t(kv), bp);
                    if (la != lb || str_memcmp(uint((sv >> 60) & 3u), uint(sv & 0xffffffffu),
                                               uint((kv >> 60) & 3u), uint(kv & 0xffffffffu), la) != 0)
                        same = false;
                }
                else if (gtab[base + 2 + 2 * i] != uint(kv & 0xffffffffu) || gtab[base + 3 + 2 * i] != uint(kv >> 32))
                    same = false;
            }
            if (same)
                return base;
        }
        s = (s + 1) & mask;
    }
    return NO_SLOT;
}

void gt_insert(uint tag, uint kn, uint tid, uint mask, uint stride)
{
    uint s = tag & mask;
    for (uint p = 0; p < GT_MAXPROBE; p++)
    {
        uint base = GT_HDR + s * stride;
        uint t = gtab[base];
        if (t == tag)
            return;
        if (t == 0)
        {
            uint prev;
            InterlockedCompareExchange(gtab[base], 0, tag, prev);
            if (prev == 0)
            {
                gtab[base + 1] = kn;
                for (uint i = 0; i < GA_NK; i++)
                {
                    uint64_t kv = uint64_t(st_key[tid * GA_NK + i]);
                    gtab[base + 2 + 2 * i] = uint(kv & 0xffffffffu);
                    gtab[base + 3 + 2 * i] = uint(kv >> 32);
                }
                uint d;
                InterlockedAdd(gtab[0], 1, d);
                return;
            }
            if (prev == tag)
                return;
        }
        s = (s + 1) & mask;
    }
}

// the string register of a side-area entry (bit 30 of the high word marks a value)
int64_t gt_side_reg(uint w)
{
    return int64_t(uint64_t(outb[w]) | (uint64_t(outb[w + 1] & 0xbfffffffu) << 32));
}

// min / max of a string (A6b), after the dispatches that added the rows: a row that has a text value found its slot in mode 0 or 2
// and left { slot, registers } in the side area (outb, written by an earlier dispatch, so visible). Here the row competes for the slot's
// word, which holds the winning row + 1: a compare-and-swap loop that only ever installs a better row (lock free: a failed swap means
// another row improved the word), comparing the bytes of the registers, which point into the task's pages (read only).
void gt_text_row(uint row, uint naggs, uint aggoff)
{
    uint sb = g_side + row * g_rw;
    uint slot = outb[sb];
    if (slot == 0)
        return;
    uint off = slot + GT_KEYW;
    uint t = 0;
    for (uint k = 0; k < naggs; k++)
    {
        uint cls = prog[aggoff + 4 * k];
        if (cls >= AC_SMIN)
        {
            if ((outb[sb + 2 + 2 * t] & 0x40000000u) != 0)
            {
                int64_t my = gt_side_reg(sb + 1 + 2 * t);
                bool bp = (cls >= AC_SMIN + 2u);
                bool smin = (((cls - AC_SMIN) & 1u) == 0);
                uint cur = gtab[off];
                for (uint it = 0; it < 65536u; it++)
                {
                    if (cur != 0)
                    {
                        int c = str_reg_cmp(my, gt_side_reg(g_side + (cur - 1) * g_rw + 1 + 2 * t), bp);
                        if (smin ? (c >= 0) : (c <= 0))
                            break;
                    }
                    uint prev;
                    InterlockedCompareExchange(gtab[off], cur, row + 1, prev);
                    if (prev == cur)
                        break;
                    cur = prev;
                    if (it == 65535u)
                        report_error(0, row, 0x7ffffffbu);
                }
            }
            t++;
        }
        off += acc_width(cls);
    }
}

void gt_accumulate(uint base, uint tid, uint naggs, uint aggoff, uint row)
{
    uint off = base + GT_KEYW;
    uint t = 0;
    for (uint k = 0; k < naggs; k++)
    {
        uint cls = prog[aggoff + 4 * k];
        if (cls >= AC_SMIN)
        {
            // min / max of a string: only the row's side entry here, the slot is updated by mode 4
            if (st_ok[tid * GA_NA + k] != 0)
            {
                uint64_t rg = uint64_t(st_val[tid * GA_NA + k]);
                uint sb = g_side + row * g_rw;
                outb[sb] = base;
                outb[sb + 1 + 2 * t] = uint(rg & 0xffffffffu);
                outb[sb + 2 + 2 * t] = uint(rg >> 32) | 0x40000000u;
            }
            t++;
        }
        else if (st_ok[tid * GA_NA + k] != 0)
        {
            int64_t v = st_val[tid * GA_NA + k];
            uint prev;
            if (cls == AC_COUNT_ANY || cls == AC_COUNT_COND)
                InterlockedAdd(gtab[off], 1, prev);
            else if (cls == AC_SUM)
            {
                InterlockedAdd(gtab[off], 1, prev);
                if (v >= 0 && v < 4294967296)
                    add_wrap(off + 1, off + 2, uint(v));
                else if (v < 0 && v > -4294967296)
                    add_wrap(off + 3, off + 4, uint(-v));
                else
                {
                    uint64_t x = uint64_t(v) ^ 0x8000000000000000ull;
                    InterlockedAdd(gtab[off + 5], 1, prev);
                    add_wrap(off + 6, off + 7, uint(x & 0xffffffffu));
                    add_wrap(off + 8, off + 9, uint(x >> 32));
                }
            }
            else
            {
                // a plain read first: the flag only goes 0 -> 1 and the extreme only improves, so a stale read costs at most
                // one atomic that changes nothing, and the common case (nothing to change) needs none
                uint b = uint(int(v)) ^ 0x80000000u;
                if (gtab[off] == 0)
                    InterlockedOr(gtab[off], 1, prev);
                uint cur = gtab[off + 1];
                uint nb = (cls == AC_MIN) ? ~b : b;
                if (nb > cur)
                    InterlockedMax(gtab[off + 1], nb, prev);
            }
        }
        off += acc_width(cls);
    }
}

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void gagg_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint wg = (kparam.page_base & 0x0fffffffu) + gid.x;     // page_base counts workgroups here; bits 28.. are the mode
    uint ppw = kparam.blcksz;                       // pages per workgroup
    uint first_page = wg * ppw;
    if (first_page >= kparam.npages)
        return;                                     // uniform: the whole group leaves
    uint last_page = min(first_page + ppw, kparam.npages);
    uint tid = lid.x;
    uint mode = kparam.page_base >> 28;

    uint coloff  = prog[0];
    uint insnoff = prog[2];
    uint ncols   = prog[3];
    uint nq      = prog[5];
    uint nins    = prog[6];
    uint qreg    = prog[7];
    uint naggs   = prog[10];
    uint aggoff  = prog[11];
    uint nkeys   = prog[12];
    uint keyoff  = prog[13];
    uint flags_base = kparam.total_lps;
    g_kstr       = prog[21];
    g_side       = prog[22];
    g_rw         = prog[23];
    uint mask    = gtab[2];
    uint stride  = gtab[3];
    uint a_cls[GA_NA];
    for (uint k = 0; k < GA_NA; k++)
        a_cls[k] = (k < naggs) ? prog[aggoff + 4 * k] : 0;

    if (tid == 0)
        sh_hits = 0;
    GroupMemoryBarrierWithGroupSync();

    for (uint page = first_page; page < last_page; page++)
    {
#ifdef ARROW_SRC
        uint nlp      = arrow_page_rows(page, ncols);
        uint row0     = arrow_page_row0(page, ncols);
#else
        uint pte      = ptab[2 * page];
        uint lp_base  = ptab[2 * page + 1];
        g_src         = (pte & 0x80000000u) != 0 ? 1u : 0u;
        uint pbase    = pte & 0x7fffffffu;
        uint pd_lower = ld16(pbase + 12);
        uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);
#endif

        for (uint lp = tid; lp < nlp; lp += SCAN_WG)
        {
#ifdef ARROW_SRC
            uint fidx = flags_base + page * AR_VP + lp;
#else
            uint fidx = flags_base + lp_base + lp;
#endif
            if (mode == 4)
            {
                gt_text_row(fidx - flags_base, naggs, aggoff);
                continue;
            }
            if (mode != 0 && outb[fidx] != (mode == 3 ? 1u : 4u))
                continue;                           // only the rows that found no slot (mode 3: the ones the CPU takes)

            Reg  regs[VM_MAX_REGS];
            bool hit = false;
#if defined(ARROW_SRC) && defined(SPEC_PROG)
            hit = spec_arrow_eval_row(row0 + lp, page, lp, false, regs);
#elif defined(ARROW_SRC)
            hit = arrow_eval_row(row0 + lp, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
#elif defined(SPEC_PROG)
            hit = spec_eval_row(pbase, page, lp, false, regs);
#else
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);
#endif
#ifdef SPEC_PROG
            if (hit && !spec_scales_ok(regs))
#else
            if (hit && !agg_scales_ok(aggoff, naggs, keyoff, nkeys, regs))
#endif
            {
                hit = false;                        // a numeric of another scale than the plan's:
                g_recheck = 1;                      // the backend evaluates the row
            }
            if (!hit)
            {
                if (mode == 0)
                    outb[fidx] = (g_recheck != 0) ? 2u : 0u;
                continue;
            }
#ifdef SPEC_PROG
            spec_stage(regs, tid);
#else
            {
                int64_t kk[GA_NK];
                uint kn = 0;
                for (uint i = 0; i < GA_NK; i++)
                {
                    kk[i] = 0;
                    if (i < nkeys)
                    {
                        Reg r = regs[prog[keyoff + i]];
                        if (r.isnull != 0) kn |= (1u << i); else kk[i] = r.v;
                    }
                    st_key[tid * GA_NK + i] = kk[i];
                }
                st_hash[tid] = hash_keys(kk, kn, g_kstr);
                st_kn[tid] = kn | 0x80000000u;
                for (uint k = 0; k < naggs; k++)
                {
                    uint cls = prog[aggoff + 4 * k];
                    uint ra  = prog[aggoff + 4 * k + 1];
                    uint rf  = prog[aggoff + 4 * k + 2];
                    bool ok  = true;
                    if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                        ok = false;                         // FILTER (WHERE ...) is not true
                    if (ok && cls != AC_COUNT_ANY && regs[ra].isnull != 0)
                        ok = false;
                    st_ok[tid * GA_NA + k] = ok ? 1u : 0u;
                    st_val[tid * GA_NA + k] = (cls == AC_COUNT_ANY || cls == AC_COUNT_COND) ? 0 : regs[ra].v;
                }
            }
#endif
            if (mode == 0)
            {
                uint prev;
                InterlockedAdd(sh_hits, 1, prev);
            }
#ifdef ARROW_SRC
            if (mode == 3)
            {
                // this row goes to the CPU: its staged key and values to the overflow list
                uint idx;
                InterlockedAdd(outb[kparam.extra], 1, idx);
                uint rb = kparam.extra + 1 + idx * OV_REC;
                outb[rb + 0] = st_kn[tid] & 15u;
                for (uint i = 0; i < GA_NK; i++)
                {
                    uint64_t kv = uint64_t(st_key[tid * GA_NK + i]);
                    outb[rb + 1 + 2 * i] = uint(kv & 0xffffffffu);
                    outb[rb + 2 + 2 * i] = uint(kv >> 32);
                }
                for (uint k = 0; k < naggs; k++)
                {
                    uint64_t vv = uint64_t(st_val[tid * GA_NA + k]);
                    outb[rb + 1 + 2 * GA_NK + 3 * k + 0] = st_ok[tid * GA_NA + k];
                    outb[rb + 1 + 2 * GA_NK + 3 * k + 1] = uint(vv & 0xffffffffu);
                    outb[rb + 1 + 2 * GA_NK + 3 * k + 2] = uint(vv >> 32);
                }
                continue;
            }
#endif
            // min/max arguments must fit 32 bits
            bool fits = true;
            for (uint k = 0; k < naggs; k++)
            {
                int64_t v = st_val[tid * GA_NA + k];
                if ((a_cls[k] == AC_MIN || a_cls[k] == AC_MAX) && st_ok[tid * GA_NA + k] != 0 &&
                    (v < -2147483648 || v > 2147483647))
                    fits = false;
            }
            if (!fits)
            {
                outb[fidx] = 1u;
                continue;
            }
            uint kn = st_kn[tid] & 15u;
            uint tag = st_hash[tid];
            if (tag == 0)
                tag = 1;
            if (mode == 1)
            {
                gt_insert(tag, kn, tid, mask, stride);
                continue;
            }
            uint base = gt_find(tag, kn, tid, mask, stride, nkeys);
            if (base == NO_SLOT)
            {
                outb[fidx] = (mode == 0) ? 4u : 1u;
                continue;
            }
            gt_accumulate(base, tid, naggs, aggoff, fidx - flags_base);
            outb[fidx] = 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0 && mode == 0 && sh_hits != 0)
    {
        uint prev;
        InterlockedAdd(gtab[1], sh_hits, prev);
    }
}
