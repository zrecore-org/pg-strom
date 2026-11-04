// The Arrow (columnar) row loader, shared by arrow.slang (GpuScan on record batches) and by the
// aggregation and join kernels built with -DARROW_SRC (agg.slang, gagg.slang, join.slang, jagg.slang,
// jgagg.slang), where an Arrow record batch is the outer relation. Include after scan_common.slang.
//
// Bindings as in scan_common.slang with a different meaning:
//   pagef  the Arrow data: a copy of the columns' buffers, packed by the service (arrow_pack_gpu)
//   ptab   the column table, 8 words per column of the program's column list:
//            [0] validity bitmap byte offset (0xffffffff: no bitmap, no NULLs)  [1] its length in bytes
//            [2] values byte offset   [3] width of a value in bytes (0: bool, a bit per value);
//                string columns: the width of an offset (4, 8) or, for a fixed-size string, of the value
//            [4] mode: 0 normal, 1 all NULL, 2 constant (a virtual column), 3 string with offsets,
//                4 fixed-size string, 5 string as { offset, length } pairs (GPU-Cache: the datum, header included,
//                at byte offset [5] + offset, at most [6] bytes of data in all; the width [3] is 8)
//            [5] normal: the conversion to the PostgreSQL representation (ACV_*);
//                mode 2: the constant's low word, 3: byte offset of the string bytes
//            [6] mode 2: the constant's high word, 3: length of the string bytes
//            [7] length of the values in bytes (a row beyond it reads as NULL)
//          after the table, at word 8 * ncols: [0] rows in this pass  [1] first row of the pass
//            [2] nonzero: row filter (GPU-Cache): pagel holds one bit per row, a row whose bit is 0 does not exist
//   pagel  the row filter if ptab[8 * ncols + 2] is nonzero, else unused (a placeholder)
// A string register is { byte offset in pagef | length << 32 } (the source bits are 0).
//
// Conversions of a normal column (the same arithmetic as arrow_fetch() in arrow_exec.c, which the
// backend's pg_datum_arrow_ref() defines, wrap-around included):
//   1 Date[day] 32 bit: days since 1970 -> days since 2000
//   2 Time[s] (32 bit) -> us, 3 Time[ms] (32 bit) -> us, 4 Time[us] as is, 5 Time[ns] -> us
//   6 Timestamp[s], 7 [ms], 8 [us], 9 [ns] (64 bit): -> us since 2000
static const uint ACV_NONE   = 0;
static const uint ACV_DATE   = 1;
static const uint ACV_TIME_S = 2;
static const uint ACV_TIME_MS = 3;
static const uint ACV_TIME_US = 4;
static const uint ACV_TIME_NS = 5;
static const uint ACV_TS_S   = 6;
static const uint ACV_TS_MS  = 7;
static const uint ACV_TS_US  = 8;
static const uint ACV_TS_NS  = 9;

#define ARROW_TS_DRIFT 946684800000000ul    // 1970-01-01 -> 2000-01-01 in microseconds

uint ar_ld8(uint o)  { return (pagef[o >> 2] >> ((o & 3u) * 8u)) & 0xffu; }
uint ar_ld16(uint o) { return ar_ld8(o) | (ar_ld8(o + 1) << 8); }
uint ar_ld32(uint o)
{
    uint sh = (o & 3u) * 8u;
    if (sh == 0)
        return pagef[o >> 2];
    return (pagef[o >> 2] >> sh) | (pagef[(o >> 2) + 1] << (32u - sh));
}
uint64_t ar_ld64(uint o) { return uint64_t(ar_ld32(o)) | (uint64_t(ar_ld32(o + 4)) << 32); }

// Load the columns of the program's column list for `row` of the record batch into the registers
// their descriptors name (the caller has set every register to NULL). Columns that are NULL stay so.
void arrow_load_row(uint row, uint coloff, uint ncols, inout Reg regs[VM_MAX_REGS])
{
    for (uint c = 0; c < ncols; c++)
    {
        uint desc = prog[coloff + c];
        uint slot = (desc >> 16) & 0xffu;
        if (slot == 0xffu)
            continue;
        uint vt = (desc >> 12) & 0xfu;
        uint t  = c * 8;
        uint mode = ptab[t + 4];

        if (mode == 1)
            continue;                               // NULL in every row
        if (mode == 2)
        {
            regs[slot].v = int64_t(uint64_t(ptab[t + 5]) | (uint64_t(ptab[t + 6]) << 32));
            regs[slot].isnull = 0;
            continue;
        }
        uint noff = ptab[t + 0];
        if (noff != 0xffffffffu)
        {
            uint nbyte = row >> 3;
            if (nbyte >= ptab[t + 1] || ((ar_ld8(noff + nbyte) >> (row & 7u)) & 1u) == 0)
                continue;                           // NULL
        }
        if (vt == SOP_VT_NULLONLY)
        {
            regs[slot].v = 0;
            regs[slot].isnull = 0;
            continue;
        }
        uint voff = ptab[t + 2];
        uint w    = ptab[t + 3];
        uint vlen = ptab[t + 7];
        if (mode == 3)
        {
            // utf8 / binary: offsets (w bytes each) into the string bytes
            uint64_t o1, o2;
            if ((uint64_t(row) + 2) * w > uint64_t(vlen))
                continue;                           // short offsets buffer: NULL, as the CPU path
            if (w == 4)
            {
                o1 = uint64_t(ar_ld32(voff + row * 4));
                o2 = uint64_t(ar_ld32(voff + row * 4 + 4));
            }
            else
            {
                o1 = ar_ld64(voff + row * 8);
                o2 = ar_ld64(voff + row * 8 + 8);
            }
            if (o1 > o2 || o2 > uint64_t(ptab[t + 6]) || o2 - o1 > 0x0fffffffull)
                continue;                           // corrupted: NULL, as the CPU path
            regs[slot].v = int64_t(uint64_t(ptab[t + 5] + uint(o1)) | ((o2 - o1) << 32));
            regs[slot].isnull = 0;
            continue;
        }
        if (mode == 5)
        {
            // GPU-Cache string: { offset | length << 32 } per row, the complete datum behind it
            if ((uint64_t(row) + 1) * 8 > uint64_t(vlen))
                continue;
            uint64_t pr = ar_ld64(voff + row * 8);
            uint soff = uint(pr & 0xffffffffu);
            uint slen = uint(pr >> 32);
            if (slen == 0 || uint64_t(soff) + uint64_t(slen) > uint64_t(ptab[t + 6]))
                continue;                           // corrupted: NULL
            uint sbase = ptab[t + 5] + soff;
            uint hdr = ((ar_ld8(sbase) & 1u) != 0u) ? 1u : 4u;     // a 1-byte (short) or a 4-byte header
            if (slen < hdr)
                continue;
            regs[slot].v = int64_t(uint64_t(sbase + hdr) | (uint64_t(slen - hdr) << 32));
            regs[slot].isnull = 0;
            continue;
        }
        if (mode == 4)
        {
            // fixed-size string (char(n)): w bytes at row * w
            if ((uint64_t(row) + 1) * w > uint64_t(vlen))
                continue;
            regs[slot].v = int64_t(uint64_t(voff + row * w) | (uint64_t(w) << 32));
            regs[slot].isnull = 0;
            continue;
        }
        int64_t v = 0;
        if (w == 0 ? ((row >> 3) >= vlen) : ((uint64_t(row) + 1) * w > uint64_t(vlen)))
            continue;                               // short buffer: NULL
        uint conv = ptab[t + 5];
        if (vt == SOP_VT_BOOL)
            v = int64_t((ar_ld8(voff + (row >> 3)) >> (row & 7u)) & 1u);
        else if (conv != ACV_NONE)
        {
            if (conv <= ACV_TIME_MS)
            {
                uint u = ar_ld32(voff + row * 4);
                if (conv == ACV_DATE)
                    v = int64_t(int(u - 10957u));
                else
                    v = int64_t(uint64_t(u) * (conv == ACV_TIME_S ? 1000000ul : 1000ul));
            }
            else
            {
                uint64_t u = ar_ld64(voff + row * 8);
                if (conv == ACV_TIME_US)
                    v = int64_t(u);
                else if (conv == ACV_TIME_NS)
                    v = int64_t(u / 1000ul);
                else
                {
                    if (conv == ACV_TS_S)
                        u = u * 1000000ul;
                    else if (conv == ACV_TS_MS)
                        u = u * 1000ul;
                    else if (conv == ACV_TS_NS)
                        u = u / 1000ul;
                    v = int64_t(u - ARROW_TS_DRIFT);
                }
            }
        }
        else if (w == 1)
        {
            uint b = ar_ld8(voff + row);
            v = int64_t(int(b ^ 0x80u) - 128);
        }
        else if (w == 2)
        {
            uint h = ar_ld16(voff + row * 2);
            v = int64_t(int(h ^ 0x8000u) - 32768);
        }
        else if (w == 4)
        {
            uint u = ar_ld32(voff + row * 4);
            v = (vt == SOP_VT_FLOAT4) ? int64_t(u) : int64_t(int(u));
        }
        else
            v = int64_t(ar_ld64(voff + row * 8));
        regs[slot].v = v;
        regs[slot].isnull = 0;
    }
}

// Evaluate `row`: load the columns, run the qual part of the program and, for a qualifying row, the
// rest of it (the projection, the aggregate arguments). Returns true if the row qualifies. A failure
// is recorded with report_error(epage, elp, err) and counts as "not qualifying".
bool arrow_eval_row(uint row, uint epage, uint elp, uint coloff, uint ncols, uint insnoff, uint nq, uint nins,
                    uint qreg, inout Reg regs[VM_MAX_REGS])
{
    if (ptab[8 * ncols + 2] != 0u && ((pagel[row >> 5] >> (row & 31u)) & 1u) == 0u)
        return false;                               // GPU-Cache: no such row (free slot, or not visible)
    g_recheck = 0;
    g_src = 0;
    [unroll]
    for (uint i = 0; i < VM_MAX_REGS; i++)
    {
        regs[i].v = 0;
        regs[i].isnull = 1;
        regs[i].sc = 0;
    }
    arrow_load_row(row, coloff, ncols, regs);

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
        report_error(epage, elp, err);
        return false;
    }
    return hit;
}

// ---------------------------------------------------------------------------------------------
// The aggregation and join kernels (ARROW_SRC) walk a record batch in "virtual pages" of AR_VP rows,
// so that their page loops stay as they are: page p of a pass holds rows [p * AR_VP, (p + 1) * AR_VP)
// of the pass, which starts at row ptab[8 * ncols + 1] of the batch and has ptab[8 * ncols] rows.
// ---------------------------------------------------------------------------------------------
#define AR_VP 256u

// the number of rows of virtual page `page` ("line pointers")
uint arrow_page_rows(uint page, uint ncols)
{
    uint total = ptab[8 * ncols];
    uint r0 = page * AR_VP;
    return (r0 >= total) ? 0u : min(AR_VP, total - r0);
}

// the batch row of the first row of virtual page `page`
uint arrow_page_row0(uint page, uint ncols)
{
    return ptab[8 * ncols + 1] + page * AR_VP;
}
