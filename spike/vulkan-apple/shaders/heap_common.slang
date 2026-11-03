// PostgreSQL heap-page scan plumbing (KDS_FORMAT_BLOCK-style): the kernels read
// raw 8 KB heap pages as copied from the relation file, parse the page header
// and line pointers, and deform tuples of  t(a int4, b int8, f float4, d int4).
//
// Layouts (src/include/storage/bufpage.h, itemid.h, access/htup_details.h):
//   PageHeaderData  24 bytes: pd_lsn(8) pd_checksum(2) pd_flags(2) pd_lower(2)
//                   pd_upper(2) pd_special(2) pd_pagesize_version(2) pd_prune_xid(4)
//   ItemIdData      4 bytes at offset 24 + 4*i: lp_off:15 lp_flags:2 lp_len:15
//   HeapTupleHeader t_xmin(4) t_xmax(4) t_cid(4) t_ctid(6) t_infomask2(2)
//                   t_infomask(2) t_hoff(1) t_bits[] ; data at t_hoff, columns aligned
//
// Tuple visibility is NOT checked: the table is VACUUM FREEZEd and the
// visibility map says every page is all-visible/all-frozen (checked by the host).
#include "common_out.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint> pages;     // 2048 words per 8 KB page

#ifndef HEAP_WG
#define HEAP_WG 128
#endif
static const uint PAGE_WORDS = 2048;
static const uint LP_NORMAL  = 1;
static const uint HEAP_HASNULL = 0x0001;

struct Tup
{
    int     a;
    int64_t b;
    float   f;
    int     d;
    uint    nulls;      // bit i set = column i is NULL (a=0 b=1 f=2 d=3)
};

// returns false for unused / dead / redirect line pointers
bool heap_deform(uint pagew, uint lpidx, out Tup t)
{
    t.a = 0; t.b = 0; t.f = 0.0f; t.d = 0; t.nulls = 0;

    uint lp       = pages[pagew + 6 + lpidx];
    uint lp_off   = lp & 0x7FFFu;
    uint lp_flags = (lp >> 15) & 3u;
    if (lp_flags != LP_NORMAL)
        return false;

    uint tw     = pagew + (lp_off >> 2);        // tuples are MAXALIGN (8) aligned
    uint w5     = pages[tw + 5];                // t_infomask | t_hoff << 16 | t_bits[0] << 24
    uint hoff   = (w5 >> 16) & 0xFFu;
    uint bits   = (w5 >> 24) & 0xFFu;
    uint dw     = tw + (hoff >> 2);             // t_hoff is MAXALIGN, so word aligned
    uint nn     = ((w5 & HEAP_HASNULL) != 0) ? bits : 0xFu;     // t_bits: bit set = NOT NULL

    t.nulls = (~nn) & 0xFu;
    uint off = 0;
    if ((nn & 1u) != 0)                          // a int4
    {
        t.a = int(pages[dw + (off >> 2)]);
        off += 4;
    }
    if ((nn & 2u) != 0)                          // b int8, 8-byte aligned
    {
        off = (off + 7u) & ~7u;
        uint lo = pages[dw + (off >> 2)];
        uint hi = pages[dw + (off >> 2) + 1];
        t.b = int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
        off += 8;
    }
    if ((nn & 4u) != 0)                          // f float4
    {
        off = (off + 3u) & ~3u;
        t.f = asfloat(pages[dw + (off >> 2)]);
        off += 4;
    }
    if ((nn & 8u) != 0)                          // d int4
    {
        off = (off + 3u) & ~3u;
        t.d = int(pages[dw + (off >> 2)]);
    }
    return true;
}

interface ITupEval
{
    // true if the tuple passes the qual; err != 0 reports an evaluation error
    bool test(Tup t, out uint err);
}

// One workgroup walks pages gid, gid + ngroups, ...; its threads split the line
// pointers of a page. The id reported for a match is blkno * 512 + offnum, the
// same number the SQL oracle derives from ctid.
void heap_scan<E : ITupEval>(E ev, uint3 gid, uint3 lid)
{
    for (uint page = gid.x; page < kparam.nrows; page += kparam.stride)
    {
        uint pagew    = page * PAGE_WORDS;
        uint pd_lower = pages[pagew + 3] & 0xFFFFu;
        uint nlp      = (pd_lower >= 24u) ? ((pd_lower - 24u) >> 2) : 0u;
        uint rounds   = (nlp + HEAP_WG - 1u) / HEAP_WG;     // uniform per workgroup

        for (uint r = 0; r < rounds; r++)
        {
            uint i   = r * HEAP_WG + lid.x;
            uint id  = page * 512u + i + 1u;
            bool hit = false;

            if (i < nlp)
            {
                Tup t;
                if (heap_deform(pagew, i, t))
                {
                    uint err = 0;
                    hit = ev.test(t, err);
                    if (err != 0)
                    {
                        report_error(id, err);
                        hit = false;
                    }
                }
            }
            emit_matches(id, hit);
        }
    }
}
