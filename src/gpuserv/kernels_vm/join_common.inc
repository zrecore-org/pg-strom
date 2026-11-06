// The pieces of the join kernels shared by join.slang (result records), jagg.slang (aggregation over a
// join) and jgagg.slang (GROUP BY over a join): the hash chain helpers. Include after scan_common.slang
// with WITH_INNER defined.
#define JOIN_MAXD 4
#define JDT_WORDS 19
static const uint NO_ITEM = 0xffffffffu;

// the first item of the chain a hash value selects in relation d (dt: its depth table row)
uint chain_first(uint dt, uint h)
{
    uint kds_end = prog[dt + 7];
    uint wi = prog[dt + 8] + (h % prog[dt + 9]) * 2;
    g_src = 2;
    uint64_t off = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
    return (off != 0 && off != 0xffffffffffffffffull) ? kds_end - uint(off) : NO_ITEM;
}

// the item after `item` in its chain (the chain pointer is at the start of the item)
uint chain_next(uint dt, uint item)
{
    uint kds_end = prog[dt + 7];
    uint wi = item / 4;
    g_src = 2;
    uint64_t nx = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1)) << 32);
    return (nx == 0 || nx == 0xffffffffffffffffull) ? NO_ITEM : kds_end - uint(nx);
}


// ---- A13: a GiST-index join depth (the depth table words [14] strategy, [15] y register, [16] byte offset
// of block 0 of the copied index pages in innerb, [17] number of blocks, [18] step limit).
// The inner rows are in the depth's hash store, indexed by row id; the backend rewrote the leaf entries of the
// copied index pages to name the row id in t_tid (bi_hi:bi_lo) and marked entries without an inner row LP_DEAD
// (gpu_join.c innerPreloadFixupGiSTLeaves). A walk is a cursor (block << 11 | next line pointer), so it resumes
// where the last match was found: down along the downlinks of the entries whose box overlaps the query box,
// back up through the parent (block, line pointer) the backend stored in pd_lsn of each page. Right-links are not
// followed (the tree is walked from the root, every page once).
#ifndef SPEC_PROG
static const uint GIST_CUR_SHIFT = 11u;

bool gf_isnan(uint f) { return (f & 0x7fffffffu) > 0x7f800000u; }

// The float4 one step toward -infinity / +infinity (a finite float4 or an infinity; the caller clamps)
uint gf_step_down(uint f)
{
    if ((f & 0x7fffffffu) >= 0x7f800000u)
        return f;
    if ((f & 0x7fffffffu) == 0u)
        return 0x80000001u;
    return ((f & 0x80000000u) != 0u) ? f + 1u : f - 1u;
}

uint gf_step_up(uint f)
{
    if ((f & 0x7fffffffu) >= 0x7f800000u)
        return f;
    if ((f & 0x7fffffffu) == 0u)
        return 1u;
    return ((f & 0x80000000u) != 0u) ? f - 1u : f + 1u;
}

// The query box of the point (x, y) as geom_box2d() makes it: rounded outward to float4 (clamped to +-FLT_MAX).
// With a tolerance (slice 3, ST_DWithin / ST_3DDWithin: hasr, rr) the box of the point grown by the tolerance: a superset of
// every geometry whose distance to the point is <= the tolerance, however PostGIS rounds the distance. The tolerance is
// widened by 2^-30 of itself (the slack of PostGIS' double arithmetic), the bounds are computed in binary64, rounded
// outward to float4 and moved one more float4 step. A NULL tolerance matches nothing; a negative, NaN or infinite one
// (PostGIS raises or gives its own answer: the distance function hands every candidate to the backend) opens the whole plane.
// Returns 0, or the error code of a non-finite coordinate; nxt[d] is the first cursor, or NO_ITEM for a NULL point.
uint gist_open(uint dt, uint d, Reg rx, Reg ry, Reg rr, bool hasr, inout uint gq[(JOIN_MAXD + 1) * 4], inout uint nxt[JOIN_MAXD + 1],
               inout uint gst[JOIN_MAXD + 1])
{
    gst[d] = 0u;
    if ((rx.isnull | ry.isnull) != 0u || (hasr && rr.isnull != 0u))
    {
        nxt[d] = NO_ITEM;
        return 0u;
    }
    uint64_t xb = uint64_t(rx.v);
    uint64_t yb = uint64_t(ry.v);
    if ((xb & F64_MAG) >= F64_INF || (yb & F64_MAG) >= F64_INF)
        return SOP_GIST_NONFINITE_CODE;
    nxt[d] = 1u;                                        // block 0, line pointer 1
    if (hasr)
    {
        uint64_t tb = uint64_t(rr.v);
        if ((tb & F64_MAG) >= F64_INF || (tb >> 63) != 0ull)
        {
            gq[4 * d + 0] = 0xff7fffffu;                // -FLT_MAX
            gq[4 * d + 1] = 0x7f7fffffu;                // +FLT_MAX
            gq[4 * d + 2] = 0xff7fffffu;
            gq[4 * d + 3] = 0x7f7fffffu;
            return 0u;
        }
        tb = f64_mul(tb, 0x3FF0000000400000ull);        // * (1 + 2^-30)
        gq[4 * d + 0] = gf_clamp(gf_step_down(f64_to_f32(f64_sub(xb, tb), 1u)));
        gq[4 * d + 1] = gf_clamp(gf_step_up(f64_to_f32(f64_add(xb, tb), 2u)));
        gq[4 * d + 2] = gf_clamp(gf_step_down(f64_to_f32(f64_sub(yb, tb), 1u)));
        gq[4 * d + 3] = gf_clamp(gf_step_up(f64_to_f32(f64_add(yb, tb), 2u)));
        return 0u;
    }
    gq[4 * d + 0] = gf_clamp(f64_to_f32(xb, 1u));       // xmin
    gq[4 * d + 1] = gf_clamp(f64_to_f32(xb, 2u));       // xmax
    gq[4 * d + 2] = gf_clamp(f64_to_f32(yb, 1u));       // ymin
    gq[4 * d + 3] = gf_clamp(f64_to_f32(yb, 2u));       // ymax
    return 0u;
}

// The next inner item (byte offset of its hash item in innerb) whose index entry overlaps the query box, or
// NO_ITEM when the walk is over. err: 0, or SOP_GIST_STEPS_CODE (step limit, or a damaged index).
uint gist_next(uint dt, uint d, in uint gq[(JOIN_MAXD + 1) * 4], inout uint cursor, inout uint steps, out uint err)
{
    err = 0u;
    uint page0   = prog[dt + 16];
    uint nblocks = prog[dt + 17];
    uint maxst   = prog[dt + 18];
    uint q0 = gq[4 * d + 0], q1 = gq[4 * d + 1], q2 = gq[4 * d + 2], q3 = gq[4 * d + 3];
    uint blk = cursor >> GIST_CUR_SHIFT;
    uint off = cursor & 2047u;
    g_src = 2;
    while (true)
    {
        uint pg = page0 + blk * 8192u;
        uint lower = ld16(pg + 12u);
        uint maxoff = (lower <= 24u) ? 0u : ((lower - 24u) >> 2);
        steps += 1u;
        if (steps > maxst)
        {
            err = SOP_GIST_STEPS_CODE;
            cursor = NO_ITEM;
            return NO_ITEM;
        }
        if (off > maxoff)
        {
            // this page is done: back to the entry of the parent that led here
            if (blk == 0u)
            {
                cursor = NO_ITEM;
                return NO_ITEM;
            }
            uint pb = ld32(pg);
            uint po = ld32(pg + 4u);
            if (pb >= nblocks || po >= 2047u)
            {
                err = SOP_GIST_STEPS_CODE;              // a page the walk cannot get back from
                cursor = NO_ITEM;
                return NO_ITEM;
            }
            blk = pb;
            off = po + 1u;
            continue;
        }
        uint lpv = ld32(pg + 24u + 4u * (off - 1u));
        if (((lpv >> 15) & 3u) != 1u)                   // not LP_NORMAL (dead: no inner row)
        {
            off += 1u;
            continue;
        }
        uint it = pg + (lpv & 0x7fffu);
        if ((ld16(it + 6u) & 0x8000u) != 0u)            // NULL key
        {
            off += 1u;
            continue;
        }
        uint k0 = ld32(it + 8u);                        // box2df: xmin, xmax, ymin, ymax
        uint k1 = ld32(it + 12u);
        uint k2 = ld32(it + 16u);
        uint k3 = ld32(it + 20u);
        bool nan = gf_isnan(k0) || gf_isnan(k1) || gf_isnan(k2) || gf_isnan(k3);
        bool ovl = gf_le(k0, q1) && gf_le(q0, k1) && gf_le(k2, q3) && gf_le(q2, k3);
        uint tid_hi = ld16(it);
        uint tid_lo = ld16(it + 2u);
        uint opq = pg + ld16(pg + 16u);
        if ((ld16(opq + 12u) & 1u) != 0u)
        {
            // leaf: a NaN key (an empty geometry) never matches
            off += 1u;
            if (!nan && ovl)
            {
                uint rowid = (tid_hi << 16) | tid_lo;
                uint wi = prog[dt + 8] + 2u * prog[dt + 9] + 2u * rowid;
                uint64_t o64 = uint64_t(pw(wi)) | (uint64_t(pw(wi + 1u)) << 32);
                cursor = (blk << GIST_CUR_SHIFT) | off;
                return prog[dt + 7] - (uint(o64) + 8u);
            }
        }
        else if (nan || ovl)
        {
            uint child = (tid_hi << 16) | tid_lo;
            if (child >= nblocks)
            {
                err = SOP_GIST_STEPS_CODE;
                cursor = NO_ITEM;
                return NO_ITEM;
            }
            blk = child;
            off = 1u;
        }
        else
            off += 1u;
    }
    return NO_ITEM;
}
#endif
