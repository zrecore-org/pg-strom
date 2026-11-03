// Scan of LIVE PostgreSQL buffers selected by a page list, with torn-read detection.
//
// Table u(a int4, b int8, f float4, d int4), no NULLs, so every tuple is exactly
//   t_hoff = 24, t_len = 48, data: a@0 b@8 f@16 d@20 (relative to t_hoff).
// Writers always keep  b = 3*a + d + 7  in every tuple version they write, so a
// tuple whose bytes violate it was read half-written. Other torn-read evidence:
// line pointers outside the page or misaligned, unexpected tuple headers,
// inconsistent page headers.
//
// The kernel takes NO locks and does NO visibility checks: dead and superseded
// versions are counted like live ones (reported in 'stale').
#include "heap_common.slang"

[[vk::binding(2, 0)]] StructuredBuffer<uint> plist;      // pairs: word offset of the page, relation block number

// stats layout in outb
static const uint S_TUPLES = 0;     // tuples examined (LP_NORMAL, well-formed)
static const uint S_MATCH  = 1;     // tuples passing  a > 5e8 AND f > 500 AND d < 25
static const uint S_IDSUM  = 2;     // sum of blkno*512+offnum over matches, mod 2^32
static const uint S_VIOL   = 3;     // invariant violations
static const uint S_BADLP  = 4;     // line pointers out of range / misaligned / too short
static const uint S_BADHDR = 5;     // unexpected t_hoff / HASNULL / t_len
static const uint S_BADPG  = 6;     // inconsistent page header
static const uint S_PAGES  = 7;     // pages examined
static const uint S_STALE  = 8;     // tuples without HEAP_XMAX_INVALID (superseded, deleted or locked)
static const uint S_NSAMP  = 9;     // number of anomalies seen (samples kept: first 8)
static const uint S_SAMP   = 16;
static const uint SAMPLE_WORDS = 8;
static const uint MAX_SAMPLES  = 8;

void sample(uint kind, uint blk, uint lpidx, uint lpword, uint a, uint blo, uint bhi, uint d)
{
    uint idx;
    InterlockedAdd(outb[S_NSAMP], 1, idx);
    if (idx < MAX_SAMPLES)
    {
        uint o = S_SAMP + idx * SAMPLE_WORDS;
        outb[o + 0] = kind;     // 1 invariant, 2 line pointer, 3 tuple header, 4 page header
        outb[o + 1] = blk;
        outb[o + 2] = lpidx;
        outb[o + 3] = lpword;
        outb[o + 4] = a;
        outb[o + 5] = blo;
        outb[o + 6] = bhi;
        outb[o + 7] = d;
    }
}

[shader("compute")]
[numthreads(HEAP_WG, 1, 1)]
void heap_live(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint tuples = 0, match = 0, idsum = 0, viol = 0, badlp = 0, badhdr = 0;
    uint badpg = 0, pgs = 0, stale = 0;

    for (uint e = gid.x; e < kparam.nrows; e += kparam.stride)
    {
        uint pagew = plist[2 * e];
        uint blk   = plist[2 * e + 1];
        uint w3    = pages[pagew + 3];
        uint w4    = pages[pagew + 4];
        uint lower = w3 & 0xFFFFu;
        uint upper = w3 >> 16;
        uint ver   = w4 >> 16;
        bool pageok = (ver == 0x2004u) && lower >= 24u && upper <= 8192u && lower <= upper;
        uint nlp = pageok ? ((lower - 24u) >> 2) : 0u;

        if (lid.x == 0)
        {
            pgs++;
            if (!pageok)
            {
                badpg++;
                sample(4, blk, 0, w3, w4, 0, 0, 0);
            }
        }
        uint rounds = (nlp + HEAP_WG - 1u) / HEAP_WG;       // uniform per workgroup
        for (uint r = 0; r < rounds; r++)
        {
            uint i = r * HEAP_WG + lid.x;
            if (i >= nlp)
                continue;
            uint lp    = pages[pagew + 6 + i];
            uint flags = (lp >> 15) & 3u;
            if (flags != LP_NORMAL)
                continue;
            uint off = lp & 0x7FFFu;
            uint len = lp >> 17;
            if (off < 24u || off + len > 8192u || len < 24u || (off & 7u) != 0u)
            {
                badlp++;
                sample(2, blk, i, lp, 0, 0, 0, 0);
                continue;
            }
            uint tw      = pagew + (off >> 2);
            uint w5      = pages[tw + 5];
            uint hoff    = (w5 >> 16) & 0xFFu;
            uint infomask = w5 & 0xFFFFu;
            if (hoff != 24u || (infomask & HEAP_HASNULL) != 0 || len != 48u)
            {
                badhdr++;
                sample(3, blk, i, lp, w5, 0, 0, 0);
                continue;
            }
            uint dw  = tw + 6;
            int  a   = int(pages[dw + 0]);
            uint blo = pages[dw + 2];
            uint bhi = pages[dw + 3];
            float f  = asfloat(pages[dw + 4]);
            int  d   = int(pages[dw + 5]);
            int64_t b = int64_t(uint64_t(blo) | (uint64_t(bhi) << 32));

            tuples++;
            if ((infomask & 0x0800u) == 0)      // HEAP_XMAX_INVALID not set
                stale++;
            if (b != int64_t(a) * 3 + int64_t(d) + 7)
            {
                viol++;
                sample(1, blk, i, lp, uint(a), blo, bhi, uint(d));
            }
            else if (a > 500000000 && f > 500.0f && d < 25)
            {
                match++;
                idsum += blk * 512u + i + 1u;
            }
        }
    }
    // one set of atomics per wave; every lane is here (uniform control flow)
    uint s_tuples = WaveActiveSum(tuples), s_match = WaveActiveSum(match), s_idsum = WaveActiveSum(idsum);
    uint s_viol = WaveActiveSum(viol), s_badlp = WaveActiveSum(badlp), s_badhdr = WaveActiveSum(badhdr);
    uint s_badpg = WaveActiveSum(badpg), s_pgs = WaveActiveSum(pgs), s_stale = WaveActiveSum(stale);
    if (WaveIsFirstLane())
    {
        InterlockedAdd(outb[S_TUPLES], s_tuples);
        InterlockedAdd(outb[S_MATCH], s_match);
        InterlockedAdd(outb[S_IDSUM], s_idsum);
        InterlockedAdd(outb[S_VIOL], s_viol);
        InterlockedAdd(outb[S_BADLP], s_badlp);
        InterlockedAdd(outb[S_BADHDR], s_badhdr);
        InterlockedAdd(outb[S_BADPG], s_badpg);
        InterlockedAdd(outb[S_PAGES], s_pgs);
        InterlockedAdd(outb[S_STALE], s_stale);
    }
}
