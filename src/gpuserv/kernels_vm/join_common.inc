// The pieces of the join kernels shared by join.slang (result records), jagg.slang (aggregation over a
// join) and jgagg.slang (GROUP BY over a join): the hash chain helpers. Include after scan_common.slang
// with WITH_INNER defined.
#define JOIN_MAXD 4
#define JDT_WORDS 14
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

