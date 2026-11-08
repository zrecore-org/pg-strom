// hash_bytes.inc: PostgreSQL's hash_bytes() (hash_any, common/hashfn.c, Bob Jenkins' lookup3) of `len` bytes
// of a string register's source: the hash of a text / bpchar join key, bit-identical to the one the backend
// puts in the inner hash items (devtype_text_hash). Included by scan_common.slang after ldw_s() / ldb_s().
// The byte-wise tail of the original is kept: the lowest byte of c is reserved for the length.

uint hb_rot(uint x, uint k) { return (x << k) | (x >> (32u - k)); }

uint hash_bytes_src(uint src, uint off, uint len)
{
    uint a = 0x9e3779b9u + len + 3923095u;
    uint b = a;
    uint c = a;
    uint i = 0;

    for (; len - i >= 12u; i += 12u)
    {
        a += ldw_s(src, off + i);
        b += ldw_s(src, off + i + 4u);
        c += ldw_s(src, off + i + 8u);
        a -= c; a ^= hb_rot(c, 4u);  c += b;
        b -= a; b ^= hb_rot(a, 6u);  a += c;
        c -= b; c ^= hb_rot(b, 8u);  b += a;
        a -= c; a ^= hb_rot(c, 16u); c += b;
        b -= a; b ^= hb_rot(a, 19u); a += c;
        c -= b; c ^= hb_rot(b, 4u);  b += a;
    }
    uint rem = len - i;
    uint o = off + i;
    if (rem >= 9u)
    {
        c += ldb_s(src, o + 8u) << 8;
        if (rem >= 10u)
            c += ldb_s(src, o + 9u) << 16;
        if (rem >= 11u)
            c += ldb_s(src, o + 10u) << 24;
    }
    if (rem >= 8u)
    {
        b += ldw_s(src, o + 4u);
        a += ldw_s(src, o);
    }
    else if (rem >= 4u)
    {
        if (rem >= 5u)
        {
            b += ldb_s(src, o + 4u);
            if (rem >= 6u)
                b += ldb_s(src, o + 5u) << 8;
            if (rem >= 7u)
                b += ldb_s(src, o + 6u) << 16;
        }
        a += ldw_s(src, o);
    }
    else if (rem >= 1u)
    {
        if (rem >= 3u)
            a += ldb_s(src, o + 2u) << 16;
        if (rem >= 2u)
            a += ldb_s(src, o + 1u) << 8;
        a += ldb_s(src, o);
    }
    c ^= b; c -= hb_rot(b, 14u);
    a ^= c; a -= hb_rot(c, 11u);
    b ^= a; b -= hb_rot(a, 25u);
    c ^= b; c -= hb_rot(b, 16u);
    a ^= c; a -= hb_rot(c, 4u);
    b ^= a; b -= hb_rot(a, 14u);
    c ^= b; c -= hb_rot(b, 24u);
    return c;
}
