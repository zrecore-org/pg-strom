// Plumbing for the row-format kernels (the VM, the hand-written kernel and the
// generated specialized kernels): the input rows and the unaligned byte loaders.
#include "common_out.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint8_t> rows;

uint ld32(uint o)
{
    return uint(rows[o]) | (uint(rows[o + 1]) << 8) |
           (uint(rows[o + 2]) << 16) | (uint(rows[o + 3]) << 24);
}

int64_t ld64(uint o)
{
    return int64_t(uint64_t(ld32(o)) | (uint64_t(ld32(o + 4)) << 32));
}
