// GpuJoin kernel (hash joins of up to JOIN_MAXD inner relations): one workgroup per heap page, one
// thread per outer line pointer. A thread evaluates the outer scan quals and then walks the join
// tree depth first as a small state machine: at depth d it looks at the next item of the hash
// chain of relation d (the backend's KDS_FORMAT_HASH buffer, imported as innerb), compares the
// stored hash, deforms the inner tuple and runs the join quals; a match goes one depth deeper (the
// hash value of depth d+1 is computed from the columns loaded so far), at the last depth it runs
// the projection and appends a record. An exhausted chain goes back one depth. For a LEFT OUTER
// join an exhausted chain without a match continues with NULL inner columns. Records are appended
// with one atomic add per subgroup.
//
// The walk is in join_walk.inc, shared with the aggregation kernels jagg.slang (aggregation over a
// join) and jgagg.slang (GROUP BY over a join); the emit of this kernel appends a match record.
//
// Program header (words): [0] outer column table, [2] instructions, [3] outer columns, [5] end of the
// scan quals, [6] end of the program, [7] scan qual register, [8] computed values, [9] their
// registers, [14] number of inner relations, [15] depth table, [16] first projection instruction.
// The depth table has 14 words per relation: [0] hash begin, [1] hash end = join qual begin,
// [2] join qual end, [3] hash register, [4] join qual register, [5] its column table, [6] its
// columns, [7] byte offset of the end of its KDS in innerb, [8] word offset of its hash slots,
// [9] number of hash slots, [10] LEFT OUTER, [11] 1: SEMI, 2: ANTI, [12] RIGHT/FULL: first word of
// this relation's outer-join map (one word per inner row, in outb after the header; 0xffffffff:
// none), [13] its number of rows.
//
// Output (outb): [0] number of matches (may exceed the capacity: the host retries), [1] first
// error, [2],[3] page and line pointer; from word 4 the outer-join map (header word [17] words, set
// by a join qual match of a RIGHT/FULL join), then per match { page, lp, byte offset of the inner
// item in innerb for each relation (0xffffffff: none, LEFT OUTER), (isnull, lo, hi) * computed
// values }.
#define WITH_INNER 1
#define JW_MODE 0
#include "scan_common.slang"
#ifdef ARROW_SRC
#include "arrow_load.inc"       // the outer relation is an Arrow record batch (see arrow_load.inc)
#endif
#include "join_common.inc"
#ifdef SPEC_PROG
#include "spec_gen.inc"           // A6: the program baked in (specialize.c)
#endif

[shader("compute")]
[numthreads(SCAN_WG, 1, 1)]
void join_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint page = kparam.page_base + gid.x;
    if (page >= kparam.npages)
        return;                                     // uniform: the whole group leaves

    uint coloff   = prog[0];
    uint insnoff  = prog[2];
    uint ncols    = prog[3];
    uint nq       = prog[5];
    uint nins     = prog[6];
    uint qreg     = prog[7];
    uint ncomp    = prog[8];
    uint compoff  = prog[9];
    uint nd       = prog[14];
    uint dtab     = prog[15];
    uint proj_begin = prog[16];
    uint cap      = kparam.total_lps;               // capacity of the match records
    uint rec_words = 2 + nd + 3 * ncomp;
    uint ojw      = prog[17];                       // the outer-join map precedes the records

#include "join_walk.inc"
}
