// Hand-written heap-page kernel for the E1 predicate, the lower bound for the
// real-page scan:  (b + (int8)a) > 2^30 AND f > 500.0 AND d < 25 AND a > 1000
#include "heap_common.slang"

struct FixedEval : ITupEval
{
    bool test(Tup t, out uint err)
    {
        err = 0;
        return ((t.nulls & 0xFu) == 0) && (t.b + int64_t(t.a) > 1073741824) &&
               (t.f > 500.0f) && (t.d < 25) && (t.a > 1000);
    }
}

[shader("compute")]
[numthreads(HEAP_WG, 1, 1)]
void heap_fixed(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    FixedEval ev;
    heap_scan<FixedEval>(ev, gid, lid);
}
