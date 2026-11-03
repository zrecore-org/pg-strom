#!/bin/sh
# PostgreSQL CPU baseline: the E1-style predicate over the heap table, with the
# table fully in shared buffers, at several parallel-worker counts.
# usage: ./pg_baseline.sh [runs]
RUNS=${1:-6}
Q="SELECT count(*) FROM t WHERE b + a::int8 > 1073741824 AND f > 500::float4 AND d < 25 AND a > 1000"
cd "$(dirname "$0")"
./pgq.sh -c "SELECT pg_prewarm('t', 'buffer')" >/dev/null
for w in 0 1 2 4 6 8; do
  {
    printf '%s\n' "SET max_parallel_workers_per_gather = $w;"
    printf '%s\n' "SET parallel_setup_cost = 0; SET parallel_tuple_cost = 0; SET min_parallel_table_scan_size = 0;"
    printf '%s\n' '\timing on'
    i=0; while [ $i -lt $RUNS ]; do printf '%s;\n' "$Q"; i=$((i+1)); done
  } | ./pgq.sh 2>&1 | awk -v w=$w '/^Time:/ { t[++n] = $2 } END {
      best = 1e12; for (i = 2; i <= n; i++) if (t[i] < best) best = t[i];   # drop the first (warm-up) run
      printf("workers=%d  best of %d = %.1f ms\n", w, n - 1, best) }'
done
