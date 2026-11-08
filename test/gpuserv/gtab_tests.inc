# GROUP BY with many groups on the GPU (A3): the session-wide hash table in device memory (gtab.slang) against the CPU.
# Sourced by gtab_dev.sh and scan_test.sh; needs q, GPU, CPU, runagg, runagg_any, check, sql, PSQL and fail.
echo "GROUP BY with many groups (session-wide hash table)"
NG=${NG:-300000}
sql "drop table if exists gm" >/dev/null
# gm: keys of 1 to 100,000 groups (uniform, skewed, NULLs, 64-bit, narrow ranges), values of every width
sql "create table gm as select i::int8 as i,
	((i * 37) % 3000)::int4 as k3k, ((i * 7919) % 20000)::int4 as k20k, ((i * 104729) % 100000)::int4 as k100k,
	(case when i % 13 = 0 then null else (i * 31) % 500 end)::int4 as kn,
	floor(50000 * power(((i * 2654435761) % 4294967296) / 4294967296.0, 4))::int4 as kskew,
	(((i * 17) % 2000) * 5000000000 - 4000000000000)::int8 as kw,
	((i * 7) % 61)::int2 as k2, ('2024-01-01'::date + ((i * 11) % 700)::int4) as kd,
	((i * 2654435761) % 4294967296 - 2147483648)::int4 as a,
	(case when i % 11 = 0 then null else ((i * 48271) % 2000001 - 1000000) end)::int4 as b,
	((i * 6700417) % 4611686018427387904 - 2305843009213693952 + (i % 3) * 2305843009213693952)::int8 as c,
	(i % 5 = 0) as e
	from generate_series(1::int8, $NG) i" >/dev/null
sql "vacuum analyze gm" >/dev/null
for t in gm; do
	runagg $t "3,000 groups" "select k3k, count(*), count(b), sum(a), min(b), max(b), avg(b) from %T group by 1 order by 1"
	runagg $t "20,000 groups, int8 sums (carry across 32 bits)" "select k20k, count(*), sum(a), sum(b), sum(c) from %T group by 1 order by 1"
	runagg $t "100,000 groups" "select k100k, count(*), sum(a), max(a), min(a) from %T group by 1 order by 1"
	runagg $t "NULL keys" "select kn, count(*), sum(a), min(b), max(b) from %T group by 1 order by 1"
	runagg $t "skewed keys (a few very hot groups)" "select kskew, count(*), sum(a), sum(b), max(b) from %T group by 1 order by 1"
	runagg $t "int8 keys beyond 32 bits" "select kw, count(*), sum(a), min(a) from %T group by 1 order by 1"
	runagg $t "min and max of int8 beyond 32 bits (rows on the CPU)" "select k3k, min(c), max(c), count(*) from %T group by 1 order by 1"
	runagg $t "two keys" "select k2, k3k % 400 as kk, count(*), sum(a), max(b) from %T group by 1, 2 order by 1, 2"
	runagg $t "two keys, one with NULLs" "select kn, k2, count(*), sum(b) from %T group by 1, 2 order by 1, 2"
	runagg $t "date key" "select kd, count(*), sum(a) from %T group by 1 order by 1"
	runagg $t "quals, FILTER and an expression key" "select (k20k + 5) % 7000 as k, count(*) filter (where e), sum(b) filter (where a > 0), max(a) from %T where b is not null and a > -2000000000 group by 1 order by 1"
	runagg $t "all values NULL in a group, count of NULLs" "select kn % 60 as k, count(*), count(b), sum(b), min(b), max(b), sum(c) from %T where i % 7 <> 3 and (b is null or i % 100 < 2) group by 1 order by 1"
	runagg $t "sum beyond 64 bits per group" "select k3k % 100 as k, sum((c % 200000000000000000) + 9000000000000000000), avg((c % 200000000000000000) + 9000000000000000000) from %T group by 1 order by 1"
	runagg_any $t "empty input" "select k20k, count(*), sum(a) from %T where i < 0 group by 1 order by 1"
	runagg_any $t "a group per row" "select i, count(*), sum(a) from %T where i <= 20000 group by 1 order by 1" "set max_parallel_workers_per_gather=0;"
	runagg $t "parallel workers" "select k20k, count(*), sum(a), max(b) from %T group by 1 order by 1" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
done
# several tasks (the table is kept and grown between them): 2.5M rows are more than one 64 MB task
sql "drop table if exists gmm" >/dev/null
sql "create table gmm as select ((i * 7919) % 150000)::int4 as k, ((i * 31) % 1000)::int4 as v, (i % 97)::int4 as w, (case when i % 17 = 0 then null else i % 1000003 end)::int4 as n from generate_series(1::int8, 2500000) i" >/dev/null
sql "vacuum analyze gmm" >/dev/null
runagg gmm "150,000 groups over several tasks (the table grows)" "select k, count(*), sum(v), min(w), max(n), sum(n), count(n) from %T group by 1 order by 1"
runagg gmm "an expression key the planner cannot estimate (a small table first)" "select (k * 3) % 200000 as g, count(*), sum(v), sum(w) from %T group by 1 order by 1"
