# Aggregation over a join (GpuPreAgg on top of GpuJoin): the join kernel aggregates the joined rows on the
# device (jagg.slang without GROUP BY, jgagg.slang with it). Sourced by scan_test.sh (and by joinagg_dev.sh)
# after the join sections: it needs their helpers (q, GPU, CPU, JOINSET, check) and tables (s1 s2 s3, sd with
# duplicate keys and NULL pay, sd2, sd3, sdn).
#
# The plan of such a query is a Custom Scan (GpuPreAgg) that shows the join ("GPU Join Quals [n]") under a
# CPU Aggregate; runjagg requires both, runjagg_cpu only compares the results (shapes that stay on the CPU).
echo "aggregation over a join (GpuPreAgg on GpuJoin)"
# Without enable_seqscan=off: its 1e10 penalty on the dimension table's scan is inside every path's cost and
# within PostgreSQL's 1% fuzz, so a sorted CPU GroupAggregate would "dominate" the hashed GpuPreAgg path.
JAGGSET_SAVE="$JOINSET"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
JAGGPAT='GPU Join Quals \[1\]'
JAGGBIG="set pg_strom.gpupreagg_max_groups=1000000;"
PAR="set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
runjagg() { # table, label, query-with-%T and %D [, settings]; GpuPreAgg over the join, results equal to the CPU's
	[ -n "$JAGG_FILTER" ] && ! echo "$2" | grep -Eq "$JAGG_FILTER" && return 0
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g"); set_extra=${4:-}
	pl=$($PSQL -d scantest -c "$GPU $JOINSET $JAGGBIG $set_extra explain $qry")
	plan=$(echo "$pl" | grep -c "Custom Scan (GpuPreAgg)"); jn=$(echo "$pl" | grep -c "$JAGGPAT")
	g=$(q "$GPU $JOINSET $JAGGBIG $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$jn" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	elif [ "$ALLOW_CPU" = on ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then
		# a CPU Vulkan device runs the test on a smaller table, where the planner's estimates pick other plans: the
		# result still must match, the plan shape is checked on the real GPU
		echo "  ok    $tbl: $label (${g#*|} rows; plan GpuPreAgg=$plan join=$jn differs on this host)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan join=$jn)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runjagg_cpu() { # the aggregate stays on the CPU (above a GpuJoin or not); the results must be equal
	[ -n "$JAGG_FILTER" ] && ! echo "$2" | grep -Eq "$JAGG_FILTER" && return 0
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g"); set_extra=${4:-}
	pl=$($PSQL -d scantest -c "$GPU $JOINSET $JAGGBIG $set_extra explain $qry")
	plan=$(echo "$pl" | grep -c "Custom Scan (GpuPreAgg)")
	g=$(q "$GPU $JOINSET $JAGGBIG $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -eq 0 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows, CPU aggregate)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

for t in s1 s2 s3; do
	JAGGPAT='GPU Join Quals \[1\]'
	runjagg $t "count, sum, max over both sides" "select count(*), sum(t.a), max(d.grp) from %T t join %D d on t.b = d.k"
	runjagg $t "NULL values from the inner side, duplicate keys" "select count(*), count(d.pay), sum(d.pay), min(d.pay), max(d.pay), avg(d.pay) from %T t join %D d on t.b = d.k"
	runjagg $t "all aggregates of an int2/int4/int8 argument from either side" "select sum(t.a), min(t.a), max(t.a), avg(t.a), sum(t.c), min(t.c), max(t.c), avg(t.b), sum(d.k), min(d.k8), max(d.k8), avg(d.grp) from %T t join %D d on t.b = d.k"
	runjagg $t "empty join result" "select count(*), count(d.pay), sum(d.pay), min(t.a), max(d.grp), avg(t.b) from %T t join %D d on t.b = d.k + 1000000"
	runjagg $t "quals on both sides" "select count(*), sum(t.a), max(d.pay) from %T t join %D d on t.b = d.k where t.a > 0 and d.grp < 8 and d.pay is not null"
	runjagg $t "expressions mixing both sides" "select sum(t.a + d.grp), max(t.b * 2 - d.k), min(d.pay + t.c), count(t.a * d.grp) from %T t join %D d on t.b = d.k"
	runjagg $t "FILTER (WHERE ...) on either side" "select count(*) filter (where d.grp < 8), sum(t.b) filter (where t.a < 0), max(d.pay) filter (where t.e), count(d.pay) filter (where d.grp > 100) from %T t join %D d on t.b = d.k"
	runjagg $t "sum beyond 64 bits" "select sum(t.c + 9000000000000000000), avg(t.c + 9000000000000000000), sum(d.pay + 9000000000000000000) from %T t join %D d on t.b = d.k"
	runjagg $t "int8 join key with a computed key expression" "select count(*), sum(d.grp), max(t.a) from %T t join %D d on t.c % 20000 = d.k8"
	runjagg $t "extra join qual" "select count(*), sum(d.grp) from %T t join %D d on t.b = d.k and t.a > d.grp"
	runjagg $t "9 aggregates (the 12-aggregate kernel)" "select count(*), count(d.pay), sum(t.a), sum(d.grp), min(t.b), max(d.k), avg(t.c), sum(d.pay), max(t.a) from %T t join %D d on t.b = d.k"
	JAGGPAT='Left Outer Join Quals \[1\]'
	runjagg $t "LEFT JOIN: count(*) against count(inner column)" "select count(*), count(d.grp), sum(d.grp), max(d.pay), sum(t.a) from %T t left join %D d on t.b = d.k + 20000"
	runjagg $t "LEFT JOIN with no match at all" "select count(*), count(d.grp), sum(d.grp), min(d.k), max(t.i) from %T t left join %D d on t.b = d.k + 1000000"
	runjagg $t "LEFT JOIN with an ON qual on the inner side, FILTER on the null-extended column" "select count(*), count(d.grp), sum(t.a) filter (where d.grp is null), sum(d.grp) from %T t left join %D d on t.b = d.k and d.grp < 3"
done
for t in s1 s2 s3; do
	JAGGPAT='Join Quals \[2\]'
	runjagg $t "two inner relations" "select count(*), sum(t.a + d.grp + e.g2), max(e.p2), min(d.pay) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2"
	runjagg $t "two inner relations, duplicate keys at both depths, FILTER on the second" "select count(*), sum(e.p2) filter (where e.g2 > 10), max(d.grp) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 where t.a > -200"
	JAGGPAT='Join Quals \[3\]'
	runjagg $t "three inner relations" "select count(*), sum(f.g3), max(e.g2), sum(d.grp), min(t.a) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 join sd3 f on t.a + 20000 = f.k3"
	JAGGPAT='Join Quals \[2\]'
	runjagg $t "inner join, then LEFT join" "select count(*), count(e.g2), sum(e.p2), sum(d.grp) from %T t join %D d on t.b = d.k left join sd2 e on t.a % 40 = e.k2 + 400"
	runjagg $t "LEFT join, then inner join" "select count(*), count(d.grp), sum(e.g2), max(d.pay) from %T t left join %D d on t.b = d.k + 20000 join sd2 e on t.a % 400 = e.k2 where t.i < 80000"
	runjagg $t "two LEFT joins, the first without matches" "select count(*), count(d.grp), count(e.g2), sum(e.g2) from %T t left join %D d on t.b = d.k + 1000000 left join sd2 e on t.a % 400 = e.k2 where t.i < 50000"
done
JAGGPAT='GPU Join Quals \[1\]'

# semi and anti joins: the aggregate sees the outer row once
JAGGPAT='GPU Semi Join Quals \[1\]'
for t in s1 s2 s3; do
	JAGGPAT='GPU Semi Join Quals \[1\]'
	runjagg $t "EXISTS" "select count(*), sum(t.a), max(t.b) from %T t where exists (select 1 from %D d where d.k = t.b)"
	runjagg $t "EXISTS with a correlated qual, duplicate inner keys" "select count(*), min(t.c) from %T t where exists (select 1 from %D d where d.k = t.b and d.grp > 99 and t.a < d.grp)"
	JAGGPAT='GPU Anti Join Quals \[1\]'
	runjagg $t "NOT EXISTS" "select count(*), sum(t.a), max(t.b), count(t.b) from %T t where not exists (select 1 from %D d where d.k = t.b)"
done
JAGGPAT='GPU Join Quals \[1\]'

# GROUP BY over a join: up to two integer keys, from either side
echo "GROUP BY over a join"
for t in s1 s2 s3; do
	JAGGPAT='GPU Join Quals \[1\]'
	runjagg $t "key from the inner side" "select d.grp, count(*), sum(t.a), max(t.b), min(d.pay), avg(t.c) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "key from the outer side" "select t.a % 10 as k, count(*), sum(d.grp), max(d.pay) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "NULL keys from the inner side" "select d.pay % 3 as k, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "FILTER and quals" "select d.grp % 4 as k, count(*) filter (where t.b > 0), sum(t.b) filter (where d.pay is null), max(t.c) from %T t join %D d on t.b = d.k where t.a > -9000 group by 1 order by 1"
	runjagg $t "sum beyond 64 bits per group" "select d.grp % 3 as k, sum(t.c + 9000000000000000000), avg(t.c + 9000000000000000000) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "int8 key" "select d.k8 % 7 as k, count(*), sum(t.a) from %T t join %D d on t.c % 20000 = d.k8 group by 1 order by 1"
	runjagg $t "min and max beyond 32 bits (overflow list)" "select d.grp as k, min(t.c + 5000000000), max(t.c + 5000000000), count(*), sum(t.c + 5000000000) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "empty join result" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k + 1000000 group by 1 order by 1"
	JAGGPAT='Left Outer Join Quals \[1\]'
	runjagg $t "LEFT JOIN: the unmatched rows form the NULL group" "select d.grp, count(*), count(d.pay), sum(t.a) from %T t left join %D d on t.b = d.k + 20000 group by 1 order by 1"
	runjagg $t "LEFT JOIN, key from the outer side, count of the null-extended column" "select t.a % 5 as k, count(*), count(d.grp), sum(d.grp) from %T t left join %D d on t.b = d.k and d.grp < 3 group by 1 order by 1"
done
for t in s1 s2 s3; do
	JAGGPAT='Join Quals \[2\]'
	runjagg $t "stacked joins, key from the second inner relation" "select e.g2, count(*), sum(t.a), max(d.grp) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 group by 1 order by 1"
	runjagg $t "stacked joins, keys from two inner relations" "select d.grp % 4 as k1, e.g2 as k2, count(*), sum(e.p2) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 group by 1, 2 order by 1, 2"
	runjagg $t "inner then LEFT join, key from the LEFT side" "select e.g2, count(*), count(e.g2), sum(d.grp) from %T t join %D d on t.b = d.k left join sd2 e on t.a % 40 = e.k2 + 400 group by 1 order by 1"
done
# keys with known statistics (the planner estimates an expression of a column at one group per row and would
# keep the aggregate on the CPU): jt has plain key columns of 7, 97 and 1500 distinct values, some dead tuples
sql "create table jt as select i, a, b, c, (i % 7)::int4 as g7, (i % 97)::int4 as g97, (i % 1500)::int4 as g1500 from s1"
sql "vacuum analyze jt"; sql "update jt set c = c + 1 where i % 10 = 0"; sql "analyze jt"
JAGGPAT='GPU Join Quals \[1\]'
runjagg jt "two keys, one from each side" "select t.g7, d.grp, count(*), sum(t.c), min(d.k) from %T t join %D d on t.b = d.k group by 1, 2 order by 1, 2"
runjagg jt "more groups than a workgroup's table (overflow list)" "select t.g1500, count(*), sum(d.grp), min(t.b), max(d.pay) from %T t join %D d on t.b = d.k group by 1 order by 1"
runjagg jt "overflow with two keys" "select t.g97, d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1, 2 order by 1, 2"
runjagg jt "overflow, outer key, min/max beyond 32 bits" "select t.g1500, min(t.c + 5000000000), max(t.c + 5000000000), count(*) from %T t join %D d on t.b = d.k group by 1 order by 1"
JAGGPAT='GPU Semi Join Quals \[1\]'
runjagg s1 "semi join, key from the outer side" "select t.a % 6 as k, count(*), sum(t.b) from %T t where exists (select 1 from %D d where d.k = t.b) group by 1 order by 1"
JAGGPAT='GPU Anti Join Quals \[1\]'
runjagg s1 "anti join, key from the outer side" "select t.a % 6 as k, count(*), sum(t.c) from %T t where not exists (select 1 from %D d where d.k = t.b) group by 1 order by 1"
JAGGPAT='GPU Join Quals \[1\]'

# parallel workers: every worker aggregates its part of the outer rows in its own session
runjagg s1 "parallel workers, no GROUP BY" "select count(*), sum(t.a), max(d.grp), min(d.pay) from %T t join %D d on t.b = d.k" "$PAR"
runjagg s1 "parallel workers, GROUP BY" "select d.grp, count(*), sum(t.a), max(t.c) from %T t join %D d on t.b = d.k group by 1 order by 1" "$PAR"
JAGGPAT='Left Outer Join Quals \[1\]'
runjagg s1 "parallel workers, LEFT JOIN with a NULL group" "select d.grp, count(*), count(d.pay) from %T t left join %D d on t.b = d.k + 20000 group by 1 order by 1" "$PAR"

# shapes that stay on the CPU aggregate (the results must still be right)
# RIGHT and FULL joins: the kernels flag the inner rows that found a partner; after the final reply the backend
# joins each unmatched inner row with NULL outer columns and projects it to a partial-aggregate row (the Agg node
# above adds it to the GPU's states). Each test below needs both halves: matched rows aggregated on the device and
# unmatched ones from the CPU.
echo "aggregation over RIGHT and FULL joins"
sql "create table if not exists sdnl as select case when grp % 4 = 0 then null else k end as k, grp, pay, txt from sd"; sql "analyze sdnl"
for t in s1 s2 s3; do
	JAGGPAT='Right Outer Join Quals \[1\]'
	runjagg $t "RIGHT JOIN, ON qual on both sides" "select count(*), count(t.i), sum(d.grp), sum(t.a), max(d.pay), min(t.b) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000"
	runjagg $t "RIGHT JOIN, every inner row has a partner or not (duplicate inner keys)" "select count(*), count(t.i), sum(d.grp), avg(t.a), max(d.k8) from %T t right join %D d on t.b = d.k"
	runjagg $t "RIGHT JOIN, ON qual on the inner side" "select count(*), count(t.i), sum(d.grp), min(d.pay) from %T t right join %D d on t.b = d.k and d.grp < 3"
	runjagg $t "RIGHT JOIN, nothing matches (the aggregates see the inner rows only)" "select count(*), count(t.i), sum(t.a), sum(d.grp), min(d.pay), max(d.k) from %T t right join %D d on t.b = d.k + 1000000"
	runjagg $t "RIGHT JOIN, empty outer side (every inner row is unmatched)" "select count(*), count(t.i), sum(d.grp), max(d.pay) from %T t right join %D d on t.b = d.k and t.i + 0 < -1"
	runjagg $t "RIGHT JOIN, inner rows with NULL keys" "select count(*), count(t.i), sum(d.grp), max(t.a) from %T t right join sdnl d on t.b = d.k and t.a + d.grp > 9000"
	runjagg $t "RIGHT JOIN, expression mixing both sides, FILTER" "select sum(t.a + d.grp), count(*) filter (where t.i is null), max(d.pay) filter (where t.e), sum(d.pay) filter (where t.i is null) from %T t right join %D d on t.b = d.k and t.a > d.grp"
	JAGGPAT='Full Outer Join Quals \[1\]'
	runjagg $t "FULL JOIN, unmatched rows on both sides" "select count(*), count(t.i), count(d.k), sum(t.a), sum(d.grp), max(d.pay) from %T t full join %D d on t.b = d.k and t.a > 9000"
	runjagg $t "FULL JOIN, nothing matches" "select count(*), count(t.i), count(d.k), sum(d.grp), sum(t.a) from %T t full join %D d on t.b = d.k + 1000000"
	runjagg $t "FULL JOIN, inner rows with NULL keys" "select count(*), count(t.i), count(d.k), sum(d.grp) from %T t full join sdnl d on t.b = d.k and t.a > 9000"
done
for t in s1 s2 s3; do
	JAGGPAT='Right Outer Join Quals \[1\]'
	runjagg $t "RIGHT JOIN, key from the inner side" "select d.grp, count(*), count(t.i), sum(t.a), max(d.pay) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 group by 1 order by 1"
	runjagg $t "RIGHT JOIN, nothing matches, GROUP BY" "select d.grp, count(*), count(t.i), sum(d.pay) from %T t right join %D d on t.b = d.k + 1000000 group by 1 order by 1"
	JAGGPAT='Full Outer Join Quals \[1\]'
	runjagg $t "FULL JOIN, key from the inner side, FILTER" "select d.grp, count(*) filter (where t.i is null), count(t.i), sum(t.a) from %T t full join %D d on t.b = d.k and t.i % 3 = 0 group by 1 order by 1"
done
# keys of the outer side (the planner needs statistics of a plain column: jt, a copy of s1 with such columns) and the NULL group
# of the unmatched inner rows
JAGGPAT='Right Outer Join Quals \[1\]'
RJTMP="create table rjt as select i, a, b, c, (i % 7)::int4 as g7, (i % 97)::int4 as g97, (i % 1500)::int4 as g1500 from s1"
sql "$RJTMP"; sql "vacuum analyze rjt"
runjagg rjt "RIGHT JOIN, key from the outer side (unmatched inner rows form the NULL group)" "select t.g7, count(*), sum(d.grp), min(d.pay), count(t.i) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 group by 1 order by 1"
runjagg rjt "RIGHT JOIN, two keys, one from each side (NULL keys of the inner side's pay too)" "select t.g7, d.grp, count(*), count(t.i), sum(d.pay) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 group by 1, 2 order by 1, 2"
runjagg rjt "RIGHT JOIN, nothing matches, key from the outer side" "select t.g7, count(*), count(t.i), sum(d.grp) from %T t right join %D d on t.b = d.k + 1000000 group by 1 order by 1"
runjagg rjt "RIGHT JOIN, more groups than a workgroup's table (overflow list)" "select t.g1500, count(*), count(t.i), sum(d.grp) from %T t right join %D d on t.b = d.k and t.a + d.grp > -5000 group by 1 order by 1"
JAGGPAT='Full Outer Join Quals \[1\]'
runjagg rjt "FULL JOIN, key from the outer side" "select t.g7, count(*), count(d.k), sum(d.grp), sum(t.b) from %T t full join %D d on t.b = d.k and t.a > 9000 group by 1 order by 1"
# an inner side with many NULL keys: they hash alike, so an outer row with a NULL key walks one long chain (about 7 s of
# kernel time on the Apple GPU for this query; the join aggregation is dispatched in slices so that no dispatch hits the watchdog)
JAGGPAT='Full Outer Join Quals \[1\]'
runjagg s2 "FULL JOIN, NULL-key inner rows and long chain walks" "select count(*), count(t.i), count(d.k), sum(d.grp) from %T t full join sdnl d on t.b = d.k and t.i % 3 = 0"
# negative values of the inner side in min/max/sum (the backend loads them from the inner store as Datums)
JAGGPAT='Right Outer Join Quals \[1\]'
runjagg s1 "RIGHT JOIN, negative int4 and int8 values of the unmatched inner rows" "select min(d.k), max(d.k), sum(d.k), min(d.pay), max(d.pay), min(d.grp - 100), count(t.i) from %T t right join %D d on t.b = d.k + 1000000"
# stacked: the unmatched inner rows of depth 1 continue through the later depths with NULL outer columns
for t in s1 s2 s3; do
	JAGGPAT='Join Quals \[2\]'
	runjagg $t "inner join, then RIGHT join" "select count(*), count(t.i), sum(e.g2), sum(d.grp) from %T t join sd2 e on t.a % 400 = e.k2 right join %D d on t.b = d.k and t.a + d.grp > 9000"
	runjagg $t "inner join, then RIGHT join, GROUP BY" "select d.grp % 5 as k, count(*), count(t.i), sum(e.g2) from %T t join sd2 e on t.a % 400 = e.k2 right join %D d on t.b = d.k and t.a + d.grp > 9000 group by 1 order by 1"
done
JAGGPAT='Right Outer Join Quals \[1\]'
runjagg s1 "RIGHT join, then LEFT join on the inner side's column" "select count(*), count(t.i), count(e.g2), sum(d.grp), sum(e.p2) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 left join sd2 e on d.grp = e.k2"
runjagg s1 "RIGHT join, then inner join on the inner side's column, GROUP BY" "select e.g2, count(*), count(t.i), sum(d.grp) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 join sd2 e on d.grp = e.k2 group by 1 order by 1"
# a parallel plan does not offer RIGHT/FULL joins: the aggregate must still be right
runjagg s1 "RIGHT JOIN, parallel workers wanted (not offered for the join: a serial GpuPreAgg or the CPU aggregate, same result)" "select count(*), count(t.i), sum(d.grp) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000" "$PAR"
# A14: the aggregating kernels used to drop an outer row whose scan quals the device cannot decide (float4 results that
# are denormal or may underflow, numerics beyond 64 bits, ...) without a word, and to stop with a kernel error when the
# projection could not be computed. Now a qual-only recheck is flagged in a bitmap behind the results (no extra pass) and
# a program whose walk or projection can recheck runs a scout pass first (scan_prog.h rck_outer / rck_walk, join_walk.inc);
# the backend joins and aggregates those rows on the CPU. The table has denormal float4 values, numerics that overflow
# 64 bits and compressed / out-of-line strings (the strings take part through NULL tests and a text qual below the join).
echo "aggregation over a join: rows the device cannot decide (recheck)"
sql "create table rkt as select s1.*, (i % 13)::numeric as n, (i % 7)::int4 as g7, (i % 2)::int4 as g2,
	case when i % 101 = 0 then (select string_agg(md5(i::text || j::text), '') from generate_series(1, 100) j)
	     when i % 53 = 0 then repeat(md5(i::text), 100) else f end as f2
	from s1 where i % 3 = 0"
sql "update rkt set d = 1e-40::float4 where i % 997 = 5"
sql "vacuum analyze rkt"
runjagg_res() { # only the results must equal the CPU's (the planner may keep the aggregate or the join on the CPU)
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g")
	g=$(q "$GPU $JOINSET $JAGGBIG" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
RKF="t.d < 'Infinity'::float4 and t.d > '-Infinity'::float4"
JAGGPAT='GPU Join Quals \[1\]'
runjagg rkt "recheck: float4 qual, denormal operands, ungrouped" "select count(*), sum(d.grp), sum(t.a), max(t.i) from %T t join %D d on t.b = d.k where t.d * 2::float4 > 0::float4"
runjagg rkt "recheck: float4 qual, GROUP BY an inner key" "select d.grp, count(*), sum(d.k), min(t.i) from %T t join %D d on t.b = d.k where t.d * 2::float4 > 0::float4 group by 1 order by 1"
runjagg rkt "recheck: float4 qual, key from the outer side" "select t.g7, count(*), sum(d.grp) from %T t join %D d on t.b = d.k where t.d * 2::float4 > 0::float4 group by 1 order by 1"
runjagg rkt "recheck: float4 in the projection (scout pass), ungrouped" "select count(*), sum((t.d * 2::float4)::int), max(d.grp) from %T t join %D d on t.b = d.k where $RKF"
runjagg rkt "recheck: float4 in the projection (scout pass), GROUP BY" "select d.grp, count(*), sum((t.d * 2::float4)::int) from %T t join %D d on t.b = d.k where $RKF group by 1 order by 1"
runjagg rkt "recheck: float4 in a FILTER, two keys" "select d.grp, t.g2, count(*) filter (where t.d * 2::float4 > 1::float4), sum(d.k) from %T t join %D d on t.b = d.k where $RKF group by 1, 2 order by 1, 2"
runjagg rkt "recheck: text qual below the join, compressed and out-of-line strings" "select count(*), sum(d.grp), sum(t.a) from %T t join %D d on t.b = d.k where length(t.f2) > 100"
runjagg rkt "recheck: NULL test of the toasted string, float4 qual" "select count(*), sum(d.grp) from %T t join %D d on t.b = d.k where t.f2 is not null and t.d * 2::float4 > 0::float4"
JAGGPAT='Left Outer Join Quals \[1\]'
runjagg rkt "recheck: LEFT JOIN, float4 ON qual" "select count(*), count(d.k), sum(d.grp), sum(t.a) from %T t left join %D d on t.b = d.k and t.d * 2::float4 > 0::float4"
runjagg rkt "recheck: LEFT JOIN, GROUP BY" "select t.a % 5, count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and t.d * 2::float4 > 0::float4 group by 1 order by 1"
runjagg_res rkt "recheck: LEFT JOIN, numeric division by zero in the ON qual" "select count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and t.i % 50 <> 0 and t.b::numeric / (t.i % 50)::numeric > 3"
runjagg_res rkt "recheck: LEFT JOIN, numeric beyond 64 bits in the ON qual, GROUP BY" "select t.a % 4, count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and (case when t.i % 50 = 0 then t.c::numeric * 100000000000000000000::numeric else t.a::numeric end) > 5 group by 1 order by 1"
JAGGPAT='Right Outer Join Quals \[1\]'
runjagg rkt "recheck: RIGHT JOIN" "select count(*), count(t.i), sum(d.grp), sum(t.a) from %T t right join %D d on t.b = d.k and t.d * 2::float4 > 0::float4"
runjagg rkt "recheck: RIGHT JOIN, GROUP BY" "select d.grp % 5, count(*), count(t.i), sum(t.a) from %T t right join %D d on t.b = d.k and t.d * 2::float4 > 0::float4 group by 1 order by 1"
JAGGPAT='Full Outer Join Quals \[1\]'
runjagg rkt "recheck: FULL JOIN" "select count(*), count(t.i), count(d.k), sum(d.grp) from %T t full join %D d on t.b = d.k and t.d * 2::float4 > 0::float4"
runjagg rkt "recheck: FULL JOIN, GROUP BY" "select d.grp % 4, count(*), count(t.i), count(d.k) from %T t full join %D d on t.b = d.k and t.d * 2::float4 > 0::float4 group by 1 order by 1"
JAGGPAT='GPU Join Quals \[1\]'
runjagg s1 "float aggregate over a join (sum(float4) adds in float4 on the GPU too)" "select count(*), sum(t.d), max(d.grp) from %T t join %D d on t.b = d.k where t.a > 9000"
runjagg_cpu s1 "text key" "select d.txt, count(*) from %T t join %D d on t.b = d.k group by 1 order by 1 limit 20"
runjagg_cpu s1 "too many estimated groups" "select t.i, count(*) from %T t join %D d on t.b = d.k where t.i < 50000 group by 1 order by 1 limit 50" "set pg_strom.gpupreagg_max_groups=16;"
JOINSET="$JAGGSET_SAVE"
