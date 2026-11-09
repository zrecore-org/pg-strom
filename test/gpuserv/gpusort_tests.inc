# GPU-Sort (A7): GROUP BY ... ORDER BY / LIMIT / window-rank through the service's sorter. Sourced by scan_test.sh
# (and gpusort_dev.sh) after the tables s1, s2 and sd exist. Every query runs on the GPU and with
# pg_strom.enabled = off; the rows must be identical *in order* (the result is hashed in output order).
#
# Settings: GPU-Sort is off by default (pg_strom.enable_gpusort), the planner only builds the path when
# the estimated group count is below pg_strom.gpupreagg_max_groups, and the CPU has an equally good Sort,
# so the GPU variants turn the first two on and enable_sort off (a plan with a Sort node is then not
# competitive, so the GPU-Sort path is picked whenever it exists). The service sorts on the device here
# whatever the number of groups (the scan_test.sh cluster sets PGSTROM_GPUSERV_SORT=gpu), and a device
# failure would show in the log as "GPU-Sort fell back".
echo "gpu-sort"
GS="$GPU set pg_strom.enable_gpusort=on; set pg_strom.cpu_fallback=off; set pg_strom.gpupreagg_max_groups=100000000; set enable_sort=off;"
GSCPU="$CPU"
qo() { # settings, query -> md5 of the rows in output order, count
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(t::text, E'\n' order by ord), '')), count(*) from (select row_number() over () as ord, y.* from ($2) y) t" 2>&1 | tail -1
}
runsort() { # table, label, query-with-%T [, extra settings]; the plan must have GPU-Sort keys
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GS $set_extra explain $qry" | grep -c "GPU-Sort keys")
	g=$(qo "$GS $set_extra" "$qry"); c=$(qo "$GSCPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GPU-Sort=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runsort_cpu() { # like runsort, but the sort must stay on the CPU (no GPU-Sort keys); the rows must still match
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GS $set_extra explain $qry" | grep -c "GPU-Sort keys")
	g=$(qo "$GS $set_extra" "$qry"); c=$(qo "$GSCPU" "$qry")
	if [ "$plan" -eq 0 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (stays on the CPU, ${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GPU-Sort=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runsort_set() { # table, label, query-with-%T: only the set of rows is compared (ties may order differently), plan has GPU-Sort
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GS $set_extra explain $qry" | grep -c "GPU-Sort keys")
	g=$(q "$GS $set_extra" "$qry"); c=$(q "$GSCPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GPU-Sort=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

for T in s1 s2; do
	# one key, ascending: the group key itself (20000 groups)
	runsort $T "group by int2, order by the key" "select a, count(*), sum(b), max(c) from %T group by a order by a"
	runsort $T "group by int2, order by the key DESC" "select a, count(*), min(b) from %T group by a order by a desc"
	# the aggregate as the key (the key column breaks the ties so that the order is total)
	runsort $T "order by sum(int4) DESC, key" "select a, sum(b) from %T group by a order by 2 desc, 1"
	runsort $T "order by sum(int4) ASC NULLS FIRST, key" "select a % 50 as k, sum(b) from %T group by 1 order by 2 nulls first, 1 desc"
	runsort $T "order by sum(int4) DESC NULLS LAST, key" "select a % 50 as k, sum(b) from %T group by 1 order by 2 desc nulls last, 1"
	runsort $T "order by sum(int8)" "select a, sum(c), count(c) from %T group by a order by 2, 1"
	runsort $T "order by count(*), key" "select a, count(*) from %T group by a order by 2 desc, 1 desc"
	runsort $T "order by max, min" "select a % 97 as k, max(b), min(c) from %T group by 1 order by 2 desc nulls last, 3 nulls first, 1"
	# two group keys (bool and an integer expression): NULL keys, ASC/DESC per key
	runsort $T "two keys, mixed directions" "select a % 3 as e2, a % 7 as m, count(*), sum(b) from %T group by 1, 2 order by e2 desc, m"
	runsort $T "two keys, by an aggregate between them" "select a % 3 as e2, a % 5 as m, sum(b) from %T group by 1, 2 order by 1, 3 desc nulls last, 2"
	# LIMIT / OFFSET pushed into the sort
	runsort $T "LIMIT" "select a, count(*), sum(b) from %T group by a order by 3 desc nulls last, 1 limit 10"
	runsort $T "LIMIT with OFFSET" "select a, sum(b) from %T group by a order by a limit 7 offset 5"
	runsort $T "LIMIT 1" "select a % 1000 as k, count(*) from %T group by 1 order by 2 desc, 1 limit 1"
	# many groups and a key with a NULL group (b has NULLs)
	runsort $T "NULL group key last/first" "select b, count(*) from %T group by b order by b nulls first limit 100"
	runsort $T "NULL group key (default)" "select b, count(*) from %T group by b order by b desc limit 100"
	runsort $T "date-like key arithmetic" "select (c % 1000) as k, sum(a) from %T group by 1 order by 2 desc, 1"
done

# ties: only the sort key column is compared (the order of equal keys is not defined)
for T in s1; do
	g=$(qo "$GS" "select cnt from (select count(*) as cnt from $T group by a % 13 order by cnt) z")
	c=$(qo "$GSCPU" "select cnt from (select count(*) as cnt from $T group by a % 13 order by cnt) z")
	[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "$T: tie-heavy sort key, key column only" $?
	g=$(qo "$GS" "select s from (select sum(b) as s from $T group by a % 4 order by s desc nulls last) z")
	c=$(qo "$GSCPU" "select s from (select sum(b) as s from $T group by a % 4 order by s desc nulls last) z")
	[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "$T: sum key with few groups" $?
done

# GpuPreAgg over GpuJoin
JS="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
runsort s1 "over a join: order by sum DESC" "select d.grp, count(*), sum(t.b) from %T t join sd d on t.b = d.k group by d.grp order by 3 desc nulls last, 1" "$JS"
runsort s1 "over a join: order by the key, LIMIT" "select d.grp, sum(t.a) from %T t join sd d on t.b = d.k group by d.grp order by 1 limit 5" "$JS"
runsort s2 "over a join: two keys" "select d.grp, t.a % 2 as e2, count(*) from %T t join sd d on t.b = d.k group by d.grp, 2 order by 3, 1, 2" "$JS"

# window-rank filter: rank() / dense_rank() / row_number() below N per partition
runsort_set s1 "rank() <= 3 per partition" "select * from (select a / 2000 as p, a, sum(b) as sb, rank() over (partition by a / 2000 order by sum(b)) r from %T group by a / 2000, a) z where r <= 3"
runsort_set s1 "dense_rank() < 3 per partition, DESC" "select * from (select a / 5000 as p, a % 700 as k, count(*) as n, dense_rank() over (partition by a / 5000 order by count(*) desc) r from %T group by a / 5000, a % 700) z where r < 3"
runsort s1 "row_number() <= 4 per partition (total order)" "select * from (select a / 1000 as p, a, sum(c) as sc, row_number() over (partition by a / 1000 order by sum(c), a) r from %T group by a / 1000, a) z where r <= 4 order by p, r"
# a window on top of GROUP BY the planner cannot feed from the sorter stays on the CPU
runsort_cpu s1 "avg() key: the sorter has no average" "select a % 40 as k, avg(b) from %T group by 1 order by 2 desc nulls last, 1"
runsort s1 "bool group key (booleq is a device function: the groups are sorted on the GPU)" "select e, count(*) from %T group by e order by e desc"
runsort_cpu s1 "float4 NaN key" "select d, count(*) from %T group by d order by d"
runsort_cpu s1 "sort key is not in the result of the GROUP BY (expression)" "select a % 40 as k, count(*) from %T group by 1 order by (a % 40) * 2 desc, 1"

# disabled: enable_gpusort off gives the CPU plan
n=$($PSQL -d scantest -c "$GS set pg_strom.enable_gpusort=off; explain select a, sum(b) from s1 group by a order by 2 desc, 1" | grep -c "GPU-Sort keys")
[ "$n" -eq 0 ]; check "enable_gpusort=off: no GPU-Sort in the plan" $?
# cpu_fallback on: the sorter cannot hand rows to the CPU, so the planner does not offer it
n=$($PSQL -d scantest -c "$GS set pg_strom.cpu_fallback=on; explain select a, sum(b) from s1 group by a order by 2 desc, 1" | grep -c "GPU-Sort keys")
[ "$n" -eq 0 ]; check "cpu_fallback=on: no GPU-Sort in the plan" $?
# parallel settings: the non-parallel GPU-Sort path may still be chosen (a Gather never sits above it); the rows must match
PARSET2="set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
g=$(qo "$GS $PARSET2" "select a, sum(b) from s1 group by a order by 2 desc, 1")
c=$(qo "$GSCPU" "select a, sum(b) from s1 group by a order by 2 desc, 1")
[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "parallel settings: sorted GROUP BY result matches" $?

# every group once: the sum of the counts is the number of rows (the GROUP BY result is complete)
g=$($PSQL -d scantest -c "$GS select sum(n), count(*) from (select a, count(*) as n from s1 group by a order by a) z" | tail -1)
c=$($PSQL -d scantest -c "$CPU select count(*), count(distinct a) from s1" | tail -1)
[ "${g%%|*}" = "${c%%|*}" ] && [ "${g##*|}" = "${c##*|}" ]; check "GPU-Sort returns every group once (counts add up)" $?

# the service must not have fallen back from the device (it would be silent otherwise)
! grep -q "GPU-Sort fell back" "$ROOT/log" 2>/dev/null; check "GPU-Sort ran on the device (no fallback message in the service log)" $?

# parallel GpuPreAgg with several workers: every worker's partial states must reach PostgreSQL (each worker
# has its own session of the service; before the fix only the last worker's rows were returned). A task is
# 64 MB of pages, so the table needs a few of them.
sql "create table pw as select i, (i % 7)::int4 as g, (i % 1000)::int8 as v from generate_series(1, 5000000) i"
sql "vacuum analyze pw"
PARSET="set max_parallel_workers_per_gather=3; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0; set parallel_leader_participation=off;"
g=$(q "$GPU $PARSET" "select g, count(*), sum(v), max(i) from pw group by g")
c=$(q "$CPU" "select g, count(*), sum(v), max(i) from pw group by g")
[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "parallel GpuPreAgg over a large table: all workers' groups are combined" $?
g=$(q "$GPU $PARSET" "select count(*), sum(v), min(i), max(i) from pw")
c=$(q "$CPU" "select count(*), sum(v), min(i), max(i) from pw")
[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "parallel GpuPreAgg without GROUP BY over a large table" $?
sql "drop table pw"
