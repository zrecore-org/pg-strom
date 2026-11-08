# Runtime CPU fallback tests (A14), sourced by scan_test.sh and fallback_dev.sh after the helpers (check, sql, q, GPU, CPU,
# fail, ROOT, PSQL) and the tables sd, a1 (Arrow) and p1 (Parquet) exist.
#
# A query whose GPU path is refused or dies must return the rows of the same query with pg_strom.enabled = off: no error, no
# duplicated and no missing rows.  The faults are injected by the service (pg_strom.gpuserv_fault, src/gpuserv/wire.c):
#   open        OpenSession is answered with an error      refuse[=N]  the N-th task: "needs more than 32 registers"
#   lost[=N]    the N-th task: "device marked lost"        drop[=N]    the connection is closed at the N-th task
#   final       XpuTaskFinal fails                         exit[=N]    the whole service exits at the N-th task
# Each query runs on tables of many small tasks (pg_strom.scan_chunk_blocks) so that a fault at task 3 finds tasks before and
# after it; the server log line "the CPU runs the rest of the node" says that the fallback happened.
echo "CPU fallback (A14)"
FBN=${FBN:-60000}
fbcount() { grep -c "the CPU runs the rest of the node" "$ROOT/log"; }
FBSET="set pg_strom.scan_chunk_blocks=16; set pg_strom.max_async_tasks=2;"
FBJOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
# an aggregation over a join: without enable_seqscan=off (its penalty would let a sorted CPU aggregate win)
FBJAGGSET="set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
FBAGGSET="set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"

sql "create table fb1 (id int4, a int2, b int4, c int8, f text, n numeric(12,2), g date, d float4, k int4)"
sql "insert into fb1 select i, ((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 40)::int) end,
	((i % 1000) / 8.0)::numeric(12,2), date '2000-01-01' + (i % 9000),
	(((i * 17) % 10000) / 7.0)::float4, ((i % 20000) - 10000)::int4 from generate_series(1, $FBN) i"
sql "vacuum analyze fb1"
# dead tuples and recently changed pages: some blocks are all-visible (read by the service itself), the others are copied
sql "update fb1 set b = b + 1 where id % 5 = 0"
sql "delete from fb1 where id % 13 = 0"
sql "analyze fb1"
sql "create table fb2 (like fb1)"
sql "insert into fb2 select * from fb1 where id % 3 = 0"
sql "analyze fb2"

# fbrun <label> <plan pattern> <kind: scan|join|agg> <faults> <query, with %T for the table> [extra settings]
# every fault must (a) return the CPU's rows (b) be logged as a fallback
fbrun() {
	label=$1; pat=$2; kind=$3; faults=$4; qry=$5; extra=${6:-}
	case $kind in join) pre="$GPU $FBJOINSET $FBSET";; jagg) pre="$GPU $FBJAGGSET $FBSET";; agg) pre="$GPU $FBAGGSET $FBSET";; *) pre="$GPU set enable_seqscan=off; $FBSET";; esac
	pre="$pre $extra"
	plan=$($PSQL -d scantest -c "$pre explain $qry" 2>&1 | grep -c "$pat")
	c=$(q "$CPU" "$qry"); g0=$(q "$pre" "$qry")
	if [ "$plan" -lt 1 ] || [ "$g0" != "$c" ]; then
		echo "  FAIL  $label: no fault (plan $pat=$plan)"; echo "        gpu: $g0"; echo "        cpu: $c"; fail=$((fail+1)); return
	fi
	for f in $faults; do
		b=$(fbcount)
		g=$(q "$pre set pg_strom.gpuserv_fault='$f';" "$qry"); a=$(fbcount)
		if [ "$g" = "$c" ] && [ "$a" -gt "$b" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label: $f (${g#*|} rows)"
		else echo "  FAIL  $label: $f (fallbacks $b -> $a)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
	done
}
FAULTS="refuse refuse=3 lost=2 drop drop=3 open final"
FAULTS_AGG="refuse refuse=3 lost=2 drop drop=4 open final"

echo "-- GpuScan on a heap table"
fbrun "scan: qual and projection" "Custom Scan (GpuScan)" scan "$FAULTS" "select id, a, f, n from fb1 where b > 100 and a < 5000"
fbrun "scan: text and numeric" "Custom Scan (GpuScan)" scan "refuse=3 drop=2" "select id, c, g, d from fb1 where f like 'ab%' or c < 0"
fbrun "scan: no row qualifies" "Custom Scan (GpuScan)" scan "refuse" "select id from fb1 where a > 30000"
fbrun "scan: every row qualifies" "Custom Scan (GpuScan)" scan "lost=2 drop=3" "select id, a from fb1 where id > 0 or id <= 0"
fbrun "scan: computed projection" "Custom Scan (GpuScan)" scan "refuse=2" "select id, a + b, c * 2 from fb1 where b is not null"

# the visibility rules of the CPU run: the transaction's own changes are seen, deleted rows are not
for f in refuse=2 drop=3; do
	qry="select id, a, b from fb2 where b > 100"
	out=$($PSQL -d scantest 2>&1 <<EOF
begin;
delete from fb2 where id % 4 = 0;
update fb2 set b = b + 7 where id % 5 = 0;
insert into fb2 select id + 1000000, a, b + 3, c, f, n, g, d, k from fb2 where id % 11 = 0;
$GPU set enable_seqscan=off; $FBSET set pg_strom.gpuserv_fault='$f';
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
$CPU
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
rollback;
EOF
)
	g=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 1p); c=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 2p)
	if [ -n "$g" ] && [ "$g" = "$c" ] && ! echo "$out" | grep -q ERROR; then echo "  ok    scan inside a transaction with its own changes: $f (${g#*|} rows)"
	else echo "  FAIL  scan inside a transaction with its own changes: $f"; echo "$out" | head; fail=$((fail+1)); fi
done

echo "-- GpuJoin on a heap table"
fbrun "join: inner" "Custom Scan (GpuJoin)" join "$FAULTS" "select fb1.id, sd.pay, sd.txt from fb1 join sd on fb1.k = sd.k where fb1.b > 0"
fbrun "join: left outer" "Custom Scan (GpuJoin)" join "refuse=3 drop=2" "select fb1.id, sd.k, sd.pay from fb1 left join sd on fb1.k = sd.k and sd.grp < 5"
fbrun "join: right outer (the unmatched inner rows)" "Custom Scan (GpuJoin)" join "refuse refuse=3 drop=2 final" "select fb1.id, sd.k, sd.pay from fb1 right join sd on fb1.k = sd.k and fb1.b > 40000"
fbrun "join: full outer" "Custom Scan (GpuJoin)" join "refuse=3 lost=2 final" "select fb1.id, sd.k from fb1 full join sd on fb1.k = sd.k and fb1.b > 40000"
fbrun "join: semi" "Custom Scan (GpuJoin)" join "refuse=2" "select id, a from fb1 t where exists (select 1 from sd d where d.k = t.k and d.grp = 3)"
fbrun "join: anti" "Custom Scan (GpuJoin)" join "drop=3" "select id, a from fb1 t where not exists (select 1 from sd d where d.k = t.k and d.grp < 16)"
fbrun "join: two inner relations" "Custom Scan (GpuJoin)" join "refuse=3 open" "select fb1.id, sd.pay, fb2.b from fb1 join sd on fb1.k = sd.k join fb2 on fb1.id = fb2.id"

echo "-- GpuPreAgg on a heap table"
fbrun "agg: without GROUP BY" "Custom Scan (GpuPreAgg)" agg "$FAULTS_AGG" "select count(*), sum(b), min(a), max(c), avg(b) from fb1 where a > -5000"
fbrun "agg: GROUP BY" "Custom Scan (GpuPreAgg)" agg "$FAULTS_AGG" "select a % 7 as k, count(*), sum(b), max(c) from fb1 group by 1"
fbrun "agg: two keys" "Custom Scan (GpuPreAgg)" agg "refuse=3 lost=2" "select a % 5, b % 3, count(*), sum(c) from fb1 group by 1, 2"
fbrun "agg: over a join" "Custom Scan (GpuPreAgg)" jagg "refuse refuse=3 drop=3 final" "select sd.grp, count(*), sum(fb1.b) from fb1 join sd on fb1.k = sd.k group by 1"
fbrun "agg: over a right join" "Custom Scan (GpuPreAgg)" jagg "refuse=3 final" "select sd.grp, count(*), count(fb1.id) from fb1 right join sd on fb1.k = sd.k and fb1.b > 40000 group by 1"
fbrun "agg: an expression that is both a qual and an aggregate argument (replay evaluates it)" "Custom Scan (GpuPreAgg)" agg "refuse drop=3" "select count(a + b), sum(a + b), count(*) from fb1 where a + b > 100"
fbrun "agg: the same with GROUP BY" "Custom Scan (GpuPreAgg)" agg "refuse" "select a % 3, count(b * 2), sum(b * 2) from fb1 where b * 2 > 100 group by 1"
fbrun "agg: recheck rows are held back (no duplicates)" "Custom Scan (GpuPreAgg)" agg "refuse=3 drop=3 final" "select a % 4, count(*), sum(length(f)) from fb1 group by 1"

echo "-- Arrow and Parquet"
for t in a1 p1; do
	fbrun "$t: scan" "Custom Scan (GpuScan)" scan "refuse refuse=3 drop=2 open" "select i, a, b, f from $t where b > 100 and a < 5000"
	fbrun "$t: aggregation" "Custom Scan (GpuPreAgg)" agg "refuse=3 drop=2 final" "select a % 7, count(*), sum(b), max(c) from $t group by 1"
	fbrun "$t: join" "Custom Scan (GpuJoin)" join "refuse=2 lost=3" "select $t.i, sd.pay from $t join sd on $t.b = sd.k where $t.b > 0"
done

echo "-- GPU-Cache"
sql "create table fc1 (id int4, a int2, b int4, f text)"
sql "insert into fc1 select i, (i % 20000 - 10000)::int2, (i * 31) % 1000, repeat(chr(97 + i % 26), (i % 20)::int) from generate_series(1, $FBN) i"
sql "analyze fc1"
sql "create trigger gc_sync after insert or update or delete on fc1 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((FBN * 3))')"
sql "update fc1 set b = b + 1 where id % 5 = 0"
sql "delete from fc1 where id % 13 = 0"
sql "analyze fc1"
fbcache() { # label, kind, faults, query
	plan=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; $FBAGGSET explain $4" 2>&1 | grep -c "GPU Cache:")
	if [ "$plan" -lt 1 ]; then echo "  FAIL  fc1: $1 (the plan does not read the GPU-Cache)"; fail=$((fail+1)); return; fi
	case $2 in agg) pat="Custom Scan (GpuPreAgg)";; *) pat="Custom Scan (GpuScan)";; esac
	fbrun "fc1: $1" "$pat" $2 "$3" "$4"
}
fbcache "scan" scan "refuse drop lost open" "select id, a, f from fc1 where b > 500 and a < 5000"
fbcache "aggregation" agg "refuse drop final" "select a % 5, count(*), sum(b) from fc1 group by 1"

echo "-- parallel workers"
PAR="set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0; set min_parallel_index_scan_size=0;"
n=$($PSQL -d scantest -c "$GPU $PAR set enable_seqscan=off; explain select id, a from fb1 where b > 100" 2>&1 | grep -c "Gather")
check "a parallel GpuScan plan is chosen" "$([ "$n" -ge 1 ]; echo $?)"
fbrun "parallel scan" "Gather" scan "refuse refuse=3 drop=2" "select id, a, f from fb1 where b > 100" "$PAR"
fbrun "parallel aggregation" "Gather" agg "refuse drop=3" "select a % 5, count(*), sum(b) from fb1 group by 1" "$PAR"
fbrun "parallel join" "Gather" jagg "refuse=2" "select fb1.id, sd.pay from fb1 join sd on fb1.k = sd.k" "$PAR"

echo "-- real refusals (the planner's limits switched off: the service rejects the program)"
NL="set pg_strom.debug_no_program_limits=on;"
# 40 referenced columns: more kernel variables than the VM has registers
wdef=$(i=1; while [ $i -le 40 ]; do printf 'c%d int4, ' $i; i=$((i+1)); done)
wsel=$(i=1; while [ $i -le 40 ]; do printf 'i + %d, ' $i; i=$((i+1)); done)
sql "create table fw (${wdef}id int4)"
sql "insert into fw select ${wsel}i from generate_series(1, 20000) i"
sql "analyze fw"
wcols=$(i=1; while [ $i -le 40 ]; do printf 'c%d, ' $i; i=$((i+1)); done)
fbrun "refused: 40 referenced columns (GpuScan)" "Custom Scan (GpuScan)" scan "-" "select ${wcols}id from fw where c1 + c40 > 100 and c2 < 30000" "$NL"
fbrun "refused: 80 columns in a join (unsupported join session)" "Custom Scan (GpuJoin)" join "-" "select x.*, y.* from fw x join fw y on x.c1 = y.c1 where x.c2 > 10 and y.c3 > 10" "$NL"
aggs=$(i=1; while [ $i -le 16 ]; do printf 'sum(b+%d), ' $i; i=$((i+1)); done)
fbrun "refused: 16 aggregates" "Custom Scan (GpuPreAgg)" agg "-" "select ${aggs}count(*) from fb1" "$NL"

echo "-- the error is raised when the fallback is off or impossible"
out=$($PSQL -d scantest -c "$GPU set pg_strom.gpu_error_fallback=off; set pg_strom.gpuserv_fault='refuse'; select count(*) from fb1 where b > 100" 2>&1)
echo "$out" | grep -q "needs more than 32 registers"; check "pg_strom.gpu_error_fallback=off: the refusal is an error" $?
out=$($PSQL -d scantest -c "$GPU set pg_strom.enable_gpusort=on; set pg_strom.gpuserv_fault='refuse'; set enable_sort=off; $FBAGGSET select a, sum(b) from fb1 group by a order by 2 desc, 1 limit 5" 2>&1)
echo "$out" | grep -q "needs more than 32 registers"; check "GPU-Sort: the refusal stays an error (the Sort node was removed)" $?

echo "-- EXPLAIN ANALYZE shows the fallback"
out=$($PSQL -d scantest -c "$GPU $FBAGGSET $FBSET set pg_strom.gpuserv_fault='drop=3'; explain (analyze, costs off, timing off, summary off) select a % 7, count(*), sum(b) from fb1 group by 1" 2>&1)
echo "$out" | grep -q "cpu-mode: 1 worker(s)"; check "Fallback-stats names the CPU mode" $?
out=$($PSQL -d scantest -c "$GPU $FBAGGSET $FBSET explain (analyze, costs off, timing off, summary off) select a % 7, count(*), sum(b) from fb1 group by 1" 2>&1)
echo "$out" | grep -q "cpu-mode"; [ $? -ne 0 ]; check "no cpu-mode without a fault" $?

echo "-- a service that is killed in the middle of a query"
b=$(fbcount)
g=$(q "$GPU $FBJOINSET $FBSET set pg_strom.gpuserv_fault='exit=2';" "select fb1.id, sd.pay from fb1 join sd on fb1.k = sd.k"); c=$(q "$CPU" "select fb1.id, sd.pay from fb1 join sd on fb1.k = sd.k")
[ "$g" = "$c" ] && [ "$(fbcount)" -gt "$b" ]; check "service exits at task 2: the rows are the CPU's ($g)" $?
n=0; while [ $n -lt 60 ]; do
	[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
check "the launcher restarted the service" "$([ $n -lt 60 ]; echo $?)"
g=$(q "$GPU $FBAGGSET" "select a % 7, count(*), sum(b) from fb1 group by 1"); c=$(q "$CPU" "select a % 7, count(*), sum(b) from fb1 group by 1")
[ "$g" = "$c" ]; check "queries run on the GPU again after the restart" $?
