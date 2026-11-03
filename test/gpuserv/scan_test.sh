#!/bin/sh
# Differential test of the Vulkan GpuScan: every query runs on the GPU and on the CPU
# (pg_strom.enabled = off) and the result sets must be identical. Needs the module and the
# service installed in the PostgreSQL given by PGBIN (make -C src install).
#
#   PGBIN=/path/to/bin GPUSERV_BIN=src/gpuserv/pgstrom-gpuserv sh test/gpuserv/scan_test.sh
#
# ALLOW_CPU=on runs it on a CPU Vulkan device (lavapipe). KEEP=1 keeps the cluster.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pgstrom_scan_test}
PORT=${PORT:-54334}
NROWS=${NROWS:-300000}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
fail=0

check() { # name rc
	if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
sql() { $PSQL -d scantest -c "$1"; }

stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
max_worker_processes = 24
CONF
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

# ---- data: NULLs, several integer widths, float4 with NaN/Infinity, short and long text
sql "create table s_base as select i,
	((i % 20000) - 10000)::int2 as a,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end as b,
	((i::int8 * 1000003) % 1000000007 - 500000000) as c,
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4
	     else (((i * 17) % 10000) / 7.0)::float4 end as d,
	(i % 3 = 0) as e,
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 200)::int) end as f
	from generate_series(1, $NROWS) i"
sql "create table s1 as select * from s_base"; sql "vacuum analyze s1"          # all pages all-visible (read from the file)
sql "create table s2 as select * from s_base"; sql "update s2 set b = b + 1 where i % 5 = 0"
sql "delete from s2 where i % 13 = 0"; sql "analyze s2"                         # dead tuples, nothing vacuumed
sql "create table s3 as select * from s_base"; sql "vacuum s3"
sql "update s3 set c = c + 7 where i % 17 = 0"; sql "analyze s3"                # mix of vacuumed and fresh pages

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
run() { # table, label, query-with-%T
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ]; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

run_any() { # like run, but the plan need not use GpuScan (the expression may stay on the CPU)
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && [ "${g#*|}" != "" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runprep() { # table, label, prepared query with $1/$2, execute arguments
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); args=$4
	pre="set plan_cache_mode=force_generic_plan; prepare pq(int, int) as select md5(coalesce(string_agg(x::text, E'\\n' order by x::text), '')), count(*) from ($qry) x; execute pq($args);"
	g=$($PSQL -d scantest -c "$GPU $pre" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $pre" 2>&1 | tail -1)
	plan=$($PSQL -d scantest -c "$GPU set plan_cache_mode=force_generic_plan; prepare pe(int, int) as $qry; explain execute pe($args)" 2>&1 | grep -c "Custom Scan (GpuScan)")
	if [ "$g" = "$c" ] && [ "$plan" -ge 1 ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

for t in s1 s2 s3; do
	echo "table $t"
	run $t "trivial qual, mixed projection" "select a, f, c from %T where i > 0"
	run $t "select * with int4 qual" "select * from %T where b < 100"
	run $t "int2/int8 qual" "select a, c from %T where a > 0 and c < 0"
	run $t "int2*int4 arithmetic, NULLs" "select a + b, c from %T where b is not null and a * 2 < b"
	run $t "float4 compare incl. NaN/Infinity" "select i, d from %T where d > 100.5::float4"
	run $t "float4 arithmetic" "select i, d * 2::float4 + 1::float4 from %T where d < 50::float4"
	run $t "bool column and OR" "select i, e from %T where e and (b > 0 or c > 0)"
	run $t "division and modulo" "select b, c / 3, c % 7 from %T where c > 1000"
	run $t "IS NULL on several columns" "select i from %T where b is null or f is null"
	run $t "casts" "select a::int4 + b, c::int4 from %T where c between -1000000 and 1000000"
	run $t "unary minus/plus, NOT" "select -a, +b from %T where not (a < 0 or b > 0)"
	run $t "varlena pass-through with qual on other column" "select f from %T where a = 123"
	run $t "10 computed columns (16-register kernel)" "select a+1, a+2, a+3, a+4, a+5, b+1, b+2, b+3, b+4, b+5 from %T where a > 9000"
	run $t "24 computed columns (32-register kernel)" "select a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11,a+12, b-1,b-2,b-3,b-4,b-5,b-6,b-7,b-8,b-9,b-10,b-11,b-12 from %T where a > 9500"
	run $t "int -> float4 -> int casts, rounding to even" "select b, (b::float4 / 2::float4)::int4, b::float4 * 3::float4 from %T where b between -100 and 100"
	run $t "float4 vs int cast compare" "select i from %T where b::float4 > 1000.5::float4 and a < 0"
	runprep $t "query parameters (generic plan)" "select a, b from %T where b < \$1 and a > \$2" "500, -100"
	echo "  -- expressions the VM does not have stay on the CPU"
	run_any $t "float8 comparison" "select i, d from %T where d > 100.5"
	run_any $t "text equality" "select i from %T where f = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"
	run_any $t "CASE" "select i, case when b > 0 then a else 0 end from %T where a > 9000"
	run_any $t "IN list" "select i from %T where a in (1, 5, 9000, 9999)"
	run_any $t "COALESCE" "select i, coalesce(b, 0) from %T where coalesce(b, 0) > 49000"
	run_any $t "IS TRUE, and supported + unsupported quals" "select i from %T where e is true and b > 40000 and f like 'a%'"
	run_any $t "text function in projection" "select i, upper(f) from %T where a > 9990"
	run $t "IS NULL on a text column" "select i from %T where f is null and a > 0"
	run $t "IS NOT NULL on a text column" "select f from %T where f is not null and b > 49000"
done

echo "aggregation (GpuPreAgg)"
runagg() { # table, label, query-with-%T [, settings]; the plan must use GpuPreAgg
	ANYPLAN=0; __runagg "$@"
}
runagg_any() { # the same, but the aggregate may also stay on the CPU (results must match)
	ANYPLAN=1; __runagg "$@"
}
__runagg() {
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
for t in s1 s2 s3; do
	runagg $t "count, sum, min, max, avg of int4 with NULLs" "select count(*), count(b), sum(b), min(b), max(b), avg(b) from %T"
	runagg $t "int2 and int8 columns" "select sum(a), min(a), max(a), avg(a), sum(c), min(c), max(c), avg(c) from %T"
	runagg $t "with quals" "select count(*), sum(a), max(c) from %T where a > 0 and b is not null"
	runagg $t "empty input" "select count(*), count(b), sum(b), min(b), max(c), avg(a) from %T where a > 100000"
	runagg $t "only NULL values" "select count(b), sum(b), min(b), max(b) from %T where b is null"
	runagg $t "expressions as arguments" "select sum(a + b), max(b * 2), min(c - 1), count(a * 2) from %T"
	runagg $t "FILTER (WHERE ...)" "select count(*) filter (where b > 0), sum(b) filter (where a < 0), max(c) filter (where e) from %T"
	runagg $t "sum beyond 64 bits" "select sum(c + 9000000000000000000), avg(c + 9000000000000000000), sum(c) from %T"
	runagg_any $t "float aggregates stay on the CPU too" "select count(*), sum(a), sum(d) from %T where a > 9000"
done
runagg s1 "parallel workers" "select count(*), sum(b), min(c), max(a), avg(b) from %T where a > -5000" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"

echo "GROUP BY"
BIGGROUPS="set pg_strom.gpupreagg_max_groups=1000000;"
for t in s1 s2 s3; do
	runagg $t "one int key" "select a % 10 as k, count(*), sum(b), min(c), max(b), avg(a) from %T group by 1 order by 1"
	runagg $t "key with NULLs" "select b % 5 as k, count(*), count(b), sum(a) from %T group by 1 order by 1"
	runagg $t "two integer keys" "select a % 4 as k1, b % 3 as k2, count(*), sum(c), max(a) from %T group by 1, 2 order by 1, 2"
	runagg_any $t "bool key (stays on the CPU)" "select a % 4 as k1, e, count(*), sum(c), max(a) from %T group by 1, 2 order by 1, 2"
	runagg $t "int8 key" "select c % 7 as k, count(*), sum(a) from %T group by 1 order by 1"
	runagg $t "with quals and FILTER" "select a % 6 as k, count(*) filter (where b > 0), sum(b) filter (where e), max(c) from %T where a > -9000 group by 1 order by 1"
	runagg_any $t "empty input" "select a % 6 as k, count(*), sum(b) from %T where a > 100000 group by 1 order by 1"
	runagg $t "sum beyond 64 bits per group" "select a % 3 as k, sum(c + 9000000000000000000), avg(c + 9000000000000000000) from %T group by 1 order by 1"
	runagg $t "more groups than a workgroup's table (CPU overflow path)" "select i % 1500 as k, count(*), sum(a), min(b), max(c) from %T group by 1 order by 1" "$BIGGROUPS"
	runagg $t "overflow with two keys" "select i % 97 as k1, i % 13 as k2, count(*), sum(a) from %T group by 1, 2 order by 1, 2" "$BIGGROUPS"
done
runagg s1 "parallel workers" "select a % 10 as k, count(*), sum(b), max(c) from %T group by 1 order by 1" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
runagg_any s1 "text key stays on the CPU" "select f, count(*) from %T group by f order by 1 limit 20"
runagg_any s1 "too many estimated groups stay on the CPU" "select i, count(*) from %T where i < 50000 group by i order by 1 limit 50" "set pg_strom.gpupreagg_max_groups=16;"
plan=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=16; explain select i % 1000, count(*) from s1 group by 1" | grep -c "Custom Scan (GpuPreAgg)"); [ "$plan" -eq 0 ]; check "a group count above pg_strom.gpupreagg_max_groups is not offloaded" $?

echo "joins"
sql "create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp,
	case when i % 9 = 0 then null else (i::int8 * 3 - 5) end as pay, repeat(chr(65 + i % 26), (i % 30)::int) as txt, i::int8 as k8
	from generate_series(1, 20000) i"
sql "insert into sd select k, grp + 100, pay, txt, k8 from sd where k % 10 = 0"          # duplicate keys: several matches per outer row
sql "analyze sd"
JOINPAT="Custom Scan (GpuJoin)"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
runjoin() { # table, label, query-with-%T and %D [, settings]; the plan must use GpuJoin
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $set_extra explain $qry" | grep -c "$JOINPAT")
	g=$(q "$GPU $JOINSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuJoin=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
for t in s1 s2 s3; do
	ANYPLAN=0
	runjoin $t "int4 = int4, pass-through columns of both sides" "select t.i, t.a, d.grp, d.pay from %T t join %D d on t.b = d.k"
	runjoin $t "int8 key" "select t.i, d.grp from %T t join %D d on t.c % 20000 = d.k8"
	runjoin $t "quals on both sides" "select t.i, d.grp, d.pay from %T t join %D d on t.b = d.k where t.a > 0 and d.grp < 8 and d.pay is not null"
	runjoin $t "computed projection mixing both sides" "select t.i, t.a + d.grp as x, d.pay * 2 as y, t.b - d.k as z from %T t join %D d on t.b = d.k"
	runjoin $t "text column of the inner side" "select t.i, d.txt from %T t join %D d on t.b = d.k"
	runjoin $t "text column of the outer side" "select d.grp, t.f from %T t join %D d on t.b = d.k where t.i < 5000"
	runjoin $t "extra join qual" "select t.i, d.grp from %T t join %D d on t.b = d.k and t.a > d.grp"
	runjoin $t "no match at all" "select t.i, d.grp from %T t join %D d on t.b = d.k + 1000000"
	runjoin $t "ctid of the outer side" "select t.ctid, d.grp from %T t join %D d on t.b = d.k where t.i % 100 = 0"
	runjoin $t "LEFT JOIN: unmatched outer rows get NULLs" "select t.i, t.b, d.grp, d.pay, d.txt from %T t left join %D d on t.b = d.k + 20000 where t.i < 60000"
	runjoin $t "LEFT JOIN with duplicate inner keys and computed values" "select t.i, t.a + d.grp as x, coalesce(d.pay, -1) as y from %T t left join %D d on t.b = d.k"
	runjoin $t "LEFT JOIN with an ON qual on the inner side" "select t.i, d.grp from %T t left join %D d on t.b = d.k and d.grp < 3"
	runjoin $t "LEFT JOIN with an ON qual on the outer side" "select t.i, d.grp from %T t left join %D d on t.b = d.k and t.a > 0"
	runjoin $t "LEFT JOIN with no match at all" "select t.i, d.grp, d.txt from %T t left join %D d on t.b = d.k + 1000000"
done
sql "create table sd2 as select (i - 1)::int4 as k2, (i % 5)::int4 as g2, (i * 11)::int8 as p2 from generate_series(1, 400) i"
sql "insert into sd2 select k2, g2 + 50, p2 + 1 from sd2 where k2 % 40 = 0"                # duplicates at the second depth
sql "create table sd3 as select (i * 3)::int4 as k3, (i % 7)::int4 as g3 from generate_series(0, 30000) i"
sql "analyze sd2"; sql "analyze sd3"
# stacked joins: several inner relations in one GpuJoin ("Join Quals [2]" in the plan). Without enable_seqscan=off
# the planner keeps the big table as the probe side; the plan must show the second depth.
JOINSET_SAVE="$JOINSET"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
JOINPAT='Join Quals \[2\]'
for t in s1 s2 s3; do
	ANYPLAN=0
	runjoin $t "two inner relations, columns of both and computed" "select t.i, t.a + d.grp + e.g2 as x, d.pay, e.p2, e.k2 from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2"
	runjoin $t "two inner relations, duplicate keys at both depths" "select t.i, d.grp, e.g2, e.p2 from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 where t.a > -200"
	runjoin $t "three inner relations" "select t.i, d.grp, e.g2, f.g3 from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 join sd3 f on t.a + 20000 = f.k3"
	runjoin $t "inner join, then LEFT join" "select t.i, d.grp, e.g2, e.p2 from %T t join %D d on t.b = d.k left join sd2 e on t.a % 40 = e.k2 + 400"
	runjoin $t "LEFT join, then inner join" "select t.i, d.grp, e.g2 from %T t left join %D d on t.b = d.k + 20000 join sd2 e on t.a % 400 = e.k2 where t.i < 80000"
	runjoin $t "two LEFT joins, the first without matches" "select t.i, d.grp, e.g2 from %T t left join %D d on t.b = d.k + 1000000 left join sd2 e on t.a % 400 = e.k2 where t.i < 50000"
	runjoin $t "text columns of two inner relations" "select t.i, d.txt, e.k2 from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2 where t.i < 30000"
done
JOINPAT="Custom Scan (GpuJoin)"
for t in s1 s2 s3; do
	runjoin $t "join of a join result (separate GpuJoins)" "select t.i, d.grp, e.g2 from %T t join %D d on t.b = d.k join sd2 e on d.grp = e.k2 where t.a < 0"
done
JOINSET="$JOINSET_SAVE"
ANYPLAN=1
runjoin s1 "aggregate on top (the join stays below the CPU aggregate)" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1 order by 1"
runjoin s1 "LEFT JOIN with a WHERE qual on the inner side stays on the CPU" "select t.i, d.grp from %T t left join %D d on t.b = d.k where d.grp is null and t.i < 3000"
runjoin s1 "float key stays on the CPU" "select t.i, d.grp from %T t join %D d on t.d::float8 = d.k::float8 where t.i < 3000"
ANYPLAN=0
runjoin s1 "parallel workers" "select t.i, d.grp from %T t join %D d on t.b = d.k" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"

. "${0%/*}/types_tests.inc"

echo "errors"
for t in s1; do
	g=$($PSQL -d scantest -c "$GPU select c / a from $t where b < 0" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select c / a from $t where b < 0" 2>&1 | head -1)
	echo "$g" | grep -q "division by zero" ; check "$t: division by zero is reported on the GPU" $?
	echo "$c" | grep -q "division by zero" ; check "$t: ... and on the CPU" $?
done

echo "parallel workers"
g=$(q "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;" "select a, c from s1 where b < 1000")
c=$(q "$CPU" "select a, c from s1 where b < 1000")
[ "$g" = "$c" ]; check "parallel GpuScan matches the CPU" $?

[ -z "$KEEP" ] && stop
echo
if [ $fail -eq 0 ]; then echo "SCAN TEST: ALL PASSED"; else echo "SCAN TEST: $fail FAILED"; exit 1; fi
