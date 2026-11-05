#!/bin/sh
# Quick development run of the "aggregation over a join" tests (joinagg_tests.inc) on a small data
# set, without the rest of scan_test.sh. Same cluster setup and helper functions as scan_test.sh
# (kept small and copied here), so run it through locked_test.sh:
#
#   sh test/gpuserv/locked_test.sh "$PWD" joinagg_dev.sh
#
# ADHOC=<file> runs the SQL file (psql -f) against the loaded data instead of the tests, and
# prints the output; NROWS (default 300000) sizes the big tables.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/a4c_jagg}
PORT=${PORT:-54394}
NROWS=${NROWS:-300000}
PSQL="${0%/*}/qtimeout.sh $ROOT/.query_timeout $PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
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
sh "${0%/*}/pgconf_test.sh" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
[ -f /tmp/a4c_trace ] && export PGSTROM_GPUSERV_TRACE=1
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

sql "create table s_base as select i,
	((i % 20000) - 10000)::int2 as a,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end as b,
	((i::int8 * 1000003) % 1000000007 - 500000000) as c,
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4
	     else (((i * 17) % 10000) / 7.0)::float4 end as d,
	(i % 3 = 0) as e,
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 200)::int) end as f
	from generate_series(1, $NROWS) i"
sql "create table s1 as select * from s_base"; sql "vacuum analyze s1"
sql "create table s2 as select * from s_base"; sql "update s2 set b = b + 1 where i % 5 = 0"
sql "delete from s2 where i % 13 = 0"; sql "analyze s2"
sql "create table s3 as select * from s_base"; sql "vacuum s3"
sql "update s3 set c = c + 7 where i % 17 = 0"; sql "analyze s3"
sql "create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp,
	case when i % 9 = 0 then null else (i::int8 * 3 - 5) end as pay, repeat(chr(65 + i % 26), (i % 30)::int) as txt, i::int8 as k8
	from generate_series(1, 20000) i"
sql "insert into sd select k, grp + 100, pay, txt, k8 from sd where k % 10 = 0"
sql "analyze sd"
sql "create table sd2 as select (i - 1)::int4 as k2, (i % 5)::int4 as g2, (i * 11)::int8 as p2 from generate_series(1, 400) i"
sql "insert into sd2 select k2, g2 + 50, p2 + 1 from sd2 where k2 % 40 = 0"
sql "create table sd3 as select (i * 3)::int4 as k3, (i % 7)::int4 as g3 from generate_series(0, 30000) i"
sql "analyze sd2"; sql "analyze sd3"
sql "create table sdn as select k, grp, pay, k8 from sd where k is not null"; sql "alter table sdn alter column k set not null"
sql "alter table sdn alter column k8 set not null"; sql "analyze sdn"

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
JOINPAT="Custom Scan (GpuJoin)"
ANYPLAN=0
runjoin() { # table, label, query-with-%T and %D [, settings]; the plan must use GpuJoin
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $set_extra explain $qry" | grep -c "$JOINPAT")
	g=$(q "$GPU $JOINSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuJoin=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runagg_any() { :; }

[ -z "$ADHOC" ] && [ -f /tmp/a4c_adhoc.sql ] && ADHOC=/tmp/a4c_adhoc.sql
if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -f /tmp/a4c_trace ] && grep "pgstrom-gpuserv: .*task" "$ROOT/log" | tail -${TRACE_LINES:-40}
	[ -f /tmp/a4c_skip_tests ] && exit 0
fi
. "${0%/*}/joinagg_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "JOINAGG DEV: ALL PASSED"; else echo "JOINAGG DEV: $fail FAILED"; exit 1; fi
