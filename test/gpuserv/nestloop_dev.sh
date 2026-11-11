#!/bin/sh
# Quick development run of the nest-loop join tests (A4: a GpuJoin without a hash key: non-equi, range, cross, IS NOT DISTINCT FROM)
# on a small data set. Run it through locked_test.sh:
#
#   PORT=54438 sh test/gpuserv/locked_test.sh "$PWD" nestloop_dev.sh
#
# ADHOC=<file> runs the SQL file (psql -f) against the loaded data first (SKIP_TESTS=1: instead of the tests); NROWS sizes the outer table.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-${PGSTROM_TEST_DATA:-/Volumes/CrucialX9/Projects/pg-strom/test-data}/a4-nestloop/nl_dev}
PORT=${PORT:-54438}
NROWS=${NROWS:-30000}
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
[ -n "$TRACE" ] && export PGSTROM_GPUSERV_TRACE=1
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -n "$SKIP_TESTS" ] && exit 0
fi
. "${0%/*}/nestloop_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "NESTLOOP DEV: ALL PASSED"; else echo "NESTLOOP DEV: $fail FAILED"; exit 1; fi
