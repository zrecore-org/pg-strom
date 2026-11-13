#!/bin/sh
# Quick development run of the min / max of text, varchar and bpchar tests (A6b, minmaxtext_tests.inc) on a small data set.
# Same cluster setup as keys_dev.sh; run it through locked_test.sh:
#
#   sh test/gpuserv/locked_test.sh "$PWD" minmaxtext_dev.sh
#
# MM_EXPECT_GPU=1 (the default) also asserts that the ungrouped C-collation shapes plan as GpuPreAgg; MM_EXPECT_GPU=0 checks the results
# only (a service without the aggregate: the planner keeps them on the CPU). NROWS sizes the tables.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-${PGSTROM_TEST_DATA:-/Volumes/CrucialX9/Projects/pg-strom/test-data}/a6b/minmaxtext_dev}
PORT=${PORT:-54425}
NROWS=${NROWS:-60000}
MM_EXPECT_GPU=${MM_EXPECT_GPU:-1}
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
n=0; while [ $n -lt 120 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 120 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

. "${0%/*}/minmaxtext_setup.inc"

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
# ADHOC=<file>: run that SQL first (SKIP_TESTS=1: and nothing else)
if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -n "$SKIP_TESTS" ] && exit 0
fi
. "${0%/*}/minmaxtext_tests.inc"

# forced overflow (A6b): without the switch to the session-wide table (PGSTROM_GPUSERV_GTAB=0) the 3000 groups of gm do not fit the workgroups'
# tables of gagg.slang, so most rows overflow to the CPU (group_cpu_rows, and the overflow lists of the Arrow kernels): the extremes of those rows
# are resolved on the host and must merge with the kernels' by bytes
echo "min / max of strings: overflow rows (PGSTROM_GPUSERV_GTAB=0)"
stop
PGSTROM_GPUSERV_GTAB=0 $PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "restart failed"; exit 1; }
n=0; while [ $n -lt 120 ]; do
	[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
check "service is ready again" "$([ $n -lt 120 ]; echo $?)"
for src in tmo tmoa tmop tmog; do
	rmc "$src: overflow rows, many groups" 'select gm, min(t collate "C"), max(v collate "C"), min(w collate "C"), count(*), sum(n) from %T group by gm' $src
	rmr "$src: overflow rows, NULL and empty extremes" 'select gm, max(t collate "C"), min(v collate "C") filter (where n > 3) from %T where g >= 10 group by gm' $src
done
rmc "heap: overflow rows, text key" 'select w, min(t collate "C"), max(v collate "C") from %T group by w' tmo
echo
if [ $fail -eq 0 ]; then echo "MINMAXTEXT DEV: ALL PASSED"; else echo "MINMAXTEXT DEV: $fail FAILED"; exit 1; fi
