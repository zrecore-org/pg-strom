#!/bin/sh
# Quick development run of the many-groups GROUP BY tests (gtab_tests.inc: the session-wide hash table of gtab.slang), without the
# rest of scan_test.sh; run it through locked_test.sh:
#
#   ROOT=... PORT=54423 sh test/gpuserv/locked_test.sh "$PWD" gtab_dev.sh
#
# The service runs with PGSTROM_GPUSERV_GTAB=1 (every GROUP BY uses the table, also the few-group queries) unless GTAB is set,
# and with PGSTROM_GPUSERV_SPECIALIZE=check unless SPEC is set. ADHOC=<file> runs the SQL file against the loaded tables first.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/gtab_dev}
PORT=${PORT:-54423}
PSQL="${0%/*}/qtimeout.sh $ROOT/.query_timeout $PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
fail=0
T0=$(date +%s)

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
export PGSTROM_GPUSERV_SPECIALIZE=${SPEC:-check}
export PGSTROM_GPUSERV_GTAB=${GTAB:-1}
export PGSTROM_GPUSERV_CACHE_DIR=$ROOT/speccache
export PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=${PGSTROM_GPUSERV_PIPELINE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pgstrom}   # the persisted pipeline cache is shared with prewarm_kernels.sh, not per test
export PGSTROM_GPUSERV_TRACE=1
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
q() { # settings, query -> md5 of the sorted result text and the row count
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
__runagg() {
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=10000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$(q "$GPU set pg_strom.gpupreagg_max_groups=10000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runagg() { ANYPLAN=0; __runagg "$@"; }
runagg_any() { ANYPLAN=1; __runagg "$@"; }

if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -n "$NOTESTS" ] && exit 0
fi
. "${0%/*}/gtab_tests.inc"
# the memory bound: pg_strom.gpuserv_group_memory is a postmaster setting, so restart the cluster with its minimum (64 MB)
stop
echo "pg_strom.gpuserv_group_memory = 64" >> "$ROOT/data/postgresql.conf"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log2" -w start >/dev/null 2>&1 || { echo "restart failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
check "service is ready with pg_strom.gpuserv_group_memory = 64" "$([ $n -lt 40 ]; echo $?)"
. "${0%/*}/gtab_mem_tests.inc"
echo
nused=$(cat "$ROOT/log" "$ROOT/log2" 2>/dev/null | grep -c "group task:.*table")
check "the session-wide table served the tasks ($nused task lines)" "$([ "$nused" -gt 0 ]; echo $?)"
if [ $fail -eq 0 ]; then echo "GTAB DEV: ALL PASSED ($(( $(date +%s) - T0 )) s)"; else echo "GTAB DEV: $fail FAILED ($(( $(date +%s) - T0 )) s)"; exit 1; fi
