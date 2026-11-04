#!/bin/sh
# Quick development run of the PostGIS tests (postgis_tests.inc) on a small data set, without the rest of scan_test.sh.
# Same cluster setup and helper functions as scan_test.sh (kept small and copied here), so run it through locked_test.sh:
#
#   sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
#
# ADHOC=<file> runs the SQL file (psql -f) against the loaded data instead of the tests; PGROWS (default 200000) sizes the
# tables. The service runs with PGSTROM_GPUSERV_SPECIALIZE=check (every dispatch runs the VM kernel and the specialized one
# and compares them) unless SPEC is set.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pg_dev}
PORT=${PORT:-54404}
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
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_SPECIALIZE=${SPEC:-check}
export PGSTROM_GPUSERV_CACHE_DIR=$ROOT/speccache
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
run() { # table, label, query-with-%T
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ]; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
run_any() { # like run, but the plan need not use GpuScan
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && [ "${g#*|}" != "" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
ANYPLAN=0
__runagg() {
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
JOINPAT="Custom Scan (GpuJoin)"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
runjoin() { # table, label, query-with-%T and %D [, settings]; the plan must use GpuJoin (unless ANYPLAN=1)
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/f8_dim/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $set_extra explain $qry" | grep -c "$JOINPAT")
	g=$(q "$GPU $JOINSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuJoin=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runagg() { ANYPLAN=0; __runagg "$@"; }
runagg_any() { ANYPLAN=1; __runagg "$@"; }

if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -n "$NOTESTS" ] && exit 0
fi
. "${0%/*}/postgis_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "POSTGIS DEV: ALL PASSED"; else echo "POSTGIS DEV: $fail FAILED"; exit 1; fi
