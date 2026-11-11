#!/bin/sh
# Quick development run of the first-use kernel compile tests (firstuse_tests.inc, A9 part 3): background prewarm, the
# readiness gate (a query whose kernel the driver has not compiled continues on the CPU), the persisted state, and the
# install-time `pgstrom-gpuserv --prewarm-all`. Needs the GPU; run it through locked_test.sh (PREWARM=off: this script
# starts from an empty pipeline cache directory of its own on purpose):
#
#   PORT=54436 PREWARM=off sh test/gpuserv/locked_test.sh "$PWD" firstuse_dev.sh
#
# The cache directory is $ROOT/pcache (PGSTROM_GPUSERV_PIPELINE_CACHE_DIR), so nothing of ~/.cache/pgstrom is used or changed.
# The Metal compiler keeps its own cache outside it: a kernel the driver has compiled before for any checkout compiles in
# seconds here, not minutes, so the stall these tests avoid is shorter than on a fresh machine (the timings say so).
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/fu_dev}
PORT=${PORT:-54436}
FBN=${FBN:-30000}
PSQL="${0%/*}/qtimeout.sh $ROOT/.query_timeout $PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
fail=0

check() { # name rc
	if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
sql() { $PSQL -d scantest -c "$1"; }

# the tests turn on what locked_test.sh turns off for the others
unset PGSTROM_GPUSERV_KERNEL_WAIT_MS PGSTROM_GPUSERV_PREWARM
mkdir -p "$ROOT"
export PGSTROM_GPUSERV_PIPELINE_CACHE_DIR="$ROOT/pcache"
export PGSTROM_GPUSERV_PREWARM_DELAY_MS=0 PGSTROM_GPUSERV_PREWARM_IDLE_MS=300

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
pg_strom.gpu_specialize = 'off'
pg_strom.gpu_prewarm = 'off'
max_worker_processes = 24
CONF
sh "${0%/*}/pgconf_test.sh" >> "$ROOT/data/postgresql.conf"
[ -z "$KEEP" ] && trap stop EXIT INT TERM

. "${0%/*}/firstuse_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "FIRSTUSE DEV: ALL PASSED"; else echo "FIRSTUSE DEV: $fail FAILED"; exit 1; fi
