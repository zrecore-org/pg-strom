#!/bin/sh
# Run the upstream-style regression suite (PGXS pg_regress, test/Makefile) against a throwaway cluster
# that is kept between runs, so the 6M-row SSBM dataset (pgstrom_init) is loaded only once.
#
#   sh test/vulkan_installcheck.sh                                   # vulkan_schedule (must be green)
#   TARGET=pending-installcheck sh test/vulkan_installcheck.sh       # pending_schedule (expected to fail)
#   TARGET=installcheck sh test/vulkan_installcheck.sh               # the whole upstream parallel_schedule
#   SCHEDULE=./my_schedule sh test/vulkan_installcheck.sh            # an arbitrary schedule file
#
# Needs the module and the service installed (make -C src PG_CONFIG=... install). With several checkouts
# sharing one Homebrew PostgreSQL run it under the test lock:
#   sh test/gpuserv/locked_test.sh <tree> test/vulkan_installcheck.sh
#
# Environment: PGBIN, GPUSERV_BIN (default: the installed one), ROOT (cluster directory, default
# /tmp/pgstrom_regress), PORT (54396), ALLOW_CPU=on (lavapipe), FRESH=1 (rebuild the cluster and the
# dataset), HOLD=1 (start the cluster and wait for $ROOT/release: an interactive session), KEEP_RUNNING=1 (leave the cluster up; stop it with pg_ctl -D $ROOT/data stop).
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:-$($PGBIN/pg_config --pkglibdir)/pgstrom-gpuserv}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pgstrom_regress}
PORT=${PORT:-54396}
TARGET=${TARGET:-vulkan-installcheck}
HERE=$(cd "$(dirname "$0")" && pwd)
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"

stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
[ -d "$ROOT/data" ] && stop
if [ "${FRESH:-0}" = 1 ]; then rm -rf "$ROOT"; fi
mkdir -p "$ROOT"
if [ ! -f "$ROOT/data/PG_VERSION" ]; then
	$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
	# the settings of test/automation_setup/postgresql.conf that the expected outputs depend on
	cat >> "$ROOT/data/postgresql.conf" <<CONF
listen_addresses = ''
shared_buffers = 1GB
work_mem = 1GB
max_worker_processes = 100
max_wal_size = 4GB
log_timezone = 'Asia/Tokyo'
timezone = 'Asia/Tokyo'
datestyle = 'iso, mdy'
lc_messages = 'en_US.UTF-8'
lc_monetary = 'en_US.UTF-8'
lc_numeric = 'en_US.UTF-8'
lc_time = 'en_US.UTF-8'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
CONF
fi
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w -o "-p $PORT -k $ROOT" start >/dev/null 2>&1 || { echo "start failed (see $ROOT/log)"; exit 1; }
n=0; while [ $n -lt 60 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
if [ $n -ge 60 ]; then echo "GPU service is not ready"; stop; exit 1; fi

if [ "${HOLD:-0}" = 1 ]; then
	# interactive session: keep the cluster (and, under locked_test.sh, the test lock) until $ROOT/release exists
	rm -f "$ROOT/release"; echo "cluster up: psql -h $ROOT -p $PORT -U postgres; touch $ROOT/release to stop"
	while [ ! -f "$ROOT/release" ]; do sleep 2; done
	stop; exit 0
fi
if [ -n "$SCHEDULE" ]; then
	PGHOST=$ROOT PGPORT=$PORT PGUSER=postgres make -C "$HERE" PG_CONFIG="$PGBIN/pg_config" installcheck REGRESS="--schedule=$SCHEDULE"
else
	PGHOST=$ROOT PGPORT=$PORT PGUSER=postgres make -C "$HERE" PG_CONFIG="$PGBIN/pg_config" "$TARGET"
fi
rc=$?
[ "${KEEP_RUNNING:-0}" = 1 ] || stop
exit $rc
