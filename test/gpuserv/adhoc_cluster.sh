#!/bin/sh
# adhoc_cluster.sh - start a KEPT throwaway cluster (ROOT=..., left by a dev script with a ROOT you set) with the service trace on, run SQL
# (ADHOC=<file>, psql -f, database scantest) and print the service's trace lines (GREP, default "join task"); run through the test lock:
#   ROOT=<dir> PORT=<port> ADHOC=q.sql sh test/gpuserv/locked_test.sh "$PWD" adhoc_cluster.sh
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN}
ROOT=${ROOT:?set ROOT}
PORT=${PORT:?set PORT}
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
sed -i '' "s/^port = .*/port = $PORT/; s#^pg_strom.gpuserv_path = .*#pg_strom.gpuserv_path = '$GPUSERV_BIN'#" "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_TRACE=1
export PGSTROM_GPUSERV_SPECIALIZE=${SPEC:-off}
export PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=${PGSTROM_GPUSERV_PIPELINE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pgstrom}
: > "$ROOT/log"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 60 ]; do
	[ "$($PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q -d scantest -f "$ADHOC" 2>&1
echo "--- service trace"
grep -E "${GREP:-join task}" "$ROOT/log" | sed 's/^.*pgstrom-gpuserv: //' | sort | uniq -c | sort -rn | head -${LINES_MAX:-30}
