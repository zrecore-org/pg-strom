#!/bin/sh
# Quick development run of the CPU fallback tests (fallback_tests.inc, A14) on a small data set, without the rest of
# scan_test.sh. Same cluster setup and helpers as scan_test.sh (kept small and copied here), so run it through
# locked_test.sh:
#
#   PORT=54421 sh test/gpuserv/locked_test.sh "$PWD" fallback_dev.sh
#
# ADHOC=<file> runs the SQL file (psql -f) against the loaded data instead of the tests. NROWS (default 120000) sizes
# the Arrow tables, FBN (default 60000) the heap tables.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/fb_dev}
PORT=${PORT:-54421}
NROWS=${NROWS:-120000}
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
[ -z "$KEEP" ] && trap stop EXIT INT TERM
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

sql "create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp,
	case when i % 9 = 0 then null else (i::int8 * 3 - 5) end as pay, repeat(chr(65 + i % 26), (i % 30)::int) as txt, i::int8 as k8
	from generate_series(1, 20000) i"
sql "analyze sd"

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}

# the Arrow and Parquet tables of scan_test.sh
ARROWDIR="$ROOT/arrow"; mkdir -p "$ARROWDIR"
ARROW_CFLAGS=$(pkg-config --cflags arrow parquet 2>/dev/null); ARROW_LIBS=$(pkg-config --libs arrow parquet 2>/dev/null)
if [ -z "$ARROW_LIBS" ] && command -v brew >/dev/null 2>&1; then
	AP=$(brew --prefix apache-arrow 2>/dev/null); ARROW_CFLAGS="-I$AP/include"; ARROW_LIBS="-L$AP/lib -lparquet -larrow -Wl,-rpath,$AP/lib"
fi
c++ -std=c++20 -O2 -DWITH_PARQUET "$(dirname "$0")/arrow_gen.cpp" -o "$ARROWDIR/arrow_gen" $ARROW_CFLAGS $ARROW_LIBS >"$ARROWDIR/gen.log" 2>&1
check "the Arrow test data generator builds" $?
"$ARROWDIR/arrow_gen" "$ARROWDIR/a1.arrow" "$NROWS" 20000 1
"$ARROWDIR/arrow_gen" "$ARROWDIR/p1.parquet" "$NROWS" 20000 1
FTDEF="(i int8, a int2, b int4, c int8, d float4, e bool, f text, dt date, ts timestamp)"
sql "create foreign table a1 $FTDEF server arrow_fdw options (file '$ARROWDIR/a1.arrow')"
sql "create foreign table p1 $FTDEF server arrow_fdw options (file '$ARROWDIR/p1.parquet')"
sql "analyze a1"; sql "analyze p1"

[ -z "$ADHOC" ] && [ -f /tmp/fb_adhoc.sql ] && ADHOC=/tmp/fb_adhoc.sql
if [ -n "$ADHOC" ]; then
	FB_SETUP_ONLY=1
	$PSQL -d scantest -f "$ADHOC" 2>&1
	grep -i "WARNING\|ERROR\|runs the rest" "$ROOT/log" | tail -20
	exit 0
fi
. "${0%/*}/fallback_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "FALLBACK DEV: ALL PASSED"; else echo "FALLBACK DEV: $fail FAILED"; exit 1; fi
