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

# tmo: strings of length 0..30 (empty strings, one group that has only '' and NULL, one group that is all NULL), multi-byte values whose
# UTF-8 byte order differs from their code point order of a locale, equal values (ties), trailing blanks, a compressible 5 kB value and an
# out-of-line (STORAGE EXTERNAL) one. char(10) columns pad every value to the same length, so ties are identical.
sql "create table tmo as select i,
	(i % 12)::int4 as g, (i % 3000)::int4 as gm, (i % 4)::int4 as g2,
	case when i % 12 = 11 then null when i % 12 = 10 then (case when i % 5 = 0 then '' end)
	     when i % 23 = 0 then null else repeat(chr(97 + i % 26), (i % 31)::int) end as t,
	case when i % 12 = 11 then null when i % 29 = 0 then null when i % 6 = 0 then repeat(chr(955 + i % 5), (i % 12)::int + 1)
	     when i % 7 = 0 then repeat(chr(8364), (i % 3)::int + 1) else repeat(chr(65 + i % 26), (i % 9)::int) end::varchar(40) as v,
	case when i % 12 = 11 then null when i % 19 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) end::char(10) as w,
	case when i % 12 = 11 then null when i % 19 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) || repeat(' ', (i % 3)::int) end::bpchar as wn,
	case when i % 40 = 0 then repeat(chr(97 + i % 4), 5000 + (i % 3)) end as big,
	case when i % 40 = 0 then repeat(md5((i % 6)::text), 160) end as ext,
	(i % 50)::int4 as n, i::int8 as pay from generate_series(1, $NROWS) i"
sql "alter table tmo alter column ext set storage external"
sql "update tmo set ext = ext where i % 1 = 0"
sql "vacuum analyze tmo"
# one row, and an empty table
sql "create table tm1 as select 'only'::text as t, 'one'::varchar(10) as v, 'x'::char(4) as w, 1::int4 as g"
sql "create table tm0 (t text, v varchar(10), w char(4), g int4)"
sql "analyze tm1"; sql "analyze tm0"

# the same strings as an Arrow file, a Parquet file and a GPU-Cache (tmoa / tmop / tmog: the first 30000 rows of tmo)
KDIR=$ROOT/arrow; mkdir -p "$KDIR"
make -C "${0%/*}/../../arrow-tools" PG_CONFIG="$PGBIN/pg_config" pg2arrow >"$ROOT/pg2arrow.log" 2>&1
check "pg2arrow builds" $?
P2A="${0%/*}/../../arrow-tools/pg2arrow"
KCOLS="i int4, g int4, gm int4, g2 int4, t text, v varchar(40), w char(10), n int4, pay int8"
KSEL="i, g, gm, g2, t, v, w, n, pay"
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tmo where i <= 30000 order by i" -s 2MB -o "$KDIR/tmo.arrow" >"$ROOT/p2a1.log" 2>&1
check "pg2arrow writes the Arrow file" $?
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tmo where i <= 30000 order by i" -s 2MB -q -o "$KDIR/tmo.parquet" >"$ROOT/p2a2.log" 2>&1
check "pg2arrow writes the Parquet file" $?
sql "create foreign table tmoa ($KCOLS) server arrow_fdw options (file '$KDIR/tmo.arrow')"
sql "create foreign table tmop ($KCOLS) server arrow_fdw options (file '$KDIR/tmo.parquet')"
sql "create table tmog as select $KSEL from tmo where i <= 30000"
sql "vacuum analyze tmog"
sql "create trigger gc_sync after insert or update or delete on tmog for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=90000')"
sql "analyze tmoa"; sql "analyze tmop"

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
echo
if [ $fail -eq 0 ]; then echo "MINMAXTEXT DEV: ALL PASSED"; else echo "MINMAXTEXT DEV: $fail FAILED"; exit 1; fi
