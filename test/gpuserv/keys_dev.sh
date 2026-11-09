#!/bin/sh
# Quick development run of the "string keys" tests (keys_tests.inc) on a small data
# set, without the rest of scan_test.sh. Same cluster setup and helper functions as scan_test.sh
# (kept small and copied here), so run it through locked_test.sh:
#
#   sh test/gpuserv/locked_test.sh "$PWD" keys_dev.sh
#
# ADHOC=<file> runs the SQL file (psql -f) against the loaded data first (SKIP_TESTS=1: instead of the tests); NROWS (default 300000) sizes the big tables.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-${PGSTROM_TEST_DATA:-/Volumes/CrucialX9/Projects/pg-strom/test-data}/a6-keys/keys_dev}
PORT=${PORT:-54425}
NROWS=${NROWS:-60000}
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

# string-key tables: tko (outer) and tki (inner); keys of length 0..30, multibyte, NULLs, trailing blanks, a
# compressible 5 kB value and an out-of-line (STORAGE EXTERNAL) one
sql "create table tko as select i,
	case when i % 23 = 0 then null else repeat(chr(97 + i % 26), (i % 31)::int) end as k,
	case when i % 29 = 0 then null when i % 6 = 0 then repeat(chr(955 + i % 5), (i % 12)::int + 1) else repeat(chr(65 + i % 26), (i % 9)::int) end::varchar(40) as kv,
	case when i % 19 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) end::char(14) as kb,
	(i % 50)::int4 as n,
	case when i % 40 = 0 then repeat(chr(97 + i % 4), 5000 + (i % 3)) end as big,
	case when i % 40 = 0 then repeat(md5((i % 6)::text), 160) end as ext,
	i::int8 as pay
	from generate_series(1, $NROWS) i"
sql "alter table tko alter column ext set storage external"
sql "update tko set ext = ext where i % 1 = 0"
sql "create table tki as select i,
	case when i % 17 = 0 then null else repeat(chr(97 + i % 26), (i % 31)::int) end as k,
	case when i % 13 = 0 then null when i % 4 = 0 then repeat(chr(955 + i % 5), (i % 12)::int + 1) else repeat(chr(65 + i % 26), (i % 9)::int) end::varchar(40) as kv,
	case when i % 11 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) || repeat(' ', (i % 3)::int) end::char(20) as kb,
	(i % 50)::int4 as n,
	repeat(chr(97 + i % 4), 5000 + (i % 3)) as big,
	repeat(md5((i % 6)::text), 160) as ext,
	(i * 3)::int4 as grp
	from generate_series(1, 4000) i"
sql "alter table tki alter column ext set storage external"
sql "update tki set ext = ext where i % 1 = 0"
sql "insert into tki select i + 5000, k, kv, kb, n, big, ext, grp from tki where i % 10 = 0"
sql "vacuum analyze tko"; sql "vacuum analyze tki"
# bool keys: tbo (3000 rows) joined to the 4 rows of tbi
sql "create table tbo as select i, case when i % 13 = 0 then null else (i % 3 = 0) end as b, (i % 5)::int4 as n, (i % 7)::int4 as m, (i % 4)::int4 as o, i::int8 as pay from generate_series(1, 3000) i"
sql "create table tbi as select * from (values (true, 10), (false, 20), (null::bool, 30), (true, 40)) v(b, w)"
sql "analyze tbo"; sql "analyze tbi"
# GROUP BY tables: tkm has 100 distinct keys when it is analyzed, then 100,000 more that the planner does not know (the group
# table is sized for the estimate and overflows); tkb is big enough for several tasks
sql "create table tkm as select md5((i % 100)::text) as k, (i % 7)::int4 as n, i::int8 as v from generate_series(1, 20000) i"
sql "analyze tkm"
sql "insert into tkm select md5((100 + i)::text), (i % 7)::int4, i::int8 from generate_series(1, 100000) i"
sql "create table tkb as select 'key' || (i % 3000) as k, (i % 11)::int4 as n, i::int8 as v, repeat('x', 40) as pad from generate_series(1, ${KEYS_BIG:-2000000}) i"
sql "vacuum analyze tkb"

# the same keys as an Arrow file, a Parquet file and a GPU-Cache (tkoa / tkop / tkog: the first 30000 rows of tko)
KDIR=$ROOT/arrow; mkdir -p "$KDIR"
make -C "${0%/*}/../../arrow-tools" PG_CONFIG="$PGBIN/pg_config" pg2arrow >"$ROOT/pg2arrow.log" 2>&1
check "pg2arrow builds" $?
P2A="${0%/*}/../../arrow-tools/pg2arrow"
KCOLS="i int4, k text, kv varchar(40), kb char(14), n int4, pay int8"
KSEL="i, k, kv, kb, n, pay"
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tko where i <= 30000 order by i" -s 2MB -o "$KDIR/tko.arrow" >"$ROOT/p2a1.log" 2>&1
check "pg2arrow writes the Arrow file" $?
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tko where i <= 30000 order by i" -s 2MB -q -o "$KDIR/tko.parquet" >"$ROOT/p2a2.log" 2>&1
check "pg2arrow writes the Parquet file" $?
sql "create foreign table tkoa ($KCOLS) server arrow_fdw options (file '$KDIR/tko.arrow')"
sql "create foreign table tkop ($KCOLS) server arrow_fdw options (file '$KDIR/tko.parquet')"
sql "create table tkog as select $KSEL from tko where i <= 30000"
sql "vacuum analyze tkog"
sql "create trigger gc_sync after insert or update or delete on tkog for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=90000')"
sql "analyze tkoa"; sql "analyze tkop"

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

# ADHOC=<file>: run that SQL first (SKIP_TESTS=1: and nothing else); TRACE=1: service trace; JAGG_FILTER=<egrep>:
# only the checks whose label matches
if [ -n "$ADHOC" ]; then
	$PSQL -d scantest -f "$ADHOC" 2>&1
	[ -n "$TRACE" ] && LC_ALL=C grep -a "pgstrom-gpuserv: .*task" "$ROOT/log" | tail -${TRACE_LINES:-40}
	[ -n "$SKIP_TESTS" ] && exit 0
fi
. "${0%/*}/keys_tests.inc"
echo
if [ $fail -eq 0 ]; then echo "KEYS DEV: ALL PASSED"; else echo "KEYS DEV: $fail FAILED"; exit 1; fi
