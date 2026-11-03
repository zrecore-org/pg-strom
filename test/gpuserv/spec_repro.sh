#!/bin/sh
# Development helper: a small cluster in check mode, the SQL file named by ADHOC (default /tmp/sp_adhoc.sql) run
# against a database with the tables s1 (integers, text), nm1 (numeric), ty1 (dates) and the dimension sd, then
# the specializer's lines of the service log. Run through locked_test.sh:
#   sh test/gpuserv/locked_test.sh "$PWD" spec_repro.sh
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/sp_repro}
PORT=${PORT:-54413}
ADHOC=${ADHOC:-/tmp/sp_adhoc.sql}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -q"
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || exit 1
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
max_worker_processes = 24
CONF
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_SPECIALIZE=${PGSTROM_GPUSERV_SPECIALIZE:-check}
export PGSTROM_GPUSERV_CACHE_DIR=$ROOT/speccache
export PGSTROM_GPUSERV_SPEC_LOG=1
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || exit 1
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -At -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database t" >/dev/null
$PSQL -At -d t -c "create extension pg_strom" >/dev/null
$PSQL -At -d t >/dev/null <<SQL
create table nm1 as select i::int4 as i, (i % 1000)::int4 as k, (i::int8 * 1000)::int8 as k8,
	case when i % 41 = 0 then null when i % 101 = 0 then 0 else ((i::int8 * 7919) % 2000000 - 1000000)::numeric / 100 end::numeric(10,2) as n1,
	case when i % 43 = 0 then null else ((i::int8 * 1000003) % 1000000000000 - 500000000000)::numeric / 1000000 end::numeric(18,6) as n2
	from generate_series(1, 300000) i;
create table s1 as select i, ((i % 20000) - 10000)::int2 as a, case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end as b,
	((i::int8 * 1000003) % 1000000007 - 500000000) as c from generate_series(1, 300000) i;
create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp, i::int8 as k8 from generate_series(1, 20000) i;
insert into sd select k, grp + 100, k8 from sd where k % 10 = 0;
create table ty1 as select i, date '1990-01-01' + ((i * 37) % 12000) as dt from generate_series(1, 300000) i;
vacuum analyze nm1; vacuum analyze s1; analyze sd; vacuum analyze ty1;
SQL
$PSQL -d t -c "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0;" -f "$ADHOC" 2>&1 | head -${OUT_LINES:-60}
echo "--- specializer log"
grep "specialize" "$ROOT/log" | tail -${LOG_LINES:-40}
