#!/bin/sh
# Benchmark of GROUP BY with many groups (A3: 1k / 10k / 100k / 1M groups, uniform and skewed keys), GPU vs 8-worker
# PostgreSQL. Run through the wrapper with a ROOT on the local drive (the cluster is kept between runs):
#
#   ROOT=/tmp/a3_groups PORT=54423 sh test/gpuserv/locked_test.sh "$PWD" groups_bench.sh
#
# Table gt(id int8, v int4 (0..999), w int4 (0..9999), k1k, k10k, k100k, k1m int4, kskew int4 (skewed, 100k values),
# kw int8 (1M groups, values beyond 32 bits), k2 / k2b int4 (two keys, 1,000 x 100 groups)), ROWS rows (default 16.8M).
# Every query runs three times on the GPU and on 8-worker PostgreSQL (EXPLAIN ANALYZE time, best of three); the service
# traces the kernel, merge and overflow time per task. The GPU result must equal the CPU's (md5).
#   LABEL_RE='1k'     only the queries whose label matches        ONLY='<sql>'   just that query
#   NOCPU=1           skip the PostgreSQL measurement             FRESH=1        rebuild the cluster and table
#   SPEC=default      PGSTROM_GPUSERV_SPECIALIZE for the service  ROWS=<n>
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/a3_groups}
PORT=${PORT:-54423}
ROWS=${ROWS:-16800000}
SPEC=${SPEC:-default}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
sql() { $PSQL -d bench -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
[ -n "$FRESH" ] && rm -rf "$ROOT"
mkdir -p "$ROOT"
if [ ! -d "$ROOT/data" ]; then
	$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
fi
grep -q "^# groups_bench" "$ROOT/data/postgresql.conf" || cat >> "$ROOT/data/postgresql.conf" <<CONF
# groups_bench
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
max_worker_processes = 24
max_parallel_workers = 16
shared_buffers = 2GB
CONF
grep -v "^pg_strom.gpuserv_path" "$ROOT/data/postgresql.conf" > "$ROOT/conf.tmp" && cp "$ROOT/conf.tmp" "$ROOT/data/postgresql.conf"
echo "pg_strom.gpuserv_path = '$GPUSERV_BIN'" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_TRACE=1
if [ "$SPEC" = default ]; then unset PGSTROM_GPUSERV_SPECIALIZE; else export PGSTROM_GPUSERV_SPECIALIZE=$SPEC; fi
: > "$ROOT/log"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; cat "$ROOT/log"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 40 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }
$PSQL -d postgres -c "select 1 from pg_database where datname = 'bench'" | grep -q 1 || $PSQL -d postgres -c "create database bench" >/dev/null
sql "create extension if not exists pg_strom" >/dev/null 2>&1
if [ "$(sql "select count(*) from pg_class where relname = 'gt'")" = 0 ]; then
	echo "building the data set ($ROWS rows) ..."
	sql "create table gt as select i::int8 as id, ((i::int8 * 7919) % 1000)::int4 as v, ((i::int8 * 104729) % 10000)::int4 as w,
		((i::int8 * 97) % 1000)::int4 as k1k, ((i::int8 * 7919) % 10000)::int4 as k10k,
		((i::int8 * 104729) % 100000)::int4 as k100k, ((i::int8 * 15485863) % 1000000)::int4 as k1m,
		floor(100000 * power(((i::int8 * 2654435761) % 4294967296) / 4294967296.0, 4))::int4 as kskew,
		(((i::int8 * 15485863) % 1000000) * 7919 + 5000000000)::int8 as kw,
		((i::int8 * 31) % 1000)::int4 as k2, ((i::int8 * 7) % 100)::int4 as k2b
		from generate_series(1, $ROWS) i"
	[ "$(sql "select count(*) from pg_class where relname = 'gt'")" = 1 ] || { echo "data set creation failed"; exit 1; }
	sql "vacuum analyze gt"
fi
echo "=== service: $(sql "select state from pgstrom.gpu_service_state()"), specialize=$SPEC, table $(sql "select pg_size_pretty(pg_table_size('gt'))")"
GPUSET="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.gpupreagg_max_groups=100000000; set pg_strom.gpupreagg_max_groups_heap=100000000;"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
bench() { # label query
	label=$1; qry=$2
	[ -n "$LABEL_RE" ] && ! echo "$label" | grep -Eq "$LABEL_RE" && return
	plan=$($PSQL -d bench -c "$GPUSET explain $qry" | head -3 | tr '\n' '|' | cut -c1-110)
	before=$(wc -l < "$ROOT/log")
	gt=""
	for r in 1 2 3; do
		ms=$($PSQL -d bench -c "$GPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
		gt="$gt $ms"
	done
	kt=$(tail -n +$((before+1)) "$ROOT/log" | grep "pgstrom-gpuserv: .*task:" | sed -n 's/.* gpu \([0-9.]*\) ms.*/\1/p' | sort -n | awk '{a[NR]=$1} END {if (NR) printf "%.1f", a[int((NR+1)/2)]; else printf "-"}')
	ntask=$(tail -n +$((before+1)) "$ROOT/log" | grep -c "pgstrom-gpuserv: .*task:")
	ct=""
	if [ -z "$NOCPU" ]; then
		for r in 1 2 3; do
			ms=$($PSQL -d bench -c "$CPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
			ct="$ct $ms"
		done
	fi
	rg=$($PSQL -d bench -c "$GPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" | tail -1)
	rc=$($PSQL -d bench -c "$CPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" | tail -1)
	[ "$rg" = "$rc" ] && same=same || same="RESULT MISMATCH"
	gb=$(echo $gt | tr ' ' '\n' | sort -n | head -1); cb=$(echo $ct | tr ' ' '\n' | sort -n | head -1)
	printf "%-22s gpu %8s ms (runs:%s)  kernel %6s ms/task (%s tasks)  cpu8 %8s ms (runs:%s)\n" "$label" "$gb" "$gt" "$kt" "$ntask" "${cb:--}" "$ct"
	echo "      [$plan] result: $same"
	tail -n +$((before+1)) "$ROOT/log" | grep "pgstrom-gpuserv: .*task:" | tail -2 | cut -c1-200
}
if [ -n "$ONLY" ]; then bench "only" "$ONLY"; else
	bench "scan only (count)"   "select count(*), sum(w), max(v) from gt"
	bench "1k groups"           "select k1k, count(*), sum(w), max(v) from gt group by k1k"
	bench "10k groups"          "select k10k, count(*), sum(w), max(v) from gt group by k10k"
	bench "100k groups"         "select k100k, count(*), sum(w), max(v) from gt group by k100k"
	bench "1M groups"           "select k1m, count(*), sum(w), max(v) from gt group by k1m"
	bench "skewed 100k"         "select kskew, count(*), sum(w), max(v) from gt group by kskew"
	bench "int8 key 1M"         "select kw, count(*), sum(w), max(v) from gt group by kw"
	bench "two keys 100k"       "select k2, k2b, count(*), sum(w), max(v) from gt group by k2, k2b"
fi
