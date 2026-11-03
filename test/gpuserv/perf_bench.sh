#!/bin/sh
# Benchmark of the GPU kernels (A6: VM vs specialized kernels, GROUP BY speed). Run through the wrapper:
#
#   sh test/gpuserv/locked_test.sh "$PWD" perf_bench.sh > /tmp/sp_bench.out 2>&1
#
# A throwaway cluster ($ROOT, default /tmp/sp_bench, port 54413) holds a 16.8M-row heap table
#   t(id int8, v int4 (0..999), w int4 (0..9999), f float4, g7 int4, g100 int4, g1000 int4)
# and a 1,000-row dimension d(k, grp (20 values), pay); the data directory is kept between runs
# (FRESH=1 rebuilds it). Every query runs three times per configuration (EXPLAIN ANALYZE execution time,
# best of three is reported, with the runs):
#   gpu    GpuScan / GpuPreAgg / GpuJoin, no parallel workers
#   cpu8   PostgreSQL without PG-Strom, 8 parallel workers (measured in the first variant only)
# The service runs with PGSTROM_GPUSERV_TRACE=1: the kernel time per task ("gpu x ms" of the task lines) is the
# median over the tasks of the 3 runs. The GPU result must equal the CPU's (md5 of the sorted rows).
#
#   SPECS="off force off"   service variants to run one after the other (the cluster is restarted for each); the
#                           value is PGSTROM_GPUSERV_SPECIALIZE ("default": leave it unset). Default: "default".
#   ONLY='<sql>'            benchmark just that query
#   NOCPU=1                 skip the PostgreSQL measurement
#   ROWS=<n>                table size (default 16800000; changes need FRESH=1)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/sp_bench}
PORT=${PORT:-54413}
ROWS=${ROWS:-16800000}
SPECS=${SPECS:-default}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
sql() { $PSQL -d bench -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
[ -n "$FRESH" ] && rm -rf "$ROOT"
mkdir -p "$ROOT"
if [ ! -d "$ROOT/data" ]; then
	$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
fi
grep -q "^# perf_bench" "$ROOT/data/postgresql.conf" || cat >> "$ROOT/data/postgresql.conf" <<CONF
# perf_bench
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
max_worker_processes = 24
max_parallel_workers = 16
shared_buffers = 2GB
CONF
# the service path may change between runs
grep -v "^pg_strom.gpuserv_path" "$ROOT/data/postgresql.conf" > "$ROOT/conf.tmp" && cp "$ROOT/conf.tmp" "$ROOT/data/postgresql.conf"
echo "pg_strom.gpuserv_path = '$GPUSERV_BIN'" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_TRACE=1

start_cluster() { # specialize-mode
	stop
	if [ "$1" = default ]; then unset PGSTROM_GPUSERV_SPECIALIZE; else export PGSTROM_GPUSERV_SPECIALIZE=$1; fi
	: > "$ROOT/log"
	$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; cat "$ROOT/log"; exit 1; }
	n=0; while [ $n -lt 40 ]; do
		[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
		n=$((n+1)); sleep 1
	done
	[ $n -lt 40 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }
	if ! $PSQL -d postgres -c "select 1 from pg_database where datname = 'bench'" | grep -q 1; then
		$PSQL -d postgres -c "create database bench" >/dev/null
	fi
	sql "create extension if not exists pg_strom" >/dev/null 2>&1
}

GPUSET="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.gpupreagg_max_groups=100000;"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
bench() { # label query
	label=$1; qry=$2
	plan=$($PSQL -d bench -c "$GPUSET explain $qry" | head -3 | tr '\n' '|' | cut -c1-110)
	before=$(wc -l < "$ROOT/log")
	gt=""
	for r in 1 2 3; do
		ms=$($PSQL -d bench -c "$GPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
		gt="$gt $ms"
	done
	# kernel ms per task: the median "gpu x ms" of the task lines the 3 runs wrote
	kt=$(tail -n +$((before+1)) "$ROOT/log" | grep "pgstrom-gpuserv: .*task:" | sed -n 's/.* gpu \([0-9.]*\) ms.*/\1/p' | sort -n | awk '{a[NR]=$1} END {if (NR) printf "%.1f", a[int((NR+1)/2)]; else printf "-"}')
	ntask=$(tail -n +$((before+1)) "$ROOT/log" | grep -c "pgstrom-gpuserv: .*task:")
	ct=""
	if [ -z "$NOCPU" ]; then
		for r in 1 2 3; do
			ms=$($PSQL -d bench -c "$CPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
			ct="$ct $ms"
		done
	fi
	# the result must be the CPU's (md5 of the sorted rows)
	rg=$($PSQL -d bench -c "$GPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" | tail -1)
	rc=$($PSQL -d bench -c "$CPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" | tail -1)
	[ "$rg" = "$rc" ] && same=same || same="RESULT MISMATCH"
	gb=$(echo $gt | tr ' ' '\n' | sort -n | head -1); cb=$(echo $ct | tr ' ' '\n' | sort -n | head -1)
	printf "%-24s gpu %8s ms (runs:%s)  kernel %6s ms/task (%s tasks)  cpu8 %8s ms (runs:%s)\n" "$label" "$gb" "$gt" "$kt" "$ntask" "${cb:--}" "$ct"
	echo "      [$plan] result: $same"
}
run_all() {
	if [ -n "$ONLY" ]; then
		bench "only" "$ONLY"
		return
	fi
	bench "agg, no qual"           "select count(*), sum(w), max(v) from t"
	bench "agg, trivial qual"      "select count(*), sum(w), max(v) from t where v + 1 > 3"
	bench "agg, arithmetic quals"  "select count(*), sum(w), max(v) from t where v * 3 + w > 2990 and w % 7 = 3"
	bench "scan, arithmetic"       "select id, v * 2 + w from t where v * 3 + w > 12980 and w < 100"
	bench "group by 7"             "select g7, count(*), sum(w), max(v) from t group by g7"
	bench "group by 7, qual"       "select g7, count(*), sum(w), max(v) from t where v + 1 > 3 group by g7"
	bench "group by 100"           "select g100, count(*), sum(w), max(v) from t group by g100"
	bench "group by 1000"          "select g1000, count(*), sum(w), max(v) from t group by g1000"
	bench "join agg"               "select count(*), sum(t.w), max(d.pay) from t join d on t.v = d.k where t.w < 500"
	bench "join group by 20"       "select d.grp, count(*), sum(t.w), max(d.pay) from t join d on t.v = d.k where t.w < 5000 group by d.grp"
}

first=1
for spec in $SPECS; do
	start_cluster "$spec"
	if [ "$first" = 1 ] && [ "$($PSQL -d bench -c "select count(*) from pg_class where relname = 't'")" = 0 ]; then
		echo "building the data set ($ROWS rows) ..."
		sql "create table t as select i::int8 as id, ((i::int8 * 7919) % 1000)::int4 as v, ((i::int8 * 104729) % 10000)::int4 as w,
			(((i::int8 * 17) % 10000) / 7.0)::float4 as f, (i % 7)::int4 as g7, ((i::int8 * 31) % 100)::int4 as g100,
			((i::int8 * 97) % 1000)::int4 as g1000 from generate_series(1, $ROWS) i"
		[ "$(sql "select count(*) from pg_class where relname = 't'")" = 1 ] || { echo "data set creation failed"; exit 1; }
		sql "vacuum analyze t"
		sql "create table d as select i::int4 as k, (i % 20)::int4 as grp, (i * 3)::int8 as pay from generate_series(0, 999) i"
		sql "analyze d"
	fi
	echo "=== service: $(sql "select state from pgstrom.gpu_service_state()"), specialize=$spec, table $(sql "select pg_size_pretty(pg_table_size('t'))")"
	run_all
	grep "pgstrom-gpuserv: specialize" "$ROOT/log" | head -${SPEC_LOG_LINES:-20}
	first=""
	NOCPU=1
done
