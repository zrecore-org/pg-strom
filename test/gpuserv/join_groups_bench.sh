#!/bin/sh
# Benchmark of GROUP BY with many groups OVER A JOIN (the join slice of A3): 1k / 10k / 100k / 1M groups, the key from the outer
# or from the inner table, GPU vs 8-worker PostgreSQL. Run it through the wrapper (the cluster is kept between runs):
#
#   GPUSERV_BIN=<installed pgstrom-gpuserv> ROOT=/tmp/a3_joingroups PORT=54424 sh test/gpuserv/locked_test.sh "$PWD" join_groups_bench.sh
#
# jf(id int8, dk int4 (0..999999, the key of jd), v int4 (0..999), w int4 (0..9999), k1k, k10k, k100k, k1m int4), ROWS rows
# (default 8M); jd(id int4, gi1k, gi10k, gi100k int4, gi1m int4 (= id), flag int4), 1,000,000 rows. Every join row finds one jd row.
# The GPU runs with pg_strom.gpu_error_fallback=off (a refused or failed program fails the query instead of running on the CPU),
# the planner caps raised (pg_strom.gpupreagg_max_groups) and the service traces each task: "kernel" is the GPU time, "merge" the
# host merge of the workgroup tables, "ovf" the number of joined rows that found no slot (the host aggregates them one by one).
# The GPU result must equal the CPU's (md5).
#   LABEL_RE='1k'   only the queries whose label matches     ONLY='<sql>'  just that query
#   NOCPU=1         skip the PostgreSQL measurement          FRESH=1        rebuild the cluster and tables     ROWS=<n>
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/a3_joingroups}
PORT=${PORT:-54424}
ROWS=${ROWS:-8000000}
CAP=${CAP:-100000000}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
sql() { $PSQL -d bench -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
[ -n "$FRESH" ] && rm -rf "$ROOT"
mkdir -p "$ROOT"
if [ ! -d "$ROOT/data" ]; then
	$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
fi
grep -q "^# join_groups_bench" "$ROOT/data/postgresql.conf" || cat >> "$ROOT/data/postgresql.conf" <<CONF
# join_groups_bench
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
: > "$ROOT/log"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; cat "$ROOT/log"; exit 1; }
n=0; while [ $n -lt 600 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 600 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }
$PSQL -d postgres -c "select 1 from pg_database where datname = 'bench'" | grep -q 1 || $PSQL -d postgres -c "create database bench" >/dev/null
sql "create extension if not exists pg_strom" >/dev/null 2>&1
if [ "$(sql "select count(*) from pg_class where relname = 'jf'")" = 0 ]; then
	echo "building the data set ($ROWS rows) ..."
	sql "create table jd as select i::int4 as id, (i % 1000)::int4 as gi1k, (i % 10000)::int4 as gi10k, (i % 100000)::int4 as gi100k,
		i::int4 as gi1m, (i % 7)::int4 as flag from generate_series(0, 999999) i"
	sql "create table jf as select i::int8 as id, ((i::int8 * 7919) % 1000000)::int4 as dk, ((i::int8 * 7919) % 1000)::int4 as v,
		((i::int8 * 104729) % 10000)::int4 as w, ((i::int8 * 97) % 1000)::int4 as k1k, ((i::int8 * 7919) % 10000)::int4 as k10k,
		((i::int8 * 104729) % 100000)::int4 as k100k, ((i::int8 * 15485863) % 1000000)::int4 as k1m
		from generate_series(1, $ROWS) i"
	[ "$(sql "select count(*) from pg_class where relname = 'jf'")" = 1 ] || { echo "data set creation failed"; exit 1; }
	sql "alter table jd add primary key (id)"
	sql "vacuum analyze jf"; sql "vacuum analyze jd"
fi
echo "=== service: $(sql "select state from pgstrom.gpu_service_state()"), jf $(sql "select pg_size_pretty(pg_table_size('jf'))"), jd $(sql "select pg_size_pretty(pg_table_size('jd'))")"
GPUSET="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.gpu_error_fallback=off; set pg_strom.gpupreagg_max_groups=$CAP; set pg_strom.gpupreagg_max_groups_heap=$CAP;"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
med() { sort -n | awk '{a[NR]=$1} END {if (NR) printf "%.1f", a[int((NR+1)/2)]; else printf "-"}'; }
bench() { # label query
	label=$1; qry=$2
	[ -n "$LABEL_RE" ] && ! echo "$label" | grep -Eq "$LABEL_RE" && return
	plan=$($PSQL -d bench -c "$GPUSET explain $qry" | grep -E "Custom Scan|GpuPreAgg|GpuJoin|HashAggregate|GroupAggregate" | head -3 | tr '\n' '|' | cut -c1-120)
	before=$(wc -l < "$ROOT/log")
	gt=""
	for r in 1 2 3; do
		ms=$($PSQL -d bench -c "$GPUSET explain (analyze, timing off) $qry" 2>&1 | grep -E "Execution Time|ERROR" | sed 's/.*Execution Time: //; s/ ms//')
		gt="$gt $ms"
	done
	new=$(tail -n +$((before+1)) "$ROOT/log" | grep "pgstrom-gpuserv: join.*task:")
	ntask=$(echo "$new" | grep -c "task:")
	kt=$(echo "$new" | sed -n 's/.* gpu \([0-9.]*\) ms.*/\1/p' | med)
	mt=$(echo "$new" | sed -n 's/.* merge \([0-9.]*\) ms.*/\1/p' | med)
	ovf=$(echo "$new" | sed -n 's/.*overflow rows \([0-9]*\).*/\1/p' | awk '{s+=$1} END {print s+0}')
	hidden=$(tail -n +$((before+1)) "$ROOT/log" | grep -c -E "task failed:|GPU execution failed|VK_ERROR|final failed:")
	ct=""
	if [ -z "$NOCPU" ]; then
		for r in 1 2 3; do
			ms=$($PSQL -d bench -c "$CPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
			ct="$ct $ms"
		done
	fi
	rg=$($PSQL -d bench -c "$GPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" 2>&1 | tail -1)
	rc=$($PSQL -d bench -c "$CPUSET select md5(string_agg(x::text, ',' order by x::text)) from ($qry) x" | tail -1)
	[ "$rg" = "$rc" ] && same=same || same="RESULT MISMATCH ($rg)"
	gb=$(echo $gt | tr ' ' '\n' | sort -n | head -1); cb=$(echo $ct | tr ' ' '\n' | sort -n | head -1)
	printf "%-26s gpu %8s ms (runs:%s)  kernel %6s ms/task merge %6s ms/task (%s tasks) ovf %s rows  cpu8 %8s ms (runs:%s) failures %s\n" \
		"$label" "$gb" "$gt" "$kt" "$mt" "$ntask" "$ovf" "${cb:--}" "$ct" "$hidden"
	echo "      [$plan] result: $same"
}
if [ -n "$ONLY" ]; then bench "only" "$ONLY"; else
	bench "join count (no groups)"     "select count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id"
	bench "outer key 1k"               "select k1k, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by k1k"
	bench "outer key 10k"              "select k10k, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by k10k"
	bench "outer key 100k"             "select k100k, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by k100k"
	bench "outer key 1M"               "select k1m, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by k1m"
	bench "inner key 1k"               "select jd.gi1k, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by jd.gi1k"
	bench "inner key 100k"             "select jd.gi100k, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by jd.gi100k"
	bench "inner key 1M"               "select jd.gi1m, count(*), sum(w), max(v) from jf join jd on jf.dk = jd.id group by jd.gi1m"
fi
