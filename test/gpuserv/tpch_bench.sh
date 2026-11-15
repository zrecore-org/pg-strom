#!/bin/sh
# TPC-H (22 queries, test/tpch/queries) on a throwaway cluster: PostgreSQL with 8 parallel workers against PG-Strom with the planner's
# own choice (the GPU nodes it picks, the rest on the CPU, also with 8 parallel workers: GPU_WORKERS=<n>). The A18 baseline: which operations dominate, where the GPU helps. Run it
# through the wrapper (the cluster is kept between runs):
#
#   GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/locked_test.sh "$PWD" tpch_bench.sh
#
# Data: dbgen output (the *.tbl files with a trailing '|' per line; build test/tpch with a C compiler, e.g.
#   make CC=cc  -> dbgen-tpch, then `dbgen-tpch -s 1 -f` in an empty directory with dists.dss)
#   TBL=<directory of the .tbl files> (default <checkout>/test-data/tpch_sf1)   ROOT=<cluster dir> (default test-data/tpch_bench)
#   FRESH=1   rebuild the cluster and tables         LABEL_RE='^(01|06|18)$'  only these queries (the file number)
#   GPUSET_EXTRA='set pg_strom.gpu_scan_tuple_cost=0.01;'  more settings for the GPU runs (cost calibration)
#   NOCPU=1   skip the PostgreSQL runs               RUNS=3 (default 2)   timeout per query QTIMEOUT=300 s
# One line per query: CPU time (best of RUNS), GPU time (best of RUNS), the number of Gpu nodes in the GPU plan, and whether the output
# of the two engines is identical (md5 of the rows; a float formatted differently shows as DIFF and is judged by hand).
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=$(cd "$HERE/../.." && pwd)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
TBL=${TBL:-$TREE/test-data/tpch_sf1}
ROOT=${ROOT:-$TREE/test-data/tpch_bench}
PORT=${PORT:-54433}
RUNS=${RUNS:-2}
QTIMEOUT=${QTIMEOUT:-300}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
[ -n "$FRESH" ] && rm -rf "$ROOT"
mkdir -p "$ROOT"
[ -d "$ROOT/data" ] || $PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
grep -q "^# tpch_bench" "$ROOT/data/postgresql.conf" || cat >> "$ROOT/data/postgresql.conf" <<CONF
# tpch_bench
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
max_worker_processes = 24
max_parallel_workers = 16
shared_buffers = 2GB
work_mem = 256MB
fsync = off
full_page_writes = off
CONF
grep -v "^pg_strom.gpuserv_path" "$ROOT/data/postgresql.conf" > "$ROOT/conf.tmp" && cp "$ROOT/conf.tmp" "$ROOT/data/postgresql.conf"
echo "pg_strom.gpuserv_path = '$GPUSERV_BIN'" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
: > "$ROOT/log"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; cat "$ROOT/log"; exit 1; }
n=0; while [ $n -lt 600 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 600 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }
$PSQL -d postgres -c "select 1 from pg_database where datname = 'tpch'" | grep -q 1 || $PSQL -d postgres -c "create database tpch" >/dev/null
sql() { $PSQL -d tpch -c "$1"; }
sql "create extension if not exists pg_strom" >/dev/null 2>&1
if [ "$(sql "select count(*) from pg_class where relname = 'lineitem'")" = 0 ]; then
	echo "loading $TBL ..."
	$PSQL -d tpch -f "$TREE/test/tpch/dss.ddl" >/dev/null 2>&1
	for t in region nation supplier customer part partsupp orders lineitem; do
		sed 's/|$//' "$TBL/$t.tbl" | $PSQL -d tpch -c "copy $t from stdin with (format text, delimiter '|')" || { echo "load of $t failed"; exit 1; }
	done
	sql "alter table region add primary key (r_regionkey); alter table nation add primary key (n_nationkey);
		alter table part add primary key (p_partkey); alter table supplier add primary key (s_suppkey);
		alter table partsupp add primary key (ps_partkey, ps_suppkey); alter table customer add primary key (c_custkey);
		alter table orders add primary key (o_orderkey); alter table lineitem add primary key (l_orderkey, l_linenumber)"
	sql "vacuum analyze"
fi
echo "=== service: $(sql "select state from pgstrom.gpu_service_state()"), lineitem $(sql "select count(*) from lineitem") rows, $(sql "select pg_size_pretty(pg_database_size('tpch'))")"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8;"
GPUSET="set pg_strom.enabled=on; set max_parallel_workers_per_gather=${GPU_WORKERS:-8}; $GPUSET_EXTRA"
# the time of a query file: the largest "Time:" of its statements (Q15 creates and drops a view)
run_ms() { # settings file
	(echo "$1"; printf '%s\n' '\timing on'; cat "$2") | timeout $QTIMEOUT $PSQL -d tpch 2>&1 | grep '^Time:' | sed 's/Time: \([0-9.]*\) ms.*/\1/' | sort -n | tail -1
}
best() { sort -n | head -1; }
printf "%-4s %-34s %10s %10s %6s  %s\n" "q" "title" "cpu8 ms" "gpu ms" "nodes" "result"
for f in "$TREE"/test/tpch/queries/tpch-*.sql; do
	q=$(basename "$f" .sql | sed 's/tpch-//')
	[ -n "$LABEL_RE" ] && ! echo "$q" | grep -Eq "$LABEL_RE" && continue
	title=$(sed -n '2p' "$f" | sed 's/^-- *TPC-H\/TPC-R *//' | cut -c1-34)
	nodes=$( (echo "$GPUSET"; echo 'explain'; cat "$f") | timeout 60 $PSQL -d tpch 2>/dev/null | grep -c "Custom Scan (Gpu")
	ct=""; gt=""
	if [ -z "$NOCPU" ]; then for r in $(seq $RUNS); do ct="$ct $(run_ms "$CPUSET" "$f")"; done; fi
	for r in $(seq $RUNS); do gt="$gt $(run_ms "$GPUSET" "$f")"; done
	rc=$( (echo "$CPUSET"; cat "$f") | timeout $QTIMEOUT $PSQL -d tpch 2>&1 | md5 -q)
	rg=$( (echo "$GPUSET"; cat "$f") | timeout $QTIMEOUT $PSQL -d tpch 2>&1 | md5 -q)
	[ "$rc" = "$rg" ] && same=same || same=DIFF
	printf "%-4s %-34s %10s %10s %6s  %s\n" "$q" "$title" "$(echo $ct | tr ' ' '\n' | best)" "$(echo $gt | tr ' ' '\n' | best)" "$nodes" "$same"
done
