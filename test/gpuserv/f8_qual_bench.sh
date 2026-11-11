#!/bin/sh
# Query-level cost of float8 arithmetic in a qual (A10 part 6, S1): every query with arithmetic next to an arithmetic-free twin of
# about the same selectivity on the same table, so the difference is what a faster float8 arithmetic can remove at most.
# Run through the wrapper (it installs the tree and sets GPUSERV_BIN):
#
#   PORT=54435 sh test/gpuserv/locked_test.sh "$PWD" f8_qual_bench.sh
#
# A throwaway cluster ($ROOT) keeps the tables between runs (FRESH=1 rebuilds):
#   tf(x, y float8 in [0, 1000))   ROWS rows (default 16800000: many tasks, so the specialized kernel)
#   ts(x, y)                       SMALL rows (default 40000: one task, the interpreter)
# Each query runs 3 times (EXPLAIN ANALYZE execution time, best of 3); "kernel" is the median "gpu x ms" of the service's task lines.
# The GPU result must equal the CPU's. SPECS="default off default+nodf off+nodf" runs the service variants one after the other
# (PGSTROM_GPUSERV_SPECIALIZE; "+nodf": PGSTROM_GPUSERV_DF64=off, the exact software binary64 only).
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/Volumes/CrucialX9/Projects/pg-strom/test-data/f8_qual_bench}
PORT=${PORT:-54435}
ROWS=${ROWS:-16800000}
SMALL=${SMALL:-40000}
SPECS=${SPECS:-default}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
sql() { $PSQL -d bench -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
[ -n "$FRESH" ] && rm -rf "$ROOT"
mkdir -p "$ROOT"
[ -d "$ROOT/data" ] || $PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
grep -q "^# f8_qual_bench" "$ROOT/data/postgresql.conf" || cat >> "$ROOT/data/postgresql.conf" <<CONF
# f8_qual_bench
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

start_cluster() { # a specialize mode (default, off, force ...), "+nodf" after it: PGSTROM_GPUSERV_DF64=off (no df64 prefilter)
	stop
	m=${1%+nodf}
	if [ "$m" = "$1" ]; then unset PGSTROM_GPUSERV_DF64; else export PGSTROM_GPUSERV_DF64=off; fi
	if [ "$m" = default ]; then unset PGSTROM_GPUSERV_SPECIALIZE; else export PGSTROM_GPUSERV_SPECIALIZE=$m; fi
	: > "$ROOT/log"
	$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; cat "$ROOT/log"; exit 1; }
	n=0; while [ $n -lt 600 ]; do
		[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
		n=$((n+1)); sleep 1
	done
	[ $n -lt 600 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }
	$PSQL -d postgres -c "select 1 from pg_database where datname = 'bench'" | grep -q 1 || $PSQL -d postgres -c "create database bench" >/dev/null
	sql "create extension if not exists pg_strom" >/dev/null 2>&1
}
GPUSET="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0;"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
bench() { # label query
	label=$1; qry=$2
	[ -n "$LABEL_RE" ] && ! echo "$label" | grep -Eq "$LABEL_RE" && return
	plan=$($PSQL -d bench -c "$GPUSET explain $qry" | head -3 | tr '\n' '|' | cut -c1-90)
	before=$(wc -l < "$ROOT/log")
	gt=""
	for r in 1 2 3; do
		ms=$($PSQL -d bench -c "$GPUSET explain (analyze, timing off) $qry" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
		gt="$gt $ms"
	done
	kt=$(tail -n +$((before+1)) "$ROOT/log" | grep "pgstrom-gpuserv: .*task:" | sed -n 's/.* gpu \([0-9.]*\) ms.*/\1/p' | sort -n | awk '{a[NR]=$1} END {if (NR) printf "%.1f", a[int((NR+1)/2)]; else printf "-"}')
	ntask=$(tail -n +$((before+1)) "$ROOT/log" | grep -c "pgstrom-gpuserv: .*task:")
	rg=$($PSQL -d bench -c "$GPUSET $qry" | tail -1)
	rc=$($PSQL -d bench -c "$CPUSET $qry" | tail -1)
	[ "$rg" = "$rc" ] && same="same ($rg)" || same="RESULT MISMATCH gpu=$rg cpu=$rc"
	gb=$(echo $gt | tr ' ' '\n' | sort -n | head -1)
	printf "%-26s gpu %8s ms (runs:%s)  kernel %6s ms/task (%s tasks)\n" "$label" "$gb" "$gt" "$kt" "$ntask"
	echo "      [$plan] result: $same"
}
run_all() {
	for t in tf ts; do
		echo "--- table $t ($(sql "select count(*) from $t") rows)"
		bench "$t no arithmetic: x<850"    "select count(*) from $t where x < 850"
		bench "$t x*1.7-3.3<850"           "select count(*) from $t where x * 1.7 - 3.3 < 850"
		bench "$t no arithmetic: x<y"      "select count(*) from $t where x < y"
		bench "$t abs(x-y)<50"             "select count(*) from $t where abs(x - y) < 50"
		bench "$t no arithmetic: x<500"    "select count(*) from $t where x < 500"
		bench "$t x*x+y*y<250000"          "select count(*) from $t where x * x + y * y < 250000"
		bench "$t x*a+y*b<c (two terms)"   "select count(*) from $t where x * 0.37 + y * 1.21 < 800"
		bench "$t x^3+y^3<5e8 (5 ops)"     "select count(*) from $t where x * x * x + y * y * y < 500000000"
		bench "$t (x*x+y*y)*.5-x*y*.3"     "select count(*) from $t where (x * x + y * y) * 0.5 - x * y * 0.3 < 100000"
		bench "$t two quals, 8 ops"        "select count(*) from $t where x * 0.37 + y * 1.21 < 800 and abs(x * x - y * y) < 400000 and (x + y) * (x - y) > -300000"
	done
}
first=1
for spec in $SPECS; do
	start_cluster "$spec"
	if [ "$first" = 1 ] && [ "$($PSQL -d bench -c "select count(*) from pg_class where relname = 'tf'")" = 0 ]; then
		echo "building the data sets ..."
		sql "select setseed(0.42)"
		sql "create table tf as select (random() * 1000)::float8 as x, (random() * 1000)::float8 as y from generate_series(1, $ROWS)"
		sql "create table ts as select (random() * 1000)::float8 as x, (random() * 1000)::float8 as y from generate_series(1, $SMALL)"
		sql "vacuum analyze tf"; sql "vacuum analyze ts"
	fi
	echo "=== service: $(sql "select state from pgstrom.gpu_service_state()"), specialize=$spec"
	run_all
	first=""
done
