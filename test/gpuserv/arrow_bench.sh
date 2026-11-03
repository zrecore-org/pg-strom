#!/bin/sh
# Benchmark of GpuPreAgg / GpuJoin over a large Arrow file (A5 part 2). Run through locked_test.sh:
#
#   BIGARROW=/tmp/a5_gen/big.arrow sh test/gpuserv/locked_test.sh "$PWD" arrow_bench.sh
#
# BIGARROW is written by arrow_gen (16.8M rows, 9 columns, 2M rows per record batch: `arrow_gen big.arrow
# 16800000 2000000 1`); if the file does not exist it is generated. Every query runs three times in three
# configurations (EXPLAIN ANALYZE execution time, warm page cache):
#   gpu       GpuPreAgg / GpuJoin over the Arrow table (the plan of this milestone)
#   gpuscan   the plan before: the aggregate / join runs on the CPU above a GpuScan
#   cpu8      PostgreSQL without PG-Strom, 8 parallel workers
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/a5b_bench}
PORT=${PORT:-54404}
BIGARROW=${BIGARROW:-/tmp/a5b_bigdata/big.arrow}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
if [ ! -f "$BIGARROW" ]; then
	mkdir -p "$(dirname "$BIGARROW")"
	ARROW_CFLAGS=$(pkg-config --cflags arrow parquet 2>/dev/null); ARROW_LIBS=$(pkg-config --libs arrow parquet 2>/dev/null)
	if [ -z "$ARROW_LIBS" ] && command -v brew >/dev/null 2>&1; then
		AP=$(brew --prefix apache-arrow 2>/dev/null); ARROW_CFLAGS="-I$AP/include"; ARROW_LIBS="-L$AP/lib -lparquet -larrow -Wl,-rpath,$AP/lib"
	fi
	c++ -std=c++20 -O2 "$(dirname "$0")/arrow_gen.cpp" -o "$ROOT/arrow_gen" $ARROW_CFLAGS $ARROW_LIBS || exit 1
	"$ROOT/arrow_gen" "$BIGARROW" 16800000 2000000 1
fi
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
max_worker_processes = 24
max_parallel_workers = 16
CONF
trap stop EXIT INT TERM
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database bench" >/dev/null
sql() { $PSQL -d bench -c "$1"; }
sql "create extension pg_strom" >/dev/null
sql "create foreign table big (i int8, a int2, b int4, c int8, d float4, e bool, f text, dt date, ts timestamp) server arrow_fdw options (file '$BIGARROW')"
sql "analyze big"
# a 4,000-row dimension table: k covers every 25th value of b (about 4% of the rows join)
sql "create table bd as select (i * 25 - 50000)::int4 as k, (i % 17)::int4 as grp, (i * 3)::int8 as pay from generate_series(0, 3999) i"
sql "analyze bd"

GPUSET="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.gpupreagg_max_groups=100000;"
OLDSET="$GPUSET set pg_strom.enable_gpupreagg=off; set pg_strom.enable_gpujoin=off;"
CPUSET="set pg_strom.enabled=off; set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
bench() { # label query
	echo "== $1: $2"
	for cfg in gpu gpuscan cpu8; do
		case $cfg in gpu) S="$GPUSET";; gpuscan) S="$OLDSET";; cpu8) S="$CPUSET";; esac
		plan=$($PSQL -d bench -c "$S explain $2" | head -3 | tr '\n' '|' | cut -c1-150)
		t=""
		for r in 1 2 3; do
			ms=$($PSQL -d bench -c "$S explain (analyze, timing off) $2" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
			t="$t $ms"
		done
		echo "   $cfg:$t ms   [$plan]"
	done
}
bench "aggregate, quals" "select count(*), sum(a), max(c) from big where b > 0"
bench "aggregate, selective qual" "select count(*), sum(a), max(c) from big where a > 9990 and b < 100"
bench "aggregate, text qual" "select count(*), sum(b) from big where f like 'abc%'"
bench "GROUP BY 20 groups" "select a, count(*), sum(b), max(c) from big where a >= -10000 and a < -9980 group by a"
bench "GROUP BY 10 groups (expression key)" "select a % 10 as k, count(*), sum(b), max(c) from big group by 1"
bench "join, selective (outer qual)" "select t.i, d.grp from big t join bd d on t.b = d.k where t.a > 9990"
bench "join, unselective" "select t.i, d.grp from big t join bd d on t.b = d.k"
bench "join aggregate" "select count(*), sum(t.a), max(d.pay) from big t join bd d on t.b = d.k"
bench "join aggregate GROUP BY inner column (17 groups)" "select d.grp, count(*), sum(t.a), max(t.c) from big t join bd d on t.b = d.k group by 1"
bench "join aggregate, 2 conditions" "select count(*), sum(t.a) from big t join bd d on t.b = d.k where t.a > 0 and d.grp < 8"
