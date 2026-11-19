#!/bin/sh
# A19 concurrency sweep: does more concurrency (parallel workers, async tasks in flight, concurrent client sessions) raise the
# throughput of the GPU path? Uses the data set of perf_bench.sh (kept in $ROOT; run perf_bench.sh once to build it). Run through the wrapper:
#
#   ROOT=$PWD/test-data/par_bench sh test/gpuserv/locked_test.sh "$PWD" par_sweep.sh > test-data/par_sweep.log 2>&1
#
# Per query: best of 3 (EXPLAIN ANALYZE execution time) for max_parallel_workers_per_gather in WORKERS, max_async_tasks in ASYNCS,
# and scan_chunk_blocks in CHUNKS, and the wall time of K concurrent client sessions running the same query (CLIENTS), reported as queries per second.
#   LABEL_RE=<regex>   only the queries whose label matches
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/sp_bench}
PORT=${PORT:-54413}
WORKERS=${WORKERS:-"0 2 4"}
ASYNCS=${ASYNCS:-"1 2 4 8"}
CLIENTS=${CLIENTS:-"1 2 4"}
CHUNKS=${CHUNKS:-"2048 4096 8192 16384 32768"}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
[ -d "$ROOT/data" ] || { echo "no data set in $ROOT: run perf_bench.sh first"; exit 1; }
stop
grep -v "^pg_strom.gpuserv_path" "$ROOT/data/postgresql.conf" > "$ROOT/conf.tmp" && cp "$ROOT/conf.tmp" "$ROOT/data/postgresql.conf"
echo "pg_strom.gpuserv_path = '$GPUSERV_BIN'" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_TRACE=1
: > "$ROOT/log"
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; tail -20 "$ROOT/log"; exit 1; }
n=0; while [ $n -lt 600 ]; do
	[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 600 ] || { echo "service not ready"; tail -20 "$ROOT/log"; exit 1; }

BASE="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set pg_strom.gpu_scan_tuple_cost=0; set pg_strom.gpupreagg_max_groups=100000; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
best3() { # settings query -> best execution time of three runs
	b=""
	for r in 1 2 3; do
		ms=$($PSQL -d bench -c "$BASE $1 explain (analyze, timing off) $2" | grep "Execution Time" | sed 's/.*Execution Time: //; s/ ms//')
		b="$b $ms"
	done
	echo $b | tr ' ' '\n' | sort -n | head -1
}
sweep() { # label query
	label=$1; qry=$2
	[ -n "$LABEL_RE" ] && ! echo "$label" | grep -Eq "$LABEL_RE" && return
	echo "== $label"
	best3 "set max_parallel_workers_per_gather=0;" "$qry" > /dev/null	# warm the page cache and the kernels
	for w in $WORKERS; do
		plan=$($PSQL -d bench -c "$BASE set max_parallel_workers_per_gather=$w; explain $qry" | tr '\n' '|' | cut -c1-150)
		line="workers=$w"
		for a in $ASYNCS; do
			line="$line  async=$a: $(best3 "set max_parallel_workers_per_gather=$w; set pg_strom.max_async_tasks=$a;" "$qry") ms"
		done
		echo "$line"
		echo "   [$plan]"
	done
	line="chunk blocks (workers=0 async=2):"
	for c in $CHUNKS; do
		line="$line  $c: $(best3 "set max_parallel_workers_per_gather=0; set pg_strom.max_async_tasks=2; set pg_strom.scan_chunk_blocks=$c;" "$qry") ms"
	done
	echo "$line"
	for k in $CLIENTS; do
		t0=$(perl -MTime::HiRes=time -e 'printf "%.0f", time()*1000')
		i=0
		while [ $i -lt $k ]; do
			( for r in 1 2 3 4 5 6 7 8 9 10; do $PSQL -d bench -c "$BASE set max_parallel_workers_per_gather=0; $qry" > /dev/null; done ) &
			i=$((i+1))
		done
		wait
		t1=$(perl -MTime::HiRes=time -e 'printf "%.0f", time()*1000')
		echo "clients=$k: $((k * 10)) queries in $((t1 - t0)) ms ($(( (k * 10 * 1000000) / (t1 - t0) )) milli-queries/s)"
	done
}
sweep "agg, no qual"      "select count(*), sum(w), max(v) from t"
sweep "scan, arithmetic"  "select id, v * 2 + w from t where v * 3 + w > 12980 and w < 100"
sweep "group by 100"      "select g100, count(*), sum(w), max(v) from t group by g100"
sweep "join group by 20"  "select d.grp, count(*), sum(t.w), max(d.pay) from t join d on t.v = d.k where t.w < 5000 group by d.grp"
