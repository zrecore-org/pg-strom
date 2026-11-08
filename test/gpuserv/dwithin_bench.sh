#!/bin/sh
# Benchmark of the distance of a point to a line / polygon on the GPU (A13 slice 4): scan quals over N_ROWS geometries, the GPU against PostgreSQL
# (PostGIS, 1 process and 4 parallel workers), plus a GiST join over the same kind of inner rows (answered by the CPU: the join kernel has no walker,
# so it is a measure of the poison path, not of the GPU). Each query runs three times per configuration, the best EXPLAIN ANALYZE execution time is
# reported, the answers must be equal (md5 of the sorted rows), and "back" is the number of rows the GPU handed to the backend (Fallback-stats).
# Run it through the wrapper (the service runs with the default kernels, SPEC=default):
#
#   sh test/gpuserv/locked_test.sh "$PWD" dwithin_bench.sh
#
# N_ROWS (default 1000000) geometries, N_INNER (default 20000) / N_OUTER (default 20000) for the join. The service is NOT in check mode here.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pg_dwithin_bench}
PORT=${PORT:-54424}
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
export PGSTROM_GPUSERV_SPECIALIZE=${SPEC:-default}
export PGSTROM_GPUSERV_CACHE_DIR=$ROOT/speccache
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
$PSQL -d postgres -c "create database scantest" >/dev/null
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
sql "create extension pg_strom" >/dev/null

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # query -> md5 of the sorted result text
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
run() { # table, label, query-with-%T
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ]; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
run_any() { # like run, but the plan need not use GpuScan
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && [ "${g#*|}" != "" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
ANYPLAN=0
__runagg() {
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
JOINPAT="Custom Scan (GpuJoin)"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
runjoin() { # table, label, query-with-%T and %D [, settings]; the plan must use GpuJoin (unless ANYPLAN=1)
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/f8_dim/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $set_extra explain $qry" | grep -c "$JOINPAT")
	g=$(q "$GPU $JOINSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$plan" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuJoin=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
runagg() { ANYPLAN=0; __runagg "$@"; }
runagg_any() { ANYPLAN=1; __runagg "$@"; }

N_ROWS=${N_ROWS:-1000000}
N_INNER=${N_INNER:-20000}
N_OUTER=${N_OUTER:-20000}
GISTON="set pg_strom.enable_gpugistindex=on; set enable_seqscan=on;"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
sql "create extension if not exists postgis" >/dev/null
sql "drop table if exists bs; drop table if exists bz; drop table if exists bp"
# N_ROWS geometries over a 1000 x 1000 square: 4 of 5 are lines of 3 vertices (Z for kind 'line3d'), every 5th a polygon of 8 vertices (a buffer ring)
sql "create table bs (id int4, kind text, geom geometry, geom3 geometry, x float8, y float8, z float8)"
sql "insert into bs select i, case i % 5 when 0 then 'polygon' else 'line' end,
	case i % 5
		when 0 then ST_MakePolygon(ST_ExteriorRing(ST_Buffer(ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0, ((i * 104729::int8) % 1000000) / 1000.0), 0.9, 2)))
		else ST_MakeLine(ARRAY[ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0, ((i * 104729::int8) % 1000000) / 1000.0),
			ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0 + 0.8 + (i % 7) / 10.0, ((i * 104729::int8) % 1000000) / 1000.0 + 0.3),
			ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0 + 1.1, ((i * 104729::int8) % 1000000) / 1000.0 + 1.2 + (i % 3) / 10.0)])
	end,
	ST_MakeLine(ARRAY[ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0, ((i * 104729::int8) % 1000000) / 1000.0, (i % 13)::float8),
		ST_MakePoint(((i * 7919::int8) % 1000000) / 1000.0 + 1.0, ((i * 104729::int8) % 1000000) / 1000.0 + 0.5, (i % 17)::float8)]),
	((i * 15485863::int8) % 1000000) / 1000.0 + 0.0137, ((i * 32452843::int8) % 1000000) / 1000.0 + 0.0291, (i % 11)::float8
	from generate_series(1, $N_ROWS) i"
sql "analyze bs" >/dev/null
bench() { # label, query, [join]
	label=$1; qry=$2; join=$3
	gt=999999; ct=999999; pt=999999
	if [ -n "$join" ]; then gs="$GPU $JOINSET $GISTON"; cs="$CPU $JOINSET set enable_nestloop=on;"; ps="set pg_strom.enabled=off; set max_parallel_workers_per_gather=4; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0; $JOINSET set enable_nestloop=on;"
	else gs="$GPU"; cs="$CPU"; ps="set pg_strom.enabled=off; set max_parallel_workers_per_gather=4; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"; fi
	for k in 1 2 3; do
		t=$($PSQL -d scantest -c "$gs explain (analyze, costs off) $qry" | grep "Execution Time" | sed 's/[^0-9.]//g')
		[ "$(echo "$t < $gt" | bc)" = 1 ] && gt=$t
		t=$($PSQL -d scantest -c "$cs explain (analyze, costs off) $qry" | grep "Execution Time" | sed 's/[^0-9.]//g')
		[ "$(echo "$t < $ct" | bc)" = 1 ] && ct=$t
		t=$($PSQL -d scantest -c "$ps explain (analyze, costs off) $qry" | grep "Execution Time" | sed 's/[^0-9.]//g')
		[ "$(echo "$t < $pt" | bc)" = 1 ] && pt=$t
	done
	g=$(q "$gs" "$qry"); c=$(q "$cs" "$qry")
	r=$($PSQL -d scantest -c "$gs explain (analyze, costs off) $qry" | grep "Fallback-stats" | head -1 | sed 's/[^0-9=]//g; s/=/ /g')
	back=0; for v in $r; do back=$((back + v)); done
	same=differ; [ "$g" = "$c" ] && same=equal
	echo "  bench $label: GPU ${gt} ms, PostgreSQL ${ct} ms (1 process), ${pt} ms (4 workers); ${g#*|} rows, answers $same, $back rows back"
	[ "$same" = equal ] || fail=$((fail+1))
}
echo "scan quals over $N_ROWS lines and polygons"
bench "count, ST_DWithin(geom, constant point, 2)" "select count(*), sum(id) from bs where st_dwithin(geom, 'POINT(500.5 500.5)'::geometry, 2)"
bench "count, ST_DWithin(geom, constant point, 50)" "select count(*), sum(id) from bs where st_dwithin(geom, 'POINT(500.5 500.5)'::geometry, 50)"
bench "count, ST_DWithin(geom, ST_MakePoint(x, y), 0.5)" "select count(*), sum(id) from bs where st_dwithin(geom, st_makepoint(x, y), 0.5)"
bench "count, ST_Distance(geom, ST_MakePoint(x, y)) < 100" "select count(*), sum(id) from bs where st_distance(geom, st_makepoint(x, y)) < 100"
bench "count, polygons only, ST_DWithin(geom, ST_MakePoint(x, y), 300)" "select count(*), sum(id) from bs where kind = 'polygon' and st_dwithin(geom, st_makepoint(x, y), 300)"
bench "count, ST_3DDWithin(Z line, constant point, 30)" "select count(*), sum(id) from bs where st_3ddwithin(geom3, 'POINT Z (500.5 500.5 5)'::geometry, 30)"
bench "count, ST_3DDWithin(Z line, ST_MakePoint(x, y, z), 400)" "select count(*), sum(id) from bs where st_3ddwithin(geom3, st_makepoint(x, y, z), 400)"
echo "GiST join, inner lines and polygons (answered by the CPU through poison records)"
sql "create table bz as select id, geom, kind from bs where id <= $N_INNER"
sql "create index bz_gist on bz using gist (geom)"
sql "create table bp as select id, x, y from bs where id <= $N_OUTER"
sql "analyze bz; analyze bp" >/dev/null
bench "join count, tolerance 2" "select count(*), sum(i.id) from bp o join bz i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 2)" join
echo
if [ $fail -eq 0 ]; then echo "DWITHIN BENCH: DONE"; else echo "DWITHIN BENCH: $fail answers differ"; exit 1; fi
