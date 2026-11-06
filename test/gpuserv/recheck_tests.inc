# A12: the per-row CPU recheck on Arrow_Fdw, Parquet and GPU-Cache sources. Sourced by scan_test.sh and recheck_dev.sh (they
# provide run, run_any, runagg, runagg_any, runjoin, q, check, sql, $PSQL, $GPU, $CPU, $fail). Skipped when the postgis
# extension is not available. New tables only (rk_*).
#
# The point-in-polygon predicate hands back every point on an edge or vertex of its polygon (the grid of arrow_points.cpp is
# full of them), so each query below has the backend re-evaluate a few thousand rows of an Arrow record batch, a Parquet row
# group or a cached row; the answer must still be the CPU's, and EXPLAIN ANALYZE must show the rows in "Fallback-stats".
echo "Recheck on Arrow / Parquet / GPU-Cache sources (A12)"
if ! $PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1; then
	echo "  -- the postgis extension is not available: skipped"
else
RCKROWS=${RCKROWS:-120000}
SQ="'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry"
HOLE="'POLYGON((-8 -8,8 -8,8 8,-8 8,-8 -8),(-3 -3,3 -3,3 3,-3 3,-3 -3),(4 4,6 4,6 6,4 6,4 4))'::geometry"
MULTI="'MULTIPOLYGON(((0 0,6 0,6 6,0 6,0 0)),((3 3,9 3,9 9,3 9,3 3)),((-8 -8,-4 -8,-4 -4,-8 -4,-8 -8),(-7 -7,-5 -7,-5 -5,-7 -5,-7 -7)))'::geometry"

RARROW="$ROOT/rck_arrow"; mkdir -p "$RARROW"
ARROW_CFLAGS=$(pkg-config --cflags arrow parquet 2>/dev/null); ARROW_LIBS=$(pkg-config --libs arrow parquet 2>/dev/null)
if [ -z "$ARROW_LIBS" ] && command -v brew >/dev/null 2>&1; then
	AP=$(brew --prefix apache-arrow 2>/dev/null); ARROW_CFLAGS="-I$AP/include"; ARROW_LIBS="-L$AP/lib -lparquet -larrow -Wl,-rpath,$AP/lib"
# aggregation over the rows the GPU decides, with the rechecked rows added by the backend (the Agg node above GpuPreAgg
# combines the partial states): without GROUP BY, with one and two keys, a FILTER, HAVING-free
for t in rk_a rk_p rk_one rk_first rk_last rk_pone; do
	echo "table $t: aggregation"
	case $t in rk_one|rk_pone) agg=runagg_any;; *) agg=runagg;; esac
	$agg $t "count(*), sum, min, max" "select count(*), sum(id), min(id), max(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
	$agg $t "count of ST_Covers(hole), avg" "select count(*), avg(id) from %T where ST_Covers($HOLE, ST_MakePoint(x, y))"
	runagg_any $t "GROUP BY id % 7" "select id % 7 as k, count(*), sum(id) from %T where ST_Intersects(ST_MakePoint(x, y), $MULTI) group by 1 order by 1"
	runagg_any $t "GROUP BY two keys" "select id % 3 as k1, id % 5 as k2, count(*), max(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y)) group by 1, 2 order by 1, 2"
	$agg $t "count(*) FILTER" "select count(*) filter (where id % 2 = 0), count(*) from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
done
for t in rk_a rk_p; do
	rck_check $t "select count(*), sum(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "aggregation"
	rck_check $t "select id % 7, count(*) from %T where ST_Contains($SQ, ST_MakePoint(x, y)) group by 1" "GROUP BY"
done

# Z, M and ZM: the point built from three and four coordinates against 2D and 3D polygons (PostGIS decides in 2D)
SQZ="'POLYGON Z((0 0 0,10 0 0,10 10 0,0 10 0,0 0 0))'::geometry"
for t in rk_a rk_p rk_first; do
	echo "table $t: Z, M and ZM points"
	run $t "scan: ST_Contains(square, point Z)" "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y, z))"
	run $t "scan: ST_Covers(polygon Z, point Z)" "select id from %T where ST_Covers($SQZ, ST_MakePoint(x, y, z))"
	run $t "scan: ST_Intersects(point ZM, multi)" "select id from %T where ST_Intersects(ST_MakePoint(x, y, z, m), $MULTI)"
	run $t "scan: ST_Contains(hole, point ZM)" "select id, z from %T where ST_Contains($HOLE, ST_MakePoint(x, y, z, m))"
	runagg_any $t "count over a point Z" "select count(*), sum(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y, z))"
	runagg_any $t "GROUP BY over a point ZM" "select id % 4, count(*) from %T where ST_Contains($HOLE, ST_MakePoint(x, y, z, m)) group by 1 order by 1"
done
fi
c++ -std=c++20 -O2 -DWITH_PARQUET "${0%/*}/arrow_points.cpp" -o "$RARROW/arrow_points" $ARROW_CFLAGS $ARROW_LIBS >"$RARROW/gen.log" 2>&1
check "the point-table generator builds" $?
# 3 record batches / row groups; a one-row file whose row is the vertex (10 10) of SQ (i = 1176); a file that starts at the
# vertex (0 0) (i = 336) and one that ends at (10 10) (i = 1176 mod 1681), each one batch of 60000 rows; a Parquet file with several row groups
"$RARROW/arrow_points" "$RARROW/rk_a.arrow" "$RCKROWS" 50000
"$RARROW/arrow_points" "$RARROW/rk_p.parquet" "$RCKROWS" 50000
"$RARROW/arrow_points" "$RARROW/rk_one.arrow" 1 1 1176
"$RARROW/arrow_points" "$RARROW/rk_first.arrow" 60000 60000 336
"$RARROW/arrow_points" "$RARROW/rk_last.arrow" 60000 60000 12
"$RARROW/arrow_points" "$RARROW/rk_pone.parquet" 1 1 1176
RFT="(id int8, x float8, y float8, z float8, m float8)"
sql "create foreign table rk_a $RFT server arrow_fdw options (file '$RARROW/rk_a.arrow')"
sql "create foreign table rk_p $RFT server arrow_fdw options (file '$RARROW/rk_p.parquet')"
sql "create foreign table rk_one $RFT server arrow_fdw options (file '$RARROW/rk_one.arrow')"
sql "create foreign table rk_first $RFT server arrow_fdw options (file '$RARROW/rk_first.arrow')"
sql "create foreign table rk_last $RFT server arrow_fdw options (file '$RARROW/rk_last.arrow')"
sql "create foreign table rk_pone $RFT server arrow_fdw options (file '$RARROW/rk_pone.parquet')"
for t in rk_a rk_p rk_one rk_first rk_last rk_pone; do sql "analyze $t"; done

# EXPLAIN ANALYZE's count of the rows the backend re-evaluated ("Fallback-stats: depth[0]=N")
rck_rows() { # table, query-with-%T -> rows re-evaluated by the backend
	r=$($PSQL -d scantest -c "$GPU explain (analyze, costs off) $(printf '%s' "$2" | sed "s/%T/$1/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
rck_check() { # table, query-with-%T: the backend must have re-evaluated at least one row
	n=$(rck_rows "$1" "$2")
	[ "$n" -gt 0 ]; check "$1: the GPU handed rows back to the backend ($n rechecked): $3" $?
}

for t in rk_a rk_p; do
	echo "table $t"
	for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI"; do
		nm=${pl%%:*}; pg=${pl#*:}
		run $t "scan: ST_Contains($nm)" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
		run $t "scan: ST_Covers($nm)" "select id from %T where ST_Covers($pg, ST_MakePoint(x, y))"
		run $t "scan: ST_Intersects($nm)" "select id, x + y from %T where ST_Intersects(ST_MakePoint(x, y), $pg)"
	done
	rck_check $t "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "ST_Contains(square)"
done
for t in rk_one rk_first rk_last rk_pone; do
	echo "table $t"
	run $t "scan: ST_Covers(square)" "select id from %T where ST_Covers($SQ, ST_MakePoint(x, y))"
	run $t "scan: ST_Contains(square)" "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
	rck_check $t "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "ST_Contains(square)"
done
# aggregation over the rows the GPU decides, with the rechecked rows added by the backend (the Agg node above GpuPreAgg
# combines the partial states): without GROUP BY, with one and two keys, a FILTER, HAVING-free
for t in rk_a rk_p rk_one rk_first rk_last rk_pone; do
	echo "table $t: aggregation"
	case $t in rk_one|rk_pone) agg=runagg_any;; *) agg=runagg;; esac
	$agg $t "count(*), sum, min, max" "select count(*), sum(id), min(id), max(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
	$agg $t "count of ST_Covers(hole), avg" "select count(*), avg(id) from %T where ST_Covers($HOLE, ST_MakePoint(x, y))"
	runagg_any $t "GROUP BY id % 7" "select id % 7 as k, count(*), sum(id) from %T where ST_Intersects(ST_MakePoint(x, y), $MULTI) group by 1 order by 1"
	runagg_any $t "GROUP BY two keys" "select id % 3 as k1, id % 5 as k2, count(*), max(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y)) group by 1, 2 order by 1, 2"
	$agg $t "count(*) FILTER" "select count(*) filter (where id % 2 = 0), count(*) from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
done
for t in rk_a rk_p; do
	rck_check $t "select count(*), sum(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "aggregation"
	rck_check $t "select id % 7, count(*) from %T where ST_Contains($SQ, ST_MakePoint(x, y)) group by 1" "GROUP BY"
done

# Z, M and ZM: the point built from three and four coordinates against 2D and 3D polygons (PostGIS decides in 2D)
SQZ="'POLYGON Z((0 0 0,10 0 0,10 10 0,0 10 0,0 0 0))'::geometry"
for t in rk_a rk_p rk_first; do
	echo "table $t: Z, M and ZM points"
	run $t "scan: ST_Contains(square, point Z)" "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y, z))"
	run $t "scan: ST_Covers(polygon Z, point Z)" "select id from %T where ST_Covers($SQZ, ST_MakePoint(x, y, z))"
	run $t "scan: ST_Intersects(point ZM, multi)" "select id from %T where ST_Intersects(ST_MakePoint(x, y, z, m), $MULTI)"
	run $t "scan: ST_Contains(hole, point ZM)" "select id, z from %T where ST_Contains($HOLE, ST_MakePoint(x, y, z, m))"
	runagg_any $t "count over a point Z" "select count(*), sum(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y, z))"
	runagg_any $t "GROUP BY over a point ZM" "select id % 4, count(*) from %T where ST_Contains($HOLE, ST_MakePoint(x, y, z, m)) group by 1 order by 1"
done

# ---- parallel workers: every worker has its own record-batch state and recheck buffers
APAR="set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
runx() { # table, label, query-with-%T, settings: like run, with more settings after $GPU
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); ex=$4
	plan=$($PSQL -d scantest -c "$GPU $ex explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$(q "$GPU $ex" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ]; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
for t in rk_a rk_p; do
	runx $t "parallel workers: ST_Contains(hole)" "select id from %T where ST_Contains($HOLE, ST_MakePoint(x, y))" "$APAR"
	runx $t "parallel workers only: ST_Covers(square)" "select id, z from %T where ST_Covers($SQ, ST_MakePoint(x, y))" "$APAR set parallel_leader_participation=off;"
	runagg $t "parallel workers: count and sum" "select count(*), sum(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "$APAR"
	runagg_any $t "parallel workers: GROUP BY" "select id % 5, count(*), min(id) from %T where ST_Intersects(ST_MakePoint(x, y), $MULTI) group by 1 order by 1" "$APAR"
done

# ---- GpuJoin with an Arrow outer relation: the point-in-polygon join through a GiST index on a heap table of polygons. The
# polygons are squares on the integer grid (the grid points sit on their edges and vertices) and a few shapes the kernel
# cannot read (Z, M, ZM, a line, a point, a big polygon, NULL, EMPTY), which come back as poison records: the outer row goes to
# the backend, which joins it on the CPU. The outer tables are small Arrow / Parquet files with several batches.
GISTON="set pg_strom.enable_gpugistindex=on; set enable_seqscan=on;"
"$RARROW/arrow_points" "$RARROW/rk_j.arrow" 8200 3000 0 clean
"$RARROW/arrow_points" "$RARROW/rk_jp.parquet" 8200 3000 0 clean
sql "create foreign table rk_j $RFT server arrow_fdw options (file '$RARROW/rk_j.arrow')"
sql "create foreign table rk_jp $RFT server arrow_fdw options (file '$RARROW/rk_jp.parquet')"
sql "analyze rk_j"; sql "analyze rk_jp"
sql "drop table if exists rk_gz"
sql "create table rk_gz (id int4, geom geometry, a int4) with (fillfactor=70)"
sql "insert into rk_gz select i,
	case i % 16
		when 0 then null
		when 1 then ST_GeomFromText('POLYGON EMPTY')
		when 2 then ST_MakeEnvelope(i % 12 - 3, (i * 3) % 12 - 3, i % 12 + 1, (i * 3) % 12 + 2)
		when 3 then ST_Force3D(ST_MakeEnvelope(i % 12 - 2, i % 9 - 2, i % 12 + 3, i % 9 + 3), 2)
		when 4 then ST_Force3DM(ST_MakeEnvelope(i % 10 - 2, i % 11 - 2, i % 10 + 3, i % 11 + 3), 4)
		when 5 then ST_Force4D(ST_MakeEnvelope(i % 13 - 3, i % 7 - 3, i % 13 + 2, i % 7 + 2), 2, 3)
		when 6 then ST_MakeLine(ST_MakePoint(i % 10, i % 9), ST_MakePoint(i % 10 + 5, i % 9 + 5))
		when 7 then ST_MakePoint(i % 10, i % 9)
		when 8 then ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(i % 11, i % 8), ST_MakePoint(i % 11 + 6, i % 8), ST_MakePoint(i % 11 + 3, i % 8 + 5), ST_MakePoint(i % 11, i % 8)]))
		when 9 then ST_Difference(ST_MakeEnvelope(i % 9 - 2, i % 8 - 2, i % 9 + 8, i % 8 + 8), ST_MakeEnvelope(i % 9 + 1, i % 8 + 1, i % 9 + 5, i % 8 + 5))
		else ST_MakeEnvelope(i % 14 - 4, (i * 5) % 14 - 4, i % 14 + 2 + i % 3, (i * 5) % 14 + 2 + i % 4)
	end, i
	from generate_series(1, 400) i"
sql "insert into rk_gz values (90001, ST_Buffer(ST_MakePoint(6, 6), 5, 700), 1)"
sql "create index rk_gz_gist on rk_gz using gist (geom)"
sql "analyze rk_gz"
rk_gistjoin() { # label, query [, settings]; the plan must be a GPU GiST Join; the rows equal the CPU's
	label=$1; qry=$2; ex=${3:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $ex explain $qry")
	pj=$(echo "$plan" | grep -c "Custom Scan (GpuJoin)"); pg=$(echo "$plan" | grep -c "GPU GiST Join")
	g=$(q "$GPU $JOINSET $GISTON $ex" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$pj" -ge 1 ] && [ "$pg" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label (plan GpuJoin=$pj GPU GiST Join=$pg)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
rk_gistagg() { # label, query [, settings]: GpuPreAgg above the GiST join (ANYPLAN=1: the plan need not use GpuPreAgg)
	label=$1; qry=$2; ex=${3:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON set pg_strom.gpupreagg_max_groups=1000000; $ex explain $qry")
	pa=$(echo "$plan" | grep -c "Custom Scan (GpuPreAgg)"); pg=$(echo "$plan" | grep -c "GPU GiST Join")
	g=$(q "$GPU $JOINSET $GISTON set pg_strom.gpupreagg_max_groups=1000000; $ex" "$qry"); c=$(q "$CPU" "$qry")
	if { [ "$pa" -ge 1 ] || [ "$ANYPLAN" = 1 ]; } && [ "$pg" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label (plan GpuPreAgg=$pa GPU GiST Join=$pg)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
rk_joins() { # outer table
	o=$1
	echo "table $o: GiST joins"
	J="select o.id, i.id from $o o join rk_gz i on"
	rk_gistjoin "$o: ST_Contains(inner polygon, point)" "$J st_contains(i.geom, st_makepoint(o.x, o.y))"
	rk_gistjoin "$o: ST_Intersects(inner polygon, point)" "$J st_intersects(i.geom, st_makepoint(o.x, o.y))"
	rk_gistjoin "$o: ST_Within(point, inner polygon)" "$J st_within(st_makepoint(o.x, o.y), i.geom)"
	rk_gistjoin "$o: point Z" "$J st_intersects(i.geom, st_makepoint(o.x, o.y, o.z))"
	rk_gistjoin "$o: point ZM" "$J st_contains(i.geom, st_makepoint(o.x, o.y, o.z, o.m))"
	rk_gistjoin "$o: outer qual, inner qual" "$J st_intersects(i.geom, st_makepoint(o.x, o.y)) and o.id % 5 = 0 and i.a % 3 = 0"
	rk_gistjoin "$o: left join" "select o.id, i.id from $o o left join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y))"
	rk_gistjoin "$o: semi join" "select o.id from $o o where exists (select 1 from rk_gz i where st_intersects(i.geom, st_makepoint(o.x, o.y)))"
	rk_gistjoin "$o: anti join" "select o.id from $o o where not exists (select 1 from rk_gz i where st_contains(i.geom, st_makepoint(o.x, o.y)))"
	rk_gistjoin "$o: right join" "select o.id, i.id from $o o right join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y))"
	rk_gistjoin "$o: count per outer row above the join" "select o.id, count(i.id) from $o o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y)) group by o.id"
	rk_gistjoin "$o: parallel workers" "$J st_contains(i.geom, st_makepoint(o.x, o.y))" "$APAR"
	ANYPLAN=0
	A="select count(*), count(i.a), sum(i.a), min(i.id), max(o.id) from $o o join rk_gz i on"
	rk_gistagg "$o: ungrouped aggregate over the join" "$A st_contains(i.geom, st_makepoint(o.x, o.y))"
	rk_gistagg "$o: ungrouped aggregate, ST_Intersects, point ZM" "$A st_intersects(i.geom, st_makepoint(o.x, o.y, o.z, o.m))"
	rk_gistagg "$o: GROUP BY an inner key" "select i.a % 5, count(*), sum(o.id), max(o.id) from $o o join rk_gz i on st_intersects(i.geom, st_makepoint(o.x, o.y)) group by i.a % 5"
	rk_gistagg "$o: aggregate, parallel workers" "$A st_contains(i.geom, st_makepoint(o.x, o.y))" "$APAR"
	ANYPLAN=1
	rk_gistagg "$o: GROUP BY an outer key (any plan)" "select o.id % 7, count(*), sum(i.a), min(i.id) from $o o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y)) group by o.id % 7"
	rk_gistagg "$o: GROUP BY two keys (any plan)" "select o.id % 3, i.a % 4, count(*), sum(o.id) from $o o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y)) group by o.id % 3, i.a % 4"
	ANYPLAN=0
}
rk_joins rk_j
rk_joins rk_jp
# the GPU must have handed rows of the Arrow outer relation to the backend
for t in rk_j rk_jp; do
	r=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON explain (analyze, costs off) select o.id, i.id from $t o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y))" | grep "Fallback-stats" | head -1 | sed 's/[^0-9=]//g; s/=/ /g')
	n=0; for v in $r; do n=$((n + v)); done
	[ "$n" -gt 0 ]; check "$t: the GiST join handed outer rows to the backend (Fallback-stats sum $n)" $?
done

# ---- GPU-Cache: the same grid in a heap table with a cache. The service builds a heap tuple of the cached row for the backend
# (the cache is not a snapshot, so the backend cannot fetch the version from the heap).
sql "create table rk_gc (id int8, x float8, y float8, z float8, m float8)"
sql "create trigger gc_sync after insert or update or delete on rk_gc for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=60000')"
sql "insert into rk_gc select i, case when i % 97 = 96 then null when i % 211 = 210 then 'Infinity'::float8 else (((i % 1681) % 41) - 8) / 2.0 end,
	case when i % 101 = 100 then null when i % 223 = 222 then 'NaN'::float8 else (((i % 1681) / 41) - 8) / 2.0 end,
	(((i % 1681) % 41) - 8) / 2.0 + (((i % 1681) / 41) - 8) / 2.0, (((i % 1681) % 41) - 8) / 2.0 - (((i % 1681) / 41) - 8) / 2.0
	from generate_series(0, 7999) i"
sql "analyze rk_gc"
rkgc() { # label, query-with-%T [, settings]: the plan must read the GPU-Cache and the rows equal the heap's
	label=$1; qry=$(printf '%s' "$2" | sed "s/%T/rk_gc/g"); ex=${3:-}
	plan=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; $ex explain $qry" 2>&1 | grep -c "GPU Cache:")
	g=$(q "$GPU set enable_seqscan=off; $ex" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    rk_gc: $label (${g#*|} rows)"
	else echo "  FAIL  rk_gc: $label (plan GPU Cache=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
echo "table rk_gc (GPU-Cache)"
for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI"; do
	nm=${pl%%:*}; pg=${pl#*:}
	rkgc "scan: ST_Contains($nm)" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
	rkgc "scan: ST_Covers($nm)" "select id, x + y from %T where ST_Covers($pg, ST_MakePoint(x, y))"
	rkgc "scan: ST_Intersects($nm, point Z)" "select id from %T where ST_Intersects(ST_MakePoint(x, y, z), $pg)"
done
rkgc "scan: ST_Contains(hole, point ZM)" "select id, z, m from %T where ST_Contains($HOLE, ST_MakePoint(x, y, z, m))"
rkgc "scan: next to a plain qual" "select id from %T where id % 3 = 0 and ST_Covers($SQ, ST_MakePoint(x, y))"
rkgc "aggregate: count, sum, min, max" "select count(*), sum(id), min(id), max(id) from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
rkgc "aggregate: GROUP BY id % 7" "select id % 7, count(*), sum(id) from %T where ST_Intersects(ST_MakePoint(x, y), $MULTI) group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1;"
rkgc "aggregate: GROUP BY two keys" "select id % 3, id % 5, count(*), max(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y)) group by 1, 2 order by 1, 2" "set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1;"
rkgc "scan, parallel workers" "select id from %T where ST_Contains($HOLE, ST_MakePoint(x, y))" "$APAR"
n=$(rck_rows rk_gc "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))")
[ "$n" -gt 0 ]; check "rk_gc: the GPU handed cached rows to the backend ($n rechecked)" $?
# a join with the cache as the outer relation
rk_gistjoin "rk_gc: ST_Contains join" "select o.id, i.id from rk_gc o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y)) and o.x <> 'Infinity'::float8 and o.y <> 'NaN'::float8"
rk_gistjoin "rk_gc: left join, point ZM" "select o.id, i.id from rk_gc o left join rk_gz i on st_intersects(i.geom, st_makepoint(o.x, o.y, o.z, o.m)) and o.x <> 'Infinity'::float8 and o.y <> 'NaN'::float8"
rk_gistagg "rk_gc: ungrouped aggregate over the join" "select count(*), sum(i.a), min(i.id), max(o.id) from rk_gc o join rk_gz i on st_contains(i.geom, st_makepoint(o.x, o.y)) and o.x <> 'Infinity'::float8 and o.y <> 'NaN'::float8"
ANYPLAN=1
rk_gistagg "rk_gc: GROUP BY an outer key (any plan)" "select o.id % 7, count(*), sum(i.a) from rk_gc o join rk_gz i on st_intersects(i.geom, st_makepoint(o.x, o.y)) and o.x <> 'Infinity'::float8 and o.y <> 'NaN'::float8 group by o.id % 7"
ANYPLAN=0

# the cache holds the latest committed state and the session's own changes: rows changed inside a transaction (at the vertex
# (0 0) and (10 10) of the square, so they are rechecked) must be seen by the recheck exactly as by the heap scan
rk_txn() { # label, ending (commit|rollback), statements, query-with-%T
	label=$1; ending=$2; stmts=$3; qry=$(printf '%s' "$4" | sed "s/%T/rk_gc/g")
	out=$($PSQL -d scantest 2>&1 <<EOF
begin;
$stmts
$GPU
set enable_seqscan=off;
explain $qry;
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
$CPU
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
$ending;
EOF
)
	n=$(echo "$out" | grep -c "GPU Cache:")
	g=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 1p); c=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 2p)
	if [ "$n" -ge 1 ] && [ -n "$g" ] && [ "$g" = "$c" ] && ! echo "$out" | grep -q "ERROR"; then echo "  ok    rk_gc: $label (${g#*|} rows)"
	else echo "  FAIL  rk_gc: $label (GPU Cache lines=$n)"; echo "$out" | head -20; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
Q="select id, x, y from %T where ST_Covers($SQ, ST_MakePoint(x, y)) and id >= 100000"
rk_txn "own INSERT of vertex points" rollback "insert into rk_gc values (100001, 0, 0, 0, 0), (100002, 10, 10, 20, 0), (100003, 5, 0, 5, 5), (100004, 5.5, 5.5, 11, 0), (100005, null, 1, 1, 1);" "$Q"
rk_txn "own UPDATE moves a row onto a vertex" rollback "update rk_gc set x = 0, y = 0 where id between 100 and 110; insert into rk_gc values (100001, 10, 0, 10, 10);" "select id, x, y from %T where ST_Contains($SQ, ST_MakePoint(x, y)) and (id between 100 and 110 or id >= 100000)"
rk_txn "own DELETE of rechecked rows" rollback "delete from rk_gc where id between 330 and 340 or id % 17 = 0;" "select id from %T where ST_Intersects(ST_MakePoint(x, y), $SQ)"
rk_txn "committed changes" commit "insert into rk_gc values (100010, 0, 10, 10, -10), (100011, 10, 0, 10, 10); update rk_gc set y = 10 where id between 200 and 205; delete from rk_gc where id between 400 and 420;" "select id, y from %T where ST_Covers($SQ, ST_MakePoint(x, y)) and (id between 190 and 430 or id >= 100000)"
rk_txn "savepoint rollback" commit "insert into rk_gc values (100020, 0, 0, 0, 0); savepoint s1; insert into rk_gc values (100021, 10, 10, 20, 0); update rk_gc set x = 10 where id = 100020; rollback to s1; insert into rk_gc values (100022, 0, 10, 10, -10);" "select id, x, y from %T where ST_Covers($SQ, ST_MakePoint(x, y)) and id >= 100000"
rk_txn "aggregate inside the transaction" rollback "insert into rk_gc values (100031, 0, 0, 0, 0), (100032, 10, 10, 20, 0); delete from rk_gc where id between 336 and 340;" "select count(*), sum(id) from %T where ST_Covers($SQ, ST_MakePoint(x, y))"
fi
