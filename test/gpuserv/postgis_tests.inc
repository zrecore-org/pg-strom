# A11: PostGIS on the GPU. Sourced by scan_test.sh and postgis_dev.sh (they provide run, run_any, q, check, sql, $PSQL,
# $GPU, $CPU, $fail). Skipped when the postgis extension is not available. New tables only (pg_*).
echo "PostGIS (A11)"
if ! $PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1; then
	echo "  -- the postgis extension is not available: skipped"
else
PGROWS=${PGROWS:-100000}

# g: a geometry of every shape the operators meet: points (no cached box), 3D and M points, lines and polygons with a cached
# box, 2-vertex lines and triangles (PostGIS keeps no box for them), empties, multipoints, a collection, NULLs, tiny and
# huge coordinates, infinity. g2: a second geometry (points near the first). x, y: a point as two float8.
pg_create() { # table
	sql "create table $1 (id int4, g geometry, g2 geometry, x float8, y float8, a int4)"
	sql "insert into $1 select i,
		case i % 20
			when 0 then null
			when 1 then ST_SetSRID(ST_MakePoint(((i * 7919) % 20001) / 1000.0 - 10, ((i * 104729) % 20001) / 1000.0 - 10), 4326)
			when 2 then ST_MakePoint(((i * 31) % 2001) / 100.0, ((i * 17) % 2001) / 100.0)
			when 3 then ST_MakePoint(((i * 31) % 2001) / 100.0, ((i * 17) % 2001) / 100.0, (i % 7)::float8)
			when 4 then ST_MakePointM(((i * 31) % 2001) / 100.0, ((i * 17) % 2001) / 100.0, (i % 5)::float8)
			when 5 then ST_GeomFromText('POINT EMPTY')
			when 6 then ST_GeomFromText('LINESTRING EMPTY')
			when 7 then ST_GeomFromText('POLYGON EMPTY')
			when 8 then ST_MakeLine(ARRAY[ST_MakePoint(i % 50, i % 40), ST_MakePoint(i % 30 + 5, i % 60), ST_MakePoint(i % 70, i % 20 + 3)])
			when 9 then ST_MakeLine(ST_MakePoint(i % 50, i % 40), ST_MakePoint(i % 30 + 5, i % 60))
			when 10 then ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(i % 40, i % 30), ST_MakePoint(i % 40 + 7, i % 30), ST_MakePoint(i % 40 + 7, i % 30 + 6), ST_MakePoint(i % 40, i % 30 + 6), ST_MakePoint(i % 40, i % 30)]))
			when 11 then ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(i % 40, i % 30), ST_MakePoint(i % 40 + 4, i % 30), ST_MakePoint(i % 40, i % 30 + 4), ST_MakePoint(i % 40, i % 30)]))
			when 12 then ST_Collect(ARRAY[ST_MakePoint(i % 25, i % 35), ST_MakePoint(i % 45 + 2, i % 15), ST_MakePoint(i % 10, i % 55)])
			when 13 then ST_GeomFromText('MULTIPOINT((1 2),(30 40),(7 8))')
			when 14 then ST_MakePoint(1e-30 * (i % 100), 1e-45 * (i % 50))
			when 15 then ST_MakePoint(1e38 * ((i % 7) - 3), 3e38 * ((i % 5) - 2))
			when 16 then ST_MakePoint('Infinity'::float8 * (CASE WHEN i % 2 = 0 THEN 1 ELSE -1 END), (i % 9)::float8)
			when 17 then ST_MakePoint(0.1 * (i % 100), 1.0 / 3.0 * (i % 90))
			when 18 then ST_Collect(ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(0,0), ST_MakePoint(10,0), ST_MakePoint(10,10), ST_MakePoint(0,10), ST_MakePoint(0,0)])), ST_MakePoint(50, 50))
			else ST_MakePoint(((i * 7919) % 20001) / 1000.0 - 10, ((i * 104729) % 20001) / 1000.0 - 10)
		end,
		case when i % 11 = 0 then null else ST_MakePoint(((i * 13) % 4001) / 100.0, ((i * 29) % 4001) / 100.0) end,
		((i * 7919) % 20001) / 1000.0 - 10, ((i * 104729) % 20001) / 1000.0 - 10,
		i % 1000
		from generate_series(1::int8, $PGROWS) i"
}
pg_create pg_1; sql "vacuum analyze pg_1"
pg_create pg_2; sql "update pg_2 set a = a + 1 where id % 5 = 0"; sql "delete from pg_2 where id % 13 = 0"; sql "analyze pg_2"

for t in pg_1 pg_2; do
	echo "table $t"
	run $t "&& a polygon" "select id from %T where g && 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry"
	run $t "&& a point" "select id from %T where g && 'POINT(5 5)'::geometry"
	run $t "&& a thin line" "select id from %T where g && 'LINESTRING(0 0, 100 0.001)'::geometry"
	run $t "~ (contains the box of) a small polygon" "select id from %T where g ~ 'POLYGON((1 1,2 1,2 2,1 2,1 1))'::geometry"
	run $t "@ (is within the box of) a big polygon" "select id from %T where g @ 'POLYGON((-100 -100,100 -100,100 100,-100 100,-100 -100))'::geometry"
	run $t "@ a polygon, with a column and NULLs" "select id, a from %T where g @ 'POLYGON((0 0,40 0,40 40,0 40,0 0))'::geometry and a < 500"
	run $t "&& between two columns" "select id from %T where g && g2"
	run $t "~ and @ between two columns" "select id from %T where g ~ g2 or g @ g2"
	run $t "NOT (g && polygon)" "select id from %T where not (g && 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry)"
	run $t "&& with other quals and a projection" "select id, a, x * 2 from %T where g && 'POLYGON((-5 -5,5 -5,5 5,-5 5,-5 -5))'::geometry and a % 3 = 0"
	run $t "IS NULL on a geometry" "select id from %T where g is null or g2 is null"
	run $t "geometry pass-through next to a box qual" "select id, g from %T where g && 'POINT(1 2)'::geometry"
	run $t "&& with a point of infinity" "select id from %T where g && ST_MakePoint('Infinity'::float8, 3)"
	run $t "&& with huge coordinates" "select id from %T where g && 'POLYGON((-1e39 -1e39,1e39 -1e39,1e39 1e39,-1e39 1e39,-1e39 -1e39))'::geometry"
	run $t "&& with tiny coordinates" "select id from %T where g && 'POLYGON((-1e-40 -1e-40,1e-29 -1e-40,1e-29 1e-44,-1e-40 1e-44,-1e-40 -1e-40))'::geometry"
	run $t "&& the box of a point just outside a float4" "select id from %T where g && 'POINT(0.1 0.3333333333333333)'::geometry"
done

# ---- point in polygon (GEOMPIP): ST_Contains / ST_Covers / ST_Intersects / ST_Within / ST_CoveredBy ----
# The kernel decides inside / outside and hands everything doubtful (on an edge or a vertex, within the error bound of the
# orientation test, non-finite or extreme coordinates, a different SRID, another geometry type) to the backend.
SQ="'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry"
HOLE="'POLYGON((-8 -8,8 -8,8 8,-8 8,-8 -8),(-3 -3,3 -3,3 3,-3 3,-3 -3),(4 4,6 4,6 6,4 6,4 4))'::geometry"
MULTI="'MULTIPOLYGON(((0 0,6 0,6 6,0 6,0 0)),((3 3,9 3,9 9,3 9,3 3)),((-8 -8,-4 -8,-4 -4,-8 -4,-8 -8),(-7 -7,-5 -7,-5 -5,-7 -5,-7 -7)))'::geometry"
CIRCLE="ST_Buffer(ST_MakePoint(1.5, -0.5), 7.3, 32)"
BOWTIE="'POLYGON((-9 -9,9 9,9 -9,-9 9,-9 -9))'::geometry"
LOOP2="'POLYGON((-5 -5,5 -5,5 5,-5 5,-5 -5,5 -5,5 5,-5 5,-5 -5))'::geometry"
CW="'POLYGON((0 0,0 8,8 8,8 0,0 0))'::geometry"

for t in pg_1 pg_2; do
	echo "table $t, point in polygon"
	for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI" "circle:$CIRCLE" "bowtie:$BOWTIE" "doubleloop:$LOOP2" "clockwise:$CW"; do
		nm=${pl%%:*}; pg=${pl#*:}
		run $t "ST_Contains($nm, ST_MakePoint(x, y))" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
		run $t "ST_Covers($nm, ST_MakePoint(x, y))" "select id from %T where ST_Covers($pg, ST_MakePoint(x, y))"
		run $t "ST_Intersects(ST_MakePoint(x, y), $nm)" "select id from %T where ST_Intersects(ST_MakePoint(x, y), $pg)"
		run $t "ST_Intersects($nm, ST_Point(x, y))" "select id from %T where ST_Intersects($pg, ST_Point(x, y))"
		run $t "ST_Within(ST_MakePoint(x, y), $nm)" "select id from %T where ST_Within(ST_MakePoint(x, y), $pg)"
		run $t "ST_CoveredBy(ST_MakePoint(x, y), $nm)" "select id from %T where ST_CoveredBy(ST_MakePoint(x, y), $pg)"
	done
	run $t "NOT ST_Contains" "select id from %T where not ST_Contains($HOLE, ST_MakePoint(x, y))"
	run $t "ST_Contains with other quals and a projection" "select id, a, x + y from %T where ST_Contains($MULTI, ST_MakePoint(x, y)) and a % 3 = 0"
	run $t "ST_Contains, the point from expressions" "select id from %T where ST_Contains($SQ, ST_MakePoint(x * 2 - 5, y / 2))"
	run $t "ST_Contains and ST_Intersects of two polygons" "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y)) or ST_Intersects(ST_MakePoint(x, y), $MULTI)"
	run $t "a SRID on both sides" "select id from %T where ST_Contains(ST_SetSRID($HOLE, 4326), ST_SetSRID(ST_MakePoint(x, y), 4326))"
	run $t "a point column (g2) in a polygon" "select id from %T where ST_Contains($MULTI, g2)"
	run $t "a point column (g2) and ST_Within" "select id from %T where ST_Within(g2, ST_Buffer(ST_MakePoint(20, 20), 12, 16))"
	run $t "ST_Intersects of a polygon and a point constant" "select id from %T where ST_Intersects($SQ, 'POINT(5 5)'::geometry) and a < 500"
done

# NULL, Infinity and NaN coordinates in columns (NULL gives NULL, the others go to the backend)
sql "create table pg_odd as select id, x, y, case when id % 7 = 0 then null else x end as xn, case when id % 11 = 0 then 'Infinity'::float8 when id % 5 = 0 then '-Infinity'::float8 else x end as xi, case when id % 13 = 0 then 'NaN'::float8 else y end as yq from pg_1"
sql "analyze pg_odd"
run pg_odd "ST_Contains with a NULL x" "select id from %T where ST_Contains($SQ, ST_MakePoint(xn, y))"
run pg_odd "NOT ST_Contains with a NULL x" "select id from %T where not ST_Contains($SQ, ST_MakePoint(xn, y))"
run pg_odd "ST_Contains with an infinite x" "select id from %T where ST_Contains($SQ, ST_MakePoint(xi, y))"
run pg_odd "ST_Intersects with a NaN y" "select id from %T where ST_Intersects($SQ, ST_MakePoint(x, yq))"

# a column of every shape (points, lines, polygons, empties, collections, NULLs, Z and M points): what is not a plain
# point in the same SRID goes to the backend; the SRID-4326 rows make the CPU raise "mixed SRID", and so must the query
# that the GPU path hands over
echo "mixed shapes and SRIDs"
mixed() { # label, query: both paths must give the same error text
	qry=$(printf '%s' "$2" | sed "s/%T/pg_1/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && echo "$g" | grep -q "mixed SRID"; then echo "  ok    pg_1: $1"
	else echo "  FAIL  pg_1: $1"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
mixed "ST_Intersects(g, polygon): mixed SRID raises the same error" "select id from %T where ST_Intersects(g, $SQ)"
mixed "ST_Contains(polygon, ST_SetSRID(point, 4326)): raises" "select id from %T where ST_Contains($SQ, ST_SetSRID(ST_MakePoint(x, y), 4326))"
mixed "ST_Within(point, polygon with another SRID): raises" "select id from %T where ST_Within(ST_MakePoint(x, y), ST_SetSRID($SQ, 3857))"
# the same column with every SRID set to 0: the shapes alone decide (a line, a collection, an empty, a Z or M point ...)
sql "create table pg_3 as select id, ST_SetSRID(g, 0) as g, g2, x, y, a from pg_1"
sql "analyze pg_3"
run pg_3 "ST_Intersects(g, square) over every shape" "select id from %T where ST_Intersects(g, $SQ)"
run pg_3 "ST_Intersects(square, g) over every shape" "select id from %T where ST_Intersects($SQ, g)"
run pg_3 "ST_Contains(multi, g) over every shape" "select id from %T where ST_Contains($MULTI, g)"
run pg_3 "ST_Covers(hole, g) over every shape" "select id from %T where ST_Covers($HOLE, g)"
run pg_3 "ST_Within(g, circle) over every shape" "select id from %T where ST_Within(g, $CIRCLE)"
run pg_3 "ST_CoveredBy(g, bowtie) over every shape" "select id from %T where ST_CoveredBy(g, $BOWTIE)"
run pg_3 "ST_Intersects(g, g2): no polygon constant" "select id from %T where ST_Intersects(g, g2)"

# Boundaries: a grid of points on integer and half-integer coordinates against polygons with integer vertices (points on
# edges, on vertices, in a hole's rim, on a horizontal edge's line beyond its end), points 0 to 6 ulps off a diagonal edge
# (where PostGIS's fused multiply-add and a plain IEEE evaluation disagree), and polygons with huge, tiny and nearly
# degenerate coordinates. Most of these rows are handed back to the backend; the answers must still be PostGIS's own.
sql "create table pg_grid as select i as id, (gx / 2.0)::float8 as x, (gy / 2.0)::float8 as y from
	(select i, ((i % 1681) % 41) - 8 as gx, ((i % 1681) / 41) - 8 as gy from generate_series(0, 100799) i) s"
sql "create table pg_ulp as select i as id, t + (k - 3)::float8 * 2.220446049250313e-16::float8 * t as x, t as y, k from
	(select i, (i % 7) as k, (0.001 + (i / 7) * 0.0005)::float8 as t from generate_series(0, 139999) i) s"
sql "analyze pg_grid"; sql "analyze pg_ulp"
TRI="'POLYGON((0 0,10 10,0 10,0 0))'::geometry"
for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI" "bowtie:$BOWTIE" "doubleloop:$LOOP2" "triangle:$TRI"; do
	nm=${pl%%:*}; pg=${pl#*:}
	run pg_grid "grid: ST_Contains($nm)" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
	run pg_grid "grid: ST_Covers($nm)" "select id from %T where ST_Covers($pg, ST_MakePoint(x, y))"
	run pg_grid "grid: ST_Intersects($nm)" "select id from %T where ST_Intersects(ST_MakePoint(x, y), $pg)"
done
run pg_ulp "ulp: ST_Contains(triangle)" "select id from %T where ST_Contains($TRI, ST_MakePoint(x, y))"
run pg_ulp "ulp: ST_Covers(triangle)" "select id from %T where ST_Covers($TRI, ST_MakePoint(x, y))"
run pg_ulp "ulp: ST_Intersects(triangle)" "select id from %T where ST_Intersects(ST_MakePoint(x, y), $TRI)"
run pg_ulp "ulp: ST_Contains(rotated triangle)" "select id from %T where ST_Contains('POLYGON((0 0,0.3 0.1,-0.2 0.7,0 0))'::geometry, ST_MakePoint(x / 40, y / 40 * 0.9))"
run pg_grid "multipolygon, a point inside one part and on the edge of the next" "select id from %T where ST_Contains('MULTIPOLYGON(((0 0,6 0,6 6,0 6,0 0)),((3 3,9 3,9 9,3 9,3 3)))'::geometry, ST_MakePoint(x, y))"
run pg_grid "the same parts in the other order" "select id from %T where ST_Contains('MULTIPOLYGON(((3 3,9 3,9 9,3 9,3 3)),((0 0,6 0,6 6,0 6,0 0)))'::geometry, ST_MakePoint(x, y))"
run pg_grid "huge polygon" "select id from %T where ST_Contains('POLYGON((-1e300 -1e300,1e300 -1e300,1e300 1e300,-1e300 1e300,-1e300 -1e300))'::geometry, ST_MakePoint(x, y))"
run pg_grid "huge coordinates of the point" "select id from %T where ST_Contains($SQ, ST_MakePoint(x * 1e300, y * 1e300))"
run pg_grid "tiny polygon" "select id from %T where ST_Contains('POLYGON((-1e-200 -1e-200,1e-200 -1e-200,1e-200 1e-200,-1e-200 1e-200,-1e-200 -1e-200))'::geometry, ST_MakePoint(x * 1e-201, y * 1e-201))"
run pg_grid "sliver polygon" "select id from %T where ST_Intersects('POLYGON((0 0,10 0.0000000000001,10 0,0 0))'::geometry, ST_MakePoint(x, y / 1e13))"
run pg_grid "polygon with a repeated vertex and a spike" "select id from %T where ST_Contains('POLYGON((0 0,0 0,5 5,10 0,10 10,10 10,5 5,0 10,0 0))'::geometry, ST_MakePoint(x, y))"
runagg pg_grid "count of a polygon test" "select count(*) from %T where ST_Contains($HOLE, ST_MakePoint(x, y))"
runagg pg_2 "count, min and max over points in a multipolygon" "select count(*), min(a), max(a) from %T where ST_Contains($MULTI, ST_MakePoint(x, y))"
runagg_any pg_2 "count by a, the points in a multipolygon" "select a, count(*) from %T where ST_Contains($MULTI, ST_MakePoint(x, y)) group by a order by a"

# The recheck is an escape hatch: a kernel that handed every row back would pass every comparison above. EXPLAIN ANALYZE
# counts the rows the backend re-evaluated ("Fallback-stats: depth[0]=N"): on random points almost none may be (the
# positive control, the integer grid full of points on edges and vertices, must show some, or the counter is not read).
recheck_rows() { # table, query -> rows re-evaluated by the backend
	r=$($PSQL -d scantest -c "$GPU explain (analyze, costs off) $(printf '%s' "$2" | sed "s/%T/$1/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
for t in pg_1 pg_2; do
	nrows=$(sql "select count(*) from $t")
	for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI" "circle:$CIRCLE"; do
		nm=${pl%%:*}; pg=${pl#*:}
		n=$(recheck_rows $t "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))")
		[ "$n" -le $((nrows / 100)) ]; check "$t: random points against $nm: $n of $nrows rows rechecked (at most 1%)" $?
	done
done
n=$(recheck_rows pg_grid "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))")
[ "$n" -gt 0 ]; check "pg_grid: points on edges and vertices are rechecked (counter read: $n rows)" $?

# A table with a GPU-Cache: the cache holds neither geometry nor a way to hand a row back to the backend, so the geometry
# predicates stay on the CPU there (an error from the service would be the alternative); the answers are still PostGIS's.
sql "create table pg_gc as select id, g2 as g, x, y from pg_1 where g2 is not null and id <= 30000"
sql "create trigger gc_sync after insert or update or delete on pg_gc for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=90000')"
sql "analyze pg_gc"
run_any pg_gc "GPU-Cache table: fused ST_Contains" "select id from %T where ST_Contains($MULTI, ST_MakePoint(x, y))"
run_any pg_gc "GPU-Cache table: ST_Intersects on a point column" "select id from %T where ST_Intersects(g, $SQ)"
run_any pg_gc "GPU-Cache table: && on a point column" "select id from %T where g && $SQ"
run_any pg_gc "GPU-Cache table: a plain qual next to ST_Contains" "select id from %T where x > 0 and ST_Contains($SQ, ST_MakePoint(x, y))"

# ---- A11 part 3: ST_DWithin of two points. PostGIS tests its own distance (which differs from plain binary64 in 8% of random
# pairs) against the tolerance, so the instruction decides only beyond a relative 2^-47 of r*r and hands the boundary, a SRID
# mismatch, a bad tolerance and every non-point back to the backend. ----
errq() { # settings, query -> the first ERROR line without the prefix
	$PSQL -d scantest -c "$1 $2" 2>&1 | sed -n 's/^.*ERROR:  *//p' | head -1
}
runerr_pg() { # table, label, query-with-%T, expected message; the GPU plan must be a GpuScan
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$(errq "$GPU" "$qry"); c=$(errq "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && [ "$g" = "$4" ]; then echo "  ok    $tbl: $label ($g)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; echo "        want: $4"; fail=$((fail+1)); fi
}
P55="'POINT(5 5)'::geometry"
for t in pg_1 pg_2; do
	run $t "ST_DWithin(point column, point constant, 3), every shape" "select id from %T where id % 20 <> 1 and ST_DWithin(g, $P55, 3)"
	run $t "ST_DWithin(constant, column, 3): the other order" "select id from %T where id % 20 <> 1 and ST_DWithin($P55, g, 3)"
	run $t "ST_DWithin(fused point, constant, 2.5)" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 2.5)"
	run $t "ST_DWithin(constant, fused point, 2.5)" "select id from %T where ST_DWithin($P55, ST_MakePoint(x, y), 2.5)"
	run $t "ST_DWithin(two fused points, tolerance from a column)" "select id from %T where ST_DWithin(ST_MakePoint(x, y), ST_MakePoint(y, x), a::float8 / 100)"
	run $t "ST_DWithin(fused point, point column, 4), SRID 0" "select id from %T where id % 20 <> 1 and ST_DWithin(ST_MakePoint(x, y), g, 4)"
	run $t "ST_DWithin(g, g2, 5): two columns" "select id from %T where id % 20 <> 1 and ST_DWithin(g, g2, 5)"
	run $t "ST_DWithin with a computed tolerance" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, abs(x) / 4 + 0.5)"
	run $t "ST_DWithin with a SRID on both fused points" "select id from %T where ST_DWithin(ST_SetSRID(ST_MakePoint(x, y), 4326), ST_SetSRID(ST_MakePoint(y, x), 4326), 3)"
	run $t "ST_DWithin(fused with SRID, g with SRID 4326)" "select id from %T where id % 20 = 1 and ST_DWithin(ST_SetSRID(ST_MakePoint(x, y), 4326), g, 6)"
	run $t "NOT ST_DWithin, with other quals and a projection" "select id, a, x * 2 from %T where not ST_DWithin(ST_MakePoint(x, y), $P55, 6) and a % 3 = 0"
	run $t "ST_DWithin, tolerance 0" "select id from %T where ST_DWithin(ST_MakePoint(x, y), ST_MakePoint(x, y), 0)"
	run $t "ST_DWithin, tolerance 0, other point" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 0) or a = 7"
	run $t "ST_DWithin, tolerance Infinity" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 'Infinity'::float8) and a = 3"
	run $t "ST_DWithin, tolerance NaN" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 'NaN'::float8) or a = 3"
	run $t "ST_DWithin, tolerance NULL" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, null) or a = 3"
	run $t "ST_DWithin, huge coordinates" "select id from %T where ST_DWithin(ST_MakePoint(x * 1e300, y * 1e300), ST_MakePoint(0, 0), 1e300)"
	run $t "ST_DWithin, coordinates whose squares overflow" "select id from %T where ST_DWithin(ST_MakePoint(x * 1e160, y * 1e160), ST_MakePoint(0, 0), 5e160) or a = 5"
	run $t "ST_DWithin, tiny coordinates" "select id from %T where ST_DWithin(ST_MakePoint(x * 1e-200, y * 1e-200), ST_MakePoint(0, 0), 3e-200) or a = 5"
	run $t "ST_DWithin, ST_Distance on the CPU next to it" "select id, ST_Distance(ST_MakePoint(x, y), $P55) from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 1.5)"
done
runerr_pg pg_1 "ST_DWithin, a negative tolerance" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, a::float8 - 500)" "Tolerance cannot be less than zero"
runerr_pg pg_1 "ST_DWithin, -Infinity as the tolerance" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, '-Infinity'::float8)" "Tolerance cannot be less than zero"
runerr_pg pg_1 "ST_DWithin, mixed SRID (a column with SRID 4326 against SRID 0)" "select id from %T where ST_DWithin(g, $P55, 3)" "LWGEOM_dwithin: Operation on mixed SRID geometries (Point, 4326) != (Point, 0)"
runerr_pg pg_1 "ST_DWithin, mixed SRID (fused point with SRID against SRID 0)" "select id from %T where ST_DWithin(ST_SetSRID(ST_MakePoint(x, y), 3857), $P55, 3)" "LWGEOM_dwithin: Operation on mixed SRID geometries (Point, 3857) != (Point, 0)"

# Boundaries: half-integer grid points at distances exactly r (the 3-4-5 triangle, 0.5 steps), tolerances one and a few ulps
# either side of the distance, and the tolerance the plain binary64 distance gives (PostGIS's own distance is above or below
# it in 8% of the pairs: the kernel must not decide those).
for r in 0 0.5 1.5 2.5 5 5.0000000000000009 4.999999999999999 10; do
	run pg_grid "grid: ST_DWithin(fused, POINT(3 4), $r)" "select id from %T where ST_DWithin(ST_MakePoint(x, y), 'POINT(3 4)'::geometry, $r)"
	run pg_grid "grid: ST_DWithin(POINT(0 0), fused, $r)" "select id from %T where ST_DWithin('POINT(0 0)'::geometry, ST_MakePoint(x, y), $r)"
done
# The tolerance is a stored column (the plain binary64 distance, computed by SQL's separate multiply, add and sqrt), so the
# expression stays shallow enough for the VM's temporary registers and the kernel really runs.
sql "create table pg_ulpd as select id, x, y, sqrt(x * x + y * y) as d from pg_ulp"
sql "create table pg_dw as select id, x, y, a, sqrt((x - 5) * (x - 5) + (y - 5) * (y - 5)) as d from pg_1"
sql "analyze pg_ulpd"; sql "analyze pg_dw"
# coordinates of infinity and NaN in a stored column (a CASE inside the call would be too deep for the VM and stay on the CPU)
sql "create table pg_inf as select id, case a % 4 when 0 then 'Infinity'::float8 when 1 then '-Infinity'::float8 when 2 then 'NaN'::float8 else x end as x, y, a from pg_1"
sql "analyze pg_inf"
run pg_inf "ST_DWithin, point coordinates of infinity and NaN" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 3) or a = 5"
run pg_inf "ST_DWithin, infinite and NaN coordinates, tolerance Infinity" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 'Infinity'::float8) or a = 5"
for k in -3 -1 0 1 3; do
	run pg_ulpd "ulp: ST_DWithin(fused, POINT(0 0), plain distance + $k ulp)" "select id from %T where ST_DWithin(ST_MakePoint(x, y), 'POINT(0 0)'::geometry, d * (1 + $k * 2.220446049250313e-16))"
done
run pg_dw "ST_DWithin, the tolerance is the plain binary64 distance" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, d)"
run pg_dw "ST_DWithin, the tolerance is 2 ulp above the plain distance" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, d * (1 + 4.440892098500626e-16)) and a % 7 = 1"
runagg pg_grid "count of a ST_DWithin test" "select count(*) from %T where ST_DWithin(ST_MakePoint(x, y), 'POINT(3 4)'::geometry, 5)"
runagg pg_2 "count, min and max over points near a point" "select count(*), min(a), max(a) from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 4)"
runagg_any pg_2 "count by a, the points near a point" "select a, count(*) from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 4) group by a order by a"

# The recheck is the escape hatch: random pairs may hand back only the rows within 2^-47 of the boundary (almost none); the
# grid, full of pairs exactly at r, must hand back many, or the counter is not being read.
for t in pg_1 pg_2; do
	nrows=$(sql "select count(*) from $t")
	n=$(recheck_rows $t "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 3.3)")
	[ "$n" -le $((nrows / 100)) ]; check "$t: random points against ST_DWithin: $n of $nrows rows rechecked (at most 1%)" $?
	n=$(recheck_rows $t "select id from %T where ST_DWithin(ST_MakePoint(x, y), ST_MakePoint(y, x), 4.4)")
	[ "$n" -le $((nrows / 100)) ]; check "$t: two fused points: $n of $nrows rows rechecked (at most 1%)" $?
done
n=$(recheck_rows pg_grid "select id from %T where ST_DWithin(ST_MakePoint(x, y), 'POINT(3 4)'::geometry, 5)")
[ "$n" -gt 0 ]; check "pg_grid: pairs exactly at the tolerance are rechecked (counter read: $n rows)" $?
n=$(recheck_rows pg_dw "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, d)")
[ "$n" -gt 0 ]; check "pg_dw: a tolerance equal to the plain distance is rechecked (counter read: $n rows)" $?

# GPU-Cache table: the same gates as the point-in-polygon predicates (no recheck path there yet), the answers are PostGIS's.
run_any pg_gc "GPU-Cache table: fused ST_DWithin" "select id from %T where ST_DWithin(ST_MakePoint(x, y), $P55, 3)"
run_any pg_gc "GPU-Cache table: ST_DWithin on a point column" "select id from %T where ST_DWithin(g, $P55, 6)"
# the register-pressure ladder of the fused-point instructions (a program the builder refuses has no CPU fallback)
. "${0%/*}/postgis_ladder.inc"
fi
