# R3 / R4: ST_Area, ST_NPoints, ST_NumPoints, ST_NumGeometries, ST_NRings, ST_NumInteriorRings, ST_X, ST_Y, ST_SRID, ST_IsEmpty on the GPU,
# against PostGIS 3.7's own bits (postgis/measure_probe.sql, area_probe*.sql, area_twin_check.c). Sourced by postgis3d_tests.inc (so by
# postgis3d_dev.sh, postgis_dev.sh and scan_test.sh); uses run, run_any, runagg, runagg_any, q, check, sql, $PSQL, $GPU, $CPU, $fail.
# A float8 projection may be computed by the CPU above the scan, so the decisive checks are equalities in a qual against the value PostGIS
# stored when the table was built (the first model of the area passed the projections and failed these). Arrow_Fdw / Parquet have no
# geometry column type, so those sources are not applicable; heap tables and a GPU-Cache table are covered.
echo "PostGIS measures and accessors (R3, R4)"
sql "drop table if exists pg_ms"
sql "create table pg_ms as select (row_number() over ())::int4 as id, g::geometry as g, cp from (values
	('POINT(1 2)'), ('POINT Z (1 2 3)'), ('POINT M (1 2 3)'), ('POINT ZM (1 2 3 4)'), ('POINT EMPTY'), ('POINT Z EMPTY'),
	('SRID=4326;POINT(1 2)'), ('SRID=900913;POINT(-1 2)'), ('POINT(NaN 1)'), ('POINT(-0 -0)'), ('SRID=32632;POINT Z (-5 1e300 1)'),
	('MULTIPOINT((1 2),(3 4))'), ('MULTIPOINT EMPTY'), ('MULTIPOINT((1 2), EMPTY)'),
	('LINESTRING(0 0,3 4)'), ('LINESTRING Z (0 0 0,3 4 5, 6 7 8)'), ('LINESTRING EMPTY'), ('LINESTRING(0 0,0 0)'),
	('MULTILINESTRING((0 0,3 4),(1 1,2 2,3 3))'), ('MULTILINESTRING EMPTY'), ('MULTILINESTRING((0 0,3 4), EMPTY)'),
	('POLYGON((0 0,4 0,4 3,0 0))'), ('POLYGON((0 0,0 3,4 3,0 0))'), ('POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,2 4,4 4,4 2,2 2))'),
	('POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,4 2,4 4,2 4,2 2),(5 5,7 5,7 7,5 7,5 5))'), ('SRID=4326;POLYGON((0 0,10 0,10 10,0 10,0 0))'),
	('POLYGON Z ((0 0 5,10 0 6,10 10 7,0 10 8,0 0 5))'), ('POLYGON M ((0 0 5,10 0 6,10 10 7,0 10 8,0 0 5))'),
	('POLYGON ZM ((0 0 5 1,10 0 6 1,10 10 7 1,0 10 8 1,0 0 5 1))'), ('POLYGON EMPTY'), ('POLYGON Z EMPTY'),
	('POLYGON((0 0,1 0,1 1,0 0))'), ('POLYGON((0 0,1e15 0,1e15 1,0 1,0 0))'),
	('POLYGON((1e15 1e15,1e15 1.000000000000001e15,1.000000000000001e15 1.000000000000001e15,1.000000000000001e15 1e15,1e15 1e15))'),
	('POLYGON((1e150 0,1e150 1e150,0 1e150,0 0,1e150 0))'), ('POLYGON((1e-200 0,1e-200 1e-200,0 1e-200,0 0,1e-200 0))'),
	('POLYGON((0 0,1 0,1 NaN,0 0))'), ('POLYGON((0.1 0.1,0.7 0.2,0.9 0.9,0.3 0.8,0.1 0.1),(0.2 0.2,0.3 0.2,0.3 0.3,0.2 0.2))'),
	('MULTIPOLYGON(((0 0,4 0,4 3,0 0)),((10 10,12 10,12 12,10 12,10 10),(10.5 10.5,11 10.5,11 11,10.5 10.5)))'), ('MULTIPOLYGON EMPTY'),
	('MULTIPOLYGON(((0 0,4 0,4 3,0 0)), EMPTY)'),
	('GEOMETRYCOLLECTION(POINT(1 2),LINESTRING(0 0,1 1),POLYGON((0 0,4 0,4 3,0 0)))'), ('GEOMETRYCOLLECTION EMPTY'),
	('GEOMETRYCOLLECTION(POINT EMPTY)'),
	('GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POLYGON((0 0,4 0,4 3,0 0)),POINT(1 1)),MULTIPOLYGON(((0 0,2 0,2 2,0 2,0 0))))'),
	('GEOMETRYCOLLECTION(MULTIPOINT((1 2),(3 4)),MULTILINESTRING((0 0,1 1)))'), ('GEOMETRYCOLLECTION(GEOMETRYCOLLECTION EMPTY, POINT(1 1))'),
	('GEOMETRYCOLLECTION(MULTIPOLYGON(((0 0,4 0,4 3,0 0)),((10 10,12 10,12 12,10 12,10 10))),POLYGON((0 0,1 0,1 1,0 0)))'),
	('CIRCULARSTRING(0 0,1 1,2 0)'), ('CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0))'), ('TRIANGLE((0 0,1 0,0 1,0 0))'),
	('POLYHEDRALSURFACE(((0 0 0,0 1 0,1 1 0,0 0 0)))'), ('TIN(((0 0 0,0 1 0,1 1 0,0 0 0)))')
) v(w), lateral (select ST_GeomFromEWKT(w) as g) q, generate_series(1, 400) cp"
sql "insert into pg_ms select 100000 + i, g, 0 from (select row_number() over () as i, g from (values
	(ST_MakePoint('Infinity', '-Infinity')), (ST_MakePoint('Infinity', 1, 2)), (null::geometry),
	(ST_MakePolygon(ST_SetPoint('LINESTRING(0 0,1 0,1 1,0 0)'::geometry, 2, ST_MakePoint(1, 'Infinity')))),
	(ST_MakePolygon(ST_SetPoint('LINESTRING(0 0,4 0,4 3,0 0)'::geometry, 1, ST_MakePoint('Infinity', 0)))),
	(ST_SetPoint('LINESTRING(0 0,1 0,1 1)'::geometry, 1, ST_MakePoint('NaN', 0)))) v(g)) w, generate_series(1, 400)"
sql "alter table pg_ms add column area float8, add column np int4, add column nump int4, add column ng int4, add column nr int4, add column nir int4,
	add column srid int4, add column em bool, add column x float8, add column y float8"
# the CPU's answers; ST_X / ST_Y raise for a non-point, so they are filled only for points
sql "set pg_strom.enabled = off; update pg_ms set area = ST_Area(g), np = ST_NPoints(g), nump = ST_NumPoints(g), ng = ST_NumGeometries(g),
	nr = ST_NRings(g), nir = ST_NumInteriorRings(g), srid = ST_SRID(g), em = ST_IsEmpty(g)"
sql "set pg_strom.enabled = off; update pg_ms set x = ST_X(g), y = ST_Y(g) where GeometryType(g) = 'POINT'"
sql "analyze pg_ms"
pushed_t() { # table, label, function name (lower case), query-with-%T: the function is in the GPU's scan quals
	qry=$(printf '%s' "$4" | sed "s/%T/$1/g")
	pl=$($PSQL -d scantest -c "$GPU explain $qry" 2>&1 | grep "GPU Scan Quals" | grep -ci "$3")
	[ "${pl:-0}" -ge 1 ]; check "$1: $2 is in the GPU's scan quals" $?
}
sql "drop table if exists pg_msp"
sql "create table pg_msp as select id, g, x, y from pg_ms, generate_series(1, 5) k where GeometryType(g) = 'POINT'"
sql "analyze pg_msp"
# a NaN stored as NaN: compare with IS NOT DISTINCT FROM (PostgreSQL's float8 equality treats NaN = NaN, which is what the GPU does too)
run pg_ms "ST_Area equals PostGIS' stored value (bit for bit), every shape" "select id from %T where ST_Area(g) is not distinct from area and area is not null"
run pg_ms "ST_Area as a value" "select id, ST_Area(g) from %T where id % 7 = 0"
run pg_ms "ST_Area compared with a constant" "select id from %T where ST_Area(g) > 5 and ST_Area(g) < 100"
run pg_ms "ST_Area of a non-polygon is 0 (points, lines, collections of those)" "select id from %T where ST_Area(g) = 0 and ST_Length(g) > 0"
run pg_ms "ST_Area next to ST_Perimeter (a polygon with a hole)" "select id from %T where ST_Area(g) > 0 and ST_Perimeter(g) > ST_Area(g)"
runagg pg_ms "min, max and count of ST_Area" "select min(ST_Area(g)), max(ST_Area(g)), count(ST_Area(g)) from %T where id < 100000 and id % 3 <> 0 and ST_Area(g) < 1e100"
runagg_any pg_ms "GROUP BY with ST_Area as the aggregated value" "select id % 5, count(*), max(ST_Area(g)) from %T where id < 100000 and ST_Area(g) < 1e100 group by 1 order by 1"
run pg_ms "ST_NPoints equals the stored value" "select id from %T where ST_NPoints(g) = np"
run pg_ms "ST_NumPoints (lines only) equals the stored value" "select id from %T where ST_NumPoints(g) is not distinct from nump and nump is not null"
run pg_ms "ST_NumPoints is NULL for everything but lines" "select id from %T where ST_NumPoints(g) is null and g is not null"
run pg_ms "ST_NumGeometries equals the stored value" "select id from %T where ST_NumGeometries(g) = ng"
run pg_ms "ST_NRings equals the stored value" "select id from %T where ST_NRings(g) = nr"
run pg_ms "ST_NumInteriorRings equals the stored value, NULL included" "select id from %T where ST_NumInteriorRings(g) is not distinct from nir and nir is not null"
run pg_ms "ST_NumInteriorRings is NULL for everything but polygons" "select id from %T where ST_NumInteriorRings(g) is null and g is not null"
run pg_ms "ST_NumInteriorRing (the old name)" "select id, ST_NumInteriorRing(g) from %T where id % 5 = 0"
run pg_ms "ST_SRID equals the stored value" "select id from %T where ST_SRID(g) = srid"
run pg_ms "ST_SRID as a value, in a comparison" "select id, ST_SRID(g) from %T where ST_SRID(g) = 4326 or ST_SRID(g) = 900913 or ST_SRID(g) = 32632"
run pg_ms "ST_IsEmpty equals the stored value" "select id from %T where (ST_IsEmpty(g) and em) or (not ST_IsEmpty(g) and not em)"
run pg_ms "NOT ST_IsEmpty with an area" "select id from %T where not ST_IsEmpty(g) and ST_Area(g) > 1"
run pg_ms "counts as values in one projection" "select id, ST_NPoints(g), ST_NRings(g), ST_NumGeometries(g), ST_SRID(g), ST_IsEmpty(g) from %T where id % 11 = 0"
runagg pg_ms "sum of the counts (integer aggregates)" "select sum(ST_NPoints(g)), sum(ST_NRings(g)), max(ST_NumGeometries(g)), count(*) filter (where ST_IsEmpty(g)) from %T"
runagg pg_ms "GROUP BY ST_SRID" "select ST_SRID(g), count(*), sum(ST_NPoints(g)) from %T group by 1 order by 1"
runagg pg_ms "GROUP BY ST_NumGeometries" "select ST_NumGeometries(g), count(*) from %T group by 1 order by 1"
# ST_X / ST_Y: a point answers its coordinate (NaN, Infinity, -0 as stored), an empty point NULL; PostGIS raises for every other type, which the
# GPU hands to the CPU, so queries that include non-points must raise the same error: test the point rows, and the error separately
run pg_msp "ST_X equals the stored value (points)" "select id from %T where ST_X(g) = x or (ST_X(g) is null and x is null)"
run pg_msp "ST_Y equals the stored value (points)" "select id from %T where ST_Y(g) = y or (ST_Y(g) is null and y is null)"
run pg_msp "ST_X and ST_Y as values" "select id, ST_X(g), ST_Y(g) from %T where ST_X(g) is not null or id % 2 = 0"
run pg_msp "ST_X in arithmetic" "select id from %T where ST_X(g) * 2 + ST_Y(g) > 1"
runagg pg_msp "min and max of ST_X" "select min(ST_X(g)), max(ST_Y(g)) from %T where id < 100000 and ST_X(g) < 'Infinity'"
g=$($PSQL -d scantest -c "$GPU select ST_X(g) from pg_ms where GeometryType(g) = 'POLYGON' limit 1" 2>&1 | sed -n 's/^.*ERROR:  *//p' | head -1)
c=$($PSQL -d scantest -c "$CPU select ST_X(g) from pg_ms where GeometryType(g) = 'POLYGON' limit 1" 2>&1 | sed -n 's/^.*ERROR:  *//p' | head -1)
[ -n "$g" ] && [ "$g" = "$c" ]; check "ST_X of a polygon raises PostGIS' error on both paths ($g)" $?
for fn in ST_Area ST_NPoints ST_NumPoints ST_NumGeometries ST_NRings ST_NumInteriorRings ST_SRID ST_IsEmpty; do
	lc=$(echo $fn | tr 'A-Z' 'a-z')
	pushed_t pg_ms "$fn" $lc "select id from %T where $fn(g) is not null"
done
pushed_t pg_msp "ST_X" st_x "select id from %T where ST_X(g) > 0"
pushed_t pg_msp "ST_Y" st_y "select id from %T where ST_Y(g) > 0"

# the Z, M and ZM versions of the 3D table pg_3d (points, lines, polygons, collections, empties in every dimension combination)
run pg_3d "ST_Area of Z, M and ZM geometries is the 2D area" "select id from %T where ST_Area(g) > 0 and ST_Area(g) = ST_Area(ST_Force2D(g))"
run pg_3d "ST_NPoints / ST_NRings / ST_NumGeometries / ST_IsEmpty / ST_SRID of Z, M and ZM" "select id, ST_NPoints(g), ST_NRings(g), ST_NumGeometries(g), ST_IsEmpty(g), ST_SRID(g) from %T where ST_NPoints(g) >= 0 or id % 2 = 0"
run pg_3d "ST_X / ST_Y of the mixed point column (Z, M, ZM, 2D, empty, NaN, Infinity, -0)" "select id, ST_X(gp), ST_Y(gp) from %T where ST_X(gp) is not null or id % 2 = 0"
run pg_3d "ST_X / ST_Y compared" "select id from %T where ST_X(g2) > ST_Y(g2) and ST_Z(g2) < 5"

# random polygons with holes, multipolygons and collections at five magnitudes (pg_3dm of postgis3d_tests.inc, Z, and 2D every 7th)
sql "set pg_strom.enabled = off; alter table pg_3dm add column ar float8, add column np int4, add column nr int4; update pg_3dm set ar = ST_Area(g), np = ST_NPoints(g), nr = ST_NRings(g)"
sql "analyze pg_3dm"
run pg_3dm "ST_Area equals the stored value, bit for bit (random polygons, holes, multi-geometries, collections)" "select id from %T where ST_Area(g) = ar"
run pg_3dm "ST_Area as a value" "select id, ST_Area(g) from %T where id % 3 = 0"
run pg_3dm "ST_NPoints and ST_NRings equal the stored values" "select id from %T where ST_NPoints(g) = np and ST_NRings(g) = nr"
runagg pg_3dm "max of ST_Area" "select max(ST_Area(g)), count(*) filter (where ST_Area(g) > 1) from %T"

# the rings of postgis/area_probe*.sql: offsets of 1e15 and 1e16 (cancellation: PostGIS' exact path, which the GPU hands back), nearly
# collinear rings, integer rectangles, 1e150 and 1e300 coordinates (PostGIS rescales), 1e-160, rings of 3 to 20000 points
sql "drop table if exists pg_mr"
sql "create table pg_mr as select i::int4 as id, ST_MakePolygon(ring) as g, ST_MakePolygon(ring, case when i % 3 = 0 then array[hole] end) as gh from (
	select i, ST_MakeLine(array_agg(ST_MakePoint(x, y) order by k) || (array_agg(ST_MakePoint(x, y) order by k))[1]) as ring,
		ST_MakeLine(array_agg(ST_MakePoint(x * 0.5 + 0.25, y * 0.5 + 0.25) order by k) || (array_agg(ST_MakePoint(x * 0.5 + 0.25, y * 0.5 + 0.25) order by k))[1]) as hole
	from (select i, k, n0,
		case i % 10 when 0 then 1e15 + round(random() * 8) when 1 then 1e16 + round(random() * 4) * 2 when 2 then k + random() * 1e-9
			when 3 then round(random() * 10) when 4 then 1e150 * (random() * 2 - 1) when 5 then 1e-160 * (random() * 2 - 1)
			when 6 then 12345.678 + random() * 1e-3 when 7 then 1e300 * (random() * 2 - 1) when 8 then random() * 100 else xr end as x,
		case i % 10 when 0 then 1e15 + round(random() * 8) when 1 then 1e16 + round(random() * 4) * 2 when 2 then k * 0.5 + 3 + random() * 1e-12
			when 3 then round(random() * 10) when 4 then 1e150 * (random() * 2 - 1) when 5 then 1e-160 * (random() * 2 - 1)
			when 6 then 765.4321 + random() * 1e-3 when 7 then 1e300 * (random() * 2 - 1) when 8 then random() * 100 else xr * 0.3 + (case when random() < 0.5 then 0 else 1e-13 end) end as y
		from (select i, k, n0, random() * 1000 as xr from (select i, 3 + (i % 9) as n0 from generate_series(1, 3000) i) s, lateral generate_series(1, n0) k) t) u
	group by i) v"
sql "set pg_strom.enabled = off; alter table pg_mr add column ar float8, add column arh float8; update pg_mr set ar = ST_Area(g), arh = ST_Area(gh)"
sql "analyze pg_mr"
run_any pg_mr "ST_Area of random, offset, collinear and extreme rings equals the stored value" "select id from %T where ST_Area(g) is not distinct from ar"
run_any pg_mr "ST_Area of the same with holes equals the stored value" "select id from %T where ST_Area(gh) is not distinct from arh"
run_any pg_mr "ST_Area of those rings as values" "select id, ST_Area(g), ST_Area(gh) from %T"
# rings of 1 to 100,000 points (a ring of fewer than 3 points is skipped by lwpoly_area; a big one may be stored compressed or out of line,
# which the GPU hands back as a whole)
sql "drop table if exists pg_mb"
sql "create table pg_mb as select n as id, ST_MakePolygon(ring) as g, ST_MakePolygon(ring, array[ST_ExteriorRing(ST_Buffer(ST_MakePoint(0, 0), 0.3, 8))]) as gh from (
	select n, ST_MakeLine(array_agg(ST_MakePoint(cos(2 * pi() * k / n) * (1 + 0.001 * sin(k * 7.0)), sin(2 * pi() * k / n) * (1 + 0.001 * sin(k * 7.0))) order by k) || ST_MakePoint(1, 0)) as ring
	from unnest(array[3,4,5,7,10,33,100,1000,5000,20000,100000]) n, lateral generate_series(0, n - 1) k group by n) q"
sql "set pg_strom.enabled = off; alter table pg_mb add column ar float8, add column arh float8, add column np int4; update pg_mb set ar = ST_Area(g), arh = ST_Area(gh), np = ST_NPoints(g)"
sql "analyze pg_mb"
run_any pg_mb "ST_Area of rings of 3 to 100,000 points equals the stored value" "select id from %T where ST_Area(g) is not distinct from ar"
run_any pg_mb "ST_Area with a hole (the exterior alone is more than the polygon)" "select id from %T where ST_Area(gh) is not distinct from arh"
run_any pg_mb "ST_NPoints of big rings" "select id from %T where ST_NPoints(g) = np"

# mixed SRIDs do not matter for ST_Area, and a polygon column with a cached box is read through the header
run pg_ms "ST_Area of SRID 4326 polygons" "select id from %T where ST_SRID(g) = 4326 and ST_Area(g) > 0"

# a table with a GPU-Cache: the cache holds the geometry as a string; the answers are still PostGIS'
sql "drop table if exists pg_msc"
sql "create table pg_msc as select id, g, area, np, nr, srid from pg_ms where id < 100000 and cp <= 40"
sql "create trigger gc_sync after insert or update or delete on pg_msc for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=100000')"
sql "analyze pg_msc"
run_any pg_msc "GPU-Cache table: ST_Area equals the stored value" "select id from %T where ST_Area(g) is not distinct from area and area is not null"
run_any pg_msc "GPU-Cache table: ST_NPoints, ST_NRings, ST_SRID" "select id from %T where ST_NPoints(g) = np and ST_NRings(g) = nr and ST_SRID(g) = srid"
run_any pg_msc "GPU-Cache table: ST_IsEmpty and ST_NumGeometries as values" "select id, ST_IsEmpty(g), ST_NumGeometries(g) from %T"
n=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; explain select id from pg_msc where ST_Area(g) > 5" | grep -c "GPU Cache:")
[ "$n" -ge 1 ]; check "pg_msc: the plan of ST_Area reads the GPU-Cache" $?
