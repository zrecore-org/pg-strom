# PostGIS in 3D: every GPU feature of postgis_tests.inc against Z, M and ZM geometries; the 3D functions that run on the GPU
# (ST_Z, ST_M, the 3- and 4-argument ST_MakePoint, ST_3DDWithin of two points); and the 3D-only functions that do not
# (ST_3DDistance, ST_3DLength ...), which must stay on the CPU and give the CPU's answer; and ST_3DIntersects of two points, &&& and the
# flag functions (ST_NDims, ST_HasZ, ST_HasM, ST_Zmflag, ST_CoordDim), which run on the GPU.
# Sourced by postgis_tests.inc (so by scan_test.sh and postgis_dev.sh); uses run, run_any, runagg_any, q, check, sql, $PSQL, $GPU,
# $CPU, $JOINSET, $fail. The 2D operators (&&, ~, @, ST_Contains and friends, ST_DWithin) ignore Z and M, as PostGIS does for the
# 2D forms.
echo "PostGIS 3D (Z, M, ZM)"
PG3=${PG3ROWS:-100000}

# g: points, lines, polygons (with and without a cached box), multipolygons, collections, empties, in Z, M and ZM; g2: 3D points
sql "drop table if exists pg_3d"
sql "create table pg_3d as select i::int4 as id,
	case i % 26
		when 0 then null
		when 1 then ST_MakePoint(bx, by, zz)
		when 2 then ST_MakePointM(bx, by, mm)
		when 3 then ST_MakePoint(bx, by, zz, mm)
		when 4 then ST_GeomFromEWKT('POINT Z EMPTY')
		when 5 then ST_Force3DZ(ST_MakeLine(ARRAY[ST_MakePoint(bx, by), ST_MakePoint(bx + 4, by + 3), ST_MakePoint(bx + 1, by + 7)]), zz)
		when 6 then ST_Force3DM(ST_MakeLine(ARRAY[ST_MakePoint(bx, by), ST_MakePoint(bx + 4, by + 3), ST_MakePoint(bx + 1, by + 7)]), mm)
		when 7 then ST_Force4D(ST_MakeLine(ARRAY[ST_MakePoint(bx, by), ST_MakePoint(bx + 4, by + 3), ST_MakePoint(bx + 1, by + 7)]), zz, mm)
		when 8 then ST_Force3DZ(ST_MakeLine(ST_MakePoint(bx, by), ST_MakePoint(bx + 5, by + 2)), zz)
		when 9 then ST_Force3DZ(ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(bx, by), ST_MakePoint(bx + 5, by), ST_MakePoint(bx, by + 5), ST_MakePoint(bx, by)])), zz)
		when 10 then ST_Force3DZ(ST_MakeEnvelope(bx, by, bx + 6, by + 5), zz)
		when 11 then ST_Force3DM(ST_MakeEnvelope(bx, by, bx + 6, by + 5), mm)
		when 12 then ST_Force4D(ST_MakeEnvelope(bx, by, bx + 6, by + 5), zz, mm)
		when 13 then ST_Force3DZ(ST_MakePolygon(ST_ExteriorRing(ST_MakeEnvelope(bx, by, bx + 8, by + 8)), ARRAY[ST_ExteriorRing(ST_MakeEnvelope(bx + 2, by + 2, bx + 5, by + 5))]), zz)
		when 14 then ST_Force3DZ(ST_Collect(ST_MakeEnvelope(bx, by, bx + 3, by + 3), ST_MakeEnvelope(bx + 5, by + 5, bx + 9, by + 8)), zz)
		when 15 then ST_Force3DZ(ST_Collect(ARRAY[ST_MakePoint(bx, by), ST_MakePoint(bx + 2, by + 1), ST_MakePoint(bx + 3, by + 9)]), zz)
		when 16 then ST_Force3DZ(ST_Collect(ST_MakeEnvelope(bx, by, bx + 4, by + 4), ST_MakePoint(bx + 7, by + 7)), zz)
		when 17 then ST_GeomFromEWKT('POLYGON Z EMPTY')
		when 18 then ST_MakePoint(1e38 * ((i % 7) - 3), by, 1e300)
		when 19 then ST_MakePoint(bx, by, 'NaN'::float8)
		when 20 then ST_Force3DZ(ST_MakeEnvelope(bx - 100, by - 100, bx + 100, by + 100), zz)
		else ST_MakePoint(bx, by, zz)
	end as g,
	case when i % 11 = 0 then null else ST_MakePoint(bx + 1, by + 2, zz) end as g2,
	case i % 10
		when 0 then null
		when 1 then ST_MakePoint(bx, by, zz)
		when 2 then ST_MakePointM(bx, by, mm)
		when 3 then ST_MakePoint(bx, by, zz, mm)
		when 4 then ST_MakePoint(bx, by)
		when 5 then ST_GeomFromEWKT('POINT Z EMPTY')
		when 6 then ST_GeomFromEWKT('POINT M EMPTY')
		when 7 then ST_MakePoint(bx, by, 'NaN'::float8)
		when 8 then ST_MakePoint(bx, by, 'Infinity'::float8)
		else ST_MakePoint(bx, by, -0.0::float8)
	end as gp,
	case when i % 5 = 0 then null else zz end as zn, case when i % 3 = 0 then null else mm end as mn,
	bx::float8 as x, by::float8 as y, zz as z, (i % 1000)::int4 as a
	from (select i, ((i * 37) % 400) / 10.0 as bx, ((i * 53) % 400) / 10.0 as by, (i % 9)::float8 as zz, (i % 5)::float8 as mm
		  from generate_series(1, $PG3) i) s"
sql "analyze pg_3d"
n3=$($PSQL -d scantest -c "select count(*) filter (where ST_NDims(g) = 3), count(*) filter (where ST_NDims(g) = 4), count(*) filter (where ST_HasM(g) and not ST_HasZ(g)) from pg_3d" 2>&1 | tail -1)
echo "  info  pg_3d: rows with 3 dimensions, with 4, M only: $n3"

BOXP="'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry"
BOXZ="ST_Force3DZ('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 7)"
BOXM="ST_Force3DM('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 7)"
BOXZM="ST_Force4D('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 7, 3)"
BIGZ="ST_Force3DZ('POLYGON((-100 -100,100 -100,100 100,-100 100,-100 -100))'::geometry, -2)"

# the box operators ignore Z and M, with 2D, Z, M and ZM operands on both sides
for c in "2d:$BOXP" "z:$BOXZ" "m:$BOXM" "zm:$BOXZM"; do
	nm=${c%%:*}; pg=${c#*:}
	run pg_3d "&& a $nm polygon" "select id from %T where g && $pg"
	run pg_3d "~ a $nm polygon" "select id from %T where g ~ $pg"
	run pg_3d "@ a big $nm polygon" "select id from %T where g @ ST_Force3DZ($BIGZ, 1)"
done
run pg_3d "&& a Z point" "select id from %T where g && ST_MakePoint(5, 5, 9)"
run pg_3d "&& a ZM point" "select id from %T where g && ST_MakePoint(5, 5, 9, 4)"
run pg_3d "&& between two columns (g, a 3D point)" "select id from %T where g && g2"
run pg_3d "~ and @ between two columns" "select id from %T where g ~ g2 or g @ g2"
run pg_3d "NOT (g && Z polygon)" "select id from %T where not (g && $BOXZ)"
run pg_3d "&& and a projection of the 3D geometry" "select id, g, z from %T where g && $BOXZ and a % 4 = 0"

# point in polygon: the polygon is a Z, M or ZM constant or a column (Z/M/ZM polygons, with or without a cached box)
for c in "2d:$BOXP" "z:$BOXZ" "m:$BOXM" "zm:$BOXZM" "bigz:$BIGZ"; do
	nm=${c%%:*}; pg=${c#*:}
	run pg_3d "ST_Contains($nm polygon, ST_MakePoint(x, y))" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
	run pg_3d "ST_Covers($nm polygon, ST_MakePoint(x, y))" "select id from %T where ST_Covers($pg, ST_MakePoint(x, y))"
	run pg_3d "ST_Intersects(ST_MakePoint(x, y), $nm polygon)" "select id from %T where ST_Intersects(ST_MakePoint(x, y), $pg)"
	run pg_3d "ST_Within(ST_MakePoint(x, y), $nm polygon)" "select id from %T where ST_Within(ST_MakePoint(x, y), $pg)"
done
run pg_3d "ST_Contains(the geometry column, ST_MakePoint(x, y))" "select id from %T where ST_Contains(g, ST_MakePoint(x, y))"
run pg_3d "ST_Intersects(the geometry column, ST_MakePoint(x, y))" "select id from %T where ST_Intersects(g, ST_MakePoint(x, y))"
run pg_3d "ST_Within(ST_MakePoint(x, y), the geometry column)" "select id from %T where ST_Within(ST_MakePoint(x, y), g)"
run pg_3d "ST_Covers(the geometry column, a Z point of g2)" "select id from %T where ST_Covers(g, g2)"
run pg_3d "ST_Contains($BOXZ, g2)" "select id from %T where ST_Contains($BOXZ, g2)"
run pg_3d "ST_Contains with the 3-argument ST_MakePoint" "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, z))"
run pg_3d "ST_Contains with the 4-argument ST_MakePoint" "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, z, 3))"
run pg_3d "ST_Contains(3-argument ST_MakePoint with a NULL Z)" "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, zn))"
run pg_3d "NOT ST_Contains(3-argument ST_MakePoint with a NULL Z)" "select id from %T where not ST_Contains($BOXZ, ST_MakePoint(x, y, zn))"
run pg_3d "ST_Contains(4-argument ST_MakePoint with a NULL M)" "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, z, mn))"
run pg_3d "ST_Intersects(3-argument ST_MakePoint, polygon)" "select id from %T where ST_Intersects(ST_MakePoint(x, y, z), $BOXP)"

# ST_DWithin ignores Z (the 2D form), so a 3D point column and a 3-argument point give the 2D answer
run pg_3d "ST_DWithin(g2, ST_MakePoint(x, y), 3)" "select id from %T where ST_DWithin(g2, ST_MakePoint(x, y), 3)"
run pg_3d "ST_DWithin(g, ST_MakePoint(20, 20), 9)" "select id from %T where ST_DWithin(g, ST_MakePoint(20, 20), 9)"
run pg_3d "ST_DWithin(g2, ST_MakePoint(x, y, z), 4)" "select id from %T where ST_DWithin(g2, ST_MakePoint(x, y, z), 4)"
run pg_3d "ST_DWithin(two fused 3D points)" "select id from %T where ST_DWithin(ST_MakePoint(x, y, z), ST_MakePoint(20, 20, 9), 8)"
run pg_3d "ST_DWithin(fused point with a NULL Z)" "select id from %T where ST_DWithin(ST_MakePoint(x, y, zn), ST_MakePoint(20, 20), 8)"

# 3D-only functions are not on the GPU: results equal the CPU's, and the function is not in the GPU's scan quals
refused() { # label, function name (lower case), query-with-%T
	qry=$(printf '%s' "$3" | sed "s/%T/pg_3d/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	pl=$($PSQL -d scantest -c "$GPU explain $qry" 2>&1 | grep "GPU Scan Quals" | grep -ci "$2")
	if [ "$g" = "$c" ] && [ "${pl:-0}" -eq 0 ] && ! echo "$g" | grep -q ERROR; then echo "  ok    pg_3d: $1 (${g#*|} rows, not on the GPU)"
	else echo "  FAIL  pg_3d: $1 (in the GPU quals: ${pl:-0})"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
pushed() { # label, function name (lower case), query-with-%T: the function is in the GPU's scan quals
	qry=$(printf '%s' "$3" | sed "s/%T/pg_3d/g")
	pl=$($PSQL -d scantest -c "$GPU explain $qry" 2>&1 | grep "GPU Scan Quals" | grep -ci "$2")
	[ "${pl:-0}" -ge 1 ]; check "pg_3d: $1 is in the GPU's scan quals" $?
}
# ST_Distance, ST_3DDistance and ST_3DMaxDistance of two points return PostGIS' own float8: sqrt(fma(dz, dz, fma(dx, dx, dy*dy))) (3D) or
# sqrt(fma(dx, dx, dy*dy)) (2D), which the GPU reproduces bit for bit with the software binary64 (postgis/dist_probe.c: 200,000 pairs; plain
# binary64 would differ in 10.9% of them). The planner calibrates the function against the installed PostGIS before it pushes it down.
# The length of a line is the same segment formula added left to right (postgis/len_probe.c: 60,000 lines, no difference).
run pg_3d "ST_3DDistance(Z point column, fused 3-argument point) < 5" "select id from %T where ST_3DDistance(g2, ST_MakePoint(x, y, z)) < 5"
run pg_3d "ST_3DDistance as a value" "select id, ST_3DDistance(g2, ST_MakePoint(x, y, z)) from %T where id % 7 = 0"
run pg_3d "ST_3DDistance of two columns, compared with a column" "select id from %T where ST_3DDistance(g2, g2) <= z"
run_any pg_3d "ST_3DDistance(mixed point column, Z point column) (2D, empty, NaN and M points are re-checked)" "select id, ST_3DDistance(gp, g2) from %T"
run pg_3d "ST_Distance (2D) of a Z point column and a fused 2-argument point" "select id, ST_Distance(g2, ST_MakePoint(x, y)) from %T where a < 900"
run pg_3d "ST_Distance ignores Z (fused 3-argument point)" "select id from %T where ST_Distance(g2, ST_MakePoint(x, y, z)) < 3"
run pg_3d "ST_3DMaxDistance of two points" "select id, ST_3DMaxDistance(g2, ST_MakePoint(x, y, z)) from %T where id % 5 = 0"
runagg pg_3d "min and max of ST_3DDistance (a float8 aggregate over the exact distance)" "select min(ST_3DDistance(g2, ST_MakePoint(x, y, z))), max(ST_3DDistance(g2, ST_MakePoint(x, y, z))), count(ST_3DDistance(g2, ST_MakePoint(x, y, z))) from %T"
run pg_3d "ST_3DDistance with arithmetic" "select id, ST_3DDistance(g2, ST_MakePoint(x, y, z)) * 2 + 1 from %T where ST_3DDistance(g2, ST_MakePoint(x, y, z)) between 1 and 4"
pushed "ST_3DDistance" st_3ddistance "select id from %T where ST_3DDistance(g2, ST_MakePoint(x, y, z)) < 5"
pushed "ST_Distance" st_distance "select id from %T where ST_Distance(g2, ST_MakePoint(x, y)) < 5"
run pg_3d "ST_3DLength of every geometry (lines on the GPU; multi-geometries and collections re-checked)" "select id from %T where ST_3DLength(g) > 6"
run pg_3d "ST_3DLength, ST_Length and ST_Length2D as values" "select id, ST_3DLength(g), ST_Length(g), ST_Length2D(g) from %T where id % 3 = 0"
run pg_3d "ST_Length compared with ST_3DLength" "select id from %T where ST_3DLength(g) > ST_Length(g)"
run pg_3d "ST_3DLength of the mixed point column (always 0)" "select id from %T where ST_3DLength(gp) = 0"
runagg pg_3d "max of ST_3DLength" "select max(ST_3DLength(g)), count(*) filter (where ST_Length(g) > 3) from %T"
pushed "ST_3DLength" st_3dlength "select id from %T where ST_3DLength(g) > 6"
pushed "ST_Length" st_length "select id from %T where ST_Length(g) > 6"

# the edge lines of postgis/dist_probe2.out, many copies of each: empty, one repeated point, overflow, NaN, underflow, 2D, M, ZM, multi-lines
sql "drop table if exists pg_3dl"
sql "create table pg_3dl as select (row_number() over ())::int4 as id, g::geometry as g from (values
	('LINESTRING Z (0 0 0, 3 4 12)'), ('LINESTRING Z (0 0 0, 3 4 12, 3 4 12, 10 10 10)'), ('LINESTRING (0 0, 3 4, 6 8)'),
	('LINESTRING M (0 0 1, 3 4 2)'), ('LINESTRING ZM (0 0 0 1, 3 4 12 2)'), ('LINESTRING Z EMPTY'), ('LINESTRING EMPTY'),
	('LINESTRING Z (1 1 1, 1 1 1)'), ('LINESTRING Z (0 0 0, 1e308 0 0)'), ('LINESTRING Z (0 0 0, 1e200 1e200 1e200)'),
	('LINESTRING Z (0 0 0, NaN 0 0)'), ('LINESTRING Z (1e-200 0 0, 0 0 0)'), ('LINESTRING Z (1e-160 1e-160 1e-160, 0 0 0)'),
	('MULTILINESTRING Z ((0 0 0, 3 4 12), (1 1 1, 2 2 2))'), ('POINT Z (1 2 3)'), ('POLYGON Z ((0 0 0, 4 0 0, 4 3 0, 0 0 0))'),
	('GEOMETRYCOLLECTION Z (LINESTRING Z (0 0 0, 3 4 12), POINT Z (1 1 1))'), ('SRID=4326;LINESTRING Z (0 0 0, 3 4 12)'),
	('LINESTRING Z (0.1 0.2 0.3, 0.4 0.5 0.6, 0.7 0.9 0.2, 1.3 1.1 0.8, 2.1 0.4 0.5, 0.3 0.3 0.3, 9.9 8.8 7.7)')) v(g),
	generate_series(1, 2500)"
sql "set pg_strom.enabled = off; alter table pg_3dl add column l3 float8, add column l2 float8; update pg_3dl set l3 = ST_3DLength(g), l2 = ST_Length(g)"
sql "analyze pg_3dl"
run_any pg_3dl "ST_3DLength equals the stored value over the edge lines" "select id from %T where ST_3DLength(g) is not distinct from l3 and l3 is not null"
run_any pg_3dl "ST_Length equals the stored value over the edge lines" "select id from %T where ST_Length(g) is not distinct from l2 and l2 is not null"
run_any pg_3dl "ST_3DLength over the edge lines" "select id, ST_3DLength(g) from %T"
run_any pg_3dl "ST_Length over the edge lines" "select id, ST_Length(g) from %T"
run_any pg_3dl "ST_Length2D over the edge lines" "select id, ST_Length2D(g) from %T"
run_any pg_3dl "ST_Perimeter over the edge lines (always 0 for lines)" "select id, ST_Perimeter(g), ST_3DPerimeter(g) from %T"
run_any pg_3dl "the edge lengths compared" "select id from %T where ST_3DLength(g) > 10 or ST_Length(g) = 5"

# lengths and perimeters of multi-geometries: PostGIS adds each member's own total to the container's running total, a polygon's rings in
# order; random lines, polygons with holes, multi-lines, multi-polygons and mixed collections at several magnitudes (Z, and 2D every 7th)
sql "create or replace function pg3d_line(n int, sc float8) returns geometry language sql as
	'select ST_MakeLine(array_agg(ST_MakePoint((random()*2-1)*sc, (random()*2-1)*sc, (random()*2-1)*sc))) from generate_series(1, n)'"
sql "create or replace function pg3d_ring(n int, sc float8) returns geometry language sql as
	'select ST_MakePolygon(ST_AddPoint(l, ST_StartPoint(l))) from (select pg3d_line(n, sc) l) q'"
sql "create or replace function pg3d_hole(sc float8) returns geometry language sql as
	'select ST_ExteriorRing(pg3d_ring(4, sc))'"
sql "drop table if exists pg_3dm"
sql "create table pg_3dm as select id, case when id % 7 = 0 then ST_Force2D(g) else g end as g from (select i::int4 as id,
	case i % 8
		when 0 then ST_Collect(ARRAY[pg3d_line(2 + i % 4, sc), pg3d_line(3, sc), pg3d_line(2 + i % 3, sc)])
		when 1 then ST_MakePolygon(ST_ExteriorRing(pg3d_ring(4 + i % 3, sc)), ARRAY[pg3d_hole(sc), pg3d_hole(sc)])
		when 2 then ST_Collect(pg3d_ring(4 + i % 3, sc), ST_MakePolygon(ST_ExteriorRing(pg3d_ring(5, sc)), ARRAY[pg3d_hole(sc)]))
		when 3 then ST_Collect(ARRAY[pg3d_line(3, sc), pg3d_ring(4, sc), ST_Collect(pg3d_line(2, sc), pg3d_line(4, sc))])
		when 4 then ST_Collect(pg3d_line(2 + i % 5, sc), pg3d_line(2 + i % 3, sc))
		when 5 then pg3d_line(2 + i % 9, sc)
		when 6 then ST_Collect(ARRAY[pg3d_ring(4, sc), pg3d_line(3, sc), ST_MakePoint(sc, sc, sc)])
		else ST_Collect(ARRAY[ST_MakePoint(1, 2, 3), ST_MakePoint(4, 5, 6)])
	end as g
	from (select i, case i % 5 when 0 then 1000.0 when 1 then 1e-3 when 2 then 1e8 when 3 then 1.0 else 1e-120 end as sc
	      from generate_series(1, 30000) i) s) t"
sql "set pg_strom.enabled = off; alter table pg_3dm add column l2 float8, add column l3 float8, add column p2 float8, add column p3 float8;
	update pg_3dm set l2 = ST_Length(g), l3 = ST_3DLength(g), p2 = ST_Perimeter(g), p3 = ST_3DPerimeter(g)"
sql "analyze pg_3dm"
run pg_3dm "ST_Length equals PostGIS' stored value, bit for bit" "select id from %T where ST_Length(g) = l2"
run pg_3dm "ST_3DLength equals the stored value" "select id from %T where ST_3DLength(g) = l3"
run pg_3dm "ST_Perimeter equals the stored value" "select id from %T where ST_Perimeter(g) = p2"
run pg_3dm "ST_3DPerimeter equals the stored value" "select id from %T where ST_3DPerimeter(g) = p3"
for fn in ST_Length ST_Length2D ST_3DLength ST_Perimeter ST_Perimeter2D ST_3DPerimeter; do
	run pg_3dm "$fn of random lines, polygons with holes, multi-geometries and collections" "select id, $fn(g) from %T where id % 2 = 0"
done
run pg_3dm "ST_3DLength and ST_3DPerimeter compared" "select id from %T where ST_3DLength(g) > ST_3DPerimeter(g) and ST_Length(g) > 0"
runagg pg_3dm "max of ST_3DPerimeter" "select max(ST_3DPerimeter(g)), count(*) filter (where ST_Perimeter(g) > 1) from %T"
pushed "ST_Perimeter" st_perimeter "select id from %T where ST_Perimeter(g) > 6"
pushed "ST_3DPerimeter" st_3dperimeter "select id from %T where ST_3DPerimeter(g) > 6"

# ST_Distance of a point and a line, a polygon (with holes), a multi-geometry or a collection (2D): the minimum over the members of the distance
# to the vertices and to the feet of the perpendiculars, computed with the fused operations PostGIS' build contracts them to; a point near an
# edge or on a vertex of a polygon is re-checked on the CPU (the winding test cannot tell), on a line vertex it is exact
sql "drop table if exists pg_3dp"
sql "create table pg_3dp as select id, g, ST_MakePoint(px, py, pz) as pt, px, py, pz,
	case when id % 4 = 0 then (select geom from ST_DumpPoints(g) limit 1) else ST_MakePoint(px, py) end as pv from (
	select id, g, (random()*2-1)*sc*1.2 as px, (random()*2-1)*sc*1.2 as py, (random()*2-1)*sc as pz from (
	select id, g, case id % 5 when 0 then 1000.0 when 1 then 1e-3 when 2 then 1e8 when 3 then 1.0 else 1e-120 end as sc from pg_3dm) a) b"
sql "set pg_strom.enabled = off; alter table pg_3dp add column dist float8, add column distv float8; update pg_3dp set dist = ST_Distance(g, pt), distv = ST_Distance(g, pv)"
sql "analyze pg_3dp"
# A float8 in a projection may be computed by the CPU above the scan, so the decisive check is an equality in a qual (always evaluated by the
# GPU) against the value PostGIS computed when the table was built: any difference in the last bit loses rows (the first model of the
# segment distance passed every other test here and failed this one).
run pg_3dp "ST_Distance equals PostGIS' stored value, bit for bit" "select id from %T where ST_Distance(g, pt) = dist"
run pg_3dp "ST_Distance to a vertex equals the stored value" "select id from %T where ST_Distance(g, pv) = distv"
run pg_3dp "ST_Distance differs from the stored value by nothing" "select id from %T where ST_Distance(pt, g) <> dist"
run pg_3dp "ST_Distance(geometry, point column)" "select id, ST_Distance(g, pt) from %T where id % 2 = 0"
run pg_3dp "ST_Distance(point column, geometry)" "select id, ST_Distance(pt, g) from %T where id % 2 = 1"
run pg_3dp "ST_Distance(geometry, fused 2-argument point)" "select id, ST_Distance(g, ST_MakePoint(px, py)) from %T where id % 3 = 0"
run pg_3dp "ST_Distance(geometry, fused 3-argument point) (Z is ignored)" "select id, ST_Distance(g, ST_MakePoint(px, py, pz)) from %T where id % 3 = 1"
run pg_3dp "ST_Distance to a point that is a vertex of the geometry (on a polygon: re-checked)" "select id, ST_Distance(g, pv) from %T where id % 5 <> 4"
run pg_3dp "ST_Distance compared" "select id from %T where ST_Distance(g, pt) < 100 or ST_Distance(g, pv) = 0"
run_any pg_3dp "ST_3DDistance of a point and a non-point (the CPU)" "select id, ST_3DDistance(g, pt) from %T where id % 5 = 0"
runagg pg_3dp "min and max of ST_Distance to the geometry" "select min(ST_Distance(g, pt)), max(ST_Distance(g, pt)), count(*) filter (where ST_Distance(g, pv) > 0) from %T"
pushed "ST_Distance (point to geometry)" st_distance "select id from %T where ST_Distance(g, g2) < 100"

# the edge pairs of postgis/dist_probe2.out, many copies of each: empty, NaN, overflow, underflow, -0, the same point, 2D operands, M, ZM
sql "drop table if exists pg_3dd"
sql "create table pg_3dd as select (row_number() over ())::int4 as id, a::geometry as ga, b::geometry as gb from (values
	('POINT Z (1 2 3)', 'POINT Z (4 6 8)'), ('POINT Z (1 2 3)', 'POINT Z (1 2 3)'), ('POINT (1 2)', 'POINT (4 6)'),
	('POINT Z (1 2 3)', 'POINT (4 6)'), ('POINT (1 2)', 'POINT Z (4 6 8)'), ('POINT M (1 2 9)', 'POINT M (4 6 9)'),
	('POINT ZM (1 2 3 4)', 'POINT ZM (4 6 8 9)'), ('POINT Z EMPTY', 'POINT Z (1 2 3)'), ('POINT EMPTY', 'POINT (1 2)'),
	('POINT Z (NaN 2 3)', 'POINT Z (1 2 3)'), ('POINT Z (1 2 3)', 'POINT Z (1 2 NaN)'), ('POINT Z (1e308 0 0)', 'POINT Z (-1e308 0 0)'),
	('POINT Z (1e-200 0 0)', 'POINT Z (0 0 0)'), ('POINT Z (1e-160 0 0)', 'POINT Z (0 0 0)'), ('POINT Z (-0 0 0)', 'POINT Z (0 -0 0)'),
	('SRID=4326;POINT Z (1 2 3)', 'SRID=4326;POINT Z (4 6 8)'), ('POINT Z (1e154 1e154 1e154)', 'POINT Z (-1e154 -1e154 -1e154)'),
	('POINT Z (1e-170 1e-170 1e-170)', 'POINT Z (-1e-170 -1e-170 -1e-170)'), ('POINT Z (3 4 0)', 'POINT Z (0 0 12)')) v(a, b),
	generate_series(1, 2500)"
sql "set pg_strom.enabled = off; alter table pg_3dd add column d3 float8, add column d2 float8; update pg_3dd set d3 = ST_3DDistance(ga, gb), d2 = ST_Distance(ga, gb)"
sql "analyze pg_3dd"
run_any pg_3dd "ST_3DDistance equals the stored value over the edge pairs (NULL, NaN, overflow and underflow rows are the CPU's)" "select id from %T where ST_3DDistance(ga, gb) is not distinct from d3 and d3 is not null"
run_any pg_3dd "ST_Distance equals the stored value over the edge pairs" "select id from %T where ST_Distance(ga, gb) is not distinct from d2 and d2 is not null"
run_any pg_3dd "ST_3DDistance over the edge pairs" "select id, ST_3DDistance(ga, gb) from %T"
run_any pg_3dd "ST_Distance over the edge pairs" "select id, ST_Distance(ga, gb) from %T"
run_any pg_3dd "ST_3DMaxDistance over the edge pairs" "select id, ST_3DMaxDistance(ga, gb) from %T"
run_any pg_3dd "the edge distances compared" "select id from %T where ST_3DDistance(ga, gb) < 6 or ST_Distance(ga, gb) = 5"

# ST_3DIntersects of two points = ST_3DDWithin(a, b, 0) (postgis/threed_probe2.out); a point without Z, an empty point, another SRID,
# NaN and the underflow range go back to the CPU
run pg_3d "ST_3DIntersects(Z point column, fused 3-argument point), all equal" "select id from %T where ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z))"
run pg_3d "ST_3DIntersects(Z point column, fused point one apart in Z)" "select id from %T where ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z + 1))"
run pg_3d "ST_3DIntersects(Z point column, fused point, a few equal)" "select id from %T where ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z + (id % 7) / 6))"
run_any pg_3d "ST_3DIntersects(mixed point column, Z point column) (2D, empty, NaN and M points are re-checked)" "select id from %T where ST_3DIntersects(gp, g2)"
run pg_3d "NOT ST_3DIntersects" "select id from %T where g2 is not null and not ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z + 1))"
pushed "ST_3DIntersects" st_3dintersects "select id from %T where ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z))"

# &&& (n-D boxes): the 2D boxes, and Z or M ranges when both operands have them (postgis/threed_probe2.out)
BOXZ4="ST_Force4D(ST_MakeEnvelope(0, 0, 10, 10), 3, 3)"
run pg_3d "&&& a ZM polygon (the earlier CPU-only case)" "select id from %T where g &&& $BOXZ4"
run pg_3d "&&& a Z polygon" "select id from %T where g &&& $BOXZ"
run pg_3d "&&& an M polygon" "select id from %T where g &&& $BOXM"
run pg_3d "&&& a 2D polygon" "select id from %T where g &&& $BOXP"
run pg_3d "&&& a Z point" "select id from %T where g &&& ST_MakePoint(5, 5, 4)"
run pg_3d "&&& an M point" "select id from %T where g &&& ST_MakePointM(5, 5, 2)"
run pg_3d "&&& a ZM point" "select id from %T where g &&& ST_MakePoint(5, 5, 4, 2)"
run pg_3d "&&& between two columns (g, a 3D point)" "select id from %T where g &&& g2"
run_any pg_3d "&&& between two columns (mixed points, 2D, NaN, empties)" "select id from %T where gp &&& g2"
run pg_3d "NOT (g &&& Z polygon)" "select id from %T where not (g &&& $BOXZ)"
run pg_3d "&& and &&& together" "select id from %T where g && $BOXP and g &&& $BOXZ4"
run pg_3d "&&& in a projection" "select id, g &&& $BOXZ4, g2 &&& $BOXM from %T where id % 5 = 0"
pushed "&&&" "&&&" "select id from %T where g &&& $BOXZ4"

# the flag functions read the header of any geometry
run pg_3d "ST_NDims = 3" "select id from %T where ST_NDims(g) = 3"
run pg_3d "ST_NDims = 4" "select id from %T where ST_NDims(g) = 4"
run pg_3d "ST_HasZ" "select id from %T where ST_HasZ(g)"
run pg_3d "ST_HasM" "select id from %T where ST_HasM(g)"
run pg_3d "ST_HasM and not ST_HasZ" "select id from %T where ST_HasM(g) and not ST_HasZ(g)"
run pg_3d "ST_Zmflag" "select id from %T where ST_Zmflag(g) = 2"
run pg_3d "ST_CoordDim of the point column" "select id from %T where ST_CoordDim(gp) = 3"
run pg_3d "flags in a projection (empty and NULL geometries included)" "select id, ST_NDims(g), ST_HasZ(gp), ST_HasM(gp), ST_Zmflag(g), ST_CoordDim(g2) from %T where id % 3 = 0"
run pg_3d "flags with arithmetic" "select id, ST_NDims(g) * 10 + ST_Zmflag(g) from %T where ST_NDims(g) + ST_Zmflag(gp) > 4"
pushed "ST_NDims" st_ndims "select id from %T where ST_NDims(g) = 3"
pushed "ST_HasZ" st_hasz "select id from %T where ST_HasZ(g)"
pushed "ST_Zmflag" st_zmflag "select id from %T where ST_Zmflag(g) = 2"
refused "ST_Force2D(g) && polygon" st_force2d "select id from %T where ST_Force2D(g) && $BOXP"

# ---- 3D functions on the GPU: ST_Z, ST_M, ST_3DDWithin ----
# gp: points of every dimension (Z, M, ZM, 2D, empties, NaN / Infinity / -0 Z); g2: Z points; zn / mn: Z / M with NULLs
run pg_3d "ST_Z of a point column in a qual" "select id from %T where ST_Z(gp) > 3"
run pg_3d "ST_M of a point column in a qual" "select id from %T where ST_M(gp) > 2"
run pg_3d "ST_Z IS NULL (2D and M points, empties, NULLs)" "select id from %T where ST_Z(gp) is null"
run pg_3d "ST_M IS NOT NULL" "select id from %T where ST_M(gp) is not null"
run pg_3d "ST_Z, ST_M in the projection" "select id, ST_Z(gp), ST_M(gp), ST_Z(g2) from %T where a % 7 = 0"
run pg_3d "arithmetic on ST_Z and ST_M" "select id, ST_Z(gp) * 2 + ST_M(gp) from %T where ST_Z(gp) between 1 and 5"
run pg_3d "ST_Z of a constant point" "select id from %T where ST_Z('POINT Z(1 2 3)'::geometry) < z"
run pg_3d "ST_Z compared with a column (NaN and Infinity Z)" "select id from %T where ST_Z(gp) >= z"
runagg_any pg_3d "aggregates of ST_Z and ST_M" "select count(ST_Z(gp)), max(ST_Z(gp)), min(ST_M(gp)), sum(ST_Z(g2)) from %T"
# a non-point raises PostgreSQL's error, from the backend (the row is handed back), in either path
for fn in ST_Z ST_M; do
	qry="select $fn(g) from pg_3d where id <= 40 and g is not null"
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	[ "$g" = "$c" ] && echo "$g" | grep -q "must have type POINT"; check "$fn of a line or polygon raises PostGIS' error ($c)" $?
done
pushed "ST_Z" st_z "select id from %T where ST_Z(gp) > 3"
pushed "ST_M" st_m "select id from %T where ST_M(gp) > 2"
pushed "ST_3DDWithin (column, fused 3-argument point)" st_3ddwithin "select id from %T where ST_3DDWithin(g2, ST_MakePoint(x, y, z + 1), 3)"
pushed "ST_3DDWithin (two fused points)" st_3ddwithin "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(20, 20, 3), 15)"
pushed "ST_Contains (3-argument fused point)" st_contains "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, z))"
pushed "ST_DWithin (4-argument fused point)" st_dwithin "select id from %T where ST_DWithin(g2, ST_MakePoint(x, y, z, 3), 4)"
refused "ST_Z of a fused point (no register to read)" st_z "select id from %T where ST_Z(ST_MakePoint(x, y, z)) > 3"

run pg_3d "ST_3DDWithin(Z point column, fused 3-argument point)" "select id from %T where ST_3DDWithin(g2, ST_MakePoint(x, y, z + 1), 3)"
run pg_3d "ST_3DDWithin(fused 4-argument point, Z point column)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z, 7), g2, 2.5)"
run pg_3d "ST_3DDWithin(two fused points)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(20, 20, 3), 15)"
run pg_3d "ST_3DDWithin(column, constant Z point)" "select id from %T where ST_3DDWithin(g2, 'POINT Z(20 20 3)'::geometry, 12)"
run pg_3d "ST_3DDWithin(mixed points, constant): the backend does what is not 3D" "select id from %T where ST_3DDWithin(gp, 'POINT Z(20 20 3)'::geometry, 15)"
run pg_3d "ST_3DDWithin with the tolerance from a column" "select id from %T where ST_3DDWithin(g2, ST_MakePoint(20, 20, 3), z * 3)"
run pg_3d "ST_3DDWithin(fused point with a NULL Z)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, zn), ST_MakePoint(20, 20, 3), 15)"
run pg_3d "NOT ST_3DDWithin" "select id from %T where not ST_3DDWithin(g2, ST_MakePoint(x, y, z), 2)"
refused "ST_3DDWithin of a 2D fused point (PostGIS raises a NOTICE)" st_3ddwithin "select id from %T where ST_3DDWithin(g2, ST_MakePoint(x, y), 3)"
for ec in "select count(*) from pg_3d where ST_3DDWithin(g2, 'POINT Z(1 1 1)'::geometry, -1)|Tolerance cannot be less than zero" \
		  "select count(*) from pg_3d where ST_3DDWithin(ST_SetSRID(g2, 4326), g2, 3)|mixed SRID"; do
	qry=${ec%%|*}; msg=${ec#*|}
	# (the first message ends with a newline, so look for the text in the whole output, not its last line)
	g=$($PSQL -d scantest -c "$GPU $qry" 2>&1 | grep -c "$msg"); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | grep -c "$msg")
	[ "$g" -ge 1 ] && [ "$c" -ge 1 ]; check "ST_3DDWithin raises PostGIS' error ($msg) from the backend" $?
done
# the decision band: pairs at the distance r itself and a hair either side (PostGIS' distance is not bit-reproducible)
sql "drop table if exists pg_3dn"
sql "create table pg_3dn as select i::int4 as id, x, y, z, qx, qy, qz, sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y) + (qz-z)*(qz-z)) as r,
	ST_3DDistance(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz)) as d3, ST_Distance(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz)) as d2 from (
	select i, (random()*2-1)*10 as x, (random()*2-1)*10 as y, (random()*2-1)*10 as z, (random()*2-1)*10 as qx, (random()*2-1)*10 as qy,
	       (random()*2-1)*10 as qz from generate_series(1, 50000) i) s"
sql "analyze pg_3dn"
for f in "1" "1 - 1e-15" "1 + 1e-15" "1 - 1e-13" "1 + 1e-13" "1 - 1e-9" "1 + 1e-9"; do
	run pg_3dn "ST_3DDWithin at r * ($f)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), r * ($f))"
done
# the tolerance is PostGIS' own distance of the pair: every row is exactly on the boundary (the GPU decides "distance <= tolerance" exactly
# when the installed PostGIS does, instead of re-checking a band around it); a few ulps below and above
D3=d3        # PostGIS' own distance of the pair, computed by the CPU when the table was built (columns d3 and d2)
D2=d2
run pg_3dn "ST_3DDistance equals PostGIS' stored value, bit for bit" "select id from %T where ST_3DDistance(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz)) = d3"
run pg_3dn "ST_Distance equals the stored value" "select id from %T where ST_Distance(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz)) = d2"
run pg_3dn "ST_3DMaxDistance equals the stored 3D distance" "select id from %T where ST_3DMaxDistance(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz)) = d3"
for f in "1" "1 - 1.2e-16" "1 + 2.3e-16" "1 - 3.4e-16" "1 + 1.1e-16"; do
	run pg_3dn "ST_3DDWithin at the stored ST_3DDistance * ($f)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), $D3 * ($f))"
	run pg_3dn "ST_DWithin at the stored ST_Distance * ($f)" "select id from %T where ST_DWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), $D2 * ($f))"
done
run pg_3dn "NOT ST_3DDWithin just below the stored distance" "select id from %T where not ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), $D3 * (1 - 1.2e-16))"
# exact Pythagorean quadruples: the distance is an integer, tolerance = the distance is true, a hair below is false
sql "drop table if exists pg_3dq"
sql "create table pg_3dq as select (row_number() over ())::int4 as id, 0::float8 as x, 0::float8 as y, 0::float8 as z, (d[1] * m)::float8 as qx, (d[2] * m)::float8 as qy,
	(d[3] * m)::float8 as qz, (d[4] * m)::float8 as r from (values (array[1,2,2,3]), (array[2,3,6,7]), (array[3,4,12,13]), (array[4,4,7,9]), (array[6,6,7,11])) q(d),
	(values (1), (2), (8), (1024), (3)) mm(m)"
run pg_3dq "ST_3DDWithin at exactly r (a Pythagorean quadruple)" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), r)"
run pg_3dq "ST_3DDWithin a hair below r" "select id from %T where ST_3DDWithin(ST_MakePoint(x, y, z), ST_MakePoint(qx, qy, qz), r * (1 - 1e-15))"

# the GPU-Cache table: ST_Z / ST_3DDWithin are answered correctly (the cache does not take geometry predicates yet)
sql "drop table if exists pg_3dc"
sql "create table pg_3dc as select id, g2 as g, x, y, z from pg_3d where g2 is not null and id <= 30000"
sql "create trigger gc_sync3 after insert or update or delete on pg_3dc for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=90000')"
sql "analyze pg_3dc"
run_any pg_3dc "GPU-Cache table: ST_Z in a qual" "select id from %T where ST_Z(g) > 3"
run_any pg_3dc "GPU-Cache table: ST_Z in the projection" "select id, ST_Z(g), x from %T where id % 9 = 0"
run_any pg_3dc "GPU-Cache table: ST_3DDWithin" "select id from %T where ST_3DDWithin(g, ST_MakePoint(x, y, z), 3)"
# a join qual: no geometry in a join program, so the backend does it
sql "drop table if exists pg_3dj"
sql "create table pg_3dj as select id, id % 50 as k from pg_3d where id <= 2000"
for qry in "select count(*) from pg_3d t join pg_3dj j on t.id = j.id and ST_Z(t.gp) > 2" \
		   "select count(*) from pg_3d t join pg_3dj j on t.id = j.id and ST_3DDWithin(t.g2, ST_MakePoint(20, 20, 3), 15)"; do
	g=$(q "$GPU $JOINSET" "$qry"); c=$(q "$CPU" "$qry")
	[ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; check "a 3D function in a join qual gives the CPU's answer (${g#*|})" $?
done

# the register-pressure ladder with the Z (and M) registers a fused 3D point keeps alive (a program the builder cannot build has no
# CPU fallback, so the planner's estimate must refuse such a qual itself: every rung runs on the GPU and matches, or plans on the CPU)
LT1="abs(x) / 4 + 0.5"
LT2="abs(x * 0.5 + y * 0.25) + 0.5"
LT3="sqrt(abs(x * 0.5 + y * 0.25)) + 0.5"
LT4="sqrt(abs(x * 0.5 + y * 0.25) + x * x) + 0.5"
LT5="round(sqrt(abs(x * 0.5 + y * 0.25) + x * x) * 4) / 4 + ceil(abs(y)) * 0.1"
i=0
for tol in "$LT1" "$LT2" "$LT3" "$LT4" "$LT5"; do
	i=$((i+1))
	run_any pg_3d "3D ladder $i: ST_3DDWithin(two computed fused 3D points, computed tolerance)" "select id from %T where ST_3DDWithin(ST_MakePoint(x + 1, y + 1, z), ST_MakePoint(y - 1, x - 1, z + 2), $tol)"
	run_any pg_3d "3D ladder $i: ST_3DDWithin(computed fused 4-argument point, constant, computed tolerance)" "select id from %T where ST_3DDWithin(ST_MakePoint(x * 0.5 + y, y * 0.5 - x, abs(z) + 1, z), 'POINT Z(25 10 5)'::geometry, $tol)"
	run_any pg_3d "3D ladder $i: ST_DWithin(two computed fused 4-argument points, computed tolerance)" "select id from %T where ST_DWithin(ST_MakePoint(x + 1, y + 1, z, x), ST_MakePoint(y - 1, x - 1, zn, y), $tol)"
	run_any pg_3d "3D ladder $i: ST_Contains(computed 3-argument point) and ST_3DDWithin together" "select id from %T where ST_Contains($BOXP, ST_MakePoint(x + 1, y + 1, zn)) and ST_3DDWithin(ST_MakePoint(x + 1, y + 1, z), g2, $tol)"
done
run_any pg_3d "3D ladder: ST_Z of a column next to two fused 3D points" "select id from %T where ST_Z(gp) > 1 and ST_3DDWithin(ST_MakePoint(x + 1, y + 1, z), ST_MakePoint(y - 1, x - 1, z + 2), 9)"

# a 3D inner geometry in a GiST join: the index key is the 2D box, the point comes from x and y
if [ "${GISTJOIN3D:-on}" = on ]; then
	sql "drop table if exists pg_3dp; drop table if exists pg_3dz"
	sql "create table pg_3dz as select id, g from pg_3d where g is not null and id <= 3000 and id % 26 <> 20"
	sql "create index pg_3dz_gist on pg_3dz using gist (g)"
	sql "create table pg_3dp as select id, x, y from pg_3d where id <= 40000"
	sql "analyze pg_3dz; analyze pg_3dp"
	GISTON="set pg_strom.enable_gpugistindex=on; set enable_seqscan=on;"
	JQ3="select o.id, i.id from pg_3dp o join pg_3dz i on i.g && st_makepoint(o.x, o.y)"
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON explain $JQ3")
	pj=$(echo "$plan" | grep -c "GPU GiST Join")
	g=$(q "$GPU $JOINSET $GISTON" "$JQ3"); c=$(q "$CPU" "$JQ3")
	if [ "$pj" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    pg_3d: GiST join over Z, M and ZM geometries (${g#*|} rows)"
	else echo "  FAIL  pg_3d: GiST join over Z, M and ZM geometries (GPU GiST Join=$pj)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
fi
