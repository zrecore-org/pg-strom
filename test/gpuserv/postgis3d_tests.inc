# PostGIS in 3D: every GPU feature of postgis_tests.inc against Z, M and ZM geometries, and the 3D-only functions, which must stay
# on the CPU and give the CPU's answer. Sourced by postgis_tests.inc (so by scan_test.sh and postgis_dev.sh); uses run, run_any, q, check,
# sql, $PSQL, $GPU, $CPU, $JOINSET, $fail. The 2D operators (&&, ~, @, ST_Contains and friends, ST_DWithin) ignore Z and M, as
# PostGIS does for the 2D forms; ST_3DDWithin, ST_3DDistance, ST_3DIntersects, &&& and the accessors are not on the GPU.
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
run_any pg_3d "ST_Contains with the 3-argument ST_MakePoint" "select id from %T where ST_Contains($BOXZ, ST_MakePoint(x, y, z))"

# ST_DWithin ignores Z (the 2D form), so a 3D point column and a 3-argument point give the 2D answer
run pg_3d "ST_DWithin(g2, ST_MakePoint(x, y), 3)" "select id from %T where ST_DWithin(g2, ST_MakePoint(x, y), 3)"
run pg_3d "ST_DWithin(g, ST_MakePoint(20, 20), 9)" "select id from %T where ST_DWithin(g, ST_MakePoint(20, 20), 9)"
run_any pg_3d "ST_DWithin(g2, ST_MakePoint(x, y, z), 4)" "select id from %T where ST_DWithin(g2, ST_MakePoint(x, y, z), 4)"

# 3D-only functions are not on the GPU: results equal the CPU's, and the function is not in the GPU's scan quals
refused() { # label, function name (lower case), query-with-%T
	qry=$(printf '%s' "$3" | sed "s/%T/pg_3d/g")
	g=$(q "$GPU" "$qry"); c=$(q "$CPU" "$qry")
	pl=$($PSQL -d scantest -c "$GPU explain $qry" 2>&1 | grep "GPU Scan Quals" | grep -ci "$2")
	if [ "$g" = "$c" ] && [ "${pl:-0}" -eq 0 ] && ! echo "$g" | grep -q ERROR; then echo "  ok    pg_3d: $1 (${g#*|} rows, not on the GPU)"
	else echo "  FAIL  pg_3d: $1 (in the GPU quals: ${pl:-0})"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
refused "ST_3DDWithin" st_3ddwithin "select id from %T where ST_3DDWithin(g2, ST_MakePoint(x, y, z + 1), 3)"
refused "ST_3DDistance" st_3ddistance "select id from %T where ST_3DDistance(g2, ST_MakePoint(x, y, z)) < 5"
refused "ST_3DIntersects" st_3dintersects "select id from %T where ST_3DIntersects(g2, ST_MakePoint(x + 1, y + 2, z))"
refused "&&& (n-D bounding boxes)" "&&&" "select id from %T where g &&& ST_Force4D(ST_MakeEnvelope(0, 0, 10, 10), 3, 3)"
refused "ST_Z of a point" st_z "select id from %T where case when GeometryType(g) = 'POINT' and not ST_IsEmpty(g) then ST_Z(g) end > 3"
refused "ST_M of a point" st_m "select id from %T where case when GeometryType(g) = 'POINTM' then ST_M(g) end > 2"
refused "ST_NDims" st_ndims "select id from %T where ST_NDims(g) = 3"
refused "ST_HasZ" st_hasz "select id from %T where ST_HasZ(g)"
refused "ST_3DLength" st_3dlength "select id from %T where case when GeometryType(g) like 'LINESTRING%' then ST_3DLength(g) end > 6"
refused "ST_Force2D(g) && polygon" st_force2d "select id from %T where ST_Force2D(g) && $BOXP"

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
