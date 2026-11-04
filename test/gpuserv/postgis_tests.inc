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
fi
