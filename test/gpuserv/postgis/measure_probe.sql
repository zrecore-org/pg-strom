-- ST_Area, ST_NPoints, ST_NumPoints, ST_NumGeometries, ST_NRings, ST_NumInteriorRings, ST_X, ST_Y, ST_SRID, ST_IsEmpty on edge cases:
-- run on PostGIS 3.7 (psql -X -At) and keep the filtered output as measure_probe.out; the GPU path must give these answers or hand the row to the CPU.
set client_min_messages = error;
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table g(id serial, g geometry);
insert into g(g) select ST_GeomFromEWKT(w) from (values
 ('POINT(1 2)'), ('POINT Z (1 2 3)'), ('POINT M (1 2 3)'), ('POINT ZM (1 2 3 4)'), ('POINT EMPTY'), ('POINT Z EMPTY'),
 ('SRID=4326;POINT(1 2)'), ('SRID=900913;POINT(-1 2)'), ('POINT(NaN 1)'), ('POINT(-0 -0)'),
 ('MULTIPOINT((1 2),(3 4))'), ('MULTIPOINT EMPTY'), ('MULTIPOINT((1 2), EMPTY)'),
 ('LINESTRING(0 0,3 4)'), ('LINESTRING Z (0 0 0,3 4 5, 6 7 8)'), ('LINESTRING EMPTY'), ('LINESTRING(0 0,0 0)'),
 ('MULTILINESTRING((0 0,3 4),(1 1,2 2,3 3))'), ('MULTILINESTRING EMPTY'), ('MULTILINESTRING((0 0,3 4), EMPTY)'),
 ('POLYGON((0 0,4 0,4 3,0 0))'), ('POLYGON((0 0,0 3,4 3,0 0))'), ('POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,2 4,4 4,4 2,2 2))'),
 ('POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,4 2,4 4,2 4,2 2),(5 5,7 5,7 7,5 7,5 5))'),
 ('POLYGON Z ((0 0 5,10 0 6,10 10 7,0 10 8,0 0 5))'), ('POLYGON M ((0 0 5,10 0 6,10 10 7,0 10 8,0 0 5))'), ('POLYGON ZM ((0 0 5 1,10 0 6 1,10 10 7 1,0 10 8 1,0 0 5 1))'),
 ('POLYGON EMPTY'), ('POLYGON Z EMPTY'), ('POLYGON((0 0,1 0,1 1,0 0))'),
 ('POLYGON((0 0,1e15 0,1e15 1,0 1,0 0))'), ('POLYGON((1e15 1e15,1e15 1.000000000000001e15,1.000000000000001e15 1.000000000000001e15,1.000000000000001e15 1e15,1e15 1e15))'),
 ('POLYGON((1e150 0,1e150 1e150,0 1e150,0 0,1e150 0))'), ('POLYGON((1e-200 0,1e-200 1e-200,0 1e-200,0 0,1e-200 0))'),
 ('POLYGON((0 0,1 0,1 NaN,0 0))'),
 ('MULTIPOLYGON(((0 0,4 0,4 3,0 0)),((10 10,12 10,12 12,10 12,10 10),(10.5 10.5,11 10.5,11 11,10.5 10.5)))'), ('MULTIPOLYGON EMPTY'),
 ('MULTIPOLYGON(((0 0,4 0,4 3,0 0)), EMPTY)'),
 ('GEOMETRYCOLLECTION(POINT(1 2),LINESTRING(0 0,1 1),POLYGON((0 0,4 0,4 3,0 0)))'), ('GEOMETRYCOLLECTION EMPTY'),
 ('GEOMETRYCOLLECTION(POINT EMPTY)'), ('GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POLYGON((0 0,4 0,4 3,0 0)),POINT(1 1)),MULTIPOLYGON(((0 0,2 0,2 2,0 2,0 0))))'),
 ('GEOMETRYCOLLECTION(MULTIPOINT((1 2),(3 4)),MULTILINESTRING((0 0,1 1)))'),
 ('GEOMETRYCOLLECTION(GEOMETRYCOLLECTION EMPTY, POINT(1 1))'),
 ('CIRCULARSTRING(0 0,1 1,2 0)'), ('CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0))'), ('TRIANGLE((0 0,1 0,0 1,0 0))'),
 ('POLYHEDRALSURFACE(((0 0 0,0 1 0,1 1 0,0 0 0)))'), ('TIN(((0 0 0,0 1 0,1 1 0,0 0 0)))')
) v(w);
insert into g(g) values (ST_MakePoint('Infinity', '-Infinity')), (ST_MakePoint('Infinity', 1, 2)),
 (ST_MakePolygon(ST_SetPoint('LINESTRING(0 0,1 0,1 1,0 0)'::geometry, 2, ST_MakePoint(1, 'Infinity')))),
 (ST_MakePolygon(ST_SetPoint('LINESTRING(0 0,4 0,4 3,0 0)'::geometry, 1, ST_MakePoint('Infinity', 0)))),
 (ST_SetPoint('LINESTRING(0 0,1 0,1 1)'::geometry, 1, ST_MakePoint('NaN', 0)));
select id, ST_AsEWKT(g) w,
  ST_Area(g) area, ST_NPoints(g) np, ST_NumPoints(g) nump, ST_NumGeometries(g) ng, ST_NRings(g) nr, ST_NumInteriorRings(g) nir, ST_NumInteriorRing(g) nir2,
  ST_SRID(g) srid, ST_IsEmpty(g) em
from g order by id;
-- ST_X, ST_Y raise for non-points: one row at a time
create temp table xy(id int, x text, y text);
do $$ declare r record; begin for r in select id, g from g order by id loop
  begin insert into xy values (r.id, ST_X(r.g)::text, ST_Y(r.g)::text);
  exception when others then insert into xy values (r.id, 'ERROR ' || sqlerrm, null); end;
end loop; end $$;
select * from xy order by id;
select ST_SRID(ST_SetSRID('POINT(1 1)'::geometry, 0)), ST_SRID(ST_SetSRID('POINT(1 1)'::geometry, 999999)), ST_SRID(ST_SetSRID('POINT(1 1)'::geometry, -5));
select ST_Area('POLYGON((0 0,4 0,4 3,0 0))'::geometry), ST_Area('SRID=4326;POLYGON((0 0,4 0,4 3,0 0))'::geometry);
