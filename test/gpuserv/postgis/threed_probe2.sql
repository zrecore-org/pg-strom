-- ST_3DIntersects(point, point) against ST_3DDWithin(a, b, 0), the flag functions and && vs &&& with mixed dimensions. Run:
--   ADHOC=test/gpuserv/postgis/threed_probe2.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
set client_min_messages = notice;
create or replace function try_text(sql text) returns text language plpgsql as $$
declare r text;
begin
  execute sql into r;
  return coalesce(r, 'NULL');
exception when others then
  return 'ERROR: ' || sqlerrm;
end $$;

create temp table c (id serial, a text, b text);
insert into c (a, b) values
 ('POINT Z (1 2 3)', 'POINT Z (1 2 3)'),
 ('POINT Z (1 2 3)', 'POINT Z (1 2 4)'),
 ('POINT Z (0 0 0)', 'POINT Z (-0 0 0)'),
 ('POINT Z (1e-200 0 0)', 'POINT Z (0 0 0)'),
 ('POINT Z (1e-160 0 0)', 'POINT Z (0 0 0)'),
 ('POINT Z (1e-155 0 0)', 'POINT Z (0 0 0)'),
 ('POINT Z (1 2 3)', 'POINT (1 2)'),
 ('POINT (1 2)', 'POINT (1 2)'),
 ('POINT (1 2)', 'POINT (1 3)'),
 ('POINT Z (1 2 3)', 'POINT M (1 2 3)'),
 ('POINT ZM (1 2 3 4)', 'POINT Z (1 2 3)'),
 ('POINT ZM (1 2 3 4)', 'POINT ZM (1 2 3 9)'),
 ('POINT Z EMPTY', 'POINT Z (1 2 3)'),
 ('POINT Z (NaN 2 3)', 'POINT Z (NaN 2 3)'),
 ('POINT Z (Infinity 2 3)', 'POINT Z (Infinity 2 3)'),
 ('POINT Z (1e308 0 0)', 'POINT Z (-1e308 0 0)'),
 ('SRID=4326;POINT Z (1 2 3)', 'SRID=3857;POINT Z (1 2 3)'),
 ('SRID=4326;POINT Z (1 2 3)', 'POINT Z (1 2 3)');
select id, a, b,
  try_text(format('select ST_3DIntersects(%L::geometry, %L::geometry)', a, b)) as intersects3d,
  try_text(format('select ST_3DDWithin(%L::geometry, %L::geometry, 0)', a, b)) as dwithin0
from c order by id;

-- flag functions
select a, try_text(format('select ST_HasZ(%L::geometry)', a)) hasz, try_text(format('select ST_HasM(%L::geometry)', a)) hasm,
       try_text(format('select ST_NDims(%L::geometry)', a)) ndims, try_text(format('select ST_Zmflag(%L::geometry)', a)) zmflag,
       try_text(format('select ST_CoordDim(%L::geometry)', a)) coorddim
from (values ('POINT (1 2)'), ('POINT Z (1 2 3)'), ('POINT M (1 2 3)'), ('POINT ZM (1 2 3 4)'), ('POINT EMPTY'), ('POINT Z EMPTY'),
             ('LINESTRING Z (0 0 0, 1 1 1)'), ('POLYGON M ((0 0 1, 1 0 1, 1 1 1, 0 0 1))'), ('GEOMETRYCOLLECTION EMPTY')) v(a);
select 'NULL input', ST_HasZ(NULL::geometry), ST_NDims(NULL::geometry);
select 'return types', pg_typeof(ST_HasZ('POINT Z (1 2 3)'::geometry)), pg_typeof(ST_NDims('POINT Z (1 2 3)'::geometry)),
       pg_typeof(ST_Zmflag('POINT Z (1 2 3)'::geometry)), pg_typeof(ST_CoordDim('POINT Z (1 2 3)'::geometry));

-- && (2D) and &&& (n-D) with mixed dimensions
create temp table g (id int, g geometry);
insert into g values
 (1, 'POINT (1 2)'), (2, 'POINT Z (1 2 3)'), (3, 'POINT Z (1 2 30)'), (4, 'POINT M (1 2 3)'), (5, 'POINT M (1 2 30)'),
 (6, 'POINT ZM (1 2 3 4)'), (7, 'POINT ZM (1 2 3 40)'), (8, 'POINT ZM (1 2 30 4)'), (9, 'POINT Z (5 6 3)'),
 (10, 'POINT EMPTY'), (11, 'POINT Z (1.00000001 2 3)'), (12, 'POINT (0.1 0.2)'),
 (13, 'POLYGON Z ((0 0 0, 10 0 5, 10 10 10, 0 10 5, 0 0 0))'), (14, 'POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))'),
 (15, 'POLYGON M ((0 0 0, 10 0 5, 10 10 10, 0 10 5, 0 0 0))');
select a.id aid, b.id bid, a.g && b.g as box2d, a.g &&& b.g as boxnd
from g a, g b where a.id < b.id and (a.g && b.g) is distinct from (a.g &&& b.g) order by 1, 2;
select 'pairs total', count(*) from g a, g b where a.id < b.id;
select 'pairs where &&& differs from &&', count(*) from g a, g b where a.id < b.id and (a.g && b.g) is distinct from (a.g &&& b.g);
-- float4 rounding of a point's box
select 'float4 edge', 'POINT Z (16777217 0 0)'::geometry &&& 'POINT Z (16777216 0 0)'::geometry, 'POINT Z (16777217 0 0)'::geometry && 'POINT Z (16777216 0 0)'::geometry;
