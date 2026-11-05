-- Edge cases of the distance and length functions of points and lines (what the kernel must hand back to the CPU). Run:
--   ADHOC=test/gpuserv/postgis/dist_probe2.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
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
 ('POINT Z (1 2 3)', 'POINT Z (4 6 8)'),
 ('POINT Z (1 2 3)', 'POINT Z (1 2 3)'),
 ('POINT (1 2)', 'POINT (4 6)'),
 ('POINT Z (1 2 3)', 'POINT (4 6)'),
 ('POINT (1 2)', 'POINT Z (4 6 8)'),
 ('POINT M (1 2 9)', 'POINT M (4 6 9)'),
 ('POINT ZM (1 2 3 4)', 'POINT ZM (4 6 8 9)'),
 ('POINT Z EMPTY', 'POINT Z (1 2 3)'),
 ('POINT EMPTY', 'POINT (1 2)'),
 ('POINT Z (NaN 2 3)', 'POINT Z (1 2 3)'),
 ('POINT Z (1 2 3)', 'POINT Z (1 2 NaN)'),
 ('POINT Z (1e308 0 0)', 'POINT Z (-1e308 0 0)'),
 ('POINT Z (1e-200 0 0)', 'POINT Z (0 0 0)'),
 ('POINT Z (1e-160 0 0)', 'POINT Z (0 0 0)'),
 ('POINT Z (-0 0 0)', 'POINT Z (0 -0 0)'),
 ('SRID=4326;POINT Z (1 2 3)', 'SRID=3857;POINT Z (4 6 8)'),
 ('SRID=4326;POINT Z (1 2 3)', 'POINT Z (4 6 8)'),
 ('SRID=4326;POINT Z (1 2 3)', 'SRID=4326;POINT Z (4 6 8)'),
 ('POINT Z (1 2 3)', 'LINESTRING Z (0 0 0, 5 5 5)'),
 ('MULTIPOINT Z ((1 2 3), (4 5 6))', 'POINT Z (0 0 0)');
select id, a, b,
  try_text(format('select ST_3DDistance(%L::geometry, %L::geometry)', a, b)) d3,
  try_text(format('select ST_Distance(%L::geometry, %L::geometry)', a, b)) d2,
  try_text(format('select ST_3DMaxDistance(%L::geometry, %L::geometry)', a, b)) dmax
from c order by id;

create temp table l (id serial, g text);
insert into l (g) values
 ('LINESTRING Z (0 0 0, 3 4 12)'),
 ('LINESTRING Z (0 0 0, 3 4 12, 3 4 12, 10 10 10)'),
 ('LINESTRING (0 0, 3 4, 6 8)'),
 ('LINESTRING M (0 0 1, 3 4 2)'),
 ('LINESTRING ZM (0 0 0 1, 3 4 12 2)'),
 ('LINESTRING Z EMPTY'),
 ('LINESTRING EMPTY'),
 ('LINESTRING Z (1 1 1, 1 1 1)'),
 ('LINESTRING Z (0 0 0, 1e308 0 0)'),
 ('LINESTRING Z (0 0 0, 1e200 1e200 1e200)'),
 ('LINESTRING Z (0 0 0, NaN 0 0)'),
 ('LINESTRING Z (1e-200 0 0, 0 0 0)'),
 ('MULTILINESTRING Z ((0 0 0, 3 4 12), (1 1 1, 2 2 2))'),
 ('POINT Z (1 2 3)'),
 ('POLYGON Z ((0 0 0, 4 0 0, 4 3 0, 0 0 0))'),
 ('GEOMETRYCOLLECTION Z (LINESTRING Z (0 0 0, 3 4 12), POINT Z (1 1 1))'),
 ('SRID=4326;LINESTRING Z (0 0 0, 3 4 12)');
select id, g,
  try_text(format('select ST_3DLength(%L::geometry)', g)) l3,
  try_text(format('select ST_Length(%L::geometry)', g)) l2,
  try_text(format('select ST_Length2D(%L::geometry)', g)) l2d
from l order by id;
