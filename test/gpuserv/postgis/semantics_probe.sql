-- A11 point-in-polygon: what the installed PostGIS answers for the cases that decide the kernel's design.
-- Run: ADHOC=test/gpuserv/postgis/semantics_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
\echo -- PostGIS version
select postgis_full_version();

create temp table poly(n text, g geometry);
insert into poly values
 ('square',      'POLYGON((0 0,10 0,10 10,0 10,0 0))'),
 ('hole',        'POLYGON((0 0,10 0,10 10,0 10,0 0),(3 3,7 3,7 7,3 7,3 3))'),
 ('bowtie',      'POLYGON((0 0,10 10,10 0,0 10,0 0))'),
 ('doubleloop',  'POLYGON((0 0,10 0,10 10,0 10,0 0,10 0,10 10,0 10,0 0))'),
 ('cw',          'POLYGON((0 0,0 10,10 10,10 0,0 0))'),
 ('multi',       'MULTIPOLYGON(((0 0,10 0,10 10,0 10,0 0)),((5 5,15 5,15 15,5 15,5 5)))'),
 ('multi_disj',  'MULTIPOLYGON(((0 0,4 0,4 4,0 4,0 0)),((6 6,10 6,10 10,6 10,6 6)))'),
 ('holeinhole',  'POLYGON((0 0,10 0,10 10,0 10,0 0),(2 2,8 2,8 8,2 8,2 2),(4 4,6 4,6 6,4 6,4 4))'),
 ('z_square',    'POLYGON Z((0 0 1,10 0 2,10 10 3,0 10 4,0 0 1))'),
 ('empty',       'POLYGON EMPTY'),
 ('srid',        'SRID=4326;POLYGON((0 0,10 0,10 10,0 10,0 0))');

create temp table pt(n text, g geometry);
insert into pt values
 ('inside',      'POINT(5 5)'),
 ('outside',     'POINT(20 20)'),
 ('edge',        'POINT(5 0)'),
 ('vertex',      'POINT(0 0)'),
 ('hole_in',     'POINT(5 5.5)'),
 ('hole_edge',   'POINT(3 5)'),
 ('hole_vertex', 'POINT(3 3)'),
 ('overlap',     'POINT(7 7)'),
 ('z_pt',        'POINT Z(5 5 100)'),
 ('empty',       'POINT EMPTY'),
 ('multipt_in',  'MULTIPOINT((5 5),(6 6))'),
 ('multipt_mix', 'MULTIPOINT((5 5),(20 20))'),
 ('srid',        'SRID=4326;POINT(5 5)'),
 ('line',        'LINESTRING(1 1,2 2)');

\echo -- n(poly), n(pt): contains, covers, intersects, within-reversed (NULL = error)
create or replace function try_bool(sql text) returns text language plpgsql as $$
declare r text;
begin
  execute sql into r;
  return coalesce(r, 'NULL');
exception when others then
  return 'ERR:' || sqlerrm;
end $$;

select p.n as poly, t.n as pt,
       try_bool(format('select st_contains(%L::geometry, %L::geometry)',   p.g, t.g)) as contains,
       try_bool(format('select st_covers(%L::geometry, %L::geometry)',     p.g, t.g)) as covers,
       try_bool(format('select st_intersects(%L::geometry, %L::geometry)', p.g, t.g)) as intersects,
       try_bool(format('select st_intersects(%L::geometry, %L::geometry)', t.g, p.g)) as intersects_rev,
       try_bool(format('select st_within(%L::geometry, %L::geometry)',     t.g, p.g)) as within,
       try_bool(format('select st_coveredby(%L::geometry, %L::geometry)',  t.g, p.g)) as coveredby
from poly p, pt t
order by p.n, t.n;
