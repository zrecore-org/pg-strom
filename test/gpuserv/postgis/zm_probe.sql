-- 3D: what the installed PostGIS answers for ST_Z / ST_M, the 3- and 4-argument ST_MakePoint and ST_3DDWithin of two points
-- (the cases that decide the kernels). Run:
--   ADHOC=test/gpuserv/postgis/zm_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
select postgis_full_version();
set client_min_messages = notice;

create or replace function try_text(sql text) returns text language plpgsql as $$
declare r text;
begin
  execute sql into r;
  return coalesce(r, 'NULL');
exception when others then
  return 'ERR[' || sqlstate || ']:' || sqlerrm;
end $$;

\echo -- ST_Z / ST_M: Z, M, ZM, 2D, empties, NaN, SRID, non-points, NULL
create temp table pt(n text, g geometry);
insert into pt values
 ('p2d',    'POINT(1 2)'),
 ('pz',     'POINT Z(1 2 3)'),
 ('pm',     'POINT M(1 2 4)'),
 ('pzm',    'POINT ZM(1 2 3 4)'),
 ('ez',     'POINT Z EMPTY'),
 ('e2',     'POINT EMPTY'),
 ('em',     'POINT M EMPTY'),
 ('srid',   'SRID=4326;POINT Z(1 2 3)'),
 ('multi',  'MULTIPOINT Z((1 2 3),(4 5 6))'),
 ('line',   'LINESTRING Z(1 2 3,4 5 6)'),
 ('poly',   'POLYGON Z((0 0 1,1 0 1,1 1 1,0 0 1))'),
 ('coll',   'GEOMETRYCOLLECTION Z(POINT Z(1 2 3))'),
 ('null',   NULL);
insert into pt values ('nanz', st_makepoint(1, 2, 'NaN'::float8)), ('infz', st_makepoint(1, 2, 'Infinity'::float8)), ('negz', st_makepoint(1, 2, -0.0::float8)),
 ('bigz', st_makepoint(1, 2, 1e308)), ('tinyz', st_makepoint(1, 2, 5e-324)), ('nanx', st_makepoint('NaN'::float8, 2, 3));
select n, try_text(format('select st_z(%L::geometry)::text', g)) as st_z, try_text(format('select st_m(%L::geometry)::text', g)) as st_m,
       try_text(format('select st_ndims(%L::geometry)::text', g)) as ndims
from pt order by n;

\echo -- ST_MakePoint with 3 and 4 arguments: NULL arguments, NaN, and the 2D operators on the result
select try_text('select st_astext(st_makepoint(1, 2, NULL))') as z_null,
       try_text('select st_astext(st_makepoint(NULL, 2, 3))') as x_null,
       try_text('select st_astext(st_makepoint(1, 2, 3, NULL))') as m_null,
       try_text('select st_astext(st_makepoint(1, 2, ''NaN''::float8))') as z_nan,
       try_text('select st_astext(st_makepoint(1, 2, 3, 4))') as zm,
       try_text('select st_z(st_makepoint(1, 2, 3))') as z_of_makepoint,
       try_text('select (st_makepoint(1, 2, NULL) && st_makepoint(1, 2))::text') as overlaps_null_z,
       try_text('select st_contains(st_makeenvelope(0, 0, 5, 5), st_makepoint(1, 2, NULL))') as contains_null_z,
       try_text('select st_contains(st_makeenvelope(0, 0, 5, 5), st_makepoint(1, 2, 7))') as contains_z,
       try_text('select st_dwithin(st_makepoint(1, 2, 3), st_makepoint(1, 2, 90), 0.5)') as dwithin_ignores_z,
       try_text('select st_dwithin(st_makepoint(1, 2, NULL), st_makepoint(1, 2), 0.5)') as dwithin_null_z,
       try_text('select st_contains(st_makeenvelope(0, 0, 5, 5), st_makepoint(1, 2, 1/0.0))') as z_division_by_zero;

\echo -- ST_3DDWithin(a, b, 5) of two points: shapes
insert into pt values ('p00', 'POINT(0 0)'), ('p34', 'POINT(3 4)'), ('p345z', 'POINT Z(3 4 0)'), ('p00z', 'POINT Z(0 0 0)'), ('p3412', 'POINT Z(0 0 5)'),
 ('p34m', 'POINT M(3 4 100)'), ('p34zm', 'POINT ZM(3 4 0 100)'), ('srid2', 'SRID=3857;POINT Z(3 4 0)');
select a.n as a, b.n as b, try_text(format('select st_3ddwithin(%L::geometry, %L::geometry, 5)', a.g, b.g)) as d3_5,
       try_text(format('select st_dwithin(%L::geometry, %L::geometry, 5)', a.g, b.g)) as d2_5,
       try_text(format('select st_3ddistance(%L::geometry, %L::geometry)', a.g, b.g)) as dist3
from pt a, pt b where a.n in ('p00', 'p00z', 'p3412', 'srid') and b.n in ('p34', 'p345z', 'p3412', 'p34m', 'p34zm', 'p00z', 'ez', 'e2', 'srid', 'srid2', 'multi', 'line', 'null', 'nanz', 'infz', 'nanx')
order by a.n, b.n;

\echo -- tolerance: zero, negative, NULL, NaN, Infinity, huge
select tol, try_text(format('select st_3ddwithin(%L::geometry, %L::geometry, %s::float8)', 'POINT Z(0 0 0)', 'POINT Z(3 4 0)', tol)) as r
from unnest(array['0','-1','-0','5','4.999999999999999','5.000000000000001','NULL','''NaN''','''Infinity''','''-Infinity''','1e308']) tol;
select try_text('select st_3ddwithin(''POINT Z(1 1 1)''::geometry, ''POINT Z(1 1 1)''::geometry, 0)') as same_point_zero_tol;
select try_text('select st_3ddwithin(''POINT Z(0 0 0)''::geometry, ''POINT Z(1 2 2)''::geometry, 3)') as pythagoras_1_2_2;
select try_text('select st_3ddwithin(''POINT Z(0 0 0)''::geometry, ''POINT Z(0.3 0.4 0)''::geometry, 0.5)') as at_half;

\echo -- overflow of the squares and extreme coordinates
select try_text('select st_3ddwithin(''POINT Z(1e200 0 0)''::geometry, ''POINT Z(0 0 0)''::geometry, 1e200)') as sq_overflow_eq,
       try_text('select st_3ddwithin(''POINT Z(1e200 1e200 1e200)''::geometry, ''POINT Z(0 0 0)''::geometry, 1.8e200)') as sq_overflow_lt,
       try_text('select st_3ddwithin(''POINT Z(1e-200 1e-200 1e-200)''::geometry, ''POINT Z(0 0 0)''::geometry, 1e-200)') as sq_underflow;

\echo -- does the CPU answer equal sqrt(dx*dx+dy*dy+dz*dz) <= r (plain operations) over many pairs, and where does it differ?
create temp table g as
  select i, x, y, z, (random()*2-1)*10 as qx, (random()*2-1)*10 as qy, (random()*2-1)*10 as qz from (
    select i, (random()*2-1)*10 as x, (random()*2-1)*10 as y, (random()*2-1)*10 as z from generate_series(1, 300000) i) s;
select count(*) as n,
       count(*) filter (where st_3ddistance(st_makepoint(x, y, z), st_makepoint(qx, qy, qz)) <> sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y) + (qz-z)*(qz-z))) as dist3_differs_plain,
       max(abs(st_3ddistance(st_makepoint(x, y, z), st_makepoint(qx, qy, qz)) - sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y) + (qz-z)*(qz-z))) / nullif(sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y) + (qz-z)*(qz-z)), 0)) as max_rel_diff
from g;
-- r = the plain distance of the pair (the most delicate case) and a hair either side
with d as (select x, y, z, qx, qy, qz, sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y) + (qz-z)*(qz-z)) as r from g)
select count(*) as n,
       count(*) filter (where st_3ddwithin(st_makepoint(x, y, z), st_makepoint(qx, qy, qz), r)) as true_at_r,
       count(*) filter (where st_3ddwithin(st_makepoint(x, y, z), st_makepoint(qx, qy, qz), r * (1 - 1e-15))) as true_below_1e15,
       count(*) filter (where st_3ddwithin(st_makepoint(x, y, z), st_makepoint(qx, qy, qz), r * (1 + 1e-15))) as true_above_1e15,
       count(*) filter (where st_3ddwithin(st_makepoint(x, y, z), st_makepoint(qx, qy, qz), r * (1 - 1e-13))) as true_below_1e13,
       count(*) filter (where st_3ddwithin(st_makepoint(x, y, z), st_makepoint(qx, qy, qz), r * (1 + 1e-13))) as true_above_1e13
from d;
