-- A11 ST_DWithin / ST_Distance of two points: what the installed PostGIS answers for the cases that decide the kernel's design.
-- Run: ADHOC=test/gpuserv/postgis/dwithin_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
select postgis_full_version();

create or replace function try_text(sql text) returns text language plpgsql as $$
declare r text;
begin
  execute sql into r;
  return coalesce(r, 'NULL');
exception when others then
  return 'ERR[' || sqlstate || ']:' || sqlerrm;
end $$;

\echo -- shapes: 2D/3D/M point, empty, other SRID, multipoint, line; ST_DWithin(a, b, 5) and ST_Distance(a, b)
create temp table pt(n text, g geometry);
insert into pt values
 ('p00',      'POINT(0 0)'),
 ('p34',      'POINT(3 4)'),
 ('p34z',     'POINT Z(3 4 100)'),
 ('p34m',     'POINT M(3 4 100)'),
 ('p34zm',    'POINT ZM(3 4 100 7)'),
 ('empty',    'POINT EMPTY'),
 ('srid',     'SRID=4326;POINT(3 4)'),
 ('multi',    'MULTIPOINT((3 4),(100 100))'),
 ('line',     'LINESTRING(3 4,100 100)'),
 ('big',      'POINT(1e308 1e308)'),
 ('tiny',     'POINT(1e-200 1e-200)');
insert into pt values ('nan', st_makepoint('NaN'::float8, 0)), ('inf', st_makepoint('Infinity'::float8, 0)), ('ninf', st_makepoint(0, '-Infinity'::float8));
select a.n as a, b.n as b,
       try_text(format('select st_dwithin(%L::geometry, %L::geometry, 5)', a.g, b.g)) as dwithin5,
       try_text(format('select st_distance(%L::geometry, %L::geometry)', a.g, b.g)) as dist
from pt a, pt b where a.n = 'p00' order by b.n;
select a.n as a, b.n as b,
       try_text(format('select st_dwithin(%L::geometry, %L::geometry, 5)', a.g, b.g)) as dwithin5,
       try_text(format('select st_distance(%L::geometry, %L::geometry)', a.g, b.g)) as dist
from pt a, pt b where a.n in ('srid', 'empty', 'p34z') and b.n in ('p00', 'p34', 'srid', 'empty') order by a.n, b.n;

\echo -- tolerance: zero, negative, NULL, NaN, Infinity, huge
select tol, try_text(format('select st_dwithin(%L::geometry, %L::geometry, %s::float8)', 'POINT(0 0)', 'POINT(3 4)', tol)) as r
from unnest(array['0','-1','-0','5','4.999999999999999','5.000000000000001','NULL','''NaN''','''Infinity''','''-Infinity''','1e308']) tol;
select try_text('select st_dwithin(''POINT(0 0)''::geometry, ''POINT(0 0)''::geometry, 0)') as same_point_zero_tol;

\echo -- exactly at the boundary: (3,4) vs 5; (0.3,0.4) vs 0.5; 1 ulp either side of r
select try_text('select st_dwithin(''POINT(0 0)''::geometry, ''POINT(0.3 0.4)''::geometry, 0.5)') as at_half;
select try_text('select st_distance(''POINT(0 0)''::geometry, ''POINT(0.3 0.4)''::geometry)::text') as dist_half;

\echo -- does the CPU answer equal sqrt(dx*dx+dy*dy) <= r (separate multiply, add, sqrt) over many near-boundary pairs?
create temp table g as
  select i, x, y, (random()*2-1)*10 as qx, (random()*2-1)*10 as qy from (
    select i, (random()*2-1)*10 as x, (random()*2-1)*10 as y from generate_series(1, 300000) i) s;
select count(*) as n,
       count(*) filter (where st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) <> sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y))) as dist_differs_plain,
       count(*) filter (where st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) <> sqrt(((qx-x)*(qx-x)) + ((qy-y)*(qy-y)))) as dist_differs_plain2,
       count(*) filter (where st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) < sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y))) as dist_below_plain,
       count(*) filter (where st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) > sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y))) as dist_above_plain,
       max(abs(st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) - sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y))) / nullif(sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y)), 0)) as max_rel_diff
from g;
-- r chosen as the CPU distance itself, +-1ulp-ish: ST_DWithin(a,b,d) must be true; and with d - tiny false.
select count(*) as n,
       count(*) filter (where not st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), st_distance(st_makepoint(x, y), st_makepoint(qx, qy)))) as not_within_own_distance,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) * (1 - 1e-15))) as within_just_below,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y)))) as within_plain
from g;
-- is the comparison on the distance or on its square? Compare squared threshold.
select count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), 3.0) <> ((qx-x)*(qx-x) + (qy-y)*(qy-y) <= 9.0)) as differs_from_squared,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), 3.0) <> (sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y)) <= 3.0)) as differs_from_sqrt
from g;
-- same, on integer-valued coordinates and a grid of tolerances so that many pairs sit exactly on the boundary
create temp table h as select a.x as x, a.y as y, b.x as qx, b.y as qy from
  (select x, y from generate_series(-20, 20) x, generate_series(-20, 20) y) a,
  (select x, y from generate_series(-3, 3) x, generate_series(-3, 3) y) b;
select count(*) as n,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), 5.0) <> ((qx-x)*(qx-x) + (qy-y)*(qy-y) <= 25.0)) as grid_differs
from h;

\echo -- boundary: is the test on d <= r or d*d <= r*r? pairs whose d*d <= r*r is true but sqrt(d*d) > r (and the reverse)
create temp table b as select x, y, qx, qy, sqrt((qx-x)*(qx-x) + (qy-y)*(qy-y)) as d from g;
select count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), d)) as within_plain_d,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), d) <> (st_distance(st_makepoint(x, y), st_makepoint(qx, qy)) <= d)) as differs_from_distance_le_r,
       count(*) filter (where st_dwithin(st_makepoint(x, y), st_makepoint(qx, qy), d) <> ((qx-x)*(qx-x) + (qy-y)*(qy-y) <= d*d)) as differs_from_squared
from b;

\echo -- geography and the 3D variants are different functions
select try_text('select st_dwithin(''POINT(0 0)''::geography, ''POINT(0 0.00001)''::geography, 2)') as geog;
select try_text('select st_3ddwithin(''POINT Z(0 0 0)''::geometry, ''POINT Z(3 4 100)''::geometry, 5)') as three_d;
select proname, pg_get_function_identity_arguments(oid) as args from pg_proc where proname in ('st_dwithin','st_distance') and pronamespace = 'public'::regnamespace order by 1, 2;
