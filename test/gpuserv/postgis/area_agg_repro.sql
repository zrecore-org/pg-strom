create extension if not exists postgis;
set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0;
drop table if exists ar;
create table ar as select (row_number() over ())::int4 as id, w, ST_GeomFromEWKT(w) as g, c from (values
 ('POINT(1 2)'),('POINT EMPTY'),('LINESTRING(0 0,3 4)'),('POLYGON((0 0,4 0,4 3,0 0))'),
 ('POLYGON((0 0,1e15 0,1e15 1,0 1,0 0))'),
 ('POLYGON((1e15 1e15,1e15 1.000000000000001e15,1.000000000000001e15 1.000000000000001e15,1.000000000000001e15 1e15,1e15 1e15))'),
 ('POLYGON((1e150 0,1e150 1e150,0 1e150,0 0,1e150 0))'),('POLYGON((1e-200 0,1e-200 1e-200,0 1e-200,0 0,1e-200 0))'),
 ('POLYGON((0 0,1 0,1 NaN,0 0))'),('MULTIPOLYGON(((0 0,4 0,4 3,0 0)), EMPTY)'),
 ('GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POLYGON((0 0,4 0,4 3,0 0)),POINT(1 1)),MULTIPOLYGON(((0 0,2 0,2 2,0 2,0 0))))'),
 ('CIRCULARSTRING(0 0,1 1,2 0)'),('CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0))'),('TRIANGLE((0 0,1 0,0 1,0 0))'),
 ('TIN(((0 0 0,0 1 0,1 1 0,0 0 0)))'),('POLYGON EMPTY')) v(w), generate_series(1, 2000) c;
analyze ar;
set pg_strom.enabled=off;
select 'cpu', w, count(ST_Area(g)) from ar where ST_Area(g) < 1e100 group by w order by w;
set pg_strom.enabled=on;
explain (costs off) select count(ST_Area(g)) from ar where ST_Area(g) < 1e100;
select 'gpu', w, count(ST_Area(g)) from ar where ST_Area(g) < 1e100 group by w order by w;
select 'gpu-total', count(ST_Area(g)), min(ST_Area(g)) from ar where ST_Area(g) < 1e100;
select 'gpu-nofilter', count(ST_Area(g)) from ar;
select 'count-star-filter', count(*) from ar where ST_Area(g) < 1e100;
select 'count-arg-nofilter-on-id', count(ST_Area(g)) from ar where id > 0;
select 'count-star-other-qual', count(*) from ar where ST_NPoints(g) >= 0;
select 'sum-npoints', sum(ST_NPoints(g)) from ar;
explain (analyze, costs off) select count(ST_Area(g)) from ar where ST_Area(g) < 1e100;
-- diagnostics: is the loss specific to ST_Area? other recheck-capable functions as aggregate arguments with a qual
set pg_strom.enabled=on;
select 'npoints-arg', count(ST_NPoints(g)), sum(ST_NPoints(g)) from ar where ST_NPoints(g) >= 0;
select 'length-arg', count(ST_Length(g)) from ar where ST_Length(g) >= 0;
select 'area-arg-idqual', count(ST_Area(g)) from ar where id > 0 and ST_Area(g) < 1e100;
select 'g-arg', count(g) from ar where ST_Area(g) < 1e100;
select 'area-plus', count(ST_Area(g) + 1) from ar where ST_Area(g) < 1e100;
select 'srid-arg', count(ST_SRID(g)) from ar where ST_NPoints(g) >= 0;
set pg_strom.enabled=off;
select 'cpu-npoints-arg', count(ST_NPoints(g)), sum(ST_NPoints(g)) from ar where ST_NPoints(g) >= 0;
select 'cpu-length-arg', count(ST_Length(g)) from ar where ST_Length(g) >= 0;
select 'cpu-area', count(ST_Area(g)) from ar where ST_Area(g) < 1e100;
-- cost and the float8 sum
drop table if exists ar2;
create table ar2 as select i as id, ST_MakePolygon(ST_MakeLine(array_agg(ST_MakePoint(cos(2*pi()*k/40.0)*(1+(i%10)), sin(2*pi()*k/40.0)*(1+(i%10))) order by k) || ST_MakePoint(1+(i%10), 0))) as g
 from generate_series(1, 100000) i, generate_series(0, 39) k group by i;
analyze ar2;
\timing on
set pg_strom.enabled=off; set max_parallel_workers_per_gather=0;
select 'cpu-count', count(*) from ar2 where ST_Area(g) > 100;
select 'cpu-sum', sum(ST_Area(g)) from ar2;
select 'cpu-minmax', min(ST_Area(g)), max(ST_Area(g)) from ar2;
set pg_strom.enabled=on; set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0;
select 'gpu-count', count(*) from ar2 where ST_Area(g) > 100;
select 'gpu-count', count(*) from ar2 where ST_Area(g) > 100;
select 'gpu-sum', sum(ST_Area(g)) from ar2;
select 'gpu-minmax', min(ST_Area(g)), max(ST_Area(g)) from ar2;
explain (costs off) select sum(ST_Area(g)) from ar2;
explain (costs off) select count(*) from ar2 where ST_Area(g) > 100;
\timing off
