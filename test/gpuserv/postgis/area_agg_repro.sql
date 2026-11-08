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
