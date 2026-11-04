-- what does geom && ST_MakePoint(x, y) do on the CPU for non-finite and out-of-float-range coordinates,
-- through a seq scan and through the GiST index
create extension if not exists postgis;
drop table if exists pg_gnf;
create table pg_gnf (id int, geom geometry);
insert into pg_gnf values (1, st_makepoint(1e39, -1e39)), (2, st_makepoint(0,0)),
  (3, st_makeenvelope(-1e38, -1e38, 1e38, 1e38)), (4, st_makepoint(3.4e38, 3.4e38)), (5, null), (6, 'POINT EMPTY');
create index on pg_gnf using gist (geom);
analyze pg_gnf;
select 'seq' as how, p.n, array(select id from pg_gnf g where g.geom && st_makepoint(p.x, p.y) order by id)
from (values (1,'NaN'::float8,0::float8),(2,'Infinity',0),(3,'-Infinity',0),(4,1e39,-1e39),(5,1e300,0),(6,3.4e38,3.4e38),(7,0,'NaN'),(8,'Infinity','Infinity')) p(n,x,y) order by 2;
set enable_seqscan = off; set enable_bitmapscan = off;
select 'idx' as how, p.n, array(select id from pg_gnf g where g.geom && st_makepoint(p.x, p.y) order by id)
from (values (1,'NaN'::float8,0::float8),(2,'Infinity',0),(3,'-Infinity',0),(4,1e39,-1e39),(5,1e300,0),(6,3.4e38,3.4e38),(7,0,'NaN'),(8,'Infinity','Infinity')) p(n,x,y) order by 2;
