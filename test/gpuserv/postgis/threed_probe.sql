-- 3D follow-ups: which functions exist, and whether ST_3DDistance of two points equals plain binary64 arithmetic. Run:
--   ADHOC=test/gpuserv/postgis/threed_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
select postgis_full_version();
select proname, pg_get_function_arguments(oid) from pg_proc
 where proname in ('st_hasz','st_hasm','st_ndims','st_zmflag','st_coorddim','st_3dintersects','st_3ddistance','st_3dlength','st_hasarc')
 order by 1, 2;

-- ST_3DDistance of two points against sqrt((dx*dx + dy*dy) + dz*dz) in plain SQL float8 arithmetic (no fused multiply-add)
create temp table p3 as
select ST_MakePoint(random()*2000-1000, random()*2000-1000, random()*2000-1000) a,
       ST_MakePoint(random()*2000-1000, random()*2000-1000, random()*2000-1000) b
from generate_series(1, 100000);
select 'pairs' k, count(*) from p3;
select 'mismatch sqrt((dx2+dy2)+dz2)' k, count(*) from p3
 where ST_3DDistance(a, b) <> sqrt((ST_X(a)-ST_X(b))*(ST_X(a)-ST_X(b)) + (ST_Y(a)-ST_Y(b))*(ST_Y(a)-ST_Y(b)) + (ST_Z(a)-ST_Z(b))*(ST_Z(a)-ST_Z(b)));
select 'mismatch hypot' k, count(*) from p3
 where ST_3DDistance(a, b) <> sqrt(power(ST_X(a)-ST_X(b), 2) + power(ST_Y(a)-ST_Y(b), 2) + power(ST_Z(a)-ST_Z(b), 2));
-- the squared distance compared with the tolerance is what ST_3DDWithin decides; here ST_3DDistance itself
select 'max relative difference' k, max(abs(ST_3DDistance(a, b) - sqrt((ST_X(a)-ST_X(b))*(ST_X(a)-ST_X(b)) + (ST_Y(a)-ST_Y(b))*(ST_Y(a)-ST_Y(b)) + (ST_Z(a)-ST_Z(b))*(ST_Z(a)-ST_Z(b)))) / nullif(ST_3DDistance(a, b), 0)) from p3;
