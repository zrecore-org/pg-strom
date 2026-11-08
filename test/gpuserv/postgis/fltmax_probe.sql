-- PostGIS' initial distance is FLT_MAX (lwgeom_mindistance2d_tolerance): what do ST_Distance / ST_DWithin answer beyond it? The GPU kernel hands every
-- distance beyond FLT_MAX (3.4028234663852886e38) to the backend. Run on any PostGIS (no pg_strom needed):
--   psql -X -At -f test/gpuserv/postgis/fltmax_probe.sql
create extension if not exists postgis;
select 'dist point-point 1e39', ST_Distance('POINT(0 0)'::geometry, 'POINT(1e39 0)'::geometry);
select 'dist point-line 1e39', ST_Distance('POINT(0 0)'::geometry, 'LINESTRING(1e39 1, 1e39 5)'::geometry);
select 'dist point-line 3.4e38', ST_Distance('POINT(0 0)'::geometry, 'LINESTRING(3.4e38 1, 3.4e38 5)'::geometry);
select 'dist 3d point-line 1e39', ST_3DDistance('POINT Z (0 0 0)'::geometry, 'LINESTRING Z (1e39 1 0, 1e39 5 0)'::geometry);
select 'dwithin point-point 1e39 tol 5e38', ST_DWithin('POINT(0 0)'::geometry, 'POINT(1e39 0)'::geometry, 5e38);
select 'dwithin point-point 1e39 tol 1e300', ST_DWithin('POINT(0 0)'::geometry, 'POINT(1e39 0)'::geometry, 1e300);
select 'dwithin point-line 1e39 tol 5e38', ST_DWithin('POINT(0 0)'::geometry, 'LINESTRING(1e39 1, 1e39 5)'::geometry, 5e38);
select 'dwithin point-line 1e39 tol 1e300', ST_DWithin('POINT(0 0)'::geometry, 'LINESTRING(1e39 1, 1e39 5)'::geometry, 1e300);
select 'dwithin point-line 1e39 tol 3.5e38', ST_DWithin('POINT(0 0)'::geometry, 'LINESTRING(1e39 1, 1e39 5)'::geometry, 3.5e38);
select 'dwithin point-line 1e39 tol 1e39', ST_DWithin('POINT(0 0)'::geometry, 'LINESTRING(1e39 1, 1e39 5)'::geometry, 1e39);
select 'dwithin 3d point-line 1e39 tol 1e300', ST_3DDWithin('POINT Z (0 0 0)'::geometry, 'LINESTRING Z (1e39 1 0, 1e39 5 0)'::geometry, 1e300);
select 'dwithin 3d point-point 1e39 tol 1e300', ST_3DDWithin('POINT Z (0 0 0)'::geometry, 'POINT Z (1e39 0 0)'::geometry, 1e300);
select 'dwithin empty line', ST_DWithin('LINESTRING EMPTY'::geometry, 'POINT(0 0)'::geometry, 1e39);
select 'dwithin empty polygon', ST_DWithin('POLYGON EMPTY'::geometry, 'POINT(0 0)'::geometry, 1);
select 'dwithin empty point-line', ST_DWithin('POINT EMPTY'::geometry, 'LINESTRING(0 0, 1 1)'::geometry, 1);
select 'dist empty line', ST_Distance('LINESTRING EMPTY'::geometry, 'POINT(0 0)'::geometry);
