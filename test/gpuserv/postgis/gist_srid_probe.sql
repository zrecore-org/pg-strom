-- does && between geometries of different SRIDs error, or compare boxes?
create extension if not exists postgis;
select st_setsrid(st_makepoint(0,0),4326) && st_makepoint(0,0) as srid_vs_none;
select st_setsrid(st_makepoint(0,0),4326) && st_setsrid(st_makepoint(0,0),3857) as srid_vs_other;
select st_setsrid(st_makepoint(0,0),4326) && st_setsrid(st_makepoint(0,0),4326) as same;
