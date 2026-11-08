-- ST_DWithin(polygon, point, d) against ST_Distance(polygon, point) for points in a hole, outside and in the body (the planner's calibration
-- vulkan_geom_dwithin_np_calibrated models it as "distance <= d"). For d = the distance, one float8 step below and above it: how many answers
-- differ from "d_dist <= tol"? Also the distance to the polygon against the distance to its hole's ring as a line. Run on any PostGIS:
--   psql -X -At -f test/gpuserv/postgis/dwithin_np_probe.sql
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table p as
select i, sc, (r1*2-1)*sc*1.7 x, (r2*2-1)*sc*1.7 y,
       ST_GeomFromText(format('POLYGON((%s %s, %s %s, %s %s, %s %s, %s %s),(%s %s, %s %s, %s %s, %s %s, %s %s))', -sc, -sc, sc, -sc, sc, sc, -sc, sc, -sc, -sc,
                              -sc/4, -sc/4, -sc/4, sc/4, sc/4, sc/4, sc/4, -sc/4, -sc/4, -sc/4)) poly,
       ST_GeomFromText(format('LINESTRING(%s %s, %s %s, %s %s, %s %s, %s %s)', -sc/4, -sc/4, -sc/4, sc/4, sc/4, sc/4, sc/4, -sc/4, -sc/4, -sc/4)) hole
from (select i, case i % 3 when 0 then 1000.0 when 1 then 1.0 else 0.001 end sc, random() r1, random() r2 from generate_series(1, 30000) i) q;
create temp table d as select *, ST_Distance(poly, ST_MakePoint(x, y)) dist, ST_Distance(hole, ST_MakePoint(x, y)) hdist from p;
select 'rows', count(*), 'in hole', count(*) filter (where abs(x) < sc/4 and abs(y) < sc/4), 'dist differs from hole-ring distance (in hole)',
       count(*) filter (where abs(x) < sc/4 and abs(y) < sc/4 and dist <> hdist) from d;
create temp table t as select *, dist - power(2.0::float8, floor(ln(dist) / ln(2.0)) - 52) dn, dist + power(2.0::float8, floor(ln(dist) / ln(2.0)) - 52) up from d where dist > 0;
select 'dwithin(dist) false', count(*) filter (where not ST_DWithin(poly, ST_MakePoint(x, y), dist)),
       'dwithin(below) true', count(*) filter (where ST_DWithin(poly, ST_MakePoint(x, y), dn)),
       'dwithin(above) false', count(*) filter (where not ST_DWithin(poly, ST_MakePoint(x, y), up)), 'rows', count(*) from t;
select 'in hole: dwithin(below) true', count(*) filter (where ST_DWithin(poly, ST_MakePoint(x, y), dn)), 'rows', count(*) from t where abs(x) < sc/4 and abs(y) < sc/4;
select 'outside: dwithin(below) true', count(*) filter (where ST_DWithin(poly, ST_MakePoint(x, y), dn)), 'rows', count(*) from t where abs(x) > sc;
