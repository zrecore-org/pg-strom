-- ST_Area of a polygon: which expression and summation order does the installed PostGIS use (ptarray_signed_area, lwpoly_area)? Dumps random
-- closed rings with PostGIS' areas of the exterior ring alone and of the polygon with a hole; area_probe.c tries the candidates. Run:
--   psql -X -At -f area_probe.sql > out.txt; cc -O0 -ffp-contract=off -o probe area_probe.c -lm && ./probe < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table pts as
select i, k,
  case i % 6 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 when 4 then 1e15 else 1e-120 end * (random() * 2 - 1) as x,
  case i % 6 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 when 4 then 1e15 else 1e-120 end * (random() * 2 - 1) as y,
  case i % 6 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 when 4 then 1e15 else 1e-120 end * (random() * 2 - 1) as z,
  case when k <= 3 + (i % 7) then 0 else 1 end as ring
from generate_series(1, 60000) i, generate_series(1, 3 + (i % 7) + 3 + (i % 5)) k;
-- ring 0 has 3 + i%7 points (closed by repeating the first), ring 1 the rest (closed)
create temp table rings as
select i, ring, array_agg(ST_MakePoint(x, y, z) order by k) || (array_agg(ST_MakePoint(x, y, z) order by k))[1] as a,
       string_agg(format('%s|%s|%s', x, y, z), '|' order by k) || '|' || (array_agg(format('%s|%s|%s', x, y, z) order by k))[1] as s,
       count(*) + 1 as n
from pts group by i, ring;
select 'A', r0.i, r0.n, r1.n, r0.s, r1.s,
  ST_Area(ST_MakePolygon(ST_MakeLine(r0.a))),
  ST_Area(ST_MakePolygon(ST_MakeLine(r1.a))),
  ST_Area(ST_MakePolygon(ST_MakeLine(r0.a), ARRAY[ST_MakeLine(r1.a)])),
  ST_Area(ST_MakePolygon(ST_MakeLine(r1.a), ARRAY[ST_MakeLine(r0.a)]))
from rings r0 join rings r1 on r0.i = r1.i and r0.ring = 0 and r1.ring = 1 order by r0.i;
