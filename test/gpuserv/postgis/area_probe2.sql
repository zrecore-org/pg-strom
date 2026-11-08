-- Harder rings for area_twin_check.c (same "A|" output as area_probe.sql): offsets of 1e15/1e16 beside unit shapes (cancellation), nearly
-- collinear rings, integer rectangles, 1e150 coordinates (rescaling), 1e-160, rings of 3 to 20000 points.
--   psql -X -At -f area_probe2.sql | grep '^A|' > out.txt; ./twin < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table pts as
select i, k, ring, case i % 10
  when 9 then xr
  when 0 then 1e15 + round(random() * 8) when 1 then 1e16 + round(random() * 4) * 2 when 2 then (k * 1.0) + random() * 1e-9
  when 3 then round(random() * 10) when 4 then 1e150 * (random() * 2 - 1) when 5 then 1e-160 * (random() * 2 - 1)
  when 6 then 12345.678 + random() * 1e-3 when 7 then 1e300 * (random() * 2 - 1) else random() * 100 end as x,
  case i % 10
  when 9 then xr * 0.3 + (case when random() < 0.5 then 0 else 1e-13 end)
  when 0 then 1e15 + round(random() * 8) when 1 then 1e16 + round(random() * 4) * 2 when 2 then (k * 1.0) * 0.5 + 3 + random() * 1e-12
  when 3 then round(random() * 10) when 4 then 1e150 * (random() * 2 - 1) when 5 then 1e-160 * (random() * 2 - 1)
  when 6 then 765.4321 + random() * 1e-3 when 7 then 1e300 * (random() * 2 - 1) else random() * 100 end as y,
  0.0 as z
from (select i, k, case when k <= n0 then 0 else 1 end as ring, random() * 1000 as xr
      from (select i, case when i % 40 = 0 then 3 when i % 40 = 1 then 2000 when i % 997 = 5 then 20000 else 3 + (i % 9) end as n0
            from generate_series(1, 20000) i) s, lateral generate_series(1, n0 + 3) k) t;
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
