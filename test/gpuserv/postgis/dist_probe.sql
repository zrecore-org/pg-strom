-- Which floating-point expression does the installed PostGIS evaluate for the distance of two points (ST_3DDistance, ST_Distance)
-- and the length of a line (ST_3DLength, ST_Length)? Dumps pairs with PostGIS' answers; dist_probe.c tries the candidate orders. Run:
--   ADHOC=test/gpuserv/postgis/dist_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh > out.txt
--   grep '^D|' out.txt | cc -O0 -ffp-contract=off -o probe dist_probe.c -lm && ./probe < ...   (see dist_probe.c)
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table p as
select i,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as ax,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as ay,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as az,
  random() as r1, random() as r2, random() as r3
from generate_series(1, 200000) i;
create temp table q as
select i, ax, ay, az,
  case i % 3 when 0 then ax + (random() * 2 - 1) * abs(ax) * 1e-6 else ax + (r1 * 2 - 1) * (case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end) end as bx,
  case i % 3 when 0 then ay + (random() * 2 - 1) * abs(ay) * 1e-6 else ay + (r2 * 2 - 1) * (case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end) end as by,
  case i % 3 when 0 then az + (random() * 2 - 1) * abs(az) * 1e-6 else az + (r3 * 2 - 1) * (case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end) end as bz
from p;
select 'D', i, ax, ay, az, bx, by, bz,
  ST_3DDistance(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz)),
  ST_Distance(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz)),
  ST_3DMaxDistance(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz))
from q order by i;
