-- ST_Distance of a point and one segment: which expression does the installed PostGIS evaluate? (A first model, the foot of the perpendicular
-- with fused operations, was wrong in the last bits.) Dumps random point/segment pairs at several magnitudes; seg_probe.c classifies each pair
-- (the point projects before A, after B, or inside the segment) and counts the mismatches of every candidate. Run:
--   ADHOC=test/gpuserv/postgis/seg_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh > out.txt
--   cc -O0 -ffp-contract=off -o probe seg_probe.c -lm && ./probe < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table s as select i,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end as sc,
  random() r1, random() r2, random() r3, random() r4, random() r5, random() r6
from generate_series(1, 200000) i;
select 'S', i, (r1*2-1)*sc*1.2, (r2*2-1)*sc*1.2, (r3*2-1)*sc, (r4*2-1)*sc, (r5*2-1)*sc, (r6*2-1)*sc,
  ST_Distance(ST_MakePoint((r1*2-1)*sc*1.2, (r2*2-1)*sc*1.2), ST_MakeLine(ST_MakePoint((r3*2-1)*sc, (r4*2-1)*sc), ST_MakePoint((r5*2-1)*sc, (r6*2-1)*sc)))
from s order by i;
