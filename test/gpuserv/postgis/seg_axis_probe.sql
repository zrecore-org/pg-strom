-- ST_Distance of a point and a segment that is vertical or horizontal (or almost): PostGIS answers |px - ax| exactly when the foot is inside the
-- segment (seg_probe.sql looked at random segments only: there the |s| * sqrt(L2) model holds, 0 differences in 89,523 rows; for an axis-aligned
-- segment it is off by 1-2 ulp in 27% of the rows). Dumps rows of kind V (vertical), H (horizontal) and N (a tiny slope: dx = k * 1e-n of the length);
-- seg_axis_probe.c counts the rows where the generic |s| * sqrt(L2) model, or |h| (the axis-aligned distance), differs. Run:
--   psql -X -At -f test/gpuserv/postgis/seg_axis_probe.sql > out.txt && cc -O0 -ffp-contract=off -o probe seg_axis_probe.c -lm && ./probe < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table a as select i, case i % 4 when 0 then 1000.0 when 1 then 1e-3 when 2 then 1e8 else 1.0 end sc,
  random() r1, random() r2, random() r3, random() r4, random() r5, (array[0.0, 0.0, 1e-300, 1e-100, 1e-30, 1e-16, 1e-12, 1e-9, 1e-5, 1e-3])[1 + i % 10] tiny from generate_series(1, 100000) i;
select 'A', i, 'V', (r1*2-1)*sc, (r2*2-1)*sc, (r3*2-1)*sc, (r4*2-1)*sc, (r3*2-1)*sc + tiny * sc, (r5*2-1)*sc,
  ST_Distance(ST_MakePoint((r1*2-1)*sc, (r2*2-1)*sc), ST_MakeLine(ST_MakePoint((r3*2-1)*sc, (r4*2-1)*sc), ST_MakePoint((r3*2-1)*sc + tiny * sc, (r5*2-1)*sc)))
from a;
select 'A', i, 'H', (r1*2-1)*sc, (r2*2-1)*sc, (r3*2-1)*sc, (r4*2-1)*sc, (r5*2-1)*sc, (r4*2-1)*sc + tiny * sc,
  ST_Distance(ST_MakePoint((r1*2-1)*sc, (r2*2-1)*sc), ST_MakeLine(ST_MakePoint((r3*2-1)*sc, (r4*2-1)*sc), ST_MakePoint((r5*2-1)*sc, (r4*2-1)*sc + tiny * sc)))
from a;
