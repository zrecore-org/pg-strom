-- ST_3DDistance / ST_3DDWithin of a 3D point and a 3D segment: which expression does the installed PostGIS evaluate? (PostGIS' lw_dist3d_pt_seg
-- projects the point on the segment, r = ((p-A).(B-A)) / |B-A|^2, takes A for r < 0, B for r > 1 and else the point c = A + r (B-A), and
-- measures the point-to-point distance to it; the fusing of every step is what the probe finds.) Dumps random point/segment pairs;
-- seg3d_probe.c classifies each pair and counts the mismatches of every candidate. Run:
--   ADHOC=test/gpuserv/postgis/seg3d_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh > out.txt
--   cc -O0 -ffp-contract=off -o probe seg3d_probe.c -lm && ./probe < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table s as select i,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end as sc,
  random() r1, random() r2, random() r3, random() r4, random() r5, random() r6, random() r7, random() r8, random() r9
from generate_series(1, 200000) i;
select 'S', i, (r1*2-1)*sc*1.2, (r2*2-1)*sc*1.2, (r3*2-1)*sc*1.2, (r4*2-1)*sc, (r5*2-1)*sc, (r6*2-1)*sc, (r7*2-1)*sc, (r8*2-1)*sc, (r9*2-1)*sc,
  ST_3DDistance(ST_MakePoint((r1*2-1)*sc*1.2, (r2*2-1)*sc*1.2, (r3*2-1)*sc*1.2),
     ST_MakeLine(ST_MakePoint((r4*2-1)*sc, (r5*2-1)*sc, (r6*2-1)*sc), ST_MakePoint((r7*2-1)*sc, (r8*2-1)*sc, (r9*2-1)*sc)))
from s order by i;

-- the points that project (almost) onto an end of the segment: t = 0, 1 and their neighbours, so that the r < 0 / r > 1 decisions are probed too
select 'T', i, px, py, pz, ax, ay, az, bx, by, bz, ST_3DDistance(ST_MakePoint(px, py, pz), ST_MakeLine(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz)))
from (select i, ax, ay, az, bx, by, bz,
        ax + t * (bx - ax) + (r1 * 2 - 1) * sc * 0.3 * 0 + (r1 * 2 - 1) * sc * 0.01 as px,
        ay + t * (by - ay) + (r2 * 2 - 1) * sc * 0.01 as py,
        az + t * (bz - az) + (r3 * 2 - 1) * sc * 0.01 as pz
      from (select i, sc, r1, r2, r3, (r4*2-1)*sc ax, (r5*2-1)*sc ay, (r6*2-1)*sc az, (r7*2-1)*sc bx, (r8*2-1)*sc by, (r9*2-1)*sc bz,
              (array[0.0, 1e-16, -1e-16, 1e-12, -1e-12, 1.0, 1.0000000000000002, 0.9999999999999999, 1 + 1e-12, 1 - 1e-12, 1e-8, -1e-8, 1 - 1e-8, 1 + 1e-8]::float8[])[1 + i % 14] t
            from s where i <= 100000) q) w
order by i;

-- segments parallel to an axis, lying in an axis plane, or of zero length (A = B): PostGIS' 3D function has no axis-aligned special case (the 2D one has:
-- seg_axis_probe.sql), the same model holds
select 'T', i, px, py, pz, ax, ay, az, bx, by, bz, ST_3DDistance(ST_MakePoint(px, py, pz), ST_MakeLine(ST_MakePoint(ax, ay, az), ST_MakePoint(bx, by, bz)))
from (select i, (r1*2-1)*sc px, (r2*2-1)*sc py, (r3*2-1)*sc pz, (r4*2-1)*sc ax, (r5*2-1)*sc ay, (r6*2-1)*sc az,
        case i % 6 when 0 then (r4*2-1)*sc when 1 then (r4*2-1)*sc when 2 then (r7*2-1)*sc when 3 then (r7*2-1)*sc when 4 then (r4*2-1)*sc else (r7*2-1)*sc end bx,
        case i % 6 when 0 then (r5*2-1)*sc when 1 then (r8*2-1)*sc when 2 then (r5*2-1)*sc when 3 then (r8*2-1)*sc when 4 then (r5*2-1)*sc else (r8*2-1)*sc end by,
        case i % 6 when 0 then (r9*2-1)*sc when 1 then (r6*2-1)*sc when 2 then (r6*2-1)*sc when 3 then (r9*2-1)*sc when 4 then (r6*2-1)*sc else (r6*2-1)*sc end bz
      from s where i > 100000) q order by i;
