-- The length of a line: which expression and summation order does the installed PostGIS use (ST_3DLength, ST_Length)? Dumps random
-- lines with PostGIS' answers; len_probe.c tries the candidates. Run:
--   ADHOC=test/gpuserv/postgis/len_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh > out.txt
--   cc -O0 -ffp-contract=off -o probe len_probe.c -lm && ./probe < out.txt
create extension if not exists postgis;
set extra_float_digits = 3;
create temp table pts as
select i, k,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as x,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as y,
  case i % 5 when 0 then 1000 when 1 then 1e-3 when 2 then 1e8 when 3 then 1 else 1e-120 end * (random() * 2 - 1) as z
from generate_series(1, 60000) i, generate_series(1, 2 + (i % 9)) k;
select 'L', i, count(*), string_agg(format('%s|%s|%s', x, y, z), '|' order by k),
  ST_3DLength(ST_MakeLine(ST_MakePoint(x, y, z) order by k)),
  ST_Length(ST_MakeLine(ST_MakePoint(x, y, z) order by k)),
  ST_Length(ST_MakeLine(ST_MakePoint(x, y) order by k)),
  ST_3DLength(ST_MakeLine(ST_MakePoint(x, y) order by k))
from pts group by i order by i;
