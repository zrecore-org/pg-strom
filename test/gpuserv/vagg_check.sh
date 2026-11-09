#!/bin/sh
# kernels_vm/vagg.inc (the soft-binary64 float8_accum / float8_combine / float8_regr_accum / float8_regr_combine that the GPU
# kernels and the service use for variance, stddev, covar, corr and regr_*) against the installed PostgreSQL's own functions, bit
# for bit, on generated data: offsets of 1e9 with unit variance, uniform, a wide range of magnitudes, integers, constants,
# NaN and Infinity, overflow (the error cases must agree), subnormals. Needs only PostgreSQL and cc: no GPU, no module, no lock.
#
#   sh test/gpuserv/vagg_check.sh [cases per kind (default 20000)]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
PORT=${PORT:-54428}
N=${1:-20000}
W=${VAGG_WORK:-$ROOT/test-data/vagg_check}
mkdir -p "$W" || exit 2
cc -O1 -o "$W/vagg_check" -I "$ROOT/src/gpuserv" "$ROOT/src/gpuserv/vagg_check.c" || { echo "vagg_check: build failed"; exit 2; }
rm -rf "$W/data"
$PGBIN/initdb -D "$W/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 2; }
echo "port = $PORT
unix_socket_directories = '$W'
fsync = off" >> "$W/data/postgresql.conf"
$PGBIN/pg_ctl -D "$W/data" -l "$W/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 2; }
trap '$PGBIN/pg_ctl -D "$W/data" -m fast -w stop >/dev/null 2>&1' EXIT INT TERM
$PGBIN/psql -h "$W" -p $PORT -U postgres -X -At -q -v ON_ERROR_STOP=1 -c "
create function vc_val(fl int) returns float8 language sql as \$\$
  select case fl
    when 0 then 1e9 + (random() - 0.5) * 2
    when 1 then (random() - 0.5) * 2
    when 2 then (random() - 0.5) * power(10::float8, floor(random() * 60) - 30)
    when 3 then floor(random() * 2001) - 1000
    when 4 then 12345.678
    when 5 then case floor(random() * 12) when 0 then 'NaN'::float8 when 1 then 'Infinity'::float8 when 2 then '-Infinity'::float8
                  else (random() - 0.5) * 100 end
    when 6 then (random() - 0.5) * 1e305 * (1 + floor(random() * 3))
    else random() * 1e-310 * floor(random() * 4) end
\$\$;
create function vc_hex(v float8) returns text language sql as \$\$ select encode(float8send(v), 'hex') \$\$;
create function vc_hexs(v float8[]) returns text language sql as \$\$
  select case when v is null then null else coalesce((select string_agg(vc_hex(e), ',' order by i) from unnest(v) with ordinality as t(e, i)), '') end \$\$;
create function vc_vstate(xs float8[]) returns float8[] language plpgsql as \$\$
declare st float8[] := '{0,0,0}'; x float8;
begin
  foreach x in array xs loop st := float8_accum(st, x); end loop;
  return st;
exception when others then return null;
end \$\$;
create function vc_rstate(ys float8[], xs float8[]) returns float8[] language plpgsql as \$\$
declare st float8[] := '{0,0,0,0,0,0}';
begin
  for i in 1 .. coalesce(array_length(xs, 1), 0) loop st := float8_regr_accum(st, ys[i], xs[i]); end loop;
  return st;
exception when others then return null;
end \$\$;
create function vc_comb(a float8[], b float8[], n int) returns float8[] language plpgsql as \$\$
begin
  if n = 3 then return float8_combine(a, b); end if;
  return float8_regr_combine(a, b);
exception when others then return null;
end \$\$;
create function vc_fmt(v float8[]) returns text language sql as \$\$ select coalesce(vc_hexs(v), 'ERR') \$\$;
" postgres >/dev/null || { echo "vagg_check: setup failed"; exit 2; }
$PGBIN/psql -h "$W" -p $PORT -U postgres -X -At -q -v ON_ERROR_STOP=1 -o "$W/cases.txt" -c "
select setseed(0.42);
with c as (
  select g, g % 8 as fl, 1 + floor(random() * 40)::int as n from generate_series(1, $N) g
), d as (
  select g, fl, n, (select array_agg(vc_val(fl) order by i) from generate_series(1, n) i) as xs from c
), e as (
  select *, (select array_agg(vc_val((fl + g) % 8) order by i) from generate_series(1, n) i) as ys from d
)
select 'V|' || vc_hexs(xs) || '|' || coalesce(vc_fmt(vc_vstate(xs)), 'ERR') || '|' ||
       coalesce(vc_hexs(vc_vstate(xs[1:n/2])), 'ERR') || '|' || coalesce(vc_hexs(vc_vstate(xs[n/2+1:n])), 'ERR') || '|' ||
       coalesce(vc_hexs(vc_comb(vc_vstate(xs[1:n/2]), vc_vstate(xs[n/2+1:n]), 3)), 'ERR')
  from e
union all
select 'R|' || vc_hexs(xs) || '|' || vc_hexs(ys) || '|' || coalesce(vc_hexs(vc_rstate(ys, xs)), 'ERR') || '|' ||
       coalesce(vc_hexs(vc_rstate(ys[1:n/2], xs[1:n/2])), 'ERR') || '|' || coalesce(vc_hexs(vc_rstate(ys[n/2+1:n], xs[n/2+1:n])), 'ERR') || '|' ||
       coalesce(vc_hexs(vc_comb(vc_rstate(ys[1:n/2], xs[1:n/2]), vc_rstate(ys[n/2+1:n], xs[n/2+1:n]), 6)), 'ERR')
  from e;
" postgres >/dev/null || { echo "vagg_check: generating the cases failed"; exit 2; }
"$W/vagg_check" "$W/cases.txt"
