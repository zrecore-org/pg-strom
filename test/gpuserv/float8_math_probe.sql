-- A10 math functions: what the installed PostgreSQL answers for sqrt, round, ceil, ceiling, floor, trunc and sign over the
-- edge values that decide the kernel (signed zeros, ties, NaN, infinities, the 2^52 boundary, subnormals), and the error
-- of a negative sqrt (message and SQLSTATE). float8out prints a negative zero as -0.
-- Run: ADHOC=test/gpuserv/float8_math_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
select version();
create temp table v(x float8);
insert into v values
 ('0'), ('-0'), ('0.5'), ('-0.5'), ('0.49999999999999994'), ('-0.49999999999999994'), ('1.5'), ('-1.5'), ('2.5'), ('-2.5'),
 ('0.9999999999999999'), ('1'), ('-1'), ('1.0000000000000002'), ('3.5'), ('4503599627370495.5'), ('4503599627370496'),
 ('4503599627370497'), ('-4503599627370495.5'), ('9007199254740992'), ('1e300'), ('-1e300'), ('5e-324'), ('-5e-324'),
 ('2.2250738585072014e-308'), ('Infinity'), ('-Infinity'), ('NaN'), ('2'), ('4'), ('1e-10'), ('-1e-10'), ('123456789.123456789');
\echo -- value | round | ceil | ceiling | floor | trunc | sign   (text, so that -0 shows)
select x::text as x, round(x)::text as round, ceil(x)::text as ceil, ceiling(x)::text as ceiling, floor(x)::text as floor,
       trunc(x)::text as trunc, sign(x)::text as sign
from v order by x;

create or replace function try_sqrt(a float8) returns text language plpgsql as $$
begin
  return sqrt(a)::text;
exception when others then
  return 'ERR ' || sqlstate || ': ' || sqlerrm;
end $$;
\echo -- sqrt (the error text and SQLSTATE)
select x::text as x, try_sqrt(x) as sqrt from v order by x;

\echo -- the SQLSTATE and text of the errors the GPU path already raises (division by zero, overflow), for comparison
create or replace function try_expr(q text) returns text language plpgsql as $$
declare r text;
begin
  execute q into r;
  return coalesce(r, 'NULL');
exception when others then
  return 'ERR ' || sqlstate || ': ' || sqlerrm;
end $$;
select try_expr('select (1.0::float8 / 0.0::float8)::text');
select try_expr('select (1e308::float8 * 10::float8)::text');
select try_expr('select sqrt(-1.0::float8)::text');
