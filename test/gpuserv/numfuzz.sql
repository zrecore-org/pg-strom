-- numfuzz.sql: random numeric operand pairs and PostgreSQL's results for numfuzz.c.
--   psql -At -v rows=300000 -f numfuzz.sql <db> | /tmp/numfuzz
-- Operands have 1..19 significant digits (some near powers of 10 and 10000 boundaries, some sharing leading digits),
-- scales 0..30 (rarely more) and either sign.
\set QUIET on
drop table if exists fz_t;
create table fz_t (a numeric, b numeric, n int, p int, s int);

create or replace function fz_digits(nd int, kind int) returns text language plpgsql volatile as $$
declare r text := ''; i int; d int;
begin
  if kind = 0 then          -- random digits
    for i in 1..nd loop r := r || (floor(random() * 10))::int::text; end loop;
  elsif kind = 1 then       -- 1 or 9 then zeros / nines
    r := (case when random() < 0.5 then '1' else '9' end);
    for i in 2..nd loop r := r || (case when random() < 0.7 then (case when substr(r,1,1) = '1' then '0' else '9' end)
                                   else (floor(random() * 10))::int::text end); end loop;
  elsif kind = 2 then       -- powers of ten and neighbours
    r := '1' || repeat('0', nd - 1);
    if random() < 0.5 then r := repeat('9', nd); end if;
  else                      -- leading digit 5 then random: lots of rounding ties
    r := '5';
    for i in 2..nd loop r := r || (case when random() < 0.5 then '0' else (floor(random() * 10))::int::text end); end loop;
  end if;
  return r;
end $$;

create or replace function fz_val(maxdig int, base text default null) returns numeric language plpgsql volatile as $$
declare nd int; sc int; digs text; neg text; k int; t text;
begin
  nd := 1 + floor(random() * random() * maxdig)::int;
  if nd > maxdig then nd := maxdig; end if;
  sc := case when random() < 0.1 then 0 when random() < 0.9 then floor(random() * 12)::int else floor(random() * 60)::int end;
  k := floor(random() * 4)::int;
  digs := fz_digits(nd, k);
  if base is not null and random() < 0.3 then       -- share the leading digits with the other operand
    digs := substr(ltrim(regexp_replace(base, '[^0-9]', '', 'g'), '0') || '1', 1, 1 + floor(random() * 4)::int) || substr(digs, 1, greatest(nd - 4, 0));
  end if;
  if random() < 0.05 then digs := '0'; end if;
  if length(digs) <= sc then digs := lpad(digs, sc + 1, '0'); end if;
  neg := case when random() < 0.5 then '-' else '' end;
  if sc > 0 then t := substr(digs, 1, length(digs) - sc) || '.' || substr(digs, length(digs) - sc + 1);
  else t := digs; end if;
  return (neg || t)::numeric;
end $$;

insert into fz_t
select a, fz_val(18, a::text), (floor(random() * 50) - 15)::int, 1 + floor(random() * 22)::int, floor(random() * 14)::int
from (select fz_val(18) as a from generate_series(1, :rows)) x;

create or replace function fz_try(sqltxt text) returns text language plpgsql as $$
declare r text;
begin
  execute sqltxt into r;
  return coalesce(r, 'NULL');
exception when others then return 'ERR';
end $$;

select a, b, n, p, s,
  fz_try(format('select (%L::numeric / %L::numeric)::text', a, b)),
  fz_try(format('select (%L::numeric %% %L::numeric)::text', a, b)),
  fz_try(format('select div(%L::numeric, %L::numeric)::text', a, b)),
  fz_try(format('select round(%L::numeric, %s)::text', a, n)),
  fz_try(format('select trunc(%L::numeric, %s)::text', a, n)),
  fz_try(format('select ceil(%L::numeric)::text', a)),
  fz_try(format('select floor(%L::numeric)::text', a)),
  fz_try(format('select sign(%L::numeric)::text', a)),
  fz_try(format('select (%L::numeric)::numeric(%s,%s)::text', a, p, s))
from fz_t;
