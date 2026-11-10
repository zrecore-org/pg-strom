-- Timing of GROUP BY over a join with three and four keys: the new GpuPreAgg-over-join, the old plan (GpuJoin, the aggregate on the CPU)
-- and PostgreSQL alone with 8 workers. Run through keys_dev.sh: ADHOC=$PWD/test/gpuserv/jkeys_bench.sql SKIP_TESTS=1 (JKEYS_ONLY=1).
-- JB_ROWS sizes the fact table (default 4000000 here).
\set QUIET 1
\pset format unaligned
\pset tuples_only on
drop table if exists jb_f; drop table if exists jb_d;
create table jb_f as select i as id, ((i::int8 * 7919) % 4000)::int4 as k, (i % 40)::int4 as a, ((i / 40) % 50)::int4 as b, ((i / 2000) % 50)::int4 as c, i::int8 as v from generate_series(1, 4000000) i;
create table jb_d as select i::int4 as k, (i % 5)::int4 as x, (i % 4)::int4 as y, (i * 3)::int8 as w from generate_series(0, 3999) i;
vacuum analyze jb_f; analyze jb_d;
set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set pg_strom.gpupreagg_max_groups=1000000; set pg_strom.gpu_error_fallback=off;
set enable_mergejoin=off; set enable_nestloop=off;
create or replace function jb_time(q text, setup text) returns text language plpgsql as $f$
declare t0 timestamptz; best numeric := 1e12; el numeric; n int; plan text; r record;
begin
  execute setup;
  for i in 1..3 loop
    t0 := clock_timestamp();
    execute 'select count(*) from (' || q || ') x' into n;
    el := extract(epoch from clock_timestamp() - t0) * 1000;
    if el < best then best := el; end if;
  end loop;
  return round(best)::text || ' ms, ' || n || ' groups';
end $f$;
\echo -- 2 keys (the kernel the wider key columns must not slow down): a % 4, d.x: 20 groups
select 'GPU new plan       ' || jb_time('select (f.a % 4), d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=0');
select 'PostgreSQL 8 workers ' || jb_time('select (f.a % 4), d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2', 'set pg_strom.enabled=off; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=8');
\echo -- 2 keys: a, d.x: 200 groups
select 'GPU new plan       ' || jb_time('select f.a, d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=0');
select 'PostgreSQL 8 workers ' || jb_time('select f.a, d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2', 'set pg_strom.enabled=off; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=8');
\echo -- 3 keys: a (40), b (50), d.x (5): 10,000 groups
select 'GPU new plan       ' || jb_time('select f.a, f.b, d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=on; set max_parallel_workers_per_gather=0');
select 'GPU join + CPU agg ' || jb_time('select f.a, f.b, d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=off; set max_parallel_workers_per_gather=0');
select 'PostgreSQL 8 workers ' || jb_time('select f.a, f.b, d.x, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=off; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=8');
\echo -- 3 keys with few groups: a % 4, d.x, d.y: 80 groups
select 'GPU new plan       ' || jb_time('select (f.a % 4), d.x, d.y, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=0');
select 'GPU join + CPU agg ' || jb_time('select (f.a % 4), d.x, d.y, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=off; set max_parallel_workers_per_gather=0');
select 'PostgreSQL 8 workers ' || jb_time('select (f.a % 4), d.x, d.y, count(*), sum(f.v), max(d.w) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3', 'set pg_strom.enabled=off; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=8');
\echo -- 4 keys: a, b, c (50), d.x: 500,000 groups upper bound (about 100,000 occur)
select 'GPU new plan       ' || jb_time('select f.a, f.b, f.c, d.x, count(*), sum(f.v) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3, 4', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=0');
select 'GPU join + CPU agg ' || jb_time('select f.a, f.b, f.c, d.x, count(*), sum(f.v) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3, 4', 'set pg_strom.enabled=on; set pg_strom.enable_gpupreagg=off; set max_parallel_workers_per_gather=0');
select 'PostgreSQL 8 workers ' || jb_time('select f.a, f.b, f.c, d.x, count(*), sum(f.v) from jb_f f join jb_d d on f.k = d.k group by 1, 2, 3, 4', 'set pg_strom.enabled=off; set pg_strom.enable_gpupreagg=on; set max_parallel_workers_per_gather=8');
