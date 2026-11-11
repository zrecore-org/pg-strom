-- reproducer tables for the NVIDIA float8 failures (negative integers, NaN); used by nvidia_probe.sh
create table np (i int4, a int2, b int4, c int8, d float4, z float8, g3 int4, g4 int4);
insert into np select i,
	((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4 else (((i * 17) % 10000) / 7.0)::float4 end,
	case i % 24 when 0 then 'NaN'::float8 when 1 then 'Infinity' when 2 then '-Infinity' when 3 then '-0' when 4 then 0
		when 18 then null else ((i % 1000) / 8.0)::float8 end,
	i % 2, i % 4
	from generate_series(1, 20000) i;
vacuum analyze np;
-- tiny table: every value is shown
create table nt (i int4, a int2, b int4, c int8);
insert into nt values (1, -9, -9, -9), (2, -8, -8, -8), (3, -7, -7, -7), (4, -1, -1, -1), (5, 0, 0, 0), (6, 1, 1, 1), (7, 7, 7, 7), (8, 9, 9, 9), (9, -3, -3, -3), (10, -5, -5, -5);
vacuum analyze nt;
-- the f8 test's size: 200000 rows (many workgroups) with the special values of z
create table np2 (i int4, a int2, b int4, g3 int4, g5 int4, z float8, w float8);
insert into np2 select i, ((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end, i % 4, i % 5,
	case i % 24 when 0 then 'NaN'::float8 when 1 then 'Infinity' when 2 then '-Infinity' when 3 then '-0' when 4 then 0
		when 5 then 5e-324 when 6 then 2.2250738585072014e-308 when 7 then 1.7976931348623157e308
		when 8 then -1.7976931348623157e308 when 9 then 1e-310 when 10 then 0.5 when 11 then -0.5
		when 12 then 1.5 when 13 then 2.5 when 14 then -2.5 when 15 then 9.2233720368547758e18
		when 16 then -9.2233720368547758e18 when 17 then 3.0000000000000004 when 18 then null
		when 19 then 4503599627370497.5 when 20 then 1e15 when 21 then -1e300 when 22 then 1e300
		else ((i % 1000) / 8.0)::float8 end,
	(((i % 997) + 1) / 16.0)::float8
	from generate_series(1, 200000) i;
vacuum analyze np2;
-- the table of float8_tests.inc (f8_1), for the max(z) NaN failure that only happens there
create table f8p (i int4, a int2, b int4, c int8, d float4, x float8, y float8, w float8, z float8, g2 int4, g3 int4, g4 int4, g5 int4, g6 int4, g7 int4);
insert into f8p select i,
	((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4 when i % 2003 = 0 then '-Infinity'::float4
	     else (((i * 17) % 10000) / 7.0)::float4 end,
	case when i % 53 = 0 then null else (((i::int8 * 7919) % 200001 - 100000) / 37.0)::float8 end,
	case when i % 59 = 0 then null when i % 5 = 0 then ((i % 2001) - 1000) + 0.5::float8
	     else (((i::int8 * 104729) % 1000003 - 500000) / 1009.0)::float8 end,
	(((i % 997) + 1) / 16.0)::float8,
	case i % 24 when 0 then 'NaN'::float8 when 1 then 'Infinity' when 2 then '-Infinity' when 3 then '-0' when 4 then 0
		when 5 then 5e-324 when 6 then 2.2250738585072014e-308 when 7 then 1.7976931348623157e308
		when 8 then -1.7976931348623157e308 when 9 then 1e-310 when 10 then 0.5 when 11 then -0.5
		when 12 then 1.5 when 13 then 2.5 when 14 then -2.5 when 15 then 9.2233720368547758e18
		when 16 then -9.2233720368547758e18 when 17 then 3.0000000000000004 when 18 then null
		when 19 then 4503599627370497.5 when 20 then 1e15 when 21 then -1e300 when 22 then 1e300
		else ((i % 1000) / 8.0)::float8 end,
	i % 2, i % 3, i % 4, i % 5, i % 6, i % 7
	from generate_series(1, 200000) i;
vacuum analyze f8p;
select relname, relpages, reltuples from pg_class where relname in ('np2', 'f8p');
