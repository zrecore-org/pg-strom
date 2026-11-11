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
