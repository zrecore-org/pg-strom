# A10: float8 on the GPU (the kernel computes it in integer arithmetic, kernels_vm/f64_soft.inc). Sourced by
# scan_test.sh and float8_dev.sh (they provide run, run_any, q, check, sql, $PSQL, $GPU, $CPU, $fail). New tables only (f8_*).
echo "float8 (A10)"
F8ROWS=${F8ROWS:-200000}

# x: ordinary values with NULLs; y: ordinary values (some exact ties of rint); w: positive divisors; z: every special
# value (NaN, +-Infinity, -0, 0, subnormals, the largest and smallest normals, halves for rint, 2^63 for the int8
# cast); d: float4 with NaN and Infinity; a, b, c: int2, int4, int8.
f8_create() { # table
	sql "create table $1 (i int4, a int2, b int4, c int8, d float4, x float8, y float8, w float8, z float8)"
	sql "insert into $1 select i,
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
			else ((i % 1000) / 8.0)::float8 end
		from generate_series(1, $F8ROWS) i"
}
f8_create f8_1; sql "vacuum analyze f8_1"                                  # all-visible pages
f8_create f8_2; sql "update f8_2 set x = x + 1 where i % 5 = 0"; sql "delete from f8_2 where i % 13 = 0"; sql "analyze f8_2"

# a query that must fail on the GPU path and on the CPU with the same message
runerr() { # table, label, query-with-%T, text of the error
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); want=$4
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$($PSQL -d scantest -c "$GPU $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if [ "$plan" -ge 1 ] && echo "$g" | grep -q "$want" && echo "$c" | grep -q "$want"; then echo "  ok    $tbl: $label ($want)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan, expected '$want')"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

for t in f8_1 f8_2; do
	echo "table $t"
	run $t "float8 compare with a constant" "select i, x from %T where x > 12.5"
	run $t "float8 =, <>, <, <=, >=" "select i, x, y from %T where x = y or x <> y and x <= 3.25 or y >= 10 or y < -9.75"
	run $t "float8 column vs column" "select i, x, y from %T where x < y and y <= 100"
	run $t "float8 between" "select i, x from %T where x between -250.5 and 250.25"
	run $t "NaN, Infinity, -0, 0 (PostgreSQL's float8 order)" "select i, z from %T where z = 'NaN' or z = 'Infinity' or z = '-Infinity' or z = 0"
	run $t "greater than a huge value (NaN is the largest)" "select i, z from %T where z > 1e300"
	run $t "less than, not equal, subnormals" "select i, z from %T where z < 1e-300 and z <> 0 or z = 5e-324"
	run $t "z compared with itself and a constant" "select i, z from %T where z = z and z >= -0.5 and z <= 2.5"
	run $t "IS NULL / IS NOT NULL" "select i, x, y, z from %T where x is null or y is not null and z is null"
	run $t "addition and subtraction" "select i, x + y, x - y, y + 0.1 from %T where i % 11 = 0"
	run $t "multiplication and division" "select i, x * y, x / w, y * w from %T where i % 7 = 3"
	run $t "nested arithmetic" "select i, (x * 2.5 + y) / w - x * x from %T where i % 13 = 4"
	run $t "specials in arithmetic (no overflow)" "select i, z + 1, z - z, z * 0, z + z from %T where z = 'NaN' or z = 'Infinity' or z = '-Infinity' or z = 0 or abs(z) between 1e-300 and 1e300 and abs(z) < 1e299"
	run $t "float8 negation and abs" "select i, -x, abs(y), -z, abs(z), +x from %T where i % 17 = 2"
	run $t "int -> float8" "select i, a::float8 + x, b::float8 * y, c::float8 / w from %T where b is not null and i % 3 = 0"
	run $t "float4 -> float8 (exact)" "select i, d::float8, d::float8 + x from %T where i % 5 = 1"
	run $t "float8 -> float4" "select i, x::float4, y::float4 from %T where abs(x) < 1e6 and abs(y) < 1e6 and i % 4 = 0"
	run $t "float8 -> int (rint, ties to even)" "select i, y::int4, (x / 7)::int8, z::int2 from %T where i % 3 = 0 and abs(z) < 30000 and z = z"
	run $t "mixed float4 / float8 comparison" "select i, d, x from %T where d < x or x >= d and d > 100.5"
	run $t "float4 vs float8 constant" "select i, d from %T where d > 100.5 and d <> 'NaN'"
	run $t "mixed float4 / float8 arithmetic" "select i, d + x, x - d, d * y, y / (d + 1000) from %T where i % 9 = 0 and d < 'Infinity' and d > '-Infinity' and d = d"
	run $t "float8 results next to other columns" "select i, a, b, x * 2, y from %T where i % 1000 = 7"
	run $t "12 computed float8 columns (16-register kernel)" "select x+1, x+2, x+3, x+4, x+5, x+6, y*2, y*3, y*4, x-y, x*y, x/w from %T where i % 211 = 0"
	run $t "24 computed float8 columns (32-register kernel)" "select x+1,x+2,x+3,x+4,x+5,x+6,x+7,x+8,x+9,x+10,x+11,x+12, y-1,y-2,y-3,y-4,y-5,y-6,y-7,y-8,y-9,y-10,y-11,y-12 from %T where i % 311 = 0"
	run $t "float8 compare after a computed float8" "select i from %T where x * y > 1000 and x / w < 5000"
	run $t "float8 pass-through with a qual on another column" "select x, y, z from %T where a = 123"
	run_any $t "float8 compare with a numeric constant that has many digits" "select i from %T where x > 12.3456789012345678901234567890"
	runerr $t "overflow (+)" "select z + z from %T where z = 1.7976931348623157e308" "value out of range: overflow"
	runerr $t "overflow (*)" "select z * 1e10 from %T where z = 1e300 or z = -1e300" "value out of range: overflow"
	runerr $t "underflow (*)" "select z * 0.5 from %T where z = 5e-324" "value out of range: underflow"
	runerr $t "underflow (/)" "select z / 1e300 from %T where z = 1e-310" "value out of range: underflow"
	runerr $t "division by zero" "select x / (z - z) from %T where z = 0" "division by zero"
	run $t "NaN divided by zero is NaN (float8, float4)" "select z / 0, d / 0::float4, d / (b - b) from %T where z = 'NaN' and d = 'NaN'"
	runerr $t "float8 -> int4 out of range" "select z::int4 from %T where z = 9.2233720368547758e18" "integer out of range"
	runerr $t "float8 -> int8 out of range" "select z::int8 from %T where z = 9.2233720368547758e18" "bigint out of range"
	runerr $t "float8 NaN -> int" "select z::int4 from %T where z = 'NaN'" "integer out of range"
	runerr $t "float8 -> float4 overflow" "select z::float4 from %T where z = 1e300" "value out of range: overflow"
	runerr $t "float8 -> float4 underflow" "select z::float4 from %T where z = 5e-324" "value out of range: underflow"
done

# joins: the float8 columns of both sides, in a qual that needs both, in the projection and in a join aggregate
sql "create table f8_dim as select (i * 3 - 50000)::int4 as k, (((i::int8 * 7919) % 2001 - 1000) / 16.0)::float8 as v,
	case when i % 11 = 0 then null else (((i::int8 * 104729) % 1999 - 999) / 8.0)::float8 end as u, (i % 5)::int4 as grp
	from generate_series(0, 40000) i"
sql "analyze f8_dim"
for t in f8_1 f8_2; do
	echo "join, table $t"
	ANYPLAN=0
	runjoin $t "float8 columns of both sides, compared in a join qual" "select t.i, d.v, t.x from %T t join f8_dim d on t.b = d.k where t.x > d.v"
	runjoin $t "float8 qual on the inner side" "select t.i, d.grp from %T t join f8_dim d on t.b = d.k where d.v < -100.5 or d.u > 100"
	runjoin $t "float8 arithmetic mixing both sides" "select t.i, t.x + d.v, d.u * t.y, t.x / (d.grp + 1), -d.v from %T t join f8_dim d on t.b = d.k where t.i % 3 = 0"
	runjoin $t "float8 compare of a computed value across the join" "select t.i, d.grp from %T t join f8_dim d on t.b = d.k where t.x * 2 > d.v + d.u"
	runjoin $t "LEFT JOIN with a float8 column of the inner side" "select t.i, d.v, d.u from %T t left join f8_dim d on t.b = d.k where t.i % 11 = 4"
	runjoin $t "aggregate over the join with a float8 qual" "select d.grp, count(*), sum(t.a) from %T t join f8_dim d on t.b = d.k where t.x > d.v group by 1 order by 1"
	ANYPLAN=1
	runjoin $t "float8 join key stays on the CPU" "select t.i, d.grp from %T t join f8_dim d on t.x = d.v"
done
