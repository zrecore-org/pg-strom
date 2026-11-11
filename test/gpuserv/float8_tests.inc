# A10: float8 on the GPU (the kernel computes it in integer arithmetic, kernels_vm/f64_soft.inc). Sourced by
# scan_test.sh and float8_dev.sh (they provide run, run_any, q, check, sql, $PSQL, $GPU, $CPU, $fail). New tables only (f8_*).
echo "float8 (A10)"
# the installed module is shared by every checkout: say so if it is not this tree's (it has the sum(float4) action)
strings "$($PGBIN/pg_config --pkglibdir)/pg_strom.dylib" 2>/dev/null | grep -q "psum::fp32" || echo "  NOTE: the installed pg_strom module lacks psum::fp32 (stale install?)"
# F8PARTS (default: all) picks the sections that run: scan (heap scans, casts, errors, math functions), join, agg, joinagg. The CI job
# runs "scan" only: on lavapipe every join and aggregate kernel is compiled by the CPU's JIT and the other sections need over an hour.
f8part() { case " ${F8PARTS:-scan join agg joinagg} " in *" $1 "*) echo f8_1 f8_2;; esac; }
F8ROWS=${F8ROWS:-200000}

# x: ordinary values with NULLs; y: ordinary values (some exact ties of rint); w: positive divisors; z: every special
# value (NaN, +-Infinity, -0, 0, subnormals, the largest and smallest normals, halves for rint, 2^63 for the int8
# cast); d: float4 with NaN and Infinity; a, b, c: int2, int4, int8.
f8_create() { # table
	sql "create table $1 (i int4, a int2, b int4, c int8, d float4, x float8, y float8, w float8, z float8, g2 int4, g3 int4, g4 int4, g5 int4, g6 int4, g7 int4)"
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
			else ((i % 1000) / 8.0)::float8 end,
		i % 2, i % 3, i % 4, i % 5, i % 6, i % 7
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

for t in $(f8part scan); do
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
	# math functions (sqrt, round = rint, ceil, ceiling, floor, trunc, sign): PostgreSQL's float.c, -0 kept except by sign
	run $t "round, ceil, ceiling, floor, trunc, sign over every special value" "select i, round(z), ceil(z), ceiling(z), floor(z), trunc(z), sign(z) from %T where i % 3 = 1"
	run $t "round, ceil, floor, trunc, sign over x and y (exact ties among y)" "select i, round(y), ceil(x), floor(y), trunc(x), sign(y), round(x), sign(x) from %T where i % 3 = 1"
	run $t "sqrt of non-negative values (w, |x|, |z| with NaN and Infinity)" "select i, sqrt(w), sqrt(abs(x)), sqrt(abs(z)) from %T where i % 3 = 2"
	run $t "sqrt(-0), sqrt(0), sqrt(Infinity), sqrt(NaN)" "select i, sqrt(z) from %T where z = 0 or z = 'Infinity' or z = 'NaN'"
	run $t "math functions as quals" "select i from %T where round(y) = 3 or ceil(x) > 100 and floor(x) < 200 or sign(z) = -1 and trunc(z) = -2"
	run $t "rounding to two decimals, a hypotenuse, ceil and floor of ratios" "select i, round(x * 100) / 100, sqrt(x * x + y * y), ceil(x / w), floor(y / w) from %T where i % 5 = 2"
	run $t "rounded values cast to integers" "select i, round(y)::int4, ceil(x)::int8, trunc(x)::int4 from %T where abs(x) < 1e6 and x = x and i % 4 = 0"
	run $t "ceil and floor next to float4 and integer columns" "select i, ceil(d::float8), floor(a::float8 / 7), round(b::float8 / 3) from %T where d = d and b is not null and d < 'Infinity' and d > '-Infinity' and i % 6 = 0"
	# sqrt raises on a negative argument: the quals below must not, on either path, whatever order they are written in
	run $t "guarded sqrt: x >= 0 and sqrt(x) > 20" "select i, x from %T where x >= 0 and sqrt(x) > 20"
	run $t "guarded sqrt, written in the other order" "select i, x from %T where sqrt(x) > 20 and x >= 0"
	run $t "x < 0 or sqrt(x) > 20" "select i, x from %T where x < 0 or sqrt(x) > 20"
	run $t "sqrt in the projection of non-negative rows" "select i, sqrt(x), round(sqrt(x) * 10) from %T where x >= 0"
	run $t "sqrt of a CASE-free guard through abs()" "select i, sqrt(abs(x) + 1) from %T where x < 0 and i % 9 = 0"
	runerr $t "sqrt of a negative number" "select sqrt(x) from %T where x < 0" "cannot take square root of a negative number"
	runerr $t "sqrt of -Infinity" "select sqrt(z) from %T where z = '-Infinity'" "cannot take square root of a negative number"
	runerr $t "sqrt of the smallest negative number" "select sqrt(-z) from %T where z = 5e-324" "cannot take square root of a negative number"
	runerr $t "sqrt of a negative number in a qual" "select i from %T where sqrt(x) > 1 and x < -5" "cannot take square root of a negative number"
done

# joins: the float8 columns of both sides, in a qual that needs both, in the projection and in a join aggregate
sql "create table f8_dim as select (i * 3 - 50000)::int4 as k, (((i::int8 * 7919) % 2001 - 1000) / 16.0)::float8 as v,
	case when i % 11 = 0 then null else (((i::int8 * 104729) % 1999 - 999) / 8.0)::float8 end as u, (i % 5)::int4 as grp
	from generate_series(0, 40000) i"
sql "analyze f8_dim"
for t in $(f8part join); do
	echo "join, table $t"
	ANYPLAN=0
	runjoin $t "float8 columns of both sides, compared in a join qual" "select t.i, d.v, t.x from %T t join f8_dim d on t.b = d.k where t.x > d.v"
	runjoin $t "float8 qual on the inner side" "select t.i, d.grp from %T t join f8_dim d on t.b = d.k where d.v < -100.5 or d.u > 100"
	runjoin $t "float8 arithmetic mixing both sides" "select t.i, t.x + d.v, d.u * t.y, t.x / (d.grp + 1), -d.v from %T t join f8_dim d on t.b = d.k where t.i % 3 = 0"
	runjoin $t "float8 compare of a computed value across the join" "select t.i, d.grp from %T t join f8_dim d on t.b = d.k where t.x * 2 > d.v + d.u"
	runjoin $t "math functions across the join" "select t.i, ceil(t.x), floor(d.v), sqrt(abs(d.u)), sign(t.y - d.v), round(t.x + d.v) from %T t join f8_dim d on t.b = d.k where t.i % 3 = 0"
	runjoin $t "LEFT JOIN with a float8 column of the inner side" "select t.i, d.v, d.u from %T t left join f8_dim d on t.b = d.k where t.i % 11 = 4"
	runjoin $t "aggregate over the join with a float8 qual" "select d.grp, count(*), sum(t.a) from %T t join f8_dim d on t.b = d.k where t.x > d.v group by 1 order by 1"
	ANYPLAN=1
	runjoin $t "float8 join key" "select t.i, d.grp from %T t join f8_dim d on t.x = d.v"
done

# float8 aggregates, with and without GROUP BY. The GPU adds in another order than a serial scan (with GROUP BY: the row order inside
# a workgroup, then the workgroups' partial sums in order), so the sums are checked on data whose sums are exact in any order
# (w: multiples of 1/16; float4 values; halves; small integers as float4) or rounded to 6 significant decimals; min and max are
# exact. A float8 / float4 sum that overflows raises PostgreSQL's error on both paths.
# like runagg, but compares the whole result (runagg compares the last row only): md5 of the output, GpuPreAgg in the plan
rungrp() { # table, label, query-with-%T [, extra settings]
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/ order by [0-9, ]*\$//"); set_extra=${4:-}
	gs="$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra"
	plan=$($PSQL -d scantest -c "$gs explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$gs $qry" 2>&1 | sort); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | sort)
	if { [ "$plan" -ge 1 ] || [ "$GRPANY" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR && [ -n "$g" ]; then echo "  ok    $tbl: $label ($(echo "$g" | wc -l | tr -d ' ') rows)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $(echo "$g" | head -3)"; echo "        cpu: $(echo "$c" | head -3)"; fail=$((fail+1)); fi
}
runerr_agg() { # table, label, query-with-%T, text of the error: the plan must use GpuPreAgg
	gmax="set pg_strom.gpupreagg_max_groups=1000000;"
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); want=$4
	plan=$($PSQL -d scantest -c "$GPU $gmax set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$GPU $gmax set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if [ "$plan" -ge 1 ] && echo "$g" | grep -q "$want" && echo "$c" | grep -q "$want"; then echo "  ok    $tbl: $label ($want)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan, expected '$want')"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
for t in $(f8part agg); do
	echo "aggregates, table $t"
	runagg $t "count, sum, min, max, avg of exact values" "select count(*), count(w), sum(w), min(w), max(w), avg(w) from %T"
	runagg $t "min, max of float8 with NULLs" "select min(x), max(x), min(y), max(y), count(x) from %T"
	runagg $t "min, max with NaN and Infinity (NaN is the largest)" "select min(z), max(z), count(z) from %T"
	runagg $t "min, max with NaN and Infinity after a CPU fallback (partial states merged by pgstrom.fmax_trans_fp64: NaN is the largest)" "select min(z), max(z), count(z) from %T" "set pg_strom.gpuserv_fault='refuse';"
	runagg $t "sum and avg rounded (inexact values)" "select round(sum(x)::numeric, 3), round(avg(y)::numeric, 6), count(y) from %T"
	runagg $t "sum of exact values with halves and big integers" "select sum(z), count(z) from %T where abs(z) < 1e10 and z <> 3.0000000000000004 and (abs(z) > 1e-300 or z = 0)"
	runagg $t "aggregates with a qual" "select sum(w), min(x), max(y), avg(w) from %T where i % 7 = 1 and a > 0"
	runagg $t "aggregates with FILTER" "select sum(w) filter (where i % 3 = 0), max(x) filter (where b > 0), min(y) filter (where a < 0), count(*) from %T"
	runagg $t "aggregate of a computed float8 argument" "select sum(w * 2 + 1), max(x * 2.5 - y), min(w / 4) from %T where i % 5 = 2"
	runagg $t "aggregates of math functions (integer-valued sums are exact in any order)" "select count(*), sum(round(x)), sum(ceil(y)), sum(floor(w)), sum(trunc(x)), max(sqrt(w)), min(sqrt(abs(y))), sum(sign(x)) from %T where x >= 0"
	runagg $t "max of sqrt over non-negative rows" "select max(sqrt(x)), min(sqrt(x)), count(*) from %T where x >= 0"
	runagg_any $t "max of sqrt with a FILTER that guards it (the argument is not evaluated for the rejected rows)" "select max(sqrt(x)) filter (where x >= 0), count(*) from %T"
	runagg $t "no row qualifies (NULL results)" "select sum(w), min(x), max(y), avg(w), count(*) from %T where a > 30000"
	runagg $t "float4: min, max, avg (double accumulation, exact)" "select min(d), max(d), avg(d), count(d) from %T"
	runagg $t "sum(float4): each addition rounded to float4 (small integers: exact in any order)" "select sum((a % 8)::float4), sum((b % 5)::float4), count(*) from %T"
	runagg $t "sum(float4) of NaN and Infinity" "select sum(d), count(d) from %T where d = 'NaN' or d = 'Infinity'"
	runagg $t "sum(float4) of +Infinity and -Infinity is NaN" "select sum(d) from %T where d = 'Infinity' or d = '-Infinity'"
	runagg $t "sum and avg of -0 (sum keeps the sign, avg starts from +0)" "select sum(z)::text, avg(z)::text, min(z)::text, max(z)::text from %T where i % 24 = 3"
	runerr_agg $t "sum(float4) overflows" "select sum((w * 0 + 3e38)::float4) from %T" "value out of range: overflow"
	rungrp $t "float8 aggregates with GROUP BY" "select g5, count(*), sum(w), min(w), max(w), avg(w) from %T group by 1"
	runagg $t "count and an int aggregate next to float8 ones" "select count(*), sum(b), max(c), sum(w), min(x) from %T"
	runerr_agg $t "sum overflows" "select sum(z) from %T where z = 1.7976931348623157e308" "value out of range: overflow"
done
# GROUP BY: every aggregate in gagg.slang / jgagg.slang (and the Arrow and GPU-Cache variants through their own tests). The rows of a
# group are added in the row order of a workgroup, the workgroups' partials in order: exact for exact data, rounded otherwise.
sql "create table f8_big as select i, (i % 7)::int4 as g, (i % 300)::int4 as g2, (((i * 31) % 1009) / 16.0)::float8 as w,
	(((i::int8 * 7919) % 200001 - 100000) / 37.0)::float8 as x, ((i % 11) - 5)::float4 as f from generate_series(1, 1000000) i"
sql "analyze f8_big"
for t in $(f8part agg); do
	echo "aggregates with GROUP BY, table $t"
	rungrp $t "min, max of float8 with NULLs, NaN and Infinity per group" "select g7, min(x), max(x), min(z), max(z), count(z) from %T group by 1"
	rungrp $t "sum and avg rounded (inexact values) per group" "select g5, round(sum(x)::numeric, 3), round(avg(y)::numeric, 6), count(y) from %T group by 1"
	rungrp $t "sum of exact specials per group" "select g3, sum(z), count(z) from %T where abs(z) < 1e10 and z <> 3.0000000000000004 and (abs(z) > 1e-300 or z = 0) group by 1"
	rungrp $t "FILTER and a qual per group" "select g4, sum(w) filter (where i % 3 = 0), max(x) filter (where b > 0), min(y) filter (where a < 0), avg(w) from %T where i % 7 <> 1 group by 1"
	rungrp $t "computed float8 arguments per group" "select g6, sum(w * 2 + 1), max(x * 2.5 - y), min(w / 4) from %T where i % 5 = 2 group by 1"
	rungrp $t "groups where every value is NULL or nothing qualifies" "select g3, min(x) filter (where x is null), sum(w) filter (where false), max(y) filter (where i < 0), count(*) from %T group by 1"
	rungrp $t "float4: min, max, avg per group" "select g5, min(d), max(d), avg(d), count(d) from %T group by 1"
	rungrp $t "sum(float4) per group (small integers: exact in any order)" "select g5, sum((a % 8)::float4), sum((b % 5)::float4) from %T group by 1"
	rungrp $t "sum(float4) of NaN and Infinity per group" "select g3, sum(d), count(d) from %T where d = 'NaN' or d = 'Infinity' or d = '-Infinity' group by 1"
	rungrp $t "sum and avg of -0 per group (sum keeps the sign, avg starts from +0)" "select g2, sum(z)::text, avg(z)::text, min(z)::text from %T where i % 24 = 3 group by 1"
	rungrp $t "int and float8 aggregates together per group" "select g5, count(*), sum(b), max(c), sum(w), min(x), avg(d) from %T group by 1"
	rungrp $t "two group keys" "select g3, b % 4, sum(w), max(x), count(*) from %T group by 1, 2 order by 1, 2"
	rungrp $t "more groups than the table holds (overflow rows aggregated by the service)" "select i % 300, sum(w), min(x), max(y), sum((a % 8)::float4), avg(w) from %T group by 1"
	runerr_agg $t "sum overflows in a group" "select g5, sum(z) from %T where z = 1.7976931348623157e308 group by 1" "value out of range: overflow"
	runerr_agg $t "sum(float4) overflows in a group" "select g5, sum((w * 0 + 3e38)::float4) from %T group by 1" "value out of range: overflow"
done
echo "aggregates with GROUP BY, 1,000,000 rows (the order of additions matters for inexact values: rounded; exact values: equal)"
rungrp f8_big "sum of exact values per group" "select g, count(*), sum(w), min(w), max(w), avg(w) from %T group by 1 order by 1"
rungrp f8_big "sum and avg of inexact values per group (rounded)" "select g, round(sum(x)::numeric, 2), round(avg(x)::numeric, 8), min(x), max(x) from %T group by 1 order by 1"
rungrp f8_big "sum(float4) of small integers per group" "select g, sum(f), min(f), max(f), avg(f) from %T group by 1 order by 1"
rungrp f8_big "300 groups (above the table capacity)" "select g2, sum(w), min(x), max(x), sum(f) from %T group by 1 order by 1"
runagg f8_big "ungrouped sums of 1,000,000 rows" "select sum(w), round(sum(x)::numeric, 2), sum(f), avg(w), min(x), max(x) from %T"
echo "join aggregates (float8, no GROUP BY)"
for t in $(f8part joinagg); do
	rungrp $t "float8 aggregates over a join" "select sum(d.v), min(t.x), max(d.u), avg(d.v), count(*) from %T t join f8_dim d on t.b = d.k" "$JOINSET"
	rungrp $t "float8 aggregates over a join with a float8 qual and a FILTER" "select sum(d.v) filter (where d.grp < 3), max(t.y), min(d.u) from %T t join f8_dim d on t.b = d.k where t.x > d.v" "$JOINSET"
	rungrp $t "float8 aggregate of a computed value across the join" "select sum(d.v * 2 + d.u), max(t.x - d.v) from %T t join f8_dim d on t.b = d.k where t.i % 3 = 0" "$JOINSET"
done
echo "join aggregates (float8, GROUP BY)"
for t in $(f8part joinagg); do
	rungrp $t "float8 aggregates over a join per group" "select d.grp, sum(d.v), min(t.x), max(d.u), avg(d.v), count(*) from %T t join f8_dim d on t.b = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	rungrp $t "float8 aggregates over a join with a float8 qual and a FILTER per group" "select d.grp, sum(d.v) filter (where d.grp < 3), max(t.y), min(d.u) from %T t join f8_dim d on t.b = d.k where t.x > d.v group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	rungrp $t "sum(float4) over a join per group" "select d.grp, sum((t.a % 8)::float4), sum((d.k % 5)::float4), count(*) from %T t join f8_dim d on t.b = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	GRPANY=1	# (the planner prices 40,000 groups above the CPU plan: the result must still be equal)
	rungrp $t "many groups over a join (overflow rows)" "select d.k, sum(d.v), min(t.x), max(d.u) from %T t join f8_dim d on t.b = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	GRPANY=0
done
GRPANY=0

# variance / stddev / covar / corr / regr_* (A10 part 5). The device accumulates PostgreSQL's own transition state (float8_accum:
# N, Sx, Sxx by the Youngs-Cramer update; float8_regr_accum: N, Sx, Sxx, Sy, Syy, Sxy), per thread in row order, merges the
# threads and workgroups with float8_combine / float8_regr_combine, and the Agg node finalizes with PostgreSQL's own functions
# (kernels_vm/vagg.inc; test/gpuserv/vagg_check.sh proves the arithmetic bit for bit against the server). The order of the rows
# differs from a serial scan, so for inexact data the last bits can differ (as between PostgreSQL's own parallel workers): the
# tests round to the digits that data supports, and use exactly representable data (multiples of 1/16, constants) where the
# result is bit-equal in any order. xo / yo have a mean of 1e9 / 2e9 and a variance of 1/3 / 3: the sum-of-squares formula
# (N * Sxx - Sx^2) of the earlier catalog loses every digit on them, Youngs-Cramer keeps them.
echo "variance, stddev, covar, corr, regr (A10)"
sql "create table f8_var as select i, (i % 5)::int4 as g, (i % 7)::int4 as g2, (i % 300)::int4 as g300,
	(1e9 + (((i::int8 * 7919) % 2001) - 1000) / 1000.0)::float8 as xo,
	(2e9 + 3.0 * (((i::int8 * 7919) % 2001) - 1000) / 1000.0 + (((i::int8 * 104729) % 1999) - 999) / 2000.0)::float8 as yo,
	(((i::int8 * 31) % 10007) / 100.0 - 50)::float8 as xs,
	((((i::int8 * 31) % 10007) / 100.0 - 50) * 0.7 + ((i * 17) % 101) / 10.0)::float8 as ys,
	case when i % 11 = 0 then null else (((i::int8 * 31) % 10007) / 100.0 - 50)::float8 end as xn,
	case when i % 13 = 0 then null else ((((i::int8 * 31) % 10007) / 100.0 - 50) * -0.4 + ((i * 17) % 101) / 10.0)::float8 end as yn,
	case when i % 5 = 3 and i % 1000 = 3 then 'NaN'::float8 when i % 5 = 4 and i % 1000 = 4 then 'Infinity'::float8
	     when i % 5 = 2 and i % 1000 = 2 then '-Infinity'::float8 else (((i::int8 * 31) % 10007) / 100.0 - 50)::float8 end as sp,
	(((i::int8 * 31) % 10007) / 100.0 - 50)::float4 as k,
	(((i * 37) % 1000) / 16.0)::float8 as dy,
	3.25::float8 as c3
	from generate_series(1, $F8ROWS) i"
sql "analyze f8_var"
VAR1="round(var_pop(xo)::numeric, 4), round(var_samp(xo)::numeric, 4), round(variance(xo)::numeric, 4), round(stddev_pop(xo)::numeric, 5), round(stddev_samp(xo)::numeric, 5), round(stddev(xo)::numeric, 5), count(xo)"
COV1="round(covar_pop(yo, xo)::numeric, 3), round(covar_samp(yo, xo)::numeric, 3), round(corr(yo, xo)::numeric, 6), round(regr_slope(yo, xo)::numeric, 5), round(regr_r2(yo, xo)::numeric, 6), round(regr_sxx(yo, xo)::numeric, 1), round(regr_syy(yo, xo)::numeric, 0), round(regr_sxy(yo, xo)::numeric, 0), round(regr_avgx(yo, xo)::numeric, 3), round(regr_avgy(yo, xo)::numeric, 3)"
REG1="round(regr_slope(ys, xs)::numeric, 6), round(regr_intercept(ys, xs)::numeric, 5), round(regr_r2(ys, xs)::numeric, 8), round(regr_sxx(ys, xs)::numeric, 1), round(regr_syy(ys, xs)::numeric, 1), round(regr_sxy(ys, xs)::numeric, 1), round(regr_avgx(ys, xs)::numeric, 5), round(regr_avgy(ys, xs)::numeric, 5), round(corr(ys, xs)::numeric, 8), round(covar_pop(ys, xs)::numeric, 5), round(covar_samp(ys, xs)::numeric, 5)"
if [ -n "$(f8part agg)" ]; then
	echo "variance and covariance, table f8_var"
	runagg f8_var "variance and stddev of values around 1e9 (cancellation-prone)" "select $VAR1 from %T"
	runagg f8_var "covariance, correlation and regression of values around 1e9 and 2e9" "select $COV1 from %T"
	runagg f8_var "regr_* and corr of ordinary values" "select $REG1 from %T"
	runagg f8_var "variance, stddev of ordinary values" "select round(var_pop(xs)::numeric, 6), round(var_samp(xs)::numeric, 6), round(stddev(xs)::numeric, 7), round(stddev_pop(ys)::numeric, 7), count(*) from %T"
	runagg f8_var "NULLs: a pair with a NULL argument is skipped" "select round(covar_pop(yn, xn)::numeric, 5), round(corr(yn, xn)::numeric, 8), round(regr_slope(yn, xn)::numeric, 7), round(regr_avgx(yn, xn)::numeric, 5), round(var_pop(xn)::numeric, 6), round(var_pop(yn)::numeric, 6), round(regr_syy(yn, xn)::numeric, 2) from %T"
	runagg f8_var "NaN in the data gives NaN" "select var_pop(sp), stddev(sp), covar_pop(sp, xs), corr(xs, sp), regr_syy(sp, xs) from %T where g = 3"
	runagg f8_var "Infinity in the data gives NaN" "select var_samp(sp), stddev_pop(sp), regr_slope(ys, sp), covar_samp(sp, ys) from %T where g = 4"
	runagg f8_var "-Infinity in the data gives NaN" "select variance(sp), corr(sp, ys) from %T where g = 2"
	runagg_any f8_var "data without specials in the same table (an OR in FILTER keeps the plan on the CPU)" "select round(var_pop(sp) filter (where g = 0 or g = 1)::numeric, 6), round(stddev(sp) filter (where g = 0 or g = 1)::numeric, 6), count(*) from %T"
	runagg f8_var "one row: var_samp is NULL, var_pop is 0" "select var_samp(xo) filter (where i = 17), var_pop(xo) filter (where i = 17), stddev_samp(xo) filter (where i = 17), stddev_pop(xo) filter (where i = 17), covar_samp(yo, xo) filter (where i = 17), covar_pop(yo, xo) filter (where i = 17), corr(yo, xo) filter (where i = 17), regr_slope(yo, xo) filter (where i = 17), regr_sxx(yo, xo) filter (where i = 17), regr_avgx(yo, xo) filter (where i = 17) from %T"
	runagg f8_var "one row with NaN" "select var_pop(sp) filter (where i = 3), corr(sp, xs) filter (where i = 3), covar_pop(sp, xs) filter (where i = 3), regr_sxx(sp, xs) filter (where i = 3) from %T"
	runagg f8_var "two rows (rounded: the order of the two rows changes the last bits of non-dyadic data)" "select round(var_samp(xs) filter (where i = 17 or i = 18)::numeric, 9), round(var_pop(xs) filter (where i = 17 or i = 18)::numeric, 9), round(covar_samp(ys, xs) filter (where i = 17 or i = 18)::numeric, 9), round(corr(ys, xs) filter (where i = 17 or i = 18)::numeric, 9) from %T"
	runagg f8_var "no row qualifies: NULL" "select var_pop(xs), stddev(xs), covar_pop(ys, xs), corr(ys, xs), regr_slope(ys, xs), regr_r2(ys, xs), regr_avgx(ys, xs), count(*) from %T where i < 0"
	runagg f8_var "constant values: variance 0, correlation NULL" "select var_pop(c3), var_samp(c3), stddev(c3), covar_pop(c3, c3), corr(c3, xs), regr_slope(c3, xs), regr_r2(xs, c3), regr_sxx(c3, c3) from %T"
	runagg f8_var "multiples of 1/16 and constants: exact in any order" "select regr_avgx(dy, dy), regr_avgy(dy, dy), covar_pop(c3, c3), var_pop(c3) from %T"
	runagg f8_var "FILTER" "select round(var_pop(xo) filter (where i % 3 = 0)::numeric, 4), round(corr(yo, xo) filter (where i % 4 = 1)::numeric, 6), round(stddev(xs) filter (where g = 2)::numeric, 7), round(covar_pop(ys, xs) filter (where false)::numeric, 3) from %T"
	runagg f8_var "a qual" "select round(var_pop(xo)::numeric, 4), round(regr_slope(yo, xo)::numeric, 5), count(*) from %T where i % 7 = 1 and xs > 0"
	runagg f8_var "float4 arguments" "select round(variance(k)::numeric, 5), round(stddev_pop(k)::numeric, 6), round(corr(k, ys)::numeric, 8), round(regr_slope(ys, k)::numeric, 7), round(covar_pop(k, k)::numeric, 5) from %T"
	runagg f8_var "computed arguments" "select round(var_pop(xs * 2 + 1)::numeric, 5), round(covar_pop(xs, ys * 3 - xs)::numeric, 4), round(corr(xs + ys, xs - ys)::numeric, 8) from %T where i % 5 = 2"
	runagg_any f8_var "variance and stddev of integers (exact integer state, A10 part 5b) agree" "select var_pop(g2), stddev(g2), variance(g) from %T"
	runagg f8_var "variance together with sum, count, min and max" "select count(*), round(sum(xs)::numeric, 3), min(xo), max(yo), round(var_pop(xs)::numeric, 6), round(corr(ys, xs)::numeric, 8), round(avg(ys)::numeric, 6) from %T"
	runagg f8_var "four regression aggregates at once (twelve accumulators)" "select round(regr_slope(ys, xs)::numeric, 6), round(regr_intercept(ys, xs)::numeric, 5), round(corr(yn, xn)::numeric, 8), round(covar_pop(yo, xo)::numeric, 3) from %T"
	runagg_any f8_var "regr_count stays on the CPU" "select regr_count(ys, xs), regr_count(yn, xn) from %T"
	runerr_agg f8_1 "variance of finite values that overflows" "select var_pop(z) from %T where z = 1.7976931348623157e308 or z = -1.7976931348623157e308" "value out of range: overflow"
	runerr_agg f8_1 "covariance of finite values that overflows" "select covar_pop(z, z) from %T where z = 1.7976931348623157e308 or z = -1.7976931348623157e308" "value out of range: overflow"
fi
if [ -n "$(f8part agg)" ]; then
	echo "variance and covariance with GROUP BY, table f8_var"
	rungrp f8_var "variance, stddev around 1e9 per group" "select g, round(var_pop(xo)::numeric, 4), round(var_samp(xo)::numeric, 4), round(stddev(xo)::numeric, 5), round(stddev_pop(xo)::numeric, 5), count(xo) from %T group by 1"
	rungrp f8_var "covariance, correlation and regression around 1e9 and 2e9 per group" "select g2, round(covar_pop(yo, xo)::numeric, 3), round(covar_samp(yo, xo)::numeric, 3), round(corr(yo, xo)::numeric, 6), round(regr_slope(yo, xo)::numeric, 5), round(regr_r2(yo, xo)::numeric, 6), round(regr_sxx(yo, xo)::numeric, 1), round(regr_avgx(yo, xo)::numeric, 3) from %T group by 1"
	rungrp f8_var "regr_* of ordinary values per group" "select g, round(regr_slope(ys, xs)::numeric, 6), round(regr_intercept(ys, xs)::numeric, 5), round(regr_r2(ys, xs)::numeric, 8), round(regr_sxy(ys, xs)::numeric, 1), round(corr(ys, xs)::numeric, 8), round(regr_avgy(ys, xs)::numeric, 5) from %T group by 1"
	rungrp f8_var "NULL pairs per group" "select g2, round(covar_pop(yn, xn)::numeric, 5), round(corr(yn, xn)::numeric, 8), round(var_pop(xn)::numeric, 6), round(var_samp(yn)::numeric, 6) from %T group by 1"
	rungrp f8_var "NaN and Infinity per group (groups 2, 3 and 4 have them)" "select g, round(var_pop(sp)::numeric, 6), round(stddev(sp)::numeric, 6), round(corr(sp, xs)::numeric, 9), round(covar_pop(xs, sp)::numeric, 6) from %T group by 1"
	
	rungrp f8_var "FILTER per group" "select g, round(var_pop(xo) filter (where i % 3 = 0)::numeric, 4), round(corr(yo, xo) filter (where i % 4 = 1)::numeric, 6), round(var_samp(xs) filter (where g2 = 2)::numeric, 6) from %T group by 1"
	rungrp f8_var "groups of one row and of two rows, and groups with no qualifying row" "select i / 2, round(var_samp(xs)::numeric, 8), round(var_pop(xs)::numeric, 8), round(covar_pop(ys, xs)::numeric, 8), round(corr(ys, xs)::numeric, 9), var_samp(xs) filter (where i % 2 = 0), var_pop(xs) filter (where i % 2 = 0), covar_samp(ys, xs) filter (where i % 2 = 1 and i < 0) from %T group by 1"
	rungrp f8_var "groups with no qualifying row" "select g, var_pop(xs) filter (where false), corr(ys, xs) filter (where i < 0), round(var_pop(xo)::numeric, 4) from %T group by 1"
	rungrp f8_var "multiples of 1/16 in groups of two (exact in any order)" "select i / 2, var_pop(dy), var_samp(dy), covar_pop(dy, dy), regr_sxy(dy, dy), regr_avgx(dy, dy) from %T group by 1"
	rungrp f8_var "constants per group" "select g, var_pop(c3), stddev(c3), corr(c3, xs), regr_slope(xs, c3), covar_samp(c3, c3) from %T group by 1"
	rungrp f8_var "two group keys" "select g, g2, round(var_pop(xs)::numeric, 5), round(corr(ys, xs)::numeric, 8) from %T group by 1, 2 order by 1, 2"
	rungrp f8_var "more groups than the table holds (overflow rows aggregated by the service)" "select g300, round(var_pop(xo)::numeric, 4), round(covar_pop(yo, xo)::numeric, 3), round(corr(yo, xo)::numeric, 5), round(stddev_samp(xs)::numeric, 6) from %T group by 1"
	rungrp f8_var "variance next to other float8 and integer aggregates" "select g, count(*), round(sum(xs)::numeric, 3), min(xo), max(yo), round(var_pop(xs)::numeric, 6), round(regr_slope(ys, xs)::numeric, 6), round(avg(xs)::numeric, 6) from %T group by 1"
	rungrp f8_var "float4 arguments per group" "select g, round(variance(k)::numeric, 5), round(corr(k, ys)::numeric, 8) from %T group by 1"
	GRPANY=1
	rungrp f8_var "variance of integers per group (A10 part 5b)" "select g, var_pop(g2), stddev(g300) from %T group by 1"
	GRPANY=0
	runerr_agg f8_1 "variance overflows in a group" "select g5, var_pop(z) from %T where z = 1.7976931348623157e308 or z = -1.7976931348623157e308 group by 1" "value out of range: overflow"
	runerr_agg f8_1 "covariance overflows in a group" "select g5, covar_samp(z, z) from %T where z = 1.7976931348623157e308 or z = -1.7976931348623157e308 group by 1" "value out of range: overflow"
fi
echo "variance and covariance, 1,000,000 rows"
sql "alter table f8_big add column xo float8, add column yo float8"
sql "update f8_big set xo = (1e9 + (((i::int8 * 7919) % 2001) - 1000) / 1000.0)::float8, yo = (2e9 + 3.0 * (((i::int8 * 7919) % 2001) - 1000) / 1000.0 + (((i::int8 * 104729) % 1999) - 999) / 2000.0)::float8"
sql "vacuum analyze f8_big"
runagg f8_big "variance, covariance and regression of values around 1e9, ungrouped" "select round(var_pop(xo)::numeric, 4), round(stddev_samp(xo)::numeric, 5), round(covar_pop(yo, xo)::numeric, 3), round(corr(yo, xo)::numeric, 6), round(regr_slope(yo, xo)::numeric, 5), round(var_samp(x)::numeric, 5), round(corr(x, w)::numeric, 7) from %T"
rungrp f8_big "variance and covariance per group" "select g, round(var_pop(xo)::numeric, 4), round(covar_samp(yo, xo)::numeric, 3), round(corr(yo, xo)::numeric, 6), round(stddev(x)::numeric, 5), round(regr_slope(w, x)::numeric, 8) from %T group by 1 order by 1"
rungrp f8_big "300 groups (above the table capacity)" "select g2, round(var_pop(xo)::numeric, 3), round(corr(yo, xo)::numeric, 5), round(regr_slope(w, x)::numeric, 7) from %T group by 1 order by 1"
echo "variance and covariance over a join"
for t in $(f8part joinagg); do
	rungrp $t "variance and covariance over a join" "select round(var_pop(d.v)::numeric, 5), round(stddev(t.x)::numeric, 5), round(corr(t.x, d.v)::numeric, 8), round(covar_pop(t.y, d.u)::numeric, 5), round(regr_slope(d.v, t.x)::numeric, 7), count(*) from %T t join f8_dim d on t.b = d.k" "$JOINSET"
	rungrp $t "variance over a join with a FILTER and a float8 qual" "select round(var_samp(d.v) filter (where d.grp < 3)::numeric, 5), round(corr(t.y, t.x)::numeric, 8) from %T t join f8_dim d on t.b = d.k where t.x > d.v" "$JOINSET"
	rungrp $t "variance and covariance over a join per group" "select d.grp, round(var_pop(d.v)::numeric, 5), round(stddev(t.x)::numeric, 5), round(corr(t.x, d.v)::numeric, 8), round(covar_samp(t.y, d.u)::numeric, 5), count(*) from %T t join f8_dim d on t.b = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	GRPANY=1
	rungrp $t "variance over a join, many groups (overflow rows)" "select d.k, round(var_pop(t.x)::numeric, 3), round(corr(t.x, d.v)::numeric, 5), round(covar_pop(t.x, d.v)::numeric, 3) from %T t join f8_dim d on t.b = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
	GRPANY=0
done
