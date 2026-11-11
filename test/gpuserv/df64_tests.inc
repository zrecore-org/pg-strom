# A10 part 6 (S1): the df64 prefilter of float8 comparisons (DFCMP). Sourced by float8_tests.inc (so by scan_test.sh and float8_dev.sh) and
# by df64_dev.sh. The prefilter may only save work: every result must equal PostgreSQL's, so the queries sit right at the thresholds
# where a double-float evaluation with an error bound can and cannot decide (margins from 1e-3 down to a few ulps, exact ties, decimals),
# next to NULLs and values outside the prefilter's range. Needs $PSQL, run, runerr, check, sql, $fail.
echo "df64 prefilter (A10 part 6)"
type runerr >/dev/null 2>&1 || runerr() { # table, label, query-with-%T, text of the error (as float8_tests.inc)
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); want=$4
	plan=$($PSQL -d scantest -c "$GPU explain $qry" | grep -c "Custom Scan (GpuScan)")
	g=$($PSQL -d scantest -c "$GPU $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if [ "$plan" -ge 1 ] && echo "$g" | grep -q "$want" && echo "$c" | grep -q "$want"; then echo "  ok    $tbl: $label ($want)"
	else echo "  FAIL  $tbl: $label (plan GpuScan=$plan, expected '$want')"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
[ -n "$ROOT" ] && [ -f "$ROOT/log" ] && { if grep -q "df64 prefilter on" "$ROOT/log"; then echo "  note: the service runs with the df64 prefilter"; else echo "  NOTE: the df64 prefilter is OFF in this service ($(sed -n 's/.*\(df64 prefilter off.*\)/\1/p' "$ROOT/log" | head -1))"; fi; }
DFROWS=${DFROWS:-120000}
sql "drop table if exists f8_df"
# x, y: values around the thresholds of the queries below. Row class i % 16: the margin of the row from its threshold is 10^-(1 + i % 16) of the operand
# (relative), alternating sides; class 15 sits a few ulps from it; i % 5 = 0 rows are decimals (exact ties); NULLs and out-of-range values mixed in.
sql "create table f8_df (i int4, x float8, y float8, w float8, c float8, z float8)"
sql "insert into f8_df select i,
	case when i % 97 = 0 then null
	     when i % 5 = 0 then ((i % 4000) / 4.0)::float8
	     else ((850.0 + 3.3) / 1.7) * (1 + (case when i % 2 = 0 then 1 else -1 end) * power(10::float8, -1 - (i % 16)) * ((i % 7) + 1) / 7.0) end,
	case when i % 89 = 0 then null
	     when i % 5 = 0 then ((i % 4000) / 4.0 + 50)::float8
	     else (((850.0 + 3.3) / 1.7) + 50.0 * (1 + (case when i % 3 = 0 then 1 else -1 end) * power(10::float8, -1 - (i % 16)))) end,
	((i % 1000) + 1) / 8.0,
	case when i % 1013 = 0 then null else 850.0 + ((i % 11) - 5) * 1e-13 * (i % 3) end,
	case i % 10 when 0 then 'NaN'::float8 when 1 then 'Infinity'::float8 when 2 then 1e-300 when 3 then 1e300 when 4 then 0 when 5 then -0.0
	     when 6 then 5e-324 when 7 then 1.7e19 when 8 then 3.3e-19 else ((i % 997) - 400) / 16.0 end
	from generate_series(1, $DFROWS) i"
sql "update f8_df set y = x + 50 where i % 17 = 0 and x is not null"          # |x - y| = 50 exactly in the reals, may be off by an ulp in doubles
sql "vacuum analyze f8_df"
t=f8_df
run $t "x*1.7 - 3.3 < 850 (a threshold the rows approach from 1e-1 to 1e-16)" "select i, x from %T where x * 1.7 - 3.3 < 850"
run $t "x*1.7 - 3.3 <= 850" "select i, x from %T where x * 1.7 - 3.3 <= 850"
run $t "x*1.7 - 3.3 > 850" "select i, x from %T where x * 1.7 - 3.3 > 850"
run $t "x*1.7 - 3.3 >= 850" "select i, x from %T where x * 1.7 - 3.3 >= 850"
run $t "x*1.7 - 3.3 = 850" "select i, x from %T where x * 1.7 - 3.3 = 850"
run $t "x*1.7 - 3.3 <> 850" "select i, x from %T where x * 1.7 - 3.3 <> 850"
run $t "850 > x*1.7 - 3.3 (the constant on the left)" "select i, x from %T where 850 > x * 1.7 - 3.3"
run $t "x*1.7 - 3.3 < c (a column on the right, a few ulps away)" "select i, x, c from %T where x * 1.7 - 3.3 < c"
run $t "abs(x - y) < 50" "select i, x, y from %T where abs(x - y) < 50"
run $t "abs(x - y) <= 50" "select i, x, y from %T where abs(x - y) <= 50"
run $t "abs(x - y) = 50" "select i, x, y from %T where abs(x - y) = 50"
run $t "abs(y - x) >= 50" "select i, x, y from %T where abs(y - x) >= 50"
run $t "x - y = -50" "select i, x, y from %T where x - y = -50"
run $t "x*x + y*y < 250000" "select i, x, y from %T where x * x + y * y < 250000"
run $t "x*x + y*y >= 1000000" "select i, x, y from %T where x * x + y * y >= 1000000"
run $t "(x + y) * (x - y) < 1000" "select i, x, y from %T where (x + y) * (x - y) < 1000"
run $t "-x * 2 > -y" "select i, x, y from %T where -x * 2 > -y"
run $t "x * w + y * c < 90000" "select i, x from %T where x * w + y * c < 90000"
run $t "x * 0.5 + y * 0.5 = (x + y) * 0.5 (ties in the reals, equal or one ulp apart in doubles)" "select i, x, y from %T where x * 0.5 + y * 0.5 = (x + y) * 0.5"
run $t "x + y + 0.1 + 0.2 = x + y + 0.3 (0.1 + 0.2 is not 0.3)" "select i from %T where x + y + 0.1 + 0.2 = x + y + 0.3"
run $t "x * 1.7 - 3.3 < 850 and abs(x - y) < 50 (two prefilters in one AND)" "select i from %T where x * 1.7 - 3.3 < 850 and abs(x - y) < 50"
run $t "x * 1.7 - 3.3 < 850 or abs(x - y) >= 50 (an OR)" "select i from %T where x * 1.7 - 3.3 < 850 or abs(x - y) >= 50"
run $t "not (x * 1.7 - 3.3 < 850)" "select i from %T where not (x * 1.7 - 3.3 < 850)"
run $t "(x * 1.7 - 3.3 < 850) is null" "select i from %T where (x * 1.7 - 3.3 < 850) is null"
run $t "(x * 1.7 - 3.3 < 850) in the target list" "select i, x * 1.7 - 3.3 < 850, abs(x - y) < 50 from %T where i % 13 = 0"
echo "  z: NaN, Infinity, 1e-300, 1e300, 0, -0, 5e-324, 1.7e19 (outside the prefilter's range: the exact path decides)"
run $t "z * 2 > 1" "select i, z from %T where z * 2 > 1"
run $t "z + z = z" "select i, z from %T where z + z = z"
run $t "z - x < 5" "select i, z from %T where z - x < 5"
run $t "abs(z - 1e19) < 1e19" "select i, z from %T where abs(z - 1e19) < 1e19"
runerr $t "1e300 * z overflows (the prefilter must not decide it)" "select i from %T where z * 1e300 > 1" "value out of range: overflow"
runerr $t "z * 1e-300 underflows on a nonzero z" "select i from %T where i % 10 = 6 and z * 1e-300 > 1" "value out of range: underflow"
echo "  a null leaf next to an overflowing one: PostgreSQL still evaluates both operands"
runerr $t "(x * 2) + (z * 1e300) with a NULL x row first" "select i from %T where (x * 2) + (z * 1e300) > 1 and i % 10 = 3" "value out of range: overflow"
echo "  float4 under the service's fast-math-off setting (the switch is global; FARITH must still equal PostgreSQL's float4 arithmetic bit for bit)"
sql "drop table if exists f8_df4"
sql "create table f8_df4 (i int4, p float4, q float4, r float4)"
sql "insert into f8_df4 select i,
	case i % 11 when 0 then 'NaN'::float4 when 1 then 'Infinity'::float4 when 2 then '-Infinity'::float4 when 3 then 1.0e-40::float4 when 4 then 3.0e38::float4
	     when 5 then -0.0::float4 else (((i::int8 * 7919) % 200001 - 100000) / 37.0)::float4 end,
	case i % 7 when 0 then 1.1754944e-38::float4 when 1 then 16777216::float4 when 2 then 0.1::float4 else (((i::int8 * 104729) % 1000003 - 500000) / 1009.0)::float4 end,
	((i % 997) + 1) / 16.0
	from generate_series(1, $DFROWS) i"
sql "vacuum analyze f8_df4"
t=f8_df4
run $t "float4: p * q + r < 50" "select i, p, q, r from %T where p * q + r < 50"
run $t "float4: (p + q) * r > 1000" "select i from %T where (p + q) * r > 1000"
run $t "float4: p * 0.1 + q * 0.7 = q * 0.7 + p * 0.1 (fused or not: the sums are one ulp apart)" "select i from %T where p * 0.1::float4 + q * 0.7::float4 = q * 0.7::float4 + p * 0.1::float4"
run $t "float4: p / r <= q" "select i, p / r, q from %T where p / r <= q"
run $t "float4: results in the target list" "select i, p * q, p + q, p - q, p * q + r, (p + q) * r from %T where i % 5 = 0"
run $t "float4 and float8 mixed: p::float8 * q::float8 + r < 50" "select i from %T where p::float8 * q::float8 + r::float8 < 50"
run_any $t "float4: abs(p - q) < r (float4 abs stays on the CPU)" "select i from %T where abs(p - q) < r"
