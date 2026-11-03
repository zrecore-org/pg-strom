# A6: numeric on the GPU. Sourced by scan_test.sh (it provides run, run_any, runagg, runagg_any, runjoin,
# q, check, sql, $PSQL, $GPU, $CPU, $fail). New tables only (nm_*).
#
# A numeric register is { unscaled int64, scale }; what does not fit (more than about 18 digits, NaN,
# Infinity, an overflowing intermediate result) is evaluated by the backend ("CPU re-evaluation of N tuples"
# with client_min_messages = debug1). Results are compared as text, so a wrong scale shows up.
echo "numeric (A6)"
NMROWS=${NMROWS:-300000}

# n1 numeric(10,2) with NULLs and zeros; n2 numeric(18,6); n3 numeric(38,10) with a few values that need more than
# 64 bits and a few tiny ones; n4 unconstrained: integers, long fractions (more than 64 bits unscaled), NaN, +-Infinity,
# values beyond 64 bits, zeros with a scale, trailing zero digits, a value with dscale 70 (long header format).
nm_create() { # table name
	sql "create table $1 as select i::int4 as i, (i % 1000)::int4 as k, (i::int8 * 1000)::int8 as k8,
		case when i % 41 = 0 then null when i % 101 = 0 then 0 else ((i::int8 * 7919) % 2000000 - 1000000)::numeric / 100 end::numeric(10,2) as n1,
		case when i % 43 = 0 then null else ((i::int8 * 1000003) % 1000000000000 - 500000000000)::numeric / 1000000 end::numeric(18,6) as n2,
		case when i % 47 = 0 then null when i % 1000 = 0 then 12345678901234567890123.4567890123
		     when i % 3001 = 0 then 0.0000000001 else ((i * 31) % 100000 - 50000)::numeric / 1000 end::numeric(38,10) as n3,
		case i % 12 when 0 then null when 1 then i::numeric when 2 then i::numeric / 7 when 3 then 'NaN'::numeric
			when 4 then -(i::numeric) / 1000 when 5 then 100000000000000000000::numeric + i when 6 then 0.00
			when 7 then 'Infinity'::numeric when 8 then 12.500000 when 9 then '-Infinity'::numeric
			when 10 then '3e-70'::numeric else (i % 100)::numeric / 4 end as n4
		from generate_series(1, $NMROWS) i"
}
nm_create nm1; sql "vacuum analyze nm1"                                  # all-visible pages
nm_create nm2; sql "update nm2 set n1 = n1 + 1 where i % 5 = 0"; sql "delete from nm2 where i % 13 = 0"; sql "analyze nm2"

for t in nm1 nm2; do
	echo "table $t"
	run $t "numeric(10,2) = constant" "select i, n1 from %T where n1 = 12.34 or n1 = 0 or n1 = -0.5"
	run $t "numeric(10,2) <> / < / <= / > / >= constants" "select i, n1 from %T where n1 <> 0.00 and n1 > -100.5 and n1 < 100 and n1 >= -50 and n1 <= 25.25"
	run $t "constants with another scale" "select i, n1, n2 from %T where n1 > 3 and n1 < 20.5000 and n2 <> 1.5"
	run $t "numeric(18,6) compare" "select i, n2 from %T where n2 > 1000.123456 and n2 <= 100000"
	run $t "columns of different scales" "select i, n1, n2 from %T where n1 < n2 and n1 > -1000"
	run $t "equality of columns of different scales" "select i, n1, n3 from %T where n1 = n3 or n2 = n3"
	run $t "numeric(38,10) (values beyond 64 bits are rechecked)" "select i, n3 from %T where n3 > 40 or n3 < -49.9"
	run $t "numeric(38,10) tiny and huge values" "select i, n3 from %T where n3 = 0.0000000001 or n3 > 1000000 or n3 = 0"
	run $t "numeric vs int4 column" "select i, n1, k from %T where n1 > k and k > 500"
	run $t "numeric vs int8 column" "select i, n2, k8 from %T where n2 < k8 and n2 > 100000"
	run $t "numeric vs int constants" "select i, n1 from %T where n1 > 5 and n1 < 100"
	run $t "IS NULL / IS NOT NULL on numeric columns" "select i, n1, n2 from %T where n1 is null or n2 is not null and n3 is null"
	run $t "IS NULL on n4 reads no value (NaN, Infinity)" "select i from %T where n4 is null"
	run $t "numeric pass-through with qual on another column" "select i, n1, n2, n3, n4 from %T where k = 7"
	run $t "numeric and integer quals together" "select i, n1, k from %T where k < 300 and n1 > 0 and i % 3 = 0"
	run $t "OR / NOT with NULL numerics" "select i from %T where not (n1 > 0) or n2 < -250000 and n3 is not null"
	echo "  -- arithmetic (a computed numeric is formed into a varlena by the service)"
	run $t "n1 + n2, n1 - n2" "select i, n1 + n2, n1 - n2 from %T where i % 7 = 0"
	run $t "n1 * n2 (scale 8; some products need more than 64 bits)" "select i, n1 * n2 from %T where i % 5 = 0"
	run $t "n2 * n2 (overflow of 64 bits: rechecked)" "select i, n2 * n2, n2 + n2 from %T where i % 11 = 0"
	run $t "n3 arithmetic with huge values" "select i, n3 + n3, n3 - 1, n3 * 3 from %T where i % 13 = 0"
	run $t "unary minus and abs" "select i, -n1, abs(n2), -n3, abs(n1) from %T where i % 9 = 0"
	run $t "unary plus" "select i, +n1 from %T where i % 17 = 0"
	run $t "arithmetic with constants" "select i, n1 + 1, n1 - 0.005, n1 * 2.5, n2 + 1000000000000 from %T where i % 19 = 0"
	run $t "arithmetic in the qual" "select i, n1, n2 from %T where n1 * 2 > n2 / 1000 + 5 and n1 + n2 < 100000"
	run $t "arithmetic in qual with NULLs and zeros" "select i, n1 from %T where n1 * n1 = 0 or n1 - n1 = 0.00"
	run $t "int to numeric arithmetic" "select i, k::numeric + n1, k8::numeric * n1, (i::int2)::numeric - n2 from %T where i % 23 = 0"
	run $t "int4 + numeric (cast on the integer)" "select i, k + n1, n2 - k from %T where i % 29 = 0"
	run $t "unconstrained numeric: integers, NaN, Infinity, long fractions" "select i, n4, n4 + 1, n4 * 2 from %T where i % 3 = 1"
	run $t "n4 compare" "select i, n4 from %T where n4 > 100 or n4 = 0 or n4 < -50"
	run_any $t "n4 compare with NaN and Infinity constants (not representable: CPU)" "select i, n4 from %T where n4 = 'NaN' or n4 > 'Infinity' or n4 < '-Infinity' or n4 = 'Infinity'"
	run $t "dscale 70 (long header format) and its product (dscale 140)" "select i, n4 * 2, n4 * n4, n4 + n4, -n4 from %T where i % 12 = 10"
	run $t "trailing zeros and scale of zero" "select i, n4, n4 + 0.1, n4 * 1.00 from %T where i % 12 in (6, 8)"
	run $t "result scale: max of the two for sums, sum for products" "select n1 + n3, n1 - n2, n1 * n3, n3 * n3 from %T where i % 31 = 0"
	echo "  -- casts"
	run $t "numeric -> int4, rounding half away from zero" "select i, n1, n1::int4, (-n1)::int4 from %T where n1 between -100 and 100 and i % 3 = 0"
	run $t "numeric -> int8 and int2" "select i, n2::int8, n1::int2 from %T where n2 between -30000 and 30000 and n1 between -300 and 300"
	run $t "numeric -> int in a qual" "select i, n1 from %T where n1::int4 = 7 or n2::int8 > 499990"
	run $t "numeric -> int4 of an expression" "select i, (n1 * 10 + n2)::int4 from %T where abs(n2) < 1000 and abs(n1) < 100"
	run $t "numeric(18,6) -> int8 with half values" "select i, (n2 * 0.5)::int8, (n2 * 5)::int4 from %T where i % 1000 = 3"
	echo "  -- stays on the CPU"
	run_any $t "numeric division" "select i, n1 / n2 from %T where n2 <> 0 and i % 7 = 0"
	run_any $t "numeric remainder" "select i, n1 % 7 from %T where i % 7 = 0"
	run_any $t "round() and trunc()" "select i, round(n1, 1), trunc(n2, 3) from %T where i % 7 = 0"
	run_any $t "numeric to float" "select i, n1::float4, n2::float8 from %T where i % 7 = 0"
	run_any $t "numeric typmod cast" "select i, n2::numeric(10,2) from %T where abs(n2) < 1000 and i % 5 = 0"
	run_any $t "sign()" "select i, sign(n1) from %T where i % 7 = 0"
	run_any $t "numeric in IN list and CASE" "select i, case when n1 > 0 then n2 else 0 end from %T where n1 in (1, 2, 3.5)"
	run_any $t "coalesce(numeric)" "select i, coalesce(n1, 0) from %T where coalesce(n1, 0) > 9990"
done

echo "numeric: aggregation"
for t in nm1 nm2; do
	runagg $t "numeric qual under count/sum/min/max" "select count(*), sum(k), min(k8), max(i) from %T where n1 > 100 and n2 < 0"
	runagg $t "numeric qual with arithmetic" "select count(*), sum(i) from %T where n1 * n2 > 1000 or n3 > 100000"
	runagg $t "count(numeric) reads only the NULLs" "select count(n1), count(n2), count(n3), count(n4), count(*) from %T where i % 2 = 0"
	runagg $t "numeric qual with Infinity/NaN rows (rechecked)" "select count(*), sum(i), max(k) from %T where n4 > 1000"
	runagg $t "numeric qual and FILTER" "select count(*) filter (where n1 > 0), sum(k) filter (where n2 < 0) from %T where n3 is not null"
	runagg $t "numeric -> int cast as an aggregate argument" "select sum(n1::int4), max(n2::int8), min((n1 * 2)::int4) from %T where abs(n1) < 1000 and abs(n2) < 1000000"
	runagg $t "GROUP BY an int key with a numeric qual" "select k % 7 as g, count(*), sum(i), max(k8) from %T where n1 > 0 and n3 < 20 group by 1 order by 1"
	runagg $t "GROUP BY with a qual over rechecked numerics" "select k % 5 as g, count(*), sum(i) from %T where n3 > 0 or n4 < 10 group by 1 order by 1"
	runagg_any $t "sum/avg/min/max of numeric stay on the CPU" "select sum(n1), avg(n2), min(n3), max(n1) from %T"
	runagg_any $t "GROUP BY a numeric key stays on the CPU" "select n1 % 3, count(*) from %T group by n1 % 3 order by 1"
done

echo "numeric: joins"
sql "create table nm_dim as select (i % 100)::int4 as k, (i * 1.5)::numeric(10,2) as w, 'd' || i as nm from generate_series(1, 100) i"
sql "analyze nm_dim"
for t in nm1; do
	ANYPLAN=0
	runjoin $t "a numeric column passes through a join" "select t.i, t.n1, t.n4, d.nm from %T t join nm_dim d on t.k = d.k where t.i < 2000"
	runjoin $t "numeric column of the inner side passes through" "select t.i, d.w, d.nm from %T t join nm_dim d on t.k = d.k where t.i < 2000"
	ANYPLAN=1
	runjoin $t "a numeric qual keeps the outer scan on the CPU" "select t.i, d.nm from %T t join nm_dim d on t.k = d.k where t.n1 > 5000 and t.i < 50000"
	runjoin $t "numeric comparison as a join qual stays on the CPU" "select t.i, d.nm from %T t join nm_dim d on t.k = d.k and t.n1 < d.w where t.i < 20000"
	runjoin $t "numeric join key stays on the CPU" "select t.i, d.nm from %T t join nm_dim d on t.n1 = d.w where t.i < 50000"
done
ANYPLAN=0

echo "numeric: errors and the backend's re-evaluation"
for t in nm1; do
	g=$($PSQL -d scantest -c "$GPU select n2::int2 from $t where n2 is not null" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select n2::int2 from $t where n2 is not null" 2>&1 | head -1)
	echo "$g" | grep -q "smallint out of range" && [ "$g" = "$c" ]; check "$t: numeric -> int2 overflow is reported alike ($g)" $?
	g=$($PSQL -d scantest -c "$GPU select n4::int8 from $t where n4 is not null and i % 12 = 3" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select n4::int8 from $t where n4 is not null and i % 12 = 3" 2>&1 | head -1)
	[ "$g" = "$c" ] && echo "$g" | grep -q "ERROR"; check "$t: NaN -> int8 is reported alike ($g)" $?
	g=$($PSQL -d scantest -c "$GPU select n1::int8 from $t where n1 is not null and n1 > 0 and i % 2 = 0" 2>&1 | head -1)
	[ -n "$g" ] && ! echo "$g" | grep -q ERROR; check "$t: numeric -> int8 of fitting values raises no error" $?
done
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from nm1 where n3 > 0" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -ge 1 ]; check "numerics beyond 64 bits are re-evaluated by the backend (debug message seen $msgs times)" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from nm1 where n1 > 5 and n2 < 100 and n1 * 2 > n2" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -eq 0 ]; check "numerics that fit are never handed back to the backend" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from nm1 where n4 is null or n4 is not null" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -eq 0 ]; check "IS NULL on numerics reads no value, so nothing is rechecked" $?
