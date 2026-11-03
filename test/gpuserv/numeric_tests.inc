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
	run $t "trailing zeros and scale of zero" "select i, n4, n4 + 0.1, n4 * 1.00 from %T where i % 12 = 6 or i % 12 = 8"
	run $t "result scale: max of the two for sums, sum for products" "select n1 + n3, n1 - n2, n1 * n3, n3 * n3 from %T where i % 31 = 0"
	echo "  -- casts"
	run $t "numeric -> int4, rounding half away from zero" "select i, n1, n1::int4, (-n1)::int4 from %T where n1 between -100 and 100 and i % 3 = 0"
	run $t "numeric -> int8 and int2" "select i, n2::int8, n1::int2 from %T where n2 between -30000 and 30000 and n1 between -300 and 300"
	run $t "numeric -> int in a qual" "select i, n1 from %T where n1::int4 = 7 or n2::int8 > 499990"
	run $t "numeric -> int4 of an expression" "select i, (n1 * 10 + n2)::int4 from %T where abs(n2) < 1000 and abs(n1) < 100"
	run $t "numeric(18,6) -> int8 with half values" "select i, (n2 * 0.5)::int8, (n2 * 5)::int4 from %T where i % 1000 = 3"
	echo "  -- division, remainder, rounding (A6 part 2)"
	run $t "numeric division (select_div_scale result scale)" "select i, n1 / n2 from %T where n2 <> 0 and i % 7 = 0"
	run $t "division by constants and of constants" "select i, n1 / 3, n2 / 7.5, 100 / n1, 1 / n2 from %T where n1 <> 0 and n2 <> 0 and i % 11 = 0"
	run $t "division of columns of different scales" "select i, n2 / n1, n1 / n3, n3 / n2 from %T where n1 <> 0 and n3 <> 0 and n2 <> 0 and i % 13 = 0"
	run $t "division in a qual" "select i from %T where n1 <> 0 and n2 / n1 > 1000 or n2 <> 0 and n1 / n2 > 0.001"
	run $t "numeric remainder" "select i, n1 % 7, n2 % n1, mod(n1, 3), n1 % 0.3 from %T where n1 <> 0 and i % 7 = 0"
	run $t "div()" "select i, div(n2, n1), div(n1, 3), div(n2, 0.07) from %T where n1 <> 0 and i % 9 = 0"
	run $t "round() and trunc() with a scale" "select i, round(n1, 1), trunc(n2, 3), round(n2, 8), trunc(n1, 5) from %T where i % 7 = 0"
	run $t "round() and trunc() with a negative scale" "select i, round(n1, -1), trunc(n2, -2), round(n2, -9), round(n1, -30) from %T where i % 7 = 0"
	run $t "round(), trunc() without a scale" "select i, round(n1), trunc(n2), round(n3), trunc(n1) from %T where i % 7 = 0"
	run $t "ceil, floor, sign" "select i, ceil(n1), floor(n2), ceiling(n1), sign(n1), sign(n2), floor(n1), ceil(n2) from %T where i % 7 = 0"
	run $t "round() of a quotient" "select i, round(n1 / n2, 4), round(n2 / n1, 2) from %T where n1 <> 0 and n2 <> 0 and i % 7 = 0"
	run $t "numeric typmod coercion (rounding)" "select i, n2::numeric(10,2), n2::numeric(12,3), n1::numeric(8,0), n3::numeric(30,1) from %T where abs(n2) < 100000 and abs(n1) < 5000 and i % 5 = 0"
	run $t "numeric typmod coercion, wider scale" "select i, n1::numeric(14,6), n1::numeric(20,10), (n1 * 3)::numeric(18,2) from %T where abs(n1) < 1000 and i % 5 = 0"
	run $t "ceil/sign/round as a qual" "select i, n1 from %T where sign(n1) < 0 and round(n2, 0) = 100 or ceil(n1) = 5 or floor(n2) = -3"
	run_any $t "division by zero is the backend's error or result alike" "select i, n1 / n2 from %T where i % 7 = 0 and n2 is not null and i < 5000 and n2 <> 0"
	run_any $t "n4: division, round, ceil of long fractions, NaN and Infinity" "select i, n4 / 7, n4 % 3, round(n4, 2), ceil(n4), sign(n4), trunc(n4, 1), n4 / n4 from %T where i % 12 in (1, 2, 4, 6, 8, 10, 11) and n4 <> 0"
	run_any $t "numeric to float" "select i, n1::float4, n2::float8 from %T where i % 7 = 0"
	run_any $t "numeric in IN list and CASE" "select i, case when n1 > 0 then n2 else 0 end from %T where n1 in (1, 2, 3.5)"
	run_any $t "coalesce(numeric)" "select i, coalesce(n1, 0) from %T where coalesce(n1, 0) > 9990"
done


# ---- randomized operand pairs for division, remainder, round/trunc, ceil/floor/sign and the typmod coercion: 1..18
# significant digits, scales 0..13, either sign, a few zeros. (The same shapes as numfuzz.sql, which checks the
# algorithms against PostgreSQL's results offline.)
sql "create table nm_fz as
	with g as (select i, abs(hashint8(i * 2 + 1)::numeric * 1000003 + hashint8(i * 2 + 2)) as r1,
	                     abs(hashint8(i * 2 + 100000001)::numeric * 999983 + hashint8(i * 2 + 100000002)) as r2,
	                     (abs(hashint8(i * 3)) % 18 + 1)::int as d1, (abs(hashint8(i * 3 + 1)) % 18 + 1)::int as d2,
	                     (abs(hashint8(i * 5)) % 14)::int as s1, (abs(hashint8(i * 5 + 1)) % 14)::int as s2,
	                     abs(hashint8(i * 7)) % 4 as sg, abs(hashint8(i * 11)) % 20 as zz,
	                     (abs(hashint8(i * 13)) % 40 - 12)::int as n, (abs(hashint8(i * 17)) % 22 + 1)::int as p, (abs(hashint8(i * 19)) % 12)::int as s
	           from generate_series(1, $NMROWS) i),
	     t as (select i, n, p, s, s1, s2, sg,
	                  lpad(case when zz = 0 then '0' else (r1 % (10::numeric ^ d1))::numeric(40,0)::text end, s1 + 1, '0') as a,
	                  lpad(case when zz = 1 then '0' else (r2 % (10::numeric ^ d2))::numeric(40,0)::text end, s2 + 1, '0') as b from g)
	select i, n, p, s,
	       ((case when sg % 2 = 1 then '-' else '' end) || left(a, length(a) - s1) || case when s1 > 0 then '.' || right(a, s1) else '' end)::numeric as a,
	       ((case when sg / 2 = 1 then '-' else '' end) || left(b, length(b) - s2) || case when s2 > 0 then '.' || right(b, s2) else '' end)::numeric as b
	from t"
sql "analyze nm_fz"
sql "delete from nm_fz where b = 0 and i % 3 <> 0"             # a few zero divisors remain: the backend raises (or skips) them
run nm_fz "random pairs: a / b" "select i, a / b from %T where b <> 0"
run nm_fz "random pairs: a % b, div(a, b)" "select i, a % b, div(a, b), mod(b, a) from %T where b <> 0 and a <> 0"
run nm_fz "random pairs: round and trunc with random scales" "select i, round(a, n), trunc(b, n), round(a, n % 7), trunc(a, s) from %T where i > 0"
run nm_fz "random pairs: ceil, floor, sign" "select i, ceil(a), floor(b), sign(a), ceiling(b), sign(b) from %T where i > 0"
run nm_fz "random pairs: numeric(p,s) coercion" "select i, a::numeric(30,5), b::numeric(20,0), a::numeric(18,9), b::numeric(12,3) from %T where abs(a) < 1000000 and abs(b) < 1000000"
run nm_fz "random pairs: quotient in a qual" "select i from %T where b <> 0 and a / b > 1.5 and a / b < 100000"
runagg nm_fz "random pairs: count over a quotient qual" "select count(*), sum(i) from %T where b <> 0 and a / b > 3"
g=$($PSQL -d scantest -c "$GPU select count(*) from nm_fz where b = 0 and a / b > 1" 2>&1 | head -1)
c=$($PSQL -d scantest -c "$CPU select count(*) from nm_fz where b = 0 and a / b > 1" 2>&1 | head -1)
echo "$g" | grep -q "division by zero" && [ "$g" = "$c" ]; check "division by zero is reported alike ($g)" $?
g=$($PSQL -d scantest -c "$GPU select n2::numeric(5,2) from nm1 where n2 is not null and abs(n2) > 1000" 2>&1 | head -1)
c=$($PSQL -d scantest -c "$CPU select n2::numeric(5,2) from nm1 where n2 is not null and abs(n2) > 1000" 2>&1 | head -1)
echo "$g" | grep -q "numeric field overflow" && [ "$g" = "$c" ]; check "numeric field overflow is reported alike ($g)" $?

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
	echo "  -- sum and avg of numerics of a known scale (A6 part 2)"
	runagg $t "sum and avg of numeric(10,2) and numeric(18,6)" "select sum(n1), avg(n1), sum(n2), avg(n2), count(n1), count(*) from %T"
	runagg $t "sum and avg of numeric(38,10): values beyond 64 bits go through the backend" "select sum(n3), avg(n3), count(n3) from %T"
	runagg $t "sum of products and sums (scale of the expression)" "select sum(n1 * n2), sum(n1 + n2), avg(n1 - n2), sum(n1 * 3), sum(n2 * 0.5), sum(abs(n1)) from %T"
	runagg $t "sum of a quotient (scale not known: CPU) and of round()" "select sum(round(n1, 1)), avg(trunc(n2, 3)), sum(ceil(n1)), sum(sign(n2)), sum(round(n2 / 7, 4)) from %T where n2 <> 0"
	runagg $t "sum with a numeric qual and FILTER" "select sum(n1) filter (where n2 > 0), avg(n2) filter (where n1 < 0), sum(n1) from %T where n3 is not null and n1 <> 0"
	runagg $t "sum over an empty input and over NULLs only" "select sum(n1), avg(n2), sum(n3) from %T where i < 0 or n1 is null and n2 is null"
	runagg $t "sum of int4 as numeric and numeric from int" "select sum(k::numeric), avg(k8::numeric), sum(n1 + k) from %T"
	runagg $t "sum and avg with an int GROUP BY" "select k % 7 as g, sum(n1), avg(n2), sum(n1 * n2), count(*) from %T group by 1 order by 1"
	runagg $t "GROUP BY an int key, numeric(38,10) values beyond 64 bits" "select k % 11 as g, sum(n3), avg(n3) from %T group by 1 order by 1"
	runagg_any $t "sum/avg of unconstrained numeric (the scale varies: CPU)" "select sum(n4), avg(n4) from %T where i % 12 in (1, 2, 4, 6, 8, 10, 11)"
	runagg_any $t "sum of unconstrained numeric with NaN and Infinity (CPU)" "select sum(n4), avg(n4), count(n4) from %T"
done


echo "numeric: TPC-H style aggregates"
sql "create table nm_li as select i as l_orderkey, (i % 7 + 1)::int as l_linenumber, (i % 5)::int as grp, (i % 3)::int as grp2,
	((i * 37) % 50 + 1)::numeric(15,2) as l_quantity,
	(((i::int8 * 7919) % 10000000) / 100.0)::numeric(15,2) as l_extendedprice,
	((i % 11) / 100.0)::numeric(15,2) as l_discount,
	((i % 9) / 100.0)::numeric(15,2) as l_tax,
	(date '1992-01-01' + (i % 2500)) as l_shipdate,
	(case when i % 97 = 0 then null else ((i % 13) - 6) / 10.0 end)::numeric(6,1) as bal,
	((i % 20)::numeric / 4)::numeric(6,2) as g2, ((i % 7)::numeric * 100)::numeric(8,0) as g0
	from generate_series(1, $NMROWS) i"
sql "vacuum analyze nm_li"
runagg nm_li "Q6: sum(extendedprice * discount) with quals" "select sum(l_extendedprice * l_discount) as revenue from %T where l_shipdate >= date '1994-01-01' and l_shipdate < date '1995-01-01' and l_discount between 0.05 and 0.07 and l_quantity < 24"
runagg nm_li "Q1 aggregates (no GROUP BY)" "select sum(l_quantity), sum(l_extendedprice), sum(l_extendedprice * (1 - l_discount)), sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)), avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*) from %T where l_shipdate <= date '1998-09-02'"
runagg nm_li "Q1 aggregates GROUP BY integer keys" "select grp, grp2, sum(l_quantity), sum(l_extendedprice), sum(l_extendedprice * (1 - l_discount)), sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)), avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*) from %T where l_shipdate <= date '1998-09-02' group by 1, 2 order by 1, 2"
runagg nm_li "GROUP BY a numeric(6,2) key" "select g2, count(*), sum(l_extendedprice), avg(l_discount), sum(l_extendedprice * (1 - l_discount)) from %T group by 1 order by 1"
runagg nm_li "GROUP BY a numeric(8,0) key and an integer key" "select g0, grp, count(*), sum(l_quantity), avg(l_tax) from %T group by 1, 2 order by 1, 2"
runagg_any nm_li "GROUP BY a numeric key of unknown scale (unconstrained numeric: CPU)" "select bal2, count(*), sum(l_quantity) from (select l_quantity, bal::numeric as bal2 from %T) x group by 1 order by 1 nulls last"
runagg nm_li "GROUP BY a numeric key with NULLs (the NULL group)" "select bal, count(*), sum(l_quantity), avg(l_extendedprice) from %T group by 1 order by 1 nulls last"
runagg nm_li "GROUP BY a numeric expression of known scale" "select round(l_extendedprice, -4) as r, count(*), sum(l_quantity) from %T group by 1 order by 1"
runagg nm_li "GROUP BY l_quantity * l_discount (scale 4)" "select l_quantity * l_discount as qd, count(*), sum(l_tax) from %T group by 1 order by 1"
runagg nm_li "GROUP BY numeric and HAVING" "select g2, sum(l_extendedprice) as s from %T group by 1 having sum(l_extendedprice) > 1000 order by 1"
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

# ---- overflow detection of the 64-bit arithmetic (regression: a wrapped signed product checked afterwards with
# "r / a != b" is undefined behaviour that the Metal compiler folded away, so a numeric product beyond 64 bits
# came back as a wrapped, wrong value, e.g. 826563.6313334870 * 467842.677)
sql "create table nm_mul as select i as id, (826563.6313334870 + (i % 7) * 1000.5)::numeric as x, (467842.677 + (i % 11))::numeric as z,
	((i::int8 * 7919) % 3037000499 + 1) as a, ((i::int8 * 104729) % 3037000499 + 1) as b, (i % 5 = 0) as neg
	from generate_series(1, 30000) i"
sql "insert into nm_mul values (0, 826563.6313334870, 467842.677, 3037000499, 3037000499, false),
	(-1, 1000000.5, 1000000.5, -4294967296, 2147483648, false),
	(-2, 9.5, 9.5, 9223372036854775807, 1, false), (-3, 1, 1, -9223372036854775807, -1, false),
	(-4, 123456789012.345678, 98765.4321, 4611686018427387904, 1, false)"
sql "analyze nm_mul"
run nm_mul "numeric product of wide fractions (13 fractional digits, beyond 64 bits unscaled)" "select id, x * z from %T where id >= 0"
run nm_mul "numeric product, sum and difference mixed" "select id, x * z, x + z, x - z, (x * z) - x from %T where id between -4 and 3000"
run nm_mul "numeric product compared against a constant" "select id from %T where x * z > 386701741993.0 and x * z < 386701741994.0"
run nm_mul "int8 products at the 64-bit boundary (fit exactly)" "select id, a * b from %T where id between -4 and 0 and id <> -2 or id > 29990"
run nm_mul "int8 sum near INT64_MAX (fits)" "select id, a + b from %T where id <= 0 or id > 29990"
for q in "select a * b * b from nm_mul where id = 29999" "select a * 4000000000 from nm_mul where id = 0" \
	 "select a + 9223372036854775807 from nm_mul where id = 0" "select a - (-9223372036854775807) from nm_mul where id = 0" \
	 "select (-a) * 3037000500 - 3037000500 from nm_mul where id = 0"; do
	g=$($PSQL -d scantest -c "$GPU $q" 2>&1 | head -1); c=$($PSQL -d scantest -c "$CPU $q" 2>&1 | head -1)
	echo "$g" | grep -q "bigint out of range" && echo "$c" | grep -q "bigint out of range"; check "int8 overflow is reported on the GPU and the CPU: $q" $?
done
