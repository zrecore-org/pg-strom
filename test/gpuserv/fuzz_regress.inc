# Regression tests of bugs the differential fuzzer (fuzz.py) found. Sourced by scan_test.sh before "errors"
# (it provides run, run_any, runagg, runjoin, q, check, sql, $PSQL, $GPU, $CPU, $fail and the tables s1 s2 s3, sd).
echo "fuzzer regressions"
for t in s1 s2 s3; do
	# unary minus of a computed operand: the constant 0 overwrote the operand's register
	run $t "unary minus of a computed value" "select -(a + b), -(c % 1000), -(a * 2) from %T where a > 9000 and b is not null"
	run $t "unary minus of a computed value in a qual" "select i from %T where -(a + 1) < -9990"
done
ANYPLAN=1
for t in s1 s2 s3; do
	# the hash of a join key is computed over its stored bytes: a key of another width never found its partner
	runjoin $t "int2 key against an int4 key" "select t.i, d.grp from %T t join %D d on t.a = d.k where t.i < 20000"
	runjoin $t "int8 key against an int4 key" "select t.i, d.grp from %T t join %D d on t.c % 20000 = d.k where t.i < 20000"
	runjoin $t "int4 expression cast to int2 against an int4 key (semi join)" "select t.i from %T t where (t.b % 1000)::int2 in (select k from %D)"
done
ANYPLAN=0
for t in s1 s2 s3; do
	# an aggregate over a constant (the constant is a kernel parameter and a projected value; setrefs turned the
	# parameter into a reference to custom_scan_tlist and the backend crashed evaluating it)
	runagg_any $t "aggregates of constants" "select sum(3), min(7), max(1 + 1), avg(5), count(3), sum(b) from %T where a > 0"
	runagg_any $t "constant aggregate argument with GROUP BY" "select a % 3 as k, sum(3), count(*) from %T group by 1 order by 1"
	# the computed projection must not run on rows a host-only qual rejects (the error would be PG-Strom's alone)
	run_any $t "computed projection that overflows, rows rejected by a host qual" "select a::int4 * 1000000 as x, i from %T where a > 0 and upper(f) = 'NEVER'"
done
for t in s1 s2 s3; do
	ANYPLAN=1
	runjoin $t "IS NULL of an inner string column inside a join qual" "select t.i, d.grp from %T t join %D d on t.b = d.k and (t.a > 100000 or d.txt is null or d.pay is null)"
	ANYPLAN=0
done
# char(n) LIKE: the value's trailing blanks are significant (bpcharlike does not strip them)
for t in ty1 ty2; do
	run $t "bpchar LIKE keeps the padding" "select i, bp from %T where bp like 'b1' or bp like '%1' or bp not like 'b2' or bp like 'b1      '"
	run $t "bpchar LIKE prefix on the padded value" "select i, bp from %T where bp like 'b1%' and bp not like '%1'"
done
# wide plans: more kernel variables than the VM has registers must stay on the CPU, not fail in the service
WCOLS=""; WSEL=""; WSUM=""; WQ=""
k=1; while [ $k -le 44 ]; do
	WCOLS="$WCOLS, ((i * $k + $k) % 21 - 10)::int4 as w$k"
	WSEL="$WSEL${WSEL:+, }w$k + $k"
	[ $k -le 14 ] && WSUM="$WSUM${WSUM:+, }sum(w$k + w$((k+1))) filter (where w$((k+2)) > 0)"
	k=$((k+1))
done
sql "create table wz as select i $WCOLS from generate_series(1, 20000) i"; sql "analyze wz"
run_any wz "44 computed columns" "select $WSEL from %T where w1 > 0"
run_any wz "14 aggregates with filters over many columns" "select $WSUM from %T where w20 < w21 and w22 <> w23"
run_any wz "aggregates with filters (the 32 registers case of the fuzzer)" "select avg(w31) filter (where (w21 > 0)), max(w34) filter (where (w22 > 0)), avg(w34 - w12) filter (where (w31 > 0)), avg(w5), max(w26 - w29) filter (where (w16 > 0)), min(w25 - w38) filter (where (w25 > 0)) from %T where w24 < w4 and w18 <> w44 and w15 < w27"
# 100 quals: the VM's AND takes at most 64 operands
Q100=""; k=1; while [ $k -le 100 ]; do Q100="$Q100${Q100:+ and }w1 <> $((k + 100))"; k=$((k+1)); done
run_any wz "100 quals" "select i from %T where $Q100 and w2 > 0"
# a deep expression: more temporaries than registers
DEEP="w1"; k=2; while [ $k -le 40 ]; do DEEP="($DEEP + w$k)"; k=$((k+1)); done
run_any wz "left-deep sum of 40 columns" "select $DEEP from %T where w1 > 5"
DEEPR="w40"; k=39; while [ $k -ge 1 ]; do DEEPR="(w$k - $DEEPR)"; k=$((k-1)); done
run_any wz "right-deep difference of 40 columns" "select i from %T where $DEEPR > 0"
for t in s1 s2 s3; do
	# float4 comparison with denormals (Metal flushed them to zero: 0 >= 1.4e-45 was true on the GPU)
	run $t "float4 >= a denormal constant" "select i, d from %T where d >= 1.4e-45::float4 and d < 1::float4"
	run $t "float4 = a denormal constant against a column" "select i from %T where d <= 1e-38::float4 and d > '-1e-38'::float4"
	run $t "float4 vs a column through arithmetic" "select i from %T where d * 0::float4 = 0::float4 and d <> -0.0::float4 and d < 0.5::float4"
done
for t in s1 s2 s3; do
	# float4 arithmetic whose result (or operand) is denormal: the backend evaluates the row (Metal flushes to zero)
	run $t "float4 product that is denormal" "select i, d * 1e-38::float4 as x from %T where i < 3000 and d < 1000::float4"
	run $t "float4 qual on a denormal quotient" "select i from %T where d / 1e38::float4 > 0::float4 and d / 1e38::float4 < 1e-37::float4 and i < 5000 and d < 1000::float4"
	run_any $t "float4 sum with a denormal operand" "select i, d + 1e-40::float4 from %T where i < 3000 and d < 1000::float4"
done
for t in s1 s2 s3; do
	# float4 -> integer casts round ties to even (Metal's round() is half away from zero)
	run $t "float4 to int2/int4/int8 casts at .5 ties" "select i, (d / 4::float4)::int4 as a, (d / 2::float4)::int2 as b, ((d - 70::float4) / 4::float4)::int8 as c from %T where d < 3000::float4"
	runagg_any $t "GROUP BY a float4 -> int2 cast" "select (d / 2::float4)::int2 as k, count(*), sum(c) from %T where d < 200::float4 group by 1 order by 1"
	# an aggregate argument that can fail is not evaluated for rows its FILTER rejects
	runagg_any $t "FILTER that rejects everything over an overflowing argument" "select count(a * 100000000 * b::int8::int4) filter (where a <> a), count(*) from %T"
done
