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
	runjoin $t "int4 expression cast to int2 against an int4 key (semi join)" "select t.i from %T t where t.b::int2 in (select k from %D)"
done
ANYPLAN=0
