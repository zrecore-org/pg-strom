# Regression tests of bugs the differential fuzzer (fuzz.py) found. Sourced by scan_test.sh before "errors"
# (it provides run, run_any, runagg, runjoin, q, check, sql, $PSQL, $GPU, $CPU, $fail and the tables s1 s2 s3, sd).
echo "fuzzer regressions"
for t in s1 s2 s3; do
	# unary minus of a computed operand: the constant 0 overwrote the operand's register
	run $t "unary minus of a computed value" "select -(a + b), -(c % 1000), -(a * 2) from %T where a > 9000 and b is not null"
	run $t "unary minus of a computed value in a qual" "select i from %T where -(a + 1) < -9990"
done
