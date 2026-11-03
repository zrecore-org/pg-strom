#!/bin/sh
# Runs only the numeric section (numeric_tests.inc) of scan_test.sh: the prelude (cluster, data, helpers) is cut
# out of scan_test.sh, so the helpers cannot drift. Same environment as scan_test.sh.
D=${0%/*}
GEN=${TMPDIR:-/tmp}/numeric_test_gen.$$.sh
awk '
	/^for t in s1 s2 s3; do/ { n++; if (n == 1) skipping = 1 }
	n == 0 { print; next }
	/^runagg\(\) \{/ { grab = 1 }
	/^for t in s1 s2 s3; do/ { if (n == 2) grab = 0 }
	/^runjoin\(\) \{/ { grabj = 1 }
	grab { print }
	grabj { print; if ($0 == "}") grabj = 0 }
' "$D/scan_test.sh" > "$GEN"
cat >> "$GEN" <<'TAIL'
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
JOINPAT="Custom Scan (GpuJoin)"
ANYPLAN=0
TAIL
echo ". \"$D/numeric_tests.inc\"" >> "$GEN"
cat >> "$GEN" <<'TAIL'
[ -z "$KEEP" ] && stop
echo
if [ $fail -eq 0 ]; then echo "NUMERIC TEST: ALL PASSED"; else echo "NUMERIC TEST: $fail FAILED"; exit 1; fi
TAIL
ROOT=${ROOT:-/tmp/pgstrom_numeric_test} PORT=${PORT:-54395} sh "$GEN"
rc=$?
rm -f "$GEN"
exit $rc
