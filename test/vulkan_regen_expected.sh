#!/bin/sh
# Regenerate the alternate expected files of the Vulkan build: <major>/expected/<test>_9.out
#
#   sh test/gpuserv/locked_test.sh <tree> test/vulkan_regen_expected.sh     # under the shared test lock
#   sh test/vulkan_regen_expected.sh [test ...]                             # default: every test of vulkan_schedule
#
# What it does (the environment is that of test/vulkan_installcheck.sh: PGBIN, ROOT, PORT, ...):
#   1. runs vulkan_schedule on the GPU                      -> <major>/results
#   2. runs it again with pg_strom.enabled forced off       -> <major>/oracle/results   (the CPU oracle)
#   3. a test whose output already matches one of its expected files is left alone. Otherwise, if its output
#      with the EXPLAIN blocks removed equals the CPU oracle's (so the result rows are those PostgreSQL
#      computes), the output becomes <test>_9.out. If the rows differ from the oracle nothing is written
#      and the test is reported: that is a bug, or a test that cannot work on this build (see
#      vulkan_known_failures.txt).
# _9 is reserved for the Vulkan build because pg_regress accepts <test>_0.out .. <test>_9.out and some tests
# already own _1 (upstream's alternates). Only the major version of PGBIN is regenerated; the other versions
# still need a host with that server (their expected files are not touched).
# SKIP_RUN=1 reuses the results of the previous runs (to adjust the comparison only).
HERE=$(cd "$(dirname "$0")" && pwd)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
VER=$($PGBIN/pg_config --version | sed 's/[^0-9]*\([0-9][0-9]*\).*/\1/')
DIR=$HERE/$VER
rc=0

if [ "${SKIP_RUN:-0}" != 1 ]; then
	sh "$HERE/vulkan_installcheck.sh" > "$DIR/regen_gpu.log" 2>&1
	TARGET=vulkan-oracle sh "$HERE/vulkan_installcheck.sh" > "$DIR/regen_oracle.log" 2>&1
fi

TESTS=$*
[ -n "$TESTS" ] || TESTS=$(sed -n 's/^test: *//p' "$HERE/vulkan_schedule" | tr ' ' '\n' | sed '/^$/d')
# pgstrom_guc prints the setting of pg_strom.enabled itself, which the oracle run changes on purpose: its
# output is taken as it is (it contains no query results).
NO_ORACLE=" pgstrom_guc "
strip_plans() {
	# EXPLAIN result blocks ("QUERY PLAN" .. "(N rows)") are plan text, not rows; the SET/RESET pg_strom.enabled
	# statements are echoed with the value the oracle run replaced
	# In tuples-only mode (\t on) an EXPLAIN has no QUERY PLAN header: the plan is the output after the
	# echoed statement (which ends with a ';') up to the next blank line (gpu_sort).
	awk '/^\\t on/ { tonly = 1 }
	     /^\\t off/ { tonly = 0 }
	     tonly && ex == 0 && /^EXPLAIN/ { ex = 1 }
	     ex == 1 { print; if ($0 ~ /;[ ]*$/) ex = 2; next }
	     ex == 2 { if ($0 ~ /^$/) { ex = 0; print "<plan>"; print "" }; next }
	     /^(SET|RESET)[ ]+pg_strom\.enabled/ { next }
	     /^ +QUERY PLAN *$/ { skip = 1; print "<plan>"; next }
	     skip && /^\([0-9]+ rows?\)$/ { skip = 0; next }
	     skip { next }
	     { print }' "$1"
}
for t in $TESTS; do
	res=$DIR/results/$t.out
	ora=$DIR/oracle/results/$t.out
	case "$NO_ORACLE" in *" $t "*) ora=$res;; esac
	if [ ! -f "$res" ] || [ ! -f "$ora" ]; then echo "no result   $t"; rc=1; continue; fi
	same=0
	for e in "$DIR/expected/$t.out" "$DIR"/expected/${t}_[0-9].out; do
		[ -f "$e" ] && cmp -s "$e" "$res" && same=1
	done
	if [ $same = 1 ]; then echo "passes      $t"; continue; fi
	if [ "$(strip_plans "$res" | cksum)" = "$(strip_plans "$ora" | cksum)" ]; then
		cp "$res" "$DIR/expected/${t}_9.out"
		echo "regenerated $t (rows equal the CPU oracle)"
	else
		echo "ROW DIFF    $t (GPU vs CPU oracle: compare $res and $ora without EXPLAIN blocks)"
		rc=1
	fi
done
exit $rc
