#!/bin/sh
# The usual verification of a tree, stage after stage, each under the shared test lock (locked_test.sh), logs in
# test-data/logs/<stamp>/ (git-ignored, on the external volume). Prints one line per stage and exits non-zero if one failed.
#
#   sh test/gpuserv/verify_all.sh                         installcheck, then the full scan_test.sh
#   STAGES="installcheck scan fuzz float8 postgis" sh test/gpuserv/verify_all.sh
#   FUZZ_SEED=11 FUZZ_COUNT=1500 STAGES=fuzz sh test/gpuserv/verify_all.sh
#
# Stages: installcheck (vulkan_installcheck.sh: expect "All 23 tests passed"), scan (scan_test.sh), fuzz (fuzz_test.sh),
# float8 / postgis / arrow / gpucache (the dev scripts, with PGSTROM_GPUSERV_SPECIALIZE=check), regen (vulkan_regen_expected.sh).
# TIMEOUT=<seconds> caps each stage (default 4200).
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
STAGES=${STAGES:-"installcheck scan"}
TIMEOUT=${TIMEOUT:-4200}
STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR=${LOGDIR:-$REPO/test-data/logs/$STAMP}
mkdir -p "$LOGDIR"
cd "$REPO" || exit 2
rc=0
for st in $STAGES; do
	log=$LOGDIR/$st.log
	case $st in
		installcheck) cmd="test/vulkan_installcheck.sh" ;;
		scan)         cmd="scan_test.sh" ;;
		fuzz)         cmd="fuzz_test.sh" ;;
		float8)       cmd="float8_dev.sh" ;;
		postgis)      cmd="postgis_dev.sh" ;;
		arrow)        cmd="arrow_dev.sh" ;;
		gpucache)     cmd="gpucache_dev.sh" ;;
		regen)        cmd="test/vulkan_regen_expected.sh" ;;
		*) echo "unknown stage $st"; exit 2 ;;
	esac
	PGSTROM_GPUSERV_SPECIALIZE=${PGSTROM_GPUSERV_SPECIALIZE:-check} perl -e 'alarm shift; exec @ARGV' "$TIMEOUT" \
		sh test/gpuserv/locked_test.sh "$PWD" $cmd > "$log" 2>&1
	e=$?
	ok=$(grep -c '^  ok' "$log"); fl=$(grep -c '^  FAIL' "$log")
	summary=$(grep -E 'ALL PASSED|tests passed|FAILED|tests failed|regenerated|FUZZ TEST|failing:' "$log" | tail -2 | tr '\n' ' ')
	echo "$st: exit=$e ok=$ok fail=$fl $summary(log: $log)"
	[ $e -eq 0 ] || rc=1
done
git checkout -- src/gpuserv/gcache_check 2>/dev/null		# `make clean` deletes this tracked binary
exit $rc
