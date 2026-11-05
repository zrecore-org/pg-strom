#!/bin/sh
# The usual verification of a tree, stage after stage, each under the shared test lock (locked_test.sh), logs in
# test-data/logs/<stamp>/ (git-ignored, on the external volume). Prints one line per stage and exits non-zero if one failed.
#
#   sh test/gpuserv/verify_all.sh                         installcheck, then the full scan_test.sh
#   STAGES="installcheck scan fuzz float8 postgis" sh test/gpuserv/verify_all.sh
#   FUZZ_SEED=11 FUZZ_COUNT=1500 STAGES=fuzz sh test/gpuserv/verify_all.sh
#
# Stages: installcheck (vulkan_installcheck.sh: expect "All 23 tests passed"), scan (scan_test.sh), scanshards (the same in
# SHARDS=<n> parallel parts, scan_shards.sh; not on an Apple GPU, see its header), fuzz (fuzz_test.sh),
# float8 / postgis / arrow / gpucache (the dev scripts, with PGSTROM_GPUSERV_SPECIALIZE=check), regen (vulkan_regen_expected.sh).
# TIMEOUT=<seconds> caps each stage (default 4200); a stage that is cut off has its cluster and service stopped.
# Every stage writes <log>.times: seconds per section and the slowest checks (timeline.pl). QUERY_TIMEOUT=<seconds> caps each psql call of
# the tests (qtimeout.sh, default 900); after one timeout the remaining queries of that test fail at once.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
STAGES=${STAGES:-"installcheck scan"}
TIMEOUT=${TIMEOUT:-4200}
STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR=${LOGDIR:-$REPO/test-data/logs/$STAMP}
mkdir -p "$LOGDIR"
cd "$REPO" || exit 2
rc=0
n=0
for st in $STAGES; do
	n=$((n+1))
	log=$LOGDIR/$n-$st.log
	case $st in
		installcheck) cmd="test/vulkan_installcheck.sh" ;;
		scan)         cmd="scan_test.sh" ;;
		scanshards)   cmd="scan_shards.sh" ;;
		fuzz)         cmd="fuzz_test.sh" ;;
		float8)       cmd="float8_dev.sh" ;;
		postgis)      cmd="postgis_dev.sh" ;;
		arrow)        cmd="arrow_dev.sh" ;;
		gpucache)     cmd="gpucache_dev.sh" ;;
		regen)        cmd="test/vulkan_regen_expected.sh" ;;
		*) echo "unknown stage $st"; exit 2 ;;
	esac
	t_start=$(date +%s)
	: > "$LOGDIR/.stamp.$n"			# crash reports newer than this belong to the stage
	# the output goes through timeline.pl (per-section and per-check seconds in <log>.times); the exit status through <log>.rc
	( PGSTROM_GPUSERV_SPECIALIZE=${PGSTROM_GPUSERV_SPECIALIZE:-check} perl -e 'alarm shift; exec @ARGV' "$TIMEOUT" \
		sh test/gpuserv/locked_test.sh "$PWD" $cmd 2>&1; echo $? > "$log.rc" ) | perl test/gpuserv/timeline.pl "$log.times" > "$log"
	e=$(cat "$log.rc" 2>/dev/null); e=${e:-1}
	secs=$(( $(date +%s) - t_start ))
	if [ "$e" -ne 0 ]; then
		# keep the evidence: the next run replaces the kept cluster and its log, and a crash report is easy to lose
		diag=$LOGDIR/$n-$st.diag
		mkdir -p "$diag"
		cname=$(basename "${cmd%% *}" .sh)
		for f in "$REPO"/test-data/$cname/log*; do [ -f "$f" ] && cp "$f" "$diag/"; done
		if [ -d "$HOME/Library/Logs/DiagnosticReports" ]; then
			find "$HOME/Library/Logs/DiagnosticReports" -maxdepth 1 \( -name 'postgres-*.ips' -o -name 'pgstrom-gpuserv-*.ips' \) \
				-newer "$LOGDIR/.stamp.$n" -exec cp {} "$diag/" \; 2>/dev/null
		fi
		echo "$st: failed, cluster log and new crash reports copied to $diag" >> "$log"
	fi
	if [ "$e" -eq 142 ]; then
		# the stage was cut off: do not leave its cluster and service running (they hold the port and the GPU)
		echo "$st: timed out after ${TIMEOUT}s, stopping its cluster" >> "$log"
		ps -ax -o command | sed -n "s|^.*postgres -D \($REPO/test-data/[^ ]*\).*\$|\1|p" | sort -u | while read -r d; do
			${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}/pg_ctl -D "$d" -m immediate stop >/dev/null 2>&1
		done
		pkill -f "pgstrom-gpuserv --serve.*$REPO/test-data" 2>/dev/null
	fi
	ok=$(grep -c '^  ok' "$log"); fl=$(grep -c '^  FAIL' "$log")
	summary=$(grep -E 'ALL PASSED|tests passed|FAILED|tests failed|regenerated|FUZZ TEST|failing:' "$log" | tail -2 | tr '\n' ' ')
	echo "$st: exit=$e ok=$ok fail=$fl ${secs}s $summary(log: $log, times: $log.times)"
	[ $e -eq 0 ] || rc=1
done
git checkout -- src/gpuserv/gcache_check 2>/dev/null		# `make clean` deletes this tracked binary
exit $rc
