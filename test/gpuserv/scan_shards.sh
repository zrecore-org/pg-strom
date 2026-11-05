#!/bin/sh
# scan_test.sh as SHARDS parallel parts, each on a cluster and port of its own, merged into one report.
#
#   SHARDS=2 sh test/gpuserv/locked_test.sh "$PWD" scan_shards.sh      # default: 1 (the same as scan_test.sh)
#
# NOT FOR AN APPLE GPU: with 3 parts on the M1 macOS killed a Metal command buffer ("Impacting Interactivity", seen as
# VK_ERROR_DEVICE_LOST) because the services together hogged the GPU. Use it where the GPU is not shared with a display
# (a Linux GPU or lavapipe, CI) and check the first runs for that error.
#
# The parts split the work (scan_test.sh: SHARD=<i>/<n>); every check runs in exactly one of them. They share the GPU, so
# a part can be slower than it would be alone; the point is the total. Each part writes its own timing summary
# (printed after its log). Needs the module and service installed (locked_test.sh does that). The exit status is
# non-zero if any part failed.
HERE=$(cd "$(dirname "$0")" && pwd)
SHARDS=${SHARDS:-1}
ROOT=${ROOT:-/tmp/pgstrom_scan_shards}
PORT=${PORT:-54334}
mkdir -p "$ROOT" || exit 2

i=1
pids=""
while [ "$i" -le "$SHARDS" ]; do
	( ROOT="$ROOT/s$i" PORT=$((PORT + i - 1)) SHARD="$i/$SHARDS" sh "$HERE/scan_test.sh" 2>&1 \
		| perl "$HERE/timeline.pl" "$ROOT/shard$i.times" > "$ROOT/shard$i.log" ) &
	pids="$pids $!"
	i=$((i + 1))
done
wait $pids

# the verdict comes from each part's last line (a pipeline loses the exit status of scan_test.sh)
rc=0
i=1
while [ "$i" -le "$SHARDS" ]; do
	echo "==== shard $i/$SHARDS ===="
	cat "$ROOT/shard$i.log"
	echo "---- shard $i timing (the cluster directory is removed after a passing run, so the summary is kept here)"
	head -14 "$ROOT/shard$i.times"
	grep -q "SCAN TEST: ALL PASSED" "$ROOT/shard$i.log" || rc=1
	i=$((i + 1))
done
echo
ok=$(cat "$ROOT"/shard*.log | grep -c '^  ok')
failed=$(cat "$ROOT"/shard*.log | grep -c '^  FAIL')
echo "shards: $SHARDS, checks ok: $ok, FAIL lines: $failed"
if [ $rc -eq 0 ]; then echo "SCAN TEST: ALL PASSED"; else echo "SCAN TEST: FAILED (see the shard logs above)"; exit 1; fi
