#!/bin/sh
# kernel_survey.sh <build dir> <pgstrom-gpuserv> [kernel names...]
#
# Compiles each kernel on its own (pgstrom-gpuserv --prewarm) and prints one line per kernel: its SPIR-V size, "ok" with the
# seconds, "FAIL" (the driver rejected the pipeline, the line says why) or "TIMEOUT". Used on the NVIDIA T4 in CI, where a rejected
# kernel is hidden by the CPU fallback (A17); the default list is the families that failed there plus passing controls.
#   SURVEY_TIMEOUT=<s>   per kernel (default 300)
BUILD=${1:?usage: kernel_survey.sh <build dir> <pgstrom-gpuserv> [kernel...]}
BIN=${2:?usage: kernel_survey.sh <build dir> <pgstrom-gpuserv> [kernel...]}
shift 2
[ $# -gt 0 ] || set -- gpuagg8_4_lite gagg16_4_lite jagg16_4_lite jgagg16_4_lite jgagg16_4_nogeom jgagg16_4 ajgagg16_4_lite ajgagg32_4_lite \
	gagg16_4_nogeom gagg16_12_nogeom gpuagg8_12_nogeom agagg16_4_lite agtab16_lite
geom() {
	case $1 in
	gpuscan*|gpuagg*|gagg*|gpuarrow*) echo "5 16";;
	aagg*|agagg*) echo "5 20";;
	*) echo "6 20";;
	esac
}
# a per-kernel limit without timeout(1) (macOS): the compile runs in the background and is killed when the time is up
run_limited() {
	"$@" > "$OUT" 2>&1 &
	pid=$!
	n=0
	while kill -0 $pid 2>/dev/null && [ $n -lt "${SURVEY_TIMEOUT:-300}" ]; do sleep 1; n=$((n+1)); done
	if kill -0 $pid 2>/dev/null; then kill $pid 2>/dev/null; wait $pid 2>/dev/null; return 124; fi
	wait $pid
}
OUT=$(mktemp)
fail=0
for k in "$@"; do
	f=$BUILD/$k.spv
	[ -f "$f" ] || { echo "$k: no such kernel in $BUILD"; continue; }
	size=$(wc -c < "$f" | tr -d ' ')
	set -- $(geom "$k")
	nb=$1; pc=$2
	t0=$(date +%s)
	run_limited "$BIN" --prewarm "$nb" "$pc" "$f"; rc=$?
	secs=$(( $(date +%s) - t0 ))
	if [ $rc = 124 ]; then echo "$k: $size bytes of SPIR-V: TIMEOUT after $secs s"; fail=1
	elif grep -q -E "failed|rejected|VK_ERROR" "$OUT"; then echo "$k: $size bytes of SPIR-V: FAIL after $secs s: $(grep -E 'failed|rejected|VK_ERROR' "$OUT" | head -1)"; fail=1
	else echo "$k: $size bytes of SPIR-V: ok in $secs s"; fi
done
rm -f "$OUT"
exit $fail
