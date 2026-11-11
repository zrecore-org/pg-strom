#!/bin/sh
# Creates the Vulkan pipeline of every kernel (or those matching PIPE_PATTERN) one by one with pgstrom-gpuserv --prewarm, without
# the pipeline cache, and prints OK / FAIL with the SPIR-V size and the time, so that a kernel the device's shader compiler
# rejects (NVIDIA: "NVVM compilation failed", seen on the T4 as a CPU fallback of GpuScan) is named. No cluster, no lock needed,
# but it uses the GPU: run it alone.
#   GPUSERV_BIN=<pgstrom-gpuserv> sh test/gpuserv/pipeline_check.sh      PIPE_PATTERN='gpuscan*' PIPE_TIMEOUT=60 DEVICE=0
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
BUILD=${BUILD:-$(cd "${0%/*}/../../src/gpuserv" && pwd)/build}
PAT=${PIPE_PATTERN:-gpuscan* gpuagg* gagg* gtab* gpuarrow* aagg*}
TMO=${PIPE_TIMEOUT:-60}
export PGSTROM_GPUSERV_PIPELINE_CACHE=off
nfail=0
for p in $PAT; do
	for f in $(cd "$BUILD" && ls $p.spv 2>/dev/null); do
		k=${f%.spv}
		case $k in
			gpuscan*|gpuagg*|gagg*|gtab*|gpuarrow*) g="5 16";;
			aagg*|agagg*|agtab*) g="5 20";;
			*) g="6 20";;
		esac
		size=$(wc -c < "$BUILD/$f" | tr -d ' ')
		t0=$(date +%s)
		out=$(timeout "$TMO" "$GPUSERV_BIN" --prewarm $g "$BUILD/$f" --device "${DEVICE:-0}" 2>&1); rc=$?
		t=$(( $(date +%s) - t0 ))
		if [ $rc -eq 0 ]; then echo "  OK    $k ($size bytes, ${t}s)"
		else nfail=$((nfail+1)); echo "  FAIL  $k ($size bytes, ${t}s, rc=$rc): $(echo "$out" | tr '\n' ' ' | cut -c1-200)"; fi
	done
done
echo "pipeline check: $nfail failed"
[ $nfail -eq 0 ]
