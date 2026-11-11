#!/bin/sh
# Which optional instruction group makes a device's shader compiler reject the scan kernel (T4: gpuscan*_nogeom, "NVVM compilation
# failed: 3")? Compiles variants of kernels_vm/scan.slang with slangc and creates each pipeline with pgstrom-gpuserv --prewarm.
#   SLANGC=... GPUSERV_BIN=... sh test/gpuserv/nvidia_bisect.sh      BISECT_TIMEOUT=60 DEVICE=0
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN}
SLANGC=${SLANGC:?set SLANGC}
SRC=$(cd "${0%/*}/../../src/gpuserv" && pwd)
OUT=${BISECT_OUT:-${TMPDIR:-/tmp}/nvbisect}; mkdir -p "$OUT"
TMO=${BISECT_TIMEOUT:-60}
export PGSTROM_GPUSERV_PIPELINE_CACHE=off
try() { # name  slang-file  entry  nbind pcsize  defs...
	name=$1; file=$2; entry=$3; nb=$4; pc=$5; shift 5
	t0=$(date +%s)
	( cd "$SRC" && "$SLANGC" "kernels_vm/$file" -I kernels_vm "$@" -target spirv -entry "$entry" -stage compute -o "$OUT/$name.spv" ) > "$OUT/$name.slangc.log" 2>&1 || { echo "  SLANGC-FAIL $name: $(head -c 300 "$OUT/$name.slangc.log")"; return; }
	t1=$(date +%s)
	size=$(wc -c < "$OUT/$name.spv" | tr -d ' ')
	out=$(timeout "$TMO" "$GPUSERV_BIN" --prewarm "$nb" "$pc" "$OUT/$name.spv" --device "${DEVICE:-0}" 2>&1); rc=$?
	t2=$(date +%s)
	if [ $rc -eq 0 ]; then echo "  OK    $name ($size bytes, slangc $((t1-t0))s, pipeline $((t2-t1))s)"
	else echo "  FAIL  $name ($size bytes, slangc $((t1-t0))s, pipeline $((t2-t1))s, rc=$rc): $(echo "$out" | tr '\n' ' ' | cut -c1-150)"; fi
}
NOF8="-DVM_NO_F8"; NONUM="-DVM_NO_NUM"; NOJ="-DVM_NO_JSON"; NOG="-DVM_NO_GEOM"
try s8_lite       scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NONUM $NOJ $NOF8 $NOG
try s8_num        scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NOJ $NOF8 $NOG
try s8_json       scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NONUM $NOF8 $NOG
try s8_f8         scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NONUM $NOJ $NOG
try s8_numjson    scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NOF8 $NOG
try s8_nogeom     scan.slang scan_main 5 16 -DVM_MAX_REGS=8 $NOG
try a8_4_f8       agg.slang  agg_main  5 16 -DVM_MAX_REGS=8 -DAGG_MAX=4 -DSCAN_WG=128 $NONUM $NOJ $NOG
BISECT_TIMEOUT=${BISECT_TIMEOUT2:-240}; TMO=$BISECT_TIMEOUT
try a8_4_nogeom   agg.slang  agg_main  5 16 -DVM_MAX_REGS=8 -DAGG_MAX=4 -DSCAN_WG=128 $NOG
