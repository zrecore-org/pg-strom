#!/bin/sh
# Why does the T4's shader compiler reject the scan kernel with the numeric group ("NVVM compilation failed: 3" for
# gpuscan*_nogeom, while the aggregation kernel with the same group compiles)? Compiles variants of kernels_vm/scan.slang
# (numeric group only, 8 registers) with slangc and creates each pipeline with pgstrom-gpuserv --prewarm.
#   SLANGC=... GPUSERV_BIN=... sh test/gpuserv/nvidia_bisect.sh      BISECT_TIMEOUT=60 DEVICE=0
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN}
SLANGC=${SLANGC:?set SLANGC}
SRC=$(cd "${0%/*}/../../src/gpuserv" && pwd)
OUT=${BISECT_OUT:-${TMPDIR:-/tmp}/nvbisect}; mkdir -p "$OUT"
TMO=${BISECT_TIMEOUT:-90}
export PGSTROM_GPUSERV_PIPELINE_CACHE=off
try() { # name  slang-file (path)  entry  nbind pcsize  defs...
	name=$1; file=$2; entry=$3; nb=$4; pc=$5; shift 5
	t0=$(date +%s)
	( cd "$SRC" && "$SLANGC" "$file" -I kernels_vm "$@" -target spirv -entry "$entry" -stage compute -o "$OUT/$name.spv" ) > "$OUT/$name.slangc.log" 2>&1 || { echo "  SLANGC-FAIL $name: $(head -c 300 "$OUT/$name.slangc.log")"; return; }
	t1=$(date +%s)
	size=$(wc -c < "$OUT/$name.spv" | tr -d ' ')
	out=$(timeout "$TMO" "$GPUSERV_BIN" --prewarm "$nb" "$pc" "$OUT/$name.spv" --device "${DEVICE:-0}" 2>&1); rc=$?
	t2=$(date +%s)
	if [ $rc -eq 0 ]; then echo "  OK    $name ($size bytes, slangc $((t1-t0))s, pipeline $((t2-t1))s)"
	else echo "  FAIL  $name ($size bytes, slangc $((t1-t0))s, pipeline $((t2-t1))s, rc=$rc): $(echo "$out" | tr '\n' ' ' | cut -c1-150)"; fi
}
NUMONLY="-DVM_MAX_REGS=8 -DVM_NO_JSON -DVM_NO_F8 -DVM_NO_GEOM"
S=$SRC/kernels_vm/scan.slang
mk() { # name sed-expression: a copy of scan.slang with an edit
	mkdir -p "$OUT/$1"; sed "$2" "$S" > "$OUT/$1/scan.slang"
	if cmp -s "$S" "$OUT/$1/scan.slang"; then echo "  (variant $1: the edit changed nothing)"; fi
}
try s_orig kernels_vm/scan.slang scan_main 5 16 $NUMONLY
mk v1 's/r.isnull | (r.sc << 8)/r.isnull/'
try v1_no_sc "$OUT/v1/scan.slang" scan_main 5 16 $NUMONLY
mk v2 's/Reg r = regs\[prog\[compoff + k\]\];/Reg r = regs[0];/'
try v2_static_reg "$OUT/v2/scan.slang" scan_main 5 16 $NUMONLY
mk v3 's/hit ? 1u : (g_recheck != 0 ? 2u : 0u)/hit ? 1u : 0u/'
try v3_no_recheck "$OUT/v3/scan.slang" scan_main 5 16 $NUMONLY
mk v4 's/^        Reg  regs\[VM_MAX_REGS\];//; s/^    uint val_base = 4 + kparam.total_lps;/    uint val_base = 4 + kparam.total_lps; Reg regs[VM_MAX_REGS];/'
try v4_regs_outside "$OUT/v4/scan.slang" scan_main 5 16 $NUMONLY
mk v5 's/^                for (uint k = 0; k < ncomp; k++)/                for (uint k = 0; k < 0; k++)/'
try v5_no_output "$OUT/v5/scan.slang" scan_main 5 16 $NUMONLY
mk v6 's/hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs);/hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nins, qreg, regs); if (!hit) regs[0].v = 0;/'
try v6_hit_use "$OUT/v6/scan.slang" scan_main 5 16 $NUMONLY
try arrow8_num kernels_vm/arrow.slang arrow_main 5 16 $NUMONLY
