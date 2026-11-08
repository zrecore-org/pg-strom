#!/bin/sh
# Cold-compile time of one hand-built variant of a VM kernel (the Metal first-compile cost): builds the SPIR-V with
# the given slangc defines and times its pipeline creation with pcache_check.
#   sh test/gpuserv/kernel_compile_time.sh <label> <shader> <entry> [slangc args ...]
# e.g. the Arrow GROUP BY over a join, 8 aggregates, 64 slots:
#   sh test/gpuserv/kernel_compile_time.sh na8c64 jgagg jgagg_main -DARROW_SRC=1 -DVM_MAX_REGS=16 -DGA_NA=8 -DGA_C=64
# Output: test-data/compile_exp/<label>.txt. Only the Metal compiler's time is meaningful when nothing else compiles.
D=$(cd "$(dirname "$0")/../.." && pwd)
label=$1; shader=$2; entry=$3; shift 3
SLANGC=${SLANGC:-$(command -v slangc || echo "$HOME/tools/slang-2026.20/bin/slangc")}
out=${COMPILE_EXP_DIR:-$D/test-data/compile_exp}
mkdir -p "$out"
cd "$D/src/gpuserv" || exit 1
"$SLANGC" kernels_vm/$shader.slang -I kernels_vm "$@" -target spirv -entry "$entry" -stage compute -o "$out/$label.spv" || exit 1
size=$(wc -c < "$out/$label.spv" | tr -d ' ')
t=$( { PGSTROM_GPUSERV_PIPELINE_CACHE=off /usr/bin/time -p nice -n 10 ./pcache_check 0 "@$out/$label.spv"; } 2>&1 | tr '\n' ' ')
echo "$label spirv_bytes=$size $* :: $t" | tee "$out/$label.txt"
