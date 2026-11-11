#!/bin/sh
# S0 of A10 part 6: measures double-float (df64, src/gpuserv/kernels_vm/df64.inc) against the software binary64 and float32 on
# this device. No cluster, no module install (like f64soft_check.sh it needs only slangc and the Vulkan loader, so no test lock;
# it does use the GPU: do not run it next to another GPU program). Every dispatch is small and finite.
#
#   sh test/gpuserv/df64_probe.sh [corr|bench|cover|all]        (default all)
#
#   corr   exact fp32 rounding of TwoSum / FastTwoSum / TwoProduct (fma) and the df64 operations, bit for bit against the host, in
#          several toolchain variants (slangc -fp-mode, `precise` on every temporary, MVK_CONFIG_FAST_MATH_ENABLED)
#   bench  ALU throughput, best of 5: float32 / df64 / software binary64 for add, mul, div, sqrt, a chain of 16 fma, compare
#   cover  host only: the fraction of rows a df64 evaluation with a proven error bound decides (float8 qual shapes)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WHAT=${1:-all}
G="$ROOT/src/gpuserv"
SLANGC=${SLANGC:-$(command -v slangc || ls "$HOME"/tools/slang-*/bin/slangc 2>/dev/null | tail -1)}
[ -n "$SLANGC" ] || { echo "df64_probe: slangc not found"; exit 2; }
make -C "$G" df64_probe >/dev/null || { echo "df64_probe: build failed"; exit 2; }
mkdir -p "$G/build"
cd "$G" || exit 2
sl() {		# sl <src> <out> <slangc flags...>
	src=$1; out=$2; shift 2
	"$SLANGC" "$src" -I kernels_vm "$@" -target spirv -entry main -stage compute -o "$out" || { echo "df64_probe: slangc failed for $out"; exit 2; }
}
# gpu_vk_open() would otherwise recompile every kernel of the persisted pipeline cache (minutes) before the probe starts
export PGSTROM_GPUSERV_PIPELINE_CACHE=off
rc=0
if [ "$WHAT" = corr ] || [ "$WHAT" = all ]; then
	echo "== (a) correctness of the error-free transformations on this device"
	sl kernels_check/df64_corr.slang build/df64_corr_def.spv
	sl kernels_check/df64_corr.slang build/df64_corr_precise.spv -fp-mode precise
	sl kernels_check/df64_corr.slang build/df64_corr_fast.spv -fp-mode fast
	sl kernels_check/df64_corr.slang build/df64_corr_dfp.spv -DDF_USE_PRECISE
	for v in "def|build/df64_corr_def.spv|" "def+MVK fast math off|build/df64_corr_def.spv|0" "def+MVK fast math on|build/df64_corr_def.spv|1" \
			 "-fp-mode precise|build/df64_corr_precise.spv|" "-fp-mode fast|build/df64_corr_fast.spv|" \
			 "-fp-mode precise+MVK fast math off|build/df64_corr_precise.spv|0" "precise temporaries|build/df64_corr_dfp.spv|" "precise temporaries+MVK fast math off|build/df64_corr_dfp.spv|0" ; do
		name=${v%%|*}; rest=${v#*|}; spv=${rest%%|*}; fm=${rest#*|}
		echo "-- variant: $name"
		if [ -n "$fm" ]; then MVK_CONFIG_FAST_MATH_ENABLED=$fm ./df64_probe corr "$spv" ${N:-1000000}; else ./df64_probe corr "$spv" ${N:-1000000}; fi || rc=1
	done
fi
if [ "$WHAT" = bench ] || [ "$WHAT" = all ]; then
	echo "== (b) ALU throughput with MVK_CONFIG_FAST_MATH_ENABLED=0 (the switch that keeps df64 exact, see (a); `precise` is 30x slower: SPIRV-Cross wraps NoContraction operations in non-inlined optnone functions) (Gop/s, best of 5; one op = one add/mul/div/sqrt(x+y)/fma-step/compare)"
	export MVK_CONFIG_FAST_MATH_ENABLED=${BENCH_FASTMATH:-0}
	for op in 0 1 2 3 4 5; do
		for var in 0 1 2; do
			sl kernels_check/df64_bench.slang build/df64_bench_${op}_$var.spv -DVAR=$var -DOP=$op
			./df64_probe bench build/df64_bench_${op}_$var.spv $op $var || rc=1
		done
	done
	sl kernels_check/df64_bench.slang build/df64_bench_6_1.spv -DVAR=1 -DOP=6
	./df64_probe bench build/df64_bench_6_1.spv 6 1 || rc=1
fi
if [ "$WHAT" = cover ] || [ "$WHAT" = all ]; then
	echo "== (c) coverage"
	./df64_probe cover ${COVER_N:-2000000} || rc=1
fi
exit $rc
