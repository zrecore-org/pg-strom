#!/bin/sh
# Offline check of the kernel generator: build spec_check, write the generated sources of every kernel kind
# and compile each with slangc (no GPU, no PostgreSQL, no lock needed):
#
#   sh test/gpuserv/spec_compile_check.sh [outdir]
#
# SLANGC=/path/to/slangc to override the compiler.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GS=$ROOT/src/gpuserv
OUT=${1:-/tmp/spec_check_out}
SLANGC=${SLANGC:-$(command -v slangc || echo "$HOME/tools/slang-2026.20/bin/slangc")}
make -C "$GS" spec_check >/dev/null || { echo "spec_check: build failed"; exit 1; }
rm -rf "$OUT"
"$GS/spec_check" "$OUT" >/dev/null || { echo "spec_check: the generator failed"; exit 1; }
fail=0
for k in scan:scan_main agg:agg_main gagg:gagg_main join:join_main jagg:jagg_main jgagg:jgagg_main gagg3:gagg_main jgagg3:jgagg_main ascan:arrow_main aagg:agg_main agagg:gagg_main ajoin:join_main ajagg:jagg_main ajgagg:jgagg_main gaggdm:gagg_main jgaggdm:jgagg_main agaggdm:gagg_main ajgaggdm:jgagg_main; do
	name=${k%%:*}; entry=${k##*:}
	s=$(date +%s)
	if "$SLANGC" "$OUT/$name/main.slang" -I "$OUT/$name" -I "$GS/kernels_vm" -target spirv -entry "$entry" -stage compute -o "$OUT/$name/out.spv" >"$OUT/$name/slangc.log" 2>&1; then
		echo "  ok    $name ($(wc -c < "$OUT/$name/out.spv" | tr -d ' ') bytes of SPIR-V, $(( $(date +%s) - s )) s)"
	else
		echo "  FAIL  $name"; head -20 "$OUT/$name/slangc.log"; fail=$((fail+1))
	fi
done
[ $fail -eq 0 ] && echo "spec_compile_check: all passed" || { echo "spec_compile_check: $fail failed"; exit 1; }
