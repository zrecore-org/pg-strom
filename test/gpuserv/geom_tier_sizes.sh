#!/bin/sh
# geom_tier_sizes.sh <outdir> [regs] [kernel] - SPIR-V size of one VM kernel per geometry group mask (R15). Prints "label bytes".
# kernel: join (default) | jgagg | scan | agg ...; the masks are VM_GEOM_MASK values (scan_common.slang VM_G_*).
OUT=${1:?usage: geom_tier_sizes.sh <outdir> [regs] [kernel]}
R=${2:-16}
K=${3:-join}
SLANGC=${SLANGC:-$(command -v slangc || echo "$HOME/tools/slang-2026.20/bin/slangc")}
D=$(cd "$(dirname "$0")/../../src/gpuserv" && pwd)
mkdir -p "$OUT"
case $K in
join) ent=join_main; extra="";;
jgagg) ent=jgagg_main; extra="-DGA_NA=4 -DGA_C=128";;
jagg) ent=jagg_main; extra="-DAGG_MAX=4";;
scan) ent=scan_main; extra="";;
agg) ent=agg_main; extra="-DAGG_MAX=4 -DSCAN_WG=128";;
gagg) ent=gagg_main; extra="-DGA_NA=4 -DGA_C=128";;
esac
LITE="-DVM_NO_NUM -DVM_NO_JSON -DVM_NO_F8"
cd "$D" || exit 1
for m in ${MASKS:-0 1 2 4 12 16 32 64 127}; do
	f="$OUT/${K}${R}_m$m.spv"
	$SLANGC kernels_vm/$K.slang -I kernels_vm -DVM_MAX_REGS=$R $extra $LITE -DVM_GEOM_MASK=$m ${EXTRA_FLAGS} -target spirv -entry $ent -stage compute -o "$f" 2>&1 | grep -E "error"
	echo "$K$R mask=$m $(wc -c < "$f" | tr -d ' ')"
done
