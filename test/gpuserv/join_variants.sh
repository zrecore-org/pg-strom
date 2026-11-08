#!/bin/sh
# join_variants.sh <outdir> [<baseline kernels_vm dir>]
# Builds the join kernel (join.slang, VM_MAX_REGS 8 and 16) in the variants that decide how much of the point-to-geometry walker a join kernel carries
# (A13 slice 4), plus - given a directory with the kernels_vm of an older commit (git archive <rev> src/gpuserv/kernels_vm | tar -x -C <dir>) -
# that commit's kernel, for comparison. The SPIR-V files and their sizes go to <outdir>; test/gpuserv/metal_compile_probe.sh compiles them with the
# driver and prints the seconds each took (Metal's compiler service gives up on some kernels: see the plan, A13 slice 4).
OUT=${1:?usage: join_variants.sh <outdir> [<baseline kernels_vm parent dir>]}
BASE=$2
SLANGC=${SLANGC:-slangc}
HERE=$(cd "$(dirname "$0")/../../src/gpuserv" && pwd)
mkdir -p "$OUT"
build() { # name, dir, flags...
	name=$1; dir=$2; shift 2
	(cd "$dir" && $SLANGC kernels_vm/join.slang -I kernels_vm "$@" -target spirv -entry join_main -stage compute -o "$OUT/$name.spv") 2>&1 | grep -E "error"
}
for r in 8 16; do
	build cur_j$r "$HERE" -DVM_MAX_REGS=$r
	build no3d_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_GEOM3D
	build lines_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_GEOM3D -DVM_NO_GEOMPOLY
	build nonum_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_NUM -DVM_NO_JSON
	build geomonly_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_NUM -DVM_NO_JSON -DVM_NO_F8
	build geom2d_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_NUM -DVM_NO_JSON -DVM_NO_F8 -DVM_NO_GEOM3D
	build geomlines_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_NUM -DVM_NO_JSON -DVM_NO_F8 -DVM_NO_GEOM3D -DVM_NO_GEOMPOLY
	build f8only_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_NUM -DVM_NO_JSON -DVM_NO_GEOM
	build nogeom_j$r "$HERE" -DVM_MAX_REGS=$r -DVM_NO_GEOM
	[ -n "$BASE" ] && build base_j$r "$BASE" -DVM_MAX_REGS=$r
done
ls -l "$OUT"/*.spv | awk '{print $5, $9}'
