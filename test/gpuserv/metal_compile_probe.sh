#!/bin/sh
# metal_compile_probe.sh - compile SPIR-V kernels with the driver, one after the other, and print the seconds each took (or that it failed).
# Run it through the test lock (the driver's compiler is a shared, heavy resource; PREWARM=off so that the wrapper does not warm the kernels first):
#
#   PREWARM=off SPV_DIR=<dir with .spv> SPV_LIST="base_j8 cur_j8" sh test/gpuserv/locked_test.sh "$PWD" metal_compile_probe.sh
#
# Kernels are join kernels (6 bindings, 20 bytes of push constants); LIMIT (default 1200) is the seconds after which a compile is abandoned.
BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
DIR=${SPV_DIR:?set SPV_DIR}
LIMIT=${LIMIT:-1200}
export PGSTROM_GPUSERV_PIPELINE_CACHE=off
for k in ${SPV_LIST:?set SPV_LIST}; do
	t0=$(date +%s)
	"$BIN" --prewarm 6 20 "$DIR/$k.spv" > "$DIR/$k.probe.log" 2>&1 &
	pid=$!
	( sleep "$LIMIT"; kill "$pid" 2>/dev/null ) &
	killer=$!
	wait "$pid"
	rc=$?
	kill "$killer" 2>/dev/null
	t1=$(date +%s)
	echo "$k: $((t1 - t0)) s, exit $rc: $(grep -h -i -E 'prewarm|failed' "$DIR/$k.probe.log" | head -2 | sed 's#/[^ ]*/##g' | tr '\n' ' ')"
done
