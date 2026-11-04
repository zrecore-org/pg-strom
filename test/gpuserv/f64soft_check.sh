#!/bin/sh
# Software binary64 (src/gpuserv/kernels_vm/f64_soft.inc): bit-for-bit comparison with the hardware double, on the host
# (the library compiled as C) and on the Vulkan device (compiled by slangc). No cluster, no module install: it needs
# only slangc and the Vulkan loader, so it does not take the test lock.
#
#   sh test/gpuserv/f64soft_check.sh [N [device index|-1]]       N operand pairs per operation (default 4000000); -1: host only
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
N=${1:-4000000}
DEV=${2:-0}
make -C "$ROOT/src/gpuserv" f64soft_check build/f64soft_check.spv >/dev/null || { echo "f64soft_check: build failed"; exit 2; }
cd "$ROOT/src/gpuserv" && ./f64soft_check "$N" "$DEV" build/f64soft_check.spv
