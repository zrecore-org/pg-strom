#!/bin/sh
# The string hash of the join kernels (src/gpuserv/kernels_vm/hash_bytes.inc) on the Vulkan device against the values of
# PostgreSQL's hashtext() for 0..40 byte strings at four alignments. Needs slangc and the Vulkan loader; no cluster, no lock.
#
#   sh test/gpuserv/hash_check.sh [device index]
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
make -C "$ROOT/src/gpuserv" hash_check build/hash_check.spv >/dev/null || { echo "hash_check: build failed"; exit 2; }
cd "$ROOT/src/gpuserv" && ./hash_check "${1:-0}" build/hash_check.spv
