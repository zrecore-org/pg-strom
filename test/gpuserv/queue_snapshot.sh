#!/bin/sh
# Run a test script through locked_test.sh on a snapshot of the committed tree (git archive HEAD), so that edits
# made while the run waits for the lock do not change what is built and tested. A development helper:
#
#   sh test/gpuserv/queue_snapshot.sh <snapshot dir> <test script name or path> [outfile]
#
# env of the caller (SPECS=..., PGSTROM_GPUSERV_SPECIALIZE=...) passes through.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SNAP=${1:?snapshot dir}
TEST=${2:?test script}
OUT=${3:-/dev/stdout}
rm -rf "$SNAP"; mkdir -p "$SNAP"
(cd "$ROOT" && git archive HEAD) | tar -x -C "$SNAP" || exit 1
sh "$SNAP/test/gpuserv/locked_test.sh" "$SNAP" "$TEST" > "$OUT" 2>&1
