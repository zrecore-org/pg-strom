#!/bin/sh
# Safety test: can the GPU read FILE-BACKED host memory whose pages are NOT in the OS cache?
# The copy of the relation file is written with F_NOCACHE so it does not enter the page cache,
# then mapped, imported and scanned with the GPU as the first reader (no CPU sanity pass).
set -e
cd "$(dirname "$0")"
export SLANGC=${SLANGC:-$HOME/tools/slang-2026.20/bin/slangc}
./pg_setup.sh >/dev/null 2>&1
trap './pg_teardown.sh >/dev/null 2>&1; rm -f /tmp/zc_cold_rel' EXIT
REL=/tmp/pgscan/data/$(./pgq.sh -c "SELECT pg_relation_filepath('t')")
SUM="sum(((ctid::text::point)[0])::bigint * 512 + ((ctid::text::point)[1])::bigint)"
E1=$(./pgq.sh -F ' ' -c "SELECT count(*), $SUM FROM t WHERE b + a::int8 > 1073741824 AND f > 500::float4 AND d < 25 AND a > 1000")
E2=$(./pgq.sh -F ' ' -c "SELECT count(*), $SUM FROM t WHERE (a > 1000000000 OR b IS NULL) AND NOT (f < 100::float4 OR d IS NULL)")
./pgq.sh -c "CHECKPOINT" >/dev/null
echo "oracle E1: $E1   E2: $E2"
python3 -I - "$REL" /tmp/zc_cold_rel <<'PY'
import os, sys, fcntl
F_NOCACHE = 48                       # macOS fcntl: bypass the page cache
src, dst = sys.argv[1:3]
fi = os.open(src, os.O_RDONLY); fcntl.fcntl(fi, F_NOCACHE, 1)
fo = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600); fcntl.fcntl(fo, F_NOCACHE, 1)
while True:
    b = os.read(fi, 8 << 20)
    if not b: break
    os.write(fo, b)
os.fsync(fo); os.close(fi); os.close(fo)
PY
echo; echo "######## file (MAP_SHARED), pages cold, GPU is the first reader"
HEAP_SRC=file HEAP_SKIP_SANITY=1 timeout 600 ./heapspike shaders /tmp/zc_cold_rel $E1 $E2 2>&1 | grep -vE "generated kernel compile|^$"
