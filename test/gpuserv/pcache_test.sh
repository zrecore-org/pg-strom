#!/bin/sh
# Persisted Vulkan pipeline cache (gpu_vk.c, pcache_open): time to create the VM join pipelines with the cache off,
# cold (empty directory) and warm (second service start), and the handling of a corrupt, truncated or foreign cache
# file. Needs the GPU but no cluster; run it through locked_test.sh (one service on the GPU at a time):
#
#   sh test/gpuserv/locked_test.sh "$PWD" pcache_test.sh        [KERNELS="gpujoin16 gpujoin16_lite gpujoin16_nogeom ..."] [DEV=0]
#
# The numbers are printed ("pcache <kernel> create_ms=.."); the OS's own shader cache (Metal) is not cleared, so
# "cold" is cold for the Vulkan cache only. The checks are about the files, not the speed (a warm run must not be
# slower than half again the cold run).
D=$(cd "$(dirname "$0")/../.." && pwd)
KERNELS=${KERNELS:-gpujoin16 gpujoin16_lite gpujoin16_nogeom}
DEV=${DEV:-0}
TESTDATA=${PGSTROM_TEST_DATA:-/Volumes/CrucialX9/Projects/pg-strom/test-data}
W=${ROOT:-$TESTDATA/pcache_test}
fail=0
check() { if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi; }

make -C "$D/src/gpuserv" pcache_check >/dev/null || { echo "pcache_test: build failed"; exit 2; }
BIN=$D/src/gpuserv/pcache_check
rm -rf "$W"; mkdir -p "$W/cache"
export PGSTROM_GPUSERV_CACHE_DIR=$W/cache PGSTROM_GPUSERV_PIPELINE_LOG=1
run() { "$BIN" "$DEV" $KERNELS >"$W/$1.out" 2>"$W/$1.err"; _rc=$?; sed 's/^/        /' "$W/$1.out"; return $_rc; }
total() { awk '{for(i=1;i<=NF;i++) if ($i ~ /^create_ms=/) {split($i,a,"="); s+=a[2]}} END{print s+0}' "$W/$1.out"; }
nfiles() { ls "$W/cache"/pipeline-*.bin 2>/dev/null | wc -l | awk '{print $1+0}'; }

echo "cache off"
(PGSTROM_GPUSERV_PIPELINE_CACHE=off; export PGSTROM_GPUSERV_PIPELINE_CACHE; run off); check "pipelines created with the cache off" $?
[ "$(nfiles)" -eq 0 ]; check "no cache file written when off" $?

echo "cold"
run cold; check "pipelines created, cold cache" $?
[ "$(nfiles)" -eq 1 ]; check "one cache file written" $?
F=$(ls "$W/cache"/pipeline-*.bin | head -1)
SZ=$(wc -c < "$F" | awk '{print $1}')
echo "        cache file: $SZ bytes"
[ "$SZ" -gt 64 ]; check "the file holds more than a header" $?
ls "$W/cache"/*.tmp >/dev/null 2>&1; [ $? -ne 0 ]; check "no temporary file left behind" $?

echo "warm"
run warm; check "pipelines created, warm cache" $?
COLD=$(total cold); WARM=$(total warm); OFF=$(total off)
echo "        total create_ms: off=$OFF cold=$COLD warm=$WARM"
[ "$WARM" -le $((COLD + COLD / 2 + 200)) ]; check "warm is not slower than cold" $?

echo "damaged files"
head -c 100 "$F" > "$F.t" && mv "$F.t" "$F"
run trunc; check "a truncated file is ignored" $?
[ "$(wc -c < "$F" | awk '{print $1}')" -ge "$SZ" ]; check "the truncated file was replaced by a full one" $?
cp "$F" "$W/good.bin"
dd if=/dev/urandom of="$F" bs=1024 count=64 2>/dev/null
run garbage; check "a garbage file is ignored" $?
[ "$(wc -c < "$F" | awk '{print $1}')" -ge "$SZ" ]; check "the garbage file was replaced by a full one" $?
# a file of another device or driver: valid frame, the Vulkan header names another vendor id (checked before the driver
# sees the data; rebuilt with a correct hash so that only the header check can reject it)
cp "$W/good.bin" "$F"
python3 - "$F" <<'PY' || { echo "  FAIL  cannot forge the foreign file"; fail=$((fail+1)); }
import struct, sys
p = sys.argv[1]
d = bytearray(open(p, 'rb').read())
blob = d[24:]
blob[8:12] = b'\xff\xff\xff\xff'
h = 1469598103934665603
n = len(blob)
for i in range(0, n - n % 8, 8):
    w = struct.unpack_from('<Q', blob, i)[0]
    h = ((h ^ w) * 1099511628211) & (2**64 - 1)
    h ^= h >> 29
for i in range(n - n % 8, n):
    h = ((h ^ blob[i]) * 1099511628211) & (2**64 - 1)
h ^= n
open(p, 'wb').write(struct.pack('<QQQ', 0x31435053474f5250, n, h) + bytes(blob))
PY
run foreign; check "a file of another device is ignored" $?
[ "$(wc -c < "$F" | awk '{print $1}')" -ge "$SZ" ]; check "the foreign file was replaced" $?
# a good file with flipped bytes in the body: the driver must reject or tolerate it, never crash
cp "$W/good.bin" "$F"
printf '\377\377\377\377\377\377\377\377' | dd of="$F" bs=1 seek=$((SZ / 2)) conv=notrunc 2>/dev/null
run body; check "a file with a damaged body does not break pipeline creation" $?

echo "no cache directory"
(PGSTROM_GPUSERV_CACHE_DIR=none; HOME=/nonexistent; XDG_CACHE_HOME=; export PGSTROM_GPUSERV_CACHE_DIR HOME XDG_CACHE_HOME; run nodir); check "pipelines created without a cache directory" $?
ls "$W/cache"/*.tmp >/dev/null 2>&1; [ $? -ne 0 ]; check "no temporary file left behind (end)" $?

[ $fail -eq 0 ] && { echo "pcache_test: ALL PASSED"; rm -rf "$W"; exit 0; }
echo "pcache_test: $fail FAILED (files kept in $W)"
exit 1
