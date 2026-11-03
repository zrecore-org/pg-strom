#!/bin/sh
# Zero-copy page import test: every page source against the SQL oracle, plus the
# coherence check. Recreates the throwaway cluster (pg_setup.sh) and removes it
# at the end (pg_teardown.sh). usage: ./zc_test.sh
set -e
cd "$(dirname "$0")"
export SLANGC=${SLANGC:-$HOME/tools/slang-2026.20/bin/slangc}
make heapspike shaders/heap_fixed.spv shaders/heap_vm.spv >/dev/null 2>&1

./pg_setup.sh >/dev/null 2>&1
trap './pg_teardown.sh >/dev/null 2>&1; rm -f /tmp/zc_rel_*; HEAP_SRC=shm HEAP_SHM_ROLE=unlink ./heapspike shaders /dev/null 0 0 0 0 >/dev/null 2>&1' EXIT

REL=/tmp/pgscan/data/$(./pgq.sh -c "SELECT pg_relation_filepath('t')")
SUM="sum(((ctid::text::point)[0])::bigint * 512 + ((ctid::text::point)[1])::bigint)"
E1=$(./pgq.sh -F ' ' -c "SELECT count(*), $SUM FROM t WHERE b + a::int8 > 1073741824 AND f > 500::float4 AND d < 25 AND a > 1000")
E2=$(./pgq.sh -F ' ' -c "SELECT count(*), $SUM FROM t WHERE (a > 1000000000 OR b IS NULL) AND NOT (f < 100::float4 OR d IS NULL)")
echo "oracle E1: $E1   E2: $E2"
./pgq.sh -c "CHECKPOINT" >/dev/null

run() {   # run <title> <env...>  -- the oracle numbers are word-split on purpose
  title=$1; shift
  echo; echo "######## $title"
  env "$@" HEAP_COHERENCE=1 ./heapspike shaders "$RELARG" $E1 $E2 2>&1 | grep -vE "^\[PASS\] (VM|SPEC)|generated kernel compile|^E[12]:|^$"
}
RELARG=$REL;                run "copy (control: memcpy into a Vulkan buffer)" HEAP_SRC=copy
RELARG=$REL;                run "anon  (anonymous mmap, imported)"            HEAP_SRC=anon
cp "$REL" /tmp/zc_rel_shared; RELARG=/tmp/zc_rel_shared
echo; echo "######## shm: load in one process"
HEAP_SRC=shm HEAP_SHM_ROLE=load ./heapspike shaders "$RELARG" $E1 $E2 2>&1 | grep -E "loaded|FAIL|rror"
RELARG=$REL;                run "shm   (POSIX shm filled by another process, attached and imported)" HEAP_SRC=shm HEAP_SHM_ROLE=scan
HEAP_SRC=shm HEAP_SHM_ROLE=unlink ./heapspike shaders "$RELARG" 0 0 0 0 2>&1 | grep shm_unlink
cp "$REL" /tmp/zc_rel_file;  RELARG=/tmp/zc_rel_file;  run "file  (relation file mmap MAP_SHARED = OS page cache, imported)" HEAP_SRC=file
cp "$REL" /tmp/zc_rel_priv;  RELARG=/tmp/zc_rel_priv;  run "filepriv (relation file mmap MAP_PRIVATE, imported)" HEAP_SRC=filepriv
