#!/bin/sh
# f64soft_corpus_hash.sh [N]: print a hash of every random draw and every operand pair that src/gpuserv/f64soft_check.c consumes
# (host part only, no GPU, no lock). The corpus must not depend on the compiler's order of evaluation: run it with clang and with
# GCC (CC=gcc) and compare the lines. Recorded value on the M1 (clang) after the rewrite of 2026-11-14, N=300000:
#   RNDHASH ddd0cc8e93da99b0 calls 65460421
ROOT=$(cd "${0%/*}/../.." && pwd); GS=$ROOT/src/gpuserv
N=${1:-300000}; CC=${CC:-cc}
W=${PGSTROM_TEST_DATA:-$ROOT/test-data}/f64soft_corpus_hash
make -C "$GS" libgpuserv.a >/dev/null 2>&1 || { echo "libgpuserv.a: build failed"; exit 2; }
rm -rf "$W"; mkdir -p "$W"
cat > "$W/rndhash.h" <<'H'
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static uint64_t rnd_raw(void);
static uint64_t H_ = 1469598103934665603ULL;
static unsigned long long NCALL_;
static void H_print(void) { fprintf(stderr, "RNDHASH %016llx calls %llu\n", (unsigned long long) H_, NCALL_); }
static void H_fold(uint64_t v) { H_ = (H_ ^ v) * 1099511628211ULL; }
static uint64_t rnd(void) { static int reg; if (!reg) { reg = 1; atexit(H_print); } uint64_t v = rnd_raw(); H_fold(v); NCALL_++; return v; }
H
mkdir -p "$W/kernels_vm"; cp "$GS/kernels_vm/f64_soft.inc" "$W/kernels_vm/"
# the generator is renamed in a private copy; the operand pairs are folded where gen_pair() and prep() hand them out
sed -e 's/^rnd(void)$/rnd_raw(void)/' -e 's/^\t\*pb = b;$/\t*pb = b; H_fold(a); H_fold(b);/' \
	-e '/^prep(int op/,/^}/ s/^\t(void) b;$/\tH_fold(*a); H_fold(*b); (void) b;/' "$GS/f64soft_check.c" > "$W/f64soft_check.c"
grep -q 'H_fold(a)' "$W/f64soft_check.c" && grep -q 'H_fold(\*a)' "$W/f64soft_check.c" || { echo "the hooks were not inserted: f64soft_check.c changed shape"; exit 2; }
VK_CFLAGS=${VK_CFLAGS:-$(pkg-config --cflags vulkan 2>/dev/null)}
VK_LIBS=${VK_LIBS:-$(pkg-config --libs vulkan 2>/dev/null || printf '%s' -lvulkan)}
$CC -O2 -std=gnu11 -I"$GS" $VK_CFLAGS -include "$W/rndhash.h" -o "$W/chk" "$W/f64soft_check.c" "$GS/libgpuserv.a" $VK_LIBS -lpthread -lm \
	> "$W/build.log" 2>&1 || { echo "build failed:"; head -20 "$W/build.log"; exit 2; }
nice -n 10 "$W/chk" "$N" -1 2>&1 | grep -E "RNDHASH|F64SOFT CHECK"
