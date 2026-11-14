#!/bin/sh
# Mutation check of the host-side self-test of the software binary64 (src/gpuserv/f64soft_check.c): each mutant breaks one
# rounding / sticky-bit corner of kernels_vm/f64_soft.inc, and f64soft_check (host part only, no GPU, no lock) must FAIL on it.
# A mutant that passes means the operand corpus does not reach that corner. The baseline must pass first.
#
#   sh test/gpuserv/f64soft_mutants.sh [N]      N operand pairs per operation (default 1000000)
# The MoltenVK-independent idea comes from the ../mongo POC 12 (planted mutants: the sticky-bit one was caught by only 12 pairs).
ROOT=$(cd "${0%/*}/../.." && pwd)
GS=$ROOT/src/gpuserv
N=${1:-1000000}
W=${PGSTROM_TEST_DATA:-$ROOT/test-data}/f64soft_mutants
CC=${CC:-cc}
make -C "$GS" libgpuserv.a >/dev/null 2>&1 || { echo "libgpuserv.a: build failed"; exit 2; }
rm -rf "$W"; mkdir -p "$W/kernels_vm"
cp "$GS/f64soft_check.c" "$W/"
VK_CFLAGS=${VK_CFLAGS:-$(pkg-config --cflags vulkan 2>/dev/null)}
VK_LIBS=${VK_LIBS:-$(pkg-config --libs vulkan 2>/dev/null || printf '%s' -lvulkan)}
build_run() { # -> exit status of f64soft_check on the current $W/kernels_vm/f64_soft.inc
	$CC -O2 -Wall -std=gnu11 -I"$GS" $VK_CFLAGS -o "$W/chk" "$W/f64soft_check.c" "$GS/libgpuserv.a" $VK_LIBS -lpthread -lm >"$W/build.log" 2>&1 || { echo "build failed"; return 99; }
	nice -n 10 "$W/chk" "$N" -1 >"$W/out.log" 2>&1
}
cp "$GS/kernels_vm/f64_soft.inc" "$W/orig.inc"
cp "$W/orig.inc" "$W/kernels_vm/f64_soft.inc"
build_run; rc=$?
[ $rc -eq 0 ] || { echo "baseline FAILED (rc $rc): fix the library or the build first"; tail -5 "$W/out.log" "$W/build.log"; exit 2; }
echo "baseline: passes"
bad=0
mutant() { # name, exact text, replacement [, n: mutate only the n-th occurrence (the text must occur at least n times); default: it must occur once]
	python3 - "$W/orig.inc" "$W/kernels_vm/f64_soft.inc" "$2" "$3" "${4:-0}" <<'PY' || { echo "  ERROR $1: the text to mutate is not in f64_soft.inc as asked"; bad=$((bad+1)); return; }
import sys
s = open(sys.argv[1]).read()
old, new, n = sys.argv[3], sys.argv[4], int(sys.argv[5])
if n == 0:
    if s.count(old) != 1:
        sys.exit(1)
    out = s.replace(old, new)
else:
    if s.count(old) < n:
        sys.exit(1)
    i = -1
    for _ in range(n):
        i = s.index(old, i + 1)
    out = s[:i] + new + s[i + len(old):]
open(sys.argv[2], "w").write(out)
PY
	build_run; rc=$?
	if [ $rc -eq 0 ] && [ "$EQUIV" = 1 ]; then echo "  equivalent $1 (survives by design, see the comment)"
	elif [ $rc -eq 0 ]; then echo "  SURVIVED $1 (the host corpus does not catch it)"; bad=$((bad+1))
	elif [ $rc -eq 99 ]; then echo "  ERROR $1 (the mutant does not build)"; bad=$((bad+1))
	else echo "  killed   $1 ($(grep -ci 'mismatch\|differ\|FAIL' "$W/out.log") reports)"; fi
}
mutant "shr_jam drops the sticky bit (n < 64)" '(x >> n) | (((x << (64u - n)) != 0ull) ? 1ull : 0ull)' '(x >> n)'
mutant "shr_jam drops the sticky bit (n >= 64)" 'return (x != 0ull) ? 1ull : 0ull;' 'return 0ull;'
mutant "rounding: ties do not go to even" 'r &= ~1ull;' ';'
mutant "rounding: ties always round down" '(sig + 0x200ull) >> 10' '(sig + 0x1FFull) >> 10'
mutant "rounding: half rounds up (no tie rule, bias)" '(sig + 0x200ull) >> 10' '(sig + 0x201ull) >> 10'
# The sticky of a shifted fma / sqr_add term that comes from the PRODUCT's low word (the `sl` of the shifted term, when the addend is the larger
# term) cannot change a rounded value: the larger term is a double (at most 53 bits, no bits at or below the rounding position), so any
# non-zero sticky below it rounds back to it, as zero does. These mutants are value-equivalent and are expected to survive (EQUIV=1).
# The sticky of the ADDEND shifted out (sh part, d >= 128) matters when the product is an exact tie, and is killed.
for n in 1 2; do
	EQUIV=1 mutant "fma/sqr_add #$n: sl term of the sticky dropped (64 < d < 128)" '|| sl != 0ull) ? 1ull : 0ull;' '|| 0) ? 1ull : 0ull;' $n
	EQUIV=1 mutant "fma/sqr_add #$n: sl term of the sticky dropped (d < 64)" 'uint64_t lost = ((sl << (64u - d)) != 0ull) ? 1ull : 0ull;' 'uint64_t lost = 0ull;' $n
	mutant "fma/sqr_add #$n: addend sticky lost (d >= 128)" 'sl = ((sh | sl) != 0ull) ? 1ull : 0ull;' 'sl = 0ull;' $n
	mutant "fma/sqr_add #$n: addend sticky lost (64 < d < 128, sh part)" '(sh << (128u - d)) != 0ull ||' '0 ||' $n
done
mutant "mul: sticky of the low product lost (bit 41 set)" 'if ((lo & ((1ull << 43) - 1ull)) != 0ull)' 'if (0)'
mutant "mul: sticky of the low product lost (bit 41 clear)" 'if ((lo & ((1ull << 42) - 1ull)) != 0ull)' 'if (0)'
echo
if [ $bad -eq 0 ]; then echo "F64SOFT MUTANTS: all killed"; else echo "F64SOFT MUTANTS: $bad survived or failed"; exit 1; fi
