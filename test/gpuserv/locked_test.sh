#!/bin/sh
# Install a build tree into the Homebrew PostgreSQL and run a test script on it, one checkout at a time.
# The installed module ($libdir/pg_strom, referenced by the SQL scripts) and the test cluster ports are
# shared, so checkouts working in parallel (worktrees) queue here.
#
#   sh test/gpuserv/locked_test.sh <tree> [scan_test.sh|service_test.sh]      (env of the test passes through)
#   sh test/gpuserv/locked_test.sh <tree> test/vulkan_installcheck.sh          (a path relative to <tree> also works)
#
# <tree> is the repository (or worktree) root. The lock is the directory /tmp/pgstrom_test.lock; a lock
# whose owner process is gone is taken over.
TREE=${1:?usage: locked_test.sh <tree> [test script name]}
TEST=${2:-scan_test.sh}
PG_CONFIG=${PG_CONFIG:-/opt/homebrew/opt/postgresql@17/bin/pg_config}
LOCK=/tmp/pgstrom_test.lock
waited=0
while ! mkdir "$LOCK" 2>/dev/null; do
	owner=$(cat "$LOCK/pid" 2>/dev/null)
	if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
	[ $waited -eq 0 ] && echo "locked_test: waiting for the test lock held by pid ${owner:-?} ($(cat "$LOCK/tree" 2>/dev/null))"
	waited=1; sleep 5
done
echo $$ > "$LOCK/pid"; echo "$TREE" > "$LOCK/tree"
trap 'rm -rf "$LOCK"' EXIT INT TERM
BLOG=/tmp/locked_test_build.$$.log
# Resource use: the build uses TEST_JOBS processes (default 3 of the machine's 8 cores) and the test runs at a lower
# priority so that the machine stays usable; the cluster, the GPU service and the compilers it starts inherit it.
# TEST_PRIORITY=nice (default: nice -n 10), background (taskpolicy -b: also throttles disk I/O, several times slower) or full.
# TEST_FULL_SPEED=1 is the same as TEST_PRIORITY=full.
LOWPRI=
[ -n "$TEST_FULL_SPEED" ] && TEST_PRIORITY=full
case "${TEST_PRIORITY:-nice}" in
	background) command -v taskpolicy >/dev/null 2>&1 && LOWPRI="taskpolicy -b" ;;
	nice)       LOWPRI="nice -n 10" ;;
	full)       ;;
	*)          echo "locked_test: unknown TEST_PRIORITY=$TEST_PRIORITY (nice, background, full)"; exit 2 ;;
esac
if ! $LOWPRI make -C "$TREE/src" PG_CONFIG="$PG_CONFIG" -j"${TEST_JOBS:-3}" install >"$BLOG" 2>&1; then
	grep -E "error:|\*\*\*" "$BLOG" | head -20
	echo "locked_test: build/install failed (log: $BLOG)"
	exit 2
fi
rm -f /tmp/locked_test_build.$$.log
PKGLIB=$("$PG_CONFIG" --pkglibdir)
# Kernels whose SPIR-V changed with this build are compiled by the driver once, here, instead of at the first query
# of the test (PREWARM=off skips it; the service keeps the result in its pipeline cache file).
case "$TEST" in perf_bench.sh) ;; *) sh "$TREE/test/gpuserv/prewarm_kernels.sh" "$TREE" "${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv}" ;; esac
case "$TEST" in */*) SCRIPT=$TREE/$TEST;; *) SCRIPT=$TREE/test/gpuserv/$TEST;; esac

# Where the throwaway cluster goes (an explicit ROOT is used as is and never removed). The local drive is nearly
# full, so tests run on the external volume ($PGSTROM_TEST_DATA, default the repository's test-data/), except the
# disk-speed-sensitive benchmarks, which run on the local drive ($PGSTROM_BENCH_ROOT, default /tmp).
OWN_ROOT=
if [ -z "$ROOT" ]; then
	name=$(basename "$SCRIPT" .sh)
	DATA_DIR=${PGSTROM_TEST_DATA:-/Volumes/CrucialX9/Projects/pg-strom/test-data}
	case "$name" in
	perf_bench|arrow_bench) OWN_ROOT=${PGSTROM_BENCH_ROOT:-/tmp}/pgstrom_$name ;;
	*) if mkdir -p "$DATA_DIR" 2>/dev/null; then OWN_ROOT=$DATA_DIR/$name; else OWN_ROOT=/tmp/pgstrom_$name; fi ;;
	esac
	ROOT=$OWN_ROOT; export ROOT
fi
GPUSERV_BIN=${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv} $LOWPRI sh "$SCRIPT"
rc=$?

# Free the disk once the test has passed (KEEP=1, KEEP_RUNNING=1, HOLD=1 and REUSE=1 keep the cluster; a failed
# test keeps it for diagnosis, the next run of the test replaces it).
if [ -n "$OWN_ROOT" ] && [ -z "$KEEP$KEEP_RUNNING$HOLD$REUSE" ]; then
	if [ $rc -eq 0 ]; then
		[ -f "$OWN_ROOT/data/postmaster.pid" ] && "$("$PG_CONFIG" --bindir)/pg_ctl" -D "$OWN_ROOT/data" -m fast -w stop >/dev/null 2>&1
		rm -rf "$OWN_ROOT"
	else
		echo "locked_test: the test failed (exit $rc); its cluster is kept in $OWN_ROOT"
	fi
fi
exit $rc
