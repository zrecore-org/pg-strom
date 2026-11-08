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
LOCK=${PGSTROM_TEST_LOCK:-/tmp/pgstrom_test.lock}
# Fair order: a waiter first takes a ticket (the directory $LOCK.queue/<number>, made atomically with mkdir, the number
# one above the highest present) and only tries for the lock when its ticket is the lowest of the live ones (a ticket
# whose process is gone is removed). Without it the waiter that happens to poll right after a release wins, and one
# agent can wait for hours behind others' repeated runs. The lock itself is the same directory as before, so scripts
# that do not queue still exclude each other and this one.
QUEUE=$LOCK.queue
mkdir -p "$QUEUE" 2>/dev/null
seq=$(ls "$QUEUE" 2>/dev/null | tail -1 | sed 's/^0*//'); seq=$(( ${seq:-0} + 1 ))
while :; do
	TICKET=$QUEUE/$(printf %010d "$seq")
	mkdir "$TICKET" 2>/dev/null && break
	seq=$((seq + 1))
done
echo $$ > "$TICKET/pid"; echo "$TREE $TEST" > "$TICKET/tree"
trap 'rm -rf "$TICKET"' EXIT INT TERM
waited=0
while :; do
	ahead=0
	for t in $(ls "$QUEUE" 2>/dev/null); do
		[ "$QUEUE/$t" = "$TICKET" ] && break
		tp=$(cat "$QUEUE/$t/pid" 2>/dev/null)
		if [ -n "$tp" ] && ! kill -0 "$tp" 2>/dev/null; then rm -rf "$QUEUE/$t"; continue; fi
		ahead=$((ahead + 1))
	done
	[ $ahead -eq 0 ] && break
	[ $waited -eq 0 ] && echo "locked_test: $ahead run(s) ahead in the queue ($(cat "$LOCK/tree" 2>/dev/null))"
	waited=1; sleep 5
done
waited=0
while ! mkdir "$LOCK" 2>/dev/null; do
	owner=$(cat "$LOCK/pid" 2>/dev/null)
	if [ -n "$owner" ] && ! kill -0 "$owner" 2>/dev/null; then rm -rf "$LOCK"; continue; fi
	[ $waited -eq 0 ] && echo "locked_test: waiting for the test lock held by pid ${owner:-?} ($(cat "$LOCK/tree" 2>/dev/null))"
	waited=1; sleep 5
done
echo $$ > "$LOCK/pid"; echo "$TREE" > "$LOCK/tree"
rm -rf "$TICKET"
trap 'rm -rf "$LOCK"' EXIT INT TERM
# LOCKED_TEST_LOCK_ONLY=<seconds>: take the lock, say so, hold it and leave (test of the queue itself)
if [ -n "$LOCKED_TEST_LOCK_ONLY" ]; then echo "locked_test: got the lock ($TEST)"; sleep "$LOCKED_TEST_LOCK_ONLY"; exit 0; fi
# A tree that is behind vulkan-gpu compiles kernels nobody else has compiled (every distinct SPIR-V is a cold compile of
# up to 5 minutes, at the front of the run, under the lock): merging first lets the checkouts share the compiled kernels.
if git -C "$TREE" rev-parse --verify -q vulkan-gpu >/dev/null 2>&1; then
	behind=$(git -C "$TREE" rev-list --count HEAD..vulkan-gpu 2>/dev/null)
	kdiff=$(git -C "$TREE" diff --name-only HEAD...vulkan-gpu -- src/gpuserv/kernels_vm 2>/dev/null | wc -l | tr -d ' ')
	[ "${behind:-0}" -gt 0 ] && echo "locked_test: note: $TREE is $behind commit(s) behind vulkan-gpu ($kdiff kernel source file(s) changed there); merge it to share compiled kernels"
fi
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
# The first start of a freshly installed binary of 450 MB is slow (about 5 s on an idle machine, much more on a busy one:
# the system checks the executable); the postmaster gives the start-up probe 60 s and a test waits 40 s for the service.
# Start it once here, outside those limits.
"${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv}" --probe >/dev/null 2>&1
# Kernels whose SPIR-V changed with this build are compiled by the driver once, here, instead of at the first query
# of the test (PREWARM=off skips it; the service keeps the result in its pipeline cache file).
case "$TEST" in perf_bench.sh) ;; *) PREWARM_FOR=$(basename "$TEST") sh "$TREE/test/gpuserv/prewarm_kernels.sh" "$TREE" "${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv}" ;; esac
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
