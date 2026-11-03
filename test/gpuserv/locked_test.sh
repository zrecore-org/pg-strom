#!/bin/sh
# Install a build tree into the Homebrew PostgreSQL and run a test script on it, one checkout at a time.
# The installed module ($libdir/pg_strom, referenced by the SQL scripts) and the test cluster ports are
# shared, so checkouts working in parallel (git worktrees) queue here.
#
#   sh test/gpuserv/locked_test.sh <tree> [scan_test.sh|service_test.sh]      (env of the test passes through)
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
if ! make -C "$TREE/src" PG_CONFIG="$PG_CONFIG" -j8 install >"$BLOG" 2>&1; then
	grep -E "error:|\*\*\*" "$BLOG" | head -20
	echo "locked_test: build/install failed (log: $BLOG)"
	exit 2
fi
rm -f /tmp/locked_test_build.$$.log
PKGLIB=$("$PG_CONFIG" --pkglibdir)
GPUSERV_BIN=${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv} sh "$TREE/test/gpuserv/$TEST"
