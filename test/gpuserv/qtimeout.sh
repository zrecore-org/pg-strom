#!/bin/sh
# Runs one psql call with a time limit so that a hung query fails a test instead of stalling the whole run.
#
#   qtimeout.sh <flag file> psql ...
#
# QUERY_TIMEOUT=<seconds> (default 900: a cold Metal compile of an interpreter kernel takes minutes). On a timeout it prints
# "ERROR: query timeout after <n>s" (so the test's comparison fails) and creates the flag file; later calls then fail at once
# instead of each waiting out its own limit. Delete the flag file to run again against a kept cluster (REUSE=1).
flag=$1
shift
t=${QUERY_TIMEOUT:-900}
if [ -e "$flag" ]; then
	echo "ERROR: query skipped after an earlier timeout (delete $flag to run again)"
	exit 124
fi
perl -e 'alarm shift; exec @ARGV' "$t" "$@"
rc=$?
if [ $rc -eq 142 ]; then
	: > "$flag"
	echo "ERROR: query timeout after ${t}s"
fi
exit $rc
