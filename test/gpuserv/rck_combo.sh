#!/bin/sh
# Several dev scripts under one hold of the test lock (a development helper):
#   DEV_SCRIPTS="recheck_dev.sh postgis_dev.sh" sh test/gpuserv/locked_test.sh "$PWD" rck_combo.sh
HERE=$(cd "$(dirname "$0")" && pwd)
rc=0
for s in ${DEV_SCRIPTS:-recheck_dev.sh}; do
	echo "=== $s"
	t=$(date +%s)
	sh "$HERE/$s" || rc=1
	echo "=== $s: $(( $(date +%s) - t )) s"
done
exit $rc
