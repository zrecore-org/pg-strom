#!/bin/sh
# Settings that keep a throwaway test cluster light on the machine; the test scripts append this to postgresql.conf.
#
#   sh pgconf_test.sh          for the dev and scan tests
#   sh pgconf_test.sh io       only the settings that cannot change a plan (the installcheck's expected output depends on the rest)
#
# The cluster is thrown away, so durability is not needed: no fsync (the external drive was the slowest part, a checkpoint
# wrote 85,000 buffers), no WAL full-page images, rare checkpoints. Parallel workers are capped (the tests that use them ask for
# two per gather), and autovacuum is off: the tests analyze what they need, and a background vacuum changes the visibility
# map the tests rely on ("dead tuples, nothing vacuumed") and burns CPU while the GPU path is being measured.
cat <<CONF
fsync = off
synchronous_commit = off
full_page_writes = off
checkpoint_timeout = 30min
CONF
if [ "$1" != io ]; then
	cat <<CONF
max_parallel_workers = 4
max_parallel_maintenance_workers = 1
autovacuum = off
max_wal_size = 2GB
CONF
fi
