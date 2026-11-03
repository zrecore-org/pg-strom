#!/bin/sh
# Throwaway PostgreSQL 17 cluster for the real-scan-path measurement.
# Everything lives under $ROOT (default /tmp/pgscan); pg_teardown.sh removes it.
# Not a brew service, no TCP listener, port 54329.
set -e
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
ROOT=${ROOT:-/tmp/pgscan}
PORT=${PORT:-54329}
NROWS=${NROWS:-16777216}

[ -e "$ROOT" ] && { echo "$ROOT exists; run pg_teardown.sh first"; exit 1; }
mkdir -p "$ROOT"
"$PGBIN/initdb" -D "$ROOT/data" -U postgres --no-sync >/dev/null

cat >> "$ROOT/data/postgresql.conf" <<CONF
listen_addresses = ''
unix_socket_directories = '$ROOT'
port = $PORT
shared_buffers = 2GB
fsync = off                    # throwaway data
synchronous_commit = off
wal_level = minimal
max_wal_senders = 0
max_wal_size = 8GB
max_worker_processes = 16
max_parallel_workers = 8
jit = off
CONF

"$PGBIN/pg_ctl" -D "$ROOT/data" -l "$ROOT/server.log" -w start >/dev/null
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -v ON_ERROR_STOP=1 -q"
$PSQL -d postgres -c "CREATE DATABASE scan" 
$PSQL -d scan <<SQL
CREATE EXTENSION pg_prewarm;
CREATE EXTENSION pg_visibility;
-- same shape and distributions as the spike's synthetic rows
CREATE TABLE t (a int4, b int8, f float4, d int4);
INSERT INTO t
SELECT CASE WHEN random() < 0.05 THEN NULL ELSE (random() * 2147483647)::int4 END,
       CASE WHEN random() < 0.05 THEN NULL ELSE ((random() * 2199023255552)::int8 - 1099511627776) END,
       CASE WHEN random() < 0.05 THEN NULL ELSE (random() * 1000)::float4 END,
       CASE WHEN random() < 0.03 THEN NULL ELSE (floor(random() * 101) - 50)::int4 END
  FROM generate_series(1, $NROWS);
VACUUM (FREEZE, ANALYZE) t;
ALTER TABLE t SET (parallel_workers = 8);
CHECKPOINT;
SQL
echo "cluster ready under $ROOT (port $PORT)"
