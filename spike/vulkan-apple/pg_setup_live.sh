#!/bin/sh
# Throwaway cluster for the live shared_buffers test: table u(a,b,f,d) whose tuples all satisfy
# b = 3*a + d + 7 (a writer-preserved invariant that makes torn reads detectable), fillfactor 70
# so HOT updates and page pruning happen in place, autovacuum off, the whole table in shared_buffers.
set -e
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
ROOT=${ROOT:-/tmp/pgscan}
PORT=${PORT:-54329}
NROWS=${NROWS:-8388608}
HERE=$(cd "$(dirname "$0")" && pwd)

[ -e "$ROOT" ] && { echo "$ROOT exists; run pg_teardown.sh first"; exit 1; }
mkdir -p "$ROOT"
"$PGBIN/initdb" -D "$ROOT/data" -U postgres --no-sync >/dev/null
cat >> "$ROOT/data/postgresql.conf" <<CONF
listen_addresses = ''
unix_socket_directories = '$ROOT'
port = $PORT
shared_buffers = 1200MB
fsync = off
synchronous_commit = off
wal_level = minimal
max_wal_senders = 0
max_wal_size = 8GB
max_worker_processes = 16
max_parallel_workers_per_gather = 0
autovacuum = off
jit = off
max_connections = 40
CONF
"$PGBIN/pg_ctl" -D "$ROOT/data" -l "$ROOT/server.log" -w start >/dev/null
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -v ON_ERROR_STOP=1 -q"
$PSQL -d postgres -c "CREATE DATABASE scan"
$PSQL -d scan <<SQL
CREATE EXTENSION pg_prewarm;
CREATE TABLE u (a int4, b int8, f float4, d int4) WITH (fillfactor = 70, autovacuum_enabled = false);
INSERT INTO u
SELECT a, a::int8 * 3 + d + 7, f, d
  FROM (SELECT (random() * 1000000000)::int4 AS a,
               (random() * 1000)::float4      AS f,
               (floor(random() * 101) - 50)::int4 AS d
          FROM generate_series(1, $NROWS)) s;
VACUUM (FREEZE, ANALYZE) u;
SELECT pg_prewarm('u', 'buffer');
CHECKPOINT;
SQL
echo "cluster ready under $ROOT (port $PORT)"
