#!/bin/sh
# Throwaway cluster with pg_strom preloaded, for the A0 (macOS port) checks.
# Everything lives under $ROOT (default /tmp/pgstrom_a0); pg_teardown.sh ROOT=... removes it.
set -e
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
ROOT=${ROOT:-/tmp/pgstrom_a0}
PORT=${PORT:-54331}
[ -e "$ROOT" ] && { echo "$ROOT exists; run: ROOT=$ROOT ./pg_teardown.sh"; exit 1; }
mkdir -p "$ROOT"
"$PGBIN/initdb" -D "$ROOT/data" -U postgres --no-sync >/dev/null
cat >> "$ROOT/data/postgresql.conf" <<CONF
listen_addresses = ''
unix_socket_directories = '$ROOT'
port = $PORT
shared_buffers = 256MB
shared_preload_libraries = 'pg_strom'
max_worker_processes = 16
fsync = off
CONF
"$PGBIN/pg_ctl" -D "$ROOT/data" -l "$ROOT/server.log" -w start
