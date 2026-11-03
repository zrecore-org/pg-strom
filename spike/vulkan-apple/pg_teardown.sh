#!/bin/sh
# Stop the throwaway cluster and delete everything under $ROOT.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
ROOT=${ROOT:-/tmp/pgscan}
[ -d "$ROOT/data" ] && "$PGBIN/pg_ctl" -D "$ROOT/data" -m fast -w stop
rm -rf "$ROOT"
echo "removed $ROOT"
