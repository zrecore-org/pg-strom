#!/bin/sh
# Does the installed PostGIS evaluate its orientation test (lw_segment_side) with fused multiply-add, and does a
# floating-point filter (Shewchuk-style error bound) decide the same sign as PostGIS whenever it calls a point certain?
# (The A11 PostGIS design rests on the answer: no bit-exact emulation, a filter and a recheck of the doubtful rows.)
#
#   sh test/gpuserv/postgis/fma_probe.sh                 N points near one edge of a triangle (default 2,000,000)
#   N=500000 SEED=3 sh test/gpuserv/postgis/fma_probe.sh
#
# Needs the postgis extension in the PostgreSQL named by PGBIN. Throwaway cluster under test-data/fma_probe (git-ignored);
# KEEP=1 keeps it. fma_probe.c has the models: plain separate roundings, and the two ways a compiler fuses a*b - c*d.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
ROOT=${ROOT:-$REPO/test-data/fma_probe}
PORT=${PORT:-54777}
N=${N:-2000000}
SEED=${SEED:-7}
psq() { "$PGBIN/psql" -h "$ROOT" -p "$PORT" -U postgres -X -At -q "$@"; }
stop() { "$PGBIN/pg_ctl" -D "$ROOT/pg" -m fast -w stop >/dev/null 2>&1; [ -n "$KEEP" ] || rm -rf "$ROOT"; }
trap stop EXIT INT TERM

rm -rf "$ROOT"; mkdir -p "$ROOT"
cc -O0 -ffp-contract=off -o "$ROOT/probe" "$HERE/fma_probe.c" -lm || exit 2
"$PGBIN/initdb" -D "$ROOT/pg" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 2; }
cat >> "$ROOT/pg/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
fsync = off
CONF
"$PGBIN/pg_ctl" -D "$ROOT/pg" -l "$ROOT/pg.log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 2; }
psq -d postgres -c "create extension postgis" >/dev/null 2>&1 || { echo "the postgis extension is not available in $PGBIN"; exit 2; }
echo "$(psq -d postgres -c 'select postgis_version()')"

"$ROOT/probe" gen "$N" "$SEED" > "$ROOT/all.csv"
L=$(head -1 "$ROOT/all.csv")
A=$(echo "$L" | sed -E 's/# A=([^ ]*) B=([^ ]*) C=([^ ]*)/\1/')
B=$(echo "$L" | sed -E 's/# A=([^ ]*) B=([^ ]*) C=([^ ]*)/\2/')
C=$(echo "$L" | sed -E 's/# A=([^ ]*) B=([^ ]*) C=([^ ]*)/\3/')
grep -v '^#' "$ROOT/all.csv" > "$ROOT/points.csv"
psq -d postgres -c "create table probe(id int, x float8, y float8)"
psq -d postgres -c "\\copy probe from '$ROOT/points.csv' with (format csv)"
psq -d postgres -c "set extra_float_digits=3; with t as (select ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(${A%,*},${A#*,}), ST_MakePoint(${B%,*},${B#*,}), ST_MakePoint(${C%,*},${C#*,}), ST_MakePoint(${A%,*},${A#*,})])) as g) select p.id || '|' || ST_Contains(t.g, ST_MakePoint(p.x, p.y))::int from probe p, t order by p.id" > "$ROOT/res.txt"
"$ROOT/probe" cmp "$SEED" "$ROOT/points.csv" "$ROOT/res.txt"
echo "random points (the realistic case):"
"$ROOT/probe" unc 20000000 "$SEED"
