# A12: the per-row CPU recheck on Arrow_Fdw, Parquet and GPU-Cache sources. Sourced by scan_test.sh and recheck_dev.sh (they
# provide run, run_any, runagg, runagg_any, runjoin, q, check, sql, $PSQL, $GPU, $CPU, $fail). Skipped when the postgis
# extension is not available. New tables only (rk_*).
#
# The point-in-polygon predicate hands back every point on an edge or vertex of its polygon (the grid of arrow_points.cpp is
# full of them), so each query below has the backend re-evaluate a few thousand rows of an Arrow record batch, a Parquet row
# group or a cached row; the answer must still be the CPU's, and EXPLAIN ANALYZE must show the rows in "Fallback-stats".
echo "Recheck on Arrow / Parquet / GPU-Cache sources (A12)"
if ! $PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1; then
	echo "  -- the postgis extension is not available: skipped"
else
RCKROWS=${RCKROWS:-120000}
SQ="'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry"
HOLE="'POLYGON((-8 -8,8 -8,8 8,-8 8,-8 -8),(-3 -3,3 -3,3 3,-3 3,-3 -3),(4 4,6 4,6 6,4 6,4 4))'::geometry"
MULTI="'MULTIPOLYGON(((0 0,6 0,6 6,0 6,0 0)),((3 3,9 3,9 9,3 9,3 3)),((-8 -8,-4 -8,-4 -4,-8 -4,-8 -8),(-7 -7,-5 -7,-5 -5,-7 -5,-7 -7)))'::geometry"

RARROW="$ROOT/rck_arrow"; mkdir -p "$RARROW"
ARROW_CFLAGS=$(pkg-config --cflags arrow parquet 2>/dev/null); ARROW_LIBS=$(pkg-config --libs arrow parquet 2>/dev/null)
if [ -z "$ARROW_LIBS" ] && command -v brew >/dev/null 2>&1; then
	AP=$(brew --prefix apache-arrow 2>/dev/null); ARROW_CFLAGS="-I$AP/include"; ARROW_LIBS="-L$AP/lib -lparquet -larrow -Wl,-rpath,$AP/lib"
fi
c++ -std=c++20 -O2 -DWITH_PARQUET "${0%/*}/arrow_points.cpp" -o "$RARROW/arrow_points" $ARROW_CFLAGS $ARROW_LIBS >"$RARROW/gen.log" 2>&1
check "the point-table generator builds" $?
# 3 record batches / row groups; a one-row file whose row is the vertex (10 10) of SQ (i = 1176); a file that starts at the
# vertex (0 0) (i = 336) and one that ends at (10 10), each in a single batch; a Parquet file with several row groups
"$RARROW/arrow_points" "$RARROW/rk_a.arrow" "$RCKROWS" 50000
"$RARROW/arrow_points" "$RARROW/rk_p.parquet" "$RCKROWS" 50000
"$RARROW/arrow_points" "$RARROW/rk_one.arrow" 1 1 1176
"$RARROW/arrow_points" "$RARROW/rk_first.arrow" 1000 1000 336
"$RARROW/arrow_points" "$RARROW/rk_last.arrow" 1000 1000 177
"$RARROW/arrow_points" "$RARROW/rk_pone.parquet" 1 1 1176
RFT="(id int8, x float8, y float8, z float8, m float8)"
sql "create foreign table rk_a $RFT server arrow_fdw options (file '$RARROW/rk_a.arrow')"
sql "create foreign table rk_p $RFT server arrow_fdw options (file '$RARROW/rk_p.parquet')"
sql "create foreign table rk_one $RFT server arrow_fdw options (file '$RARROW/rk_one.arrow')"
sql "create foreign table rk_first $RFT server arrow_fdw options (file '$RARROW/rk_first.arrow')"
sql "create foreign table rk_last $RFT server arrow_fdw options (file '$RARROW/rk_last.arrow')"
sql "create foreign table rk_pone $RFT server arrow_fdw options (file '$RARROW/rk_pone.parquet')"
for t in rk_a rk_p rk_one rk_first rk_last rk_pone; do sql "analyze $t"; done

# EXPLAIN ANALYZE's count of the rows the backend re-evaluated ("Fallback-stats: depth[0]=N")
rck_rows() { # table, query-with-%T -> rows re-evaluated by the backend
	r=$($PSQL -d scantest -c "$GPU explain (analyze, costs off) $(printf '%s' "$2" | sed "s/%T/$1/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
rck_check() { # table, query-with-%T: the backend must have re-evaluated at least one row
	n=$(rck_rows "$1" "$2")
	[ "$n" -gt 0 ]; check "$1: the GPU handed rows back to the backend ($n rechecked): $3" $?
}

for t in rk_a rk_p; do
	echo "table $t"
	for pl in "square:$SQ" "hole:$HOLE" "multi:$MULTI"; do
		nm=${pl%%:*}; pg=${pl#*:}
		run $t "scan: ST_Contains($nm)" "select id from %T where ST_Contains($pg, ST_MakePoint(x, y))"
		run $t "scan: ST_Covers($nm)" "select id from %T where ST_Covers($pg, ST_MakePoint(x, y))"
		run $t "scan: ST_Intersects($nm)" "select id, x + y from %T where ST_Intersects(ST_MakePoint(x, y), $pg)"
	done
	rck_check $t "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "ST_Contains(square)"
done
for t in rk_one rk_first rk_last rk_pone; do
	echo "table $t"
	run $t "scan: ST_Covers(square)" "select id from %T where ST_Covers($SQ, ST_MakePoint(x, y))"
	run $t "scan: ST_Contains(square)" "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))"
	rck_check $t "select id from %T where ST_Contains($SQ, ST_MakePoint(x, y))" "ST_Contains(square)"
done
fi
