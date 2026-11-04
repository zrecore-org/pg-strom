# A11: the register-pressure ladder for the fused-point instructions (GEOMDWITHIN, GEOMPIP). The program builder keeps the x and y
# registers of every fused point alive while it evaluates the other operands; a program it cannot build has no CPU fallback, so the
# planner's estimate must refuse such a qual itself. Every rung must either run on the GPU and match the CPU, or plan on the CPU
# (run_any compares the answers and fails on any ERROR). Sourced by postgis_tests.inc (inside its postgis-available branch).
LT1="abs(x) / 4 + 0.5"
LT2="abs(x * 0.5 + y * 0.25) + 0.5"
LT3="sqrt(abs(x * 0.5 + y * 0.25)) + 0.5"
LT4="sqrt(abs(x * 0.5 + y * 0.25) + x * x) + 0.5"
LT5="round(sqrt(abs(x * 0.5 + y * 0.25) + x * x) * 4) / 4 + ceil(abs(y)) * 0.1"
i=0
for tol in "$LT1" "$LT2" "$LT3" "$LT4" "$LT5"; do
	i=$((i+1))
	run_any pg_1 "ladder $i: ST_DWithin(two computed fused points, computed tolerance)" "select id from %T where ST_DWithin(ST_MakePoint(x + 1, y + 1), ST_MakePoint(y - 1, x - 1), $tol)"
	run_any pg_1 "ladder $i: ST_DWithin(computed fused point, constant, computed tolerance)" "select id from %T where ST_DWithin(ST_MakePoint(x * 0.5 + y, y * 0.5 - x), 'POINT(5 5)'::geometry, $tol)"
	run_any pg_1 "ladder $i: ST_DWithin(deep fused point, deep fused point, computed tolerance)" "select id from %T where ST_DWithin(ST_MakePoint(sqrt(abs(x) + 1) * 2, abs(y * 0.5 + x)), ST_MakePoint(ceil(x) + floor(y), round(x * y) / 8), $tol)"
	run_any pg_1 "ladder $i: ST_DWithin with SRIDs and a point column" "select id from %T where id % 20 <> 1 and ST_DWithin(ST_SetSRID(ST_MakePoint(x + 1, y + 1), 0), g, $tol)"
	run_any pg_1 "ladder $i: ST_DWithin next to a point-in-polygon with a computed fused point" "select id from %T where ST_Contains($SQ, ST_MakePoint(x + 1, y + 1)) and ST_DWithin(ST_MakePoint(x + 1, y + 1), ST_MakePoint(y - 1, x - 1), $tol)"
done
run_any pg_1 "ladder: ST_Contains(computed fused point), depth 2" "select id from %T where ST_Contains($SQ, ST_MakePoint(x * 0.5 + y, y * 0.5 - x))"
run_any pg_1 "ladder: ST_Contains(computed fused point), depth 4" "select id from %T where ST_Contains($HOLE, ST_MakePoint(sqrt(abs(x * 0.5 + y * 0.25) + x * x), round(sqrt(abs(y) + 1) * 4) / 4))"
run_any pg_1 "ladder: ST_Contains(deep fused point) and ST_Contains(deep fused point)" "select id from %T where ST_Contains($SQ, ST_MakePoint(abs(x) + 0.5, abs(y) + 0.5)) or ST_Contains($HOLE, ST_MakePoint(sqrt(abs(x)) - 1, sqrt(abs(y)) - 1))"
run_any pg_1 "ladder: two point-in-polygon tests and ST_DWithin together" "select id from %T where ST_Contains($SQ, ST_MakePoint(x + 1, y + 1)) and ST_Intersects(ST_MakePoint(y - 1, x - 1), $HOLE) and ST_DWithin(ST_MakePoint(x + 1, y + 1), ST_MakePoint(y - 1, x - 1), 6)"
