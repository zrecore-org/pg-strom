# R5: ST_Intersects / ST_Contains / ST_Within / ST_Covers / ST_CoveredBy of two lines / polygons / multi-geometries
# (kernels_vm/geom_pred.inc). Sourced by postgis_tests.inc (and alone by postgis_dev.sh with PRED_ONLY=1). The kernel decides a pair only where
# the orientation tests are certain; touching boundaries, a vertex on an edge, collinear overlap, mixed SRIDs, empties, big pairs are the
# backend's. The answers are PostGIS' (GEOS) own: every query is compared with pg_strom.enabled=off, and the recheck counter shows that the
# GPU decided the rows (a plain comparison also passes when every row is handed back).
echo "PostGIS R5: geometry against geometry predicates"
$PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1
if ! type recheck_rows >/dev/null 2>&1; then
recheck_rows() { # table, query -> rows re-evaluated by the backend
	r=$($PSQL -d scantest -c "$GPU explain (analyze, costs off) $(printf '%s' "$2" | sed "s/%T/$1/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
fi
PRN=${PRN:-12000}
sql "create or replace function pg_pr_shape(k int, x float8, y float8) returns geometry language sql immutable as \$f\$
	select case k % 8
		when 0 then ST_GeomFromText(format('POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))', x, y, x + 10, y, x + 10, y + 10, x, y + 10, x, y))
		when 1 then ST_GeomFromText(format('POLYGON((%s %s,%s %s,%s %s,%s %s))', x, y, x + 12, y, x, y + 9, x, y))
		when 2 then ST_GeomFromText(format('POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s),(%s %s,%s %s,%s %s,%s %s,%s %s))', x, y, x + 24, y, x + 24, y + 24, x, y + 24, x, y,
			x + 8, y + 8, x + 16, y + 8, x + 16, y + 16, x + 8, y + 16, x + 8, y + 8))
		when 3 then ST_GeomFromText(format('LINESTRING(%s %s,%s %s,%s %s)', x, y, x + 7, y + 13, x + 19, y + 4))
		when 4 then ST_GeomFromText(format('MULTIPOLYGON(((%s %s,%s %s,%s %s,%s %s,%s %s)),((%s %s,%s %s,%s %s,%s %s,%s %s)))', x, y, x + 6, y, x + 6, y + 6, x, y + 6, x, y,
			x + 8, y + 8, x + 14, y + 8, x + 14, y + 14, x + 8, y + 14, x + 8, y + 8))
		when 5 then ST_GeomFromText(format('MULTILINESTRING((%s %s,%s %s),(%s %s,%s %s))', x, y, x + 15, y + 15, x + 15, y, x, y + 15))
		when 6 then ST_GeomFromText(format('LINESTRING(%s %s,%s %s)', x, y, x + 11, y + 5))
		else ST_GeomFromText(format('LINESTRING(%s %s,%s %s,%s %s,%s %s)', x, y, x + 3, y + 22, x + 9, y + 2, x + 21, y + 21))
	end
\$f\$"
# a, b: pairs of shapes at pseudo-random places 0..60 apart (offsets are not exact binary fractions: no accidental touching); every 40th pair
# shares a vertex (the boundaries touch: the backend decides)
sql "create table pg_pr as select i id,
	pg_pr_shape(i, ((i * 7919) % 60000) / 1000.0 + 0.0137, ((i * 104729) % 60000) / 1000.0 + 0.0291) a,
	case when i % 40 = 0 then pg_pr_shape(i / 8 + 3, ((i * 7919) % 60000) / 1000.0 + 0.0137, ((i * 104729) % 60000) / 1000.0 + 0.0291)
		else pg_pr_shape(i / 8, ((i * 7919) % 60000) / 1000.0 + 0.0137 + ((i * 31) % 4001) / 100.0 - 20, ((i * 104729) % 60000) / 1000.0 + 0.0291 + ((i * 17) % 4001) / 100.0 - 20) end b,
	i % 1000 n
	from generate_series(1, $PRN) i"
sql "insert into pg_pr values (-1, NULL, 'POINT(1 1)'::geometry, 0), (-2, 'POLYGON EMPTY'::geometry, pg_pr_shape(0, 1, 1), 0), (-3, 'LINESTRING EMPTY'::geometry, pg_pr_shape(3, 1, 1), 0),
	(-5, 'MULTIPOINT((1 1),(2 2))'::geometry, pg_pr_shape(0, 1, 1), 0),
	(-6, 'GEOMETRYCOLLECTION(POLYGON((0 0,10 0,10 10,0 10,0 0)),POINT(50 50))'::geometry, pg_pr_shape(0, 1, 1), 0),
	(-7, pg_pr_shape(0, 0, 0), pg_pr_shape(0, 0, 0), 0), (-8, pg_pr_shape(0, 0, 0), pg_pr_shape(0, 10, 0), 0),
	(-9, pg_pr_shape(0, 0, 0), pg_pr_shape(0, 2, 2), 0), (-10, pg_pr_shape(2, 0, 0), pg_pr_shape(0, 10, 10), 0),
	(-11, pg_pr_shape(2, 0, 0), pg_pr_shape(0, 2, 2), 0), (-12, pg_pr_shape(2, 0, 0), pg_pr_shape(0, 9, 9), 0),
	(-13, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'LINESTRING(5 5,5 5.000000000000001)'::geometry, 0),
	(-14, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'LINESTRING(0 5,10 5)'::geometry, 0),
	(-15, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'LINESTRING(-1e30 5,1e30 5)'::geometry, 0),
	(-16, 'POLYGON((0 0,5 10,10 0,0 7,10 7,0 0))'::geometry, 'LINESTRING(5 5,5 5.5)'::geometry, 0),
	(-17, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((1e-320 1e-320,1 1e-320,1 1,1e-320 1,1e-320 1e-320))'::geometry, 0),
	(-18, ST_Buffer('POINT(0 0)'::geometry, 10, 32), ST_Buffer('POINT(3 3)'::geometry, 10, 32), 0),
	(-19, ST_Buffer('POINT(0 0)'::geometry, 10, 32), ST_Buffer('POINT(50 3)'::geometry, 5, 32), 0)"
sql "analyze pg_pr"
# Z, M and ZM versions of the same pairs (the predicates are 2D)
sql "create table pg_pr_z as select id, ST_Force3DZ(a, 5) a, ST_Force3DZ(b, 7) b, n from pg_pr"
sql "create table pg_pr_m as select id, ST_Force3DM(a, 5) a, ST_Force3DM(b, 7) b, n from pg_pr"
sql "create table pg_pr_zm as select id, ST_Force4D(a, 5, 6) a, ST_Force4D(b, 7, 8) b, n from pg_pr"
sql "analyze pg_pr_z"; sql "analyze pg_pr_m"; sql "analyze pg_pr_zm"
# a pentagram ring winds twice around its centre (the winding rule and the parity rule disagree there), a bowtie
sql "create table pg_pr_inv (id int4, a geometry, b geometry)"
sql "insert into pg_pr_inv select i, 'POLYGON((0 0,10 20,20 0,0 12,20 12,0 0))'::geometry,
	case i % 4 when 0 then ST_MakeLine(ST_MakePoint(10, 8 + i * 0.01), ST_MakePoint(10.1, 8.2)) when 1 then ST_MakeEnvelope(9.9, 8.1, 10.2 + i * 0.001, 8.3)
	when 2 then ST_MakeLine(ST_MakePoint(-5, 5), ST_MakePoint(-6, 5 + i * 0.01)) else ST_MakeLine(ST_MakePoint(1, 1), ST_MakePoint(19, 1.3 + i * 0.001)) end
	from generate_series(1, 400) i"
sql "analyze pg_pr_inv"

CP="'POLYGON((20 20,60 25,50 55,25 50,20 20))'::geometry"
CPH="'POLYGON((10 10,70 10,70 70,10 70,10 10),(30 30,50 30,50 50,30 50,30 30))'::geometry"
CL="'LINESTRING(5 5,70 66,12 40)'::geometry"
CM="'MULTIPOLYGON(((0 0,25 0,25 25,0 25,0 0)),((40 40,70 40,70 70,40 70,40 40)))'::geometry"
# mixed SRIDs: PostGIS raises, so the backend must (the kernel hands the row back); same SRID 4326 on both sides is decided
sql "create table pg_pr_srid as select id, ST_SetSRID(a, 4326) a, ST_SetSRID(b, case when id % 2 = 0 then 4326 else 0 end) b, n from pg_pr where id between 1 and 40"
run pg_pr_srid "mixed SRIDs raise the same error" "select id from %T where ST_Intersects(a, b)"
sql "create table pg_pr_srid2 as select id, ST_SetSRID(a, 4326) a, ST_SetSRID(b, 4326) b, n from pg_pr where id > 0"
run pg_pr_srid2 "ST_Intersects with the same SRID 4326" "select id from %T where ST_Intersects(a, b)"
for t in pg_pr pg_pr_z pg_pr_m pg_pr_zm; do
	echo "table $t"
	run $t "ST_Intersects(a, b)" "select id from %T where ST_Intersects(a, b)"
	run $t "ST_Intersects(b, a)" "select id from %T where ST_Intersects(b, a)"
	run $t "NOT ST_Intersects(a, b)" "select id from %T where not ST_Intersects(a, b)"
	run $t "ST_Contains(a, b)" "select id from %T where ST_Contains(a, b)"
	run $t "ST_Within(a, b)" "select id from %T where ST_Within(a, b)"
	run $t "ST_Covers(a, b)" "select id from %T where ST_Covers(a, b)"
	run $t "ST_CoveredBy(a, b)" "select id from %T where ST_CoveredBy(a, b)"
	run $t "ST_Contains(b, a) or ST_Within(b, a)" "select id from %T where ST_Contains(b, a) or ST_CoveredBy(b, a)"
	for c in "CP:$CP" "CPH:$CPH" "CL:$CL" "CM:$CM"; do
		cn=${c%%:*}; cv=${c#*:}
		run $t "ST_Intersects(a, $cn)" "select id from %T where ST_Intersects(a, $cv) and n < 800"
		run $t "ST_Contains($cn, b)" "select id from %T where ST_Contains($cv, b)"
		run $t "ST_Within(a, $cn)" "select id from %T where ST_Within(a, $cv)"
		run $t "ST_Covers(a, $cn)" "select id from %T where ST_Covers(a, $cv)"
		run $t "ST_CoveredBy(b, $cn)" "select id, n from %T where ST_CoveredBy(b, $cv) and n % 3 = 0"
	done
	runagg_any $t "count of ST_Intersects" "select count(*) from %T where ST_Intersects(a, b)"
	runagg_any $t "count and sum of ST_Contains" "select count(*), sum(n) from %T where ST_Contains(a, b)"
	runagg_any $t "count by n % 5 of ST_Intersects" "select n % 5, count(*) from %T where ST_Intersects(a, $CP) group by 1 order by 1"
	run $t "ST_Intersects and a box test" "select id from %T where a && b and ST_Intersects(a, b)"
	run $t "ST_Intersects or ST_Contains" "select id from %T where ST_Intersects(a, b) or n = 1"
done
for t in pg_pr pg_pr_zm; do
	echo "table $t (decided on the GPU)"
	nrows=$(sql "select count(*) from $t")
	for q in "ST_Intersects(a, b)" "ST_Contains(a, b)" "ST_Within(a, b)" "ST_Covers(a, b)" "ST_CoveredBy(a, b)" "ST_Intersects(a, $CP)" "ST_Contains($CPH, b)" "ST_Within(a, $CM)"; do
		n=$(recheck_rows $t "select id from %T where $q")
		[ "$n" -le $((nrows / 5)) ]; check "$t: $q: $n of $nrows rows rechecked (at most 20%)" $?
	done
	n=$(recheck_rows $t "select id from %T where ST_Intersects(a, b)")
	[ "$n" -gt 0 ]; check "$t: the touching pairs, empties and big pairs are rechecked (counter read: $n)" $?
	pl=$($PSQL -d scantest -c "$GPU explain select id from $t where ST_Intersects(a, b)" 2>&1 | grep "GPU Scan Quals" | grep -ci "st_intersects")
	[ "${pl:-0}" -ge 1 ]; check "$t: ST_Intersects of two columns is in the GPU's scan quals" $?
done
run pg_pr_inv "ST_Intersects with a pentagram ring (winding versus parity)" "select id from %T where ST_Intersects(a, b)"
run pg_pr_inv "ST_Contains with a pentagram ring" "select id from %T where ST_Contains(a, b)"
run pg_pr_inv "ST_Within with a pentagram ring" "select id from %T where ST_Within(b, a)"

# ---- R5 remainder: a point / multipoint against a line, polygon (also Z, M, ZM); R8: Z/M polygons in the point-in-polygon path ----
echo "PostGIS R5 remainder: points against lines, multipoints, Z/M polygons"
sql "create table pg_pt as select i id, a, p, mp, n from (select i, pg_pr_shape(i, ((i * 7919) % 60000) / 1000.0 + 0.0137, ((i * 104729) % 60000) / 1000.0 + 0.0291) a,
	case when i % 7 = 0 then ST_MakePoint(((i * 7919) % 60000) / 1000.0 + 0.0137, ((i * 104729) % 60000) / 1000.0 + 0.0291)
		when i % 11 = 0 then ST_MakePoint(((i * 7919) % 60000) / 1000.0 + 0.0137 + 3.5, ((i * 104729) % 60000) / 1000.0 + 0.0291 + 6.5)
		else ST_MakePoint(((i * 7919) % 60000) / 1000.0 + 0.0137 + ((i * 31) % 301) / 20.0, ((i * 104729) % 60000) / 1000.0 + 0.0291 + ((i * 17) % 301) / 20.0) end p,
	ST_Collect(ARRAY[ST_MakePoint(((i * 7919) % 60000) / 1000.0 + 1.1, ((i * 104729) % 60000) / 1000.0 + 1.3), ST_MakePoint(((i * 7919) % 60000) / 1000.0 + 5.7, ((i * 104729) % 60000) / 1000.0 + 8.9),
		ST_MakePoint(((i * 7919) % 60000) / 1000.0 + ((i * 3) % 40), ((i * 104729) % 60000) / 1000.0 + ((i * 5) % 40))]) mp,
	i % 1000 n from generate_series(1, $PRN) i) s"
sql "insert into pg_pt values (-1, NULL, 'POINT(1 1)', 'POINT(1 1)', 0), (-2, 'LINESTRING(0 0,10 10)', 'POINT EMPTY', 'MULTIPOINT((1 1),(2 2))', 0),
	(-3, 'LINESTRING(0 0,10 10)', 'POINT(5 5)', 'MULTIPOINT((1 1),(2 2))', 0), (-4, 'LINESTRING(0 0,10 10)', 'POINT(5 5.000000000000001)', 'MULTIPOINT((1 1),(20 20))', 0),
	(-5, 'POLYGON((0 0,10 0,10 10,0 10,0 0))', 'POINT(0 5)', 'MULTIPOINT((1 1),(10 10))', 0), (-6, 'POLYGON((0 0,10 0,10 10,0 10,0 0))', 'POINT(3 3)', 'MULTIPOINT((1 1),(2 2))', 0),
	(-7, 'POLYGON((0 0,10 0,10 10,0 10,0 0))', 'POINT(13 3)', 'MULTIPOINT((1 1),(12 2))', 0), (-8, 'POLYGON EMPTY', 'POINT(1 1)', 'MULTIPOINT((1 1))', 0),
	(-9, 'LINESTRING(0 0,10 0)', 'POINT(5 0)', 'MULTIPOINT EMPTY', 0), (-10, 'MULTIPOINT((1 1),(2 2))', 'POINT(1 1)', 'MULTIPOINT((1 1))', 0),
	(-11, 'POINT(1 1)', 'POINT(1 1)', 'MULTIPOINT((1 1))', 0), (-12, 'LINESTRING(0 0,0 0)', 'POINT(0 0)', 'MULTIPOINT((0 0))', 0)"
sql "analyze pg_pt"
sql "create table pg_pt_z as select id, ST_Force3DZ(a, 5) a, ST_Force3DZ(p, 2) p, ST_Force3DZ(mp, 1) mp, n from pg_pt"
sql "create table pg_pt_m as select id, ST_Force3DM(a, 5) a, ST_Force3DM(p, 2) p, ST_Force3DM(mp, 1) mp, n from pg_pt"
sql "create table pg_pt_zm as select id, ST_Force4D(a, 5, 6) a, ST_Force4D(p, 2, 3) p, ST_Force4D(mp, 1, 4) mp, n from pg_pt"
sql "analyze pg_pt_z"; sql "analyze pg_pt_m"; sql "analyze pg_pt_zm"
# polygons only (Z/M polygons against a point: the R8 case, also a fused ST_MakePoint)
sql "create table pg_pg_z as select id, a, p from pg_pt_z where ST_GeometryType(a) in ('ST_Polygon', 'ST_MultiPolygon') or id < 0"
sql "create table pg_pg_zm as select id, a, p from pg_pt_zm where ST_GeometryType(a) in ('ST_Polygon', 'ST_MultiPolygon') or id < 0"
sql "analyze pg_pg_z"; sql "analyze pg_pg_zm"
for t in pg_pt pg_pt_z pg_pt_m pg_pt_zm; do
	echo "table $t"
	run $t "ST_Intersects(a, p)" "select id from %T where ST_Intersects(a, p)"
	run $t "ST_Intersects(p, a)" "select id from %T where ST_Intersects(p, a)"
	run $t "ST_Disjoint(a, p)" "select id from %T where ST_Disjoint(a, p)"
	run $t "ST_Contains(a, p)" "select id from %T where ST_Contains(a, p)"
	run $t "ST_Within(p, a)" "select id from %T where ST_Within(p, a)"
	run $t "ST_Covers(a, p)" "select id from %T where ST_Covers(a, p)"
	run $t "ST_CoveredBy(p, a)" "select id from %T where ST_CoveredBy(p, a)"
	run $t "ST_Contains(p, a)" "select id from %T where ST_Contains(p, a)"
	run $t "ST_Intersects(a, mp)" "select id from %T where ST_Intersects(a, mp)"
	run $t "ST_Contains(a, mp)" "select id from %T where ST_Contains(a, mp)"
	run $t "ST_Covers(a, mp)" "select id from %T where ST_Covers(a, mp)"
	run $t "ST_Within(mp, a)" "select id from %T where ST_Within(mp, a)"
	run $t "ST_CoveredBy(mp, a)" "select id from %T where ST_CoveredBy(mp, a)"
	run $t "ST_Contains(mp, a)" "select id from %T where ST_Contains(mp, a)"
	run $t "ST_Disjoint(mp, a)" "select id from %T where ST_Disjoint(mp, a)"
	run $t "ST_Intersects(mp, CP)" "select id from %T where ST_Intersects(mp, $CP)"
	run $t "ST_Contains(CL, p)" "select id from %T where ST_Contains($CL, p)"
	run $t "ST_Intersects(CL, mp)" "select id from %T where ST_Intersects($CL, mp)"
	runagg_any $t "count of ST_Intersects(a, p)" "select count(*), sum(n) from %T where ST_Intersects(a, p)"
	runagg_any $t "count of ST_Disjoint(a, mp)" "select count(*) from %T where ST_Disjoint(a, mp)"
done
for t in pg_pg_z pg_pg_zm; do
	echo "table $t (Z / M polygons against a fused point)"
	run $t "ST_Contains(a, ST_MakePoint(x, y))" "select id from %T where ST_Contains(a, ST_MakePoint(ST_X(p) + 0.001, ST_Y(p) - 0.002))"
	run $t "ST_Within(ST_MakePoint, a)" "select id from %T where ST_Within(ST_MakePoint(ST_X(p) + 0.001, ST_Y(p) - 0.002), a)"
	run $t "ST_Intersects with a fused point" "select id from %T where ST_Intersects(a, ST_MakePoint(ST_X(p), ST_Y(p)))"
	run $t "ST_Disjoint with a fused point" "select id from %T where ST_Disjoint(a, ST_MakePoint(ST_X(p) + 0.001, ST_Y(p)))"
	run $t "ST_Disjoint with a fused 3D point" "select id from %T where ST_Disjoint(ST_MakePoint(ST_X(p), ST_Y(p), 3), a)"
done
for t in pg_pt pg_pt_zm; do
	echo "table $t (decided on the GPU)"
	nrows=$(sql "select count(*) from $t")
	for q in "ST_Intersects(a, p)" "ST_Contains(a, p)" "ST_Within(p, a)" "ST_Intersects(a, mp)" "ST_Contains(a, mp)" "ST_Disjoint(a, p)" "ST_Intersects($CL, mp)"; do
		n=$(recheck_rows $t "select id from %T where $q")
		[ "$n" -le $((nrows / 5)) ]; check "$t: $q: $n of $nrows rows rechecked (at most 20%)" $?
	done
done
t=pg_pg_zm
nrows=$(sql "select count(*) from $t")
n=$(recheck_rows $t "select id from %T where ST_Contains(a, ST_MakePoint(ST_X(p) + 0.001, ST_Y(p) - 0.002))")
[ "$n" -le $((nrows / 5)) ]; check "$t: Z/M polygon against a fused point: $n of $nrows rows rechecked (at most 20%)" $?

# ---- R6: ST_Disjoint, ST_Touches, ST_Crosses, ST_Overlaps, ST_Equals ----
echo "PostGIS R6: ST_Disjoint / ST_Touches / ST_Crosses / ST_Overlaps / ST_Equals"
for t in pg_pr pg_pr_z pg_pr_m pg_pr_zm; do
	echo "table $t"
	for f in ST_Disjoint ST_Touches ST_Crosses ST_Overlaps ST_Equals; do
		run $t "$f(a, b)" "select id from %T where $f(a, b)"
		run $t "NOT $f(a, b)" "select id from %T where not $f(a, b)"
		run $t "$f(b, a)" "select id from %T where $f(b, a)"
		for c in "CP:$CP" "CPH:$CPH" "CL:$CL" "CM:$CM"; do
			cn=${c%%:*}; cv=${c#*:}
			run $t "$f(a, $cn)" "select id from %T where $f(a, $cv) and n < 800"
		done
	done
	runagg_any $t "count of ST_Disjoint" "select count(*) from %T where ST_Disjoint(a, b)"
	runagg_any $t "count of ST_Crosses / ST_Overlaps" "select count(*), sum(n) from %T where ST_Crosses(a, b) or ST_Overlaps(a, b)"
	runagg_any $t "count by n % 5 of ST_Touches" "select n % 5, count(*) from %T where not ST_Touches(a, $CP) group by 1 order by 1"
	run $t "ST_Crosses and ST_Intersects" "select id from %T where ST_Intersects(a, b) and not ST_Crosses(a, b) and not ST_Disjoint(a, b)"
done
run pg_pr_srid "mixed SRIDs raise the same error (ST_Touches)" "select id from %T where ST_Touches(a, b)"
run pg_pr_srid "mixed SRIDs raise the same error (ST_Disjoint)" "select id from %T where ST_Disjoint(a, b)"
run pg_pr_srid2 "ST_Crosses with the same SRID 4326" "select id from %T where ST_Crosses(a, b)"
run pg_pr_srid2 "ST_Disjoint with the same SRID 4326" "select id from %T where ST_Disjoint(a, b)"
run pg_pr_inv "ST_Disjoint with a pentagram ring" "select id from %T where ST_Disjoint(a, b)"
run pg_pr_inv "ST_Touches with a pentagram ring" "select id from %T where ST_Touches(a, b)"
run pg_pr_inv "ST_Crosses with a pentagram ring" "select id from %T where ST_Crosses(a, b)"
# pairs of lines: crossing, collinear, shared end points, T junctions
sql "create table pg_ll as select i id, ST_MakeLine(ARRAY[ST_MakePoint(i * 0.37, 0.011), ST_MakePoint(i * 0.37 + 7.3, 9.1 + (i % 7))]) a,
	case i % 5 when 0 then ST_MakeLine(ST_MakePoint(i * 0.37 + 6, 0.5), ST_MakePoint(i * 0.37 + 1, 12.3))
		when 1 then ST_MakeLine(ST_MakePoint(i * 0.37 + 100, 0.5), ST_MakePoint(i * 0.37 + 101, 12.3))
		when 2 then ST_MakeLine(ST_MakePoint(i * 0.37, 0.011), ST_MakePoint(i * 0.37 - 7.3, 9.1))
		when 3 then ST_MakeLine(ST_MakePoint(i * 0.37 + 3.65, 4.5555), ST_MakePoint(i * 0.37 + 20, 4.5555))
		else ST_MakeLine(ARRAY[ST_MakePoint(i * 0.37 + 1, 0.011), ST_MakePoint(i * 0.37 + 3, 2.5), ST_MakePoint(i * 0.37 + 9, 3.1)]) end b, i % 1000 n
	from generate_series(1, 4000) i"
sql "insert into pg_ll values (-1, 'LINESTRING(0 0,10 0)', 'LINESTRING(5 0,15 0)', 0), (-2, 'LINESTRING(0 0,10 10)', 'LINESTRING(0 10,10 0)', 0), (-3, 'LINESTRING(0 0,10 10)', 'LINESTRING(0 0,5 5,10 10)', 0),
	(-4, 'LINESTRING(0 0,10 10)', 'LINESTRING(10 10,20 0)', 0), (-5, 'MULTILINESTRING((0 0,10 10),(0 10,10 0))', 'MULTILINESTRING((0 0,10 10),(0 10,10 0))', 0), (-6, 'LINESTRING(0 0,10 10)', 'LINESTRING(5 5,5 20)', 0),
	(-7, 'MULTILINESTRING((0 0,10 10),(20 20,30 30))', 'LINESTRING(0 10,10 0)', 0), (-8, 'LINESTRING(0 0,0 0)', 'LINESTRING(0 0,0 0)', 0)"
sql "analyze pg_ll"
for f in ST_Disjoint ST_Touches ST_Crosses ST_Overlaps ST_Equals ST_Intersects; do
	run pg_ll "$f of two lines" "select id from %T where $f(a, b)"
done
for t in pg_pr pg_pr_zm; do
	echo "table $t (R6 decided on the GPU)"
	nrows=$(sql "select count(*) from $t")
	for q in "ST_Disjoint(a, b)" "ST_Touches(a, b)" "ST_Crosses(a, b)" "ST_Overlaps(a, b)" "ST_Crosses(a, $CL)"; do
		n=$(recheck_rows $t "select id from %T where $q")
		[ "$n" -le $((nrows / 5)) ]; check "$t: $q: $n of $nrows rows rechecked (at most 20%)" $?
	done
	pl=$($PSQL -d scantest -c "$GPU explain select id from $t where ST_Crosses(a, b) and ST_Touches(a, b) is not null and ST_Equals(a, b) is not null" 2>&1 | grep "GPU Scan Quals" | grep -ci "st_crosses")
	[ "${pl:-0}" -ge 1 ]; check "$t: ST_Crosses / ST_Touches / ST_Equals are in the GPU's scan quals" $?
done
pl=$($PSQL -d scantest -c "$GPU explain select id from pg_pg_zm where ST_Contains(a, ST_MakePoint(ST_X(p) + 0.001, ST_Y(p) - 0.002)) and ST_Disjoint(a, ST_MakePoint(ST_X(p), ST_Y(p)))" 2>&1 | grep "GPU Scan Quals" | grep -ci "st_disjoint")
[ "${pl:-0}" -ge 1 ]; check "pg_pg_zm: ST_Disjoint with a fused point is in the GPU's scan quals" $?
