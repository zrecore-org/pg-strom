# A13: spatial join through a GiST index. Sourced by gist_dev.sh (and scan_test.sh once green); they provide q, check, sql,
# $PSQL, $GPU, $CPU, $JOINSET, $fail. Skipped when the postgis extension is not available.
# The join is "inner.geom && ST_MakePoint(outer.x, outer.y)" with a gist_geometry_ops_2d index on the inner geometry, run on
# the GPU with the index walked in the service and compared with the same query on the CPU (pg_strom.enabled=off).
echo "GiST spatial join (A13)"
if ! $PSQL -d scantest -c "create extension if not exists postgis" >/dev/null 2>&1; then
	echo "  -- the postgis extension is not available: skipped"
else
GZ=${GZ:-6000}
GP=${GP:-20000}
GISTON="set pg_strom.enable_gpugistindex=on; set enable_seqscan=on;"
# inner: shapes of every kind (a multi-level index at this size), fillfactor 60 so updates of a non-indexed column are HOT
sql "drop table if exists pg_gz; drop table if exists pg_gp"
sql "create table pg_gz (id int4, geom geometry, a int4, b int4) with (fillfactor=60)"
sql "insert into pg_gz select i,
	case i % 25
		when 0 then null
		when 1 then ST_GeomFromText('POINT EMPTY')
		when 2 then ST_GeomFromText('POLYGON EMPTY')
		when 3 then ST_SetSRID(ST_MakePoint((i * 7919) % 1000 / 10.0, (i * 104729) % 1000 / 10.0), 4326)
		when 4 then ST_MakeLine(ST_MakePoint(i % 97, i % 89), ST_MakePoint(i % 53 + 3, i % 71 + 2))
		when 5 then ST_MakePoint(1e39, -1e39)
		when 6 then ST_MakePoint((i * 31) % 1001 / 10.0, (i * 17) % 1001 / 10.0, (i % 7)::float8)
		else ST_MakeEnvelope((i * 37) % 960 / 10.0, (i * 53) % 960 / 10.0, (i * 37) % 960 / 10.0 + (i % 5 + 1) * 0.7, (i * 53) % 960 / 10.0 + (i % 3 + 1) * 0.9)
	end, i % 10, i
	from generate_series(1, $GZ) i"
sql "create index pg_gz_gist on pg_gz using gist (geom)"
sql "create table pg_gp (id int4, x float8, y float8)"
sql "insert into pg_gp select i, ((i * 7919) % 100001) / 1000.0, ((i * 104729) % 100001) / 1000.0 from generate_series(1, $GP) i"
sql "insert into pg_gp values (-4, -1e300, 1e300), (-5, null, 1), (-6, 1, null), (-7, 1e300, 0)" >/dev/null
sql "create table pg_gnf (id int4, x float8, y float8)"
sql "insert into pg_gnf values (1, 'NaN', 1), (2, 1, 'NaN'), (3, 'Infinity', 1), (4, 1, '-Infinity')" >/dev/null
sql "analyze pg_gz; analyze pg_gp"

# gistjoin: label, query. The plan must use GpuJoin and report the GiST index, and the rows must equal the CPU rows
gistjoin() {
	label=$1; qry=$2; set_extra=${3:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $set_extra explain $qry")
	pj=$(echo "$plan" | grep -c "Custom Scan (GpuJoin)"); pg=$(echo "$plan" | grep -c "GPU GiST Join")
	g=$(q "$GPU $JOINSET $GISTON $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$pj" -ge 1 ] && [ "$pg" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label (plan GpuJoin=$pj GPU GiST Join=$pg)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
JQ="select o.id, i.id from pg_gp o join pg_gz i on i.geom && st_makepoint(o.x, o.y)"
gistjoin "point against geometry boxes" "$JQ"
gistjoin "swapped operands" "select o.id, i.id from pg_gp o join pg_gz i on st_makepoint(o.x, o.y) && i.geom"
gistjoin "inner qual" "$JQ and i.a = 3"
gistjoin "outer qual" "$JQ and o.id % 4 = 0"
gistjoin "left join" "select o.id, i.id from pg_gp o left join pg_gz i on i.geom && st_makepoint(o.x, o.y)"
gistjoin "count per outer row" "select o.id, count(i.id) from pg_gp o join pg_gz i on i.geom && st_makepoint(o.x, o.y) group by o.id"
gistjoin "semi join (EXISTS)" "select o.id from pg_gp o where exists (select 1 from pg_gz i where i.geom && st_makepoint(o.x, o.y))"
gistjoin "anti join (NOT EXISTS)" "select o.id from pg_gp o where not exists (select 1 from pg_gz i where i.geom && st_makepoint(o.x, o.y))"
gistjoin "right join: inner rows without a partner" "select o.id, i.id from pg_gp o right join pg_gz i on i.geom && st_makepoint(o.x, o.y)"
# PostgreSQL itself refuses a FULL JOIN on && (not merge- or hash-joinable), so there is no CPU result to compare with: the GPU's
# FULL JOIN must have left + right - inner rows
fj=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select count(*) from pg_gp o full join pg_gz i on i.geom && st_makepoint(o.x, o.y)" 2>&1 | tail -1)
lj=$($PSQL -d scantest -c "$CPU select count(*) from pg_gp o left join pg_gz i on i.geom && st_makepoint(o.x, o.y)" | tail -1)
rj=$($PSQL -d scantest -c "$CPU select count(*) from pg_gp o right join pg_gz i on i.geom && st_makepoint(o.x, o.y)" | tail -1)
ij=$($PSQL -d scantest -c "$CPU select count(*) from pg_gp o join pg_gz i on i.geom && st_makepoint(o.x, o.y)" | tail -1)
[ "$fj" = "$((lj + rj - ij))" ]; check "full join: left + right - inner rows ($fj = $lj + $rj - $ij)" $?

# a probe point with a non-finite coordinate is an error (the CPU answers by its own rules), never a wrong result
for i in 1 2 3 4; do
	e=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select count(*) from pg_gnf o join pg_gz i on i.geom && st_makepoint(o.x, o.y) where o.id = $i" 2>&1 | tail -1)
	check "non-finite probe point $i is an error" "$(echo "$e" | grep -q 'non-finite' ; echo $?)"
done

# the index has more than one level
lv=$(sql "select (select count(*) from pg_class where relname = 'pg_gz_gist' and relpages > 2)")
check "the index has several pages" "$([ "$lv" = 1 ]; echo $?)"

# deleted rows, a HOT update (a non-indexed column; the index entry points at the chain root), a non-HOT update (the geometry)
sql "delete from pg_gz where id % 11 = 0"
sql "update pg_gz set b = b + 1 where id % 7 = 0"
sql "select pg_stat_force_next_flush()" >/dev/null
hot=$(sql "select n_tup_hot_upd from pg_stat_user_tables where relname = 'pg_gz'")
check "the update was HOT ($hot tuples)" "$([ "${hot:-0}" -gt 0 ]; echo $?)"
sql "update pg_gz set geom = ST_Translate(geom, 5, 5) where id % 13 = 0 and geom is not null"
gistjoin "after delete, HOT and non-HOT update" "$JQ"
sql "vacuum pg_gz"
gistjoin "after vacuum" "$JQ"
sql "delete from pg_gz where id % 3 = 0"
gistjoin "with dead tuples not yet vacuumed" "$JQ"

# an inner row visible only to another snapshot is not joined
$PSQL -d scantest -c "begin; delete from pg_gz where id % 2 = 0; $GPU $JOINSET $GISTON select md5(coalesce(string_agg(o.id || '/' || i.id, ',' order by o.id, i.id), '')) from pg_gp o join pg_gz i on i.geom && st_makepoint(o.x, o.y); rollback;" > /dev/null 2>&1
gistjoin "after a rolled back delete" "$JQ"

# walking the tree has a step limit; hitting it is an error, never a short result
e=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON set pg_strom.gpugist_max_steps=2; $JQ" 2>&1 | tail -1)
check "step limit raises an error" "$(echo "$e" | grep -q 'ERROR' ; echo $?)"

# inserts that split index pages while the join copies the index: in one snapshot the GPU join must still equal the CPU join (a
# "kept changing" error asks for a retry and is not a wrong answer)
sql "drop table if exists pg_gc"
sql "create table pg_gc (id int4, geom geometry)"
sql "insert into pg_gc select i, ST_MakeEnvelope(i % 90, (i * 7) % 90, i % 90 + 1.5, (i * 7) % 90 + 1.5) from generate_series(1, 3000) i"
sql "create index pg_gc_gist on pg_gc using gist (geom)"
sql "analyze pg_gc"
( for r in 1 2 3 4 5 6; do $PSQL -d scantest -c "insert into pg_gc select 100000 + i, ST_MakeEnvelope(i % 95, (i * 11) % 95, i % 95 + 1.0, (i * 11) % 95 + 1.0) from generate_series(1, 4000) i" >/dev/null 2>&1; done ) &
writer=$!
cmp_ok=0; cmp_bad=0; cmp_retry=0
for r in 1 2 3 4 5 6 7 8; do
	res=$($PSQL -d scantest -c "begin isolation level repeatable read; $GPU $JOINSET $GISTON
		select md5(coalesce(string_agg(o.id || '/' || i.id, ',' order by o.id, i.id), '')) from pg_gp o join pg_gc i on i.geom && st_makepoint(o.x, o.y) where o.id < 3000;
		set pg_strom.enabled=off;
		select md5(coalesce(string_agg(o.id || '/' || i.id, ',' order by o.id, i.id), '')) from pg_gp o join pg_gc i on i.geom && st_makepoint(o.x, o.y) where o.id < 3000; commit;" 2>&1)
	if echo "$res" | grep -q "kept changing"; then cmp_retry=$((cmp_retry+1))
	elif [ $(echo "$res" | grep -E '^[0-9a-f]{32}$' | sort -u | wc -l) -eq 1 ] && [ $(echo "$res" | grep -cE '^[0-9a-f]{32}$') -eq 2 ]; then cmp_ok=$((cmp_ok+1))
	else cmp_bad=$((cmp_bad+1)); echo "$res" | head -5; fi
done
wait $writer
[ "$cmp_bad" = 0 ] && [ $((cmp_ok + cmp_retry)) -ge 1 ]; check "joins while the index splits: GPU = CPU in one snapshot ($cmp_ok equal, $cmp_retry asked to retry, $cmp_bad different)" $?

# off by default: the plan has no GPU GiST Join
off=$($PSQL -d scantest -c "$GPU $JOINSET explain $JQ" | grep -c "GPU GiST Join")
check "pg_strom.enable_gpugistindex is off by default" "$([ "$off" = 0 ]; echo $?)"
# ---- slice 2: ST_Contains / ST_Intersects / ST_Within through the index, the function itself runs in the join program ----
# The index answers "geom && point"; the join program runs the point-in-polygon function on the inner geometry column (a string
# register into the inner buffer). A row the kernel cannot decide (a Z/M polygon, a line, a
# SRID mismatch) is joined by the backend on the CPU (poison record -> KDS_FORMAT_RECHECK), so the answer is always PostGIS's.
echo "GiST spatial join, point-in-polygon (A13 slice 2)"
sql "drop table if exists pg_g2z; drop table if exists pg_g2p"
sql "create table pg_g2z (id int4, geom geometry, a int4) with (fillfactor=70)"
sql "insert into pg_g2z select i,
	case i % 20
		when 0 then null
		when 1 then ST_GeomFromText('POLYGON EMPTY')
		when 2 then ST_MakeEnvelope(i % 50, (i * 3) % 50, i % 50 + 4, (i * 3) % 50 + 3)
		when 3 then ST_MakePolygon(ST_MakeLine(ARRAY[ST_MakePoint(i % 50, i % 47), ST_MakePoint(i % 50 + 6, i % 47), ST_MakePoint(i % 50 + 3, i % 47 + 5), ST_MakePoint(i % 50, i % 47)]))
		when 4 then ST_Difference(ST_MakeEnvelope(i % 50, i % 47, i % 50 + 8, i % 47 + 8), ST_MakeEnvelope(i % 50 + 2, i % 47 + 2, i % 50 + 6, i % 47 + 6))
		when 5 then ST_Collect(ST_MakeEnvelope(i % 50, i % 47, i % 50 + 2, i % 47 + 2), ST_MakeEnvelope(i % 50 + 4, i % 47 + 4, i % 50 + 6, i % 47 + 6))
		when 6 then ST_MakeLine(ST_MakePoint(i % 50, i % 47), ST_MakePoint(i % 50 + 5, i % 47 + 5))
		when 7 then ST_MakePoint(i % 50, i % 47)
		when 8 then ST_Force3D(ST_MakeEnvelope(i % 50, i % 47, i % 50 + 5, i % 47 + 5), 2)
		when 9 then ST_Force4D(ST_MakeEnvelope(i % 50, i % 47, i % 50 + 5, i % 47 + 5), 2, 3)
		when 10 then ST_Force3DM(ST_MakeEnvelope(i % 50, i % 47, i % 50 + 5, i % 47 + 5), 4)
		when 11 then ST_Buffer(ST_MakePoint(i % 50 + 5, i % 47 + 5), 4, 100)
		else ST_MakeEnvelope(i % 52 - 1, (i * 7) % 52 - 1, i % 52 + 1 + i % 3, (i * 7) % 52 + 1 + i % 4)
	end, i
	from generate_series(1, 3000) i"
# one polygon of 3000 vertices (about 48 kB: stored out of line / compressed; the inner preload detoasts it) and a compressible one
sql "insert into pg_g2z values (90001, ST_Buffer(ST_MakePoint(30, 30), 12, 750), 1), (90002, ST_Buffer(ST_MakePoint(20, 40), 15, 250), 2)"
sql "create index pg_g2z_gist on pg_g2z using gist (geom)"
sql "create table pg_g2p (id int4, x float8, y float8)"
# integer and half-integer points: on vertices, on edges, on the boundary of holes
sql "insert into pg_g2p select i, (i % 120) / 2.0, (i / 120) / 2.0 from generate_series(0, 14399) i"
sql "insert into pg_g2p values (-1, null, 1), (-2, 1, null), (-3, 1e300, 1e300)"
sql "analyze pg_g2z; analyze pg_g2p"
J2="select o.id, i.id from pg_g2p o join pg_g2z i on"
gistjoin "ST_Contains(inner polygon, point)" "$J2 st_contains(i.geom, st_makepoint(o.x, o.y))"
gistjoin "ST_Intersects(inner polygon, point)" "$J2 st_intersects(i.geom, st_makepoint(o.x, o.y))"
gistjoin "ST_Intersects(point, inner polygon)" "$J2 st_intersects(st_makepoint(o.x, o.y), i.geom)"
gistjoin "ST_Within(point, inner polygon)" "$J2 st_within(st_makepoint(o.x, o.y), i.geom)"
gistjoin "with an inner qual" "$J2 st_contains(i.geom, st_makepoint(o.x, o.y)) and i.a % 3 = 0"
gistjoin "with an outer qual" "$J2 st_intersects(i.geom, st_makepoint(o.x, o.y)) and o.id % 5 = 0"
gistjoin "left join" "select o.id, i.id from pg_g2p o left join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))"
gistjoin "semi join" "select o.id from pg_g2p o where exists (select 1 from pg_g2z i where st_intersects(i.geom, st_makepoint(o.x, o.y)))"
gistjoin "anti join" "select o.id from pg_g2p o where not exists (select 1 from pg_g2z i where st_contains(i.geom, st_makepoint(o.x, o.y)))"
gistjoin "count per outer row (an aggregate above the join)" "select o.id, count(i.id) from pg_g2p o join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y)) group by o.id"
gistjoin "fused 3-argument point" "$J2 st_intersects(i.geom, st_makepoint(o.x, o.y, 2))"
gistjoin "fused point with SRID 0 set explicitly" "$J2 st_contains(i.geom, st_setsrid(st_makepoint(o.x, o.y), 0))"
# RIGHT and FULL joins: the kernel's flags of matched inner rows and the backend's re-joined rows (poison) add up
gistjoin "right join" "select o.id, i.id from pg_g2p o right join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))"
gistjoin "right join, intersects" "select o.id, i.id from pg_g2p o right join pg_g2z i on st_intersects(i.geom, st_makepoint(o.x, o.y))"
# PostgreSQL cannot plan a FULL JOIN on this clause: the GPU's FULL JOIN must have left + right - inner rows
fj=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select count(*) from pg_g2p o full join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))" 2>&1 | tail -1)
lj=$($PSQL -d scantest -c "$CPU select count(*) from pg_g2p o left join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))" | tail -1)
rj=$($PSQL -d scantest -c "$CPU select count(*) from pg_g2p o right join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))" | tail -1)
ij=$($PSQL -d scantest -c "$CPU select count(*) from pg_g2p o join pg_g2z i on st_contains(i.geom, st_makepoint(o.x, o.y))" | tail -1)
[ "$fj" = "$((lj + rj - ij))" ]; check "full join: left + right - inner rows ($fj = $lj + $rj - $ij)" $?
# the outer polygon column (the wrong side) is not pushed into the join program
plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON explain select o.id, i.id from pg_g2p o join pg_g2z i on st_contains(st_buffer(st_makepoint(o.x, o.y), 1), i.geom)" 2>&1)
check "a function of the outer side as the polygon stays off the GiST join" "$(echo "$plan" | grep -q 'GPU GiST Join'; [ $? -ne 0 ]; echo $?)"
# rows the kernel hands back: the Z/M/ZM polygons, lines, points, the big polygons. Confirm that some existed in the join.
nb=$($PSQL -d scantest -c "$CPU select count(*) from pg_g2z where geom is not null and (ST_HasZ(geom) or ST_HasM(geom) or GeometryType(geom) <> 'POLYGON' or ST_NPoints(geom) > 200)" | tail -1)
check "the data has rows for the CPU path ($nb)" "$([ "${nb:-0}" -gt 0 ]; echo $?)"
# a mixed SRID is an error on the CPU; the GPU path asks the CPU, so the error is the same
sql "drop table if exists pg_g2s"
sql "create table pg_g2s (id int4, x float8, y float8)"
sql "insert into pg_g2s values (1, 10.5, 10.5)"
sql "create table if not exists pg_g2zs (id int4, geom geometry)"
sql "truncate pg_g2zs"
sql "insert into pg_g2zs values (1, ST_SetSRID(ST_MakeEnvelope(10, 10, 12, 12), 4326))"
sql "create index if not exists pg_g2zs_gist on pg_g2zs using gist (geom)"
sql "analyze pg_g2zs; analyze pg_g2s"
eg=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select o.id, i.id from pg_g2s o join pg_g2zs i on st_contains(i.geom, st_makepoint(o.x, o.y))" 2>&1 | grep -i 'mixed SRID' | head -1)
ec=$($PSQL -d scantest -c "$CPU select o.id, i.id from pg_g2s o join pg_g2zs i on st_contains(i.geom, st_makepoint(o.x, o.y))" 2>&1 | grep -i 'mixed SRID' | head -1)
check "mixed SRID: the same error on both ($eg)" "$([ -n "$eg" ] && [ "$eg" = "$ec" ]; echo $?)"
# changes of the inner table: deleted, HOT updated and moved rows
sql "delete from pg_g2z where id % 7 = 0"
sql "update pg_g2z set a = a + 1 where id % 5 = 0"
sql "update pg_g2z set geom = ST_Translate(geom, 3, 2) where id % 11 = 3 and geom is not null"
gistjoin "after delete and updates (contains)" "$J2 st_contains(i.geom, st_makepoint(o.x, o.y))"
gistjoin "after delete and updates (intersects)" "$J2 st_intersects(i.geom, st_makepoint(o.x, o.y))"
sql "vacuum pg_g2z"
gistjoin "after vacuum" "$J2 st_intersects(i.geom, st_makepoint(o.x, o.y))"

# ---- slice 3: ST_DWithin / ST_3DDWithin through the index ----
# The index answers "geom && box(point, tolerance)": the walk grows the point's box by the tolerance (a superset: widened for the
# rounding of PostGIS' distance and by one float4 step), and the join program decides the pair with the exact ST_DWithin. Inner
# values it cannot decide (a line, a polygon, a point with other SRID/dimensions rules, an empty geometry, a bad tolerance)
# go back to the backend as poison records.
echo "GiST spatial join, ST_DWithin (A13 slice 3)"
sql "drop table if exists pg_g3z; drop table if exists pg_g3p; drop table if exists pg_g3n"
sql "create table pg_g3z (id int4, geom geometry, a int4)"
sql "insert into pg_g3z select i,
	case i % 16
		when 0 then null
		when 1 then ST_GeomFromText('POINT EMPTY')
		when 8 then ST_MakePoint(i % 50, i % 47, (i % 7)::float8)
		when 9 then ST_MakePointM(i % 50, i % 47, (i % 5)::float8)
		when 10 then ST_MakePoint(i % 50, i % 47, (i % 7)::float8, (i % 5)::float8)
		when 11 then ST_MakeLine(ST_MakePoint(i % 50, i % 47), ST_MakePoint(i % 50 + 4, i % 47 + 3))
		when 12 then ST_MakeEnvelope(i % 50, i % 47, i % 50 + 3, i % 47 + 2)
		when 13 then ST_MakePoint(i % 50 + 0.1, i % 47 + 0.2)
		when 14 then ST_MakePoint(i % 50 + 0.30000000000000004, i % 47)
		else ST_MakePoint(i % 50, i % 47)
	end, i
	from generate_series(1, 3200) i"
sql "insert into pg_g3z values (90001, ST_MakePoint(1e39, 0), 1), (90002, ST_MakePoint(-1e39, 1e39), 2), (90003, ST_MakePoint(1e-300, 1e-300), 3)"
sql "create index pg_g3z_gist on pg_g3z using gist (geom)"
sql "create table pg_g3p (id int4, x float8, y float8, r float8)"
# half-integer points: distances of 0.5, 1, 1.5, 5 (3-4-5) from the grid points; tolerances 0 and exact distances
sql "insert into pg_g3p select i, (i % 100) / 2.0, (i / 100) / 2.0, (i % 11) / 2.0 from generate_series(0, 4999) i"
sql "insert into pg_g3p values (-1, null, 1, 1), (-2, 1, null, 1), (-3, 1, 1, null), (-4, 1e39, 0, 1), (-5, 3, 4, 5), (-6, 0.1, 0.1, 0.1), (-7, 0.5, 0.5, 0.70710678118654757)"
sql "analyze pg_g3z; analyze pg_g3p"
J3="select o.id, i.id from pg_g3p o join pg_g3z i on"
gistjoin "ST_DWithin(inner, point, constant)" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 1.5)"
gistjoin "ST_DWithin(point, inner, constant)" "$J3 st_dwithin(st_makepoint(o.x, o.y), i.geom, 1.5)"
gistjoin "tolerance 0" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 0)"
gistjoin "tolerance of the outer row" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r)"
gistjoin "tolerance an expression" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r * 2 + 0.25)"
gistjoin "boundary: distance equals the tolerance" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 5)"
gistjoin "with an inner qual" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 2) and i.a % 3 = 0"
gistjoin "with an outer qual" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 2) and o.id % 5 = 0"
gistjoin "fused 3-argument point (2D distance)" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y, 2), 1.5)"
gistjoin "ST_3DDWithin (fused point with Z)" "$J3 st_3ddwithin(i.geom, st_makepoint(o.x, o.y, (o.id % 5)::float8), 2.5)"
gistjoin "ST_3DDWithin, the point first" "$J3 st_3ddwithin(st_makepoint(o.x, o.y, (o.id % 5)::float8), i.geom, o.r)"
gistjoin "ST_3DDWithin, four-argument point" "$J3 st_3ddwithin(i.geom, st_makepoint(o.x, o.y, 1, 2), 2.5)"
gistjoin "left join" "select o.id, i.id from pg_g3p o left join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)"
gistjoin "right join" "select o.id, i.id from pg_g3p o right join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)"
gistjoin "semi join" "select o.id from pg_g3p o where exists (select 1 from pg_g3z i where st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75))"
gistjoin "anti join" "select o.id from pg_g3p o where not exists (select 1 from pg_g3z i where st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75))"
gistjoin "count per outer row" "select o.id, count(i.id) from pg_g3p o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 1.5) group by o.id"
fj=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select count(*) from pg_g3p o full join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)" 2>&1 | tail -1)
lj=$($PSQL -d scantest -c "$CPU select count(*) from pg_g3p o left join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)" | tail -1)
rj=$($PSQL -d scantest -c "$CPU select count(*) from pg_g3p o right join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)" | tail -1)
ij=$($PSQL -d scantest -c "$CPU select count(*) from pg_g3p o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75)" | tail -1)
[ "$fj" = "$((lj + rj - ij))" ]; check "full join: left + right - inner rows ($fj = $lj + $rj - $ij)" $?
# tolerance that is not an outer expression (the inner row's column) keeps the clause off the index
plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON explain select o.id, i.id from pg_g3p o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), i.a / 100.0)" 2>&1)
check "a tolerance of the inner side stays off the GiST join" "$(echo "$plan" | grep -q 'GPU GiST Join'; [ $? -ne 0 ]; echo $?)"
# one outer row with a tolerance the walk cannot use: NULL, NaN, infinity (PostGIS' own answer through the backend), negative (its error)
sql "create table pg_g3n (id int4, x float8, y float8, r float8)"
sql "insert into pg_g3n values (1, 10, 10, null), (2, 10, 10, 'NaN'), (3, 10, 10, 'Infinity')"
sql "analyze pg_g3n"
for id in 1 2 3; do
	gistjoin "tolerance case $id" "select o.id, i.id from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) where o.id = $id"
done
sql "insert into pg_g3n values (4, 10, 10, -1)"
sql "analyze pg_g3n"
eg=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON select o.id, i.id from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) where o.id = 4" 2>&1 | grep -i 'tolerance' | head -1)
ec=$($PSQL -d scantest -c "$CPU select o.id, i.id from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) where o.id = 4" 2>&1 | grep -i 'tolerance' | head -1)
check "negative tolerance: the same error on both ($eg)" "$([ -n "$eg" ] && [ "$eg" = "$ec" ]; echo $?)"
# the pairs the kernel hands back: lines, polygons, empty geometries, NULL tolerance rows exist in the data
nb=$($PSQL -d scantest -c "$CPU select count(*) from pg_g3z where geom is not null and (GeometryType(geom) <> 'POINT' or ST_IsEmpty(geom))" | tail -1)
check "the data has rows for the CPU path ($nb)" "$([ "${nb:-0}" -gt 0 ]; echo $?)"
sql "delete from pg_g3z where id % 9 = 0"
sql "update pg_g3z set geom = ST_Translate(geom, 1, 1) where id % 13 = 3 and geom is not null"
gistjoin "after delete and updates" "$J3 st_dwithin(i.geom, st_makepoint(o.x, o.y), 1.5)"

# ---- GpuPreAgg over a GiST join with a geometry qual (A13) ----
# The service runs a scout pass first (the join kernel only marks the outer rows it cannot decide: Z/M polygons, lines, big or
# compressed values, bad tolerances), the aggregating kernels skip those rows, and the backend joins them on the CPU and
# aggregates them into partial states that the Agg node above GpuPreAgg combines. The result must be PostGIS's, whatever the mix.
echo "GpuPreAgg over a GiST join (A13)"
AGGSET="set pg_strom.gpupreagg_max_groups=1000000;"
# gistagg: label, query [, settings]. The plan must have GpuPreAgg above a GPU GiST Join and the rows must equal the CPU rows
gistagg() {
	label=$1; qry=$2; set_extra=${3:-}
	plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $AGGSET $set_extra explain $qry")
	pa=$(echo "$plan" | grep -c "Custom Scan (GpuPreAgg)"); pg=$(echo "$plan" | grep -c "GPU GiST Join")
	g=$(q "$GPU $JOINSET $GISTON $AGGSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$pa" -ge 1 ] && [ "$pg" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label (plan GpuPreAgg=$pa GPU GiST Join=$pg)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
# gistagg_any: the same comparison, the plan need not have GpuPreAgg (the planner may not offer it for the shape; and where the join's row
# estimate equals the group estimate it sees no gain in pre-aggregating, though the data reduces a lot: use it for those shapes)
gistagg_any() {
	label=$1; qry=$2; set_extra=${3:-}
	g=$(q "$GPU $JOINSET $GISTON $AGGSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
C2="st_contains(i.geom, st_makepoint(o.x, o.y))"
A2="select count(*), count(i.a), sum(i.a), min(i.id), max(o.id) from pg_g2p o join pg_g2z i on"
gistagg "ungrouped aggregate, ST_Contains" "$A2 $C2"
gistagg "ungrouped aggregate, ST_Intersects" "$A2 st_intersects(i.geom, st_makepoint(o.x, o.y))"
gistagg "ungrouped aggregate, ST_Within (point first)" "$A2 st_within(st_makepoint(o.x, o.y), i.geom)"
gistagg "avg and float8 sum, ungrouped" "select round(avg(i.a), 6), round(sum(o.x)::numeric, 3), count(*) from pg_g2p o join pg_g2z i on $C2"
gistagg "GROUP BY an outer key" "select o.id % 7, count(*), sum(i.a), min(i.id), max(i.a) from pg_g2p o join pg_g2z i on $C2 group by o.id % 7"
gistagg "GROUP BY an inner key" "select i.a % 5, count(*), sum(o.id), min(o.id), max(o.id) from pg_g2p o join pg_g2z i on $C2 group by i.a % 5"
gistagg_any "GROUP BY two keys" "select o.id % 3, i.a % 4, count(*), sum(o.id), round(avg(i.a), 6) from pg_g2p o join pg_g2z i on $C2 group by o.id % 3, i.a % 4"
gistagg "GROUP BY an outer key, ST_Intersects" "select o.id % 7, count(*), sum(i.a) from pg_g2p o join pg_g2z i on st_intersects(i.geom, st_makepoint(o.x, o.y)) group by o.id % 7"
gistagg "GROUP BY an inner key, ST_Within" "select i.a % 5, count(*), sum(o.id) from pg_g2p o join pg_g2z i on st_within(st_makepoint(o.x, o.y), i.geom) group by i.a % 5"
gistagg_any "with an inner qual" "select o.id % 7, count(*), sum(i.a) from pg_g2p o join pg_g2z i on $C2 and i.a % 3 = 0 group by o.id % 7"
gistagg_any "with an outer qual" "select i.a % 5, count(*), sum(o.id) from pg_g2p o join pg_g2z i on $C2 where o.id % 5 = 0 group by i.a % 5"
gistagg "fused 3-argument point" "select o.id % 7, count(*), sum(i.a) from pg_g2p o join pg_g2z i on st_intersects(i.geom, st_makepoint(o.x, o.y, 2)) group by o.id % 7"
gistagg "LEFT join: unmatched outer rows counted" "select o.id % 7, count(*), count(i.id), sum(i.a) from pg_g2p o left join pg_g2z i on $C2 group by o.id % 7"
gistagg "LEFT join, ungrouped" "select count(*), count(i.id), sum(i.a), max(o.id) from pg_g2p o left join pg_g2z i on $C2"
# only the big polygons (3000 and 1000 vertices, stored compressed or out of line): per polygon counts
gistagg_any "the big polygons, GROUP BY the polygon" "select i.id, count(*), sum(o.id) from pg_g2p o join pg_g2z i on $C2 where i.id >= 90001 group by i.id"
gistagg_any "semi join (EXISTS), ungrouped" "select count(*), sum(o.id) from pg_g2p o where exists (select 1 from pg_g2z i where st_intersects(i.geom, st_makepoint(o.x, o.y)))"
gistagg_any "semi join (EXISTS), GROUP BY" "select o.id % 7, count(*) from pg_g2p o where exists (select 1 from pg_g2z i where st_contains(i.geom, st_makepoint(o.x, o.y))) group by o.id % 7"
gistagg_any "anti join (NOT EXISTS), ungrouped" "select count(*), sum(o.id) from pg_g2p o where not exists (select 1 from pg_g2z i where st_contains(i.geom, st_makepoint(o.x, o.y)))"
gistagg_any "anti join (NOT EXISTS), GROUP BY" "select o.id % 7, count(*) from pg_g2p o where not exists (select 1 from pg_g2z i where st_intersects(i.geom, st_makepoint(o.x, o.y))) group by o.id % 7"
# no outer row matches: count 0, sum NULL, no group
gistagg_any "empty result, ungrouped" "select count(*), sum(i.a), min(i.id), max(o.id) from pg_g2p o join pg_g2z i on $C2 where o.x < -5"
gistagg_any "empty result, GROUP BY" "select o.id % 7, count(*) from pg_g2p o join pg_g2z i on $C2 where o.x < -5 group by o.id % 7"
# RIGHT and FULL joins stay on Aggregate -> GpuJoin; the answers are the same
plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $AGGSET explain select count(*), count(o.id) from pg_g2p o right join pg_g2z i on $C2" 2>&1)
check "aggregate over a RIGHT join stays off GpuPreAgg" "$(echo "$plan" | grep -q 'Custom Scan (GpuPreAgg)'; [ $? -ne 0 ]; echo $?)"
gistagg_any "aggregate over a RIGHT join" "select count(*), count(o.id), sum(i.a) from pg_g2p o right join pg_g2z i on $C2"
gistagg_any "GROUP BY over a RIGHT join" "select i.a % 5, count(*), count(o.id) from pg_g2p o right join pg_g2z i on $C2 group by i.a % 5"

# outer rows that match many polygons and also a Z/M polygon or a line: the poison of such a row drops ALL its contributions on the
# GPU and the CPU adds the whole row exactly once. pg_g4z: 30 overlapping polygons (every point of 5..12 lies in all of them), a Z, an
# M and a ZM polygon, a line along y = x, a point, an empty and a NULL geometry, and far-away padding so that the index is worth
# walking. pg_g4p: 400 points on a half-integer grid (some on the line y = x, one on the inner point), a NULL row.
sql "drop table if exists pg_g4z; drop table if exists pg_g4p"
sql "create table pg_g4z (id int4, geom geometry, a int4)"
sql "insert into pg_g4z select i, ST_MakeEnvelope(i % 5, i % 3, 12 + i % 9, 12 + i % 7), i from generate_series(1, 30) i"
sql "insert into pg_g4z values
	(101, ST_Force3D(ST_MakeEnvelope(2, 2, 9, 9), 1), 101),
	(102, ST_Force3DM(ST_MakeEnvelope(8, 8, 16, 16), 2), 102),
	(103, ST_Force4D(ST_MakeEnvelope(14, 0, 19, 6), 1, 2), 103),
	(104, ST_MakeLine(ST_MakePoint(0, 0), ST_MakePoint(20, 20)), 104),
	(105, ST_MakePoint(10.5, 10.5), 105),
	(106, ST_GeomFromText('POLYGON EMPTY'), 106),
	(107, null, 107)"
sql "insert into pg_g4z select 1000 + i, ST_MakeEnvelope(100 + i % 50, 100 + i / 50, 101 + i % 50, 101 + i / 50), 1000 + i from generate_series(1, 1500) i"
sql "create index pg_g4z_gist on pg_g4z using gist (geom)"
sql "create table pg_g4p (id int4, x float8, y float8)"
sql "insert into pg_g4p select i, (i % 20) + 0.5, (i / 20) + 0.5 from generate_series(0, 399) i"
sql "insert into pg_g4p values (500, null, 1)"
sql "analyze pg_g4z; analyze pg_g4p"
C4="st_intersects(i.geom, st_makepoint(o.x, o.y))"
# the data: some outer rows hit a Z/M/line/point inner row, some do not, and some of those also hit ordinary polygons
nz=$($PSQL -d scantest -c "$CPU select count(distinct o.id) from pg_g4p o join pg_g4z i on $C4 where ST_HasZ(i.geom) or ST_HasM(i.geom) or GeometryType(i.geom) <> 'POLYGON'" | tail -1)
nn=$($PSQL -d scantest -c "$CPU select count(distinct o.id) from pg_g4p o join pg_g4z i on $C4 where not (ST_HasZ(i.geom) or ST_HasM(i.geom) or GeometryType(i.geom) <> 'POLYGON')" | tail -1)
nb=$($PSQL -d scantest -c "$CPU select count(*) from (select o.id from pg_g4p o join pg_g4z i on $C4 group by o.id having count(*) > 20 and bool_or(ST_HasZ(i.geom) or ST_HasM(i.geom) or GeometryType(i.geom) <> 'POLYGON')) x" | tail -1)
check "pg_g4: outer rows with many matches and a poisoned partner ($nb of $nz special, $nn ordinary)" "$([ "${nb:-0}" -gt 0 ] && [ "${nz:-0}" -lt 400 ] && [ "${nz:-0}" -gt 0 ]; echo $?)"
gistagg "pg_g4 ungrouped (Intersects)" "select count(*), count(i.id), sum(i.a), min(i.id), max(o.id) from pg_g4p o join pg_g4z i on $C4"
gistagg "pg_g4 ungrouped (Contains)" "select count(*), sum(i.a), min(o.id), max(o.id) from pg_g4p o join pg_g4z i on st_contains(i.geom, st_makepoint(o.x, o.y))"
gistagg_any "pg_g4 GROUP BY an outer key" "select o.id % 3, count(*), sum(i.a), min(i.id), max(i.id) from pg_g4p o join pg_g4z i on $C4 group by o.id % 3"
gistagg_any "pg_g4 GROUP BY the outer row" "select o.id, count(*), sum(i.a) from pg_g4p o join pg_g4z i on $C4 group by o.id"
gistagg_any "pg_g4 GROUP BY an inner key" "select i.a % 10, count(*), sum(o.id) from pg_g4p o join pg_g4z i on $C4 group by i.a % 10"
gistagg_any "pg_g4 GROUP BY two keys" "select o.id % 2, i.a % 3, count(*), sum(o.id) from pg_g4p o join pg_g4z i on $C4 group by o.id % 2, i.a % 3"
gistagg_any "pg_g4 LEFT join" "select o.id % 4, count(*), count(i.id), sum(i.a) from pg_g4p o left join pg_g4z i on $C4 group by o.id % 4"
gistagg_any "pg_g4 semi join" "select count(*), sum(o.id) from pg_g4p o where exists (select 1 from pg_g4z i where $C4)"
gistagg_any "pg_g4 anti join" "select count(*), sum(o.id) from pg_g4p o where not exists (select 1 from pg_g4z i where $C4)"
gistagg_any "pg_g4 RIGHT join" "select count(*), count(o.id), sum(i.a) from pg_g4p o right join pg_g4z i on $C4"

# ST_DWithin / ST_3DDWithin (slice 3 tables): inner lines, polygons and empty points go back to the backend, points are decided
D3="st_dwithin(i.geom, st_makepoint(o.x, o.y), 1.5)"
gistagg "ST_DWithin ungrouped, constant tolerance" "select count(*), sum(i.a), min(i.id), max(o.id) from pg_g3p o join pg_g3z i on $D3"
gistagg "ST_DWithin GROUP BY an outer key" "select o.id % 7, count(*), sum(i.a) from pg_g3p o join pg_g3z i on $D3 group by o.id % 7"
gistagg "ST_DWithin GROUP BY an inner key" "select i.a % 5, count(*), sum(o.id) from pg_g3p o join pg_g3z i on $D3 group by i.a % 5"
gistagg "ST_DWithin, tolerance of the outer row" "select o.id % 5, count(*), sum(i.a) from pg_g3p o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) group by o.id % 5"
gistagg "ST_DWithin, tolerance 0" "select count(*), sum(i.a), max(o.id) from pg_g3p o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0)"
gistagg_any "ST_DWithin, the point first, two keys" "select o.id % 3, i.a % 4, count(*), sum(o.id) from pg_g3p o join pg_g3z i on st_dwithin(st_makepoint(o.x, o.y), i.geom, 1.5) group by o.id % 3, i.a % 4"
gistagg "ST_3DDWithin ungrouped (fused Z)" "select count(*), sum(i.a), min(i.id) from pg_g3p o join pg_g3z i on st_3ddwithin(i.geom, st_makepoint(o.x, o.y, (o.id % 5)::float8), 2.5)"
gistagg "ST_3DDWithin GROUP BY (per-row tolerance)" "select i.a % 4, count(*), sum(o.id) from pg_g3p o join pg_g3z i on st_3ddwithin(i.geom, st_makepoint(o.x, o.y, (o.id % 5)::float8), o.r) group by i.a % 4"
gistagg "ST_3DDWithin, four-argument point (ZM)" "select o.id % 7, count(*), sum(i.a) from pg_g3p o join pg_g3z i on st_3ddwithin(i.geom, st_makepoint(o.x, o.y, 1, 2), 2.5) group by o.id % 7"
gistagg "ST_DWithin LEFT join" "select o.id % 7, count(*), count(i.id), sum(i.a) from pg_g3p o left join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75) group by o.id % 7"
gistagg_any "ST_DWithin semi join" "select count(*), sum(o.id) from pg_g3p o where exists (select 1 from pg_g3z i where st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75))"
gistagg_any "ST_DWithin anti join" "select count(*), sum(o.id) from pg_g3p o where not exists (select 1 from pg_g3z i where st_dwithin(i.geom, st_makepoint(o.x, o.y), 0.75))"
gistagg_any "ST_DWithin empty result" "select count(*), sum(i.a) from pg_g3p o join pg_g3z i on $D3 where o.x > 1000 and o.x < 1e299"
# a NULL, NaN or infinite tolerance (pg_g3n rows 1 to 3): PostGIS' own answer through the backend; a negative one raises its error
gistagg_any "NULL, NaN and infinite tolerance, ungrouped" "select count(*), count(i.id), sum(i.a) from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) where o.id <= 3"
gistagg_any "NULL, NaN and infinite tolerance, GROUP BY" "select o.id, count(*), sum(i.a) from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r) where o.id <= 3 group by o.id"
eg=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $AGGSET select count(*), sum(i.a) from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r)" 2>&1 | grep -i 'tolerance' | head -1)
ec=$($PSQL -d scantest -c "$CPU select count(*), sum(i.a) from pg_g3n o join pg_g3z i on st_dwithin(i.geom, st_makepoint(o.x, o.y), o.r)" 2>&1 | grep -i 'tolerance' | head -1)
check "negative tolerance in an aggregate: the same error on both ($eg)" "$([ -n "$eg" ] && [ "$eg" = "$ec" ]; echo $?)"
plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $AGGSET explain select count(*), count(o.id) from pg_g3p o right join pg_g3z i on $D3" 2>&1)
check "ST_DWithin aggregate over a RIGHT join stays off GpuPreAgg" "$(echo "$plan" | grep -q 'Custom Scan (GpuPreAgg)'; [ $? -ne 0 ]; echo $?)"
# GPU-Sort is not used over such a join (it needs the service to see every row)
plan=$($PSQL -d scantest -c "$GPU $JOINSET $GISTON $AGGSET set pg_strom.enable_gpusort=on; explain select o.id % 7 k, count(*) from pg_g2p o join pg_g2z i on $C2 group by o.id % 7 order by 2 desc, 1" 2>&1)
check "no GPU-Sort over a geometry GiST join" "$(echo "$plan" | grep -q 'GPU-Sort\|gpusort'; [ $? -ne 0 ]; echo $?)"
gistagg_any "GROUP BY ... ORDER BY over the join" "select k, c from (select o.id % 7 k, count(*) c from pg_g2p o join pg_g2z i on $C2 group by o.id % 7 order by 2 desc, 1) x"

fi
