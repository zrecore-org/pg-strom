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

# off by default: the plan has no GPU GiST Join
off=$($PSQL -d scantest -c "$GPU $JOINSET explain $JQ" | grep -c "GPU GiST Join")
check "pg_strom.enable_gpugistindex is off by default" "$([ "$off" = 0 ]; echo $?)"
fi
