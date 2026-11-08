#!/bin/sh
# A14 coverage assertion: a corpus of join and GROUP BY shapes (coverage_shapes.py) is planned with the GPU enabled and the
# cost knobs that let the GPU path win. Every shape must contain GpuJoin / GpuPreAgg unless it is listed, with a reason, in
# coverage_cpu_allowed.txt ("shape-id (glob) | reason | by-design or gap:<milestone>"); a listed shape that reaches the GPU
# fails too, so the list can only shrink. Every shape that reaches the GPU is executed and its rows (md5 of the sorted rows)
# must equal the same query with pg_strom.enabled = off. Prints the number of shapes on the GPU / allowlisted and the gap
# table (shapes blocked per reason).
#
#   sh test/gpuserv/locked_test.sh "$PWD" coverage_test.sh
#
# COVERAGE_ONLY=<regex>   only the shapes whose id matches (development); COVERAGE_ROWS=0: EXPLAIN only, no row comparison;
# ALLOWLIST=<file>        another allowlist (an empty one lists every shape that stays on the CPU, with the planner's debug
#                         message, in $ROOT/work/cpu_shapes.tsv); REUSE=1: keep the data of an earlier run in $ROOT;
# NROWS (default 100000) sizes the fact table cf, BIGROWS (1000000) the table with the many-groups keys.
HERE=$(cd "$(dirname "$0")" && pwd)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/a14_coverage}
PORT=${PORT:-54422}
NROWS=${NROWS:-100000}
BIGROWS=${BIGROWS:-1000000}
ALLOWLIST=${ALLOWLIST:-$HERE/coverage_cpu_allowed.txt}
PSQL="$HERE/qtimeout.sh $ROOT/.query_timeout $PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
PY="python3 -I $HERE/coverage_shapes.py"
fail=0
check() { # name rc
	if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
sql() { $PSQL -d covtest -c "$1"; }
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
T0=$(date +%s)
lap() { T1=$(date +%s); echo "  [$(( T1 - T0 )) s] $1"; T0=$T1; }

stop
if [ "$REUSE" = 1 ] && [ -f "$ROOT/.loaded" ]; then
	rm -f "$ROOT/.query_timeout"
else
	rm -rf "$ROOT"; mkdir -p "$ROOT"
	$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
	cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
max_worker_processes = 24
CONF
	sh "$HERE/pgconf_test.sh" >> "$ROOT/data/postgresql.conf"
fi
trap stop EXIT INT TERM
# plans and rows are checked here; the specialized kernels have their own tests (spec_dev.sh) and their background compiles
# only slow this run down
export PGSTROM_GPUSERV_SPECIALIZE=${PGSTROM_GPUSERV_SPECIALIZE:-off}
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
check "service is ready" "$([ $n -lt 40 ]; echo $?)"

if [ ! -f "$ROOT/.loaded" ]; then
$PSQL -d postgres -c "create database covtest" >/dev/null
sql "create extension pg_strom" >/dev/null
sql "create extension if not exists postgis" >/dev/null 2>&1
HAVE_POSTGIS=$($PSQL -d covtest -c "select count(*) from pg_extension where extname = 'postgis'")

# ---- heap tables. cf: the fact table (keys of every type with 2000 distinct values, group columns with 5, values with NULLs);
# cd and the other dimensions repeat the key columns; every key value of cf has exactly one partner in cd.
$PSQL -d covtest <<SQL >/dev/null
create table cf as select i::int8 as id,
	(i % 2000)::int2 as k2, (i % 2000)::int4 as k4, (i % 2000)::int8 * 1000003 as k8,
	((i % 2000) / 4.0)::float4 as kf4, ((i % 2000) / 8.0)::float8 as kf8, ((i % 2000) / 4.0)::numeric(12,2) as kn,
	('key' || (i % 2000))::text as kt, ('v' || (i % 2000))::varchar(20) as kv, ('w' || (i % 2000))::char(8) as kw,
	date '2000-01-01' + (i % 2000) as kd, timestamp '2000-01-01' + (i % 2000) * interval '1 minute' as kts,
	md5((i % 2000)::text)::uuid as ku, (i % 2 = 0) as kb,
	(i % 5)::int2 as g2, (i % 5)::int4 as g4, (i % 5)::int8 as g8, ((i % 5) / 2.0)::float4 as gf4, ((i % 5) / 2.0)::float8 as gf8,
	(i % 5)::numeric(12,2) as gn, ('g' || (i % 5))::text as gt, ('gv' || (i % 5))::varchar(20) as gv, ('gw' || (i % 5))::char(8) as gw,
	date '2020-01-01' + (i % 5) as gd, timestamp '2020-01-01' + (i % 5) * interval '1 hour' as gts, md5((i % 5)::text)::uuid as gu,
	(i % 2 = 0) as gb,
	(i % 500)::int4 as m2, (i % 300)::int4 as m3, (i % 100)::int4 as m4,
	case when i % 7 = 0 then null else (i % 64)::int2 end as v2,
	case when i % 5 = 1 and i % 3 = 0 then null else (i % 1000)::int4 end as v4,
	case when i % 11 = 0 then null else (i::int8 * 7919) % 100000 end as v8,
	case when i % 13 = 0 then null else ((i % 64) / 8.0)::float4 end as vf4,
	case when i % 17 = 0 then null else ((i % 512) / 16.0)::float8 end as vf8,
	(case when i % 19 = 0 then null else ((i % 1000) / 4.0) end)::numeric(12,2) as vn,
	case when i % 19 = 0 then null else ((i % 1000) / 4.0)::numeric end as vnu,
	case when i % 23 = 0 then null else 't' || (i % 100) end as vt,
	date '2010-01-01' + (i % 300) as vd, timestamp '2010-01-01' + (i % 300) * interval '1 hour' as vts,
	case when i % 29 = 0 then null else (i % 3 = 0) end as vb
	from generate_series(1, $NROWS) i;
create table cd as select j::int2 as k2, j::int4 as k4, j::int8 * 1000003 as k8, (j / 4.0)::float4 as kf4, (j / 8.0)::float8 as kf8,
	(j / 4.0)::numeric(12,2) as kn, ('key' || j)::text as kt, ('v' || j)::varchar(20) as kv, ('w' || j)::char(8) as kw,
	date '2000-01-01' + j as kd, timestamp '2000-01-01' + j * interval '1 minute' as kts, md5(j::text)::uuid as ku,
	(j % 7)::int4 as dg, (j * 3)::int4 as dp, 'dim' || j as dtx from generate_series(0, 1999) j;
create table cdb as select * from (values (true, 1), (false, 2)) v(kb, dp);
create table cdx as select j::int4 as k4, (j * 5)::int4 as dp from generate_series(0, 2999) j;
create table cdn as select (j * 2)::int4 as k4 from generate_series(0, 999) j;
create table cd2 as select j::int4 as m2, (j * 7)::int4 as dp2, (j % 5)::int4 as dg2 from generate_series(0, 399) j;
create table cd3 as select j::int4 as m3, (j * 11)::int4 as dp3 from generate_series(0, 249) j;
create table cd4 as select j::int4 as m4, (j * 5)::int4 as dp4 from generate_series(0, 99) j;
create table cbig as select i, (i % 5)::int4 as g5, (i % 500)::int4 as g500, (i % 50000)::int4 as g50k, i::int4 as g1m,
	(i % 1000)::int4 as v, 'k' || (i % 50000) as tg from generate_series(1, $BIGROWS) i;
create table cj as select i, jsonb_build_object('a', i % 5, 'b', 'x' || i) as j from generate_series(1, 20000) i;
create table cpart (like cf) partition by hash (id);
create table cpart_0 partition of cpart for values with (modulus 3, remainder 0);
create table cpart_1 partition of cpart for values with (modulus 3, remainder 1);
create table cpart_2 partition of cpart for values with (modulus 3, remainder 2);
insert into cpart select * from cf;
create view cfv as select * from cf where v4 > 5;
create function cov_udf(int) returns int language plpgsql immutable as 'begin return \$1 % 7; end';
-- the dimension tables of the other test scripts (the Arrow / Parquet / GPU-Cache sources join with them)
create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp,
	case when i % 9 = 0 then null else (i::int8 * 3 - 5) end as pay, repeat(chr(65 + i % 26), (i % 30)::int) as txt, i::int8 as k8
	from generate_series(1, 20000) i;
insert into sd select k, grp + 100, pay, txt, k8 from sd where k % 10 = 0;
create table sd2 as select (i - 1)::int4 as k2, (i % 5)::int4 as g2, (i * 11)::int8 as p2 from generate_series(1, 400) i;
insert into sd2 select k2, g2 + 50, p2 + 1 from sd2 where k2 % 40 = 0;
create table sd3 as select (i * 3)::int4 as k3, (i % 7)::int4 as g3 from generate_series(0, 30000) i;
create table sdn2 as select (i / 8.0)::numeric as kn, (i % 17)::int4 as grp from generate_series(0, 8000) i;
SQL
for t in cf cd cdb cdx cdn cd2 cd3 cd4 cbig cj cpart sd sd2 sd3 sdn2; do $PSQL -d covtest -c "analyze $t" >/dev/null; done
lap "heap tables"

# ---- spatial tables (the GiST join tests)
if [ "$HAVE_POSTGIS" = 1 ]; then
GZ=${GZ:-3000}; GP=${GP:-5000}
$PSQL -d covtest <<SQL >/dev/null
create table pg_gz (id int4, geom geometry, a int4, b int4) with (fillfactor=60);
insert into pg_gz select i,
	case i % 25
		when 0 then null
		when 1 then ST_GeomFromText('POINT EMPTY')
		when 3 then ST_MakePoint((i * 7919) % 1000 / 10.0, (i * 104729) % 1000 / 10.0)
		when 4 then ST_MakeLine(ST_MakePoint(i % 97, i % 89), ST_MakePoint(i % 53 + 3, i % 71 + 2))
		when 6 then ST_MakePoint((i * 31) % 1001 / 10.0, (i * 17) % 1001 / 10.0, (i % 7)::float8)
		else ST_MakeEnvelope((i * 37) % 960 / 10.0, (i * 53) % 960 / 10.0, (i * 37) % 960 / 10.0 + (i % 5 + 1) * 0.7, (i * 53) % 960 / 10.0 + (i % 3 + 1) * 0.9)
	end, i % 10, i
	from generate_series(1, $GZ) i;
create index pg_gz_gist on pg_gz using gist (geom);
create table pg_gn as select * from pg_gz;
create table pg_gp (id int4, x float8, y float8);
insert into pg_gp select i, ((i * 7919) % 100001) / 1000.0, ((i * 104729) % 100001) / 1000.0 from generate_series(1, $GP) i;
create table pg_gb as select i as id, ST_MakeEnvelope(i % 90, (i * 7) % 90, i % 90 + 2, (i * 7) % 90 + 2) as geom from generate_series(1, 300) i;
create table pg_gb2 as select i as id, ST_MakeEnvelope(i % 90, (i * 7) % 90, i % 90 + 2, (i * 7) % 90 + 2) as geom from generate_series(1, 150) i;
create table pg_gpt as select i as id, ST_MakePoint(((i * 7919) % 100001) / 1000.0, ((i * 104729) % 100001) / 1000.0) as geom from generate_series(1, 5000) i;
analyze pg_gz; analyze pg_gn; analyze pg_gp; analyze pg_gb; analyze pg_gb2; analyze pg_gpt;
SQL
fi
lap "spatial tables"

# ---- Arrow / Parquet / GPU-Cache sources
ARROWDIR="$ROOT/arrow"; mkdir -p "$ARROWDIR"
GENDIR=${COVERAGE_GEN_DIR:-$(dirname "$ROOT")/coverage_gen_cache}; mkdir -p "$GENDIR"
ARROW_CFLAGS=$(pkg-config --cflags arrow parquet 2>/dev/null); ARROW_LIBS=$(pkg-config --libs arrow parquet 2>/dev/null)
if [ -z "$ARROW_LIBS" ] && command -v brew >/dev/null 2>&1; then
	AP=$(brew --prefix apache-arrow 2>/dev/null); ARROW_CFLAGS="-I$AP/include"; ARROW_LIBS="-L$AP/lib -lparquet -larrow -Wl,-rpath,$AP/lib"
fi
for g in arrow_gen arrow_gen_ext; do
	if [ ! -x "$GENDIR/$g" ] || [ "$HERE/$g.cpp" -nt "$GENDIR/$g" ]; then
		c++ -std=c++20 -O2 -DWITH_PARQUET "$HERE/$g.cpp" -o "$GENDIR/$g" $ARROW_CFLAGS $ARROW_LIBS >"$GENDIR/$g.log" 2>&1
		check "the Arrow test data generator $g builds" $?
	fi
done
AROWS=${AROWS:-60000}
"$GENDIR/arrow_gen" "$ARROWDIR/a1.arrow" "$AROWS" 25000 1
"$GENDIR/arrow_gen" "$ARROWDIR/p1.parquet" "$AROWS" 25000 1
"$GENDIR/arrow_gen_ext" dim "$ARROWDIR/ad.arrow" 7000
"$GENDIR/arrow_gen_ext" dim "$ARROWDIR/adp.parquet" 9000
"$GENDIR/arrow_gen_ext" types "$ARROWDIR/at1.arrow" 60000 25000
FTDEF="(i int8, a int2, b int4, c int8, d float4, e bool, f text, dt date, ts timestamp)"
DDEF="(k int4, grp int4, pay int8, txt text, k8 int8)"
ATDEF="(i int8, ts_s timestamp, ts_ms timestamp, ts_us timestamp, ts_ns timestamp, d32 date, t_s time, t_ms time, t_us time, t_ns time, u8 text, lu text, fx char(5), g int4, v int8, x float8)"
sql "create foreign table a1 $FTDEF server arrow_fdw options (file '$ARROWDIR/a1.arrow')"
sql "create foreign table p1 $FTDEF server arrow_fdw options (file '$ARROWDIR/p1.parquet')"
sql "create foreign table ad $DDEF server arrow_fdw options (file '$ARROWDIR/ad.arrow')"
sql "create foreign table adp $DDEF server arrow_fdw options (file '$ARROWDIR/adp.parquet')"
sql "create foreign table at1 $ATDEF server arrow_fdw options (file '$ARROWDIR/at1.arrow')"
for t in a1 p1 ad adp at1; do sql "analyze $t" >/dev/null; done
# GPU-Cache table (the trigger is created after the load: the cache is built from the heap at its first use)
GCDEF="id int4, a int2, b int4, c int8, d float4, e bool, f text, g date, h timestamp, n numeric, x float8, v varchar(20), w char(8), k float2, u uuid"
GCNR=${GCNR:-30000}
sql "create table gc1 ($GCDEF)"
sql "insert into gc1 select i, ((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4 else (((i * 17) % 10000) / 7.0)::float4 end,
	(i % 3 = 0),
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 200)::int) end,
	date '2000-01-01' + (i % 9000),
	case when i % 13 = 0 then null else timestamp '2000-01-01' + (i % 100000) * interval '1 minute' end,
	((i % 1000) / 8.0)::numeric, i / 13.0, 'v' || (i % 50), 'w' || (i % 7), ((i % 100)::float4)::float2, md5(i::text)::uuid
	from generate_series(1, $GCNR) i"
sql "vacuum analyze gc1"
sql "create trigger gc_sync after insert or update or delete on gc1 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((GCNR * 3))')"
# the same logical table as an Arrow file, a Parquet file and a GPU-Cache (af / pf / gcf: the first 30,000 rows of cf)
make -C "$HERE/../../arrow-tools" PG_CONFIG="$PGBIN/pg_config" pg2arrow >"$GENDIR/pg2arrow.log" 2>&1
check "pg2arrow builds" $?
P2A="$HERE/../../arrow-tools/pg2arrow"
FROWS=${FROWS:-30000}
# pg2arrow has no uuid type and no numeric without a declared precision: the Arrow and Parquet files leave out ku, gu and vnu
AFSEL=$($PSQL -d covtest -c "select string_agg(attname, ', ' order by attnum) from pg_attribute where attrelid = 'cf'::regclass and attnum > 0 and attname not in ('ku', 'gu', 'vnu')")
$P2A -h "$ROOT" -p "$PORT" -u postgres -d covtest -c "select $AFSEL from cf where id <= $FROWS order by id" -s 2MB -o "$ARROWDIR/af.arrow" >"$GENDIR/pg2arrow_af.log" 2>&1
check "pg2arrow writes the Arrow file" $?
$P2A -h "$ROOT" -p "$PORT" -u postgres -d covtest -c "select $AFSEL from cf where id <= $FROWS order by id" -s 2MB -q -o "$ARROWDIR/pf.parquet" >"$GENDIR/pg2arrow_pf.log" 2>&1
check "pg2arrow writes the Parquet file" $?
CFCOLS=$($PSQL -d covtest -c "select string_agg(attname || ' ' || format_type(atttypid, atttypmod), ', ' order by attnum) from pg_attribute where attrelid = 'cf'::regclass and attnum > 0 and attname not in ('ku', 'gu', 'vnu')")
sql "create foreign table af ($CFCOLS) server arrow_fdw options (file '$ARROWDIR/af.arrow')"
sql "create foreign table pf ($CFCOLS) server arrow_fdw options (file '$ARROWDIR/pf.parquet')"
sql "analyze af"; sql "analyze pf"
sql "create table gcf as select * from cf where id <= $FROWS"
sql "vacuum analyze gcf"
sql "create trigger gc_sync after insert or update or delete on gcf for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((FROWS * 3))')"
lap "Arrow, Parquet, GPU-Cache tables"
: > "$ROOT/.loaded"
fi

if [ -n "$ADHOC" ]; then	# ADHOC=<file>: run that SQL (psql -f) against the loaded data and stop
	$PSQL -d covtest -f "$ADHOC" 2>&1
	exit 0
fi
W="$ROOT/work"; rm -rf "$W"; mkdir -p "$W"
runchunks() { # prefix: run every DIR/<prefix>_N.sql into <prefix>_N.out
	i=0
	while [ -f "$W/$1_$i.sql" ]; do
		$PSQL -d covtest -f "$W/$1_$i.sql" > "$W/$1_$i.out" 2>&1
		i=$((i+1))
	done
}
$PY emit-explain "$W" || { echo "generator failed"; exit 1; }
runchunks explain
$PY classify "$W"
lap "EXPLAIN of $(wc -l < "$W/class.tsv" | tr -d ' ') shapes: $(wc -l < "$W/gpu_ids.txt" | tr -d ' ') on the GPU"
$PY emit-diag "$W"; runchunks diag
lap "planner messages of the shapes that stay on the CPU"
if [ "${COVERAGE_ROWS:-1}" != 0 ]; then
	$PY emit-rows "$W" gpu; $PY emit-rows "$W" cpu
	runchunks rows_gpu; lap "rows on the GPU"
	runchunks rows_cpu; lap "rows on the CPU (oracle)"
fi
$PY report "$W" "$ALLOWLIST"
rc=$?
echo "  (work files: $W; the shapes that stay on the CPU with the planner's reason: $W/cpu_shapes.tsv)"
if [ $rc -ne 0 ] || [ $fail -ne 0 ]; then echo "COVERAGE: FAILED"; exit 1; fi
exit 0
