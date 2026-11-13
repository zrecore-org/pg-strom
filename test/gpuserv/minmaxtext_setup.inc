# Tables of the min / max of text tests (A6b): tmo (many string shapes), tm1, tm0 and the Arrow / Parquet / GPU-Cache copies tmoa / tmop / tmog.
# Sourced by minmaxtext_dev.sh and scan_test.sh (needs sql, check, PSQL, PGBIN, ROOT, PORT, NROWS).  MMROWS sizes tmo (default NROWS).
# tmo: strings of length 0..30 (empty strings, one group that has only '' and NULL, one group that is all NULL), multi-byte values whose
# UTF-8 byte order differs from their code point order of a locale, equal values (ties), trailing blanks, a compressible 5 kB value and an
# out-of-line (STORAGE EXTERNAL) one. char(10) columns pad every value to the same length, so ties are identical.
sql "create table tmo as select i,
	(i % 12)::int4 as g, (i % 3000)::int4 as gm, (i % 4)::int4 as g2,
	case when i % 12 = 11 then null when i % 12 = 10 then (case when i % 5 = 0 then '' end)
	     when i % 23 = 0 then null else repeat(chr(97 + i % 26), (i % 31)::int) end as t,
	case when i % 12 = 11 then null when i % 29 = 0 then null when i % 6 = 0 then repeat(chr(955 + i % 5), (i % 12)::int + 1)
	     when i % 7 = 0 then repeat(chr(8364), (i % 3)::int + 1) else repeat(chr(65 + i % 26), (i % 9)::int) end::varchar(40) as v,
	case when i % 12 = 11 then null when i % 19 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) end::char(10) as w,
	case when i % 12 = 11 then null when i % 19 = 0 then null else repeat(chr(97 + i % 7), (i % 11)::int) || repeat(' ', (i % 3)::int) end::bpchar as wn,
	case when i % 40 = 0 then repeat(chr(97 + i % 4), 5000 + (i % 3)) end as big,
	case when i % 40 = 0 then repeat(md5((i % 6)::text), 160) end as ext,
	(i % 50)::int4 as n, i::int8 as pay from generate_series(1, ${MMROWS:-$NROWS}) i"
sql "alter table tmo alter column ext set storage external"
sql "update tmo set ext = ext where i % 1 = 0"
sql "vacuum analyze tmo"
# one row, and an empty table
sql "create table tm1 as select 'only'::text as t, 'one'::varchar(10) as v, 'x'::char(4) as w, 1::int4 as g"
sql "create table tm0 (t text, v varchar(10), w char(4), g int4)"
sql "analyze tm1"; sql "analyze tm0"

# the same strings as an Arrow file, a Parquet file and a GPU-Cache (tmoa / tmop / tmog: the first 30000 rows of tmo)
KDIR=$ROOT/arrow; mkdir -p "$KDIR"
make -C "${0%/*}/../../arrow-tools" PG_CONFIG="$PGBIN/pg_config" pg2arrow >"$ROOT/pg2arrow.log" 2>&1
check "pg2arrow builds" $?
P2A="${0%/*}/../../arrow-tools/pg2arrow"
KCOLS="i int4, g int4, gm int4, g2 int4, t text, v varchar(40), w char(10), n int4, pay int8"
KSEL="i, g, gm, g2, t, v, w, n, pay"
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tmo where i <= 30000 order by i" -s 2MB -o "$KDIR/tmo.arrow" >"$ROOT/p2a1.log" 2>&1
check "pg2arrow writes the Arrow file" $?
$P2A -h "$ROOT" -p "$PORT" -u postgres -d scantest -c "select $KSEL from tmo where i <= 30000 order by i" -s 2MB -q -o "$KDIR/tmo.parquet" >"$ROOT/p2a2.log" 2>&1
check "pg2arrow writes the Parquet file" $?
sql "create foreign table tmoa ($KCOLS) server arrow_fdw options (file '$KDIR/tmo.arrow')"
sql "create foreign table tmop ($KCOLS) server arrow_fdw options (file '$KDIR/tmo.parquet')"
sql "create table tmog as select $KSEL from tmo where i <= 30000"
sql "vacuum analyze tmog"
sql "create trigger gc_sync after insert or update or delete on tmog for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=90000')"
sql "analyze tmoa"; sql "analyze tmop"
