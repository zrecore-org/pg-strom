# A6: temporal types and text on the GPU. Sourced by scan_test.sh (it provides run, run_any, runagg,
# runagg_any, runjoin, q, check, sql, $PSQL, $GPU, $CPU, $fail). New tables only (ty_*).
echo "types: temporal and text (A6)"
TYROWS=${TYROWS:-300000}

# ---- data. dt: dates with NULLs and +-infinity; ts/tz: timestamps with microseconds, NULLs and +-infinity;
# tm: time of day. f: text with NULLs, empty strings, multibyte UTF-8 and values longer than 127 bytes (4-byte
# header); vc/bp: varchar and blank-padded char; big/big2: values over 2 KB, stored compressed (big) or out of
# line uncompressed (big2, STORAGE EXTERNAL), which the GPU cannot read: the backend evaluates those rows.
ty_create() { # table name
	sql "create table $1 (i int4, dt date, dt2 date, ts timestamp, tz timestamptz, tm time, f text, g text,
		vc varchar(20), bp char(8), big text, big2 text, dt3 date, ts2 timestamp, tm2 time)"
	sql "alter table $1 alter column big2 set storage external"
	sql "insert into $1 select i,
		case when i % 41 = 0 then null when i % 997 = 0 then 'infinity'::date when i % 991 = 0 then '-infinity'::date
		     else date '1990-01-01' + ((i * 37) % 12000) end,
		date '2000-01-01' + (i % 5000),
		case when i % 43 = 0 then null when i % 1009 = 0 then 'infinity'::timestamp when i % 1013 = 0 then '-infinity'::timestamp
		     else timestamp '1999-12-31 23:59:59.5' + ((i::int8 * 7919) % 500000000) * interval '1 second' + (i % 1000) * interval '1 microsecond' end,
		case when i % 47 = 0 then null when i % 1019 = 0 then 'infinity'::timestamptz when i % 1021 = 0 then '-infinity'::timestamptz
		     else timestamptz '2000-01-01 00:00:00+00' + ((i::int8 * 6007) % 500000000) * interval '1 second' + (i % 777) * interval '1 microsecond' end,
		case when i % 53 = 0 then null else time '00:00:00' + ((i * 911) % 86400) * interval '1 second' + (i % 1000000) * interval '1 microsecond' end,
		case i % 12 when 0 then null when 1 then '' when 2 then 'abc' when 3 then 'abcdef'
			when 4 then 'héllo wörld' when 5 then '日本語のテキスト' when 6 then 'ABC'
			when 7 then repeat('xy', 100) when 8 then 'abc' || i::text when 9 then 'ab'
			when 10 then repeat('z', 130) || 'abc' else 'a%b_c' || (i % 7)::text end,
		case when i % 9 = 0 then null when i % 2 = 0 then 'abc' else 'abc' || (i % 3)::text end,
		case when i % 13 = 0 then null else ('v' || (i % 50)::text)::varchar(20) end,
		case when i % 17 = 0 then null else ('b' || (i % 20)::text)::char(8) end,
		case when i % 50 = 0 then repeat('compressible', 400)
		     when i % 50 = 1 then (select string_agg(md5((i * k)::text), '') from generate_series(1, 250) k)
		     when i % 50 = 2 then 'short' when i % 50 = 3 then null else 'nb' || i::text end,
		case when i % 60 = 0 then repeat('external', 600) when i % 60 = 1 then 'short' when i % 60 = 2 then null
		     else 'ex' || i::text end,
		case when i % 31 = 0 then null when i % 97 = 0 then 'infinity'::date else date '2000-01-01' + (i % 40) end,
		case when i % 37 = 0 then null else timestamp '2010-01-01' + (i % 30) * interval '1 day' + (i % 7) * interval '1 microsecond' end,
		time '00:00:00' + (i % 24) * interval '1 hour' + (i % 3) * interval '1 microsecond'
		from generate_series(1, $TYROWS) i"
}
ty_create ty1; sql "vacuum analyze ty1"                                  # all-visible pages
ty_create ty2; sql "update ty2 set dt2 = dt2 + 1 where i % 5 = 0"; sql "delete from ty2 where i % 13 = 0"; sql "analyze ty2"
sql "create table ty_dim as select (date '1990-01-01' + i * 3)::date as d, i::int4 as k,
	(timestamp '1999-12-31 23:59:59.5' + (i * 7919 % 500000000) * interval '1 second' + (i % 1000) * interval '1 microsecond') as t,
	'dim' || i as nm from generate_series(0, 4000) i"
sql "analyze ty_dim"
n=$($PSQL -d scantest -c "select count(*) from ty1 where pg_column_size(big) < octet_length(big) or pg_column_compression(big) is not null")
[ "$n" -gt 0 ]; check "the data has compressed strings ($n rows)" $?
toid=$($PSQL -d scantest -c "select oid from pg_class where relname = 'ty1'")
n=$($PSQL -d scantest -c "select count(distinct chunk_id) from pg_toast.pg_toast_$toid")
[ "$n" -gt 0 ]; check "the data has strings stored out of line ($n toast values)" $?
ND_OK=0
$PSQL -d scantest -c "create collation case_insens (provider = icu, locale = 'und-u-ks-level2', deterministic = false)" >/dev/null 2>&1 && ND_OK=1

for t in ty1 ty2; do
	echo "table $t"
	run $t "date compare" "select i, dt from %T where dt > '2000-01-01'"
	run $t "date =, <>, <=, infinity" "select i, dt from %T where dt = '2010-06-15' or dt <> '1995-03-01' and dt <= '1990-01-10' or dt = 'infinity' or dt = '-infinity'"
	run $t "timestamp range" "select i, ts from %T where ts >= '2015-01-01 00:00:00' and ts < '2016-01-01'"
	run $t "timestamp microseconds and infinity" "select i, ts from %T where ts > '2000-01-02 03:04:05.000123' and ts <> 'infinity' and ts <= 'infinity'"
	run $t "timestamptz compare" "select i, tz from %T where tz > '2008-01-01 00:00:00+00' and tz <= '2012-06-01 12:00:00+00'"
	run $t "time between" "select i, tm from %T where tm between '10:00' and '12:30:15.5'"
	run $t "IS NULL / IS NOT NULL on temporal columns" "select i, dt, ts, tz, tm from %T where dt is null or ts is not null and tm is null"
	run $t "date vs timestamp (columns)" "select i, dt, ts from %T where dt < ts or dt = ts"
	run $t "timestamp vs date (constant)" "select i from %T where ts >= date '2015-01-01' and ts < date '2015-02-01'"
	run $t "date vs timestamp constant" "select i from %T where dt > timestamp '2000-01-01 12:00:00' and dt <> timestamp 'infinity'"
	run $t "date + integer" "select i, dt2 + 10, 7 + dt2, dt2 - 3 from %T where dt2 + 10 > '2005-01-01'"
	run $t "date - date" "select i, dt2 - date '2000-01-01', dt2 - dt2 from %T where i % 7 = 0"
	run $t "timestamp::date and date::timestamp" "select i, ts::date, dt2::timestamp from %T where ts::date = dt2 or dt2::timestamp > ts"
	run $t "computed temporal values in the projection" "select i, ts::date, dt2 + 1 from %T where ts::date > '2010-01-01' and ts < 'infinity'"
	run $t "temporal pass-through with other columns" "select dt, ts, tz, tm, f from %T where i % 1000 = 5"
	run $t "temporal and integer quals together" "select i, dt, tm from %T where i > 500 and dt > '2005-05-05' and tm < '12:00' and i % 3 = 0"
	echo "  -- text"
	run $t "text = constant" "select i, f from %T where f = 'abc'"
	run $t "text <> constant" "select i, f from %T where f <> 'abc'"
	run $t "text = multibyte constant" "select i from %T where f = 'héllo wörld' or f = '日本語のテキスト'"
	run $t "text = empty string" "select i from %T where f = ''"
	run $t "text = long constant (4-byte header)" "select i from %T where f = repeat('xy', 100) or f = repeat('z', 130) || 'abc'"
	run $t "text = text (two columns)" "select i, f, g from %T where f = g"
	run $t "NOT (text = ...) and NULLs" "select i from %T where not (f = 'abc') or g = 'abc1'"
	run $t "LIKE prefix" "select i, f from %T where f like 'ab%'"
	run $t "LIKE suffix" "select i from %T where f like '%bc'"
	run $t "LIKE substring" "select i, f from %T where f like '%bc%'"
	run $t "LIKE exact and match-all" "select i from %T where f like 'abc' or f like '%' and i < 20"
	run $t "NOT LIKE" "select i from %T where f not like 'a%' and f not like '%z%'"
	run $t "LIKE with an escaped %" "select i, f from %T where f like 'a\\%b%'"
	run $t "LIKE on multibyte text" "select i from %T where f like 'hé%' or f like '%語の%' or f like '日本%'"
	run $t "LIKE on a long value" "select i from %T where f like '%zabc' or f like 'xyxyxy%'"
	run $t "length() of ASCII, multibyte and long values" "select i, length(f) from %T where length(f) = 5 or length(f) > 100 or length(f) = 8"
	run $t "length() in the projection" "select i, length(f) + 1 from %T where f like '%o%' "
	run $t "varchar equality and LIKE" "select i, vc from %T where vc = 'v7' or vc like 'v1%'"
	run $t "char(n) ignores trailing blanks" "select i, bp from %T where bp = 'b7' or bp = 'b13   ' or bp like 'b1%'"
	run $t "char(n) ordering in the C collation" "select i from %T where bp collate \"C\" < 'b5' and bp collate \"C\" >= 'b1'"
	run $t "text ordering in the C collation" "select i from %T where f collate \"C\" < 'abd' and f collate \"C\" >= 'ab'"
	run $t "text ordering in the C collation, <= and >" "select i from %T where f collate \"C\" <= 'abc' and f collate \"C\" > 'a'"
	run $t "text qual with a qual on another column" "select i, a_b from (select i, f as a_b, dt from %T) x where a_b like 'abc%' and dt > '2000-01-01'"
	run $t "text and temporal pass-through" "select i, f, ts from %T where f = 'abc' and ts > '2010-01-01'"
	run $t "compressed or external value: LIKE on big" "select i, length(big) from %T where big like 'compress%' or big like 'nb1%'"
	run $t "compressed or external value: equality on big" "select i from %T where big = 'short' or big = 'nb101'"
	run $t "external value: equality on big2" "select i from %T where big2 = 'short' or big2 = 'ex121' or big2 like 'external%'"
	run $t "external value: length(big2)" "select i, length(big2) from %T where length(big2) > 3000"
	run $t "toasted and plain strings in one qual" "select i, f from %T where (big = 'short' or f = 'abc') and big2 is not null"
	run $t "IS NULL on toasted columns reads no value" "select i from %T where big is null or big2 is null"
	echo "  -- stays on the CPU"
	run_any $t "text ordering in the default collation" "select i from %T where f < 'abd' and f >= 'ab'"
	run_any $t "LIKE with _" "select i from %T where f like 'a_c%'"
	run_any $t "LIKE with a % between literals" "select i from %T where f like 'a%c%'"
	run_any $t "ILIKE" "select i from %T where f ilike 'AB%'"
	run_any $t "regular expression" "select i from %T where f ~ '^ab'"
	run_any $t "timestamptz vs date (time zone)" "select i from %T where tz > date '2010-01-01' and tz < timestamp '2012-01-01 00:00:00'"
	run_any $t "timestamp + interval" "select i, ts + interval '1 day', dt + interval '1 month' from %T where i < 100"
	run_any $t "upper(text) in the projection" "select i, upper(f) from %T where f = 'abc'"
	run_any $t "IN list of text" "select i from %T where f in ('abc', 'ab')"
	[ $ND_OK = 1 ] && run_any $t "text and a non-deterministic collation" "select i from %T where f collate case_insens = 'ABC'"
	run_any $t "length() in a GROUP BY expression" "select length(f), count(*) from %T group by 1 order by 1"
done
pre="set plan_cache_mode=force_generic_plan; prepare pq(text, int) as select md5(coalesce(string_agg(x::text, E'\\n' order by x::text), '')), count(*) from (select i, f from ty1 where f = \$1 and i < \$2) x; execute pq('abc', 50000);"
g=$($PSQL -d scantest -c "$GPU $pre" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $pre" 2>&1 | tail -1)
plan=$($PSQL -d scantest -c "$GPU set plan_cache_mode=force_generic_plan; prepare pe(text, int) as select i, f from ty1 where f = \$1 and i < \$2; explain execute pe('abc', 50000)" 2>&1 | grep -c "Custom Scan (GpuScan)")
[ "$g" = "$c" ] && [ "$plan" -ge 1 ] && ! echo "$g" | grep -q ERROR; check "ty1: text query parameter (generic plan) (GpuScan=$plan)" $?

echo "types: aggregation and GROUP BY"
for t in ty1 ty2; do
	runagg $t "min/max/count of temporal columns" "select count(*), count(dt), count(ts), min(dt), max(dt), min(ts), max(ts), min(tz), max(tz), min(tm), max(tm) from %T"
	runagg $t "temporal aggregates with quals" "select min(dt), max(dt2), min(ts), max(tm) from %T where dt > '2000-01-01' and tm > '06:00'"
	runagg $t "aggregates with a text qual" "select count(*), sum(i), min(ts), max(dt2) from %T where f like 'ab%' and f <> 'ab'"
	runagg $t "aggregates with a toasted text qual" "select count(*), sum(i) from %T where big like 'compress%' or big2 = 'ex121'"
	runagg $t "count(text) reads only the NULLs" "select count(f), count(big), count(big2), count(*) from %T where i % 2 = 0"
	runagg $t "text qual and FILTER" "select count(*) filter (where f = 'abc'), max(i) filter (where length(f) > 100) from %T"
	runagg $t "empty input" "select count(*), min(dt), max(ts), min(tm) from %T where dt > '2100-01-01'"
	runagg $t "GROUP BY date" "select dt3, count(*), sum(i), max(dt2) from %T group by 1 order by 1"
	runagg $t "GROUP BY date with NULLs and infinity" "select dt3, count(*), min(dt) from %T where i > 0 group by 1 order by 1"
	runagg $t "GROUP BY timestamp::date (64-bit min/max overflow to the CPU)" "select ts2::date as d, count(*), min(ts2), max(ts2) from %T where f like '%bc%' group by 1 order by 1"
	runagg $t "GROUP BY time of day" "select tm2, count(*), max(i) from %T group by 1 order by 1"
	runagg $t "GROUP BY timestamp and min/max" "select ts2, count(*), min(tz), max(dt3) from %T group by 1 order by 1"
	runagg $t "GROUP BY two temporal keys" "select dt3, tm2, count(*), sum(i) from %T group by 1, 2 order by 1, 2"
	runagg $t "GROUP BY with a text qual" "select dt3, count(*), sum(i) from %T where f like '%bc%' group by 1 order by 1"
	runagg $t "GROUP BY with a toasted text qual" "select tm2, count(*) from %T where big like 'nb%' or big like 'compress%' or big2 like 'ex%' or big2 = 'short' group by 1 order by 1"
	runagg $t "GROUP BY date + integer" "select dt3 + 3 as k, count(*) from %T where i > 5 group by 1 order by 1"
	runagg $t "GROUP BY a high-cardinality date (overflow to the CPU)" "select dt, count(*) from %T where dt > '1990-01-01' group by 1 order by 1"
done
runagg_any ty1 "text key" "select f, count(*) from %T group by 1 order by 1"

echo "types: joins on temporal keys"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
JOINPAT="Custom Scan (GpuJoin)"
for t in ty1 ty2; do
	ANYPLAN=0
	runjoin $t "date key" "select t.i, d.k, d.nm from %T t join ty_dim d on t.dt = d.d"
	runjoin $t "timestamp key" "select t.i, d.k from %T t join ty_dim d on t.ts = d.t"
	runjoin $t "date key and temporal pass-through of both sides" "select t.i, t.dt, d.d, d.t from %T t join ty_dim d on t.dt = d.d where t.i < 60000"
	runjoin $t "LEFT JOIN on a date key" "select t.i, d.k from %T t left join ty_dim d on t.dt = d.d where t.i < 30000"
	runjoin $t "date key with a date computed on the GPU" "select t.i, t.dt2 + d.k as x, d.d - t.dt2 as y from %T t join ty_dim d on t.dt = d.d where t.dt2 < '2008-01-01' and d.d < 'infinity'"
	ANYPLAN=1
	runjoin $t "a text qual keeps the outer scan of a join on the CPU" "select t.i, d.k from %T t join ty_dim d on t.dt = d.d where t.f = 'abc'"
	runjoin $t "text key" "select t.i, d.k from %T t join ty_dim d on t.f = d.nm"
done
ANYPLAN=0

echo "types: errors"
for t in ty1; do
	g=$($PSQL -d scantest -c "$GPU select dt2 + 2147483647 from $t where dt2 is not null" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select dt2 + 2147483647 from $t where dt2 is not null" 2>&1 | head -1)
	echo "$g" | grep -q "date out of range" && echo "$c" | grep -q "date out of range"; check "$t: date + integer overflow is reported alike ($g)" $?
	g=$($PSQL -d scantest -c "$GPU select dt - dt2 from $t where dt is not null" 2>&1 | head -1)
	c=$($PSQL -d scantest -c "$CPU select dt - dt2 from $t where dt is not null" 2>&1 | head -1)
	echo "$g" | grep -q "infinite dates" && echo "$c" | grep -q "infinite dates"; check "$t: subtracting an infinite date is reported alike ($g)" $?
	run $t "far dates are valid" "select i, dt2 + 100000000, dt2 - 100000000 + 100000000 from %T where dt2 < '2020-01-01' and i % 10 = 0"
done
# the strings the GPU cannot read were evaluated by the backend
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from ty1 where big like 'compress%' or big2 = 'ex121'" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -ge 1 ]; check "toasted strings are re-evaluated by the backend (debug message seen $msgs times)" $?
msgs=$($PSQL -d scantest -c "$GPU set client_min_messages=debug1; select count(*) from ty1 where f = 'abc' or dt > '2000-01-01'" 2>&1 | grep -c "CPU re-evaluation")
[ "$msgs" -eq 0 ]; check "plain strings are never handed back to the backend" $?
