# Arrow_Fdw / Parquet as the outer (scanned) relation of GpuPreAgg and GpuJoin (A5 part 2), and the
# value types the Arrow loader reads (dates, times, timestamps of every unit, strings). Sourced by
# scan_test.sh after its Arrow section (and by arrow_dev.sh): it needs their helpers (q, GPU, CPU, check,
# sql, PSQL) and tables (a1 a2 a3 p1: Arrow IPC with several record batches, two files, a Parquet file;
# the heap tables sd, sd2, sd3, whose data the Arrow dimension tables ad and ad2 repeat).
#
# Every query runs on the GPU and with pg_strom.enabled = off; the md5 of the sorted result rows must be
# equal. The plan must show the offloaded shape (GpuPreAgg / GpuJoin over the Arrow table), except in the
# "stays on the CPU" checks.
echo "arrow: aggregation and joins over Arrow / Parquet"
ARROWDIR="${ARROWDIR:-$ROOT/arrow}"; mkdir -p "$ARROWDIR"
c++ -std=c++20 -O2 -DWITH_PARQUET "${0%/*}/arrow_gen_ext.cpp" -o "$ARROWDIR/arrow_gen_ext" $ARROW_CFLAGS $ARROW_LIBS >"$ARROWDIR/gen_ext.log" 2>&1
check "the extra Arrow test data generator builds" $?
"$ARROWDIR/arrow_gen_ext" dim "$ARROWDIR/ad.arrow" 7000                 # 3 record batches
"$ARROWDIR/arrow_gen_ext" dim "$ARROWDIR/adp.parquet" 9000              # a Parquet dimension table
"$ARROWDIR/arrow_gen_ext" dim2 "$ARROWDIR/ad2.arrow" 300
"$ARROWDIR/arrow_gen_ext" types "$ARROWDIR/at1.arrow" 120000 50000      # 3 batches, the last one short
"$ARROWDIR/arrow_gen_ext" types "$ARROWDIR/at2.arrow" 1500 1500         # one small batch
sql "create foreign table ad (k int4, grp int4, pay int8, txt text, k8 int8) server arrow_fdw options (file '$ARROWDIR/ad.arrow')"
sql "create foreign table adp (k int4, grp int4, pay int8, txt text, k8 int8) server arrow_fdw options (file '$ARROWDIR/adp.parquet')"
sql "create foreign table ad2 (k2 int4, g2 int4, p2 int8) server arrow_fdw options (file '$ARROWDIR/ad2.arrow')"
ATDEF="(i int8, ts_s timestamp, ts_ms timestamp, ts_us timestamp, ts_ns timestamp, d32 date, t_s time, t_ms time, t_us time, t_ns time, u8 text, lu text, fx char(5), g int4, v int8, x float8)"
sql "create foreign table at1 $ATDEF server arrow_fdw options (file '$ARROWDIR/at1.arrow')"
sql "create foreign table at2 $ATDEF server arrow_fdw options (file '$ARROWDIR/at2.arrow')"
sql "analyze ad"; sql "analyze adp"; sql "analyze ad2"; sql "analyze at1"; sql "analyze at2"
# heap dimension tables with date and timestamp keys that match the Arrow columns' values
sql "create table dd as select (date '1970-01-01' + (10000 + n)) as d, (n % 7)::int4 as w from generate_series(0, 19999, 3) n"
sql "create table dts as select (timestamp '1970-01-01' + (1500000000 + i * 37) * interval '1 second') as t, (i % 9)::int4 as w from generate_series(1, 3000) i"
sql "analyze dd"; sql "analyze dts"

[ -n "$ARROW_SETUP_ONLY" ] && return 0
AGGSET="set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"
AJOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
AJOINSET2="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
APAR="set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
# the placeholders: %T the Arrow table, %D the dimension (sd heap or ad Arrow), %E sd2 / ad2, %F sd3
adim=sd; adim2=sd2
__achk() { # table label query settings pattern(s of the plan: all must be present) ... compared with the CPU
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/$adim/g; s/%E/$adim2/g; s/%F/sd3/g"); set_extra=$4; shift 4
	pl=$($PSQL -d scantest -c "$GPU $set_extra explain $qry" 2>&1)
	ok=1
	# the one-batch tables of a few hundred rows are too small for the planner to choose the GPU: results only
	if [ "$ARROWANY" != 1 ] && [ "$tbl" != a2 ] && [ "$tbl" != at2 ]; then
		for pat in "$@"; do echo "$pl" | grep -q "$pat" || ok=0; done
	fi
	g=$(q "$GPU $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$ok" = 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan patterns: $*; matched=$ok)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
aagg() { # table label query [settings]: GpuPreAgg directly over the Arrow table (no GpuScan below an Aggregate)
	__achk "$1" "$2" "$3" "$AGGSET ${4:-}" "Custom Scan (GpuPreAgg)"
}
aagg_any() { # the plan may stay on the CPU (the planner's group estimate for an expression key): results equal
	ARROWANY=1; aagg "$@"; ARROWANY=0
}
aagg_cpu() { # the shape stays on the CPU (an Aggregate above a GpuScan or nothing): results equal
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/$adim/g; s/%E/$adim2/g")
	g=$(q "$GPU $AGGSET ${4:-}" "$qry"); c=$(q "$CPU" "$qry")
	pl=$($PSQL -d scantest -c "$GPU $AGGSET ${4:-} explain $qry" 2>&1 | grep -c "Custom Scan (GpuPreAgg)")
	if [ "$pl" -eq 0 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows, stays on the CPU)"
	else echo "  FAIL  $tbl: $label (GpuPreAgg=$pl)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
ajagg_any() {
	ARROWANY=1; ajagg "$@"; ARROWANY=0
}
ajoin() { # table label query [settings]: GpuJoin whose outer side is the Arrow table
	__achk "$1" "$2" "$3" "$AJOINSET ${4:-}" "Custom Scan (GpuJoin)"
}
ajoin2() { # stacked joins ("Join Quals [2]")
	__achk "$1" "$2" "$3" "$AJOINSET2 ${4:-}" "Custom Scan (GpuJoin)" 'Join Quals \[2\]'
}
ajagg() { # aggregation over a join: GpuPreAgg listing the join
	__achk "$1" "$2" "$3" "$AJOINSET2 set pg_strom.gpupreagg_max_groups=1000000; ${4:-}" "Custom Scan (GpuPreAgg)" 'Join Quals \['
}

for t in a1 a2 a3 p1; do
	echo "table $t: aggregation (GpuPreAgg over the Arrow table)"
	aagg $t "count, sum, min, max, avg of int4 with NULLs" "select count(*), count(b), sum(b), min(b), max(b), avg(b) from %T"
	aagg $t "int2 and int8 columns" "select sum(a), min(a), max(a), avg(a), sum(c), min(c), max(c), avg(c) from %T"
	aagg $t "with quals" "select count(*), sum(a), max(c) from %T where a > 0 and b is not null"
	aagg_any $t "empty input" "select count(*), count(b), sum(b), min(b), max(c), avg(a) from %T where a > 100000"
	aagg $t "only NULL values" "select count(b), sum(b), min(b), max(b) from %T where b is null"
	aagg $t "expressions as arguments" "select sum(a + b), max(b * 2), min(c - 1), count(a * 2) from %T"
	aagg $t "FILTER (WHERE ...)" "select count(*) filter (where b > 0), sum(b) filter (where a < 0), max(c) filter (where e) from %T"
	aagg $t "sum beyond 64 bits" "select sum(c + 9000000000000000000), avg(c + 9000000000000000000), sum(c) from %T"
	aagg $t "count of date, timestamp and text columns (NULL-ness)" "select count(dt), count(ts), count(f), count(*) filter (where f is null) from %T where a > 0"
	aagg $t "text qual" "select count(*), sum(a) from %T where f = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"
	aagg $t "text LIKE qual" "select count(*), max(b) from %T where f like 'abc%' or f like '%zz'"
	aagg $t "date qual" "select count(*), sum(a), min(c) from %T where dt > '2000-01-01'::date and dt <= '2010-06-30'::date"
	aagg $t "timestamp qual" "select count(*), sum(b) from %T where ts >= '2017-07-15 00:00:00'::timestamp and ts < '2017-07-16 00:00:00'::timestamp"
	aagg $t "9 aggregates (the 12-aggregate kernel)" "select count(*), count(b), sum(a), sum(c), min(b), max(a), avg(c), sum(b), max(c) from %T"
	echo "table $t: GROUP BY"
	aagg $t "one int key" "select a % 10 as k, count(*), sum(b), min(c), max(b), avg(a) from %T group by 1 order by 1"
	aagg $t "key with NULLs" "select b % 5 as k, count(*), count(b), sum(a) from %T group by 1 order by 1"
	aagg $t "two integer keys" "select a % 4 as k1, b % 3 as k2, count(*), sum(c), max(a) from %T group by 1, 2 order by 1, 2"
	aagg_any $t "int8 key" "select c % 7 as k, count(*), sum(a) from %T group by 1 order by 1"
	aagg $t "with quals and FILTER" "select a % 6 as k, count(*) filter (where b > 0), sum(b) filter (where e), max(c) from %T where a > -9000 group by 1 order by 1"
	aagg_any $t "empty input" "select a % 6 as k, count(*), sum(b) from %T where a > 100000 group by 1 order by 1"
	aagg $t "sum beyond 64 bits per group" "select a % 3 as k, sum(c + 9000000000000000000), avg(c + 9000000000000000000) from %T group by 1 order by 1"
	aagg $t "more groups than a workgroup's table (overflow list)" "select i % 1500 as k, count(*), sum(a), min(b), max(c) from %T group by 1 order by 1"
	aagg $t "overflow with two keys" "select i % 97 as k1, i % 13 as k2, count(*), sum(a) from %T group by 1, 2 order by 1, 2"
	aagg_any $t "many groups: the session-wide table (a2 and at2 are too small to switch)" "select i % 9000 as k, count(*), sum(a), min(b), max(c), sum(c) from %T group by 1 order by 1"
	aagg_any $t "three keys (session-wide table)" "select a % 7 as k1, b % 5 as k2, i % 3 as k3, count(*), sum(c), max(a) from %T group by 1, 2, 3 order by 1, 2, 3"
	aagg_any $t "four keys with NULLs, many groups" "select a % 7 as k1, b % 11 as k2, i % 13 as k3, c % 17 as k4, count(*), sum(a) from %T group by 1, 2, 3, 4 order by 1, 2, 3, 4"
	aagg $t "date key" "select dt, count(*), sum(a) from %T where dt is not null group by 1 order by 1 limit 100" "set pg_strom.gpupreagg_max_groups=100000000;"
	aagg_any $t "date key (with the NULL group), two keys" "select dt, a % 2 as k2, count(*), max(b) from %T where a > 9990 group by 1, 2 order by 1, 2"
	aagg_any $t "timestamp key" "select ts, count(*) from %T where a > 9995 group by 1 order by 1"
	aagg_cpu $t "text key stays on the CPU" "select f, count(*) from %T group by f order by 1 limit 20"
	aagg $t "sum and min of a float4 column cast to float8 (exact in a double)" "select count(*), sum(d::float8), min(d::float8), max(d::float8) from %T where a > 9000"
	aagg $t "sum(float4): every addition rounded to float4 (A10)" "select count(*), sum(d) from %T where a > 9000"
done
aagg a1 "parallel workers" "select count(*), sum(b), min(c), max(a), avg(b) from %T where a > -5000" "$APAR"
aagg a1 "GROUP BY, parallel workers" "select a % 10 as k, count(*), sum(b), max(c) from %T group by 1 order by 1" "$APAR"
aagg p1 "GROUP BY, parallel workers (Parquet)" "select a % 10 as k, count(*), sum(b), max(c) from %T group by 1 order by 1" "$APAR"
# every process fetches its own partial result (the leader does not scan: the workers' rows must not be lost)
aagg p1 "parallel workers only, no GROUP BY (Parquet)" "select count(*), sum(a), min(c) from %T" "$APAR set parallel_leader_participation=off;"
aagg a1 "parallel workers only, no GROUP BY" "select count(*), sum(a), min(c) from %T" "$APAR set parallel_leader_participation=off;"
aagg p1 "parallel workers only, GROUP BY (Parquet)" "select a % 3 as k, count(*), sum(b) from %T group by 1 order by 1" "$APAR set parallel_leader_participation=off;"

for t in a1 a2 a3 p1; do
	echo "table $t: joins with the Arrow table as the outer relation"
	adim=sd; adim2=sd2
	ajoin $t "pass-through columns of both sides" "select t.i, t.a, d.grp, d.pay from %T t join %D d on t.b = d.k where t.i < 60000"
	ajoin $t "text, date and timestamp of the Arrow side" "select t.i, t.f, t.dt, t.ts, d.grp from %T t join %D d on t.b = d.k where t.i < 60000"
	ajoin $t "int8 key" "select t.i, d.grp from %T t join %D d on t.c % 20000 = d.k8 where t.i < 80000"
	ajoin $t "quals on both sides" "select t.i, d.grp, d.pay from %T t join %D d on t.b = d.k where t.a > 0 and d.grp < 8 and d.pay is not null"
	ajoin $t "computed projection mixing both sides" "select t.i, t.a + d.grp as x, d.pay * 2 as y, t.b - d.k as z from %T t join %D d on t.b = d.k where t.i < 90000"
	ajoin $t "text column of the inner side" "select t.i, d.txt from %T t join %D d on t.b = d.k where t.i < 40000"
	ajoin $t "extra join qual" "select t.i, d.grp from %T t join %D d on t.b = d.k and t.a > d.grp where t.i < 100000"
	ajoin $t "no match at all" "select t.i, d.grp from %T t join %D d on t.b = d.k + 1000000"
	ajoin $t "LEFT JOIN: unmatched outer rows get NULLs" "select t.i, t.b, d.grp, d.pay, d.txt from %T t left join %D d on t.b = d.k + 20000 where t.i < 60000"
	ajoin $t "LEFT JOIN with duplicate inner keys" "select t.i, t.a + d.grp as x, coalesce(d.pay, -1) as y from %T t left join %D d on t.b = d.k where t.i < 70000"
	ajoin $t "LEFT JOIN with an ON qual on the outer side" "select t.i, d.grp from %T t left join %D d on t.b = d.k and t.a > 0 where t.i < 70000"
	ajoin $t "semi join (EXISTS)" "select t.i, t.a, t.f from %T t where exists (select 1 from %D d where d.k = t.b and d.grp > 99) and t.i < 90000"
	ajoin $t "anti join (NOT EXISTS)" "select t.i, t.b from %T t where not exists (select 1 from %D d where d.k = t.b) and t.i < 50000"
	ajoin $t "RIGHT JOIN" "select t.i, d.k, d.grp from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000"
	ajoin $t "FULL JOIN" "select t.i, d.k, d.grp from %T t full join %D d on t.b = d.k and t.a + d.grp > 9900"
done
for t in a1 a3 p1; do
	echo "table $t: stacked joins and joins with an Arrow dimension table"
	adim=sd; adim2=sd2
	ajoin2 $t "two inner relations" "select t.i, t.a + d.grp + e.g2 as x, d.pay, e.p2, e.k2 from %T t join %D d on t.b = d.k join %E e on t.a % 400 = e.k2 where t.i < 90000"
	ajoin2 $t "three inner relations" "select t.i, d.grp, e.g2, f.g3 from %T t join %D d on t.b = d.k join %E e on t.a % 400 = e.k2 join %F f on t.a + 20000 = f.k3 where t.i < 90000"
	ajoin2 $t "inner join, then LEFT join" "select t.i, d.grp, e.g2, e.p2 from %T t join %D d on t.b = d.k left join %E e on t.a % 40 = e.k2 + 400 where t.i < 90000"
	ajoin2 $t "LEFT join, then inner join" "select t.i, d.grp, e.g2 from %T t left join %D d on t.b = d.k + 20000 join %E e on t.a % 400 = e.k2 where t.i < 80000"
	adim=ad; adim2=ad2
	ajoin $t "Arrow dimension table (pass-through of both)" "select t.i, t.a, d.grp, d.pay, d.txt from %T t join %D d on t.b = d.k where t.i < 60000"
	ajoin2 $t "two Arrow dimension tables" "select t.i, d.grp, e.g2, e.p2 from %T t join %D d on t.b = d.k join %E e on t.a % 400 = e.k2 where t.i < 90000"
	adim=adp
	ajoin $t "Parquet dimension table" "select t.i, d.grp, d.txt from %T t join %D d on t.b = d.k where t.i < 60000"
	adim=sd; adim2=sd2
done

for t in a1 a2 a3 p1; do
	echo "table $t: aggregation over a join"
	adim=sd; adim2=sd2
	ajagg $t "count, sum, max over both sides" "select count(*), sum(t.a), max(d.grp) from %T t join %D d on t.b = d.k"
	ajagg $t "NULL values from the inner side, duplicate keys" "select count(*), count(d.pay), sum(d.pay), min(d.pay), max(d.pay), avg(d.pay) from %T t join %D d on t.b = d.k"
	ajagg $t "empty join result" "select count(*), count(d.pay), sum(d.pay), min(t.a), max(d.grp), avg(t.b) from %T t join %D d on t.b = d.k + 1000000"
	ajagg $t "quals and FILTER on both sides" "select count(*) filter (where d.grp < 8), sum(t.b) filter (where t.a < 0), max(d.pay) filter (where t.e) from %T t join %D d on t.b = d.k where t.a > -9000"
	ajagg $t "sum beyond 64 bits" "select sum(t.c + 9000000000000000000), avg(d.pay + 9000000000000000000) from %T t join %D d on t.b = d.k"
	ajagg $t "text qual on the Arrow side" "select count(*), sum(d.grp) from %T t join %D d on t.b = d.k where t.f like 'a%'"
	ajagg $t "LEFT JOIN: count(*) against count(inner column)" "select count(*), count(d.grp), sum(d.grp), max(d.pay), sum(t.a) from %T t left join %D d on t.b = d.k + 20000"
	ajagg $t "EXISTS" "select count(*), sum(t.a), max(t.b) from %T t where exists (select 1 from %D d where d.k = t.b)"
	ajagg $t "NOT EXISTS" "select count(*), sum(t.a), count(t.b) from %T t where not exists (select 1 from %D d where d.k = t.b)"
	ajagg $t "two inner relations" "select count(*), sum(t.a + d.grp + e.g2), max(e.p2), min(d.pay) from %T t join %D d on t.b = d.k join %E e on t.a % 400 = e.k2"
	ajagg $t "GROUP BY the plain outer key column" "select t.a, count(*), sum(d.grp) from %T t join %D d on t.b = d.k group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=100000;"
	ajagg $t "GROUP BY a key of the inner side" "select d.grp, count(*), sum(t.a), max(t.c) from %T t join %D d on t.b = d.k group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	ajagg_any $t "GROUP BY an expression of the outer side" "select t.a % 20 as k, count(*), sum(d.grp) from %T t join %D d on t.b = d.k group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	ajagg_any $t "GROUP BY two keys (outer and inner), the overflow list" "select t.a % 200 as k1, d.grp, count(*), sum(d.pay) from %T t join %D d on t.b = d.k group by 1, 2 order by 1, 2" "set pg_strom.gpupreagg_max_groups=100000;"
	ajagg $t "GROUP BY the NULL group of a LEFT join" "select d.grp, count(*), sum(t.a) from %T t left join %D d on t.b = d.k + 20000 group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	ajagg $t "RIGHT JOIN: matched rows on the device, the unmatched inner rows from the backend" "select count(*), count(t.i), sum(d.grp), sum(t.a), max(d.pay) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000"
	ajagg $t "RIGHT JOIN, nothing matches" "select count(*), count(t.i), sum(d.grp), min(d.pay) from %T t right join %D d on t.b = d.k + 1000000"
	ajagg $t "FULL JOIN" "select count(*), count(t.i), count(d.k), sum(d.grp), sum(t.a) from %T t full join %D d on t.b = d.k and t.a > 9000"
	ajagg $t "RIGHT JOIN, GROUP BY a key of the inner side" "select d.grp, count(*), count(t.i), sum(t.a) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000 group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	# A14: float4 results that are denormal are decided by the backend (qual only: flagged; projection: scout pass)
	ajagg $t "recheck: float4 underflow in the scan qual" "select count(*), sum(d.grp), sum(t.a) from %T t join %D d on t.b = d.k where t.d / 1e38::float4 > 0::float4"
	ajagg $t "recheck: float4 underflow in the scan qual, GROUP BY" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k where t.d / 1e38::float4 > 0::float4 group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	ajagg $t "recheck: float4 underflow in the projection, GROUP BY" "select d.grp, count(*), sum((t.d / 1e38::float4 * 1e38::float4)::int) from %T t join %D d on t.b = d.k where t.d < 'Infinity'::float4 and t.d > '-Infinity'::float4 group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	ajagg $t "recheck: LEFT JOIN, float4 underflow in the ON qual" "select count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and t.d / 1e38::float4 > 0::float4"
	adim=ad; adim2=ad2
	ajagg $t "Arrow dimension table: GROUP BY" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
	adim=sd; adim2=sd2
done
ajagg a1 "parallel workers" "select count(*), sum(t.a), max(d.grp) from %T t join %D d on t.b = d.k" "$APAR"
ajagg a1 "GROUP BY, parallel workers" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1 order by 1" "$APAR set pg_strom.gpupreagg_max_groups=1000;"
ajagg p1 "GROUP BY, parallel workers (Parquet)" "select d.grp, count(*), sum(t.a) from %T t join %D d on t.b = d.k group by 1 order by 1" "$APAR set pg_strom.gpupreagg_max_groups=1000;"

# A12: rows the device cannot decide on an Arrow / Parquet source. A numeric qual used to keep the whole table on the CPU
# ("numeric in Arrow expressions"); now the VM computes it and hands back the rows it cannot decide (a numeric beyond 64
# bits, a division by zero, a float4 result that underflows): the service names such a row by (record batch, row) and the
# backend re-evaluates it from the file. The answer must be the CPU's, the plan must be the GPU's, and EXPLAIN ANALYZE must
# show rows handed back ("Fallback-stats").
echo "arrow: rows the device cannot decide (A12: float4 underflow, numerics beyond 64 bits, division by zero)"
ark_rows() { # table, settings, query-with-%T -> rows re-evaluated by the backend (EXPLAIN ANALYZE's "Fallback-stats: depth[0]=N")
	r=$($PSQL -d scantest -c "$GPU $2 explain (analyze, costs off) $(printf '%s' "$3" | sed "s/%T/$1/g; s/%D/sd/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
ark_check() { # table, settings, query-with-%T, label: the backend must have re-evaluated at least one row
	n=$(ark_rows "$1" "$2" "$3")
	[ "$n" -gt 0 ]; check "$1: the GPU handed rows back to the backend ($n rechecked): $4" $?
}
# (a CASE with a numeric result is not pushed down, so the overflowing product stands alone: |c| above about 2.1e6 needs more than 64 bits)
ARK_NUM="(c::numeric * c::numeric * c::numeric)"
ARK_NUMT="(t.c::numeric * t.c::numeric * t.c::numeric)"
for t in a1 p1; do
	echo "table $t"
	run $t "recheck: float4 underflow in a qual" "select i, d from %T where d / 1e38::float4 > 0::float4"
	run $t "recheck: numeric beyond 64 bits in a qual" "select i, a from %T where $ARK_NUM > 5"
	run $t "recheck: numeric division by zero in a qual" "select i from %T where i % 50 <> 0 and b::numeric / (i % 50)::numeric > 3"
	run $t "recheck: numerics in the projection" "select i, a::numeric * 100000 / 7, case when i % 50 = 0 then c::numeric * c::numeric * c::numeric else a::numeric end from %T where a > 0"
	aagg $t "recheck: ungrouped aggregate, numeric qual" "select count(*), sum(a), max(i) from %T where $ARK_NUM > 5"
	aagg $t "recheck: ungrouped aggregate, float4 underflow qual" "select count(*), sum(b), min(i) from %T where d / 1e38::float4 > 0::float4"
	aagg_any $t "recheck: GROUP BY, numeric qual" "select a % 7, count(*), sum(b) from %T where $ARK_NUM > 5 group by 1 order by 1"
	aagg_any $t "recheck: GROUP BY, float4 underflow qual" "select a % 7, count(*), sum(b) from %T where d / 1e38::float4 > 0::float4 group by 1 order by 1"
	aagg_any $t "recheck: numeric argument of an aggregate" "select count(*), sum(a::numeric * 3), sum(b) from %T where a > 0 and i % 50 <> 0 and b::numeric / (i % 50)::numeric > 3"
done
for t in a1 p1; do
	ark_check $t "" "select i, a from %T where $ARK_NUM > 5" "numeric beyond 64 bits (scan)"
	ark_check $t "" "select i from %T where i % 50 <> 0 and b::numeric / (i % 50)::numeric > 3" "numeric division by zero (scan)"
	ark_check $t "" "select i, d from %T where d / 1e38::float4 > 0::float4" "float4 underflow (scan)"
	ark_check $t "$AGGSET" "select count(*), sum(a), max(i) from %T where $ARK_NUM > 5" "numeric beyond 64 bits (aggregate)"
	ark_check $t "$AGGSET" "select a % 7, count(*), sum(b) from %T where d / 1e38::float4 > 0::float4 group by 1" "float4 underflow (GROUP BY)"
	adim=sd; adim2=sd2
	# a numeric is not computed inside a join program (plan A6 "known gaps"): the plan may stay on the CPU, the answer must not change
	ARROWANY=1
	ajoin $t "recheck: numeric qual below the join" "select t.i, d.grp from %T t join %D d on t.b = d.k where $ARK_NUMT > 5"
	ajagg $t "recheck: numeric qual below the join, aggregate" "select count(*), sum(d.grp), sum(t.a) from %T t join %D d on t.b = d.k where $ARK_NUMT > 5"
	ajagg $t "recheck: LEFT JOIN, numeric ON qual, aggregate" "select count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and $ARK_NUMT > 5"
	ARROWANY=0
	ajoin $t "recheck: float4 underflow qual below the join" "select t.i, d.grp from %T t join %D d on t.b = d.k where t.d / 1e38::float4 > 0::float4"
	ajoin $t "recheck: LEFT JOIN, float4 underflow ON qual" "select t.i, d.grp from %T t left join %D d on t.b = d.k and t.d / 1e38::float4 > 0::float4 where t.i < 60000"
	ark_check $t "$AJOINSET" "select t.i, d.grp from %T t join %D d on t.b = d.k where t.d / 1e38::float4 > 0::float4" "float4 underflow (join)"
done
# many groups (the overflow list of the GROUP BY kernels) next to a numeric qual
aagg_any a1 "recheck: GROUP BY with many groups, numeric qual" "select i % 4000 as k, count(*), sum(a) from %T where $ARK_NUM > 5 group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=100000;"

echo "arrow: value types (timestamps of every unit, dates, times, strings)"
GPUQ="$GPU $AGGSET"
for t in at1 at2; do
	run $t "timestamp[s] qual and pass-through" "select i, ts_s from %T where ts_s > '2017-07-15 00:00:00'::timestamp and ts_s < '2017-07-15 06:00:00'::timestamp"
	run $t "timestamp[ms] qual" "select i, ts_ms from %T where ts_ms >= '2017-07-14 02:40:10'::timestamp and ts_ms < '2017-07-14 02:41:00'::timestamp"
	run $t "timestamp[us] qual" "select i, ts_us from %T where ts_us >= '2017-07-14 02:40:10'::timestamp and ts_us < '2017-07-14 03:41:00'::timestamp"
	run $t "timestamp[ns] qual" "select i, ts_ns from %T where ts_ns >= '2017-07-14 02:40:10'::timestamp and ts_ns < '2017-07-14 06:41:00'::timestamp"
	run $t "date qual" "select i, d32 from %T where d32 >= '2000-01-01'::date and d32 < '2000-03-01'::date"
	run $t "time[s] qual" "select i, t_s from %T where t_s > '12:00:00'::time and t_s < '12:30:00'::time"
	run $t "time[ms] qual" "select i, t_ms from %T where t_ms > '03:00:00'::time and t_ms < '03:10:00'::time"
	run $t "time[us] qual" "select i, t_us from %T where t_us > '03:00:00'::time and t_us < '03:50:00'::time"
	run $t "time[ns] qual" "select i, t_ns from %T where t_ns > '20:00:00'::time and t_ns < '20:30:00'::time"
	run $t "all temporal columns projected" "select i, ts_s, ts_ms, ts_us, ts_ns, d32, t_s, t_ms, t_us, t_ns from %T where g = 3 and i < 5000"
	run $t "temporal column equality between columns" "select i from %T where ts_s = ts_s and d32 is not null and t_s < t_ms"
	run $t "utf8 equality" "select i, u8 from %T where u8 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'"
	run $t "utf8 LIKE" "select i, u8 from %T where u8 like 'bbb%'"
	run $t "large_utf8 qual" "select i, lu from %T where lu like 'CC%' and lu is not null"
	run $t "fixed_size_binary (char(5)) equality and LIKE" "select i, fx from %T where fx = 'abbcd' or fx like 'c%'"
	run $t "string comparison of two columns" "select i from %T where u8 < lu and u8 is not null"
	run_any $t "string function in a projection" "select i, upper(u8), length(lu) from %T where g = 2 and i < 3000"
	aagg_any $t "group by timestamp[s]" "select ts_s, count(*), sum(g) from %T group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000000;"
	aagg $t "group by date" "select d32, count(*), max(v) from %T group by 1 order by 1"
	aagg $t "group by time[s]" "select t_s, count(*), min(v) from %T group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000000;"
	aagg_any $t "group by timestamp[ms], timestamp[ns]" "select ts_ms, ts_ns, count(*) from %T where i < 3000 and g = 1 group by 1, 2 order by 1, 2"
	aagg_any $t "group by date, time[us]" "select d32, t_us, count(*), max(v) from %T where i < 3000 and g = 1 group by 1, 2 order by 1, 2"
	aagg_any $t "group by time[s], time[ms]" "select t_s, t_ms, count(*) from %T where i < 3000 and g = 2 group by 1, 2 order by 1, 2"
	aagg $t "aggregates with a utf8 qual and a char(5) qual" "select count(*), sum(v), max(i) from %T where u8 like 'a%' and fx <> 'abbcd'"
	aagg $t "group by an integer key with a timestamp[ns] qual" "select g, count(*), sum(v), min(v), max(v) from %T where ts_ns > '2017-07-14 02:40:10'::timestamp group by 1 order by 1"
	run $t "float8 (Arrow double) compare incl. NaN, Infinity, -0" "select i, x from %T where x > 100.5 or x = 'NaN' or x = '-Infinity' or x = 0"
	run $t "float8 column arithmetic" "select i, x * 2.5 + 1, x / 3, -x, abs(x) from %T where i % 7 = 3 and abs(x) < 1e6 and abs(x) > 1e-100"
	run $t "float8 cast to int and float4" "select i, x::int4, (x / 7)::float4 from %T where i % 5 = 1 and abs(x) < 1e9 and x = x"
	aagg $t "float8 (Arrow double): count, min, max with NaN and Infinity" "select count(x), min(x), max(x) from %T"
	aagg $t "float8 (Arrow double): sum and avg of exact values" "select sum(x), avg(x), count(x) from %T where i % 20 = 6 or i % 20 = 7"
	aagg $t "float8 aggregate of a computed value" "select max(x * 2 + 1), min(x - 1) from %T where i % 20 > 7"
	run $t "float8 (Arrow double) math functions over every value" "select i, round(x), ceil(x), ceiling(x), floor(x), trunc(x), sign(x) from %T where i % 3 = 1"
	run $t "float8 (Arrow double) sqrt of abs()" "select i, sqrt(abs(x)), round(sqrt(abs(x)) * 100) from %T where i % 3 = 0"
	run $t "float8 (Arrow double) guarded sqrt as a qual" "select i, x from %T where x >= 0 and sqrt(x) > 3"
	aagg $t "float8 (Arrow double): aggregates of math functions" "select count(*), sum(round(x)), max(ceil(x)), min(floor(x)), sum(sign(x)), max(sqrt(abs(x))) from %T where abs(x) < 1e9"
	# an error from the kernel reaches the user from an Arrow table as from a heap table (there is no recheck path here)
	qry="select sqrt(-abs(x)) from $t where abs(x) > 1 and abs(x) < 1e300 and i % 5 = 0"
	g=$($PSQL -d scantest -c "$GPU $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	pl=$($PSQL -d scantest -c "$GPU explain $qry" 2>&1 | grep -c "Custom Scan (GpuScan)")
	if [ "$pl" -ge 1 ] && echo "$g" | grep -q "cannot take square root of a negative number" && echo "$c" | grep -q "cannot take square root of a negative number"; then
		echo "  ok    $t: sqrt of a negative number raises PostgreSQL's message (Arrow)"
	elif [ "$ARROWANY" != 1 ] && [ "$t" != a2 ] && [ "$t" != at2 ]; then
		echo "  FAIL  $t: sqrt of a negative number (plan GpuScan=$pl)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1))
	else
		echo "  ok    $t: sqrt of a negative number (small table: CPU plan, message checked: $(echo "$g" | grep -c "cannot take square root"))"
	fi
	run $t "float8 IS NULL and a float8 / int8 mix" "select i, x + v::float8 from %T where x is not null and v > 0 and i % 3 = 0 and abs(x) < 1e9"
done
adim=sd; adim2=sd2
for t in at1 at2; do
	ajoin $t "join on a date key (heap table of dates)" "select t.i, t.d32, d.w from %T t join dd d on t.d32 = d.d where t.i < 30000"
	ajoin $t "join on a timestamp[s] key" "select t.i, t.ts_s, d.w from %T t join dts d on t.ts_s = d.t"
	ajagg $t "join aggregate on a date key" "select count(*), sum(t.v), max(d.w) from %T t join dd d on t.d32 = d.d"
	ajagg_any $t "join aggregate grouped by the date key" "select d.d, count(*), sum(t.v) from %T t join dd d on t.d32 = d.d group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=100000;"
	ajagg_any $t "join aggregate grouped by a time and an inner key" "select d.w, t.t_s, count(*) from %T t join dd d on t.d32 = d.d where t.t_s < '00:10:00'::time group by 1, 2 order by 1, 2" "set pg_strom.gpupreagg_max_groups=100000;"
	ajagg $t "join aggregate on a timestamp[s] key grouped by the inner column" "select d.w, count(*), sum(t.v), min(t.i) from %T t join dts d on t.ts_s = d.t group by 1 order by 1" "set pg_strom.gpupreagg_max_groups=1000;"
done
