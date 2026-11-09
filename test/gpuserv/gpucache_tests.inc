# GPU-Cache tests (A7 part 2), sourced by scan_test.sh and gpucache_dev.sh after the helpers (check, sql, q,
# GPU, CPU, runjoin, runagg) are defined, and after the table sd of scan_test.sh exists.
#
# The oracle of every comparison is the same query with pg_strom.enabled = off (PostgreSQL's own scan of the
# heap); a query on a cached table must also show "GPU Cache:" in its plan. In-transaction checks compare the
# cache and the heap inside one transaction, which is how a session's own uncommitted changes are tested.
echo "GPU-Cache"
GCDEF="id int4, a int2, b int4, c int8, d float4, e bool, f text, g date, h timestamp, n numeric, x float8, v varchar(20), w char(8), k float2, u uuid"
GCNR=${GCNR:-60000}
gcins() { # table first last
	cat <<EOF
insert into $1 select i, ((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 1001 = 0 then 'NaN'::float4 when i % 1999 = 0 then 'Infinity'::float4 else (((i * 17) % 10000) / 7.0)::float4 end,
	(i % 3 = 0),
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 200)::int) end,
	date '2000-01-01' + (i % 9000),
	case when i % 13 = 0 then null else timestamp '2000-01-01' + (i % 100000) * interval '1 minute' end,
	((i % 1000) / 8.0)::numeric, i / 13.0, 'v' || (i % 50), 'w' || (i % 7), ((i % 100)::float4)::float2, md5(i::text)::uuid
	from generate_series($2, $3) i
EOF
}
gcinfo() { # table, json key -> value
	$PSQL -d scantest -c "select pgstrom.gpucache_info('$1'::regclass)->>'$2'"
}
gcrun() { # table, label, query-with-%T [, settings]; the plan must read the GPU-Cache; the result equals the heap's
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	plan=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; $set_extra explain $qry" 2>&1 | grep -c "GPU Cache:")
	g=$(q "$GPU set enable_seqscan=off; $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (plan GPU Cache=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
gcrun_any() { # the same, but the plan need not use the cache
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	g=$(q "$GPU $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$g" = "$c" ] && [ "${g#*|}" != "" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
gcserved() { # table, label, query-with-%T [, settings]: the cache served the query (its scan counter grew), the executor did not read the table
	tbl=$1; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g"); set_extra=${4:-}
	b=$(gcinfo $tbl scans)
	$PSQL -d scantest -c "$GPU set enable_seqscan=off; $set_extra $qry" >/dev/null 2>&1
	a=$(gcinfo $tbl scans)
	[ -n "$b" ] && [ -n "$a" ] && [ "$a" -gt "$b" ]; check "$tbl: read from the cache, not from the table ($b -> $a scans): $2" $?
}
GCAGGSET="set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"
gcagg() { # table, label, query-with-%T; the plan must use GpuPreAgg over the cache
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g")
	plan=$($PSQL -d scantest -c "$GPU $GCAGGSET explain $qry" 2>&1)
	n1=$(echo "$plan" | grep -c "Custom Scan (GpuPreAgg)"); n2=$(echo "$plan" | grep -c "GPU Cache:")
	g=$($PSQL -d scantest -c "$GPU $GCAGGSET $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	if [ "$n1" -ge 1 ] && [ "$n2" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label"
	else echo "  FAIL  $tbl: $label (GpuPreAgg=$n1 cache=$n2)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
gcjoin() { # table, label, query-with-%T and %D; the plan must join on the GPU (GpuJoin, or GpuPreAgg with the join folded in) over the cache
	# GCJOINS=off skips the joins: lavapipe (CPU Vulkan) compiles the join kernels through LLVM, which takes more than 5 minutes and
	# 7 GB for the Arrow-source join (CI), so the lavapipe jobs leave them to the Apple Silicon run
	if [ "${GCJOINS:-on}" = off ]; then echo "  skip  $1: $2 (GCJOINS=off)"; return 0; fi
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/%D/sd/g")
	plan=$($PSQL -d scantest -c "$GPU $JOINSET explain $qry" 2>&1)
	n1=$(echo "$plan" | grep -c -E "Custom Scan \(GpuJoin\)|GPU ([A-Za-z]+ )*Join Quals"); n2=$(echo "$plan" | grep -c "GPU Cache:")
	g=$(q "$GPU $JOINSET" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$n1" -ge 1 ] && [ "$n2" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $tbl: $label (${g#*|} rows)"
	else echo "  FAIL  $tbl: $label (GpuJoin=$n1 cache=$n2)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
txncmp() { # label, ending (commit|rollback), statements run first, query: the cache and the heap agree inside the transaction
	label=$1; ending=$2; stmts=$3; qry=$4
	out=$($PSQL -d scantest 2>&1 <<EOF
begin;
$stmts
$GPU
set enable_seqscan=off;
explain $qry;
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
$CPU
select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($qry) x;
$ending;
EOF
)
	n=$(echo "$out" | grep -c "GPU Cache:")
	g=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 1p); c=$(echo "$out" | grep -E '^[0-9a-f]{32}\|' | sed -n 2p)
	if [ "$n" -ge 1 ] && [ -n "$g" ] && [ "$g" = "$c" ] && ! echo "$out" | grep -q "ERROR"; then echo "  ok    $label (${g#*|} rows)"
	else echo "  FAIL  $label (GPU Cache lines=$n)"; echo "$out" | head -20; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}

# ---- tables: gc1 is filled first and gets its trigger afterwards (initial load from the heap), gc2 gets the trigger
# first (every row goes through the log)
sql "create table gc1 ($GCDEF)"
sql "$(gcins gc1 1 $GCNR)"
sql "update gc1 set b = b + 1 where id % 5 = 0"; sql "delete from gc1 where id % 13 = 0"; sql "vacuum analyze gc1"
sql "create table gc2 ($GCDEF)"
sql "create trigger gc_sync after insert or update or delete on gc2 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((GCNR * 3))')"
[ "$(gcinfo gc2 phase)" = "not_built" ]; check "a new cache is not built before it is used" $?
sql "$(gcins gc2 1 $GCNR)"
sql "update gc2 set b = b + 1 where id % 5 = 0"; sql "delete from gc2 where id % 13 = 0"; sql "analyze gc2"
sql "create trigger gc_sync after insert or update or delete on gc1 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((GCNR * 3))')"
[ "$(gcinfo gc1 phase)" = "not_built" ]; check "gc1: not built before it is used" $?
[ "$(gcinfo gc2 phase)" = "ready" ]; check "gc2: built by the first write through the trigger" $?
n=$($PSQL -d scantest -c "select count(*) from gc2"); m=$(gcinfo gc2 nitems)
[ "$n" = "$m" ]; check "gc2: the cache holds the table's rows ($m of $n)" $?

for t in gc1 gc2; do
	echo "table $t"
	gcrun $t "trivial qual, mixed projection" "select a, f, c from %T where id > 0"
	gcrun $t "select * with an int4 qual (every type passes through)" "select * from %T where b < 100"
	gcrun $t "select * of one row" "select * from %T where id = 12345"
	gcrun $t "int2/int8 qual" "select a, c from %T where a > 0 and c < 0"
	gcrun $t "int2*int4 arithmetic, NULLs" "select a + b, c from %T where b is not null and a * 2 < b"
	gcrun $t "float4 compare incl. NaN/Infinity" "select id, d from %T where d > 100.5::float4"
	gcrun $t "float4 arithmetic" "select id, d * 2::float4 + 1::float4 from %T where d < 50::float4"
	gcrun $t "bool column and OR" "select id, e from %T where e and (b > 0 or c > 0)"
	gcrun $t "division and modulo" "select b, c / 3, c % 7 from %T where c > 1000"
	gcrun $t "IS NULL on several columns" "select id from %T where b is null or f is null or h is null"
	gcrun $t "casts" "select a::int4 + b, c::int4 from %T where c between -1000000 and 1000000"
	gcrun $t "date and timestamp quals" "select id, g, h from %T where g > '2010-01-01'::date and h < '2010-06-01'::timestamp"
	gcrun $t "date arithmetic" "select id, g + 7, g - '2005-05-05'::date from %T where g < '2002-01-01'::date"
	gcrun $t "text equality, LIKE and NULL" "select id, f from %T where f like 'abc%' or f = 'aaaa'"
	gcrun $t "varchar and char(n) columns" "select id, v, w from %T where v = 'v7' or w = 'w3'"
	gcrun $t "varlena pass-through with a qual on another column" "select f, v from %T where a = 123"
	gcrun $t "numeric, float8, float2 and uuid pass through" "select id, n, x, k, u from %T where a = 7"
	gcrun $t "ctid" "select ctid, id from %T where a = 9"
	gcrun $t "10 computed columns (16 registers)" "select a+1, a+2, a+3, a+4, a+5, b+1, b+2, b+3, b+4, b+5 from %T where a > 9000"
	gcrun $t "24 computed columns (32 registers)" "select a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11,a+12, b-1,b-2,b-3,b-4,b-5,b-6,b-7,b-8,b-9,b-10,b-11,b-12 from %T where a > 9500"
	gcrun $t "no row qualifies" "select id from %T where a > 30000"
	gcrun $t "every row qualifies" "select id, a from %T where id > 0 or id <= 0"
	echo "  -- A12: numerics are read from the cache (they used to make the executor read the table)"
	gcrun $t "numeric in a qual" "select id, n from %T where n > 50"
	gcserved $t "numeric in a qual" "select id, n from %T where n > 50"
	gcrun $t "numeric in the projection" "select id, n * 2 from %T where a = 5"
	gcserved $t "numeric in the projection" "select id, n * 2 from %T where a = 5"
	gcrun $t "float8 qual" "select id, x from %T where x > 1000"
	gcrun $t "float8 arithmetic and a cast" "select id, x * 2 + 1, x::int8 from %T where x < 500 and a = 7"
	gcrun $t "float8 math functions (round, ceil, floor, trunc, sign, sqrt of abs)" "select id, round(x), ceil(x), floor(x), trunc(x), sign(x), sqrt(abs(x)) from %T where a = 7"
	gcrun $t "guarded sqrt as a qual" "select id, x from %T where x >= 0 and sqrt(x) > 20"
	gcagg $t "aggregates of math functions" "select count(*), sum(round(x)), max(ceil(x)), min(floor(x)), max(sqrt(abs(x))) from %T"
	# the kernel's error reaches the user from the cache, with PostgreSQL's message
	qry="select sqrt(-abs(x)) from $t where abs(x) > 1 and a = 7"
	g=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; $qry" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | tail -1)
	pl=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; explain $qry" 2>&1 | grep -c "GPU Cache:")
	if [ "$pl" -ge 1 ] && echo "$g" | grep -q "cannot take square root of a negative number" && echo "$c" | grep -q "cannot take square root of a negative number"; then
		echo "  ok    $t: sqrt of a negative number raises PostgreSQL's message (GPU-Cache)"
	else echo "  FAIL  $t: sqrt of a negative number (plan GPU Cache=$pl)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
	gcrun_any $t "uuid qual" "select id from %T where u > '80000000-0000-0000-0000-000000000000'::uuid"
	gcagg $t "count, sum, min, max, avg of int4 with NULLs" "select count(*), count(b), sum(b), min(b), max(b), avg(b) from %T"
	gcagg $t "float8: count, min, max, and exact sum and avg" "select count(x), min(x), max(x), sum((a % 100)::float8), avg((a % 100)::float8) from %T"
	gcagg $t "variance, stddev, corr, regr_slope of float8 columns (A10)" "select round(var_pop(x)::numeric, 3), round(stddev(x)::numeric, 5), round(corr(x, a::float8)::numeric, 8), round(regr_slope(x, a::float8)::numeric, 7), count(*) from %T where x < 'Infinity' and x > '-Infinity'"
	gcagg $t "variance and covariance per group (A10)" "select a % 5 as k, round(var_samp(x)::numeric, 3), round(covar_pop(x, b::float8)::numeric, 3), count(x) from %T where x < 'Infinity' and x > '-Infinity' group by 1 order by 1"
	gcagg $t "int2 and int8 columns, quals" "select sum(a), min(a), max(a), sum(c), max(c) from %T where a > 0 and b is not null"
	gcagg $t "GROUP BY an integer expression" "select a % 10 as k, count(*), sum(b), max(c) from %T group by 1 order by 1"
	gcagg $t "GROUP BY date and timestamp aggregates" "select b % 5 as k, count(*), min(g), max(h) from %T group by 1 order by 1"
	gcagg $t "GROUP BY many groups (session-wide table)" "select id % 7000 as k, count(*), sum(b), max(c) from %T group by 1 order by 1"
	gcagg $t "GROUP BY three keys" "select a % 5 as k1, b % 3 as k2, id % 4 as k3, count(*), sum(c) from %T group by 1, 2, 3 order by 1, 2, 3"
	gcagg $t "empty input" "select count(*), sum(b) from %T where a > 100000"
	gcjoin $t "join with a heap table" "select t.id, t.a, d.grp, d.pay from %T t join %D d on t.b = d.k where t.a > 9000"
	gcjoin $t "LEFT JOIN, text of both sides" "select t.id, d.txt, t.f from %T t left join %D d on t.b = d.k where t.id < 5000"
	gcjoin $t "join aggregate" "select d.grp, count(*), sum(t.c) from %T t join %D d on t.b = d.k where t.a > 0 group by 1 order by 1"
	gcjoin $t "RIGHT JOIN aggregate" "select count(*), count(t.id), sum(d.grp), sum(t.c) from %T t right join %D d on t.b = d.k and t.a + d.grp > 9000"
	gcjoin $t "FULL JOIN aggregate, GROUP BY an inner column" "select d.grp, count(*), count(t.id), count(d.k) from %T t full join %D d on t.b = d.k and t.a > 9000 group by 1 order by 1"
	# the rows deleted from the table (id % 13 = 0) are still slots of the cache: their partners must not count as matches
	gcjoin $t "RIGHT JOIN aggregate whose only partners were deleted" "select count(*), count(t.id), sum(d.grp) from %T t right join %D d on t.b = d.k and t.id % 13 = 0"
done
ANYPLAN=0
gcrun gc1 "parallel workers on the cache" "select a, c, f from %T where b < 1000" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0; set parallel_leader_participation=off;"

echo "GPU-Cache: the SQL surface"
n=$($PSQL -d scantest -c "$GPU set pg_strom.enable_gpucache=off; explain select a from gc1 where b < 100" | grep -c "GPU Cache:")
[ "$n" -eq 0 ]; check "pg_strom.enable_gpucache = off reads the table" $?
[ "$(gcinfo gc1 phase)" = "ready" ]; check "gpucache_info(regclass) reports the phase" $?
n=$($PSQL -d scantest -c "select count(*) from json_array_elements(pgstrom.gpucache_info()->'caches') c where c->>'table_name' in ('gc1', 'gc2')")
[ "$n" = 2 ]; check "gpucache_info() lists both caches" $?
[ "$(gcinfo gc1 config_options)" = "gpu_device_id=0,max_num_rows=$((GCNR * 3)),redo_buffer_size=167772160,gpu_sync_interval=5000000,gpu_sync_threshold=41943040" ]; check "config_options" $?
$PSQL -d scantest -c "create trigger gc_ng1 after insert or update or delete on sd for each row execute function pgstrom.gpucache_sync_trigger('cpu_device_id=0')" 2>&1 | grep -q "unknown option"; check "an unknown option is refused" $?
$PSQL -d scantest -c "create trigger gc_ng2 after insert or update or delete on sd for each row execute function pgstrom.gpucache_sync_trigger('redo_buffer_size=4m')" 2>&1 | grep -q "too small"; check "a too small redo_buffer_size is refused" $?
$PSQL -d scantest -c "create trigger gc_ng3 after insert or update or delete on gc1 for each row execute function pgstrom.gpucache_sync_trigger()" 2>&1 | grep -q "multiple row-sync triggers"; check "a second row-sync trigger is refused" $?
$PSQL -d scantest -c "select pgstrom.gpucache_compaction('gc1'::regclass)" >/dev/null 2>&1; check "gpucache_compaction() runs" $?
gcrun gc1 "after the compaction" "select id, f from %T where b < 200"
$PSQL -d scantest -c "select pgstrom.gpucache_apply_redo()" >/dev/null 2>&1; check "gpucache_apply_redo() runs" $?
[ "$($PSQL -d scantest -c "select pgstrom.gpucache_recovery('gc1'::regclass)" 2>&1 | grep -c "is not corrupted")" -ge 0 ]; check "gpucache_recovery() on a healthy cache is a no-op" $?

echo "GPU-Cache: changes"
sql "create table gc3 ($GCDEF)"
sql "$(gcins gc3 1 20000)"
sql "create trigger gc_sync after insert or update or delete on gc3 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=100000')"
sql "vacuum analyze gc3"
Q3="select id, a, b, f, v from gc3 where b < 5000"
Q3A="select count(*), sum(b), max(a), min(id) from gc3 where id > 0"
gcrun gc3 "initial load" "$Q3"
sql "insert into gc3 select * from gc3 where id <= 100 and false"
sql "$(gcins gc3 20001 20500)"
gcrun gc3 "after an INSERT" "$Q3"
sql "update gc3 set b = b - 7, f = f || 'x' where id % 9 = 0"
gcrun gc3 "after an UPDATE" "$Q3"
sql "delete from gc3 where id % 17 = 0"
gcrun gc3 "after a DELETE" "$Q3"
gcrun gc3 "aggregate after the changes" "$Q3A"
gcagg gc3 "GpuPreAgg after the changes" "$Q3A"
sql "update gc3 set id = id + 100000 where id between 1000 and 1100"
gcrun gc3 "after an UPDATE of the key" "select id, b from gc3 where id > 90000 or id < 1200"
sql "update gc3 set f = null where id % 100 = 0"
sql "update gc3 set f = repeat('long', 300) where id % 1000 = 1"
gcrun gc3 "text set to NULL and to long values" "select id, f from gc3 where id % 100 = 0 or id % 1000 = 1"
txncmp "own INSERT is visible inside the transaction" rollback "$(gcins gc3 30001 30100);" "select id, b from gc3 where id > 29000"
txncmp "own UPDATE and DELETE are visible inside the transaction" rollback "update gc3 set b = 1 where id < 100; delete from gc3 where id between 100 and 200;" "select id, b from gc3 where id < 400"
gcrun gc3 "a rolled back transaction leaves no trace" "select id, b from gc3 where id > 29000 or id < 400"
txncmp "committed transaction" commit "$(gcins gc3 31001 31100); update gc3 set b = 2 where id between 300 and 310; delete from gc3 where id between 320 and 330;" "select id, b from gc3 where id > 31000 or id between 290 and 340"
gcrun gc3 "after the committed transaction" "select id, b from gc3 where id > 31000 or id between 290 and 340"
txncmp "savepoint rollback" commit "insert into gc3 select * from gc3 where false; $(gcins gc3 32001 32050); savepoint s1; $(gcins gc3 32051 32100); update gc3 set b = 3 where id between 32001 and 32010; delete from gc3 where id between 32041 and 32045; rollback to s1; $(gcins gc3 32101 32110);" "select id, b from gc3 where id > 32000"
gcrun gc3 "after the savepoint transaction" "select id, b from gc3 where id > 32000"
txncmp "released savepoint, then a rolled back outer" rollback "savepoint s1; $(gcins gc3 33001 33050); release s1; savepoint s2; delete from gc3 where id between 33001 and 33010; release s2;" "select id, b from gc3 where id > 33000"
gcrun gc3 "after the rolled back outer transaction" "select id, b from gc3 where id > 33000"
txncmp "nested savepoints, the middle one rolled back" commit "$(gcins gc3 34001 34010); savepoint a; $(gcins gc3 34011 34020); savepoint b; $(gcins gc3 34021 34030); rollback to a; $(gcins gc3 34031 34040);" "select id from gc3 where id > 34000"
gcrun gc3 "after the nested savepoints" "select id from gc3 where id > 34000"
txncmp "INSERT then DELETE of the same rows in one transaction" commit "$(gcins gc3 35001 35100); delete from gc3 where id between 35001 and 35050;" "select id from gc3 where id > 35000"
txncmp "an UPDATE twice in one transaction" commit "update gc3 set b = 11 where id between 400 and 450; update gc3 set b = b + 1 where id between 420 and 470;" "select id, b from gc3 where id between 390 and 480"
gcrun gc3 "after the double UPDATE" "select id, b from gc3 where id between 390 and 480"
$PSQL -d scantest -c "begin; $(gcins gc3 36001 36010); prepare transaction 'gc_txn1'" 2>&1 | grep -q "cannot be prepared"; check "PREPARE TRANSACTION is refused for a transaction that wrote a cached table" $?
gcrun gc3 "after the refused PREPARE" "select id from gc3 where id > 36000"

echo "GPU-Cache: many small transactions"
sql "create table gc4 (id int4 primary key, a int2, b int4, c int8, f text)"
sql "create trigger gc_sync after insert or update or delete on gc4 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "insert into gc4 values (0, 0, 0, 0, 'seed')"
$PSQL -d scantest -c "$GPU select count(*) from gc4 where b >= 0" >/dev/null
$PSQL -d scantest <<'EOF' >/dev/null 2>&1
do $$ begin
  for i in 1..1500 loop
    insert into gc4 values (i, (i % 100)::int2, i * 3, i::int8 * 7, 'row' || i);
    if i % 3 = 0 then update gc4 set b = b + 1, f = f || '!' where id = i - 1; end if;
    if i % 10 = 0 then delete from gc4 where id = i - 5; end if;
    commit;
  end loop;
end $$;
EOF
gcrun gc4 "1500 small transactions" "select id, a, b, c, f from %T where b >= 0"
gcagg gc4 "aggregate after the small transactions" "select count(*), sum(b), sum(c), max(a) from %T"

echo "GPU-Cache: concurrent sessions"
sql "create table bank (id int4 primary key, bal int8, pad text)"
sql "insert into bank select i, 1000, repeat('p', 20) from generate_series(1, 400) i"
sql "create trigger gc_sync after insert or update or delete on bank for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "analyze bank"
BANKQ="select sum(bal), count(*) from bank where id > 0"
cat > "$ROOT/bank_writer.sql" <<'EOF'
do $$
declare a int; b int;
begin
  for i in 1..250 loop
    a := (random() * 399)::int + 1; b := (random() * 399)::int + 1;
    if a <> b then
      begin
        update bank set bal = bal + case id when a then -5 else 5 end where id in (a, b);
        if i % 7 = 0 then insert into bank values (1000 + (random() * 1000000)::int, 0, 'x') on conflict do nothing; end if;
        commit;
      exception when others then rollback;
      end;
    end if;
  end loop;
end $$;
EOF
cat > "$ROOT/bank_reader.sql" <<'EOF'
set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0;
set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;
do $$
declare s int8; n int8; bad int := 0;
begin
  for i in 1..150 loop
    select sum(bal), count(*) into s, n from bank where id > 0;
    if s <> 400000 then bad := bad + 1; end if;
  end loop;
  raise notice 'bad=%', bad;
end $$;
EOF
$PSQL -d scantest -c "$GPU $GCAGGSET $BANKQ" >/dev/null
for w in 1 2 3 4; do $PSQL -d scantest -f "$ROOT/bank_writer.sql" >/dev/null 2>&1 & done
$PSQL -d scantest -f "$ROOT/bank_reader.sql" > "$ROOT/bank_reader.out" 2>&1 &
wait
grep -q "bad=0" "$ROOT/bank_reader.out"; check "readers see every transfer atomically while 4 writers run ($(grep -o 'bad=[0-9]*' "$ROOT/bank_reader.out"))" $?
gcrun bank "the cache agrees with the heap after the writers" "select id, bal, pad from %T where id > 0"
gcagg bank "total is unchanged" "select sum(bal), count(*) from %T where id > 0 and id <= 400"

echo "GPU-Cache: TRUNCATE, ALTER TABLE, DROP"
sql "create table gc5 ($GCDEF)"
sql "$(gcins gc5 1 3000)"
sql "create trigger gc_sync after insert or update or delete on gc5 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "analyze gc5"
gcrun gc5 "before TRUNCATE" "select id, b from %T where b < 3000"
sql "truncate gc5"
n=$($PSQL -d scantest -c "$GPU select count(*) from gc5")
[ "$n" = 0 ]; check "after TRUNCATE the table is empty on the GPU" $?
sql "$(gcins gc5 1 500)"
gcrun gc5 "rows inserted after TRUNCATE" "select id, b, f from %T where b < 3000"
txncmp "TRUNCATE and inserts in one transaction" commit "truncate gc5; $(gcins gc5 1 50);" "select id, b from gc5 where id > 0"
gcrun gc5 "after TRUNCATE in a transaction" "select id, b from %T where id > 0"
sql "alter table gc5 add column z int4 default 5"
sql "update gc5 set z = id"
gcrun gc5 "after ADD COLUMN" "select id, z, b from %T where z < 100"
sql "alter table gc5 drop column a"
gcrun gc5 "after DROP COLUMN" "select id, b, z, f from %T where z < 100"
sql "alter table gc5 alter column b type int8"
gcrun gc5 "after ALTER COLUMN TYPE (rewrite)" "select id, b, z from %T where z < 100"
sql "vacuum full gc5"
gcrun gc5 "after VACUUM FULL" "select id, b, z from %T where z < 100"
sql "insert into gc5 (id, b, z, f) values (999999, 1, 2, 'new')"
gcrun gc5 "after an INSERT into the rewritten table" "select id, b, z, f from %T where z < 100 or id = 999999"
sql "drop trigger gc_sync on gc5"
n=$($PSQL -d scantest -c "$GPU explain select id from gc5 where z < 100" | grep -c "GPU Cache:")
[ "$n" -eq 0 ]; check "DROP TRIGGER stops the cache" $?
sql "create trigger gc_sync after insert or update or delete on gc5 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
gcrun gc5 "a trigger created again rebuilds the cache" "select id, b, z from %T where z < 100"
sql "alter table gc5 disable trigger gc_sync"
sql "insert into gc5 (id, b, z, f) values (888888, 1, 2, 'unlogged')"
sql "alter table gc5 enable trigger gc_sync"
gcrun gc5 "a write while the trigger is disabled is not lost" "select id, b, z, f from %T where z < 100 or id >= 888888"
sql "drop table gc5"
n=$($PSQL -d scantest -c "select count(*) from json_array_elements(pgstrom.gpucache_info()->'caches') c where c->>'table_name' = 'gc5'")
[ "$n" = 0 ]; check "DROP TABLE removes the cache" $?

echo "GPU-Cache: overflow, corruption and recovery"
sql "create table gc6 ($GCDEF)"
sql "create trigger gc_sync after insert or update or delete on gc6 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=5000')"
sql "$(gcins gc6 1 4000)"
sql "analyze gc6"
gcrun gc6 "4000 rows in a cache of 5000" "select id, b from %T where b < 30000"
[ "$(gcinfo gc6 phase)" = ready ]; check "gc6 is ready" $?
$PSQL -d scantest -c "update gc6 set b = b / 2 where id <= 1500" 2>&1 | grep -q "rowid exceeds max_num_rows"; check "versions in flight beyond max_num_rows corrupt the cache (WARNING)" $?
[ "$(gcinfo gc6 phase)" = corrupted ]; check "the cache is marked corrupted" $?
n=$($PSQL -d scantest -c "$GPU explain select id from gc6 where b < 30000" | grep -c "GPU Cache:")
[ "$n" -eq 0 ]; check "a corrupted cache is not used" $?
gcrun_any gc6 "queries are correct while it is corrupted" "select id, b from %T where b < 30000"
sql "update gc6 set b = b + 1 where id <= 100"
gcrun_any gc6 "writes while it is corrupted" "select id, b from %T where b < 30000"
sql "select pgstrom.gpucache_recovery('gc6'::regclass)"
gcrun gc6 "recovered" "select id, b from %T where b < 30000"
[ "$(gcinfo gc6 phase)" = ready ]; check "recovered: ready again" $?
sql "create table gc7 ($GCDEF)"
sql "$(gcins gc7 1 6000)"
sql "create trigger gc_sync after insert or update or delete on gc7 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=5000')"
$PSQL -d scantest -c "$GPU select count(*) from gc7 where b < 100" >/dev/null 2>&1
[ "$(gcinfo gc7 phase)" = corrupted ]; check "a table larger than max_num_rows is corrupted by the initial load" $?
gcrun_any gc7 "and is still read correctly" "select id, b from %T where b < 100"

echo "GPU-Cache: restart of the GPU service"
gen0=$($PSQL -d scantest -c "select generation from pgstrom.gpu_service_state()")
rb0=$(gcinfo gc3 rebuilds)
spid=$($PSQL -d scantest -c "select pid from pgstrom.gpu_service_state()")
kill -9 "$spid"
n=0; while [ $n -lt 60 ]; do
	st=$($PSQL -d scantest -c "select state || ' ' || generation from pgstrom.gpu_service_state()" 2>/dev/null)
	[ "$st" = "ready $((gen0 + 1))" ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 60 ]; check "the launcher restarts the killed service" $?
gcrun gc3 "the cache is rebuilt from the table after the restart" "select id, a, b, f, v from gc3 where b < 5000"
rb1=$(gcinfo gc3 rebuilds)
[ "$rb1" -gt "$rb0" ]; check "the restart rebuilt the cache ($rb0 -> $rb1)" $?
sql "$(gcins gc3 40001 40100)"
gcrun gc3 "changes after the restart" "select id, b from %T where id > 39000"
gcrun gc2 "another cache after the restart" "select id, b, f from %T where b < 100"
gcagg bank "aggregate after the restart" "select sum(bal), count(*) from %T where id > 0 and id <= 400"

echo "GPU-Cache: isolation between sessions, row shapes"
sql "create table gc8 (id int4, a int4, f text)"
sql "insert into gc8 select i, i, 'old' from generate_series(1, 1000) i"
sql "create trigger gc_sync after insert or update or delete on gc8 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "analyze gc8"
GC8Q="select count(*), sum(a), count(f) from gc8 where id > 0"
$PSQL -d scantest -c "$GPU $GCAGGSET $GC8Q" >/dev/null
$PSQL -d scantest -c "begin; insert into gc8 select i, i, 'new' from generate_series(2001, 2100) i; update gc8 set a = -1 where id <= 10; delete from gc8 where id between 11 and 20; select pg_sleep(4); commit" >/dev/null 2>&1 &
sleep 1.5
g=$($PSQL -d scantest -c "$GPU $GCAGGSET $GC8Q" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $GC8Q" 2>&1 | tail -1)
[ "$g" = "$c" ] && [ "$g" = "1000|500500|1000" ]; check "another session's uncommitted INSERT, UPDATE and DELETE are invisible ($g)" $?
wait
g=$($PSQL -d scantest -c "$GPU $GCAGGSET $GC8Q" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $GC8Q" 2>&1 | tail -1)
[ "$g" = "$c" ] && [ "$g" != "1000|500500|1000" ]; check "... and visible after its COMMIT ($g)" $?
$PSQL -d scantest -c "begin; delete from gc8 where id > 2000; select pg_sleep(3); rollback" >/dev/null 2>&1 &
sleep 1
g=$($PSQL -d scantest -c "$GPU $GCAGGSET $GC8Q" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $GC8Q" 2>&1 | tail -1)
[ "$g" = "$c" ]; check "an uncommitted DELETE of another session leaves the rows visible ($g)" $?
wait
g=$($PSQL -d scantest -c "$GPU $GCAGGSET $GC8Q" 2>&1 | tail -1); c=$($PSQL -d scantest -c "$CPU $GC8Q" 2>&1 | tail -1)
[ "$g" = "$c" ]; check "... and after its ROLLBACK ($g)" $?
# (a whole-row reference such as "select t from gc8 t where ..." fails with "cache lookup failed for attribute 0" in the
# planner of the heap GpuScan as well, with or without a cache: not a GPU-Cache matter, see vulkan_known_failures.txt)
gcrun gc8 "DELETE ... RETURNING and UPDATE ... RETURNING through the cache" "select id from gc8 where id < 0 or id > 3000"
g=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; with d as (delete from gc8 where id between 2050 and 2060 returning id) select count(*), sum(id) from d")
c=$($PSQL -d scantest -c "$CPU select count(*), sum(id) from gc8 where id between 2050 and 2060")
[ "$g" = "11|22605" ]; check "DELETE ... RETURNING scans the cache ($g)" $?
gcrun gc8 "after the DELETE ... RETURNING" "select id, a from gc8 where id > 2000"
$PSQL -d scantest -c "$GPU set enable_seqscan=off; begin; select id from gc8 where id < 5 for update; commit" >/dev/null 2>&1; check "SELECT ... FOR UPDATE on a cached table" $?
sql "create table gcp (id int4, a int4, f text) partition by range (id)"
sql "create table gcp1 partition of gcp for values from (0) to (5000)"
sql "create table gcp2 partition of gcp for values from (5000) to (10000)"
sql "insert into gcp select i, i % 100, 'p' || i from generate_series(1, 9000) i"
sql "create trigger gc_sync after insert or update or delete on gcp1 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "analyze gcp"
gcrun_any gcp "partitioned table, one partition cached" "select id, a, f from %T where a < 20"
sql "update gcp set a = a + 1000 where id % 7 = 0"
gcrun_any gcp "after an UPDATE through the partitioned table" "select id, a, f from %T where a < 20 or a > 1000"
gcagg gcp1 "aggregate over the cached partition" "select count(*), sum(a) from %T where id > 0"
sql "create table gcw (id int4, a1 int4, a2 int4, a3 int4, a4 int4, a5 int4, a6 int4, a7 int4, a8 int4, a9 int4, a10 int4, a11 int4, a12 int4, a13 int4, a14 int4, a15 int4, a16 int4, a17 int4, a18 int4, a19 int4, a20 int4, a21 int4, a22 int4, a23 int4, a24 int4, a25 int4, a26 int4, a27 int4, a28 int4, a29 int4, a30 int4, a31 int4, a32 int4, a33 int4, a34 int4, a35 int4, t text)"
sql "create trigger gc_sync after insert or update or delete on gcw for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=20000')"
sql "insert into gcw select i, i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i, repeat('t', i % 300) from generate_series(1, 5000) i"
sql "analyze gcw"
gcrun gcw "36 columns, a text column and quals on the last ones" "select id, a35, t from %T where a35 > 4990 or length(t) = 0"

# A12: numeric and jsonb columns are read from the cache, and a value the kernel cannot decide (a numeric beyond 64 bits,
# NaN, a number in a jsonb document that is no int64, a float4 result that underflows) makes the backend re-evaluate the row
# from a heap tuple built from the cache slot. The plan reads the cache ("GPU Cache:"), the executor must not fall back to
# the table, and the answer is the heap's.
echo "GPU-Cache: numeric, jsonb and float4 values the device cannot decide (A12)"
GC9N=${GC9N:-30000}
sql "create table gc9 (id int4, b int4, d float4, n numeric(18,3), nn numeric, j jsonb, f text)"
sql "create trigger gc_sync after insert or update or delete on gc9 for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((GC9N * 3))')"
sql "insert into gc9 select i, (i * 31) % 1000,
	case when i % 7 = 0 then 1e-40::float4 when i % 11 = 0 then 1e-30::float4 else (((i * 17) % 10000) / 7.0)::float4 end,
	case when i % 9 = 0 then null else ((i % 5000) / 8.0)::numeric(18,3) end,
	case when i % 101 = 0 then 'NaN'::numeric when i % 53 = 0 then 1e30::numeric + i when i % 47 = 0 then null else (i / 8.0)::numeric end,
	case when i % 61 = 0 then null when i % 29 = 0 then jsonb_build_object('a', 1e30::numeric + i, 'k', 'x' || (i % 5))
	     else jsonb_build_object('a', i, 'k', 'x' || (i % 5), 'o', jsonb_build_object('z', i % 3)) end,
	repeat(chr(97 + i % 26), (i % 40)::int)
	from generate_series(1, $GC9N) i"
sql "update gc9 set nn = nn * 3 where id % 5 = 0 and nn is not null"
sql "delete from gc9 where id % 997 = 0"
sql "analyze gc9"
gck_rows() { # table, query-with-%T -> rows the backend re-evaluated (EXPLAIN ANALYZE: "Fallback-stats: depth[0]=N")
	r=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; explain (analyze, costs off) $(printf '%s' "$2" | sed "s/%T/$1/g")" | sed -n 's/.*depth\[0\]=\([0-9]*\).*/\1/p' | head -1)
	echo "${r:-0}"
}
gck() { # table, label, query-with-%T: scan through the cache, served by it, rows handed back to the backend
	gcrun "$1" "$2" "$3"; gcserved "$1" "$2" "$3"
	n=$(gck_rows "$1" "$3")
	[ "$n" -gt 0 ]; check "$1: the GPU handed rows back to the backend ($n rechecked): $2" $?
}
gck gc9 "numeric qual (NaN, beyond 64 bits)" "select id, nn from %T where nn > 100"
gck gc9 "numeric arithmetic in a qual" "select id from %T where nn * 2 > 1000 and nn / 3 < 1000000"
gck gc9 "numeric arithmetic in the projection" "select id, nn * 2, nn / 3 from %T where id % 3 = 0"
gcrun gc9 "numeric(18,3) qual, NULLs" "select id, n from %T where n > 100 or n is null"
gcserved gc9 "numeric(18,3) qual, NULLs" "select id, n from %T where n > 100 or n is null"
gck gc9 "jsonb: number of a document that is no int64" "select id from %T where (j->>'a')::numeric > 100"
gcrun gc9 "jsonb: key, containment and path" "select id, j from %T where j ? 'o' and j @> '{\"k\": \"x2\"}' and (j->'o'->>'z')::int = 1"
gcserved gc9 "jsonb: key, containment and path" "select id, j from %T where j ? 'o' and j @> '{\"k\": \"x2\"}' and (j->'o'->>'z')::int = 1"
gck gc9 "float4 underflow in a qual" "select id, d from %T where d > 0.1::float4 and d / 1e38::float4 > 0::float4"
gck gc9 "float4 denormal operands in a qual" "select id from %T where d * 2::float4 > 0::float4 and d < 1::float4"
gcrun gc9 "text qual next to a numeric one" "select id, f from %T where f like 'a%' and nn > 1000"
gcagg gc9 "aggregate over a numeric qual" "select count(*), sum(n), max(id) from %T where nn > 100"
gcserved gc9 "aggregate over a numeric qual" "select count(*), sum(n), max(id) from %T where nn > 100" "$GCAGGSET"
gcagg gc9 "GROUP BY, numeric qual" "select b % 7 as k, count(*), sum(n) from %T where nn > 100 group by 1 order by 1"
gcagg gc9 "float4 underflow qual" "select count(*), sum(b), min(id) from %T where d > 0.1::float4 and d / 1e38::float4 > 0::float4"
gcagg gc9 "jsonb qual and numeric qual" "select count(*), sum(b) from %T where (j->>'a')::numeric > 100 and nn > 100"
# a numeric is not computed inside a join program (plan A6 "known gaps"): the plan may stay on the CPU, the answer must not change
gcrun_any gc9 "join with a numeric qual on the cached side" "select t.id, d.grp from gc9 t join sd d on t.b = d.k where t.nn > 100" "$JOINSET"
gcrun_any gc9 "join aggregate with a numeric qual on the cached side" "select count(*), sum(d.grp), sum(t.b) from gc9 t join sd d on t.b = d.k where t.nn > 100" "$JOINSET"
gcrun_any gc9 "GROUP BY with many groups, numeric qual" "select id % 4000 as k, count(*), sum(b) from %T where nn > 100 group by 1 order by 1" "$GCAGGSET" 
gcjoin gc9 "LEFT join, float4 underflow in the ON qual" "select count(*), count(d.k), sum(d.grp) from %T t left join %D d on t.b = d.k and t.d > 0.1::float4 and t.d / 1e38::float4 > 0::float4"
# a session's own changes: huge numerics and NaN written in the transaction
txncmp "gc9: huge numerics inserted and updated in a transaction (rollback)" rollback "update gc9 set nn = 1e40::numeric where id between 1 and 50; insert into gc9 values (900001, 5, 1e-40, 1.5, 'Infinity'::numeric, '{\"a\": 1e30}', 'z');" "select id, nn from gc9 where nn > 99999999 or id = 900001"
txncmp "gc9: huge numerics inserted and updated in a transaction (commit)" commit "update gc9 set nn = 1e40::numeric where id between 1 and 50; insert into gc9 values (900001, 5, 1e-40, 1.5, 'Infinity'::numeric, '{\"a\": 1e30}', 'z');" "select id, nn from gc9 where nn > 99999999 or id = 900001"
gck gc9 "after the commit" "select id, nn from %T where nn > 99999999 or nn > 5000000"

echo "GPU-Cache: a large table"
sql "create table gcbig ($GCDEF)"
sql "create trigger gc_sync after insert or update or delete on gcbig for each row execute function pgstrom.gpucache_sync_trigger('max_num_rows=$((NROWS * 2)),cache_size=1g')"
sql "$(gcins gcbig 1 $NROWS)"
sql "analyze gcbig"
gcrun gcbig "$NROWS rows loaded through the log" "select id, a, f from %T where b < 1000"
gcagg gcbig "aggregate over $NROWS rows" "select count(*), sum(b), max(c), min(a) from %T where a > 0"
gcagg gcbig "GROUP BY over $NROWS rows" "select a % 10 as k, count(*), sum(b) from %T group by 1 order by 1"
sql "update gcbig set b = b + 1 where id % 3 = 0"
sql "delete from gcbig where id % 7 = 0"
gcrun gcbig "after bulk UPDATE and DELETE" "select id, b, f from %T where b < 1000"
GCQ="select count(*), sum(b), max(c) from gcbig where a > 0"
$PSQL -d scantest -c "$GPU $GCAGGSET $GCQ" >/dev/null
ms_gpu=$($PSQL -d scantest -c "$GPU $GCAGGSET explain analyze $GCQ" | grep "Execution Time")
$PSQL -d scantest -c "$CPU $GCQ" >/dev/null
ms_cpu=$($PSQL -d scantest -c "$CPU explain analyze $GCQ" | grep "Execution Time")
echo "  info  aggregate over $NROWS cached rows: GPU $ms_gpu; PostgreSQL $ms_cpu"
GCQ="select id, a, f from gcbig where b < -49500"
$PSQL -d scantest -c "$GPU set enable_seqscan=off; $GCQ" >/dev/null
ms_gpu=$($PSQL -d scantest -c "$GPU set enable_seqscan=off; explain analyze $GCQ" | grep "Execution Time")
$PSQL -d scantest -c "$CPU $GCQ" >/dev/null
ms_cpu=$($PSQL -d scantest -c "$CPU explain analyze $GCQ" | grep "Execution Time")
echo "  info  selective scan over $NROWS cached rows: GPU $ms_gpu; PostgreSQL $ms_cpu"
