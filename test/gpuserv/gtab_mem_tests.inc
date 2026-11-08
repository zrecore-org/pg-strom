# The memory bound of the group table (pg_strom.gpuserv_group_memory = 64 MB, set by gtab_dev.sh): half bounds the device table, the whole the
# groups kept on the host. Needs q, GPU, CPU, runagg, check, sql, PSQL; the table gm of gtab_tests.inc.
echo "GROUP BY under pg_strom.gpuserv_group_memory = 64"
# 300,000 groups, three aggregates: about 55 MB on the host (fits), but the device table is capped at 262,144 slots (32 MB), so the rows that
# find no slot take the CPU path; the result is exact
runagg gm "300,000 groups: the device table is capped, the CPU path takes the rest" "select i, count(*), sum(a), max(b) from %T group by 1 order by 1"
# more groups than the session may keep (700,000 x ~180 bytes > 64 MB): a clean error (or the CPU's result if a fallback took over), never a wrong answer
sql "drop table if exists gmem" >/dev/null
sql "create table gmem as select i::int4 as k, (i % 1000)::int4 as v, (i % 77)::int4 as w from generate_series(1, 700000) i" >/dev/null
sql "vacuum analyze gmem" >/dev/null
BIG="set pg_strom.gpupreagg_max_groups_heap=100000000; set pg_strom.gpupreagg_max_groups=100000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"
QM="select k, count(*), sum(v), max(w) from gmem group by k"
plan=$($PSQL -d scantest -c "$GPU $BIG explain $QM" | grep -c "Custom Scan (GpuPreAgg)")
[ "$plan" -ge 1 ]; check "the 700,000-group query is planned on the GPU" $?
out=$($PSQL -d scantest -c "$GPU $BIG select count(*), sum(c) from ($QM offset 0) x(k, c, s, m)" 2>&1 | tail -1)
ref=$($PSQL -d scantest -c "$CPU select count(*), sum(c) from ($QM offset 0) x(k, c, s, m)" 2>&1 | tail -1)
if echo "$out" | grep -q "ERROR"; then
	echo "$out" | grep -q "gpuserv_group_memory"; check "700,000 groups exceed the session's memory: the error names pg_strom.gpuserv_group_memory ($out)" $?
else
	[ "$out" = "$ref" ]; check "700,000 groups: the result equals the CPU's ($out)" $?
fi
