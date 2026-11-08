# The memory bound of the group table (pg_strom.gpuserv_group_memory = 64 MB, set by gtab_dev.sh): half bounds the device table, the whole the
# groups kept on the host. Needs q, GPU, CPU, runagg, check, sql, PSQL.
echo "GROUP BY under pg_strom.gpuserv_group_memory = 64"
sql "drop table if exists gmem" >/dev/null
sql "create table gmem as select i::int4 as k, (i % 1000)::int4 as v from generate_series(1, 1500000) i" >/dev/null
sql "vacuum analyze gmem" >/dev/null
# more groups than the device table holds (at most 32 MB): the rows that find no slot take the CPU path, the result is exact
runagg gmem "300,000 groups: the table is capped, the CPU path takes the rest" "select k, count(*), sum(v), max(v) from %T where k <= 300000 group by 1 order by 1"
# more groups than the session may keep (1.5M x ~180 bytes > 64 MB): a clean error (or the CPU's result if a fallback took over), never a wrong answer
BIG="set pg_strom.gpupreagg_max_groups_heap=100000000; set pg_strom.gpupreagg_max_groups=100000000;"
out=$($PSQL -d scantest -c "$GPU $BIG select count(*), sum(c), sum(s) from (select k, count(*) as c, sum(v) as s from gmem group by k) x" 2>&1 | tail -1)
ref=$($PSQL -d scantest -c "$CPU select count(*), sum(c), sum(s) from (select k, count(*) as c, sum(v) as s from gmem group by k) x" 2>&1 | tail -1)
if echo "$out" | grep -q "ERROR"; then
	echo "$out" | grep -q "gpuserv_group_memory"; check "1,500,000 groups exceed the session's memory: the error names pg_strom.gpuserv_group_memory ($out)" $?
else
	[ "$out" = "$ref" ]; check "1,500,000 groups: the result equals the CPU's ($out)" $?
fi
