# A10 part 5b: variance, stddev, var_pop, var_samp, stddev_pop, stddev_samp of int2, int4 and int8 on the GPU. The device accumulates
# PostgreSQL's integer accumulator state EXACTLY (N, the sum of x as a signed 128-bit and the sum of x^2 as an unsigned 192-bit
# integer: kernels_vm/scan_common.slang ivs_acc), in any order, and the Agg node's final function hands the state to PostgreSQL's
# own numeric_var_samp / numeric_stddev_pop ... (aggfuncs.c __ivar_final), so the numeric result must equal the CPU's digit for digit,
# with no rounding in the queries. Sourced by intvar_dev.sh (it provides q, check, sql, runagg, runagg_any, $PSQL, $GPU, $CPU, $fail).
echo "variance / stddev of integers (A10 part 5b)"
IVROWS=${IVROWS:-200000}
# a: int2, b: int4 with NULLs, c: int8 over the whole range (the squares need up to 126 bits and the sums more than 64), s: small int8,
# k: constant, jk: a join key; the rows i % 6 = 0 / 1 of c are at the ends of the int8 range
sql "create table iv_t as select i, (i % 5)::int4 as g, (i % 7)::int4 as g2, (i % 300)::int4 as g300,
	((((i::int8 * 7919) % 65536) - 32768))::int2 as a,
	case when i % 11 = 0 then null else (((i::int8 * 1103515245) % 4294967296) - 2147483648)::int4 end as b,
	case i % 6 when 0 then 9223372036854775807 - (i % 100) when 1 then -9223372036854775807 + (i % 100)
	     else ((i::int8 * 7919) % 2000001 - 1000000) * 4611686018427 + (i % 1000) end as c,
	case when i % 13 = 0 then null else ((i::int8 * 31) % 1001 - 500) end as s,
	7::int4 as k,
	((i % 30000) * 3 - 50000)::int4 as jk
	from generate_series(1, $IVROWS) i"
sql "vacuum analyze iv_t"
sql "create table iv_dim as select (i * 3 - 50000)::int4 as k, ((i::int8 * 7919) % 2001 - 1000)::int8 * 3000000000 as v, ((i * 37) % 60000 - 30000)::int2 as u, (i % 5)::int4 as grp
	from generate_series(0, 40000) i"
sql "analyze iv_dim"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
GRPANY=0
rungrp() { # table, label, query-with-%T [, extra settings]
	tbl=$1; label=$2; qry=$(printf '%s' "$3" | sed "s/%T/$tbl/g; s/ order by [0-9, ]*\$//"); set_extra=${4:-}
	gs="$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; $set_extra"
	plan=$($PSQL -d scantest -c "$gs explain $qry" | grep -c "Custom Scan (GpuPreAgg)")
	g=$($PSQL -d scantest -c "$gs $qry" 2>&1 | sort); c=$($PSQL -d scantest -c "$CPU $qry" 2>&1 | sort)
	if { [ "$plan" -ge 1 ] || [ "$GRPANY" = 1 ]; } && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR && [ -n "$g" ]; then echo "  ok    $tbl: $label ($(echo "$g" | wc -l | tr -d ' ') rows)"
	else echo "  FAIL  $tbl: $label (plan GpuPreAgg=$plan)"; echo "        gpu: $(echo "$g" | head -3)"; echo "        cpu: $(echo "$c" | head -3)"; fail=$((fail+1)); fi
}
# a program has at most 12 accumulators (SCAN_AGG_MAX) and a variance takes three: four of them, or three and a count
F4() { echo "var_pop($1), var_samp($1), stddev_pop($1), stddev_samp($1)"; }
F2() { echo "variance($1), stddev($1)"; }

echo "no GROUP BY"
runagg iv_t "int2: var_pop, var_samp, stddev_pop, stddev_samp" "select $(F4 a) from %T"
runagg iv_t "int4 with NULLs" "select $(F4 b) from %T"
runagg iv_t "int8 over the whole range (192-bit sums of squares)" "select $(F4 c) from %T"
runagg iv_t "int8 of moderate size and NULLs" "select $(F4 s) from %T"
runagg iv_t "variance and stddev aliases with count" "select $(F2 a), variance(c), count(*) from %T"
runagg iv_t "three types at once (nine accumulators)" "select var_pop(a), stddev_samp(b), variance(c) from %T"
runagg iv_t "variance next to count, sum, min, max" "select count(*), sum(a), min(b), max(c), var_samp(s), avg(a) from %T"
runagg iv_t "constant values: variance 0, stddev 0" "select var_pop(k), var_samp(k), stddev_pop(k), stddev(k) from %T"
runagg iv_t "one row: var_samp is NULL, var_pop is 0" "select var_samp(a) filter (where i = 17), var_pop(a) filter (where i = 17), stddev_samp(c) filter (where i = 17), stddev_pop(c) filter (where i = 17) from %T"
runagg iv_t "two rows" "select var_samp(c) filter (where i >= 17 and i <= 18), var_pop(b) filter (where i >= 17 and i <= 18), stddev(a) filter (where i < 3), stddev_pop(s) filter (where i < 3) from %T"
runagg iv_t "no row qualifies: NULL" "select var_pop(a), stddev(c), variance(b), count(*) from %T where i < 0"
runagg iv_t "only NULL arguments: NULL" "select var_pop(b), var_samp(b), stddev(b) from %T where i % 11 = 0"
runagg iv_t "the extremes of int8: 9223372036854775807 and -9223372036854775807" "select var_pop(c), var_samp(c), stddev_pop(c), stddev_samp(c) from %T where i % 6 < 2"
runagg iv_t "the extremes of int2 and int4" "select var_pop(a), stddev_samp(a), var_samp(b), stddev_pop(b) from %T where i % 97 < 3"
runagg iv_t "FILTER" "select var_pop(a) filter (where i % 3 = 0), stddev(b) filter (where i % 4 = 1), var_samp(c) filter (where g = 2), var_pop(s) filter (where false) from %T"
runagg iv_t "a qual" "select var_pop(c), stddev(a), variance(b), count(*) from %T where i % 7 = 1 and a > 0"
runagg iv_t "computed arguments" "select var_pop(a + 1), stddev_samp(a * 2), var_samp(b / 3), variance(s - 1) from %T where i % 5 = 2"
runagg iv_t "the four aggregates of an int8 column with different filters (twelve accumulators)" "select var_pop(c) filter (where g = 0), var_samp(c) filter (where g = 1), stddev_pop(c) filter (where g = 2), stddev_samp(c) filter (where g = 3) from %T"

echo "GROUP BY"
rungrp iv_t "per group, every function, int2 / int4 / int8" "select g, $(F4 a) from %T group by 1 order by 1"
rungrp iv_t "per group, int4 and int8" "select g, var_pop(b), stddev_samp(b), var_pop(c), stddev_samp(c) from %T group by 1 order by 1"
rungrp iv_t "per group of a second key" "select g2, var_samp(s), stddev_pop(s), variance(c) from %T group by 1 order by 1"
rungrp iv_t "two group keys" "select g, g2, var_pop(a), stddev(c) from %T group by 1, 2 order by 1, 2"
GRPANY=1	# (the planner prices 300 groups above the CPU plan on a small table: the result must still be equal)
rungrp iv_t "300 groups" "select g300, var_pop(c), stddev_samp(b), variance(a) from %T group by 1 order by 1"
GRPANY=0
rungrp iv_t "groups of one and of two rows (FILTER i < 10 leaves one or two rows per group)" "select g, var_samp(a) filter (where i < 10), var_pop(a) filter (where i < 10), stddev_samp(c) filter (where i < 10), var_samp(s) filter (where i < 10) from %T group by 1 order by 1"
rungrp iv_t "groups with no qualifying row" "select g, var_pop(a) filter (where false), stddev(c) filter (where i < 0), var_pop(s) from %T group by 1 order by 1"
rungrp iv_t "FILTER per group" "select g, var_pop(a) filter (where i % 3 = 0), stddev(b) filter (where i % 4 = 1), var_samp(c) filter (where g2 = 2) from %T group by 1 order by 1"
rungrp iv_t "constants per group" "select g, var_pop(k), stddev_samp(k) from %T group by 1 order by 1"
rungrp iv_t "variance next to other aggregates per group" "select g, count(*), sum(a), min(b), max(c), var_pop(s), avg(a) from %T group by 1 order by 1"

echo "over a join"
rungrp iv_t "variance over a join" "select var_pop(d.v), stddev(t.c), var_samp(d.u), count(*) from %T t join iv_dim d on t.jk = d.k" "$JOINSET"
rungrp iv_t "variance over a join with a FILTER and a qual" "select var_samp(d.v) filter (where d.grp < 3), stddev_pop(t.b), count(*) from %T t join iv_dim d on t.jk = d.k where t.a > d.u" "$JOINSET"
rungrp iv_t "variance over a join per group" "select d.grp, var_pop(d.v), stddev_samp(t.c), variance(d.u), count(*) from %T t join iv_dim d on t.jk = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
GRPANY=1
rungrp iv_t "variance over a join, many groups (overflow rows)" "select d.k, var_pop(t.c), stddev(t.a) from %T t join iv_dim d on t.jk = d.k group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
GRPANY=0
rungrp iv_t "variance over a LEFT join" "select d.grp, var_pop(t.a), count(*) from iv_dim d left join %T t on t.jk = d.k and t.i < 3000 group by 1 order by 1" "$JOINSET set enable_seqscan=on;"
echo "plans"
for fn in "var_pop(a)" "stddev(b)" "variance(c)"; do
	n=$($PSQL -d scantest -c "$GPU set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; explain select $fn from iv_t" | grep -c "Custom Scan (GpuPreAgg)")
	check "$fn plans as GpuPreAgg" "$([ "$n" -ge 1 ]; echo $?)"
done
# numeric stays on the CPU and agrees
runagg_any iv_t "variance of numeric stays on the CPU" "select var_pop(s::numeric), stddev(a::numeric) from %T"
