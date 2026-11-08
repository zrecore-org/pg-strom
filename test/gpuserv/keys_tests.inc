# String join keys (A6): text, varchar and bpchar as hash-join keys, GpuJoin and GpuPreAgg over the join. The kernel hashes
# the payload bytes with PostgreSQL's hash_bytes() (bpchar without trailing blanks) and compares them bytewise. Sourced by
# keys_dev.sh (needs q, GPU, CPU, check, PSQL and the tables tko / tki).
echo "string join keys"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
JAGGBIG="set pg_strom.gpupreagg_max_groups=1000000;"
runkj() { # label, query-with-%T/%D; the plan must contain GpuJoin or GpuPreAgg-over-join; results equal to the CPU's
	label=$1; qry=$(printf '%s' "$2" | sed "s/%T/tko/g; s/%D/tki/g"); set_extra=${3:-}
	pl=$($PSQL -d scantest -c "$GPU $JOINSET $JAGGBIG $set_extra explain $qry")
	plan=$(echo "$pl" | grep -c "Custom Scan (Gpu\(Join\|PreAgg\))")
	g=$(q "$GPU $JOINSET $JAGGBIG $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	elif [ "$ALLOW_CPU" = on ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows; plan differs on this host)"
	else echo "  FAIL  $label (plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
}
for col in k kv kb big ext; do
	runkj "$col: inner join" "select t.i, d.i from %T t join %D d on t.$col = d.$col"
	runkj "$col: LEFT join" "select t.i, d.i from %T t left join %D d on t.$col = d.$col"
	runkj "$col: semi join" "select t.i from %T t where t.$col in (select d.$col from %D d)"
	runkj "$col: anti join" "select t.i from %T t where not exists (select 1 from %D d where d.$col = t.$col)"
	runkj "$col: count, sum over the join" "select count(*), sum(t.pay), max(d.grp), min(d.i) from %T t join %D d on t.$col = d.$col"
	runkj "$col: GROUP BY an integer over the join" "select t.n, count(*), sum(d.grp) from %T t join %D d on t.$col = d.$col group by t.n"
done
runkj "text key and an integer key" "select t.i, d.i from %T t join %D d on t.k = d.k and t.n = d.n"
runkj "text key with a qual on both sides" "select t.i, d.i from %T t join %D d on t.k = d.k where t.n < 20 and d.grp > 300"
runkj "RIGHT join on a text key" "select t.i, d.i from %T t right join %D d on t.k = d.k"
runkj "FULL join on a bpchar key" "select t.i, d.i from %T t full join %D d on t.kb = d.kb"
# Arrow, Parquet and GPU-Cache outer relations (the inner side is the heap table tki)
for src in tkoa tkop tkog; do
	for col in k kv kb; do
		runkj "$src.$col: inner join" "select t.i, d.i from $src t join %D d on t.$col = d.$col" "set pg_strom.enabled=on;"
		runkj "$src.$col: LEFT join" "select t.i, d.i from $src t left join %D d on t.$col = d.$col"
		runkj "$src.$col: count, sum, max over the join" "select count(*), sum(t.pay), max(d.grp) from $src t join %D d on t.$col = d.$col"
		runkj "$src.$col: GROUP BY an integer over the join" "select t.n, count(*), sum(d.grp) from $src t join %D d on t.$col = d.$col group by t.n"
	done
done
# an Arrow file as the inner relation
runkj "heap outer, Arrow inner, text key" "select t.i, a.i from %T t join tkoa a on t.k = a.k where t.n < 10"
# Timing (informational): aggregation over a join on a text key against an integer key of the same shape. The text key makes the
# join program poison-capable (rck_walk), so the aggregating kernel runs a scout pass first; the integer key does not.
tm() { # label, query: milliseconds of three runs on the GPU path, and on the CPU
	best=999999999
	for r in 1 2 3; do
		t0=$(perl -MTime::HiRes=time -e 'printf "%d", time()*1000')
		$PSQL -d scantest -c "$GPU $JOINSET $JAGGBIG $2" >/dev/null
		t1=$(perl -MTime::HiRes=time -e 'printf "%d", time()*1000')
		[ $((t1 - t0)) -lt $best ] && best=$((t1 - t0))
	done
	echo "  time  $1: ${best} ms"
}
tm "GPU, aggregation over a join, text key" "select t.n, count(*), sum(d.grp) from tko t join tki d on t.k = d.k group by t.n"
tm "GPU, aggregation over a join, int key" "select t.n, count(*), sum(d.grp) from tko t join tki d on t.n = d.n group by t.n"
tm "GPU, join without aggregation, text key" "select count(*) from (select t.i, d.i from tko t join tki d on t.k = d.k) x"
echo
