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
echo
