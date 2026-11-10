# GROUP BY keys over a join (A3, join part): three and four keys, and float keys. The workgroup tables of jgagg.slang hold four
# keys (SCAN_GROUP_KEYS_JOIN); a float key is staged as x + 0.0 (-0 = +0, NaN = NaN) like the scan's. Sourced by keys_dev.sh (needs
# q, GPU, CPU, check, PSQL, fail and the tables tko / tki / tfo / tfi / tbo / tbi).
echo "GROUP BY keys over a join"
JKSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off; set pg_strom.gpupreagg_max_groups=1000000; set pg_strom.gpu_error_fallback=off;"
rungj() { # label, query (tables written out); the plan must be a GpuPreAgg over the join and the rows equal to the CPU's;
	# the CPU fallback is off, so a program the service refuses fails the check instead of running on the CPU
	label=$1; qry=$2; set_extra=${3:-}
	pl=$($PSQL -d scantest -c "$GPU $JKSET $set_extra explain $qry")
	plan=$(echo "$pl" | grep -c "Custom Scan (GpuPreAgg)")
	g=$(q "$GPU $JKSET $set_extra" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	elif [ "$ALLOW_CPU" = on ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows; plan differs on this host)"
	else echo "  FAIL  $label (plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; echo "$pl" | head -14 | sed 's/^/        plan| /'; fail=$((fail+1)); fi
}
# tjo (200,000 rows) joins tjd (3,000 rows) on k; tjo.k = i % 4000, so a quarter of tjo has no partner. The keys are plain columns with few values.
J="tjo t join tjd d on t.k = d.k"
rungj "two keys over a join (the existing kernel path)" "select t.a, d.x, count(*), sum(t.pay) from $J group by 1, 2"
rungj "three keys: outer, inner, outer" "select t.a, d.x, t.b, count(*), sum(t.pay), min(d.w), max(d.k) from $J group by t.a, d.x, t.b"
rungj "three keys, one of them mostly NULL (outer)" "select t.an, d.x, t.c, count(*), sum(d.w) from $J group by t.an, d.x, t.c"
rungj "four keys from both sides" "select t.a, d.x, d.y, t.b, count(*), sum(t.pay), max(d.w) from $J group by t.a, d.x, d.y, t.b"
rungj "four keys with NULLs on both sides" "select t.an, d.xn, t.c, d.y, count(*), count(d.xn), sum(t.pay) from $J group by t.an, d.xn, t.c, d.y"
rungj "more groups than a workgroup's table holds (the overflow list), four keys" "select t.a, t.b, t.e, d.z, count(*), sum(d.w) from $J group by t.a, t.b, t.e, d.z"
rungj "three keys over a LEFT join (NULL-extended inner keys)" "select t.a, d.x, d.y, count(*), count(d.k), sum(t.pay) from tjo t left join tjd d on t.k = d.k group by t.a, d.x, d.y"
rungj "three keys over a RIGHT join" "select t.a, d.x, d.y, count(*), count(t.id), sum(d.w) from tjo t right join tjd d on t.k = d.k and t.a < 4 group by t.a, d.x, d.y"
rungj "four keys over a FULL join" "select t.b, t.c, d.y, d.z, count(*), count(t.id), count(d.k) from tjo t full join tjd d on t.k = d.k and t.b < 3 group by t.b, t.c, d.y, d.z"
rungj "three keys, FILTER and min/max beyond 32 bits" "select t.a, d.x, d.y, count(*) filter (where t.pay % 2 = 0), max(t.pay * 100000), min(t.pay * 100000) from $J group by t.a, d.x, d.y"
rungj "three keys, semi join" "select t.a, t.b, t.c, count(*), sum(t.pay) from tjo t where t.k in (select d.k from tjd d where d.y < 2) group by t.a, t.b, t.c"
rungj "three keys, anti join" "select t.a, t.b, t.c, count(*), sum(t.pay) from tjo t where not exists (select 1 from tjd d where d.k = t.k and d.y < 3) group by t.a, t.b, t.c"
rungj "three keys, two joined relations" "select t.a, d.x, e.z, count(*), sum(e.w) from tjo t join tjd d on t.k = d.k join tjd e on e.k = d.k and e.y < 2 group by t.a, d.x, e.z"
rungj "three keys from the inner side only" "select d.x, d.y, d.z, count(*), sum(t.pay) from $J group by d.x, d.y, d.z"
rungj "float key (outer float4) with two ints" "select t.fa, d.x, t.a, count(*), sum(d.w) from $J group by t.fa, d.x, t.a"
rungj "float8 key of the inner side with -0, NaN, three keys" "select d.fz + 0::float8, t.fa, d.x, t.a, count(*), sum(t.pay) from $J group by d.fz, t.fa, d.x, t.a"
# float keys over a join: the key of the outer side, of the inner side (0 and -0, NaN, NULL, infinity), and mixed with integers
rungj "float4 key of the outer side" "select t.f4g, count(*), sum(d.w), min(t.n) from tfo t join tfi d on t.n = d.wn group by t.f4g"
rungj "float8 key of the outer side" "select t.f8g, count(*), sum(d.w), min(t.n) from tfo t join tfi d on t.n = d.wn group by t.f8g"
rungj "float4 key of the inner side (-0 and +0 are one group)" "select d.f4 + 0::float4, count(*), sum(t.pay) from tfo t join tfi d on t.n = d.wn group by d.f4"
rungj "float8 key of the inner side" "select d.f8 + 0::float8, count(*), sum(t.pay) from tfo t join tfi d on t.n = d.wn group by d.f8"
rungj "float4, float8 and an int" "select t.f4g, d.f8 + 0::float8, d.w % 3, count(*), sum(t.pay) from tfo t join tfi d on t.n = d.wn group by t.f4g, d.f8, d.w % 3"
rungj "float4 key of the outer side over a LEFT join" "select t.f4g, count(*), count(d.w) from tfo t left join tfi d on t.n = d.w group by t.f4g"
rungj "float key with a float aggregate (the slot owners add in order)" "select t.f4g, count(*), sum(t.f8g), min(t.f4g) from tfo t join tfi d on t.n = d.wn group by t.f4g"
rungj "float key with a float aggregate, three keys" "select t.f4g, t.n, d.w % 2, count(*), sum(t.f8g), max(t.f8g) from tfo t join tfi d on t.n = d.wn group by 1, 2, 3"
# bool keys over a join (already worked with up to two keys)
rungj "bool key over a join" "select t.b, count(*), sum(d.w) from tbo t join tbi d on t.n = d.w % 5 group by t.b"
rungj "four keys with a bool over a join" "select t.b, t.n, t.m, t.o, count(*), sum(t.pay), max(d.w) from tbo t join tbi d on t.b = d.b group by t.b, t.n, t.m, t.o"
rungj "three keys with a bool from the inner side" "select d.b, t.n, t.m, count(*), sum(t.pay) from tbo t join tbi d on t.b = d.b group by d.b, t.n, t.m"
# Arrow and Parquet outer relations go through the same kernel (ajgagg); the Arrow files of keys_dev.sh are not built here, so
# keys_tests.inc covers them with two keys and this file with the heap tables
echo
