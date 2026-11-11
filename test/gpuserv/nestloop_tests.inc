# Nest-loop join tests (A4), sourced by nestloop_dev.sh after the helpers (check, sql, q, GPU, CPU, fail, ROOT, PSQL, NROWS).
# A join without a hash key (non-equi, range, cross, IS NOT DISTINCT FROM) is a GpuJoin whose inner rows all sit on one hash chain
# (a constant key): every outer row is compared with every inner row. Each query must plan as "GPU Nest Loop" and return the
# rows of the same query with pg_strom.enabled = off.
echo "GPU nest-loop joins"
sql "create table nlo as select i as id,
	case when i % 11 = 0 then null else (i % 1000)::int4 end as a,
	(i * 7 % 5000)::int8 as b,
	case when i % 13 = 0 then null else date '2020-01-01' + (i % 400) end as c,
	(i % 50)::int2 as x,
	case when i % 17 = 0 then null else 'k' || (i % 40) end as t
	from generate_series(1, $NROWS) i"
sql "create table nli as select j as w,
	case when j % 7 = 0 then null else ((j * 5) % 1000)::int4 end as lo,
	case when j % 7 = 0 then null else ((j * 5) % 1000 + 3)::int4 end as hi,
	case when j % 9 = 0 then null else date '2020-01-01' + ((j * 3) % 400) end as d,
	(j % 50)::int2 as y,
	case when j % 10 = 0 then null else 'k' || (j % 40) end as t,
	(j * 11)::int8 as pay
	from generate_series(1, 200) j"
sql "create table nli2 as select j as w2, (j % 20)::int4 as m, (j % 13)::int4 as lo2 from generate_series(1, 40) j"
sql "vacuum analyze nlo"; sql "analyze nli"; sql "analyze nli2"

NLSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
nlrun() { # label, query [, extra settings]; the plan must be a GpuJoin with a nest loop, the rows equal to the CPU's
	label=$1; qry=$2; set_extra=${3:-}
	pl=$($PSQL -d scantest -c "$GPU $NLSET $set_extra explain $qry")
	plan=$(echo "$pl" | grep -c "GPU Nest Loop")
	g=$(q "$GPU $NLSET $set_extra set pg_strom.gpu_error_fallback=off;" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows)"
	elif [ "$ALLOW_CPU" = on ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    $label (${g#*|} rows; plan differs on this host)"
	else echo "  FAIL  $label (nest loop in plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; echo "$pl" | head -14 | sed 's/^/        plan| /'; fail=$((fail+1)); fi
}

nlrun "range: a between lo and hi" "select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi"
nlrun "non-equi: a < lo on a small outer and inner" "select o.id, i.w from nlo o join nli i on o.a < i.lo where o.id <= 100 and i.w < 20"
nlrun "non-equi: a > hi and a < hi + 3, NULLs on both sides" "select o.id, i.w, i.hi from nlo o join nli i on o.a > i.hi and o.a < i.hi + 3"
nlrun "date range with a computed bound" "select o.id, i.w from nlo o join nli i on o.c >= i.d and o.c < i.d + 1 where o.id <= 3000"
nlrun "int8 against int4 (b > pay / 11 + 440)" "select o.id, i.w from nlo o join nli i on o.b > i.pay / 11 * 20 and o.b <= i.pay / 11 * 20 + 40 where o.id <= 2000"
nlrun "<> and a range" "select o.id, i.w from nlo o join nli i on o.x <> i.y and o.a between i.lo and i.lo + 1 where o.id <= 5000"
nlrun "projection mixing both sides" "select o.id + i.w, o.b * 2, i.pay - o.a from nlo o join nli i on o.a between i.lo and i.hi where o.id <= 8000"
nlrun "ctid of both sides" "select o.ctid, i.ctid from nlo o join nli i on o.a between i.lo and i.hi and i.w < 40 where o.id <= 8000"
# outer joins
nlrun "LEFT join: unmatched outer rows" "select o.id, i.w from nlo o left join nli i on o.a between i.lo and i.hi where o.id <= 6000"
nlrun "LEFT join, ON qual on the inner side only" "select o.id, i.w from nlo o left join nli i on o.a between i.lo and i.hi and i.w > 100 where o.id <= 6000"
nlrun "LEFT join, ON qual on the outer side only" "select o.id, i.w from nlo o left join nli i on o.a between i.lo and i.hi and o.x < 10 where o.id <= 6000"
nlrun "LEFT join, nothing matches" "select o.id, i.w from nlo o left join nli i on o.a > i.hi + 5000 where o.id <= 3000"
nlrun "RIGHT join: unmatched inner rows" "select o.id, i.w from nlo o right join nli i on o.a between i.lo and i.hi and o.id <= 500"
nlrun "RIGHT join, nothing matches" "select o.id, i.w from nlo o right join nli i on o.a > i.hi + 5000 where true"
# PostgreSQL cannot run a FULL join without a hash or merge key at all, so the oracle is the LEFT join plus the inner rows without a partner
FULLQ="select o.id, i.w from nlo o full join nli i on o.a between i.lo and i.hi and o.id <= 400 and i.w <= 150"
FULLO="select o.id, i.w from nlo o left join nli i on o.a between i.lo and i.hi and o.id <= 400 and i.w <= 150 union all select null, i.w from nli i where not exists (select 1 from nlo o where o.a between i.lo and i.hi and o.id <= 400 and i.w <= 150)"
plan=$($PSQL -d scantest -c "$GPU $NLSET explain $FULLQ" | grep -c "GPU Nest Loop")
g=$(q "$GPU $NLSET set pg_strom.gpu_error_fallback=off;" "$FULLQ"); c=$(q "$CPU" "$FULLO")
if [ "$plan" -ge 1 ] && [ "$g" = "$c" ]; then echo "  ok    FULL join (${g#*|} rows, oracle: LEFT join + anti join)"
else echo "  FAIL  FULL join (nest loop in plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
# semi and anti
nlrun "EXISTS with a range" "select o.id from nlo o where o.id <= 8000 and exists (select 1 from nli i where o.a between i.lo and i.hi and i.w > 30)"
nlrun "EXISTS with a non-equi qual" "select o.id from nlo o where o.id <= 3000 and exists (select 1 from nli i where i.w < 10 and o.a < i.lo)"
nlrun "NOT EXISTS with a range" "select o.id from nlo o where o.id <= 8000 and not exists (select 1 from nli i where o.a between i.lo and i.hi)"
nlrun "NOT EXISTS with a non-equi qual" "select o.id from nlo o where o.id <= 3000 and not exists (select 1 from nli i where i.w < 10 and o.a < i.lo)"
# IS NOT DISTINCT FROM
nlrun "IS NOT DISTINCT FROM, NULLs match NULLs" "select o.id, i.w from nlo o join nli i on o.a is not distinct from i.lo where o.id <= 4000"
nlrun "IS NOT DISTINCT FROM, LEFT join" "select o.id, i.w from nlo o left join nli i on o.c is not distinct from i.d where o.id <= 800"
nlrun "IS NOT DISTINCT FROM on text, anti" "select o.id from nlo o where o.id <= 3000 and not exists (select 1 from nli i where o.t is not distinct from i.t and i.w < 25)"
nlrun "IS NOT DISTINCT FROM and a range together" "select o.id, i.w from nlo o join nli i on o.a is not distinct from i.lo and o.b >= i.pay where o.id <= 20000"
# cross join
nlrun "cross join with qual on both sides" "select o.id, i.w from nlo o cross join nli i where o.id <= 150 and i.w <= 40"
nlrun "cross join, LEFT with an ON true" "select o.id, i.w from nlo o left join nli i on true where o.id <= 100 and i.w <= 30" 
# stacked: a nest loop with a hash join and a second nest loop
nlrun "nest loop then hash join" "select o.id, i.w, e.w2 from nlo o join nli i on o.a between i.lo and i.hi join nli2 e on e.m = o.x where o.id <= 3000"
nlrun "hash join then nest loop" "select o.id, i.w, e.w2 from nlo o join nli2 e on e.m = o.x join nli i on o.a between i.lo and i.hi where o.id <= 3000"
nlrun "two nest loops" "select o.id, i.w, e.w2 from nlo o join nli i on o.a between i.lo and i.hi join nli2 e on o.x < e.lo2 and o.x > e.lo2 - 2 where o.id <= 3000"
nlrun "nest loop then LEFT nest loop" "select o.id, i.w, e.w2 from nlo o join nli i on o.a between i.lo and i.hi left join nli2 e on o.x = e.m + 100 or o.x > e.lo2 + 40 where o.id <= 2000"
# aggregation over a nest loop (GpuPreAgg over the join)
nlrun "aggregate over a nest loop" "select count(*), sum(o.b), min(i.pay), max(o.id) from nlo o join nli i on o.a between i.lo and i.hi"
nlrun "GROUP BY over a nest loop" "select i.y, count(*), sum(o.b) from nlo o join nli i on o.a between i.lo and i.hi group by i.y"

# parallel workers must not change the rows
nlrun "parallel outer scan" "select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi" "set max_parallel_workers_per_gather=2; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"

# the pair cap: a join over the limit stays on the CPU, with the same rows
for cap in 1000 100000; do
	qry="select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi"
	plan=$($PSQL -d scantest -c "$GPU $NLSET set pg_strom.gpujoin_nestloop_max_pairs=$cap; explain $qry" | grep -c "GPU Nest Loop")
	g=$(q "$GPU $NLSET set pg_strom.gpujoin_nestloop_max_pairs=$cap;" "$qry"); c=$(q "$CPU" "$qry")
	if [ "$plan" -eq 0 ] && [ "$g" = "$c" ]; then echo "  ok    pair cap $cap: no GPU nest loop, rows equal (${g#*|} rows)"
	else echo "  FAIL  pair cap $cap (nest loop in plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
done
qry="select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi"
plan=$($PSQL -d scantest -c "$GPU $NLSET set pg_strom.enable_gpunestloop=off; explain $qry" | grep -c "GPU Nest Loop")
check "pg_strom.enable_gpunestloop=off: no GPU nest loop" "$([ "$plan" -eq 0 ]; echo $?)"
# an equi join keeps its hash key
plan=$($PSQL -d scantest -c "$GPU $NLSET explain select o.id, i.w from nlo o join nli i on o.a = i.lo" | grep -c "GPU Nest Loop")
check "an equi join is still a hash join" "$([ "$plan" -eq 0 ]; echo $?)"

# the CPU fallback (A14) of a nest-loop node
fbcount() { grep -c "the CPU runs the rest of the node" "$ROOT/log"; }
for qry in "select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi" \
		   "select o.id, i.w from nlo o left join nli i on o.a between i.lo and i.hi and o.id <= 3000" \
		   "select o.id from nlo o where not exists (select 1 from nli i where o.a between i.lo and i.hi)"; do
	c=$(q "$CPU" "$qry")
	for f in refuse=2 drop; do
		b=$(fbcount)
		g=$(q "$GPU $NLSET set pg_strom.scan_chunk_blocks=16; set pg_strom.gpuserv_fault='$f';" "$qry"); a=$(fbcount)
		if [ "$g" = "$c" ] && [ "$a" -gt "$b" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    fallback $f: $(echo "$qry" | cut -c1-60) (${g#*|} rows)"
		else echo "  FAIL  fallback $f: $qry (fallbacks $b -> $a)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
	done
done
c=$(q "$CPU" "$FULLO")
for f in refuse=2 drop; do
	b=$(fbcount)
	g=$(q "$GPU $NLSET set pg_strom.scan_chunk_blocks=16; set pg_strom.gpuserv_fault='$f';" "$FULLQ"); a=$(fbcount)
	if [ "$g" = "$c" ] && [ "$a" -gt "$b" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    fallback $f: FULL join (${g#*|} rows)"
	else echo "  FAIL  fallback $f: FULL join (fallbacks $b -> $a)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
done

# the runtime guard: a stale estimate (z = 1 is planned for one row, every row has it) must not run the product on the GPU
sql "create table nls as select i as id, (i % 1000)::int4 as a, 0::int4 as z from generate_series(1, $NROWS) i"
sql "analyze nls"
sql "update nls set z = 1"
qry="select o.id, i.w from nls o join nli i on o.a between i.lo and i.hi where o.z = 1"
pre="$GPU $NLSET set pg_strom.gpujoin_nestloop_max_pairs=1e6;"
plan=$($PSQL -d scantest -c "$pre explain $qry" | grep -c "GPU Nest Loop")
c=$(q "$CPU" "$qry")
b=$(fbcount); g=$(q "$pre" "$qry"); a=$(fbcount)
if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && [ "$a" -gt "$b" ] && grep -q "exceed pg_strom.gpujoin_nestloop_max_pairs" "$ROOT/log" && ! echo "$g" | grep -q ERROR; then echo "  ok    runtime pair guard: a stale estimate continues on the CPU (${g#*|} rows)"
else echo "  FAIL  runtime pair guard (plan=$plan fallbacks $b -> $a)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi
pre="$GPU $NLSET set pg_strom.gpujoin_nestloop_max_pairs=1e12;"
b=$(fbcount); g=$(q "$pre" "$qry"); a=$(fbcount)
if [ "$g" = "$c" ] && [ "$a" -eq "$b" ]; then echo "  ok    the same query under a large cap stays on the GPU (${g#*|} rows)"
else echo "  FAIL  large cap (fallbacks $b -> $a)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi

# no pinned inner buffer for a nest loop (pg_strom.pinned_inner_buffer_threshold would otherwise take a big GpuScan inner)
sql "create table nlb as select j as w, (j % 1000)::int4 as lo, ((j % 1000) + 2)::int4 as hi, repeat('x', 60) as pad from generate_series(1, 100000) j"
sql "analyze nlb"
qry="select o.id, i.w from nlo o join nlb i on o.a between i.lo and i.hi where o.id <= 40"
pre="$GPU $NLSET set pg_strom.pinned_inner_buffer_threshold=1; set pg_strom.gpujoin_nestloop_max_pairs=1e12;"
plan=$($PSQL -d scantest -c "$pre explain $qry" | grep -c "GPU Nest Loop")
g=$(q "$pre" "$qry"); c=$(q "$CPU" "$qry")
if [ "$plan" -ge 1 ] && [ "$g" = "$c" ] && ! echo "$g" | grep -q ERROR; then echo "  ok    pinned inner buffer threshold set: the nest loop works (${g#*|} rows)"
else echo "  FAIL  pinned inner threshold (nest loop in plan=$plan)"; echo "        gpu: $g"; echo "        cpu: $c"; fail=$((fail+1)); fi

# NL_BENCH=1 (with a big NROWS, e.g. 2000000): wall time of the GPU nest loop against PostgreSQL's own plan (8 workers)
if [ -n "$NL_BENCH" ]; then
	echo "nest-loop timings (ms, best of 3): GPU / PostgreSQL"
	nlms() { perl -MTime::HiRes=time -e '$t=time; system(@ARGV); printf "%d\n", (time-$t)*1000' -- "$@"; }
	nlbest() { b=999999999; for k in 1 2 3; do t=$(nlms $PSQL -d scantest -c "$1 select count(*), sum(x.w) from ($2) x" | tail -1); [ "$t" -lt "$b" ] && b=$t; done; echo $b; }
	PGP="set max_parallel_workers_per_gather=8; set parallel_setup_cost=0; set parallel_tuple_cost=0; set min_parallel_table_scan_size=0;"
	for spec in "range|select o.id, i.w from nlo o join nli i on o.a between i.lo and i.hi" \
				"non-equi|select o.id, i.w from nlo o join nli i on o.a > i.hi and o.a < i.hi + 3" \
				"anti|select o.id, 1 as w from nlo o where not exists (select 1 from nli i where o.a between i.lo and i.hi)" \
				"null-safe|select o.id, i.w from nlo o join nli i on o.a is not distinct from i.lo"; do
		l=${spec%%|*}; s=${spec#*|}
		g=$(nlbest "$GPU $NLSET set pg_strom.gpujoin_nestloop_max_pairs=1e12;" "$s")
		c=$(nlbest "$CPU $PGP" "$s")
		echo "  $l: $g / $c"
	done
fi
