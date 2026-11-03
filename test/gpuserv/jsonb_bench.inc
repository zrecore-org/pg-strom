# A8: timing of jsonb quals (GPU vs CPU) on JBBENCH rows (default 3,000,000), documents of about 100-150 bytes.
#   JBINC=jsonb_bench.inc sh test/gpuserv/locked_test.sh <tree> jsonb_test.sh
JBBENCH=${JBBENCH:-3000000}
echo "jsonb bench ($JBBENCH rows)"
sql "create table jb_bench as select i, (i % 5)::int as grp,
	format('{\"id\":%s,\"user\":\"user%s\",\"status\":\"%s\",\"score\":%s,\"price\":%s,\"tags\":[\"t%s\",\"t%s\"],\"geo\":{\"lat\":%s,\"city\":\"c%s\"},\"active\":%s,\"k5\":1,\"k6\":2,\"k7\":3,\"k8\":4}',
		i, i % 1000, (array['new','open','closed','hold'])[1 + i % 4], i % 1000, (i % 20000) / 100.0, i % 13, i % 29, (i % 180) - 90, i % 100, (i % 3 = 0)::text)::jsonb as d
	from generate_series(1, $JBBENCH) i"
sql "vacuum analyze jb_bench"
tm() { # label, settings, query -> elapsed ms and the result
	s=$(perl -MTime::HiRes=time -e 'print time')
	r=$($PSQL -d scantest -c "$2 $3" 2>&1 | tail -1)
	e=$(perl -MTime::HiRes=time -e 'print time')
	printf "  %-34s %8.0f ms  %s\n" "$1" "$(echo "($e - $s) * 1000" | bc -l)" "$r"
}
for q in \
 "select count(*) from jb_bench where d->>'status' = 'open'" \
 "select count(*) from jb_bench where (d->>'score')::int > 990" \
 "select count(*) from jb_bench where d ? 'k8' and d @> '{\"status\":\"closed\",\"active\":true}'" \
 "select count(*), sum((d->>'score')::int) from jb_bench where d->'geo'->>'city' = 'c7'" \
 "select grp, count(*), sum((d->>'score')::int), max((d->>'score')::int) from jb_bench where d @> '{\"status\":\"new\"}' group by 1 order by 1"; do
	echo "query: $q"
	for k in 1 2; do tm "GPU (run $k)" "$GPU set pg_strom.gpupreagg_max_groups=1000000;" "$q"; done
	for k in 1 2; do tm "CPU (run $k)" "$CPU" "$q"; done
	tm "CPU 2 workers" "set max_parallel_workers_per_gather=2; set pg_strom.enabled=off;" "$q"
done
