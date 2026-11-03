# A6: timing of numeric aggregates (GPU vs CPU) on NMBENCH rows (default 3,000,000).
#   NUMINC=numeric_bench.inc sh test/gpuserv/locked_test.sh <tree> numeric_test.sh
NMBENCH=${NMBENCH:-3000000}
echo "numeric bench ($NMBENCH rows)"
sql "create table nb_li as select i as l_orderkey, (i % 5)::int as grp, ((i * 37) % 50 + 1)::numeric(15,2) as l_quantity,
	(((i::int8 * 7919) % 10000000) / 100.0)::numeric(15,2) as l_extendedprice, ((i % 11) / 100.0)::numeric(15,2) as l_discount,
	((i % 9) / 100.0)::numeric(15,2) as l_tax, (date '1992-01-01' + (i % 2500)) as l_shipdate
	from generate_series(1, $NMBENCH) i"
sql "vacuum analyze nb_li"
tm() { # label, settings, query -> elapsed ms and the result
	s=$(perl -MTime::HiRes=time -e 'print time')
	r=$($PSQL -d scantest -c "$2 $3" 2>&1 | tail -1)
	e=$(perl -MTime::HiRes=time -e 'print time')
	printf "  %-34s %8.0f ms  %s\n" "$1" "$(echo "($e - $s) * 1000" | bc -l)" "$r"
}
for q in \
 "select sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)), sum(l_extendedprice * (1 - l_discount)), avg(l_quantity), avg(l_discount), count(*) from nb_li where l_shipdate <= date '1998-09-02'" \
 "select sum(l_extendedprice * l_discount) from nb_li where l_shipdate >= date '1994-01-01' and l_shipdate < date '1995-01-01' and l_discount between 0.05 and 0.07 and l_quantity < 24" \
 "select grp, sum(l_extendedprice * (1 - l_discount)), avg(l_quantity), count(*) from nb_li group by 1 order by 1"; do
	echo "query: $q"
	for k in 1 2; do tm "GPU (run $k)" "$GPU set pg_strom.gpupreagg_max_groups=1000000;" "$q"; done
	for k in 1 2; do tm "CPU (run $k)" "$CPU" "$q"; done
	tm "CPU 2 workers" "set max_parallel_workers_per_gather=2; set pg_strom.enabled=off;" "$q"
done
