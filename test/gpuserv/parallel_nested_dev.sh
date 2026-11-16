#!/bin/sh
# A18 regression test (found by TPC-H Q18): a parallel GpuJoin whose inner side contains a GPU aggregate under a Gather must not run
# inside the parallel workers (every worker then saw only a part of the aggregated table: rows were lost), and the shared-memory
# buffers of a non parallel-aware GPU node that runs inside a worker must be released with the node (WARNING "shared-memory ...
# leaks, and still alive"). The Q18-shaped query runs with parallel workers and zero GPU costs and must return the rows of the CPU
# plan and print no WARNING. Run it through the wrapper:
#
#   sh test/gpuserv/locked_test.sh "$PWD" parallel_nested_dev.sh          (ROWS=<orders> default 1500000 = TPC-H SF1, smaller sizes do not give the plan shape; lineitem = 4 x ROWS)
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ROOT=${ROOT:-/tmp/pgstrom_parnest}
PORT=${PORT:-54437}
ROWS=${ROWS:-1500000}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
max_worker_processes = 24
max_parallel_workers = 16
fsync = off
CONF
trap stop EXIT INT TERM
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 300 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
[ $n -lt 300 ] || { echo "service not ready"; exit 1; }
$PSQL -d postgres -c "create database t" >/dev/null
$PSQL -d t -c "create extension pg_strom" >/dev/null 2>&1
$PSQL -d t <<SQL >/dev/null
create table customer as select i as c_custkey, 'Customer#' || i as c_name from generate_series(1, $ROWS / 10) i;
create table orders as select i as o_orderkey, (i % ($ROWS / 10)) + 1 as o_custkey, (i % 5000)::numeric(12,2) as o_totalprice, date '1995-01-01' + (i % 2000) as o_orderdate from generate_series(1, $ROWS) i;
create table lineitem as select (i / 4) + 1 as l_orderkey, (1 + (i::bigint * 7919) % 50)::numeric(12,2) as l_quantity from generate_series(0, 4 * $ROWS - 1) i;
alter table customer add primary key (c_custkey); alter table orders add primary key (o_orderkey);
create index on lineitem (l_orderkey);
vacuum analyze;
SQL
Q="select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
 from customer, orders, lineitem
 where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 130)
   and c_custkey = o_custkey and o_orderkey = l_orderkey
 group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate, o_orderkey"
CPU="set pg_strom.enabled=off; set max_parallel_workers_per_gather=4;"
GPU="set pg_strom.enabled=on; set max_parallel_workers_per_gather=4; set parallel_setup_cost=0; set parallel_tuple_cost=0;
 set min_parallel_table_scan_size=0; set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set pg_strom.gpu_scan_tuple_cost=0;"
c=$($PSQL -d t -c "$CPU $Q" 2>&1 | md5 -q)
g=$($PSQL -d t -c "$GPU $Q" 2>&1 | md5 -q)
rows=$($PSQL -d t -c "$CPU $Q" | wc -l)
nodes=$($PSQL -d t -c "$GPU explain $Q" | grep -c "Custom Scan (Gpu")
warn=$($PSQL -d t -c "$GPU $Q" 2>&1 >/dev/null | grep -c WARNING)
echo "rows (CPU plan): $rows, Gpu nodes in the GPU plan: $nodes, warnings: $warn"
fail=0
[ "$c" = "$g" ] && echo "  ok    the GPU plan returns the rows of the CPU plan" || { echo "  FAIL  the GPU plan differs from the CPU plan"; fail=1; }
[ "$warn" = 0 ] && echo "  ok    no WARNING (no leaked shared-memory buffer)" || { echo "  FAIL  $warn WARNING lines"; fail=1; }
[ "$rows" -ge 10 ] || { echo "  FAIL  the query returns too few rows ($rows)"; fail=1; }
[ "$nodes" -ge 1 ] && echo "  ok    the plan uses a Gpu node" || { echo "  FAIL  no Gpu node in the plan"; fail=1; }
# an aggregate argument that is parallel unsafe (the grouping rel is not parallel safe): no GpuPreAgg may run under a Gather (A18 review)
$PSQL -d t -c "create function unsafe_f(numeric) returns numeric as 'select \$1' language sql immutable parallel unsafe" >/dev/null
UQ="select o_custkey % 3 as g, sum(unsafe_f(o_totalprice)) from orders group by 1 order by 1"
uc=$($PSQL -d t -c "$CPU $UQ" 2>&1 | md5 -q)
ug=$($PSQL -d t -c "$GPU $UQ" 2>&1 | md5 -q)
under=$($PSQL -d t -c "$GPU explain $UQ" | grep -A1 "Gather" | grep -c "Custom Scan (GpuPreAgg")
[ "$uc" = "$ug" ] && echo "  ok    parallel-unsafe aggregate argument: same rows" || { echo "  FAIL  parallel-unsafe aggregate argument: rows differ"; fail=1; }
[ "$under" = 0 ] && echo "  ok    no GpuPreAgg directly under a Gather for it" || { echo "  FAIL  GpuPreAgg under a Gather with a parallel-unsafe aggregate argument"; fail=1; }
[ "$fail" = 0 ] && echo "PARALLEL NESTED DEV: ALL PASSED" || echo "PARALLEL NESTED DEV: FAILED"
exit $fail
