#!/bin/sh
# First-query latency per kernel tier (src/gpuserv/scan_caps.h: full / lite / nogeom): a fresh service answers a first
# query of each kind with the Vulkan pipeline cache off, so what is measured is the driver's pipeline creation of the
# interpreter kernel the query needs (Metal's own cache is whatever it holds: run twice, or after a change of kernels_vm/,
# to see the cold compile). The same queries run once with the tiers (the default) and once with PGSTROM_GPUSERV_TIERS=full
# (every program on the full kernels, what the service did before the middle tier). The service logs each pipeline it
# creates (PGSTROM_GPUSERV_PIPELINE_LOG=1: SPIR-V size and creation time), which shows the tier a query really used.
#
#   sh test/gpuserv/locked_test.sh "$PWD" tier_latency.sh
#
# Specialization is off, so the interpreter's pipeline is the only compile in the way. No pass/fail: it prints a table.
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
PKGLIB=$($PGBIN/pg_config --pkglibdir)
GPUSERV_BIN=${GPUSERV_BIN:-$PKGLIB/pgstrom-gpuserv}
ROOT=${ROOT:-/tmp/pgstrom_tier_latency}
PORT=${PORT:-54417}
NROWS=${NROWS:-300000}
PSQL="$PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
D=${0%/*}

stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = ${ALLOW_CPU:-off}
max_worker_processes = 24
CONF
sh "$D/pgconf_test.sh" >> "$ROOT/data/postgresql.conf"

startpg() { # <extra env words> : start the cluster with a fresh service
	env PGSTROM_GPUSERV_SPECIALIZE=off PGSTROM_GPUSERV_PIPELINE_CACHE=off PGSTROM_GPUSERV_PIPELINE_LOG=1 \
		PGSTROM_GPUSERV_CACHE_DIR=$ROOT/cache $1 \
		$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
	n=0
	while [ $n -lt 60 ]; do
		[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
		n=$((n+1)); sleep 1
	done
}

startpg ""
$PSQL -d postgres -c "create database tl" >/dev/null
$PSQL -d tl -c "create extension pg_strom" >/dev/null
$PSQL -d tl -c "create table t as select i as k, (i % 20000)::int4 as a, (i * 0.37)::float8 as x, ((i % 100000) / 100.0)::numeric(12,2) as n from generate_series(1, $NROWS) i" >/dev/null
$PSQL -d tl -c "vacuum analyze t" >/dev/null
$PSQL -d tl -c "create table u as select i as k, (i % 1000)::float8 as y from generate_series(1, 100000) i" >/dev/null
$PSQL -d tl -c "vacuum analyze u" >/dev/null
stop

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
# label|query|plan pattern. The float8 join aggregate has a join qual that mixes both sides, which is part of the join program
# (a qual on one side only is evaluated by the scan kernel of that side): that program uses the float8 group, so it needs
# the nogeom join kernel. The other join aggregate runs its scans and its join on the lite kernels.
QUERIES='int agg (lite)|select sum(a), count(*) from t where a > 10|GpuPreAgg
float8 agg (nogeom)|select sum(x * 2.0 + 1.0) from t where x * 1.5 > 100|GpuPreAgg
numeric agg (nogeom)|select sum(n * 2), count(*) from t where n > 5.5|GpuPreAgg
int join agg (lite)|select count(*), sum(t.a) from t join u on t.k = u.k where t.a > 10|GpuPreAgg
float8 join agg (nogeom)|select count(*), sum(t.a) from t join u on t.k = u.k and t.x + u.y > 100|GpuPreAgg'

# Every query gets a fresh service (so the first query of each kind pays for its own interpreter pipeline) and the line
# names the pipelines that service created by SPIR-V size, which shows the tier the query really used.
run_mode() { # <label> <env words>
	echo "== $1"
	echo "$QUERIES" | while IFS='|' read -r label qry pat; do
		startpg "$2"
		plan=$($PSQL -d tl -c "$GPU $JOINSET explain $qry" | grep -c "$pat")
		first=$($PSQL -d tl -c "\\timing on" -c "$GPU $JOINSET $qry" 2>&1 | grep '^Time:' | tail -1)
		second=$($PSQL -d tl -c "\\timing on" -c "$GPU $JOINSET $qry" 2>&1 | grep '^Time:' | tail -1)
		pipes=$(grep "pipeline .* created" "$ROOT/log" | sed 's/.*gpuserv: pipeline //' | awk '$1 > 100000 {printf "%s%s B in %s s", sep, $1, $6; sep = "; "}')
		printf '  %-26s plan %-10s first %s, second %s; pipelines: %s\n' "$label" "$([ "$plan" -ge 1 ] && echo "$pat" || echo "NOT $pat")" "${first#Time: }" "${second#Time: }" "$pipes"
		stop
		: > "$ROOT/log"
	done
}

run_mode "tiers (default)" ""
run_mode "all programs on the full kernels (PGSTROM_GPUSERV_TIERS=full)" "PGSTROM_GPUSERV_TIERS=full"
[ -z "$KEEP" ] && rm -rf "$ROOT"
