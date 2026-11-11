# First-use kernel compile tests (A9 part 3), sourced by firstuse_dev.sh after the helpers (check, sql, PSQL, fail, ROOT,
# PGBIN, PORT, GPUSERV_BIN, stop) and the postgresql.conf of a cluster that has not been started yet.
#
# What is tested (see src/gpuserv/kprewarm.c and the plan's "A9 part 3"):
#  1. an empty pipeline cache and no prewarm: a query whose kernel is not compiled returns the CPU's rows at once (the
#     service refuses the task, the backend continues on the CPU, A14) while the kernel compiles in the background; the
#     next run of it is served by the GPU
#  2. a wait budget: a task waits for the compile, a node that cannot continue on the CPU and "-1" wait for it however long
#  3. the persisted state: a restart finds the compiled kernels ready, a damaged pipeline cache file is rebuilt
#  4. the background prewarm of the "lite" set (progress through pgstrom.gpu_prewarm_state() / gpu_kernel_state(), a
#     query during it, a restart that resumes it)
#  5. the install-time `pgstrom-gpuserv --prewarm-all` and the first query after it
echo "First-use kernel compile (A9 part 3)"
now_ms() { perl -MTime::HiRes=time -e 'printf "%d\n", time() * 1000'; }
fnow() { date +%s; }

start_cluster() { # <extra conf lines...>
	for l in "$@"; do echo "$l" >> "$ROOT/data/postgresql.conf"; done
	$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; tail -20 "$ROOT/log"; exit 1; }
	n=0; while [ $n -lt 60 ]; do
		[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
		n=$((n+1)); sleep 1
	done
	[ $n -lt 60 ]
}
# the state functions
pw() { $PSQL -d postgres -c "select $1 from pgstrom.gpu_prewarm_state()"; }
ks() { $PSQL -d postgres -c "select $1 from pgstrom.gpu_kernel_state() where $2"; }
# wait until the SQL condition (against gpu_prewarm_state()) is true, at most $2 seconds
waitfor() {
	t=0; while [ $t -lt "${2:-120}" ]; do
		[ "$($PSQL -d postgres -c "select ($1) from pgstrom.gpu_prewarm_state()" 2>/dev/null)" = t ] && return 0
		t=$((t+1)); sleep 1
	done
	return 1
}

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
CPU="set max_parallel_workers_per_gather=0; set pg_strom.enabled=off;"
q() { # settings, query -> md5 of the sorted result text|rows
	$PSQL -d scantest -c "$1 select md5(coalesce(string_agg(x::text, E'\n' order by x::text), '')), count(*) from ($2) x" 2>&1 | tail -1
}
fbcount() { grep -c "the CPU runs the rest of the node" "$ROOT/log"; }
refused() { pw refused_tasks; }
waited() { pw waited_tasks; }

start_cluster; check "the service is ready" $?
$PSQL -d postgres -c "create database scantest" >/dev/null
sql "create extension pg_strom" >/dev/null
sql "create table sd as select (i - 10001)::int4 as k, (i % 17)::int4 as grp,
	case when i % 9 = 0 then null else (i::int8 * 3 - 5) end as pay, repeat(chr(65 + i % 26), (i % 30)::int) as txt from generate_series(1, 20000) i"
sql "create table fb1 (id int4, a int2, b int4, c int8, f text, n numeric(12,2), k int4)"
sql "insert into fb1 select i, ((i % 20000) - 10000)::int2,
	case when i % 7 = 0 then null else ((i * 31) % 100000 - 50000)::int4 end,
	((i::int8 * 1000003) % 1000000007 - 500000000),
	case when i % 11 = 0 then null else repeat(chr(97 + i % 26), (i % 40)::int) end,
	((i % 1000) / 8.0)::numeric(12,2), ((i % 20000) - 10000)::int4 from generate_series(1, $FBN) i"
sql "vacuum analyze fb1"; sql "analyze sd"

SCANQ="select id, a, f, n from fb1 where b > 100 and a < 5000"
AGGQ="select count(*), sum(b), min(a), max(c), avg(b) from fb1 where a > -5000"
GRPQ="select a % 7 as k, count(*), sum(b), max(c) from fb1 group by 1"
JOINQ="select fb1.id, sd.pay, sd.txt from fb1 join sd on fb1.k = sd.k where fb1.b > 0"
SCANSET="$GPU set enable_seqscan=off;"
AGGSET="$GPU set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off;"
JOINSET="$GPU set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
plan_is() { # settings, query, pattern
	[ "$($PSQL -d scantest -c "$1 explain $2" 2>&1 | grep -c "$3")" -ge 1 ]
}
plan_is "$SCANSET" "$SCANQ" "Custom Scan (GpuScan)"; check "plan: GpuScan" $?
plan_is "$AGGSET" "$AGGQ" "Custom Scan (GpuPreAgg)"; check "plan: GpuPreAgg" $?
plan_is "$AGGSET" "$GRPQ" "Custom Scan (GpuPreAgg)"; check "plan: GpuPreAgg with GROUP BY" $?
plan_is "$JOINSET" "$JOINQ" "Custom Scan (GpuJoin)"; check "plan: GpuJoin" $?
CS=$(q "$CPU" "$SCANQ"); CA=$(q "$CPU" "$AGGQ"); CG=$(q "$CPU" "$GRPQ"); CJ=$(q "$CPU" "$JOINQ")

echo "-- the state functions on an empty cache, prewarm off"
[ "$(pw phase)" = off ]; check "gpu_prewarm_state(): phase 'off'" $?
[ "$(pw mode)" = off ]; check "gpu_prewarm_state(): mode off" $?
NK=$(pw kernels); [ "${NK:-0}" -ge 150 ]; check "gpu_prewarm_state(): the service knows its kernels ($NK)" $?
[ "$(ks 'count(*)' "state <> 'not compiled'")" = 0 ]; check "gpu_kernel_state(): nothing is compiled yet" $?
[ "$(ks 'count(*)' "kernel = 'gpuscan8_lite' and tier = 'lite' and not in_set")" = 1 ]; check "gpu_kernel_state(): a row per kernel with its tier" $?
[ "$(refused)" = 0 ]; check "no task refused so far" $?

echo "-- a cold kernel does not stall the query: the CPU answers while the compile goes on"
b=$(fbcount); t0=$(now_ms)
g=$(q "$SCANSET set pg_strom.gpu_kernel_wait=0;" "$SCANQ"); t1=$(now_ms); a=$(fbcount)
[ "$g" = "$CS" ]; check "cold GpuScan with wait=0 returns the CPU's rows (${g#*|} rows, $((t1 - t0)) ms)" $?
[ "$a" -gt "$b" ] && [ "$(refused)" -ge 1 ]; check "the node continued on the CPU (fallbacks $b -> $a, refused tasks $(refused))" $?
K1=$(grep "is not compiled yet" "$ROOT/log" | tail -1 | sed 's/.*kernel \([^ ]*\) is not compiled yet.*/\1/')
echo "        first query on a cold kernel ($K1): $((t1 - t0)) ms"
[ "$(pw demanded_kernels)" -ge 1 ]; check "the kernel was queued for the background compile ($K1)" $?
t0=$(now_ms)
t=0; while [ $t -lt 180 ]; do [ "$(ks 'state' "kernel = '$K1'")" = ready ] && break; t=$((t+1)); sleep 1; done
t1=$(now_ms)
[ "$(ks 'state' "kernel = '$K1'")" = ready ]; check "the kernel compiled in the background ($((t1 - t0)) ms later)" $?
[ "$(ks 'compile_seconds > 0' "kernel = '$K1'")" = t ]; check "its compile time is recorded" $?
r0=$(refused); b=$(fbcount); t0=$(now_ms)
g=$(q "$SCANSET set pg_strom.gpu_kernel_wait=0;" "$SCANQ"); t1=$(now_ms); a=$(fbcount)
[ "$g" = "$CS" ]; check "the same query after the compile returns the same rows ($((t1 - t0)) ms)" $?
[ "$a" = "$b" ] && [ "$(refused)" = "$r0" ]; check "and runs on the GPU (no fallback, no refusal)" $?

echo "-- a wait budget"
r0=$(refused); w0=$(waited); b=$(fbcount); t0=$(now_ms)
g=$(q "$AGGSET set pg_strom.gpu_kernel_wait=120000;" "$AGGQ"); t1=$(now_ms); a=$(fbcount)
[ "$g" = "$CA" ]; check "GpuPreAgg with a 120 s budget: the CPU's rows ($((t1 - t0)) ms)" $?
[ "$a" = "$b" ] && [ "$(refused)" = "$r0" ] && [ "$(waited)" -gt "$w0" ]; check "it waited for the compile and ran on the GPU (waited tasks $w0 -> $(waited))" $?
r0=$(refused); b=$(fbcount); t0=$(now_ms)
g=$(q "$AGGSET set pg_strom.gpu_kernel_wait=0; set pg_strom.gpu_error_fallback=off;" "$GRPQ"); t1=$(now_ms); a=$(fbcount)
[ "$g" = "$CG" ]; check "a node that may not continue on the CPU (gpu_error_fallback=off) waits: the CPU's rows ($((t1 - t0)) ms)" $?
[ "$a" = "$b" ] && [ "$(refused)" = "$r0" ]; check "and was not refused" $?
t0=$(now_ms)
g=$(q "$JOINSET set pg_strom.gpu_kernel_wait=-1;" "$JOINQ"); t1=$(now_ms)
[ "$g" = "$CJ" ]; check "gpu_kernel_wait=-1 (no gate): the join waits for its kernel in place ($((t1 - t0)) ms)" $?
[ "$(refused)" = "$r0" ]; check "and was not refused" $?
echo "        compile in place of a cold join kernel: $((t1 - t0)) ms"

echo "-- the persisted state"
NREADY=$(ks 'count(*)' "state = 'ready'")
stop
ls "$ROOT/pcache"/kernels-*.txt >/dev/null 2>&1; check "the readiness manifest is next to the pipeline cache file" $?
start_cluster; check "the service restarts" $?
[ "$(pw from_disk)" = "$NREADY" ] && [ "$NREADY" -ge 2 ]; check "after a restart the $NREADY compiled kernels are ready without a compile" $?
r0=$(refused); b=$(fbcount)
g=$(q "$SCANSET set pg_strom.gpu_kernel_wait=0;" "$SCANQ")
[ "$g" = "$CS" ] && [ "$(refused)" = 0 ] && [ "$(fbcount)" = "$b" ]; check "a query on a kernel compiled before the restart runs on the GPU at once" $?
stop
F=$(ls "$ROOT/pcache"/pipeline-*.bin | grep -v spec | head -1)
head -c 100 "$F" > "$F.t" && mv "$F.t" "$F"
start_cluster; check "the service starts with a damaged pipeline cache file" $?
[ "$(pw from_disk)" = 0 ] && [ "$(ks 'count(*)' "state = 'ready'")" = 0 ]; check "the manifest is not trusted without its pipeline cache: every kernel is compiled again" $?
b=$(fbcount)
g=$(q "$SCANSET set pg_strom.gpu_kernel_wait=0;" "$SCANQ")
[ "$g" = "$CS" ] && [ "$(fbcount)" -gt "$b" ]; check "and a query returns the CPU's rows while it does" $?
t=0; while [ $t -lt 180 ]; do [ "$(ks 'state' "kernel = '$K1'")" = ready ] && break; t=$((t+1)); sleep 1; done
[ "$(ks 'state' "kernel = '$K1'")" = ready ]; check "the pipeline cache file was rebuilt" $?
[ "$(wc -c < "$F" | tr -d ' ')" -gt 1000 ]; check "the file is a full cache again" $?

echo "-- the background prewarm (set lite)"
stop
start_cluster "pg_strom.gpu_prewarm = 'lite'"; check "the service starts with pg_strom.gpu_prewarm = lite" $?
[ "$(pw mode)" = lite ]; check "mode lite" $?
TGT=$(pw target); [ "${TGT:-0}" -ge 30 ]; check "the lite set has $TGT kernels (lite and nogeom tiers of the common ones)" $?
[ "$(ks 'count(*)' "in_set and tier in ('full', 'geometry')")" = 0 ]; check "the set holds no geometry and no full-tier kernel" $?
waitfor "ready >= 3 or phase = 'done'" 120; check "kernels become ready one after the other (ready $(pw ready) of $(pw target))" $?
echo "        $(pw "phase, ready, target, current_kernel")"
# a query during the prewarm: correct rows, not stalled
t0=$(now_ms); g=$(q "$SCANSET set pg_strom.gpu_kernel_wait=1000;" "$SCANQ"); t1=$(now_ms)
[ "$g" = "$CS" ]; check "a query during the prewarm returns the CPU's rows ($((t1 - t0)) ms)" $?
# restart in the middle: the next start resumes (the compiled ones come from disk)
R1=$(pw ready)
if [ "$(pw phase)" = running ]; then
	stop
	start_cluster; check "restart in the middle of the prewarm" $?
	[ "$(pw from_disk)" -ge "$R1" ]; check "the restart resumes: $(pw from_disk) kernels ready from disk (at least the $R1 of before)" $?
else
	echo "  note  the prewarm had already finished (only a cold compiler makes it long enough to interrupt)"
fi
t0=$(now_ms)
waitfor "phase = 'done'" 900; check "the lite set is compiled ($(( ($(now_ms) - t0) / 1000 )) s more)" $?
[ "$(pw "ready = target and failed = 0")" = t ]; check "every kernel of the set is ready, none failed ($(pw "ready, target, failed"))" $?
echo "        $(pw "phase, kernels, target, ready, failed, from_disk, elapsed_seconds")"
b=$(fbcount); r0=$(refused)
g1=$(q "$SCANSET set pg_strom.gpu_kernel_wait=0;" "$SCANQ"); g2=$(q "$AGGSET set pg_strom.gpu_kernel_wait=0;" "$AGGQ")
g3=$(q "$AGGSET set pg_strom.gpu_kernel_wait=0;" "$GRPQ"); g4=$(q "$JOINSET set pg_strom.gpu_kernel_wait=0;" "$JOINQ")
[ "$g1" = "$CS" ] && [ "$g2" = "$CA" ] && [ "$g3" = "$CG" ] && [ "$g4" = "$CJ" ]; check "after the prewarm scan, aggregation, GROUP BY and join return the CPU's rows" $?
[ "$(fbcount)" = "$b" ] && [ "$(refused)" = "$r0" ]; check "and all four ran on the GPU: no fallback, no refusal" $?

echo "-- the install-time prewarm (pgstrom-gpuserv --prewarm-all)"
stop
CLIDIR="$ROOT/pcache_cli"; mkdir -p "$CLIDIR"
L=$(PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=$CLIDIR "$GPUSERV_BIN" --prewarm-list 2>&1)
echo "$L" | head -1 | sed 's/^/        /'
echo "$L" | grep -q "to compile"; check "--prewarm-list shows the set and what is still to compile" $?
t0=$(now_ms)
PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=$CLIDIR "$GPUSERV_BIN" --prewarm-all > "$ROOT/prewarm_all.log" 2>&1; rc=$?
t1=$(now_ms)
tail -1 "$ROOT/prewarm_all.log" | sed 's/^/        /'
[ $rc -eq 0 ]; check "--prewarm-all compiles the set ($(( (t1 - t0) / 1000 )) s)" $?
ls "$CLIDIR"/kernels-*.txt >/dev/null 2>&1; check "and writes the manifest" $?
t0=$(now_ms)
PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=$CLIDIR "$GPUSERV_BIN" --prewarm-all > "$ROOT/prewarm_all2.log" 2>&1; rc=$?
t1=$(now_ms)
head -1 "$ROOT/prewarm_all2.log" | grep -q " 0 to compile"; check "a second run has nothing to compile ($((t1 - t0)) ms)" $?
# a first query on the cluster that uses it: no stall, no fallback
export PGSTROM_GPUSERV_PIPELINE_CACHE_DIR=$CLIDIR
start_cluster "pg_strom.gpu_prewarm = 'off'"; check "a cluster starts on the prewarmed cache" $?
b=$(fbcount); t0=$(now_ms)
g=$(q "$JOINSET set pg_strom.gpu_kernel_wait=0;" "$JOINQ"); t1=$(now_ms)
[ "$g" = "$CJ" ] && [ "$(fbcount)" = "$b" ] && [ "$(refused)" = 0 ]; check "the first query on the prewarmed install runs on the GPU at once ($((t1 - t0)) ms, nothing refused)" $?
echo "        first join query after --prewarm-all: $((t1 - t0)) ms"
