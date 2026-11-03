# A6: kernel specialization (specialize.c, spec_mgr.c). Sourced by scan_test.sh after its other sections: it needs
# their helpers (run, runagg, runjoin, runjagg, q, check, sql, stop, $PSQL, $PGBIN, $GPU, $CPU, $ROOT, $PORT, $fail)
# and tables (s1 s2 s3, sd sd2 sd3, nm1, ty1, a1).
#
# The rest of scan_test.sh already runs with PGSTROM_GPUSERV_SPECIALIZE=check (every dispatch runs the VM kernel and
# the specialized one and compares their outputs); this section restarts the service in other configurations:
#   force      the specialized kernel alone (the first task waits for its compilation), against the CPU
#   cold/warm  an empty cache directory, then the same programs after a restart (SPIR-V from the disk cache)
#   no compiler, a cache directory that is not writable, the mode "off"
#   two sessions that need the same new kernel at once, other programs on the same connection
echo "kernel specialization (A6)"

spec_restart() { # restart the service with the environment the caller exported; the log starts empty
	stop
	: > "$ROOT/log"
	$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "  FAIL  the cluster does not start"; fail=$((fail+1)); return 1; }
	n=0; while [ $n -lt 40 ]; do
		[ "$($PSQL -d postgres -c 'select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
		n=$((n+1)); sleep 1
	done
	[ $n -lt 40 ] || { echo "  FAIL  the service does not become ready"; fail=$((fail+1)); return 1; }
}
spec_log_count() { # extended regex -> number of matching lines in the service log
	c=$(grep -cE "$1" "$ROOT/log" 2>/dev/null); echo "${c:-0}"
}
spec_wait_idle() { sleep 1; }

# the shapes: scan qualifiers and computed projections, aggregates, GROUP BY (with overflow), joins, join aggregates,
# numeric, text and date/time columns; every kernel kind (scan, agg, group, join, join agg, join group) is in it
spec_shapes() {
	t=$1
	ANYPLAN=0
	JOINPAT="Custom Scan (GpuJoin)"
	JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_seqscan=off; set enable_mergejoin=off; set enable_nestloop=off;"
	run $t "scan: trivial qual, mixed projection" "select a, f, c from %T where i > 0"
	run $t "scan: int arithmetic, NULLs" "select a + b, c from %T where b is not null and a * 2 < b"
	run $t "scan: float4 compare and arithmetic, NaN" "select i, d * 2::float4 + 1::float4 from %T where d < 50::float4 or d > 100.5::float4"
	run $t "scan: division, modulo, casts" "select b, c / 3, c % 7, a::int4 + b from %T where c > 1000"
	run $t "scan: AND / OR short circuits with NULLs" "select i from %T where (b > 0 and a < 0) or (c < 0 and b is null) or e"
	run $t "scan: text IS NULL, varlena pass-through" "select f from %T where f is not null and b > 49000"
	run $t "scan: 24 computed columns (32-register program)" "select a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11,a+12, b-1,b-2,b-3,b-4,b-5,b-6,b-7,b-8,b-9,b-10,b-11,b-12 from %T where a > 9500"
	runagg $t "agg: count, sum, min, max, avg with NULLs" "select count(*), count(b), sum(b), min(b), max(b), avg(b) from %T"
	runagg $t "agg: quals, expressions, FILTER" "select count(*) filter (where b > 0), sum(a + b), max(c * 2) from %T where a > 0 and b is not null"
	runagg $t "agg: sum beyond 64 bits, empty input" "select sum(c + 9000000000000000000), count(*) from %T where a > -5000 union all select sum(c), count(*) from %T where a > 100000"
	runagg $t "group: one int key" "select a % 10 as k, count(*), sum(b), min(c), max(b), avg(a) from %T group by 1 order by 1"
	runagg $t "group: two keys, NULLs, FILTER" "select a % 4 as k1, b % 3 as k2, count(*) filter (where e), sum(c), max(a) from %T where a > -9000 group by 1, 2 order by 1, 2"
	runagg $t "group: more groups than a workgroup's table" "select i % 1500 as k, count(*), sum(a), min(b), max(c) from %T group by 1 order by 1" "$BIGGROUPS"
	runjoin $t "join: both sides, computed projection" "select t.i, t.a + d.grp as x, d.pay * 2 as y, t.b - d.k as z from %T t join %D d on t.b = d.k"
	runjoin $t "join: LEFT, ON qual on the inner side" "select t.i, d.grp from %T t left join %D d on t.b = d.k and d.grp < 3"
	runjoin $t "join: text of both sides, extra join qual" "select t.i, d.txt, t.f from %T t join %D d on t.b = d.k and t.a > d.grp where t.i < 50000"
}
spec_join_shapes() {
	t=$1
	JOINSET="set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1; set enable_mergejoin=off; set enable_nestloop=off;"
	JAGGPAT='GPU Join Quals \[1\]'
	runjagg $t "join agg: both sides, NULLs, duplicate keys" "select count(*), sum(t.a), max(d.grp), count(d.pay), sum(d.pay), min(d.pay), avg(t.b) from %T t join %D d on t.b = d.k"
	runjagg $t "join agg: FILTER, expressions mixing sides" "select count(*) filter (where d.grp < 8), sum(t.a + d.grp), max(t.b * 2 - d.k) from %T t join %D d on t.b = d.k where t.a > 0"
	runjagg $t "join group: key from the inner side" "select d.grp, count(*), sum(t.a), max(t.c) from %T t join %D d on t.b = d.k group by 1 order by 1"
	runjagg $t "join group: two keys, sum beyond 64 bits" "select d.grp, t.a % 3 as k2, sum(t.c + 9000000000000000000), count(d.pay) from %T t join %D d on t.b = d.k group by 1, 2 order by 1, 2"
	runjagg $t "join group: more groups than a table (overflow list)" "select t.b as k, count(*), sum(d.grp) from %T t join %D d on t.b = d.k group by 1 order by 1" "$BIGGROUPS"
	JAGGPAT='Left Outer Join Quals \[1\]'
	runjagg $t "join agg: LEFT JOIN, null-extended group" "select d.grp, count(*), count(d.grp), sum(t.a) from %T t left join %D d on t.b = d.k + 20000 group by 1 order by 1"
	JAGGPAT='Join Quals \[2\]'
	runjagg $t "join agg: two inner relations" "select count(*), sum(t.a + d.grp + e.g2), max(e.p2), min(d.pay) from %T t join %D d on t.b = d.k join sd2 e on t.a % 400 = e.k2"
}
spec_have() { [ -n "$(sql "select to_regclass('$1')")" ]; }
spec_misc_shapes() {
	spec_have nm1 || return 0
	run nm1 "numeric: compare and arithmetic" "select i, n1 * n2 from nm1 where n1 > 3 and n1 < 20.5000 and n2 <> 1.5"
	runagg nm1 "numeric: aggregate with a numeric qual" "select count(*), sum(k), min(k8), max(i) from nm1 where n1 > 100 and n2 < 0"
	run ty1 "text: LIKE, equality, length" "select i, f from ty1 where f like 'ab%' or f = 'abc' or length(f) > 100"
	run ty1 "date and timestamp compare, date arithmetic" "select i, dt2 + 10, ts from ty1 where dt > '2000-01-01' and dt2 + 10 > '2005-01-01'"
	runagg ty1 "temporal aggregates with a text qual" "select count(*), sum(i), min(ts), max(dt2) from ty1 where f like 'ab%' and f <> 'ab'"
	runagg ty1 "GROUP BY date" "select dt3, count(*), sum(i), max(dt2) from ty1 group by 1 order by 1"
	spec_have a1 && runagg_any a1 "Arrow table: aggregate (stays on the VM)" "select count(*), sum(a), max(c), min(b) from a1 where a > 0"
}

SPEC_SAVE_MODE=$PGSTROM_GPUSERV_SPECIALIZE
SPEC_SAVE_CACHE=$PGSTROM_GPUSERV_CACHE_DIR
SPEC_SAVE_SLANGC=$PGSTROM_GPUSERV_SLANGC

# ---- 1. cold cache, the specialized kernels alone (force), against the CPU
echo "  -- force: specialized kernels only, empty cache"
export PGSTROM_GPUSERV_SPECIALIZE=force
export PGSTROM_GPUSERV_CACHE_DIR="$ROOT/speccache_force"
rm -rf "$PGSTROM_GPUSERV_CACHE_DIR"
spec_restart
for t in s1 s2; do spec_shapes $t; done
spec_join_shapes s1
spec_misc_shapes
n=$(spec_log_count 'specialize: .* kernel [0-9a-f]+ in use')
[ "$n" -ge 10 ]; check "force: $n specialized kernels were used" $?
for k in "scan" "agg" "group" "join" "join agg" "join group"; do
	[ "$(spec_log_count "specialize: $k kernel [0-9a-f]+ in use")" -ge 1 ]; check "force: a '$k' kernel was used" $?
done
[ "$(spec_log_count 'specialize: .* not used')" -eq 0 ]; check "force: no kernel failed to compile or load" $?
ncomp=$(spec_log_count 'ready: slangc')
[ "$ncomp" -ge 6 ]; check "force: $ncomp kernels compiled by slangc" $?
ls "$PGSTROM_GPUSERV_CACHE_DIR"/*.spv >/dev/null 2>&1; check "force: SPIR-V files are in the cache directory" $?
[ "$(stat -c %a "$PGSTROM_GPUSERV_CACHE_DIR" 2>/dev/null || stat -f %Lp "$PGSTROM_GPUSERV_CACHE_DIR")" = 700 ]; check "force: the cache directory is private (0700)" $?

# ---- 2. warm cache after a service restart: no compilation, the same results
echo "  -- warm cache after a restart"
spec_restart
for t in s1; do spec_shapes $t; done
spec_join_shapes s1
nw=$(spec_log_count 'ready: disk cache')
[ "$nw" -ge 6 ]; check "warm: $nw kernels came from the disk cache" $?
[ "$(spec_log_count 'ready: slangc')" -eq 0 ]; check "warm: nothing was compiled again" $?

# ---- 3. check mode (the default of this script) on a program the cache has not seen: both kernels, compared
echo "  -- check: VM and specialized kernel compared on every dispatch"
export PGSTROM_GPUSERV_SPECIALIZE=check
export PGSTROM_GPUSERV_CACHE_DIR="$ROOT/speccache_check"
rm -rf "$PGSTROM_GPUSERV_CACHE_DIR"
spec_restart
spec_shapes s3
spec_join_shapes s2
spec_misc_shapes
nc=$(spec_log_count 'specialize: check ok')
[ "$nc" -ge 20 ]; check "check: $nc dispatches compared equal" $?
[ "$(spec_log_count 'CHECK FAILED')" -eq 0 ]; check "check: no difference between the VM and the specialized kernels" $?

# ---- 4. two sessions that need the same new kernel at once: one compilation
echo "  -- two sessions, one new kernel"
export PGSTROM_GPUSERV_SPECIALIZE=force
export PGSTROM_GPUSERV_CACHE_DIR="$ROOT/speccache_two"
rm -rf "$PGSTROM_GPUSERV_CACHE_DIR"
spec_restart
before=$(spec_log_count 'agg kernel [0-9a-f]+ queued')
Q2="select count(*), sum(a + 31337), max(c) from s1 where b > -31337"
( $PSQL -d scantest -c "$GPU $Q2" > "$ROOT/two1.out" 2>&1 ) &
( $PSQL -d scantest -c "$GPU $Q2" > "$ROOT/two2.out" 2>&1 ) &
wait
cmp -s "$ROOT/two1.out" "$ROOT/two2.out" && [ -s "$ROOT/two1.out" ] && ! grep -q ERROR "$ROOT/two1.out"; check "two sessions: both got a result, and the same one" $?
c=$($PSQL -d scantest -c "$CPU $Q2" 2>&1 | tail -1)
[ "$(tail -1 "$ROOT/two1.out")" = "$c" ]; check "two sessions: the result is the CPU's" $?
after=$(spec_log_count 'agg kernel [0-9a-f]+ queued')
[ $((after - before)) -eq 1 ]; check "two sessions: the kernel was queued once ($((after - before)))" $?
# different programs one after the other on one connection
QS1="select count(*), sum(a + 1) from s1 where b > 10"; QS2="select count(*), sum(a + 2), max(b) from s1 where b > 20"
out=$($PSQL -d scantest -c "$GPU" -c "$QS1" -c "$QS2" -c "$QS1" 2>&1 | tr '\n' ' ')
c=$($PSQL -d scantest -c "$CPU" -c "$QS1" -c "$QS2" -c "$QS1" 2>&1 | tr '\n' ' ')
[ "$out" = "$c" ]; check "one connection, three programs (the first twice): results equal the CPU's" $?

# ---- 5. the compiler is missing, nothing cached: the VM runs everything
echo "  -- no compiler"
export PGSTROM_GPUSERV_SPECIALIZE=force
export PGSTROM_GPUSERV_SLANGC=/nonexistent/slangc
export PGSTROM_GPUSERV_CACHE_DIR="$ROOT/speccache_nocc"
rm -rf "$PGSTROM_GPUSERV_CACHE_DIR"
spec_restart
spec_shapes s1
[ "$(spec_log_count 'specialize: .* in use')" -eq 0 ]; check "no compiler: no specialized kernel was used" $?
[ "$(spec_log_count 'slangc not found|stays on the VM|not used')" -ge 1 ]; check "no compiler: the service said so in its log" $?
unset PGSTROM_GPUSERV_SLANGC
[ -n "$SPEC_SAVE_SLANGC" ] && export PGSTROM_GPUSERV_SLANGC=$SPEC_SAVE_SLANGC

# ---- 6. a cache directory that is not writable: compiled each time, in a scratch directory
echo "  -- cache directory not writable"
export PGSTROM_GPUSERV_CACHE_DIR="$ROOT/speccache_ro"
rm -rf "$PGSTROM_GPUSERV_CACHE_DIR"; mkdir -p "$PGSTROM_GPUSERV_CACHE_DIR"; chmod 500 "$PGSTROM_GPUSERV_CACHE_DIR"
spec_restart
spec_shapes s1
if [ "$(id -u)" != 0 ]; then
	[ "$(spec_log_count 'is not usable')" -ge 1 ]; check "read-only cache: the service reported the directory" $?
	[ "$(spec_log_count 'specialize: .* in use')" -ge 3 ]; check "read-only cache: kernels were compiled and used all the same" $?
	ls "$PGSTROM_GPUSERV_CACHE_DIR"/*.spv >/dev/null 2>&1; [ $? -ne 0 ]; check "read-only cache: nothing was written there" $?
fi
chmod 700 "$PGSTROM_GPUSERV_CACHE_DIR"

# ---- 7. off: the VM only
echo "  -- off"
export PGSTROM_GPUSERV_SPECIALIZE=off
spec_restart
spec_shapes s1
[ "$(spec_log_count 'specialize:')" -eq 0 ]; check "off: the specializer did nothing" $?

# ---- back to the configuration of the rest of the script
if [ -n "$SPEC_SAVE_MODE" ]; then export PGSTROM_GPUSERV_SPECIALIZE=$SPEC_SAVE_MODE; else unset PGSTROM_GPUSERV_SPECIALIZE; fi
if [ -n "$SPEC_SAVE_CACHE" ]; then export PGSTROM_GPUSERV_CACHE_DIR=$SPEC_SAVE_CACHE; else unset PGSTROM_GPUSERV_CACHE_DIR; fi
spec_restart
