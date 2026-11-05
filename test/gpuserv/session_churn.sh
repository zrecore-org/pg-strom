#!/bin/sh
# Opens and closes many GPU sessions (one backend each) and ends them in different ways; none may hang or fail. Regression test
# for the backend that hung in close(2) on the service socket (xpuClientCloseSession closed the descriptor while the session worker
# thread was polling it; macOS blocks there). Run it through locked_test.sh:
#
#   sh test/gpuserv/locked_test.sh "$PWD" session_churn.sh        # CHURN=<n> rounds (default 150), CHURN_PAR=<n> clients (default 3)
#
# Every psql call has a limit (QUERY_TIMEOUT, default 60 s here): a hung close shows up as "query timeout".
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pg_session_churn}
PORT=${PORT:-54444}
export QUERY_TIMEOUT=${QUERY_TIMEOUT:-60}
PSQL="${0%/*}/qtimeout.sh $ROOT/.query_timeout $PGBIN/psql -h $ROOT -p $PORT -U postgres -X -At -q"
CHURN=${CHURN:-150}
PAR=${CHURN_PAR:-3}
fail=0

check() { # name rc
	if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
stop() { $PGBIN/pg_ctl -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; }
stop
rm -rf "$ROOT"; mkdir -p "$ROOT"
$PGBIN/initdb -D "$ROOT/data" -U postgres >/dev/null 2>&1 || { echo "initdb failed"; exit 1; }
cat >> "$ROOT/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$ROOT'
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
max_worker_processes = 24
CONF
sh "${0%/*}/pgconf_test.sh" >> "$ROOT/data/postgresql.conf"
trap stop EXIT INT TERM
export PGSTROM_GPUSERV_CACHE_DIR=$ROOT/speccache
$PGBIN/pg_ctl -D "$ROOT/data" -l "$ROOT/log" -w start >/dev/null 2>&1 || { echo "start failed"; exit 1; }
n=0; while [ $n -lt 40 ]; do
	[ "$($PSQL -d postgres -c 'create extension if not exists pg_strom; select state from pgstrom.gpu_service_state()' 2>/dev/null | tail -1)" = ready ] && break
	n=$((n+1)); sleep 1
done
check "service is ready" "$([ $n -lt 40 ]; echo $?)"
$PSQL -d postgres -c "create database churn" >/dev/null
$PSQL -d churn -c "create extension pg_strom; create table t as select i, (i % 1000)::int4 as a, (i * 7 % 10007)::int4 as b from generate_series(1, 60000) i; analyze t" >/dev/null

GPU="set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.enabled=on;"
want=$($PSQL -d churn -c "set pg_strom.enabled=off; select count(*) from t where a > 500" | tail -1)
plan=$($PSQL -d churn -c "$GPU explain select count(*) from t where a > 500" | grep -c "Custom Scan (Gpu")
check "the churn query runs on the GPU (plan)" "$([ "$plan" -ge 1 ]; echo $?)"

# one client: queries that end normally, an UPDATE over a GPU scan, a rolled back UPDATE, a query cut off by LIMIT, an error
client() {
	k=0; bad=0
	while [ $k -lt $CHURN ]; do
		r=$($PSQL -d churn -c "$GPU select count(*) from t where a > 500" 2>&1 | tail -1)
		[ "$r" = "$want" ] || { bad=$((bad+1)); echo "client $1 round $k: count gave '$r', wanted '$want'" >> "$ROOT/bad.log"; }
		# concurrent bulk updates of overlapping rows may deadlock (a GpuScan returns tuples in another order than a sequential
		# scan, like a parallel scan); that is PostgreSQL's answer and fine. The concurrent updates exercise EvalPlanQual,
		# which is what used to crash the backend.
		$PSQL -d churn -c "$GPU update t set b = b where a % 97 = 0" > "$ROOT/out.$1" 2>&1 || { grep -q "deadlock detected" "$ROOT/out.$1" || { bad=$((bad+1)); echo "client $1 round $k: update: $(grep -m1 -E 'ERROR|psql:' "$ROOT/out.$1")" >> "$ROOT/bad.log"; }; }
		$PSQL -d churn -c "$GPU begin; update t set b = b + 1 where a < 10; rollback" > "$ROOT/out.$1" 2>&1 || { grep -q "deadlock detected" "$ROOT/out.$1" || { bad=$((bad+1)); echo "client $1 round $k: rollback: $(grep -m1 -E 'ERROR|psql:' "$ROOT/out.$1")" >> "$ROOT/bad.log"; }; }
		$PSQL -d churn -c "$GPU select i from t where a > 100 limit 3" > "$ROOT/out.$1" 2>&1 || { bad=$((bad+1)); echo "client $1 round $k: limit: $(tail -1 "$ROOT/out.$1")" >> "$ROOT/bad.log"; }
		$PSQL -d churn -c "$GPU select i / (a - a) from t where a > 500" > /dev/null 2>&1		# division by zero: an error
		k=$((k+1))
	done
	echo $bad > "$ROOT/bad.$1"
}
i=1
while [ $i -le $PAR ]; do client $i & i=$((i+1)); done
wait
total=0; i=1
while [ $i -le $PAR ]; do total=$((total + $(cat "$ROOT/bad.$i"))); i=$((i+1)); done
check "$PAR clients x $CHURN rounds of 5 sessions: none failed or timed out" "$([ "$total" -eq 0 ]; echo $?)"
[ -s "$ROOT/bad.log" ] && head -5 "$ROOT/bad.log"
check "no query timed out" "$([ ! -e "$ROOT/.query_timeout" ]; echo $?)"
r=$($PSQL -d churn -c "select pg_stat_get_db_numbackends(oid) from pg_database where datname = 'churn'" | tail -1)
check "no backend is left behind ($r connected)" "$([ "$r" -le 1 ]; echo $?)"
n=$($PSQL -d postgres -c "select restart_count from pgstrom.gpu_service_state()" 2>/dev/null | tail -1)
check "the service was not restarted (restart_count ${n:-?})" "$([ "${n:-1}" = 0 ]; echo $?)"
echo
if [ $fail -eq 0 ]; then echo "SESSION CHURN: ALL PASSED"; else echo "SESSION CHURN: $fail FAILED"; exit 1; fi
