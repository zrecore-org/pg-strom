#!/bin/sh
# Lifecycle test of the GPU service (milestone A1): launcher, control segment, handshake,
# restart after a crash, clean shutdown. Uses a throwaway cluster under $ROOT.
#
#   GPUSERV_BIN=/path/to/pgstrom-gpuserv PGBIN=/path/to/postgres/bin ./service_test.sh
#
# The module must already be installed in that PostgreSQL (make -C src install).
PGBIN=${PGBIN:-/opt/homebrew/opt/postgresql@17/bin}
GPUSERV_BIN=${GPUSERV_BIN:?set GPUSERV_BIN to the pgstrom-gpuserv binary}
ALLOW_CPU=${ALLOW_CPU:-off}
ROOT=${ROOT:-/tmp/pgstrom_svc_test}
PORT=${PORT:-54332}
fail=0

q() { "$PGBIN/psql" -h "$ROOT" -p "$PORT" -U postgres -d postgres -X -At "$@"; }
check() {   # check <description> <condition-exit-status>
    if [ "$2" -eq 0 ]; then echo "  ok    $1"; else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
wait_for() {   # wait_for <seconds> <command...>: poll until the command succeeds
    n=$1; shift
    while [ "$n" -gt 0 ]; do "$@" >/dev/null 2>&1 && return 0; sleep 1; n=$((n-1)); done
    return 1
}
cleanup() {
    "$PGBIN/pg_ctl" -D "$ROOT/data" -m immediate stop >/dev/null 2>&1
    rm -rf "$ROOT"
}
trap cleanup EXIT

rm -rf "$ROOT"; mkdir -p "$ROOT"
"$PGBIN/initdb" -D "$ROOT/data" -U postgres --no-sync >/dev/null || exit 1
cat >> "$ROOT/data/postgresql.conf" <<CONF
listen_addresses = ''
unix_socket_directories = '$ROOT'
port = $PORT
shared_buffers = 128MB
shared_preload_libraries = 'pg_strom'
pg_strom.gpuserv_path = '$GPUSERV_BIN'
pg_strom.gpu_allow_cpu_device = $ALLOW_CPU
max_worker_processes = 16
fsync = off
CONF
"$PGBIN/pg_ctl" -D "$ROOT/data" -l "$ROOT/server.log" -w start >/dev/null || { cat "$ROOT/server.log"; exit 1; }

echo "startup"
grep -q "Vulkan device 0" "$ROOT/server.log"; check "the postmaster logged the probed device" $?
wait_for 20 grep -q "VM self-test" "$ROOT/server.log"
check "the service opened the device and ran the VM self-test" $?
grep -E "pgstrom-gpuserv\[[0-9]+\]: (opened|VM self-test|ready)" "$ROOT/server.log" | sed 's/^[^p]*//; s/^/        /' | head -3
wait_for 20 sh -c "'$PGBIN/psql' -h '$ROOT' -p $PORT -U postgres -d postgres -X -At -c 'CREATE EXTENSION IF NOT EXISTS pg_strom' "
check "CREATE EXTENSION pg_strom" $?
wait_for 20 sh -c "[ \"\$('$PGBIN/psql' -h '$ROOT' -p $PORT -U postgres -d postgres -X -At -c 'SELECT state FROM pgstrom.gpu_service_state()')\" = ready ]"
check "the service reports state 'ready'" $?

echo "SQL"
ndev=$(q -c "SELECT count(*) FROM pgstrom.gpu_devcaps()")
[ "${ndev:-0}" -ge 1 ]; check "gpu_devcaps() lists the probed device(s) ($ndev)" $?
sel=$(q -c "SELECT count(*) FROM pgstrom.gpu_devcaps() WHERE selected AND tier <> 'none'")
[ "$sel" = 1 ]; check "exactly one usable device is selected" $?
q -c "SELECT device_name, tier, device_type, caps FROM pgstrom.gpu_devcaps() WHERE selected" | sed 's/^/        /'
pong=$(q -c "SELECT pgstrom.gpu_service_ping()")
case "$pong" in ok\ pid=*) r=0;; *) r=1;; esac; check "gpu_service_ping(): $pong" $r

echo "crash and restart"
pid1=$(q -c "SELECT pid FROM pgstrom.gpu_service_state()")
gen1=$(q -c "SELECT generation FROM pgstrom.gpu_service_state()")
kill -9 "$pid1"
wait_for 20 sh -c "[ \"\$('$PGBIN/psql' -h '$ROOT' -p $PORT -U postgres -d postgres -X -At -c 'SELECT state FROM pgstrom.gpu_service_state()')\" = ready ] && [ \"\$('$PGBIN/psql' -h '$ROOT' -p $PORT -U postgres -d postgres -X -At -c 'SELECT generation FROM pgstrom.gpu_service_state()')\" -gt $gen1 ]"
check "the service came back after kill -9 (no PostgreSQL restart)" $?
pid2=$(q -c "SELECT pid FROM pgstrom.gpu_service_state()")
[ "$pid2" != "$pid1" ] && [ -n "$pid2" ]; check "a new service process ($pid1 -> $pid2)" $?
rc=$(q -c "SELECT restart_count FROM pgstrom.gpu_service_state()")
[ "$rc" -ge 1 ]; check "restart_count recorded ($rc)" $?
q -c "SELECT pgstrom.gpu_service_ping()" >/dev/null 2>&1; check "gpu_service_ping() works again" $?
q -c "SELECT 1" >/dev/null 2>&1; check "ordinary queries kept working throughout" $?

echo "shutdown"
pmpid=$(head -1 "$ROOT/data/postmaster.pid")
svc=$(q -c "SELECT pid FROM pgstrom.gpu_service_state()")
"$PGBIN/pg_ctl" -D "$ROOT/data" -m fast -w stop >/dev/null 2>&1; check "PostgreSQL stops cleanly" $?
sleep 1
kill -0 "$svc" 2>/dev/null; [ $? -ne 0 ]; check "the service process is gone ($svc)" $?
ls /tmp/.pg_strom.*.sock "$ROOT"/.pg_strom.*.sock 2>/dev/null | grep -q . ; [ $? -ne 0 ]; check "the service socket was removed" $?

echo
if [ "$fail" -eq 0 ]; then echo "SERVICE TEST: ALL PASSED"; else echo "SERVICE TEST: $fail FAILED"; echo "--- server log:"; tail -30 "$ROOT/server.log"; fi
exit $fail
