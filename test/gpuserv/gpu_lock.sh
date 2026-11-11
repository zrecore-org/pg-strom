#!/bin/sh
# Run a command while holding the machine's one GPU lock, in fair (first come, first served) order.
#
#   sh gpu_lock.sh <label> -- <command> [args...]       the command runs with the lock held; its exit status is returned
#   sh gpu_lock.sh <label> --status                      who holds the lock and who waits (no waiting)
#
# Several projects share this Mac's one GPU (PG-Strom's tests, the MongoDB GPU experiments in ../mongo, ...): two GPU
# programs at once distort timings, and on Apple Silicon macOS can kill a Metal command buffer ("Impacting Interactivity").
# Every GPU run of every project therefore goes through the same lock. It is the protocol of test/gpuserv/locked_test.sh
# (same directory, same files), so a locked_test.sh run and a gpu_lock.sh run exclude each other and queue in one line:
#
#   lock         /tmp/pgstrom_test.lock/{pid,lstart,tree,start,beat}   (PGSTROM_TEST_LOCK moves it; set it identically)
#   queue        /tmp/pgstrom_test.lock.queue/<10-digit ticket>/{pid,lstart,tree}
#
# Rules for everybody (also in doc/gpu-sharing.md):
#   - Hold the lock only while the GPU is in use. Build, edit, think and wait for people or the network outside it.
#   - One lock per job, not one for a whole session: a long piece of work is split into jobs that each queue again, so the
#     other project's turn comes in between (FIFO alternates the projects automatically).
#   - GPU_LOCK_EST_MIN=<minutes> says how long the job will take; the report to the waiters shows it. A job expected to run
#     more than 90 minutes should be split (a cold kernel compile can be 3 hours: warm first, in its own job).
#   - Never run a GPU program outside the lock, even "just a quick probe".
#   - A holder's heartbeat ($LOCK/beat) is touched every minute; a lock whose owner process is gone is taken over.
LABEL=${1:?usage: gpu_lock.sh <label> -- <command> | --status}
shift
LOCK=${PGSTROM_TEST_LOCK:-/tmp/pgstrom_test.lock}
QUEUE=$LOCK.queue

lstart_of() { ps -o lstart= -p "$1" 2>/dev/null | sed 's/^ *//'; }
alive() { # <pid> <recorded start time>
	[ -n "$1" ] || return 1
	kill -0 "$1" 2>/dev/null || return 1
	[ -z "$2" ] || [ "$(lstart_of "$1")" = "$2" ]
}
age_of() { m=$(stat -f %m "$1" 2>/dev/null) || return 0; echo $(( $(date +%s) - m )); }
status() {
	if [ -d "$LOCK" ]; then
		hs=$(cat "$LOCK/start" 2>/dev/null); held=0; [ -n "$hs" ] && held=$(( ($(date +%s) - hs) / 60 ))
		echo "held for $held min by: $(cat "$LOCK/tree" 2>/dev/null) (pid $(cat "$LOCK/pid" 2>/dev/null), heartbeat $(age_of "$LOCK/beat")s ago)"
	else
		echo "free"
	fi
	for t in $(ls "$QUEUE" 2>/dev/null); do
		tp=$(cat "$QUEUE/$t/pid" 2>/dev/null)
		alive "$tp" "$(cat "$QUEUE/$t/lstart" 2>/dev/null)" && echo "waiting: $(cat "$QUEUE/$t/tree" 2>/dev/null)"
	done
}
if [ "$1" = "--status" ]; then status; exit 0; fi
[ "$1" = "--" ] || { echo "usage: gpu_lock.sh <label> -- <command> | --status" >&2; exit 2; }
shift
[ $# -gt 0 ] || { echo "gpu_lock.sh: no command" >&2; exit 2; }
WHO="$LABEL${GPU_LOCK_EST_MIN:+ (est. $GPU_LOCK_EST_MIN min)}: $*"

mkdir -p "$QUEUE" 2>/dev/null
seq=$(ls "$QUEUE" 2>/dev/null | tail -1 | sed 's/^0*//'); seq=$(( ${seq:-0} + 1 ))
while :; do
	TICKET=$QUEUE/$(printf %010d "$seq")
	mkdir "$TICKET" 2>/dev/null && break
	seq=$((seq + 1))
done
echo $$ > "$TICKET/pid"; lstart_of $$ > "$TICKET/lstart"; echo "$WHO" > "$TICKET/tree"
trap 'rm -rf "$TICKET"' EXIT INT TERM
waited=0; n=0
while :; do
	ahead=0
	for t in $(ls "$QUEUE" 2>/dev/null); do
		[ "$QUEUE/$t" = "$TICKET" ] && break
		tp=$(cat "$QUEUE/$t/pid" 2>/dev/null)
		if [ -n "$tp" ] && ! alive "$tp" "$(cat "$QUEUE/$t/lstart" 2>/dev/null)"; then rm -rf "$QUEUE/$t"; continue; fi
		ahead=$((ahead + 1))
	done
	[ $ahead -eq 0 ] && break
	[ $waited -eq 0 ] && echo "gpu_lock: $ahead job(s) ahead in the queue; lock: $(cat "$LOCK/tree" 2>/dev/null)"
	waited=1; sleep 5
	n=$((n + 1)); [ $((n % 120)) -eq 0 ] && { echo "gpu_lock: still waiting"; status; }
done
while ! mkdir "$LOCK" 2>/dev/null; do
	owner=$(cat "$LOCK/pid" 2>/dev/null)
	if [ -n "$owner" ] && ! alive "$owner" "$(cat "$LOCK/lstart" 2>/dev/null)"; then rm -rf "$LOCK"; continue; fi
	sleep 5
done
echo $$ > "$LOCK/pid"; lstart_of $$ > "$LOCK/lstart"; echo "$WHO" > "$LOCK/tree"; date +%s > "$LOCK/start"; touch "$LOCK/beat"
rm -rf "$TICKET"
( me=$$; while kill -0 "$me" 2>/dev/null; do touch "$LOCK/beat" 2>/dev/null; sleep 60; done ) >/dev/null 2>&1 &
HB=$!
trap 'kill $HB 2>/dev/null; rm -rf "$LOCK"' EXIT INT TERM
echo "gpu_lock: got the lock ($WHO)"
"$@"
rc=$?
echo "gpu_lock: released (exit $rc)"
exit $rc
