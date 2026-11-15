#!/bin/sh
# A first-come-first-served turn for the NVIDIA eGPU on the Windows host (desktop-nvg0avo, driven through the Remote Control
# session "win"), shared by the projects on this Mac (pg-strom, mongo, exactmesh). The GPU itself is not on this machine, so the
# lock is here, on the Mac the agents run on; it is the queue and lock protocol of gpu_lock.sh with its own lock path
# (/tmp/win_egpu.lock, /tmp/win_egpu.lock.queue), so it never blocks the local Apple GPU nor the other way round.
#
#   sh win_gpu_lock.sh hold <label> [max-minutes]   WAIT in the queue; when it is your turn print "HOLDING", then keep the turn until
#                                                    release (or max-minutes, default 60, after which the turn ends). Run it in the
#                                                    BACKGROUND (a long-running command) and read its output for "HOLDING".
#   sh win_gpu_lock.sh release <label>              give the turn back (also fine to call twice)
#   sh win_gpu_lock.sh status                       who holds it and who waits (no waiting)
#
# Rules (the user's, 2026-11-15): only the holder asks `win` to run anything on the GPU, one GPU program at a time; the turn covers
# the GPU work only (build, edit and think before and after); keep a turn short (minutes, not hours); a job that needs a long
# cold shader compile (the NVIDIA driver takes 15-40 min on the big join kernels) gets its own turn with an honest max-minutes;
# if the GPU is lost on the Windows side (nvidia-smi fails, "Invalid physicalDevice"), release at once and tell the user, nobody
# resets drivers or Windows settings from here. Tell `win` at the start of the job which project holds the turn.
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${WIN_GPU_LOCK:-/tmp/win_egpu.lock}
REL=/tmp/win_egpu.release
case "$1" in
status) PGSTROM_TEST_LOCK=$LOCK sh "$HERE/gpu_lock.sh" x --status ;;
release) [ -n "$2" ] || { echo "usage: win_gpu_lock.sh release <label>" >&2; exit 2; }
	touch "$REL.$2"; echo "release requested for $2" ;;
hold) label=${2:?usage: win_gpu_lock.sh hold <label> [max-minutes]}
	max=${3:-60}
	rm -f "$REL.$label"
	GPU_LOCK_EST_MIN=$max PGSTROM_TEST_LOCK=$LOCK exec sh "$HERE/gpu_lock.sh" "$label" -- sh "$HERE/win_gpu_lock.sh" _turn "$label" "$max" ;;
_turn) label=$2; end=$(( $(date +%s) + $3 * 60 ))
	echo "HOLDING: the Windows GPU is yours ($label), max $3 min; give it back with: sh $HERE/win_gpu_lock.sh release $label"
	while [ ! -e "$REL.$label" ] && [ $(date +%s) -lt $end ]; do sleep 5; done
	rm -f "$REL.$label"; echo "turn over ($label)" ;;
*) sed -n 2,19p "$0"; exit 2 ;;
esac
