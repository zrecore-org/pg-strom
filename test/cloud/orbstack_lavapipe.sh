#!/bin/sh
# Reproduce the CI's Linux lavapipe jobs on an Apple silicon Mac: an Ubuntu 24.04 arm64 container (OrbStack or Docker Desktop)
# builds the tree and runs stages of test/cloud/gpu_check.sh against Mesa's lavapipe (CPU Vulkan, subgroup size 4), which is
# what the GitHub-hosted Linux runners have. The tree is copied into the container (the host's objects stay untouched).
#
#   sh test/cloud/orbstack_lavapipe.sh                 # STAGES=gpucache, NROWS=20000 (the CI's GPU-Cache step)
#   STAGES="service float8" sh test/cloud/orbstack_lavapipe.sh
# Output: test-data/scratch/orb_lavapipe.log (the container is removed afterwards; logs of the tests are in /work/*.log inside it,
# printed on failure).
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
STAGES=${STAGES:-gpucache}
mkdir -p "$REPO/test-data/scratch"
exec docker run --rm --platform linux/arm64 -v "$REPO":/src:ro \
	-e STAGES="$STAGES" -e NROWS="${NROWS:-20000}" -e F8ROWS="${F8ROWS:-20000}" -e F8PARTS="${F8PARTS:-scan}" \
	ubuntu:24.04 bash -c '
set -e
apt-get update -qq && apt-get install -y -qq sudo rsync >/dev/null
useradd -m -s /bin/bash ci && echo "ci ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ci
mkdir /work && rsync -a --exclude test-data --exclude .git --exclude "*.o" --exclude "*.dylib" --exclude "*.so" --exclude "*.bc" /src/ /work/ && chown -R ci /work
su ci -c "cd /work && WORK=/work ALLOW_CPU=on STAGES=\"$STAGES\" NROWS=$NROWS F8ROWS=$F8ROWS F8PARTS=$F8PARTS sh test/cloud/gpu_check.sh"
rc=$?
for f in /work/*.log; do [ "$rc" -ne 0 ] && echo "=== $f" && tail -40 "$f"; done
exit $rc
'
