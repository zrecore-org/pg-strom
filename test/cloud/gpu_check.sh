#!/bin/sh
# Run the GPU tests of the Vulkan port on a rented (or any) Ubuntu 22.04/24.04 x86_64 or arm64 machine with a GPU, and print
# one summary. Meant for an hour on a cloud box (AWS g4dn = NVIDIA T4, g4ad = AMD Radeon Pro V520, Lambda, RunPod, Vast.ai)
# that you shut down afterwards; the same sequence as the `linux-gpu` job of .github/workflows/vulkan-check.yml.
#
#   curl -fsSL https://raw.githubusercontent.com/zrecore/pg-strom/vulkan-gpu/test/cloud/gpu_check.sh | sh
#   (or clone the repository and run: sh test/cloud/gpu_check.sh)
#
# Environment: BRANCH (default vulkan-gpu), REPO (default https://github.com/zrecore/pg-strom), WORK (default ~/pgstrom-gpu-check),
# STAGES (default "service float8 postgis gist gpucache"; add "scan" for the 2500-check scan_test.sh, about an hour on a GPU).
# NVIDIA: the Vulkan ICD ships with the graphics driver package (libnvidia-gl-<major>), which a compute-only image may lack: the
# script installs it when `vulkaninfo` lists no NVIDIA device. AMD/Intel: Mesa (mesa-vulkan-drivers).
# Needs sudo. The tests refuse a CPU device (ALLOW_CPU=off), so a missing GPU driver fails the service test instead of passing
# on lavapipe.
set -u
BRANCH=${BRANCH:-vulkan-gpu}
REPO=${REPO:-https://github.com/zrecore/pg-strom}
WORK=${WORK:-$HOME/pgstrom-gpu-check}
STAGES=${STAGES:-"service float8 postgis gist gpucache"}
PG_MAJOR=${PG_MAJOR:-17}
SLANG_VERSION=${SLANG_VERSION:-2026.20}
ARCH=$(uname -m)
case "$ARCH" in
	x86_64)  SLANG_ASSET=slang-$SLANG_VERSION-linux-x86_64.tar.gz;  SLANG_SHA=e863e568c204810d6e8bc31852f056aef6763c55ea9925db2db7affa060347c9 ;;
	aarch64) SLANG_ASSET=slang-$SLANG_VERSION-linux-aarch64.tar.gz; SLANG_SHA=b0345743e2153d8c2bd33f56fb5696bcef63fc03513f054a1ae1b8d3dd0152a9 ;;
	*) echo "unsupported architecture $ARCH"; exit 2 ;;
esac
say() { echo "== $*"; }
mkdir -p "$WORK" && cd "$WORK" || exit 2

say "packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends ca-certificates curl wget lsb-release gnupg git build-essential pkgconf \
	libvulkan-dev mesa-vulkan-drivers vulkan-tools libnuma-dev libkrb5-dev libicu-dev postgresql-common >/dev/null
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y >/dev/null
sudo apt-get install -y postgresql-$PG_MAJOR postgresql-server-dev-$PG_MAJOR >/dev/null
sudo chown -R "$USER" /usr/lib/postgresql/$PG_MAJOR /usr/share/postgresql/$PG_MAJOR
wget -q "https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb"
sudo apt-get install -y ./apache-arrow-apt-source-latest-*.deb >/dev/null
sudo apt-get update -qq && sudo apt-get install -y libarrow-dev libparquet-dev >/dev/null

say "Slang $SLANG_VERSION"
D=$HOME/tools/slang-$SLANG_VERSION
mkdir -p "$D"
curl -fsSL -o /tmp/slang.tar.gz "https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/$SLANG_ASSET"
echo "$SLANG_SHA  /tmp/slang.tar.gz" | sha256sum -c - || exit 2
tar -xzf /tmp/slang.tar.gz -C "$D"
SLANGC=$(find "$D" -type f -name slangc | head -n 1)
export SLANGC

say "the GPU"
nvidia-smi 2>&1 | head -12 || true
if command -v nvidia-smi >/dev/null 2>&1 && ! vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
	DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
	echo "no NVIDIA Vulkan device: installing libnvidia-gl-$DRV"
	sudo apt-get install -y "libnvidia-gl-$DRV" >/dev/null 2>&1 || sudo apt-get install -y "libnvidia-gl-$DRV-server" >/dev/null 2>&1 || true
fi
vulkaninfo --summary 2>&1 | grep -E "deviceName|deviceType|driverName|driverInfo|apiVersion" | head -20

say "source"
if [ -d src ] && [ -f Makefile.common ]; then SRC=$PWD
elif [ -d pg-strom ]; then (cd pg-strom && git fetch -q origin "$BRANCH" && git checkout -q -B "$BRANCH" "origin/$BRANCH"); SRC=$PWD/pg-strom
else git clone -q --branch "$BRANCH" --depth 1 "$REPO" pg-strom && SRC=$PWD/pg-strom
fi
cd "$SRC" || exit 2
git log -1 --format='%h %s' 2>/dev/null

say "build"
make -C src PG_CONFIG=/usr/lib/postgresql/$PG_MAJOR/bin/pg_config WITH_GPUSERV=1 -j"$(nproc)" install >"$WORK/build.log" 2>&1 \
	|| { tail -30 "$WORK/build.log"; echo "BUILD FAILED"; exit 1; }

export PGBIN=/usr/lib/postgresql/$PG_MAJOR/bin GPUSERV_BIN=/usr/lib/postgresql/$PG_MAJOR/lib/pgstrom-gpuserv ALLOW_CPU=off
fail=0
for st in $STAGES; do
	case $st in
		service)  cmd="test/gpuserv/service_test.sh" ;;
		float8)   cmd="test/gpuserv/float8_dev.sh" ;;
		postgis)  cmd="test/gpuserv/postgis_dev.sh" ;;
		gist)     cmd="test/gpuserv/gist_dev.sh" ;;
		gpucache) cmd="test/gpuserv/gpucache_dev.sh" ;;
		scan)     cmd="test/gpuserv/scan_test.sh" ;;
		*) echo "unknown stage $st"; continue ;;
	esac
	say "$st"
	if sh "$cmd" >"$WORK/$st.log" 2>&1; then echo "$st: PASSED ($(grep -c '  ok  ' "$WORK/$st.log") checks)"
	else echo "$st: FAILED"; grep -E "FAIL|ERROR" "$WORK/$st.log" | head -8; fail=$((fail+1)); fi
done

say "summary"
vulkaninfo --summary 2>/dev/null | grep deviceName | head -2
if [ "$fail" -eq 0 ]; then echo "ALL STAGES PASSED"; else echo "$fail STAGE(S) FAILED; logs in $WORK"; fi
exit "$fail"
