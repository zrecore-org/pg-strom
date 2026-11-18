#!/bin/sh
# Collect the measurement logs behind the numbers in .claude/hw-acceleration.md into one dated, hashed folder
# (test-data/paper/<stamp>/, git-ignored, on the external volume) so a paper can cite them after a session restart.
#   sh test/gpuserv/paper_archive.sh            copy logs (each file below MAXKB, default 2048) + MANIFEST + a .tar.gz
# The folder gets: logs/ (copies), MANIFEST.txt (git head, machine, tool versions, sha256 and size of every file),
# and the tarball next to it. Nothing is deleted or modified in test-data/.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO" || exit 2
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$REPO/test-data/paper/$STAMP}
MAXKB=${MAXKB:-2048}
mkdir -p "$OUT/logs" || exit 2
list=$OUT/files.txt
: > "$list"
# patterns: TPC-H and calibration, numeric/decode attribution, benchmarks, sort, df64, A14/A3 runs, Arrow/cold, verify runs
for pat in 'test-data/tpch*' 'test-data/dbg_*.log' 'test-data/a10-*.log' 'test-data/a3-*bench*.log' 'test-data/a6-num*.log' \
	'test-data/arrow_*.log' 'test-data/affected_chain*.log' 'test-data/cov_run*.log' 'test-data/*rsort*' 'test-data/*sort*.log' \
	'test-data/verify_all*.log' 'test-data/logs/2026111*/*.times' 'test-data/logs/2026111*/*.log' 'test-data/logs/2026111*/*.rc'; do
	for f in $pat; do
		[ -f "$f" ] || continue
		kb=$(( ($(wc -c < "$f") + 1023) / 1024 ))
		[ "$kb" -le "$MAXKB" ] || { echo "skipped (>${MAXKB} KB): $f" >> "$OUT/skipped.txt"; continue; }
		echo "$f" >> "$list"
	done
done
sort -u "$list" -o "$list"
while IFS= read -r f; do
	d="$OUT/logs/$(dirname "$f")"; mkdir -p "$d"; cp -p "$f" "$d/"
done < "$list"
# GitHub Actions logs of the successful manual runs (the NVIDIA T4 measurements of A17 live only there; GitHub keeps logs
# about 90 days). CI=0 skips this (needs gh, takes a few minutes).
if [ "${CI:-1}" = 1 ] && command -v gh >/dev/null; then
	mkdir -p "$OUT/logs/ci"
	for id in $(gh run list --workflow vulkan-check.yml --limit 100 --json databaseId,event,conclusion --jq '.[]|select(.event=="workflow_dispatch" and .conclusion=="success")|.databaseId'); do
		gh run view "$id" --json createdAt,conclusion,headSha,jobs > "$OUT/logs/ci/run-$id.json" 2>/dev/null
		timeout 180 gh run view "$id" --log > "$OUT/logs/ci/run-$id.log" 2>/dev/null
	done
fi
{
	echo "stamp: $STAMP"
	echo "git head: $(git rev-parse HEAD) ($(git log -1 --format=%s | cut -c1-80))"
	echo "git branch: $(git rev-parse --abbrev-ref HEAD)"
	echo "host: $(uname -a)"
	echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)  cores: $(sysctl -n hw.ncpu 2>/dev/null)  mem: $(sysctl -n hw.memsize 2>/dev/null)"
	echo "postgres: $(/opt/homebrew/opt/postgresql@17/bin/postgres --version 2>/dev/null)"
	echo "slangc: $(${SLANGC:-$HOME/tools/slang-2026.20/bin/slangc} -v 2>&1 | head -1)"
	echo "vulkan device: $(sed -n 's/.*Vulkan device 0: \(.*\)/\1/p' test-data/logs/*/2-scan.diag/log 2>/dev/null | head -1)"
	echo
	echo "sha256  bytes  file"
	(cd "$OUT/logs" && find . -type f | sort | while IFS= read -r f; do
		printf '%s  %s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(wc -c < "$f" | tr -d ' ')" "$f"
	done)
} > "$OUT/MANIFEST.txt"
(cd "$(dirname "$OUT")" && tar -czf "$STAMP.tar.gz" "$STAMP")
echo "archived $(wc -l < "$list" | tr -d ' ') files, $(du -sh "$OUT" | cut -f1), tarball $(dirname "$OUT")/$STAMP.tar.gz"
[ -f "$OUT/skipped.txt" ] && echo "skipped $(wc -l < "$OUT/skipped.txt" | tr -d ' ') large files (see skipped.txt)"
