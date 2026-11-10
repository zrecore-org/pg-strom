#!/bin/sh
# prewarm_kernels.sh <tree> <pgstrom-gpuserv>
#
# After a build: compile the interpreter kernels whose SPIR-V changed since the last run once, one after the other,
# so that a test does not pay for it at its first query (the driver's compile of a full kernel takes from seconds to
# minutes; the result lands in the pipeline cache file of the service, src/gpuserv/gpu_vk.c, which every later service
# start reuses).
#
# The kernels checked are the ones the tests use most (the smallest register file of each family, 4 aggregates, full,
# lite and nogeom; and the 32-register aggregation-over-join kernels, which the geometry programs need and which cost
# about 5 minutes each to compile when cold). Their checksums are kept in <cache dir>/prewarm.sums; the first run only records them. A kernel whose
# checksum is unchanged is not touched, so an edit of one .slang file costs the compile of the kernels it changed.
#   PREWARM=off        skip this step
#   PREWARM=force      compile every kernel of the list now, whatever the checksums say (after deleting the cache file)
#   PREWARM=all        also the other register/aggregate variants (can take an hour after an edit of scan_common.slang)
#   PGSTROM_GPUSERV_PIPELINE_CACHE=off or PGSTROM_GPUSERV_CACHE_DIR=none: nothing to warm, skipped
TREE=${1:?usage: prewarm_kernels.sh <tree> <pgstrom-gpuserv>}
BIN=${2:?usage: prewarm_kernels.sh <tree> <pgstrom-gpuserv>}
[ "${PREWARM:-default}" = off ] && exit 0
[ "$PGSTROM_GPUSERV_PIPELINE_CACHE" = off ] && exit 0
[ "$PGSTROM_GPUSERV_CACHE_DIR" = none ] && exit 0
BUILD=$TREE/src/gpuserv/build
[ -d "$BUILD" ] || exit 0
if [ -n "$PGSTROM_GPUSERV_PIPELINE_CACHE_DIR" ]; then DIR=$PGSTROM_GPUSERV_PIPELINE_CACHE_DIR
elif [ -n "$PGSTROM_GPUSERV_CACHE_DIR" ]; then DIR=$PGSTROM_GPUSERV_CACHE_DIR
elif [ -n "$XDG_CACHE_HOME" ]; then DIR=$XDG_CACHE_HOME/pgstrom
else DIR=$HOME/.cache/pgstrom; fi
mkdir -p "$DIR" 2>/dev/null || exit 0
SUMS=$DIR/prewarm.sums
NEW=$DIR/prewarm.sums.$$
OLD_EMPTY=0
[ -s "$SUMS" ] || OLD_EMPTY=1

# name  nbind  pc_size
if [ "${PREWARM:-default}" = all ]; then
	LIST=$(cd "$BUILD" && ls gpuscan*.spv gpuagg*.spv gagg*.spv gpuarrow*.spv aagg*.spv agagg*.spv gpujoin*.spv jagg*.spv jgagg*.spv ajoin*.spv ajagg*.spv ajgagg*.spv 2>/dev/null | sed 's/\.spv$//')
else
	LIST=
	for k in gpuscan8 gpuagg8_4 gagg16_4 gpujoin8 jagg16_4 jgagg16_4 gpuarrow8 aagg16_4 agagg16_4 ajoin16 ajagg16_4 ajgagg16_4 \
			 jagg32_4 jgagg32_4 ajagg32_4 ajgagg32_4; do
		LIST="$LIST $k ${k}_lite ${k}_nogeom"
	done
	# R15: the geometry tiers of the join kernel families (3-8 MB, a fraction of the full kernel's compile time); the register-8 join kernel
	# has none (a program with 8 registers uses the 16-register one)
	for k in gpujoin16 jagg16_4 jgagg16_4 ajoin16 ajagg16_4 ajgagg16_4 jagg32_4 jgagg32_4 ajagg32_4 ajgagg32_4; do
		LIST="$LIST ${k}_gpip ${k}_gdw ${k}_gwalk"
	done
fi
# Only what the test script needs (PREWARM_FOR=<script name>, set by locked_test.sh): a script whose programs use no
# geometry never runs the full tier of a kernel (its programs use the lite or the nogeom variant), and the full tier is
# the one that costs about 5 minutes per kernel when cold, so those scripts warm only the _lite and _nogeom variants.
# The geometry scripts, the whole-suite scripts and any script not named here keep the whole list. A variant that a
# script needs and the set leaves out is compiled at its first use, as before.
#   PREWARM_SET=all|nongeom overrides the choice by script name
case "${PREWARM_SET:-$PREWARM_FOR}" in
	nongeom|float8_dev.sh|arrow_dev.sh|keys_dev.sh|fallback_dev.sh|gtab_dev.sh|spec_dev.sh|joinagg_dev.sh|gpucache_dev.sh|gpusort_dev.sh|hash_check.sh)
		FILTERED=
		for k in $LIST; do case "$k" in *_lite|*_nogeom) FILTERED="$FILTERED $k";; esac; done
		LIST=$FILTERED ;;
esac
# The geometry scripts of the join families run their geometry programs on the geometry tiers (point in polygon, ST_DWithin, the walker);
# the full join kernel (9.8 MB, 5 minutes cold) is only for a program that mixes groups (compiled at its first use, as any variant the set leaves out)
case "${PREWARM_SET:-$PREWARM_FOR}" in
	gist_dev.sh|postgis_dev.sh|postgis3d_dev.sh|recheck_dev.sh)
		FILTERED=
		for k in $LIST; do
			case "$k" in
				gpujoin8|jagg16_4|jgagg16_4|ajoin16|ajagg16_4|ajgagg16_4|jagg32_4|jgagg32_4|ajagg32_4|ajgagg32_4) ;;
				*) FILTERED="$FILTERED $k";;
			esac
		done
		LIST=$FILTERED ;;
esac
# PREWARM_LIST_ONLY=1: print the kernels that would be checked and stop (test of the sets)
[ -n "$PREWARM_LIST_ONLY" ] && { for k in $LIST; do echo "$k"; done; exit 0; }
geom() {	# nbind and push constant bytes by kernel family (scan_exec.c, arrow_exec.c)
	case $1 in
	gpuscan*|gpuagg*|gagg*|gpuarrow*) echo "5 16";;
	aagg*|agagg*) echo "5 20";;
	*) echo "6 20";;
	esac
}
checksum() { shasum -a 256 "$1" | cut -d' ' -f1; }

: > "$NEW"
TODO=
for k in $LIST; do
	f=$BUILD/$k.spv
	[ -f "$f" ] || continue
	s=$(checksum "$f")
	echo "$k $s" >> "$NEW"
	if [ "$PREWARM" = force ]; then TODO="$TODO $k"; continue; fi
	[ $OLD_EMPTY = 1 ] && continue
	old=$(grep "^$k " "$SUMS" | cut -d' ' -f2)
	[ "$old" = "$s" ] || TODO="$TODO $k"
done
if [ $OLD_EMPTY = 1 ] && [ "$PREWARM" != force ]; then
	echo "prewarm: recorded the kernel checksums (the next change of a kernel is compiled before the test)"
	mv "$NEW" "$SUMS"
	exit 0
fi
if [ -z "$TODO" ]; then rm -f "$NEW"; exit 0; fi
echo "prewarm: kernels changed since the last run:$TODO"
t0=$(date +%s)
rc=0
for k in $TODO; do
	set -- $(geom "$k")
	nice -n 10 "$BIN" --prewarm "$1" "$2" "$BUILD/$k.spv" || rc=1
done
echo "prewarm: $(( $(date +%s) - t0 )) s"
# kernels that failed to compile are tried again next time
if [ $rc = 0 ]; then mv "$NEW" "$SUMS"; else rm -f "$NEW"; fi
exit 0
