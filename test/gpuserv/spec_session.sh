#!/bin/sh
# One lock hold for the specialization checks of a development cycle: the quick specialization tests
# (spec_dev.sh, on a small data set), then the benchmark (perf_bench.sh) with the variants in SPECS
# (default "off default"), and with SWEEP=1 a sweep of the GROUP BY kernel's table size and pages per
# workgroup over the GROUP BY queries. Run it through locked_test.sh.
D=${0%/*}
sh "$D/spec_dev.sh"
rc=$?
echo "=== benchmark"
SPECS=${SPECS:-off default} sh "$D/perf_bench.sh"
if [ -n "$SWEEP" ]; then
	echo "=== sweep of the GROUP BY kernel (force mode, table slots, pages per workgroup)"
	NOCPU=1 LABEL_RE='group' SPECS="force force:PGSTROM_GPUSERV_SPEC_GA_C=16 force:PGSTROM_GPUSERV_SPEC_GA_C=32 force:PGSTROM_GPUSERV_SPEC_GA_C=64 force:PGSTROM_GPUSERV_SPEC_GA_C=256 force:PGSTROM_GPUSERV_GROUP_PPW=8 force:PGSTROM_GPUSERV_GROUP_PPW=16 force:PGSTROM_GPUSERV_GROUP_PPW=64" sh "$D/perf_bench.sh"
fi
exit $rc
