#!/bin/sh
# One lock hold for the specialization checks of a development cycle: the quick specialization tests
# (spec_dev.sh, on a small data set) and then the benchmark (perf_bench.sh) with the variants in SPECS
# (default "off default"). Run it through locked_test.sh.
D=${0%/*}
sh "$D/spec_dev.sh"
rc=$?
echo "=== benchmark"
SPECS=${SPECS:-off default} sh "$D/perf_bench.sh"
exit $rc
