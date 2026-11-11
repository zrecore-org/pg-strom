#!/bin/sh
# Differential reproducer for the NVIDIA failures of float8_dev (negative integers through %, max with NaN): builds
# a SQL file from nvidia_probe_queries.txt (each query once with the GPU, once with pg_strom.enabled=off, plus the plan
# line) and runs it through float8_dev.sh (ADHOC, NOTESTS). Run it like float8_dev.sh (GPUSERV_BIN, PGBIN, ...).
D=${0%/*}
OUT=${PROBE_SQL:-/tmp/nvidia_probe.sql}
{
	cat "$D/nvidia_probe_setup.sql"
	echo "set pg_strom.gpu_setup_cost=0; set pg_strom.gpu_tuple_cost=0; set max_parallel_workers_per_gather=0; set pg_strom.gpupreagg_max_groups=1000000; set cpu_operator_cost=0.1; set cpu_tuple_cost=0.1;"
	while IFS= read -r q; do
		[ -z "$q" ] && continue
		printf '\\echo ## %s\n' "$q"
		echo "set pg_strom.enabled=on;"
		printf '\\echo plan:\nexplain (costs off) %s;\n\\echo gpu:\n%s;\n' "$q" "$q"
		echo "set pg_strom.enabled=off;"
		printf '\\echo cpu:\n%s;\n' "$q"
	done < "$D/nvidia_probe_queries.txt"
} > "$OUT"
ADHOC=$OUT NOTESTS=1 sh "$D/float8_dev.sh"
