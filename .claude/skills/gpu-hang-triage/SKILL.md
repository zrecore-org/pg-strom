---
name: gpu-hang-triage
description: Diagnose a PG-Strom test or query that hangs, stalls, or fails at a GPU dispatch (kernel did not finish, device lost, service restarts, connection lost, self-test failed). Use when a verification run stops making progress.
---

# A hung or failing GPU run

Gather evidence before killing anything; the cluster is kept, so you can still look.

1. **Is it slow or hung?** `ps -ax -o pid,etime,time,stat,command | grep -E 'gpuserv --serve|MTLCompilerServ|postgres:'`.
   - `MTLCompilerService` CPU time growing = a cold Metal compile (minutes per interpreter variant after a change to
     `kernels_vm/`). The same query is seconds on the second run. Wait; do not kill.
   - Service CPU flat and the backend waiting = a real hang.
2. **Which side waits?** `sample <pid> 1 -file test-data/scratch/x.txt`, then read the "Call graph".
   - Service in `gpu_vk_pipeline_create_ex` = the driver is compiling.
   - Backend in `__waitAndFetchNextXpuCommand` = waiting for a result (should now error with "the connection to the GPU
     service was lost" if the service died).
   - Backend in `close` (state `U`, unkillable) = macOS kernel wait on the service socket; it ends when the service is
     killed (`kill -9 <service pid>`). Seen once during the `gpu_cache` installcheck test, cause not found.
3. **Service log**: `LC_ALL=C grep -a "did not finish\|device was lost\|self-test\|ready:" <cluster>/log*`. A line
   `kernel did not finish within N ms (device marked lost)` is the watchdog (default 10 s; the first dispatch of a pipeline
   gets 600 s; `PGSTROM_GPUSERV_TIMEOUT_MS` overrides both). The service then exits and the launcher restarts it (`generation`).
4. **`VM self-test FAILED` with garbage text** after editing `src/gpuserv/*.h`: stale objects. `touch src/gpuserv/*.c` and
   rebuild.
5. **Reproduce small**: start the kept cluster by hand (`pg_ctl -D <cluster>/data -l <log> start`, optionally with
   `PGSTROM_GPUSERV_TIMEOUT_MS=3000` or `PGSTROM_GPUSERV_SPECIALIZE=off|check` in the environment) and run the one query with
   `perl -e 'alarm 100; exec @ARGV' psql ...`. Stop it with `pg_ctl -m immediate stop` and `pkill -9 -f 'pgstrom-gpuserv --serve'`.
6. **More evidence**: `PGSTROM_GPUSERV_TRACE=1` (xpucode and per-task timings), `PGSTROM_GPUSERV_SPEC_LOG=1` (specializer),
   `sh test/gpuserv/vk_validation.sh` (Vulkan validation layer).

Cluster data dirs are under `test-data/`; the test log is `test-data/logs/<stamp>/`. Do not use `rm -rf` to clear them
(a safety check blocks it); use a fresh directory.
