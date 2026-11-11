# Sharing the GPU between projects (development machine)

This Mac's one Apple GPU is used by several projects at the same time: PG-Strom's tests and benchmarks, and the MongoDB GPU
experiments in `../mongo`. Two GPU programs at once distort every timing, and on Apple Silicon macOS can kill a Metal command buffer
("Impacting Interactivity"). All GPU work goes through one fair queue.

**Use** (any project, any language):

```
sh /Volumes/CrucialX9/Projects/pg-strom/test/gpuserv/gpu_lock.sh <label> -- <command> [args...]
sh /Volumes/CrucialX9/Projects/pg-strom/test/gpuserv/gpu_lock.sh <label> --status     # who holds it, who waits
GPU_LOCK_EST_MIN=20 sh .../gpu_lock.sh mongo-bench -- ./run_bench.sh                    # say how long it will take
```

The command runs with the lock held and its exit status is returned. `test/gpuserv/locked_test.sh` (PG-Strom's installer and
test wrapper) uses the same lock and the same queue, so PG-Strom runs and `gpu_lock.sh` runs wait in one line, first come first served.

**Rules**
1. Hold the lock only while the GPU is in use. Build, edit, think and wait for people or the network outside it.
2. One lock per job. Split long work into jobs that each queue again: FIFO then alternates the projects. Expect to be
   asked to split a job expected to run over 90 minutes (a cold kernel compile can take hours: warm in its own job).
3. Say the expected duration with `GPU_LOCK_EST_MIN`; it is shown to the waiters.
4. No GPU program outside the lock, not even a quick probe (a Vulkan or Metal device probe is a GPU program).
5. A process that dies releases the lock (the holder's process is checked; the heartbeat file is touched every minute).

**Files** (the protocol, for tools that cannot call the script): lock directory `/tmp/pgstrom_test.lock` (`pid`, `lstart`, `tree`,
`start`, `beat`), queue `/tmp/pgstrom_test.lock.queue/<10-digit ticket>/` (`pid`, `lstart`, `tree`). `PGSTROM_TEST_LOCK` moves
both; every participant must use the same value.

**Disk and caches.** The local drive is nearly full by design. Large caches go to the external volume
(`/Volumes/CrucialX9/...`); see the notes on the PG-Strom kernel cache in `CLAUDE.md`.
