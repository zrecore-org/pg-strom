---
name: vulkan-verify
description: How to test a change on the PG-Strom vulkan-gpu branch: which tests to run for which files, the full verification before a push, the test lock, timing files, and the quick reduced-size pass. Use before running or choosing tests, and before committing or pushing.
---

# Verifying a change (vulkan-gpu branch)

Standing rule from the user: **commit and push when green** (full verification below). When the user says
"don't commit until all items are done and green", that overrides it.

## 1. Pick the tests

```
sh test/gpuserv/affected.sh                 # changed files -> the dev scripts to run (RUN=1 runs them)
sh test/gpuserv/affected.sh origin/vulkan-gpu..HEAD
```

Dev scripts take a few minutes: `postgis_dev.sh`, `postgis3d_dev.sh`, `gist_dev.sh`, `float8_dev.sh`, `arrow_dev.sh`,
`gpucache_dev.sh`, `joinagg_dev.sh`, `gpusort_dev.sh`, `spec_dev.sh`, `recheck_dev.sh` (A12: geometry rechecks on Arrow, Parquet and GPU-Cache sources). Run each as
`PGSTROM_GPUSERV_SPECIALIZE=check sh test/gpuserv/locked_test.sh "$PWD" <script>`.
`sh test/gpuserv/spec_compile_check.sh` (no GPU, no lock) must pass after any change to kernels or the specializer.

## 2. Full verification (before a push)

```
STAGES="installcheck scan" sh test/gpuserv/verify_all.sh      # expect "All 23 tests passed" and ALL PASSED
```

- Always build and install first: `make -C src PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config install`.
  After editing a header in `src/gpuserv/` touch the `.c` files (`touch src/gpuserv/*.c`): the Makefile does not track
  headers and stale objects give a garbage self-test failure.
- Everything goes through `locked_test.sh` (one global lock; the installed `$libdir/pg_strom` is shared). Do not install or
  edit `test/gpuserv/*.inc` while a run is going: the shell sources those files as it reaches them.
- A cold Metal compile (first run after any change to `kernels_vm/`) takes minutes per kernel variant. A slow first pass is
  not a hang; look at `ps` for `MTLCompilerService` CPU time before assuming one.
- Quick pass for plumbing changes: `NROWS=30000 STAGES="installcheck scan" TIMEOUT=3000 sh test/gpuserv/verify_all.sh`.
- Optional stages: `fuzz`, `float8`, `postgis`, `arrow`, `gpucache` (see the header of `verify_all.sh`).

## Keeping the machine usable

`locked_test.sh` builds with `TEST_JOBS` jobs (default 3 of 8 cores) and runs the test at `nice -n 10` (`TEST_PRIORITY=nice`, the
default; `background` = `taskpolicy -b` also throttles disk I/O and is about 5x slower; `TEST_FULL_SPEED=1` turns it off). The
cluster settings come from `test/gpuserv/pgconf_test.sh` (fsync off, autovacuum off, parallel workers capped). GPU load is not
limited: on macOS long kernels can make the display lag and trigger "Impacting Interactivity" (see gpu-hang-triage); never
run several services at once on the Apple GPU (`scan_shards.sh` is for Linux/CI).

## 3. Reading results

- `test-data/logs/<stamp>/<n>-<stage>.log` and `.log.times` (seconds per group of checks and the slowest checks, from
  `timeline.pl`). The stage line in the output ends with the wall seconds.
- Every `psql` call in the scripts has a limit (`QUERY_TIMEOUT`, default 900 s, `qtimeout.sh`). After one timeout the
  rest of that test fails at once; the flag file is `<ROOT>/.query_timeout`.
- `TIMEOUT=<s>` caps a stage (default 4200); a stage that is cut off has its cluster and service stopped.
- A failing test keeps its cluster under `test-data/<name>` for diagnosis; `KEEP=1` keeps a passing one.

## 4. Commit

Short message with a `vulkan:` / `test:` / `docs:` prefix and the trailer
`Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Scratch files go under `test-data/scratch/`, never `/tmp`.
