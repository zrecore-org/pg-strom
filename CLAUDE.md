# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PG-Strom is a PostgreSQL extension (C/C++ host code + GPU kernels) that offloads scan / join / group-by to GPUs, reads Apache Arrow / Parquet via a foreign data wrapper (Arrow_Fdw), and (on the CUDA line) supports GPU-Cache and GPU-Sort. Requires PostgreSQL v15+ (see `src/pg_strom.h`).

**Two lines of development.** `master` is the CUDA/NVIDIA product (Linux, CUDA toolkit; nothing builds or runs on a Mac). The `vulkan-gpu` branch (formerly `apple-silicon`) is a hard fork in which CUDA is gone: GPU work runs through Vulkan compute (Slang kernels compiled to SPIR-V) in a separate executable, and it is validated on Apple Silicon (MoltenVK) only. Arm64 Linux with lavapipe (a CPU Vulkan device) is used to check the host code. The CUDA sources (`cuda_*.cu`, `xpu_*.cu`, `gpu_service.c`, `gpu_cache.c`, `extra.c`, ...) are no longer in this branch; read them with `git show master:src/<file>` when porting a milestone. **The plan, status and findings are in `.claude/hw-acceleration.md`; read its "A2 status" first.**

Where the branch stands: GpuScan (heap, Arrow IPC, Parquet row groups), GpuJoin (inner/LEFT/RIGHT/FULL/semi/anti hash joins, up to four inner relations) and GpuPreAgg (with and without GROUP BY on up to two integer/date-time keys, also directly over a join; all three also with an Arrow/Parquet table as the scanned relation, `kernels_vm/arrow_load.inc` + `-DARROW_SRC`) run on the GPU. Types on the GPU: bool, int1-8, float4, date/time/timestamp, numeric (values that fit 64 bits, the rest re-checked on the CPU), text (equality, simple LIKE, C-collation ordering); strings and numerics that the kernel cannot read (compressed, out of line, too big) take the per-row CPU re-evaluation path (`KDS_FORMAT_RECHECK`). GpuPreAgg returns partial aggregate states that PostgreSQL's Agg node finalizes. GROUP BY is only faster than PostgreSQL for few groups (`pg_strom.gpupreagg_max_groups`, default 16). GPU-Sort of GpuPreAgg groups and GPU-Cache (A7: `gpu_cache.c` + `src/gpuserv/gcache.c`, scan/aggregate/join over a cached table through the Arrow kernels with a row filter) work; jsonb quals run on the GPU (`->`, `->>`, `?`, `jsonb_typeof`, `@>` against a constant, `(->>)::numeric/int`; a jsonb or text value *computed* in the target list stays on the CPU, A8); kernel specialization (A6) is not done; `parquet_cache.c` is a stub. `README.md` and `doc/features.md` carry the user-facing list; the plan's status sections are authoritative.

## Build / install (vulkan-gpu branch)

All builds go through PGXS against a specific `pg_config`; always pass `PG_CONFIG`:

```
make -C src PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config install   # module + pgstrom-gpuserv
make -C src/gpuserv                                                          # the service alone (needs slangc and the Vulkan loader)
```

- `WITH_GPUSERV=auto|1|0` (default auto): build the service when `slangc` and the Vulkan headers are found; `1` makes a missing prerequisite an error.
- `PGSTROM_DEBUG=1`: `-g -O0 -DPGSTROM_DEBUG_BUILD`. `PGSTROM_FLAGS_CUSTOM`: extra host flags.
- `libarrow` and `libparquet` are required (`brew install apache-arrow pkgconf` on macOS); `libnuma` is optional. Host C++ uses C++20. GNU make 3.81 and the macOS `sh` are supported (no `echo -n`, `\#` inside `$(shell)`).
- The extension must be loaded at postmaster start: `shared_preload_libraries = 'pg_strom'`. The postmaster probes the Vulkan devices by running `pgstrom-gpuserv --probe`, and a launcher background worker runs and restarts the service (`pg_strom.gpuserv_path`, `pg_strom.gpu_allow_cpu_device`, `pg_strom.gpu_visible_devices`).
- `arrow-tools/` (`pg2arrow`, `arrow2csv`) build on macOS (`make -C arrow-tools PG_CONFIG=... pg2arrow arrow2csv`, `compat/endian.h`); `tsv2arrow` / `vcf2arrow` were not tried.

## Tests

GPU-Cache: `sh test/gpuserv/locked_test.sh "$PWD" gpucache_dev.sh` runs `test/gpuserv/gpucache_tests.inc` alone (the full `scan_test.sh` sources it); `make -C src/gpuserv gcache_check && src/gpuserv/gcache_check` checks the cache store without a GPU.

```
GPUSERV_BIN=src/gpuserv/pgstrom-gpuserv sh test/gpuserv/service_test.sh     # service lifecycle (throwaway cluster)
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/scan_test.sh        # GPU vs CPU result comparison of GpuScan
sh test/gpuserv/locked_test.sh <tree> jsonb_test.sh                           # only the jsonb section of scan_test.sh (jsonb_tests.inc)
make -C test/port && test/port/port_test                                    # portability layer
```

The differential fuzzer (`test/gpuserv/fuzz.py`, standard library Python; random expression trees, aggregates, GROUP BY and joins over edge-value tables, each query run on the GPU path and with `pg_strom.enabled=off`; comparison rule and generator in the script header) runs through the same lock: `FUZZ_SEED=1 FUZZ_COUNT=2000 sh test/gpuserv/locked_test.sh "$PWD" fuzz_test.sh` (or `FUZZ_TIME=<seconds>`; `FUZZ_ARGS="--only N"` reproduces query N of a seed, `DBG_SQL=<file>` runs ad-hoc SQL against its tables). Failing queries are minimized and appended to `test/gpuserv/fuzz_corpus.txt`, which the next run replays first; bugs it found have regression tests in `test/gpuserv/fuzz_regress.inc` (part of `scan_test.sh`).

`sh test/gpuserv/locked_test.sh <tree> [scan_test.sh|service_test.sh]` installs a checkout and runs a test while holding a global lock (the installed `$libdir/pg_strom` is shared, so parallel worktrees must queue). Both shell tests build a throwaway cluster and need the module and service installed (`make install` above). `ALLOW_CPU=on` runs them on lavapipe (Linux). `PGSTROM_GPUSERV_TRACE=1` in the postmaster's environment makes the service log each session's xpucode and per-task timings; `PGSTROM_GPUSERV_NO_MAP=1` disables the zero-copy file mapping.

The upstream regression suite (PGXS `pg_regress` from `test/`, database `contrib_regression_pg_strom`, dataset revision in `test/Makefile`) has a Vulkan subset. `test/vulkan_schedule` (23 tests) must pass, `test/pending_schedule` holds what cannot yet (`conv_text_inet`; every other test file is listed there with a reason), and `test/vulkan_known_failures.txt` explains each difference from upstream's expected output. Run it against a throwaway cluster that keeps its 2 GB dataset between runs (about 3 minutes, plus about 20 seconds for the first dataset load; needs the module installed, so under the shared test lock):

```
sh test/gpuserv/locked_test.sh "$PWD" test/vulkan_installcheck.sh                        # vulkan_schedule: expect "All 23 tests passed"
TARGET=pending-installcheck sh test/gpuserv/locked_test.sh "$PWD" test/vulkan_installcheck.sh   # pending_schedule (fails by design)
sh test/gpuserv/locked_test.sh "$PWD" test/vulkan_regen_expected.sh                      # regenerate the PG17 alternate expected files
make -C test PG_CONFIG=... vulkan-installcheck                                           # the same against a running server (PGHOST/PGPORT)
```

`pg_regress` accepts `<test>.out` and `<test>_0..9.out`: `<test>_9.out` is this branch's (plan text differs because float8, numeric, text, jsonb and joins are not on the GPU yet); `vulkan_regen_expected.sh` writes it only after the output without its `EXPLAIN` blocks equals the same test run with `pg_strom.enabled` forced off (`make vulkan-oracle`), and reports a `ROW DIFF` otherwise. Only PostgreSQL 17 has `_9` files: on a host with 15, 16, 18 or 19 run the regen script with `PGBIN` pointing at that server (until then `vulkan-installcheck` means something on 17 only). The order of parallel-append children in `partition` can flip with the random ANALYZE sample (it did once on the earlier tree): if so, keep the flipped output as `partition_8.out` too. `arrow-tools` (`pg2arrow`, `arrow2csv`) now build on macOS and the Arrow tests need them. Run a single test with `make -C test ... installcheck REGRESS=--schedule=<file>` (a schedule file with one `test:` line) or invoke `pg_regress` directly with the options of `test/Makefile` (`--use-existing`).

## Architecture

1. **PostgreSQL backend side (planner/executor hooks).** `main.c` is the entry point. `gpu_scan.c`, `gpu_join.c` and `gpu_preagg.c` register CustomScan paths. `executor.c` is the shared backend runtime: it opens a session on the GPU service, sends tasks, receives results. `fallback.c` re-executes work on the CPU. `relscan.c` and `brin.c` read heap blocks (copied under the buffer lock, visibility checked on the CPU), `arrow_fdw.c` + `arrow_meta.cpp` / `parquet_read.cpp` are Arrow_Fdw. `gpu_client.c` is the backend's view of the service (`gpuserv_ready_accept`, the connection), `gpu_device.c` describes the selected device for the planner, `gpu_launcher.c` runs the probe and the service and exposes `pgstrom.gpu_devcaps()` / `gpu_service_state()` / `gpu_service_ping()`.

2. **GPU service process: `src/gpuserv/`** (own `Makefile`, libc + Vulkan only, no PostgreSQL headers). `pgstrom-gpuserv` is exec'd by the launcher because a forked process cannot use Metal's shader compiler. It talks to backends over a Unix socket with the unchanged `XpuCommand` protocol (`XpuCommandTag__OpenSession`, `XpuTaskExec`, `XpuTaskFinal` in `src/xpu_common.h`, which compiles standalone in C with `-DCUDA_MAXTHREADS_PER_BLOCK=512`). `wire.c` serves connections, `scan_prog.c` translates the planner's `kern_expression` trees into a program for the scan VM (`cpu_vm.c` is the same VM on the CPU, used for GROUP BY overflow rows), `scan_exec.c` uploads pages and runs the kernels (scan, aggregation, join; the join maps the backend's shared-memory inner hash stores), `scan_caps.[ch]` says which opcodes the VM implements (shared with the planner), `gpu_vk.[ch]` is the Vulkan layer, `devcaps.c` probes capabilities by running probe kernels and assigns a tier (none/base/desktop), `kernels/` are the probe kernels and `kernels_vm/` the VM (`scan_common.slang` shared evaluation, `scan.slang`, `agg.slang`, `gagg.slang` and `join.slang` entry points, variants by register count and aggregate count). SPIR-V is embedded into C arrays by `gen_kspv.sh`.

### SQL expression pushdown (`codegen.c`)

`codegen.c` turns PostgreSQL expression trees into an opcode stream (`kern_expression`). Device types and functions are registered with X-macros in `src/xpu_opcodes.h` (`TYPE_OPCODE`, `FUNC_OPCODE`). **What is actually pushed down is limited by `src/gpuserv/scan_caps.c`**: `codegen.c` refuses functions and expression nodes the Vulkan VM lacks, so they stay on the CPU. To make a function run on the GPU, add its classification to `scan_caps.c`, its translation to `scan_prog.c` and its implementation to `kernels_vm/scan.slang`; extend `test/gpuserv/scan_test.sh` with a differential query.

### SQL objects and upgrades

SQL-level definitions live in `src/sql/pg_strom--<ver>.sql` (base) and `pg_strom--<from>--<to>.sql` (upgrade scripts); a fresh install runs the whole chain, so **every C symbol an old script refers to must still exist in the module** (removed features keep a stub, see `gpu_cache.c`). New files must be added to `__STROM_SQL` in `src/Makefile`; the current version is in `src/pg_strom.control` and `Makefile.common`.

### Other directories

- `spike/`: the Apple Silicon feasibility experiments (see the appendix of the plan).
- `doc/`: user documentation for the Vulkan port (install, parameters, features, troubleshooting; Markdown). `man/*.md` is the MkDocs source of the generated HTML site in `docs/` and still describes the CUDA product (not hand-edited `docs/`; folding `doc/` into `man/` is open, A9).
- `deadcode/`: retired code kept for reference; not built.
- `fluentd/`, `files/`: fluentd plugin, systemd/spec/deb packaging templates. The templates were updated for Vulkan (vulkan-loader/libvulkan1, `pgstrom-gpuserv` in the file list, `WITH_GPUSERV=1`, slangc via `SLANGC`) but **never built** (A9 status in the plan). CI: `.github/workflows/vulkan-check.yml` (lavapipe on Linux, macOS build only, a manual self-hosted Apple Silicon job; not run yet); `build-check.yml` is the CUDA line's.
- `test/ssbm`, `test/tpch`, `test/pydatagen`: data generators for tests and benchmarks.

## Conventions

- Source files carry a PostgreSQL-License header and use PostgreSQL code style (tabs, `palloc`/`elog`, `PG_TRY`).
- Gate version-specific PostgreSQL behavior with `PG_VERSION_NUM` (compat helpers in `pg_compat.h`) instead of dropping support for older majors.
- Commit messages in history are short, lowercase-prefixed summaries such as `bugfix: <what was wrong>`; older branch commits are prefixed `apple-silicon:`, new ones `vulkan:`.
