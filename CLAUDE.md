# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PG-Strom is a PostgreSQL extension (C/C++ host code + GPU kernels) that offloads scan / join / group-by to GPUs, reads Apache Arrow / Parquet via a foreign data wrapper (Arrow_Fdw), and (on the CUDA line) supports GPU-Cache and GPU-Sort. Requires PostgreSQL v15+ (see `src/pg_strom.h`).

**Two lines of development.** `master` is the CUDA/NVIDIA product (Linux, CUDA toolkit; nothing builds or runs on a Mac). The `vulkan-gpu` branch (formerly `apple-silicon`) is a hard fork in which CUDA is gone: GPU work runs through Vulkan compute (Slang kernels compiled to SPIR-V) in a separate executable, and it is validated on Apple Silicon (MoltenVK) only. Arm64 Linux with lavapipe (a CPU Vulkan device) is used to check the host code. The CUDA sources (`cuda_*.cu`, `xpu_*.cu`, `gpu_service.c`, `gpu_cache.c`, `extra.c`, ...) are no longer in this branch; read them with `git show master:src/<file>` when porting a milestone. **The plan, status and findings are in `.claude/hw-acceleration.md`; read its "A2 status" first.**

Where the branch stands: GpuScan and GpuPreAgg without GROUP BY run on the GPU (heap tables); GpuPreAgg returns partial aggregate states that PostgreSQL's Agg node finalizes. GROUP BY (rest of A3), GpuJoin (A4), Arrow/Parquet ingest (A5), GPU-Sort/GPU-Cache (A7) and jsonb (A8) are planned; `gpu_join.c` is in the tree but not registered, and `gpu_cache.c` / `parquet_cache.c` are stubs.

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
- `arrow-tools/` (`pg2arrow`, ...) does not build on macOS yet.

## Tests

```
GPUSERV_BIN=src/gpuserv/pgstrom-gpuserv sh test/gpuserv/service_test.sh     # service lifecycle (throwaway cluster)
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/scan_test.sh        # GPU vs CPU result comparison of GpuScan
make -C test/port && test/port/port_test                                    # portability layer
```

Both shell tests build a throwaway cluster and need the module and service installed (`make install` above). `ALLOW_CPU=on` runs them on lavapipe (Linux). `PGSTROM_GPUSERV_TRACE=1` in the postmaster's environment makes the service log each session's xpucode and per-task timings; `PGSTROM_GPUSERV_NO_MAP=1` disables the zero-copy file mapping.

The upstream regression suite is PGXS `installcheck` from `test/` (`make -C test PG_CONFIG=... installcheck`; schedule `test/parallel_schedule`, expected output per PG major version in `test/<15..19>/expected/`, database `contrib_regression_pg_strom`, dataset revision in `test/Makefile`). **On this branch most of it does not pass yet**: the expected files assert CUDA-era plan shapes (for example float8 expressions on the GPU) and cover join, aggregate and sort. The result rows of the scan-only tests match; the `EXPLAIN` text does not. Run a single test by invoking `pg_regress` directly with the options of `test/Makefile` (`--use-existing`) and a schedule file.

## Architecture

1. **PostgreSQL backend side (planner/executor hooks).** `main.c` is the entry point. `gpu_scan.c` (and `gpu_join.c`, `gpu_preagg.c`, not registered yet) register CustomScan paths. `executor.c` is the shared backend runtime: it opens a session on the GPU service, sends tasks, receives results. `fallback.c` re-executes work on the CPU. `relscan.c` and `brin.c` read heap blocks (copied under the buffer lock, visibility checked on the CPU), `arrow_fdw.c` + `arrow_meta.cpp` / `parquet_read.cpp` are Arrow_Fdw. `gpu_client.c` is the backend's view of the service (`gpuserv_ready_accept`, the connection), `gpu_device.c` describes the selected device for the planner, `gpu_launcher.c` runs the probe and the service and exposes `pgstrom.gpu_devcaps()` / `gpu_service_state()` / `gpu_service_ping()`.

2. **GPU service process: `src/gpuserv/`** (own `Makefile`, libc + Vulkan only, no PostgreSQL headers). `pgstrom-gpuserv` is exec'd by the launcher because a forked process cannot use Metal's shader compiler. It talks to backends over a Unix socket with the unchanged `XpuCommand` protocol (`XpuCommandTag__OpenSession`, `XpuTaskExec`, `XpuTaskFinal` in `src/xpu_common.h`, which compiles standalone in C with `-DCUDA_MAXTHREADS_PER_BLOCK=512`). `wire.c` serves connections, `scan_prog.c` translates the planner's `kern_expression` trees into a program for the scan VM, `scan_exec.c` uploads pages and runs the kernels (scan and aggregation), `scan_caps.[ch]` says which opcodes the VM implements (shared with the planner), `gpu_vk.[ch]` is the Vulkan layer, `devcaps.c` probes capabilities by running probe kernels and assigns a tier (none/base/desktop), `kernels/` are the probe kernels and `kernels_vm/` the VM (`scan_common.slang` shared evaluation, `scan.slang` and `agg.slang` entry points, variants by register count and aggregate count). SPIR-V is embedded into C arrays by `gen_kspv.sh`.

### SQL expression pushdown (`codegen.c`)

`codegen.c` turns PostgreSQL expression trees into an opcode stream (`kern_expression`). Device types and functions are registered with X-macros in `src/xpu_opcodes.h` (`TYPE_OPCODE`, `FUNC_OPCODE`). **What is actually pushed down is limited by `src/gpuserv/scan_caps.c`**: `codegen.c` refuses functions and expression nodes the Vulkan VM lacks, so they stay on the CPU. To make a function run on the GPU, add its classification to `scan_caps.c`, its translation to `scan_prog.c` and its implementation to `kernels_vm/scan.slang`; extend `test/gpuserv/scan_test.sh` with a differential query.

### SQL objects and upgrades

SQL-level definitions live in `src/sql/pg_strom--<ver>.sql` (base) and `pg_strom--<from>--<to>.sql` (upgrade scripts); a fresh install runs the whole chain, so **every C symbol an old script refers to must still exist in the module** (removed features keep a stub, see `gpu_cache.c`). New files must be added to `__STROM_SQL` in `src/Makefile`; the current version is in `src/pg_strom.control` and `Makefile.common`.

### Other directories

- `spike/`: the Apple Silicon feasibility experiments (see the appendix of the plan).
- `docs/`: generated HTML site (not hand-edited source); not yet updated for the Vulkan port (milestone A9).
- `deadcode/`: retired code kept for reference; not built.
- `fluentd/`, `files/`: fluentd plugin, systemd/spec/deb packaging templates (spec no longer mentions CUDA; full packaging rewrite is A9).
- `test/ssbm`, `test/tpch`, `test/pydatagen`: data generators for tests and benchmarks.

## Conventions

- Source files carry a PostgreSQL-License header and use PostgreSQL code style (tabs, `palloc`/`elog`, `PG_TRY`).
- Gate version-specific PostgreSQL behavior with `PG_VERSION_NUM` (compat helpers in `pg_compat.h`) instead of dropping support for older majors.
- Commit messages in history are short, lowercase-prefixed summaries such as `bugfix: <what was wrong>`; older branch commits are prefixed `apple-silicon:`, new ones `vulkan:`.
