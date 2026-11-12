# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PG-Strom is a PostgreSQL extension (C/C++ host code + GPU kernels) that offloads scan / join / group-by to GPUs, reads Apache Arrow / Parquet via a foreign data wrapper (Arrow_Fdw), and (on the CUDA line) supports GPU-Cache and GPU-Sort. Requires PostgreSQL v15+ (see `src/pg_strom.h`).

**Two lines of development.** `master` is the CUDA/NVIDIA product (Linux, CUDA toolkit; nothing builds or runs on a Mac). The `vulkan-gpu` branch (formerly `apple-silicon`) is a hard fork in which CUDA is gone: GPU work runs through Vulkan compute (Slang kernels compiled to SPIR-V) in a separate executable, and it is validated on Apple Silicon (MoltenVK) only. Arm64 Linux with lavapipe (a CPU Vulkan device) is used to check the host code. The CUDA sources (`cuda_*.cu`, `xpu_*.cu`, `gpu_service.c`, `gpu_cache.c`, `extra.c`, ...) are no longer in this branch; read them with `git show master:src/<file>` when porting a milestone. **The plan, status and findings are in `.claude/hw-acceleration.md`; read its "A2 status" first.**

Where the branch stands: GpuScan, GpuJoin (hash and nest-loop), GpuPreAgg (with/without GROUP BY, many groups), Arrow/Parquet, GPU-Cache, GPU-Sort, jsonb, numeric, float8 (df64 prefilter), PostGIS predicates/measures/distances run on the GPU through Vulkan; any refused or failed GPU node continues on the CPU (A14). The full feature-by-feature list, limits and rationale are in `.claude/branch-status.md`; the plan's status sections (`.claude/hw-acceleration.md`, has a TOC) are authoritative.

## Build / install (vulkan-gpu branch)

All builds go through PGXS against a specific `pg_config`; always pass `PG_CONFIG`:

```
make -C src PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config install   # module + pgstrom-gpuserv
make -C src/gpuserv                                                          # the service alone (needs slangc and the Vulkan loader)
```

- `WITH_GPUSERV=auto|1|0` (default auto): build the service when `slangc` and the Vulkan headers are found; `1` makes a missing prerequisite an error.
- The service Makefile writes `.d` dependency files, so a changed header rebuilds what includes it. The join kernel families (join, jagg, jgagg, Arrow twins) also have geometry tiers `_gpip` / `_gdw` / `_gwalk` (R15: `scan_join_tier()`, `-DVM_GEOM_MASK`; 3-7 MB of SPIR-V instead of the full tier's 9.8 MB; a program picks the smallest that has its geometry groups). Every VM kernel is built in three tiers: the full one, `<name>_nogeom` (`-DVM_NO_GEOM`: numeric, jsonb and float8 but no geometry) and `<name>_lite` (`-DVM_NO_NUM/JSON/F8/GEOM`: none of the optional groups). `scan_prog.feat` says which groups a program uses and `scan_feat_tier()` (`scan_caps.h`) picks the tier: no group -> lite, geometry -> full, anything else -> nogeom; the pipeline arrays of `scan_engine` are indexed by register variant + variants-per-tier * tier. The tiers compile in a fraction of the time (the join kernel is 9.6 MB of SPIR-V full, 2.98 MB nogeom, 0.9 MB lite; Metal needs minutes for the full one, once after every change of `kernels_vm/`). A new geometry opcode must be classified in `scan_vm_op_feat()` (`scan_caps.c`, the range 130-199 is the geometry group) or `src/gpuserv/feat_check` fails (`spec_compile_check.sh` runs it).
- `PGSTROM_DEBUG=1`: `-g -O0 -DPGSTROM_DEBUG_BUILD`. `PGSTROM_FLAGS_CUSTOM`: extra host flags.
- `libarrow` and `libparquet` are required (`brew install apache-arrow pkgconf` on macOS); `libnuma` is optional. Host C++ uses C++20. GNU make 3.81 and the macOS `sh` are supported (no `echo -n`, `\#` inside `$(shell)`).
- The extension must be loaded at postmaster start: `shared_preload_libraries = 'pg_strom'`. The postmaster probes the Vulkan devices by running `pgstrom-gpuserv --probe`, and a launcher background worker runs and restarts the service (`pg_strom.gpuserv_path`, `pg_strom.gpu_allow_cpu_device`, `pg_strom.gpu_visible_devices`).
- `arrow-tools/` (`pg2arrow`, `arrow2csv`) build on macOS (`make -C arrow-tools PG_CONFIG=... pg2arrow arrow2csv`, `compat/endian.h`); `tsv2arrow` / `vcf2arrow` were not tried.

## Tests

Full command reference, lock/queue behaviour, tooling and the regression suite: `.claude/testing.md` (read it before running tests; also the `vulkan-verify` skill). Essentials:

```
sh test/gpuserv/locked_test.sh "$PWD" <script>.sh      # installs, runs under the global GPU lock
sh test/gpuserv/affected.sh                           # which dev scripts the changed files need
sh test/gpuserv/locked_test.sh "$PWD" test/vulkan_installcheck.sh   # expect "All 23 tests passed"
```

Never run two GPU programs at once; keep the local drive above 3 GB free; tests run at `nice -n 10` with 3 jobs.


## Architecture

1. **PostgreSQL backend side (planner/executor hooks).** `main.c` is the entry point. `gpu_scan.c`, `gpu_join.c` and `gpu_preagg.c` register CustomScan paths. `executor.c` is the shared backend runtime: it opens a session on the GPU service, sends tasks, receives results. `fallback.c` re-executes work on the CPU. `relscan.c` and `brin.c` read heap blocks (copied under the buffer lock, visibility checked on the CPU), `arrow_fdw.c` + `arrow_meta.cpp` / `parquet_read.cpp` are Arrow_Fdw. `gpu_client.c` is the backend's view of the service (`gpuserv_ready_accept`, the connection), `gpu_device.c` describes the selected device for the planner, `gpu_launcher.c` runs the probe and the service and exposes `pgstrom.gpu_devcaps()` / `gpu_service_state()` / `gpu_service_ping()`.

2. **GPU service process: `src/gpuserv/`** (own `Makefile`, libc + Vulkan only, no PostgreSQL headers). `pgstrom-gpuserv` is exec'd by the launcher because a forked process cannot use Metal's shader compiler. It talks to backends over a Unix socket with the unchanged `XpuCommand` protocol (`XpuCommandTag__OpenSession`, `XpuTaskExec`, `XpuTaskFinal` in `src/xpu_common.h`, which compiles standalone in C with `-DCUDA_MAXTHREADS_PER_BLOCK=512`). `wire.c` serves connections, `scan_prog.c` translates the planner's `kern_expression` trees into a program for the scan VM (`cpu_vm.c` is the same VM on the CPU, used for GROUP BY overflow rows), `scan_exec.c` uploads pages and runs the kernels (scan, aggregation, join; the join maps the backend's shared-memory inner hash stores), `scan_caps.[ch]` says which opcodes the VM implements (shared with the planner), `gpu_vk.[ch]` is the Vulkan layer, `devcaps.c` probes capabilities by running probe kernels and assigns a tier (none/base/desktop), `specialize.c` turns a program into Slang source (registers as constants, every instruction a call of `vm_step()` with constant operands, so the generated kernel runs the interpreter's own helpers; the column loads call `deform_col()` / `arrow_load_col()` the same way; a program with an opcode it does not list stays on the VM), `spec_mgr.c` compiles it in a worker thread (`slangc` subprocess, the kernel sources embedded in the binary by `gen_ksrc.sh`, SPIR-V cached by a hash of the source in `$XDG_CACHE_HOME/pgstrom`), creates the pipeline there and hands it to `spec_pipe()`, which the task functions of `scan_exec.c` / `arrow_exec.c` call before each dispatch (`spec_dispatch()` in `spec_verify.c`, which in `check` mode also runs the VM kernel and compares), `kernels/` are the probe kernels and `kernels_vm/` the VM (`scan_common.slang` shared evaluation, `scan.slang`, `agg.slang`, `gagg.slang` and `join.slang` entry points, variants by register count and aggregate count). SPIR-V is embedded into C arrays by `gen_kspv.sh`.

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
