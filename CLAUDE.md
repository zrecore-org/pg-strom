# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PG-Strom is a PostgreSQL extension (C/C++ host code + GPU device code) that offloads scan / join / group-by to GPUs, reads Apache Arrow / Parquet via a foreign data wrapper (Arrow_Fdw), and supports GPU-Cache and GPU-Sort. Requires PostgreSQL v15+ (see `src/pg_strom.h`).

**Two lines of development.** `master` is the CUDA/NVIDIA product (Linux, CUDA toolkit; nothing builds or runs on a Mac). The `apple-silicon` branch is a hard fork that replaces CUDA with Vulkan compute (Slang kernels, SPIR-V) and is validated on Apple Silicon (MoltenVK) only; Linux arm64 with lavapipe (a CPU Vulkan device) is used for host-code checks. The plan, status and findings are in `.claude/hw-acceleration.md`; read its "A2 status" first. On the branch, CUDA code is still present until milestone A2.5 removes it; the build uses `src/nocuda/` stubs when no CUDA toolkit is found (`-DPGSTROM_NO_CUDA`). The Vulkan build offers GpuScan only so far (GpuPreAgg, GpuJoin, GPU-Cache and SELECT INTO direct are compiled out).

## Build / install

All builds go through PGXS against a specific `pg_config`; always pass `PG_CONFIG`:

```
PG_CONFIG=/path/to/pg_config make -C src clean
PG_CONFIG=/path/to/pg_config make -C src -j$(nproc) install   # what CI runs
```

Useful switches (see `src/Makefile`, `Makefile.common`, `src/Makefile.cuda`):
- `PGSTROM_DEBUG=1`: `-g -O0 -DPGSTROM_DEBUG_BUILD`
- `WITH_POSTGIS=1`: also builds `gist.c` and `xpu_postgis.cu` (slated for removal in v6.3)
- `WITH_FATBIN=1` and `make -C src fatbin`: prebuild the GPU fatbin (debug only)
- `PGSTROM_FLAGS_CUSTOM`, `NVCC_FLAGS_CUSTOM`: extra host/nvcc flags
- Packaging (RHEL): `make rpm` from the top directory; `make deb` for Debian. `VERSION`/`RELEASE` are set in `Makefile.common`.

`libarrow`, `libparquet`, `libnuma` are detected by `pkgconf`; their features compile out if missing (`HAS_LIBARROW`, `HAS_PARQUET`, `HAS_LIBNUMA`). Host C++ uses C++20.

The extension must be loaded at postmaster start: `shared_preload_libraries = '$libdir/pg_strom'`.

`arrow-tools/` holds standalone tools (`pg2arrow`, `arrow2csv`, `tsv2arrow`, `vcf2arrow`) built separately and used by the tests.

### apple-silicon branch: build and test

```
make -C src PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config install   # module + pgstrom-gpuserv
make -C src/gpuserv                                                          # the service alone (needs slangc and the Vulkan loader)
GPUSERV_BIN=src/gpuserv/pgstrom-gpuserv sh test/gpuserv/service_test.sh     # service lifecycle (throwaway cluster)
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/scan_test.sh        # GPU vs CPU result comparison of GpuScan
make -C test/port && test/port/port_test                                    # portability layer
```

`PGSTROM_GPUSERV_TRACE=1` in the postmaster's environment makes the service log each session's xpucode and per-task timings; `PGSTROM_GPUSERV_NO_MAP=1` disables the zero-copy file mapping. `ALLOW_CPU=on` lets the tests use lavapipe. GNU make 3.81 and the macOS `sh` are supported (no `echo -n`, `\#` in `$(shell)`).

## Tests

Regression tests are PGXS `installcheck`, run from `test/` against an already-installed, running PostgreSQL with a GPU:

```
make -C test PG_CONFIG=/path/to/pg_config installcheck
```

- Schedule is `test/parallel_schedule`; SQL in `test/sql/`, `test/input/*.source` is templated.
- Expected output is **per PG major version**: `test/<15|16|17|18|19>/expected/*.out`. A behavior change usually means updating the `.out` file for every supported version.
- Database is `contrib_regression_pg_strom`. `REGRESS_REVISION` in `test/Makefile` (and `pgstrom.regression_testdb_revision()`) controls whether the test dataset is rebuilt. Bump it when the dataset definition changes.
- Run one test: add it to a custom schedule, or invoke pg_regress directly with the same options as `test/Makefile` (`--use-existing --dbname=contrib_regression_pg_strom`) and the test name as the argument; the Makefile has no single-test target.
- CI (`.github/workflows/build-check.yml`, self-hosted runner) runs `test/automation_setup/run_ci_tests.sh --pgver <x.y>`; tests expected to fail are listed in `test/automation_setup/known_failures.txt`.
- There is no separate lint step.

## Architecture

Split into three cooperating parts that span many files:

1. **PostgreSQL backend side (planner/executor hooks).** `main.c` is the entry point (GUCs, hooks, shmem). `gpu_scan.c`, `gpu_join.c`, `gpu_preagg.c` register CustomScan paths (GpuScan / GpuJoin / GpuPreAgg) that replace standard plan nodes. `executor.c` is the shared backend runtime for these: it manages `XpuConnection`s, sends tasks, and receives results. `fallback.c` re-executes work on the CPU when a GPU kernel cannot complete (e.g. unsupported data). `relscan.c`, `brin.c`, `select_into.c` (direct `SELECT INTO` writes), `arrow_fdw.c` + `arrow_meta.cpp` (Arrow FDW), `parquet_read.cpp` / `parquet_cache.c` (Parquet) and `gpu_cache.c` (GPU-resident table cache) supply data to or alongside those nodes.

2. **GPU service process.** `gpu_service.c` is a separate background worker that owns the CUDA contexts, GPU memory, and worker threads. Backends talk to it over a socket using the `XpuCommand` protocol (`XpuCommandTag__OpenSession`, `XpuTaskExec`, `XpuTaskFinal`, ... in `src/xpu_common.h`). Backends never call CUDA kernels directly. `gpu_device.c` handles device enumeration/attributes (`gpu_devattrs.h` is generated from `cuda.h`).

3. **Device code (`*.cu`).** `cuda_gpuscan.cu`, `cuda_gpujoin.cu`, `cuda_gpupreagg.cu`, `cuda_gpusort.cu` are the kernels. `xpu_common.cu` plus `xpu_{basetype,numeric,timelib,textlib,misclib,jsonlib,postgis}.cu` implement device-side PostgreSQL types and functions. These are compiled to a fatbin at build/install time (file name includes the nvcc version and an md5 of the sources) and loaded by the GPU service.

### SQL expression pushdown (`codegen.c`)

`codegen.c` turns PostgreSQL expression trees into a device-side opcode stream (`kern_expression`) that the GPU interprets. Device types and functions are registered with X-macros in `src/xpu_opcodes.h` (`TYPE_OPCODE`, `FUNC_OPCODE`), which is included by both the host and the device code. To add a device type or function, expect to touch `xpu_opcodes.h`, the matching `xpu_*.cu` / `.h` implementation, and the `devtype`/`devfunc` tables in `codegen.c`. Host and device share `xpu_common.h` and `cuda_common.h`, so struct layout changes affect both sides.

### SQL objects and upgrades

SQL-level definitions live in `src/sql/pg_strom--<ver>.sql` (base) and `pg_strom--<from>--<to>.sql` (upgrade scripts). New files must be added to `__STROM_SQL` in `src/Makefile`, and the current version is in `src/pg_strom.control` and `Makefile.common`.

### Other directories

- `docs/`: generated HTML site (not hand-edited source) with install and reference documentation.
- `deadcode/`: retired code kept for reference; not built.
- `fluentd/`, `files/`: fluentd plugin, systemd/spec/deb packaging templates.
- `test/ssbm`, `test/tpch`, `test/pydatagen`: data generators for tests and benchmarks.

## Conventions

- Source files carry a PostgreSQL-License header and use PostgreSQL code style (tabs, `palloc`/`elog`, `PG_TRY`).
- Gate version-specific PostgreSQL behavior with `PG_VERSION_NUM` (compat helpers in `pg_compat.h`) instead of dropping support for older majors.
- Commit messages in history are short, lowercase-prefixed summaries such as `bugfix: <what was wrong>`.
