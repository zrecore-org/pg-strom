PG-Strom
========
PG-Strom is an extension for PostgreSQL database.
It is designed to accelerate mostly batch and analytics workloads with
utilization of GPU and NVME-SSD, and Apache Arrow columnar.

For more details, see the documentation below.
http://heterodb.github.io/pg-strom/

Software is fully open source, distributed under PostgreSQL License.

The `vulkan-gpu` branch: PG-Strom on Vulkan
-------------------------------------------
This branch is a hard fork of the CUDA product (which stays on `master`) that
runs the GPU work through **Vulkan compute** instead of CUDA, so that it does not
depend on NVIDIA hardware. Kernels are written in Slang and compiled to SPIR-V;
the GPU service is a separate executable (`pgstrom-gpuserv`, in `src/gpuserv/`)
that the extension starts and restarts, so a driver crash does not take down
PostgreSQL.

**Status (experimental).**

| | |
|---|---|
| Validated on | Apple Silicon (M1 Pro, MoltenVK, macOS), PostgreSQL 17. Arm64 Linux with lavapipe (a CPU Vulkan device) checks the host code. |
| Designed for, **not tested** | NVIDIA, AMD and Intel GPUs on Linux and Windows. Their behavior and speed are unknown. |
| Works | GpuScan on heap tables and on Arrow / Parquet foreign tables: bool, int1/2/4/8, float4, float8 (computed in integer arithmetic, bit-identical to the CPU), date/time/timestamp, numeric (values that fit 64 bits, with division, rounding and `sum`/`avg`; others are re-checked on the CPU) and text (`=`, `LIKE` patterns, C-collation ordering) quals and projections run on the GPU; everything else stays on the CPU and returns the same rows. UPDATE and DELETE scans. Query parameters. Aggregation (count, sum, avg, min, max of integers, and of float8 without GROUP BY) with and without GROUP BY on up to two integer/date-time keys, also directly over a join, also over Arrow/Parquet sources. Hash joins (GpuJoin): inner, LEFT, RIGHT, FULL, semi and anti, on integer/date-time keys, up to four inner relations. GPU-Cache: a table with the `pgstrom.gpucache_sync_trigger()` row trigger is kept resident in the GPU service and scans, aggregates and joins of it read the cache (latest committed state plus the session's own changes). |
| Not yet | Fast GROUP BY with many groups (the GPU wins up to about 100 groups; `pg_strom.gpupreagg_max_groups` = 128), nest-loop joins and joins on text/numeric keys, Decimal columns of Arrow/Parquet files, numeric inside join programs, float8, numeric `min`/`max`, numeric in joins, GPU-Sort of scan/join results (GroupBy results can be sorted on the GPU: experimental, off by default), jsonb values returned from the GPU (jsonb *quals* run on the GPU: `->`, `->>`, `?`, `@>`, casts to numbers), SELECT INTO direct. GPU-Direct Storage is dropped; PostGIS is partial (the bounding-box operators and point-in-polygon predicates run on the GPU). See `doc/features.md`. |
| Speed | On a 16.8M-row table (M1 Pro, against 8-process PostgreSQL on the same Mac): a selective scan about 3x, an aggregate 3.5-5x, GROUP BY with up to about 100 dense integer groups 4-5.7x, an aggregate over a join 4.8x, GROUP BY over a join 3.5x - with the per-query generated kernels (`slangc` needed at run time, else the interpreter: about half of that); GROUP BY with 1,000 groups and wide results are slower (see the plan, "A6"). |

**Build and try it** (macOS shown; needs PostgreSQL 15 or later with its
development files, `slangc` from the Slang project, the Vulkan loader and
MoltenVK, and Apache Arrow with Parquet):

```
brew install postgresql@17 apache-arrow pkgconf          # plus slang and Vulkan loader/MoltenVK
make -C src PG_CONFIG=$(brew --prefix postgresql@17)/bin/pg_config install
# postgresql.conf:  shared_preload_libraries = 'pg_strom'
# SQL:              CREATE EXTENSION pg_strom;  SELECT * FROM pgstrom.gpu_devcaps();
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/scan_test.sh   # GPU vs CPU result comparison
```

Without a usable Vulkan device the extension loads and PostgreSQL runs normally,
with no GPU offload.

**User documentation for this branch is in [`doc/`](doc/)**: [install](doc/install.md)
(platforms, prerequisites, build, first checks), [parameters](doc/parameters.md) (every
configuration parameter, with what is new, renamed or removed since the CUDA product),
[features](doc/features.md) (what runs on the GPU and what does not; the table above can
lag behind it) and [troubleshooting](doc/troubleshooting.md) (service state, probe logs,
`EXPLAIN` labels). `CLAUDE.md` describes the code layout and test workflow;
`.claude/hw-acceleration.md` is the plan, the measured findings and the milestone
status. The documentation site linked above describes the CUDA product and has
not been updated for this branch.
