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
| Works | GpuScan on heap tables: bool, int1/2/4/8 and float4 quals and projections run on the GPU; everything else stays on the CPU and returns the same rows. UPDATE and DELETE scans. Query parameters. Aggregation (count, sum, avg, min, max of integers), with and without GROUP BY on up to two integer keys. Inner hash joins on integer keys with one inner relation (GpuJoin). |
| Not yet | Fast GROUP BY with many groups (the GPU only wins for roughly 16 groups or fewer today), outer / nest-loop / multi-relation GpuJoin (inner hash joins on integer keys with one inner relation work), GPU-Sort, GPU-Cache, Arrow/Parquet on the GPU, jsonb, float8 and numeric on the GPU, SELECT INTO direct. GPU-Direct Storage and PostGIS are dropped. |
| Speed | On a 16.8M-row table: a selective scan is about 1.4x and an aggregate 2.6-4x faster than 8-process PostgreSQL on the same Mac; wide results are slower (see the plan). |

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
with no GPU offload. `CLAUDE.md` describes the code layout and test workflow;
`.claude/hw-acceleration.md` is the plan, the measured findings and the milestone
status. The documentation site linked above describes the CUDA product and has
not been updated for this branch.
