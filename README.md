PG-Strom
========
[![Vulkan Check](https://github.com/zrecore-org/pg-strom/actions/workflows/vulkan-check.yml/badge.svg?branch=vulkan-gpu)](https://github.com/zrecore-org/pg-strom/actions/workflows/vulkan-check.yml?query=branch%3Avulkan-gpu)

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
| Works | GpuScan on heap tables and on Arrow / Parquet foreign tables: bool, int1/2/4/8, float4, float8 (computed in integer arithmetic, bit-identical to the CPU), date/time/timestamp, numeric (values that fit 64 bits, with division, rounding and `sum`/`avg`; others are re-checked on the CPU) and text (`=`, `LIKE` patterns, C-collation ordering) quals and projections run on the GPU; everything else stays on the CPU and returns the same rows. UPDATE and DELETE scans. Query parameters. Aggregation (count, sum, avg, min, max of integers, float4 and float8, min/max of numeric(p,s), bool_and/bool_or, variance/stddev of integers and floats) with and without GROUP BY on up to four integer/date-time/bool/float keys over a join (three or four over a scan too), up to two over Arrow/Parquet sources, also directly over a join, also over Arrow/Parquet sources. Hash joins (GpuJoin): inner, LEFT, RIGHT, FULL, semi and anti, on integer/date-time, text/varchar/bpchar, uuid, bool, float and numeric keys, up to four inner relations. GPU-Cache: a table with the `pgstrom.gpucache_sync_trigger()` row trigger is kept resident in the GPU service and scans, aggregates and joins of it read the cache (latest committed state plus the session's own changes). A node whose GPU session is refused or dies (a program the service cannot translate, a lost device, a service that restarts) continues on the CPU and returns the CPU plan's rows instead of failing the query (`pg_strom.gpu_error_fallback`; not for GPU-Sort and pinned buffers). |
| Not yet | Fast GROUP BY with many groups over a join (scans scale to a million groups; a join wins up to about 100 groups, `pg_strom.gpupreagg_max_groups` = 128), nest-loop joins with LATERAL or an unbounded pair count, GROUP BY on a text/uuid/numeric key over a join (scans take them), GROUP BY on more than four keys, variance/stddev of numeric, `regr_count`, `bit_and`, `min`/`max` of text, Decimal columns of Arrow/Parquet files, `sum`/`avg`/`min`/`max` of unconstrained numeric, GPU-Sort of scan/join results (GroupBy results can be sorted on the GPU: experimental, off by default), jsonb values returned from the GPU (jsonb *quals* run on the GPU: `->`, `->>`, `?`, `@>`, casts to numbers), SELECT INTO direct. GPU-Direct Storage is dropped; PostGIS is partial (the bounding-box operators, point-in-polygon predicates and line/polygon-against-line/polygon `ST_Intersects`/`ST_Contains`/`ST_Within`/`ST_Covers`/`ST_CoveredBy` in scan quals (touching boundaries are re-checked on the CPU), distances, lengths, `ST_Area`, the counts `ST_NPoints`/`ST_NRings`/..., `ST_X`/`ST_Y`/`ST_SRID`/`ST_IsEmpty` run on the GPU, on heap and GPU-Cache tables only: Arrow_Fdw has no geometry column; `ST_GeometryType`, `ST_StartPoint`/`ST_EndPoint` stay on the CPU). See `doc/features.md`. |
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

Credits and licensing
---------------------
PG-Strom is distributed under the PostgreSQL License (see [`LICENSE`](LICENSE)):
Copyright (c) 2012-2024 KaiGai Kohei and (c) 2017-2024 HeteroDB, Inc. The `vulkan-gpu`
branch is a derivative of their work (the CUDA product on `master`, <https://github.com/heterodb/pg-strom>):
the planner integration, the expression code generator (`codegen.c`), the Arrow_Fdw
foreign data wrapper, the `XpuCommand` protocol and many of the SQL definitions come from
it. The Vulkan service, the Slang kernels and the other changes on this branch are
released under the same license.

Third-party software the branch builds on or tests against. None of it is copied into
the repository unless stated; each project keeps its own license.

| Project | Used for | License |
|---|---|---|
| [PostgreSQL](https://www.postgresql.org/) | the database and its extension API | PostgreSQL License |
| [Apache Arrow / Parquet](https://arrow.apache.org/) (`libarrow`, `libparquet`) | Arrow_Fdw, Parquet scans, `arrow-tools` | Apache-2.0 |
| [Slang](https://shader-slang.org/) (`slangc`) | the GPU kernels are written in Slang and compiled to SPIR-V | Apache-2.0 with LLVM exceptions |
| [Vulkan](https://www.vulkan.org/) headers and loader (Khronos) | GPU access | Apache-2.0 |
| [MoltenVK](https://github.com/KhronosGroup/MoltenVK) | Vulkan on Apple Silicon (Metal) | Apache-2.0 |
| [Mesa lavapipe](https://www.mesa3d.org/) | CPU Vulkan device used in CI | MIT |
| [PostGIS](https://postgis.net/) | the extension whose `geometry` type and functions the GPU path accelerates; it is called at planning time to check results (not linked) | GPL-2.0-or-later |
| J. R. Shewchuk, [*Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*](https://www.cs.cmu.edu/~quake/robust.html), Discrete & Computational Geometry 18 (1997) | the error bound of the orientation filter for point-in-polygon | public domain algorithm |
| [IEEE 754-2008](https://doi.org/10.1109/IEEESTD.2008.4610935); T. J. Dekker, [*A floating-point technique for extending the available precision*](https://doi.org/10.1007/BF01397083), Numerische Mathematik 18 (1971); D. Knuth, *The Art of Computer Programming* vol. 2, section 4.2.2 (error-free transformations) | the software binary64 (`f64_soft.inc`) and the double-float prefilter (`df64.inc`) | published algorithms |

**PostGIS note (decided 2026-11-11: accept GPL for that file).** The GPU `ST_Area`
(`src/gpuserv/kernels_vm/area_ring.inc`) reproduces the algorithm of PostGIS 3.7's
`ptarray_signed_area` (`liblwgeom/ptarray.c`) so that results are bit for bit the same. PostGIS
is GPL-2.0-or-later, so that one file carries `SPDX-License-Identifier: GPL-2.0-or-later` and
the license text is in [`LICENSES/GPL-2.0.txt`](LICENSES/GPL-2.0.txt). Everything else stays under the PostgreSQL
License. `cpu_vm.c` and the Slang kernels include that file, so a **binary** of
`pgstrom-gpuserv` (and its compiled kernels) is a combined work that must be distributed
under the GPL terms (source offer included); the source tree as a whole remains
PostgreSQL-licensed file by file. Other geometry kernels follow PostGIS's documented
behaviour without copying its code.
