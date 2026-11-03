# PG-Strom on Vulkan: Apple Silicon first, CUDA removed (branch `vulkan-gpu`)

> The branch was created as `apple-silicon` and renamed `vulkan-gpu` after A2 (it is the Vulkan port, validated on Apple Silicon only). Older text below keeps the old name where it describes history.

## Context and decisions

The goal changed from "add Vulkan next to CUDA" to "no longer depend on NVIDIA; port the acceleration to Vulkan for cross-platform GPU use". Your decisions, and how I read them:

| Decision | Reading in this plan |
|---|---|
| Clean replacement: remove CUDA, no CUDA fallback or oracle | Applies to the new branch only. The oracle for tests is PostgreSQL with `pg_strom.enabled=off`. |
| Drop GPUDirect Storage and `heterodb-extra` | Dropped. Storage is read through the CPU. |
| Work lives on a new git branch `vulkan-gpu` (created as `apple-silicon`) | Created from `master` when implementation starts. **`master` stays the CUDA line (`cuda-legacy`) and is not touched by this work.** |
| Maintain `cuda-legacy` releases | The CUDA line continues on `master`; you do not test it (see next row). It is never ported or modified here. |
| "I do not want to test NVIDIA / AMD / Intel hardware" | **Apple Silicon is the only validated platform.** The code stays Vulkan-portable by construction, but other vendors are labelled *unvalidated* in docs and `pgstrom.gpu_device_info`. The earlier request for Linux NVIDIA/AMD/Intel as CI platforms is reduced to "designed for, not tested". |
| Scan up to 1.5x and aggregation up to 2x slower than CUDA is acceptable | Recorded as the acceptance criterion **if anyone later measures on NVIDIA**. It cannot be a gate on this branch, because it needs NVIDIA hardware. The gate here is PostgreSQL's own parallel scan on the same Mac. |
| First Vulkan release scope: core (scan, aggregate, join), Arrow_Fdw and Parquet, jsonb, GpuSort/window and GPU-Cache | All four are in scope, sequenced by difficulty. jsonb comes last and is the most uncertain. |

**Consequences of these choices (stated plainly)**
- **This is a permanent hard fork** of the CUDA product for the `vulkan-gpu` branch. Upstream CUDA changes can still be merged into `master`, but not into the branch once CUDA files are deleted.
- **Nothing in this plan proves performance or correctness on NVIDIA, AMD or Intel.** That risk is accepted and written down, not mitigated by hardware.
- **PostGIS is dropped** (fp64 and about 6k lines; Apple has no fp64).

## Corrections to earlier conclusions (verified in code)

1. **PG-Strom never lets the GPU read live `shared_buffers`.** `src/relscan.c:439-500` reads each page with `ReadBufferExtended`, takes `BUFFER_LOCK_SHARE`, copies the 8 KB page into `xcmd_buf`, runs the visibility check on the CPU (invisible tuples are marked unused in the copy), and unlocks. `src/executor.c:171-195` then writes the buffer over a UNIX socket. So the torn-read risk I flagged does not apply, the "copy under content lock" that macOS forces is how PG-Strom already works everywhere, and the spike's "visibility checks not implemented" limitation is moot for the kernel path.
2. **The zero-copy end-to-end figures were optimistic.** 7.2 ms (specialized) and 35 ms (VM) leave out the mandatory backend copy (about 21 ms per 823 MB). Realistic figures on this Mac are **about 28 ms and 56 ms**, against 163 ms for 8-worker PostgreSQL (5.8x and 2.9x faster). Zero-copy still helps on the service side, by replacing the socket copy with a shared-memory ring.
3. **Zero-copy only holds on unified memory** (Apple, integrated GPUs). On discrete GPUs every byte crosses PCIe. That path is designed for but not validated here.
4. **The live-`shared_buffers` module and kernel (`zcscan/`, `heap_live.slang`) are no longer needed.** The Metal-in-forked-process and multithreaded-postmaster findings still stand and are why the service must be an exec'd process on macOS.

## Target architecture

### Branch and repository policy
- Create `vulkan-gpu` (first named `apple-silicon`) from `master` HEAD. `master` keeps CUDA.
- Delete CUDA, `heterodb-extra`, GPUDirect and PostGIS only on this branch (milestone A2).

### Platform scope and capability tiers
- **Base tier (implemented and validated):** Vulkan 1.2 with MoltenVK on Apple Silicon. int64 arithmetic, 8/16-bit storage, 32-bit subgroup ballot and arithmetic, 32 KiB shared memory, unified memory with host-memory import. **No fp64, no 64-bit buffer atomics, and no Vulkan memory model** (MoltenVK reports `vulkanMemoryModel` false; found in A1). The base tier therefore must not rely on acquire/release atomics: lock-free structures in A3 need explicit barriers (`GroupMemoryBarrier`/`DeviceMemoryBarrier` plus relaxed atomics). The desktop tier requires the memory model.
- **Desktop tier (implemented after A3 as milestone A3b, experimental and unvalidated):** adds 64-bit atomics and fp64. The kernel source keeps two small shim modules (`xpu_atomics.slang`, `xpu_fp64.slang`) behind `#if PGSTROM_CAP_*`, so adding it later is additive. I recommend writing the base variant only now, rather than shipping untested desktop code.
- **Capability probe (`devcaps`):** at service start, each capability is on only if the feature flag, the extension **and a probe kernel** all agree. The spike showed MoltenVK reports int64 subgroup operations that Metal then rejects, so flags alone over-report. Results are cached per device UUID and driver version and published to a shared segment so the planner can read them.
- **Planner gating:** each `devtype`, `devfunc` and aggregate in `codegen.c` gets a `required_caps` field, and pushdown is refused when the device lacks one. On Apple: `float8` operators and `sum(float8)` stay on the CPU; integer sums and counts are pushed down; `min/max(int8)` in GROUP BY stay on the CPU until a design exists (partial results plus a reduce pass is the leading option). No fp64 emulation, because it is not IEEE-exact.

### Process model: an exec'd GPU service
- **Why:** a forked process cannot use Metal's shader compiler, and PostgreSQL 17 aborts if the postmaster becomes multithreaded (both reproduced earlier). So the GPU service becomes a separately exec'd binary, `pgstrom-gpuserv`.
- **Launcher:** a small background worker that never touches Vulkan spawns the service with `posix_spawn`, watches it and restarts it. A crash or hang of the GPU driver then kills only the service, not PostgreSQL.
- **Shared state:** `gpuServSharedState`, the GPU-Cache shared head and the error level move to a named POSIX shared-memory segment created by the postmaster with `shm_open` (no threads).
- **PostgreSQL internals:** the roughly 79 `elog`/`ereport`/interrupt sites in `gpu_service.c` (the architect's count; I did not re-count) go through a small shim.
- **Page ingest:** backends keep copying pages under the share lock (unchanged `relscan.c`) but write them into a per-session shared-memory ring instead of a `StringInfo` sent over the socket. The socket then carries only headers and offsets. The join inner buffer already uses this pattern (`.pgstrom_shmbuf_*`, `gpu_service.c:2217`).

### Memory model
Every allocation becomes an explicit buffer (VMA) with a placement class; nothing is managed memory.
- **Unified memory (Apple, validated):** host-visible, device-local, persistently mapped; the ingest ring is imported with `VK_EXT_external_memory_host`.
- **Discrete GPUs (designed for):** device-local buffers, explicit copies on a transfer queue. A debug GUC `pg_strom.gpu_memory_model=discrete` forces this path on the Mac so its *correctness* can be tested locally (not its speed).
- `cuMemPrefetchAsync`-style growth of the final buffer becomes suspend, grow, device-to-device copy, resume, reusing the existing suspend/resume logic. `cuMemcpyPeer` (`gpu_service.c:3627`) becomes a host-staged copy.
- Pointers become offsets (GPU-Cache's `hslots`/`next`, `kern_multirels` pointers). Buffer device address is used only for buffers above `maxStorageBufferRange`.

### Kernels
- **One Slang source.** A single op library, `xpu_ops.slang`, implements every opcode once; both the VM's `switch` and generated specialized kernels call it. `xpu_opcodes.h` is included unchanged (spike-verified). The `TypeOpCode`/`FuncOpCode` enums move out of `xpu_common.h` into a shared header.
- **VM first, specialization second.** The prebuilt VM SPIR-V ships with the product (`slangc` pinned, build-time only). Runtime specialization (spike: 8-13x faster, cold compile 0.2-6 s, pays back after about 7-9 scans) loads `libslang` with `dlopen`, compiles in a background thread while the VM serves the first scans, and caches SPIR-V on disk by program hash plus a persisted `VkPipelineCache`. If `libslang` is absent, everything runs on the VM.
- **Sizing:** no assumption of 32-thread warps or 512-thread blocks. `WaveGetLaneCount()` and `uint4` ballots, workgroup size as a specialization constant, dispatches split to stay well under about 1 s. `KVEC_UNITSZ` (`xpu_common.h:1834`) becomes a fixed constant.
- **Concurrency rule: no inter-workgroup spin-waits anywhere** (Vulkan guarantees no forward progress across workgroups).
  - *PreAgg NoGroups* (`cuda_gpupreagg.cu:1041-1107`): one-workgroup init dispatch creates the final tuple; later workgroups reduce in shared memory and merge with one atomic per accumulator.
  - *PreAgg GroupBy* (`:1632-1713`, `:1138-1185`): insert-only lock-free hash; reserve space, write key and initial accumulators, publish the tuple offset with a 32-bit CAS and release semantics; losers compare keys and either reuse the winner's tuple or probe on; a full table raises a suspend flag, the host grows and resumes. 64-bit sums use the 16-bit-limb technique verified in the spike.
  - *Join and scan row-count lock* (`cuda_gpujoin.cu:841-878`): per-subgroup atomic reservation.

### Renames and removals (branch only)
- **Delete:** `src/*.cu`, `src/Makefile.cuda`, `src/extra.c`, `src/heterodb_extra.h`, `pg_strom.gpudirect_*`, `pg_strom.extra_ereport_level`, `pgstrom.license_query()`, `estimate_cuda_stack_size` and the EXPLAIN line "CUDA Stack Size" (`executor.c:3185`), `docs/ssd2gpu`.
- **Rename or replace:** `pg_strom.cuda_visible_devices` → `pg_strom.gpu_visible_devices`; `pg_strom.cuda_toolkit_basedir` and `cuda_stack_limit` removed; `DEVKIND__NVIDIA_GPU` → `DEVKIND__VULKAN_GPU`; `pgstrom.gpu_device_info()` gets a Vulkan attribute set (device name, UUID, driver, tier, devcaps bits, memory heaps) instead of CUDA attributes from `gpu_devattrs.h`.
- **Replace without the closed module:** device validation (dropped) and optimal-GPU selection (single device for now; PCI-bus-info based selection is a later option). Without the module PG-Strom today keeps only one GPU anyway.
- **Expected outputs:** about 4 files per PostgreSQL version (15-19) contain CUDA/GPU strings (`CUDA Stack Size` numbers, `GPU Cache: GPU0 ...`); edit mechanically.
- **Build and packaging:** `src/Makefile` loses `-lcuda` and the nvcc rules and gains slangc, Vulkan loader and MoltenVK detection; spec and deb files drop the `heterodb-extra` Recommends and `CUDA_PATH`.

## Milestones (all on branch `vulkan-gpu`; every gate runs on this Mac)

| | Scope | Exit criteria |
|---|---|---|
| **A0** Branch and platform (**done**, see below) | Create the branch. macOS portability: event loop, `shm_open`+`ftruncate`, `F_NOCACHE`, `st_mtimespec`, `gettid`, byte swapping, runtime page size. A temporary no-CUDA stub so the host code builds without the CUDA toolkit. Linux check in an arm64 container. | Builds and loads on macOS and on arm64 Linux with no GPU; `pgstrom_guc` passes (with a no-GPU expected file); `pgstrom_init` completes and loads the dataset; the portability test passes on both. |
| **A1** Service skeleton (**core done**, see below) | Exec'd `pgstrom-gpuserv`, launcher worker, control segment, ingest ring, `gpu_vk.c` (buffers, copy, dispatch, fence, pipeline), devcaps probe, VM scaffold. | Killing the service needs no PostgreSQL restart. Probe output matches the spike's findings. |
| **A2** GpuScan on the VM, then remove CUDA (**in progress**, see below) | Heap `BLOCK`/`ROW` formats, quals, projection, error buffer, CPU fallback. Types: bool, int1-8, float2/4/8 (fp64 gated). **Then delete CUDA** as listed above; disable join, aggregate, sort and GPU-Cache planning with a one-time NOTICE; move their tests to a `pending_schedule`. | `dtype_int`, `dtype_float` and scan-only `dexpr_*` tests pass. Clean build with no CUDA. Expected files fixed for PostgreSQL 15-19 text. Scan 2.9x or better than 8-worker PostgreSQL on a 16.8M-row table. |
| **A3** GpuPreAgg (**in progress**: aggregation without GROUP BY done, see "A3 status") | NoGroups, then lock-free GroupBy, aggregate catalog with `required_caps`. | `gpupreagg` tests pass; aggregate queries match the CPU oracle; faster than parallel PostgreSQL on the same table. |
| **A3b** Desktop tier (experimental, decided) | `xpu_atomics.slang` and `xpu_fp64.slang` shims selected by `PGSTROM_CAP_*`; `float8` operators and `sum`/`min`/`max` on `int8` pushed down when the probed `devcaps` allow; buffers above `maxStorageBufferRange` through BDA (lavapipe reports only 128 MiB); no subgroup-size assumption (lavapipe reports 4). Nothing here changes the Apple path. | Kernel unit tests and a regression subset pass on lavapipe in the Linux container (fp64 results bit-exact against the CPU, 64-bit atomic aggregates exact), with the tier also forced to base on lavapipe; the Apple results are unchanged; `gpu_device_info` and the docs label the tier experimental and unvalidated; no performance claims. |
| **A4** GpuJoin | Hash, nest-loop and outer joins; suspend/resume with state as offsets in `kern_gputask`; pinned inner buffer in shared memory. | `gpujoin` tests pass; join queries match the CPU oracle. |
| **A5** Arrow_Fdw and Parquet | The arrow and parquet source formats in the scan kernel, ingest through the shared-memory ring or import of the mapped file. | `arrow_*` and `parquet_cache` tests pass. |
| **A6** Types and specialization | text, numeric (int128 as 64-bit limbs), time; `libslang` specialization with a planner threshold; differential fuzzer (VM vs specialized vs CPU). | `dtype_*`, `dfunc_*` pass; fuzzer clean for 24 h. |
| **A7** GpuSort/window and GPU-Cache | Bitonic sort and window-rank kernels (one global pass per launch, no inter-workgroup waits); GPU-Cache with offset-based hash and log apply. | `gpu_sort` and `gpu_cache` tests pass. |
| **A8** jsonb | Port the jsonb tree walking over raw buffers (about 1.3k lines, mostly redesign). Cut line: if it blows the budget, ship jsonb on the CPU and say so. | `dtype_jsonb` passes or is explicitly marked CPU-only. |
| **A9** Hardening and docs | Rewrite `install`, `troubles`, `ref_params`; packaging; CI. | Full regression schedule green on this Mac; docs state that only Apple Silicon is validated. |

## A0 status (done)

Branch created from `master` (committed since; see `git log`). Verified on this Mac (macOS 27, Apple Silicon, PostgreSQL 17.11 from Homebrew, Arrow 26.0.0) and in an arm64 Ubuntu 24.04 container (OrbStack, PostgreSQL 17.11, Arrow 26.0.0).

**What changed (about 48 files, 237 insertions)**
- **Build:** `Makefile.common` escapes `#` so GNU make 3.81 (macOS) works; `src/Makefile.cuda` enters a no-CUDA mode when no toolkit is found; `src/Makefile` builds against the stub, keeps Arrow's include flags (`PG_CXXFLAGS` was reset with `:=`), links `libc++` on Darwin, adds `-mfp16-format` only when the compiler accepts it (GCC 13 on aarch64 does not), and detects libraries with `printf` instead of `echo -n` (macOS `/bin/sh` prints the `-n`).
- **`src/nocuda/`** (temporary, deleted at A2): stub `cuda.h`, `cufile.h`, `cudaProfiler.h`, `gpu_devattrs.h` and `cuda_stub.c`. `cuInit` succeeds and `cuDeviceGetCount` reports zero devices, so the host code takes its existing no-GPU path.
- **`src/pg_portable.[ch]`** (new, libc only): event loop (epoll on Linux, round-robin `poll` elsewhere), `pgstromOpenDirect` (`O_DIRECT` or `F_NOCACHE`), `pgstromPreallocate`, `gettid`, `strdupa`, `st_mtim`/`st_ctim` and `SIGPOLL` aliases.
- **Call sites:** `gpu_service.c` (event loop, `uint64` for PostgreSQL's atomics), `misc.c` (`shm_open`/`shm_unlink` with a shared name helper, runtime page size), `select_into.c` and `parquet_cache.c` (`O_DIRECT`), `xpu_common.h` (byte swapping), `parquet_read.cpp`.
- **Arrow/Parquet:** the Arrow and Parquet headers are now included under `#pragma GCC visibility push(default)`. Without it, `dynamic_cast` to `parquet::Int32Statistics` returned NULL on macOS (the module is built `-fvisibility=hidden`) and crashed the backend when creating a Parquet foreign table.
- **Tests:** 33 files use `printf %s` instead of `` `echo -n` `` (same macOS `sh` problem); `test/ssbm` builds on macOS (`malloc.h`, `O_LARGEFILE`, a header on the compile line); new `test/port/port_test.c`; new `test/17/expected/pgstrom_guc_1.out` (no-GPU expectation).

**Verification**

| Check | macOS | arm64 Linux |
|---|---|---|
| Module builds, no CUDA | yes (0 errors) | yes (0 errors) |
| Server starts with `shared_preload_libraries = 'pg_strom'`, `CREATE EXTENSION` works | yes | yes |
| `pg_strom.enabled` = on, 0 devices, normal PostgreSQL plans, no GPU service worker | yes | yes |
| `test/port/port_test` (27 checks: event loop fairness, timeout, hang-up, direct I/O, shm sizing, `gettid`, `st_mtim`) | all pass | all pass |
| `pgstrom_guc` | pass (alternative expected file) | pass |
| `pgstrom_init`: dataset load | completes, 0 errors, revision 20260430, `lineorder` 6,001,215 rows, `plineorder` 4,286,946 rows, 14 Arrow/Parquet foreign tables | not run |
| Arrow_Fdw on a Parquet file (CPU path) | `SELECT count(*)` = 259,138, correct | not run |

**Findings that change the plan or the A0 scope**
- **Process-shared mutex, condition variable and rwlock work on macOS 27** (tested, including a cross-process wake-up), so the planned replacement is unnecessary here. Older macOS versions were not tested.
- **`PAGE_SIZE` (4096) stays** as the device-ABI constant; only the host `mmap` rounding needed the runtime page size.
- **Arrow and Parquet are a hard build requirement** (libarrow 23 or newer, with Parquet; there is no guard to compile without them). On macOS that means `brew install apache-arrow pkgconf`, which pulls 16 dependencies including `llvm`, `grpc` and `aws-sdk-cpp`.
- **GPU-specific GUCs only exist when a device is found** (15 of them, for example `pg_strom.enable_gpuscan`, `gpu_setup_cost`, `gpu_mempool_*`), by upstream design. That is why `pgstrom_guc` needs a no-GPU expected file. A1 registers them when the Vulkan probe finds a device.
- **`pgstrom_init.out` in the repo is stale relative to its own SQL** (it expects revision 20240530 and does not echo the newer partitioned-table section). It does not pass on any platform as written; I did not change it. Success here means the script completes and sets the revision.

**Not done, and known gaps**
- `arrow-tools` (`pg2arrow`, `arrow2csv`) do not build on macOS yet (`endian.h` and probably more). The Arrow tests that use them belong to A5.
- Only PostgreSQL 17 was tested. The 15, 16, 18 and 19 expected-output directories do not have the no-GPU alternative.
- `make installcheck` as a whole was not run (without a GPU most tests are expected to differ); `pgstrom_init` and `pgstrom_guc` were run through `pg_regress` directly.
- A few cosmetic warnings remain (`%lu` against `uint64_t`, an Arrow `Make` deprecation).
- The module was installed into the Homebrew PostgreSQL tree for the A0 tests and removed again; A1 testing installed it again (see below), and it is still there.

## A1 status (core done)

Verified on this Mac (Apple M1 Pro, MoltenVK) and in the arm64 Linux container (lavapipe, a CPU Vulkan device, enabled with `pg_strom.gpu_allow_cpu_device`).

**What exists**
- **`src/gpu_launcher.c`:** the postmaster runs the probe (`pgstrom-gpuserv --probe`, with a timeout) at start-up, creates the control segment (named POSIX shm `/pgsctl.<port>`), and registers a launcher background worker. The worker fork+execs `pgstrom-gpuserv --serve`, passes a lifeline pipe on fd 3, restarts the service with back-off (1 s to 30 s, reset after 60 s of uptime) and stops it on shutdown (SIGTERM, SIGKILL after a grace period). GUCs: `pg_strom.gpuserv_path`, `pg_strom.gpu_allow_cpu_device`, `pg_strom.gpuserv_probe_timeout`.
- **`src/gpuserv/`** (libc and Vulkan only, no PostgreSQL headers): `gpuserv_proto.[ch]` (control segment layout, probe text format, socket handshake), `gpu_vk.[ch]` (device open, buffers, host-memory import, pipelines, dispatch with fence timeout), `devcaps.c` (probe kernels and tiers NONE/BASE/DESKTOP), `gpuserv_main.c` (`--probe`, `--serve`), `vm_selftest.[ch]` and `kernels_vm/` (the opcode-switch VM scaffold, built from the real `xpu_common.h` enums by `gen_opcodes.sh`), `kernels/` (14 probe kernels). SPIR-V is embedded as C arrays by `gen_kspv.sh`.
- **SQL (`pg_strom--6.2--7.0.sql`, default version 7.0):** `pgstrom.gpu_devcaps()`, `pgstrom.gpu_service_state()`, `pgstrom.gpu_service_ping()`.
- **Build:** `src/Makefile` builds the service through `WITH_GPUSERV` hook rules (warns and skips when `slangc` or Vulkan is missing, unless `WITH_GPUSERV=1`) and installs it into `$(pkglibdir)`.
- **Startup self-test:** before saying "ready" the service runs the VM kernel on 65,536 synthetic rows (a four-term qual with int64 arithmetic and NULLs, plus an `int4div` by zero case) and compares the match set and error code with a CPU three-valued-logic reference. A device that fails is reported as failed, not used.

**Verification (`test/gpuserv/service_test.sh`, a throwaway cluster)**

| Check | Apple M1 Pro | arm64 Linux, lavapipe |
|---|---|---|
| Probe result | tier base | tier desktop |
| Service opens the device, VM self-test | ok: 10,092 of 65,536 rows matched, `int4div` error reported | same result |
| `CREATE EXTENSION`, state `ready`, `gpu_devcaps()`, `gpu_service_ping()` | pass | pass |
| `kill -9` of the service: new process, `restart_count` = 1, ping works, queries keep working, no PostgreSQL restart | pass | pass |
| Clean shutdown: PostgreSQL stops, service gone, socket removed | pass | pass |
| Probe without `--allow-cpu` refuses the CPU device | n/a | pass (`tier=none`, "CPU device") |
| `pgstrom_guc` (no-GPU alternative), `test/port/port_test` | pass | pass (A0) |

**Findings**
- **Base tier has no Vulkan memory model** (see Platform scope). An earlier tier rule that required it classified Apple as NONE; the requirement moved to the desktop tier.
- **Feature flags over-report** as expected: MoltenVK claims 64-bit subgroup arithmetic and Metal rejects it. The probe kernels catch this, so reported and measured capabilities are stored separately (`reported_caps`, `caps`).
- Building on Linux needs `libkrb5-dev`: PostgreSQL's `libpq/libpq.h` includes `gssapi/gssapi.h`.

**Not done yet in A1**
- The shared-memory ingest ring and its host-import test (the socket handshake exists; no data path yet).
- Devcaps cache file keyed by device UUID and driver version.
- `pgstrom.gpu_device_info()` still reports the CUDA-era attributes (replaced when CUDA is deleted in A2).
- GPU GUCs are still registered only when the old CUDA path finds a device; they must be registered from the probe result in A2.
- Packaging details (the service is installed by `make install`, no spec/deb changes).

## A2 status (in progress)

**Design decision: keep the wire protocol.** The exec'd service speaks the existing `XpuCommand` frames (`OpenSession`, `XpuTaskExec`, `XpuTaskFinal`; reply `Success` or `Error`), built from the real `xpu_common.h`. `xpu_common.h` compiles in plain C with only `-DCUDA_MAXTHREADS_PER_BLOCK=512`, so the service uses the real structs (`kern_session_info`, `kern_expression`, `kern_data_store`, `XpuCommand`) with no mirror to keep in sync. Backend code (`executor.c`, `gpu_scan.c`, `relscan.c`, `fallback.c`, `codegen.c`) therefore stays as it is, and the old `gpu_service.c` is replaced rather than ported.

**Sub-steps**

| Step | Content | State |
|---|---|---|
| A2.1 | Wire server in the service (`src/gpuserv/wire.[ch]`): framing, connection table, OpenSession, task and final replies, error replies, `PGSTROM_GPUSERV_TRACE=1` dump of the session's xpucode | done |
| A2.2 | Planner integration: `gpuDevAttrs` filled from the probed devcaps, GPU GUCs registered from the probe, `gpuserv_ready_accept()` reads the control segment, backend connects to the control segment's socket, old CUDA worker not registered (`PGSTROM_NO_CUDA`) | done |
| A2.3 | Scan execution: heap-page decode (`BLOCK` source, remaining pages mapped from the file by the service), `kern_expression` to VM program translator, projection into a `KDS_FORMAT_ROW` result, error reporting | **done** (core; CPU fallback and the gaps below remain) |
| A2.4 | Planner gating by what the VM implements, query parameters, int/float4 casts, PostgreSQL-style kernel error messages | **done**; CPU fallback for kernel errors and breadth beyond int/float4/bool move to later (see below) |
| A2.5 | Delete CUDA (list under "Renames and removals"), fix expected files | **done** except the expected-file alternates (see below) |

**Verified so far (Apple M1 Pro)**
- `EXPLAIN` on a 3M-row table shows `Custom Scan (GpuScan)`, `GPU Scan Quals`, `Scan-Engine: VFS with GPU0`; the planner now offers GpuScan driven only by the Vulkan probe (plans are unchanged without a device).
- Executing it opens a session on the service and sends the tasks; the service answers with a clean `Error` ("GpuScan execution is not implemented in the Vulkan service yet"), which the backend reports as an ordinary error. The A1 lifecycle test still passes.
- Real xpucode from the planner for `WHERE v < 10 AND w + 1 > 5`: `Packed[LoadVars depth=0 resno2->slot0 resno3->slot1]` and `BoolExpr_And(int4lt(Var slot0, Const), int84gt(int84pl(Var slot1, Const), Const))`; slots carry type, length, alignment and a kvec offset. This is the input the A2.3 linearizer must consume.

**A2.3 result (Apple M1 Pro; `test/gpuserv/scan_test.sh`)**

What runs on the GPU: one workgroup per heap page, one thread per line pointer. Each thread deforms the columns the program needs (null bitmap, alignment, varlena skipping), runs the qual program of the scan VM (bool, int1/2/4/8 and float4 with PostgreSQL's overflow, zero-divide and NaN rules; int casts; three-valued AND/OR/NOT; IS [NOT] NULL on any column type), and for qualifying tuples evaluates the computed projection expressions. The service then forms the result tuples on the CPU, copying pass-through columns (any type, varlena and external TOAST pointers included, ctid and tableoid) straight from the source tuple.

- **Differential test:** 14 + 2 query shapes on three tables, each GPU result set compared with the CPU's by md5: a table whose pages are all-visible (pages mapped from the file), one with UPDATEs and DELETEs and no vacuum (pages loaded and visibility-checked by the backend), and a mix. NULLs, NaN and Infinity, 200-byte text, int2/int4/int8 mixed arithmetic, casts, division and modulo, `IS NULL` on text, 10 and 24 computed columns (the 16- and 32-register kernels), a division-by-zero error on both sides, and parallel workers. **All pass on the M1 Pro, with and without the zero-copy path (`PGSTROM_GPUSERV_NO_MAP=1`), and on arm64 Linux with lavapipe (desktop tier, subgroup size 4, copy path).** UPDATE and DELETE (planned as GpuScans projecting `ctid`) work.
- **Plan gating for now:** GpuPreAgg, GpuJoin, GPU-Cache and SELECT INTO direct are not registered in the Vulkan build (their init is compiled out), so aggregate and join queries run on the CPU instead of failing. Each comes back with its milestone (A3, A4, A7).
- **Page ingest:** the pages the backend loaded go into a small buffer; the relation file's own pages are not copied: a 512 MB window of the file is `mmap`ed and imported as a Vulkan buffer (`VK_EXT_external_memory_host`) and **kept between tasks**. Importing a fresh mapping per task was no faster than `pread` (the GPU pays the page-in on first touch, about 15 ms per 64 MB); a persistent window is what the spike measured. A per-page table tells the kernel where each page is. If a window is not usable (pages past the end of the file, a buffer limit smaller than a window as on lavapipe) every page is copied.
- **Kernel findings:** workgroup size (32 to 256) makes no difference; **register-file size does** (32 registers 3.6 ms, 8 registers 2.1 ms per 64 MB), so there are three kernel variants (8, 16 and 32 registers) chosen by the program. Loads are 32-bit words (tuples are 8-byte aligned). The kernel writes one flag per line pointer instead of compacting with atomics, so the service reads matches in heap order with no sort (a `qsort` of 8.4M records had cost 1.5 s).

**Performance against the gate (16.8M rows, 964 MB, 8 CPU cores, server-side execution time of `EXPLAIN ANALYZE`)**

| Query | GPU (VM) | PostgreSQL, 8 processes | PostgreSQL, 1 process |
|---|---|---|---|
| `SELECT id ... WHERE v < 10 AND w + 1 > 5` (0.9% of rows) | **about 90 ms** (first run 175 ms: file windows paged in) | 124 ms | 621 ms |
| `SELECT id, v, w, e ... WHERE v < 500` (50% of rows) | 980 ms | **195 ms** | 671 ms |

**The A2 gate (scan at least 2.9x faster than 8-worker PostgreSQL) is not met: the selective scan is 1.4x faster.** Where the time goes for the selective query, per 64 MB task: GPU 2.2 ms (about 30 GB/s; the spike's VM kernel on the same data class was about 35 ms for 863 MB, the same figure), flag scan and tuple forming 1.7 ms, and about 25 ms of the 90 ms is the backend (checking blocks, sending tasks, receiving results). The service is single-threaded, so tasks do not overlap. Specialized kernels (A6) take the GPU part to about a fifth, and the service can pipeline (form tuples for task N while the GPU runs task N+1) and form tuples on several threads. For the 50% query the cost is moving 8.4M tuples (about 400 MB) through the socket and forming them on one thread; that is inherent to the socket protocol (the CUDA service sent results the same way). Offload pays on selective scans and, in A3, aggregation, where only a few rows come back.

**A2.4 result**
- **Planner gating:** `src/gpuserv/scan_caps.[ch]` is the single description of what the scan VM implements (function opcodes classified by name: `intAB` comparison and arithmetic, `float4` comparison and arithmetic, casts between the integer types and to/from float4, bool equality). It is linked into both the module and the service. In the Vulkan build `codegen.c` refuses to push down any function the VM lacks and the node types it lacks (DistinctExpr, BooleanTest, CoerceViaIO, COALESCE, GREATEST/LEAST, CASE, `IN` lists). The planner then keeps those quals and projections on the CPU, so such queries are correct instead of failing: float8 comparisons, text equality and `LIKE`, `CASE`, `IN (...)`, `COALESCE`, `IS TRUE`, `upper()` all return the CPU's result (tested, mixed with GPU-capable quals in the same query). There is no separate `required_caps` table yet; fp64 and 64-bit atomics gating arrives with the desktop tier (A3b), by adding opcodes to `scan_caps.c`.
- **Query parameters** (prepared statements with a generic plan, and stable functions of constants, which the planner turns into parameters) run on the GPU: a parameter is a session constant.
- **int to float4 and float4 to int casts** (round half to even, as `rint()`; range and NaN errors).
- **Error messages** carry a reason: "division by zero", "integer out of range" (by width), "value out of range: overflow" and "underflow", with the function and the block and line pointer.
- **`EXPLAIN ANALYZE` counters fixed (an upstream bug, also on `master`):** after the scan ended, `pgstromSharedStateShutdownDSM()` re-derived the per-relation counters' address from `sizeof(pgstromSharedScanState)` while init and attach use `sizeof(pgstromSharedState)`, so `exec: raw -> in` and `ntuples=` showed unrelated memory (for example 10705 -> 5696 for a 300,000-row scan). They now read `exec: 300000 -> 63997`, with and without parallel workers.
- **Tests:** `scan_test.sh` now has 41 query shapes per run: it adds float casts with half-even rounding, a generic-plan prepared statement, seven CPU-stays expression shapes, and checks the "division by zero" text. All pass on the M1 Pro (zero-copy and copy path) and on arm64 Linux with lavapipe.

**A2.5 result: CUDA is gone from the branch**
- **Deleted:** all `cuda_*.cu` and `xpu_*.cu` device code, `Makefile.cuda` and the fatbin build, `gpu_service.c` (the old service, about 6,800 lines), `extra.c` and `heterodb_extra.h` (the closed-source GPU-Direct module), `select_into.c` (and the `CREATE TABLE AS` direct-write hook and `pg_strom.enable_select_into_direct` in `main.c`), `gist.c` and `xpu_postgis.*` (PostGIS) with every `WITH_POSTGIS` block, `nocuda/` (the stubs, `cuda.h`, `cufile.h`), `src/gpu_devattrs.h` and its generation, `estimate_cuda_stack_size` and the `CUDA Stack Size` EXPLAIN line (also removed from 14 expected files), the CUDA 13 compat macros, the `CUDA_PATH` and `heterodb-extra` lines of the RPM spec. The module links with no CUDA symbol and is 144 KB smaller. The CUDA sources stay available on `master`.
- **Rewritten or stubbed:** `gpu_device.c` (planner's device description from the probe; `GetOptimalGpuFor*` return none because there is no GPU-Direct; `pgstrom.gpu_device_info()` now lists the probe's attributes), `gpu_client.c` (new: `gpuserv_ready_accept`, `pg_strom.max_async_tasks`, `pg_strom.debug_level`, the connection to the service), `gpu_cache.c` and `parquet_cache.c` (stubs that keep the SQL-visible symbols; the features return in A7 and A5), `parquet_read.cpp` (GPU-memory read paths and the GPU-Direct cache lookup removed; the host-memory reader remains), `pg_strom.h`, `Makefile`.
- **GUCs removed:** `pg_strom.cuda_visible_devices` (now `pg_strom.gpu_visible_devices`, device indexes), `cuda_toolkit_basedir`, `cuda_stack_limit`, `gpudirect_enabled`, `gpudirect_threshold`, `gpudirect_driver`, `extra_ereport_level` (now `pg_strom.debug_level`), `gpu_mempool_*`, `enable_select_into_direct`, `parquet_cache_*`. `pgstrom.license_query()` is dropped by the 7.0 script (its C symbol stays for the older scripts of a fresh install).
- **A latent bug found and fixed:** the executor-start hook could switch on SELECT INTO direct mode for unlogged and temporary `CREATE TABLE AS`; the Vulkan service does not write tables, so such a table would have come out empty. The whole feature is gone.
- **`xpu_common.h` gained `strom_io_chunk` / `strom_io_vector`** (they lived in the deleted `heterodb_extra.h`; they are part of the backend-to-service protocol).
- **Kept on purpose:** `cuda_common.h` (device-neutral shared structs that A3, A4 and A7 still use; it will be renamed then), the `CUDA_MAXTHREADS_PER_BLOCK` constant (512) that `xpu_common.h` needs, `kern_session_info.cuda_stack_size` (wire field, now 0).
- **Verified after the deletion:** `scan_test.sh` (macOS: all pass; arm64 Linux with lavapipe: all pass), `service_test.sh`, `port_test`, a fresh `CREATE EXTENSION` (every symbol referenced by the SQL scripts exists in the module; this caught a missing `pgstrom_parquet_cache_info`).
- **The upstream regression suite on the Vulkan build:** the dataset load (`pgstrom_init`, 6.0M `lineorder` rows) works. `dtype_int`, `dtype_float`, `dfunc_math`, `dexpr_*`, `dtype_numeric/time/text` all fail on `EXPLAIN` text only as far as I looked (plans show a Seq Scan or a GpuScan with different quals and projections because float8, numeric, text and time stay on the CPU); the result rows match. **Not done:** alternate expected files (`*_1.out`) for the base tier on each PostgreSQL version (15-19), and a `vulkan_schedule` / `pending_schedule` split of `parallel_schedule`. That is mechanical but large; it is better done once A3 and A4 settle what the plans look like. Until then `scan_test.sh` is the correctness gate.
- **Still mentions CUDA:** `docs/` (generated HTML; A9), the CI workflow and `test/automation_setup`, the `deb` packaging, and `arrow-tools` / `test/ssbm` Makefiles.

**Not done yet in the Vulkan GpuScan (known gaps)**
- **No CPU fallback.** A kernel error (division by zero, overflow) fails the query. The CUDA service did the same unless `pg_strom.cpu_fallback` was on (default off), and the difference from PostgreSQL (a row the CPU would not have evaluated, because of short-circuit order, fails) is the same as before. Supporting the GUC means returning failing chunks through the `KDS_FORMAT_FALLBACK` path; not started.
- **Breadth:** only bool, int1/2/4/8 and float4 are computed on the GPU. Everything else (float8, numeric, text, date/time, jsonb) stays on the CPU until A6 (and fp64 on desktop devices until A3b). Pass-through columns of any type already work.
- Sources other than heap pages (`ROW`, Arrow, Parquet), GPU-Cache tasks and xmin/xmax/cmin/cmax columns are not supported.
- The A1 VM self-test still uses the old synthetic-row VM; it should move to the scan VM.
- `pgstrom_guc` and other upstream tests have not been re-run against the Vulkan build with a device present (the expected files assume either all GPU GUCs or none).

**Findings for A2.3 (made before implementing)**
- The CUDA scan kernel is warp-centric (stair sums, warp contexts, per-depth kvecs). The VM is one thread per row, so the heap-page walk and the result writer are new Slang code, not a port of `cuda_gpuscan.cu`.
- `XpuTaskExec` for a `BLOCK` source carries the pages loaded by the backend plus a path and an iovec for pages the service must read itself (all-visible pages, previously read by GPUDirect or VFS). Without GPUDirect the service reads them with `pread`.
- Final reply for a plain scan: `Success`, `scan_relidx = -1`, `final_plan_task = true`, no chunks. Task reply: `Success` with `kern_exec_results` plus one `KDS_FORMAT_ROW` chunk (row index array, tuples packed from the end).
- `count(*)` queries carry a `groupby_actions` xpucode (`AggFuncs`): they are GpuPreAgg (A3), not GpuScan. Plain `SELECT ... WHERE` is the A2.3 target.
- Leftover CUDA-era GUCs still register (`pg_strom.cuda_visible_devices`, `gpudirect_*`, `extra_ereport_level`); they go in A2.5.

## A3 status (in progress)

**Design: PostgreSQL finalizes, the service produces partial states.** The planner now always builds the two-phase shape the CUDA code used only when `pg_strom.cpu_fallback` was on: `Aggregate (CPU) -> Custom Scan (GpuPreAgg)`. The GpuPreAgg node returns PostgreSQL-format partial aggregate states (`kagg_state__*` in `bytea`, `int8` for counts), and PostgreSQL's own Agg node combines and finalizes them with the existing `pgstrom.fsum_int`, `fcount`, `favg_*`, `fminmax_*` functions. So the service needs no final buffer, no finalization and no `MERGE_FINAL_BUFFER`; the one-row-per-session partial result is returned in the reply to `XpuTaskFinal`, as the CUDA service did for this shape. (The planner's "complete aggregation" path, where the service finalizes, is not used.)

**Done: aggregation without GROUP BY** (count(*), count(x), sum/avg/min/max of int1/2/4/8, including `FILTER (WHERE ...)` and expression arguments):
- **Planner:** `pg_strom.enable_gpupreagg` is registered again; the aggregate catalog offers only the actions the service implements (`scan_vm_agg_action_supported()` in `scan_caps.c`: NROWS, PSUM/PAVG INT and INT64, PMIN/PMAX INT32 and INT64). float8, numeric, text and so on stay on the CPU.
- **Kernel (`kernels_vm/agg.slang`)** shares the row evaluation with the scan kernel (`scan_common.slang`: page loads, deform, VM, `eval_row`). One workgroup runs over 32 pages; each thread accumulates the qualifying tuples of its line pointers in registers, the workgroup reduces in shared memory (no atomics, nothing shared between workgroups, nothing that needs the Vulkan memory model, so it is valid on the base tier) and writes one partial record per aggregate. Sums are carried as the sum of the low 32 bits and the sum of the high 32 bits, and the service adds them up in 128 bits, so `sum(int8)` is exact beyond 64 bits (checked against PostgreSQL's numeric).
- **Service:** `scan_prog.c` has an aggregation mode (SaveExpr arguments compiled to registers, an aggregate table in the program); `scan_exec.c` runs the tasks, keeps a per-session `agg_state` (counts, `__int128` sums, min/max) and `scan_exec_agg_final()` builds the partial-state row for `XpuTaskFinal`; each parallel worker has its own session and PostgreSQL combines the workers' rows.
- **Tests:** `scan_test.sh` has 28 aggregate checks per run: NULLs, empty input, NULL-only columns, expression arguments, FILTER, sums beyond 64 bits, float aggregates (stay on the CPU), parallel workers; all pass on the M1 Pro and on arm64 Linux with lavapipe.
- **Performance (16.8M rows, server-side time of `EXPLAIN ANALYZE`, 8 CPU cores):**

| Query | GPU | 8-process PostgreSQL |
|---|---|---|
| `count(*), sum(w), max(v), avg(v) ... WHERE v < 500` | **67 ms** | 172 ms |
| `count(*), sum(id), min(w)` (no WHERE) | **40 ms** | 159 ms |
| `count(*) ... WHERE v < 10` | **35 ms** | 115 ms |

  That is 2.6x to 4.0x faster than parallel PostgreSQL; the A3 criterion ("faster than parallel PostgreSQL on the same table") holds for this shape. The GPU takes about 2 ms per 64 MB task. Two measured lessons: **the size of the per-thread accumulator arrays dominates** (12 entries 8.2 ms per task, 4 entries 4.3 ms; dynamically indexed arrays spill), so the kernel has variants by aggregate count (4, 8, 12) and register count (8, 16, 32), 9 SPIR-V modules; and **one workgroup per page is too small a unit** (the reduction cost about as much as the work), now 32 pages per workgroup.

**Next: GROUP BY** (the main use of aggregation, e.g. the SSB queries). Design, following the "no inter-workgroup waits" rule and keeping everything valid on the base tier:
- Each workgroup runs over a run of pages and builds a **shared-memory hash table** of its groups (key columns plus the same accumulators as above; capacity bounded by the 32 KiB of shared memory, a few hundred groups). Rows of a group that does not fit are marked **overflow**; the service aggregates those on the CPU from the source tuples.
- The service merges the workgroups' (key, state) records into a per-session hash table keyed by the group key bytes (the same merge as for the no-group case, per group) and returns one partial row per group at `XpuTaskFinal`. PostgreSQL's Agg node still combines across workers.
- Key types: bool and int1/2/4/8 first (hash of the key in the kernel), text and other varlena keys later (compare through the source tuple).
- Open question: for high-cardinality keys (more groups than a workgroup's table) most rows overflow and the GPU adds little; the planner should then keep the aggregation on the CPU. The group-count estimate that the planner already has (`num_groups`) can gate it.

## Effort (an estimate, not measured)

About **20-35 engineer-months** in total: the architect's 17-30 for a Vulkan-only product, minus vendor-hardware work, plus Arrow/Parquet and jsonb that were not in that figure. The largest uncertainties are the expression and type library, the lock-free PreAgg redesign, and jsonb. A rough split: A0-A2 about 5-7, A3 2-4, A4 3-5, A5 2-3, A6 3-6, A7 2-4, A8 2-4, A9 1-2.

## Validation matrix

| Layer | This Mac | Optional, free |
|---|---|---|
| Host unit tests (linearizer, ABI asserts) | yes | Linux VM |
| Kernel unit tests (`vk_unit`), base tier | yes | |
| Regression schedule vs the CPU oracle | yes | |
| Discrete-memory code path (forced by GUC) | correctness only | |
| Desktop-tier shims (64-bit atomics, fp64) and Linux host build | no | lavapipe in an OrbStack Linux VM (CPU Vulkan, no vendor GPU): catches forward-progress and ordering bugs |
| NVIDIA / AMD / Intel behaviour and performance | **not tested** | |

I recommend the optional lavapipe VM track because it needs no NVIDIA/AMD/Intel hardware and is the only way to exercise the Linux host code and the desktop-tier logic. Say so if you would rather skip it.

## Desktop tier: pros and cons (decided: implement after A3, experimental)

Evidence from the Linux container: lavapipe (Mesa's CPU Vulkan, Vulkan 1.4) reports `shaderInt64`, `shaderFloat64`, 64-bit buffer and shared atomics, buffer device address, the Vulkan memory model, and `VK_EXT_external_memory_host`. All the Apple spike kernels (scan, 16-bit-limb sum, the opcode VM, specialized kernels) passed on it, and Apple's Metal-rejected int64 `subgroupAdd` compiles there. Two numbers from it matter for the design: **subgroup size 4** (code that assumes 32 breaks) and **`maxStorageBufferRange` of 128 MiB**, which a base tier that assumes at least 1 GiB would exclude (BDA or chunking is needed). The VM was 3.2x slower than the hand-written kernel there, against 10x on the M1; CPU timings say nothing about GPUs.

| | Implementing the desktop tier (64-bit atomics, fp64) |
|---|---|
| **Pros** | `float8`, the most common floating type in PostgreSQL, can be pushed down (on Apple it stays on the CPU). 64-bit atomics make PreAgg simpler and faster: `sum`, `min`, `max` on `int8` and 64-bit hash-slot CAS need no limb or two-pass workaround. It delivers the stated cross-platform objective (NVIDIA, AMD, Intel on Linux). lavapipe gives a free correctness oracle for the logic, including exact IEEE fp64, with no GPU needed. BDA lifts the 4 GiB buffer limit. |
| **Cons** | It cannot be validated on real hardware, which you have ruled out; lavapipe passing is not the same as an NVIDIA, AMD or Intel GPU passing (driver quirks, NVIDIA's proprietary int64-atomics support is unverified, AMD wave64, Intel subgroups of 8 or 16, forward progress, memory ordering). Two compiled kernel variants double the test matrix and can drift apart. Discrete GPUs change the memory design (PCIe copies) and performance, none of which can be measured here. It costs effort that the Apple-only product does not need. |

**Recommendation:** keep the tier abstraction and the probed `devcaps` now (cheap, and Apple needs them anyway). Defer the desktop shims until after A3, then implement only what lavapipe can verify exactly (fp64 arithmetic, 64-bit atomic sum/min/max), behind capability bits so the Apple path is untouched, and ship it labelled experimental and unvalidated. Do not tune performance for it. **Decided:** the desktop tier is implemented after A3 as milestone A3b, on exactly these terms (verifiable-by-lavapipe features only, behind capability bits, experimental, unvalidated, no performance tuning).

## Risks

**Top 3 that could invalidate the approach**
1. **Cost of the expression and type engine** (about 670 opcodes, varlena in the VM, superlinear Metal compile times). Mitigation: one op library, VM first, type breadth in A6, CPU fallback through gating, specialization optional.
2. **Everything outside Apple is unvalidated.** The Vulkan code is portable by design, but memory-model, subgroup and driver behaviour on other vendors is unknown. Mitigation: devcaps probe, lavapipe VM, honest labelling.
3. **MoltenVK dependency.** Regressions or translation bugs on the only validated platform. Mitigation: pin the version; Slang can emit MSL if MoltenVK blocks us (behind `gpu_vk.c`).

**Others:** the A2 to A4 window leaves the branch without join/aggregate offload; watchdog timeouts on long dispatches (bounded dispatch size); devices with a very small `maxStorageBufferRange` fall below the base tier; hard-fork maintenance.

## Assumptions to confirm (defaults used unless you object)

1. `master` stays CUDA and untouched; all work is on `vulkan-gpu`.
2. Linux and non-Apple vendors are "designed for, not tested"; the optional lavapipe VM track is used for host-code portability.
3. Desktop-tier shims are implemented after A3 (milestone A3b) as an experimental, unvalidated tier, tested on lavapipe only.
4. `libslang` is an optional `dlopen`ed runtime dependency; servers without it run the VM only.
5. Minimums: Vulkan 1.2 plus the needed extensions, macOS 14 or later, PostgreSQL 15-19 kept.
6. The 1.5x/2x-versus-CUDA criterion is recorded but not a gate on this branch.

## Verification (how each stage is checked end to end)

- **Correctness:** the existing `pg_regress` schedule runs with `pg_strom.enabled=on` against the CPU result of the same query (`pg_strom.enabled=off`); `test/Makefile` gets a `vulkan_schedule` that grows per milestone, with `pending_schedule` holding what is not ported yet. Plan-shape tests confirm offload or correct gating.
- **Kernel level:** `test/vk_unit/` standalone harness (no PostgreSQL) runs one kernel or opcode on synthetic buffers against a C reference; the spike harnesses under `spike/vulkan-apple/` are its starting point.
- **Fuzzing and soak:** random qual expressions compared across CPU, VM and specialized kernels; concurrency soak for the lock-free hash.
- **Performance:** each of A2-A4 compares against PostgreSQL's own parallel plan on the same Mac and table (spike baseline: 163 ms with 8 workers for 16.8M rows; the realistic end-to-end GPU figure today is about 28 ms specialized and 56 ms VM including the page copy).
- **Process model:** kill the service mid-query and check the backend errors cleanly and the query succeeds on the CPU.

## Critical files

`src/gpu_service.c`, `src/relscan.c`, `src/executor.c`, `src/xpu_common.h`, `src/xpu_opcodes.h`, `src/codegen.c`, `src/cuda_gpupreagg.cu`, `src/cuda_gpujoin.cu`, `src/gpu_device.c`, `src/gpu_cache.c`, `src/extra.c` (deleted), `src/Makefile`, `src/Makefile.cuda` (deleted), `test/Makefile`.

## Kernel source language: Slang (carried over, decision unchanged)

- Slang has a real preprocessor, so `xpu_opcodes.h` can be included unchanged.
- It supports generics and interfaces, which suit the type library.
- It emits SPIR-V and also MSL, which keeps a native Metal path open if MoltenVK limits block us.
- Rejected: GLSL (no generics, macro-heavy), HLSL/DXC (no Metal path), clspv (poor pointer model, thin maintenance), and CUDA-to-SPIR-V translators such as chipStar/ZLUDA (they emit OpenCL-flavoured SPIR-V, which Vulkan and MoltenVK reject).

**Avoiding two diverging code bases.** The kernel bodies must differ, so the shared contract is the ABI:
1. `xpu_opcodes.h` is the single opcode source for `codegen.c`, CUDA and Slang.
2. A new `src/xpu_layout.def` (X-macro of fields and offsets) generates static asserts for the C structs and `src/vulkan/xpu_layout.slang` offset accessors.
3. One test suite and oracle (CPU PostgreSQL) for the single Vulkan backend.
4. Pure scalar helpers go in a shared `xpu_scalar_ops.inc`, written in the common C/Slang subset.

## Kernel ABI and build (carried over; superseded where the plan above says otherwise)

- **Memory.** Use VMA, with a pool per existing `gpuMemoryPool`.
  - Apple: unified memory, so every buffer is device-local and host-visible, persistently mapped. Prefetch is a no-op.
  - NVIDIA: a host-visible staging ring feeds device-local buffers via `vkCmdCopyBuffer`. Read-back happens on demand. ReBAR heaps are used when present.
  - Buffer growth: the kernel reports overflow, the host reallocates and relaunches, reusing the existing suspend/resume logic.
- **Shader build.** `Makefile.vulkan` runs `slangc -target spirv` per entry point. The entry points are packed into `pgstrom-vulkan-<slangc-ver>-<md5>.spvpack`, mirroring the fatbin naming and runtime rebuild logic (`gpu_service.c:640-915`). A `VkPipelineCache` is persisted under `$PGDATA/pg_strom/`.
- **Kernel ABI.** One descriptor set with fixed bindings (session and program, task state, `kds_src`, `kds_dst`, inner/extra, scratch arena, error buffer), plus push constants of at most 128 bytes. Buffers are `RWByteAddressBuffer`, offset-based, with 64-bit loads done as two 32-bit loads. BDA is not required in M1 and is gated by `devcaps.BDA` (it works on MoltenVK per the spike).
- **Subgroups.** Rewrite the stair-sum and scan primitives (`cuda_gpuscan.cu:33-76`, `cuda_common.h:26-32`) with `WaveActive*`/`WavePrefixSum`. Request size 32 via `VK_EXT_subgroup_size_control` where available, with a shared-memory scan fallback. Workgroup size is a specialization constant (default 256), capped by device limits.

## Appendix: spike results (Apple Silicon, M1 Pro, MoltenVK 1.2.9, macOS 27)

Everything in this appendix was measured on one M1 Pro (16 GB) with code under `spike/vulkan-apple/` (untracked). Four conclusions were corrected by later verification; the corrections are marked in place. Milestone names inside the appendix (M0, M1a, M1b, M1c, M2, M3, M4, M5) belong to the earlier plan, where CUDA was kept; the current milestones are A0-A9 above (roughly: M1a/M1b -> A2, M1c and the VM/specialization work -> A6, M2 -> A3, M3 -> A4, M4 -> A6, M5 -> A7).

### Apple capability spike (done)

Code: `spike/vulkan-apple/` (`make run`). Every row below was either exercised by a running kernel or read from `vulkaninfo`; the source is marked.

| Capability | Result | Source |
|---|---|---|
| Unaligned byte-wise heap-row deform (8-bit storage, scalar layout) | works | kernel, verified vs CPU on 16M rows |
| int64 arithmetic, compare, shifts | works | kernel |
| Subgroup ballot compaction + one 32-bit `atomicAdd` per subgroup | works, exact index set matches CPU | kernel |
| 64-bit sum from 32-bit pieces (16-bit limb `subgroupAdd`, lo/hi atomics with carry) | works, exact match | kernel |
| Buffer device address (raw pointer in push constants) | **works** | kernel |
| Unified memory: `DEVICE_LOCAL\|HOST_VISIBLE\|HOST_COHERENT`, persistently mapped, no staging | works | kernel; 1 heap of 16 GiB |
| int64 `subgroupAdd` | **rejected by the Metal shader compiler**, although `shaderSubgroupExtendedTypes=true` | kernel |
| 64-bit buffer/shared atomics | **not available** (`shaderBufferInt64Atomics=false`, `shaderSharedInt64Atomics=false`) | vulkaninfo |
| FP64 | **not available** (`shaderFloat64=false`) | vulkaninfo |
| float32 atomic add | available | vulkaninfo, not run |
| Subgroup size / ops | 32 fixed (min 4, max 32); basic, vote, arithmetic, ballot, shuffle, shuffle-relative, quad | vulkaninfo |
| Limits | shared memory 32 KiB, 1024 invocations per workgroup, push constants 4096 B, storage buffer range 4 GiB, 31 storage buffers per stage | vulkaninfo |
| Extensions | `VK_EXT_external_memory_host`, `VK_EXT_subgroup_size_control`, `VK_KHR_portability_subset` (must be enabled), `VK_EXT_shader_atomic_float` | vulkaninfo |

Timing from the same run (16M rows, 218 MB): the GPU scan took about 2 ms, against 35-45 ms for one CPU thread. That is hot-cache, best-of-5, versus a single-threaded loop. It shows the kernel runs at memory-bandwidth speed, not a real speedup claim. A fair comparison needs the real PostgreSQL path (M1b).

**What this changes in the plan**
- **BDA works on MoltenVK.** The earlier worry ("poorly supported") was wrong on this device. BDA stays optional for M1 (offset ABI is still the portable default), but it is a viable fallback for pointer-heavy structures later. Keep it behind `devcaps.BDA`.
- **Feature flags over-report.** `shaderSubgroupExtendedTypes` claimed int64 subgroup ops, and Metal rejected them. `devcaps` must be filled by **probing**: at device init, compile and run a tiny kernel per capability, not just read `VkPhysicalDeviceFeatures`. Add `devcaps.INT64_SUBGROUP_ARITH` and default it off on Apple.
- **No 64-bit atomics on M1-class devices.** M2 aggregates (`sum`, `count`, `min`, `max` on 64-bit) cannot use CAS or native atomics. Sum works with the limb + carry technique above. `min`/`max` on 64-bit and any 128-bit accumulator still need a design (per-workgroup partials plus a second reduction pass is the leading option). This is now the main technical risk for M2.
- **M1 scope is unblocked.** Everything GpuScan needs (deform, int/float32/int64 quals, ballot compaction, error buffer) ran on the hardware. float8 remains gated off, as planned.

### Slang port of the scan kernel (done)

`spike/vulkan-apple/shaders/scan.slang` is a line-for-line port of `scan.comp` (Slang 2026.20, official macOS aarch64 release). It compiled on the first attempt with `slangc -target spirv -entry main -stage compute`, passes `spirv-val --target-env vulkan1.2`, and on the M1 Pro produces the same 1,078,674 matches, with an identical index set to the CPU reference. Across five runs both builds landed between 1.6 and 2.2 ms (best-of-5 each) with no consistent difference. That is too coarse to rank them: at this size the kernel is bandwidth-bound and the timer includes submit and fence overhead.

- The port is smaller: HLSL-style wave intrinsics (`WaveActiveCountBits`, `WavePrefixCountBits`, `WaveIsFirstLane`, `WaveReadLaneFirst`) replace the GLSL extension boilerplate, and `StructuredBuffer<uint8_t>` and `int64_t` map straight to SPIR-V 8-bit storage and Int64.
- Slang requests `UniformAndStorageBuffer8BitAccess`, where glslang requested `StorageBuffer8BitAccess`. Device creation must enable both 8-bit storage features; `devcaps` should probe for them.
- This validates the toolchain for one small kernel only. It does **not** yet cover the parts of the plan that carry the real risk for Slang: including `xpu_opcodes.h` unchanged, generics or interfaces for the type library, the opcode-switch VM, and compile time on a large shader set. Treat the choice of Slang as supported, not settled, until M1a has a representative slice of the VM compiling.

### Opcode-switch VM prototype in Slang (done)

Code: `spike/vulkan-apple/` (`make run-vm`): `shaders/vm.slang` (the VM), `vmspike.c` (tree linearizer, CPU references, checks), `gen_opcodes.sh`. Measured on the M1 Pro, 16M rows of 21-byte unaligned rows.

**What works**
- **`xpu_opcodes.h` is used unchanged by Slang.** It compiles, and 670 opcode entries expand into the enums. Only a warning appears: the header leaves three helper macros (`__FUNC_OPCODE`, `__FUNC_LOCALE_OPCODE`, `__FUNC_RECURSIVE_OPCODE`) defined at the end, so including it a second time redefines them. Adding three `#undef` lines at the end of the header would silence that. It is not applied.
- **One source of numbering.** The `TypeOpCode`/`FuncOpCode` enums live in `xpu_common.h`, not in `xpu_opcodes.h`. `gen_opcodes.sh` extracts them verbatim for both the C harness and Slang, and the VM dispatches correctly on the values the C side emits (for example `int4pl`=85, `int4div`=160). A real implementation should move the enum into a shared header (an M0 refactor) instead of extracting it with awk.
- **No recursion, no function pointers.** The host flattens a tree into a register program `{op, dst, a, b, imm_lo, imm_hi}`, and the shader runs a `switch(op)` loop. `AND`/`OR` become accumulate-and-jump steps with SQL three-valued logic. `VarExpr` selects the heap-read by `TypeOpCode`, which replaces the `xpu_datum_operators` vtable.
- **Correct on all 5 tests**, each checked against an independently written CPU reference (exact index sets for 2.6M and 7.4M matches): arithmetic plus compare under `AND`; `NULL` tests with `OR`/`NOT`; a short-circuit case where `NOT(d = 0) AND a / d > 3` must not raise the zero-divide that 150k rows would cause; and two cases that must report the right error code (`int4div` 160, `int4pl` 85) at a row that really errors.
- **`int8pl` overflow is detectable without int128**, using a sign test on int64. The CUDA templates use `int128_t` even for `int8` add and subtract. `int8mul` was not implemented, so its overflow check is still open.

**What does not work well: speed**

| Kernel (same predicate E1, 16M rows) | Time |
|---|---|
| Hand-written kernel | 3.0 ms |
| Opcode-switch VM | 31 ms (10.5x slower) |
| One CPU thread | 39 ms |

The VM is compute-bound: it moves about 11 GB/s on a chip with several times that bandwidth. I tried the obvious fixes, and none helped much: a 4-entry register file with no zero-init (26.7 ms), and the program in a uniform buffer (33.9 ms, no gain). That points at interpretation overhead itself (per-instruction decode and dispatch, and int64 arithmetic for every register op), not at instruction fetch or register spilling. This is one expression on one GPU, so treat 10x as an order of magnitude, not a constant.

**Pipeline creation cost (cold = fresh shader text, so Metal cannot reuse a cached build)**

| Kernel | Cold | Warm |
|---|---|---|
| VM | about 105 ms | about 3 ms |
| Small specialized kernel | about 23 ms | about 0.5 ms |
| `slangc` to SPIR-V (both) | about 0.15 s | n/a |

**What this changes in the plan**
- **The VM alone is not enough on Apple.** It is a good correctness baseline and costs one pipeline build, but a 10x gap on an expression-heavy qual removes most of the GPU's advantage over parallel CPU workers.
- **Add per-expression specialization as a follow-on (M1c).** Generate a kernel from the linearized program, compile it, and cache the pipeline by program hash. The M1c experiment measured what it buys: generated kernels ran 8-13x faster than the VM and within 1.24x of the hand-written kernel, at a cold cost of roughly 0.2-6 s depending on expression size (see below). The planner needs a rows-times-size threshold. The M1c experiment below confirmed this with generated Slang text compiled by the `slangc` CLI. Whether to embed `libslang` or emit SPIR-V directly instead was not tested.
- **The VM stays as the fallback and the correctness oracle** for the specialized path, and for expressions too large or too cold to be worth compiling.
- **Fixed-width registers only cover fixed-width types.** Registers are 16 bytes (int64 value plus null flag), which fits bool, int2/4/8 and float4. text, numeric, jsonb and other varlena types need arena offsets and variable-length results. The prototype does not address that, and it is the largest remaining unknown for M4.

**Not covered:** 12 of the roughly 670 function opcodes are implemented; NaN ordering for floats; `CASE`, `COALESCE`, scalar-array ops; projection and `LoadVars`; any type wider than 8 bytes; error-reporting parity with CUDA's `kern_errorbuf`; and behaviour on NVIDIA.

### M1c experiment: specialized kernels generated from the expression trees (done)

Code: `generate`/`specialize` in `spike/vulkan-apple/vmspike.c` (about 150 lines of C emitting Slang text), plus `shaders/common.slang` shared with the VM. The generator walks the same trees as the linearizer and emits one kernel per expression: typed locals (`int`, `int64_t`, `float`, `bool`) with explicit NULL flags, no register file, and each `AND`/`OR` as a `do { ... break; } while (false);` block so a short-circuit is a structured jump. It covers the same 12 opcodes as the VM. The text is compiled with the `slangc` CLI, cached by a hash of the source, and loaded as a pipeline. `make run-vm` runs everything; `SPEC_COLD=1` forces fresh shader text so neither `slangc` nor Metal can reuse a cached build.

**Correctness.** All five expressions pass the same checks as the VM: exact match sets against the CPU reference, the right error opcode at a row that really errors, and the short-circuit case raising no zero-divide. The specialized kernel is therefore a drop-in for the VM on this opcode subset.

**Speed (16.8M rows, data already resident, best of 5)**

| Expression | VM | Specialized | Hand-written | VM / specialized |
|---|---|---|---|---|
| E1 (4 terms, 20 insns) | 31.4 ms | 3.7 ms | 3.0 ms | 8.6x |
| E2 (`NULL` tests, `OR`, `NOT`) | 36.4 ms | 2.8 ms | n/a | 13x |
| E5 (short-circuit) | 23.7 ms | 3.1 ms | n/a | 7.7x |
| N=4 terms (27 nodes) | 52.1 ms | 6.0 ms | n/a | 8.7x |
| N=16 (111 nodes) | 215.5 ms | 20.5 ms | n/a | 10.5x |
| N=64 (447 nodes) | 886.6 ms | 90.3 ms | n/a | 9.8x |
| N=256 (1,791 nodes) | 3,582.6 ms | 376.2 ms | n/a | 9.5x |

- **The generated E1 kernel is 1.24x the hand-written one**, inside the M1c target of 1.5x. Specialization recovers essentially all of the 10x.
- **The gap is a steady 8-13x at every size**, so it is not an artifact of small expressions. Large expressions are compute-bound in both forms, and the specialized form is about 9.5x faster per node throughout.

**Cost of specializing (cold, fresh shader text)**

| Expression | Nodes | `slangc` | Metal pipeline | Total |
|---|---|---|---|---|
| E1-E5 | 12-30 | about 140-150 ms | about 24-27 ms (E1: 76 ms, includes first-use start-up) | about 170-225 ms |
| N=4 | 27 | 147 ms | 39 ms | 186 ms |
| N=16 | 111 | 172 ms | 96 ms | 268 ms |
| N=64 | 447 | 325 ms | 517 ms | 842 ms |
| N=256 | 1,791 | 1,630 ms | 4,796 ms | 6.4 s |

Two runs of the scaling cases agreed within a few percent. Generating the source takes under 1 ms even at N=256. With a warm Metal cache, pipeline creation is about 1 ms. `slangc` runs as a separate process, so about 140 ms of its time is fixed start-up and compile overhead; whether `libslang` or direct SPIR-V emission would cut that was not tested.

**Break-even** (rows scanned before the compile cost is repaid, from the measurements above)

| Expression | Compile | Saved per 16.8M-row scan | Pays back after |
|---|---|---|---|
| E1 | about 175-225 ms | 28 ms | about 100-135M rows |
| N=4 | 186 ms | 46 ms | about 68M rows |
| N=16 | 268 ms | 195 ms | about 23M rows |
| N=64 | 842 ms | 796 ms | about 18M rows |
| N=256 | 6.4 s | 3.2 s | about 34M rows |

**What this means for the plan**
- **Specialization is the fast path and the VM is the fallback (revised by the real-page measurement below: run the VM first, specialize when it pays back).** On a one-off scan below roughly 20-130M rows (depending on expression size) the compile cost exceeds the saving, so the VM is faster end to end. Above that, or whenever the same expression runs repeatedly and the pipeline is cached by hash, specialization wins by about 9x. A planner threshold on estimated rows times expression size is needed. A persistent on-disk cache (the Metal cache already behaves this way for identical shader text) removes the cost for repeated expressions.
- **Metal compile time is superlinear**: 4x the nodes (N=64 to N=256) cost 9x the pipeline time. Very large predicates should stay on the VM. Removing duplicate loads in the generator (each `Var` currently reloads its column) might shrink the kernel and its compile time; that was not tried.
- **The VM penalty only shows when data is resident and the kernel is compute-bound.** These rows are 21 bytes. With wide rows, or when heap pages have to be fetched and moved first, scan I/O dominates and the VM's cost is partly hidden. That moves break-even later, not earlier. The real PostgreSQL path (M1b) must be measured before fixing the threshold.
- **Not tested:** `libslang` embedding, direct SPIR-V emission, compile in a background thread while the VM runs the first scan, error-path performance (E4 is slow in both forms because millions of rows contend on one error word), and anything on NVIDIA.

### Real PostgreSQL scan path measurement (done)

> **Correction (verified in code, `src/relscan.c:439-500`, `src/executor.c:171-195`):** PG-Strom copies every page under `BUFFER_LOCK_SHARE` into the command buffer and performs the tuple-visibility check **on the CPU** (invisible tuples are marked unused in the copy) before anything reaches the GPU. So the "visibility checks not implemented" limitation below does not apply to the real kernel path. The end-to-end figures that follow use kernel time plus a 21 ms page copy; the later zero-copy section removes only the *service-side* copy, not this backend copy.

Code: `spike/vulkan-apple/` (`pg_setup.sh`, `pg_baseline.sh`, `heapspike.c`, `shaders/heap_common.slang`; `make run-heap`). A throwaway PostgreSQL 17.11 cluster (Homebrew, under `/tmp`, no TCP, port 54329) holds `t(a int4, b int8, f float4, d int4)` with 16,777,216 rows (823 MB, 105,381 pages, same distributions and NULL rates as the earlier row tests). It was `VACUUM (FREEZE)`d and the visibility map confirms all 105,381 pages are all-visible and all-frozen, which is what lets the kernel skip per-tuple visibility checks. The harness copies the relation file's raw 8 KB pages into a GPU buffer and scans them with a Slang kernel that parses the page header, the line pointers and the tuple headers, deforms the four columns (null bitmap, 8-byte alignment of `int8`) and evaluates the predicate. The page, line-pointer and tuple-header offsets were checked against the installed PostgreSQL 17 headers.

**Correctness.** Every kernel run matched the SQL oracle exactly, on two predicates: the match count and the sum of `blkno * 512 + offnum` over all matches (derived from `ctid` in SQL), for 2,589,788 (E1) and 7,405,661 (E2) matches.

**Speed, same table and predicate E1 (16.8M rows, table fully in shared buffers, best of 5)**

| Path | Time | Notes |
|---|---|---|
| PostgreSQL, no parallelism | 909 ms | `SELECT count(*)`, JIT off |
| PostgreSQL, 8 workers | 163 ms | 6 performance + 2 efficiency cores; 6 workers: 171 ms |
| GPU VM, kernel only | 34.8 ms | 41.4 ms for E2 |
| GPU specialized, kernel only | 7.2 ms | 5.6 ms for E2; identical to the hand-written kernel |
| Page transfer into the GPU buffer | 21 ms | `memcpy` from ordinary memory, about 42 GB/s |
| Reading the pages from the OS cache with `pread` | 193 ms | what the service would pay if it read files itself |

End to end per scan (kernel + page copy): **VM about 56 ms** (2.9x faster than 8-worker PostgreSQL, 16x faster than one process); **specialized about 28 ms** (5.8x and 32x). Neither the workgroup size (64-256 threads) nor the workgroup count (2,048-65,535) changed any kernel time by more than a few percent.

**What this changes in the plan**
- **The hand-written and specialized kernels are memory-bandwidth-bound**: 823 MB in 7.2 ms is about 120 GB/s, the same rate as the row-format kernel earlier. They cannot go faster on this chip. The VM is compute-bound at about 25 GB/s.
- **The page copy is now the dominant cost of the fast path**: 21 ms of the 28 ms specialized total. Avoiding the copy (importing host memory into Vulkan with `VK_EXT_external_memory_host`) would matter more than any further kernel work; the zero-copy test below confirms it works on this device. The service must also consume pages from shared memory, not read relation files, since `pread` cost 9x more than the copy.
- **The VM penalty is smaller end to end than in the kernel-only tests**: 56 ms vs 28 ms (2x), because the copy is common to both. The earlier concern that I/O would hide part of the VM cost is confirmed: roughly a quarter of the 10x is hidden once the copy is included.
- **Specialization threshold, revised.** The VM alone already beats parallel PostgreSQL by about 3x. A cold specialized kernel costs 170-240 ms to build here (`slangc` 140-170 ms plus pipeline 26-70 ms), against about 28 ms saved per 16.8M-row scan, so it pays back only after roughly 7-9 scans of this size, about 115-150M rows (6-7 GB of heap), or immediately if the pipeline is already cached. So: **run the VM first, specialize in the background or only for expressions that recur or for scans well above about 100M rows.** The earlier "specialize by default" recommendation is withdrawn for one-off scans.
- **M1b target, restated with a measured baseline.** The plan compared against "the CPU"; the real baseline is PostgreSQL's own parallel seq scan, 163 ms for this table on 8 workers. A kernel-plus-copy time below that is the bar; the VM already clears it by 2.9x.

**Not measured, and likely to narrow the margin**
- **Integration overhead:** moving pages from PostgreSQL's shared buffers to the service, launching per-chunk kernels, and returning results. The GPU side here returns only match ids; PostgreSQL's side includes the aggregate. A `GpuPreAgg` that aggregates on the device would avoid shipping matches, but that is M2.
- **Visibility checks:** only a fully frozen, all-visible table was tested. Pages that are not all-visible need per-tuple checks, which were not implemented.
- **Cold data:** the table was in shared buffers and the OS cache. A cold scan is dominated by storage in both systems.
- **Row shape:** 4 fixed-width columns, no varlena, no TOAST, about 160 rows per page. Wider or text-heavy rows change the balance, and the VM cannot represent text yet.
- **Contention:** PostgreSQL workers and the GPU shared the same memory system but were never run at the same time.
- **Unexplained:** the specialized E2 kernel (5.6 ms) is faster than E1 (7.2 ms) although both deform the same bytes; I did not investigate.
- Single machine (M1 Pro, 16 GB), one PostgreSQL version, one run of setup.

### Zero-copy import of host memory (done)

> **Correction:** the end-to-end numbers in "What this changes" below (about 7.2 ms specialized and 35 ms VM) leave out the mandatory backend copy under the share lock (about 21 ms per 823 MB). Realistic figures on this Mac are about **28 ms specialized and 56 ms VM**, against 163 ms for 8-worker PostgreSQL (5.8x and 2.9x faster). Zero-copy applies to unified memory only (Apple, integrated GPUs); on discrete GPUs every byte crosses PCIe. The statement that the service should import PostgreSQL's memory directly is replaced by the per-session shared-memory ingest ring in the plan above.

Code: `spike/vulkan-apple/` (`vk_buf_import` in `vkutil.h`; `HEAP_SRC` modes in `heapspike.c`; `zc_test.sh`, `zc_cold.sh`). `VK_EXT_external_memory_host` wraps existing host memory as a Vulkan storage buffer, so the kernel reads it in place. The device reports `minImportedHostPointerAlignment` = 16 KB, and the imported memory type is device-local, host-visible, coherent and cached (flags 0xf), the same type as normal allocations. The throwaway cluster was recreated for this and removed afterwards. Same table shape, 16.8M rows, 863 MB.

| Page source (all imported with no copy) | Import time | Kernel (hand-written E1) | Matches SQL oracle |
|---|---|---|---|
| Control: `memcpy` into a Vulkan buffer | 19 ms copy per scan | 7.2 ms | yes |
| Anonymous `mmap`, filled by `pread` | 0.1 ms | 7.2 ms | yes |
| POSIX shared memory (`shm_open`), filled by a different process, attached by name | 0.1 ms | 7.2 ms | yes |
| Relation file, `mmap` `MAP_SHARED` (the OS page cache) | 0.1 ms | 7.2 ms | yes |
| Relation file, `mmap` `MAP_PRIVATE` | 0.1 ms | 7.2 ms | yes |

For every source the VM and the specialized kernels also matched the oracle on E1 and E2. Imported memory is no slower for the kernel than a normal buffer.

**Coherence (does the GPU see CPU writes with no copy in between?)** After a scan, the CPU zeroed column `a` in 1,000 matching tuples through the mapping, and the GPU rescanned. Every imported source showed exactly 1,000 fewer matches (the GPU read the writes in place). For shared memory the writes went through a second mapping of the same region. The control (copied buffer) correctly showed the old count, so the test can detect a stale view. The `MAP_PRIVATE` case also showed the writes; I would not have predicted that for copy-on-write memory, and I have no explanation, so nothing should rely on it.

**Cold file pages.** A copy of the relation file written with `F_NOCACHE` had 0% of its pages in the OS cache (checked with `mincore`). With the GPU as the first reader, the scan was still correct (oracle matched), the first dispatch took **333 ms** against 7.3 ms afterwards, and residency was 100% when it finished. So the pages are brought in on first use at about storage speed (863 MB in roughly 325 ms, 2.6 GB/s), with no error. The mechanism (pre-faulted at submit, or faulted during execution) was not determined.

**What this changes in the plan**
- **The page copy disappears.** End to end per scan on warm memory is now the kernel time: **about 7.2 ms specialized and about 35 ms for the VM**, against 163 ms for 8-worker PostgreSQL (22x and 4.7x faster) and 909 ms for one process. The earlier figures that included a 21 ms copy (28 ms and 56 ms) no longer apply on Apple.
- **Specialization matters more again.** With the copy gone, the VM is about 5x slower end to end, not 2x. Payback is unchanged (compile 170-240 ms against about 28 ms saved per scan, so about 7-9 scans), but the benefit per scan is 5x instead of 2x.
- **The service can map shared memory or the relation files directly.** On Apple, M1b should import PostgreSQL's memory (or mapped files) rather than copy pages.

**Not tested, and the real design risks**
- **PostgreSQL's actual `shared_buffers`.** Only my own mappings were imported. In the real design the GPU service is a background worker forked from the postmaster, so it already has `shared_buffers` mapped and needs no named region. PostgreSQL aligns `BufferBlocks` to 4 KB, so the import range would have to start at the next 16 KB boundary and skip at most one leading buffer. Importing a range above 4 GB (`maxStorageBufferRange`) would need several imports. Neither was tried.
- **Concurrent writers.** PostgreSQL backends modify buffers under content locks that a GPU kernel cannot take. A scan over live buffers can see torn pages unless the buffers are pinned and protected, or the pages are snapshotted. My coherence test only wrote between scans. This is the main correctness risk of in-place reads.
- **Very cold or very large files.** A first dispatch that has to page in several gigabytes could run into a GPU command-buffer time limit; I only tested 863 MB (333 ms).
- **Memory pressure.** Whether imported pages are pinned in RAM for the life of the buffer was not checked.
- **NVIDIA/Linux:** this is `VK_EXT_external_memory_host` on MoltenVK. The CUDA path uses `cuMemHostRegister`, a different mechanism, and was not looked at.

### Live shared_buffers with a concurrent writer (blocked on macOS; superseded, platform findings still valid)

> **Superseded:** PG-Strom never reads live `shared_buffers` from the GPU (see the correction under "Real PostgreSQL scan path measurement"), so the torn-read question this section leaves open does not arise in the real design, and the `zcscan/` module and `heap_live.slang` kernel are not needed. What stays valid: a forked process cannot use Metal's shader compiler, the postmaster cannot pre-initialise Metal under PostgreSQL 17, and an exec'd helper cannot attach the anonymous buffer pool; together they are why the GPU service is an exec'd process on macOS. The "What this means for the plan" bullets below that talk about in-place reads of live buffers are moot.

Code: `spike/vulkan-apple/zcscan/` (a PostgreSQL module: pins a table's buffers, imports the whole `shared_buffers` array into Vulkan, scans three ways), `shaders/heap_live.slang` (a lock-free, bounds-checked scan that detects torn reads through a per-tuple invariant `b = 3a + d + 7` that every writer preserves, bad line pointers, bad tuple headers and bad page headers), `pg_setup_live.sh`, `live/writer.sql`, `live/reader.sql`, `forktest.c`, `forktest2.c`. Everything built and ran up to the first GPU dispatch inside a backend; no torn-read measurement could be taken.

**What worked.** PostgreSQL's real `shared_buffers` (an anonymous shared `mmap`, 1,199 MB, 153,600 buffers) was imported into Vulkan without error. `BufferBlocks` is only 4 KB-aligned, so 12,288 leading bytes had to be skipped to reach the 16 KB import alignment (one or two leading buffers cannot be scanned in place). The module compiled against the Homebrew PostgreSQL 17 headers with PGXS (`MODULE_big`; `SHLIB_LINK` is ignored for a plain `MODULES` build).

**What blocked the test (each reproduced; the first two without PostgreSQL)**
1. **A forked process cannot use Metal's shader compiler unless its parent already initialised Metal.** A backend, like any `fork()` without `exec`, fails with `Unable to reach MTLCompilerService ... Connection init failed at lookup with error 3 - No such process`. `forktest.c`: compiling a fresh shader in a normal process works (about 75 ms); forking first, with the parent never touching Vulkan, and compiling in the child is rejected with that error. A warm Metal cache did not help, and neither did `setsid()` (ruled out).
2. **The postmaster cannot pre-initialise Metal either.** Creating even a bare Vulkan instance makes the process multithreaded (`pthread_is_threaded_np()` goes from 0 to 1), and PostgreSQL 17 aborts: `FATAL: postmaster became multithreaded during startup`. So the obvious workaround (initialise in `shared_preload_libraries`) is impossible on this PostgreSQL.
3. **An exec'd helper cannot reach `shared_buffers`.** The buffer pool is anonymous memory that another process cannot attach. `shared_memory_type = sysv` would make it attachable, but `kern.sysv.shmmax` here is 4 MB (4,194,304), so a 1.2 GB pool needs root to raise the limit.

**What this means for the plan**
- **On macOS the GPU service must be a separately executed process, not a PostgreSQL background worker.** The Linux design (a bgworker forked from the postmaster) does not carry over. This changes M1b and the macOS portability track (P): the service is spawned with `exec` and talks to backends over the existing socket protocol.
- **Reading live buffers in place is not available on macOS with stock PostgreSQL.** The only way the GPU service sees PostgreSQL pages is a copy that a backend makes into a named shared region (POSIX shm, which the zero-copy test above showed imports fine). The safe protocol for that copy is a shared content lock per buffer. So the practical macOS design is: backend copies pages under lock into shared memory, and the GPU scans that region in place. The copy (about 21 ms per 823 MB measured earlier, plus locking) comes back, but it stays on the CPU side and the GPU still reads without a second copy.
- **The torn-read risk for in-place reads of live buffers remains untested.** It still applies to the Linux design, where a bgworker can map `shared_buffers`. The prepared kernel and extension are the starting point, but they need a Linux machine (or a way around blockers 1-3).
- **Pinning:** a backend that pins buffers stops PostgreSQL pruning and defragmenting those pages (cleanup locks wait for all pins). An external helper cannot pin buffers at all. Neither effect was measured.

**Not covered by the spike (still open)**
- M2 or newer Apple GPUs (the plan assumed they may add 64-bit atomic min/max; unverified).
- Subgroup-size behavior under load (`VK_EXT_subgroup_size_control`), pipeline cache persistence, and multi-threaded queue submission.
- Anything on NVIDIA/Linux (M1a).
