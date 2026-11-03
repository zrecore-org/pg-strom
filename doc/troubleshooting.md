# Troubleshooting (Vulkan build)

Everything below is checkable in the source named next to it. Where the behavior is unknown or not
tested, the text says so. The server log is the main tool: the probe, the launcher and the service all
write to it (the service's `stderr` is the postmaster's log).

## 1. Is the GPU service up?

```sql
SELECT * FROM pgstrom.gpu_service_state();
```

| `state` | Meaning | What to do |
|---|---|---|
| `unavailable` | No GPU service is configured: the start-up probe found no usable device, the service binary is missing or not executable, or the probe failed or timed out. All other columns are NULL. PostgreSQL runs normally, without GPU offload, and the GPU parameters of [parameters.md](parameters.md) (sections 4 to 6) are not registered. | Read the first lines of the server log after start-up (section 2). |
| `starting` | The launcher started the service (`pid`, `generation`); it is opening the device and running its self-test. | Wait. If it never becomes `ready`, see `last_error` and the log. |
| `ready` | The service accepts sessions. | `SELECT pgstrom.gpu_service_ping();` connects to it and returns `ok pid=... generation=... device=... tier=...`. |
| `failed` | The service exited or could not be started; `last_error` says `service (pid N) exited with status S`, `killed by signal N` or `could not start '<path>': <reason>`. The launcher starts it again after a back-off of 1 s that doubles up to 30 s (it starts over after a run longer than a minute). `restart_count` counts the restarts. | Look at the log lines of the service just before the exit. |
| `stopped` | PostgreSQL is shutting down. | |

`generation` increments with every service start; a changed `generation` tells you the service was
restarted. Killing the service with `kill -9 <pid>` is safe: the launcher starts a new one, queries that
were running fail, later queries work (`service_test.sh` checks this), and PostgreSQL is not restarted.

The planner offers a GPU plan **only while the state is `ready`** (`gpuserv_ready_accept()` in the
path creation code of GpuScan, GpuJoin and GpuPreAgg). A plan made while the service was ready and
executed (or a prepared statement re-executed) after it died fails with `the GPU service is not ready`;
run it again once `gpu_service_state()` says `ready`.

## 2. The start-up probe and what its log means

The postmaster runs `pgstrom-gpuserv --probe` while loading the module (the server does not finish
starting before it returns, at most `pg_strom.gpuserv_probe_timeout` ms). Log lines, in order:

| Log line | Meaning |
|---|---|
| `PG-Strom version 7.0.0 built for PostgreSQL 17 (githash: ...)` | The module loaded. |
| `PG-Strom: GPU service '<path>' is not available (<errno text>); running without GPU` | `pg_strom.gpuserv_path` does not point to an executable file. Was `make install` run with `WITH_GPUSERV` not `0`, and did the build print "pgstrom-gpuserv was not built (needs slangc and the Vulkan headers)"? |
| `PG-Strom: GPU probe did not finish within N ms, killed it` | A driver hang (or a very slow CPU device). Raise `pg_strom.gpuserv_probe_timeout` or fix the driver. |
| `PG-Strom: GPU probe '<path> --probe' failed (status 0x...)` | The probe crashed. Run it by hand (below). |
| `PG-Strom: Vulkan device 0: <name> (<driver>), tier <tier> - <reason>` | One line per Vulkan device found, with its tier and the reason when it is not the best tier. |
| `PG-Strom: no usable Vulkan device; running without GPU` | Every device is tier `none` (or not allowed by `pg_strom.gpu_visible_devices`). |
| `PG-Strom: GPU0 <name> (Vulkan, tier <tier>), RAM <n>MB` | A device was selected; GPU GUCs are registered. |
| `PG-Strom: GPU service started (pid N, generation G)` | The launcher started the service. |
| `pgstrom-gpuserv[N]: ready: device 0 (...), tier base, socket ...` | The service finished its self-test (`VM self-test ok: ...`) and listens. |

Run the probe yourself, as the user that runs PostgreSQL:

```
<pkglibdir>/pgstrom-gpuserv --probe --verbose            # add --allow-cpu for a CPU device
```

`--verbose` prints one line per probe kernel (`ok`, `FAIL (...): <reason>`, or `skip` when the device does not report the capability). The same facts are in SQL:

```sql
SELECT dev_id, device_name, tier, tier_reason, device_type, caps, reported_caps FROM pgstrom.gpu_devcaps();
```

`reported_caps` is what the driver claims, `caps` what the probe kernels proved; a capability in the first
and not in the second is a driver that over-reports.

### MoltenVK "[mvk-error] ... Shader library compile failed" lines at start-up are expected

On Apple Silicon the log contains, right after the version line:

```
[mvk-error] VK_ERROR_INITIALIZATION_FAILED: Shader library compile failed (Error code 3):
program_source:19:46: error: no matching function for call to 'simd_sum'
    ... simd_sum((long(ulong(gl_GlobalInvocationID.x + 1u)) * 4294967297l) - 1000000000000l);
[mvk-error] VK_ERROR_INVALID_SHADER_NV: Compute shader function could not be compiled into pipeline. See previous logged error.
```

This is a probe kernel failing on purpose-built grounds: it tests 64-bit subgroup arithmetic
(`subgroup_arith64`), which MoltenVK reports as available but the Metal shader compiler rejects (plan,
Apple capability spike). The probe records the capability as missing and carries on; the device is
still tier `base` and works. (The message names `simd_sum` on a `long`; it is not about the desktop
tier, which asks for 64-bit atomics, fp64 and the memory model.) If the log shows `[mvk-error]` lines
**after** the service is ready, during queries, that is not this case: a kernel failed to compile or run.

## 3. No device found

| Symptom | Likely cause and check |
|---|---|
| `tier none`, reason `CPU device (use --allow-cpu)` | The only device is a CPU implementation (lavapipe). Set `pg_strom.gpu_allow_cpu_device = on` in `postgresql.conf` and restart. Such a device is for testing; it is never faster than the CPU. |
| Zero devices listed on Linux | No Vulkan driver (ICD) installed or visible: install the Mesa or vendor driver (`mesa-vulkan-drivers`), check with `vulkaninfo --summary` as the PostgreSQL user, and check access to `/dev/dri/renderD*` (the `render` group is the usual requirement; general Vulkan practice, not tested on this branch). |
| Zero devices on macOS | The Vulkan loader did not find MoltenVK's ICD file. The validated machine uses the LunarG SDK system install (`/usr/local/lib`, `/usr/local/share/vulkan/icd.d`). With Homebrew's `molten-vk` the loader needs to find its `MoltenVK_icd.json`; the standard Vulkan environment variables (`VK_DRIVER_FILES`, `VK_ICD_FILENAMES`) can point at it, set in the postmaster's environment. Not verified with Homebrew by the author. |
| `tier none`, reason `missing capability <name>` | The device lacks something every device needs (int64, 8-bit storage, subgroup ballot, 32-bit subgroup arithmetic). Nothing to configure. |
| `tier none`, `missing capability host_import or unified_memory` | The device can neither import host memory nor share memory with the CPU. Discrete GPUs without `VK_EXT_external_memory_host` end here: the "discrete GPU" memory path in the plan is designed, not implemented. |
| `tier none`, `device open failed: ...` | The Vulkan device could not be created; the text after the colon is the driver's reason. |
| Device present but not used | `pg_strom.gpu_visible_devices` excludes it, or an earlier usable device was selected (only one device is used). |

## 4. A query does not use the GPU

1. `SHOW pg_strom.enabled;` (and `enable_gpuscan`, `enable_gpupreagg`, `enable_gpujoin`). If `SHOW` says
   the parameter is unrecognized, no device was found (section 1).
2. Is the service `ready` right now?
3. Is the table big enough? The planner compares costs; for a test, `SET pg_strom.gpu_setup_cost = 0;
   SET pg_strom.gpu_tuple_cost = 0; SET max_parallel_workers_per_gather = 0;` (what `scan_test.sh` does).
4. Is everything in the query something the GPU can do? One unsupported function (`float8`
   comparison, `upper()`, `CASE`, a `LIKE` with `_`, ...) keeps that qual or column on the CPU; a
   qual that stays on the CPU is typically visible as a `Filter:` line on the Custom Scan. See [features.md](features.md).
5. GROUP BY: more than `pg_strom.gpupreagg_max_groups` (16) estimated groups stays on PostgreSQL.
6. Joins: only hash joins, up to four inner relations, integer/bool/temporal keys; an outer join whose
   WHERE clause tests the null-extended side stays on PostgreSQL.
7. Arrow tables: every referenced column must have a supported Arrow type, otherwise the plain
   Arrow_Fdw scan is used.

## 5. Reading `EXPLAIN`

```
Custom Scan (GpuScan) on t
  GPU Projection: id, v
  GPU Scan Quals: ((v < 10) AND ((w + 1) > 5)) [plan: 16800000 -> 150000, exec: 16800000 -> 151234]
  Scan-Engine: VFS with GPU0; buffer=12, vfs=1234, ntuples=16800000
```

(Illustrative; the label set is from `src/executor.c`.) The plan node names are `GpuScan`, `GpuJoin` and
`GpuPreAgg`.

| Label | Meaning |
|---|---|
| `GPU Projection` | Columns and expressions the node produces. |
| `GPU Scan Quals` | Quals of the scanned (outer) relation evaluated on the GPU. `[plan: A -> B]` are the planner's row estimates before and after (with `COSTS`), `[exec: A -> B]` the rows seen and passed (with `ANALYZE`). |
| `GPU Join Quals [i]`, `GPU Left Outer Join Quals [i]`, `GPU Right Outer Join Quals [i]`, `GPU Full Outer Join Quals [i]`, `GPU Semi Join Quals [i]`, `GPU Anti Join Quals [i]` | The join conditions of the `i`-th inner relation of this GpuJoin (`[1]` to `[4]`); the word is the join type. Each has its own `[plan: ... exec: ...]`; a RIGHT/FULL join shows `exec: A+B -> C`, where B is the unmatched inner rows emitted at the end. A bracketed `[expr]` is a qual from the WHERE clause that applies after an outer join. |
| `GPU Outer Hash [i]`, `GPU Inner Hash [i]` | The hash keys on the outer side and on the inner relation. |
| `GPU Group Key` | The GROUP BY keys of a GpuPreAgg. |
| `Scan-Engine: VFS with GPU0` | Storage is read by the CPU (VFS) and fed to GPU0 (the only mode in this build; there is no GPU-Direct). With `ANALYZE` it adds `buffer=`, `vfs=` (blocks read through the buffer manager or the file) and `ntuples=`. |
| `Fallback-stats: depth[0]=N, ...` | With `ANALYZE`, when non-zero: tuples handled by the CPU fallback path, per join depth. Whether the CPU re-evaluation of compressed or TOAST-ed text is counted here was not checked. |
| `GPU Pinned Buffer` | Not produced in this build. |
| `GPU-Sort keys`, `GPU-Sort Limit`, `Window-Rank Filter` | A GpuPreAgg whose groups the service sorts (needs `pg_strom.enable_gpusort = on`, `pg_strom.cpu_fallback = off`, no parallel workers; the plan has no Aggregate or Sort above it). If the CPU would have to evaluate rows (compressed/external values) the query fails with "GPU-Sort cannot combine ... rows": turn `enable_gpusort` off for it. |
| Arrow: `referenced: ...`, file lists, `Parquet-cache` | From Arrow_Fdw; the Parquet cache is a stub. |

When an aggregate uses the GPU the plan is `Aggregate -> Custom Scan (GpuPreAgg)` (PostgreSQL combines the
partial states); an aggregate over a join or an Arrow scan is `Aggregate -> Custom Scan (GpuJoin / GpuScan)`. `EXPLAIN (VERBOSE)` with
`pg_strom.explain_developer_mode = on` dumps the xpucode programs (`LoadVars OpCode`, `Scan Quals
OpCode`, ...).

## 6. Errors during execution

| Message | Meaning |
|---|---|
| `the GPU service is not ready` | Section 1. |
| `failed on connect('<socket>'): ...` | The service's socket is gone (service restarting). The path is `socket_path` in `gpu_service_state()`: `<unix_socket_directories>/.pg_strom.<port>.gpuserv.sock`, or `/tmp/.pg_strom.<port>.gpuserv.sock` when that would be too long for a socket path. |
| `division by zero` / `integer out of range` / `value out of range: overflow` with function, block and line pointer | A real data error found by the kernel. Unlike PostgreSQL, the GPU may evaluate a qual on a row that PostgreSQL's evaluation order would have filtered out first; there is no CPU retry (`pg_strom.cpu_fallback` does not add one). Rewrite the expression, or `SET pg_strom.enabled = off` for that query. |
| `kernel did not finish within N ms (device marked lost)` | A GPU dispatch exceeded the time limit (default 10 s, 300 s on a CPU device; `PGSTROM_GPUSERV_TIMEOUT_MS` overrides). The query that hit the limit fails; the service then logs `the GPU device was lost ... exiting` and exits, and the launcher starts a fresh service (the `generation` in `gpu_service_state()` increases). Queries sent in between fail with a connection error or `device is lost`. |
| `vkCreateComputePipelines failed (shader compiler rejected the module?)` or `[mvk-error]` lines during a query | The driver's shader compiler rejected a kernel. Report the driver, `gpu_devcaps()` output and the query; `PGSTROM_GPUSERV_TRACE=1` adds the program. |
| `cannot allocate N bytes of device memory`, `out of memory` | The device (or its share of unified memory) is too small for the task or the join's inner buffer. A record batch or an inner store larger than the device's storage buffer limit is a task error. |
| `only plain hash joins are implemented by the Vulkan service`, `inner relation N is not a hash store the service can use`, `the inner buffer holds N relations (1 to 4 are supported)` | A join shape the service does not implement (pinned or partitioned inner buffer, for example after setting `pg_strom.pinned_inner_buffer_threshold` above 0). The planner normally avoids these. |
| `PG-Strom must be loaded via shared_preload_libraries` | Set it in `postgresql.conf` and restart; `CREATE EXTENSION` alone is not enough. |

## 7. Switches for diagnosis

Set these in the environment of the postmaster (they are inherited by the service) and restart:

| Variable | Use |
|---|---|
| `PGSTROM_GPUSERV_TRACE=1` | The service logs each session's xpucode (the translated program) and, per task, its timings, e.g. `scan task: 7936 pages (7936 from file) pages-in 3.1 ms (mapped), setup 0.4 ms, gpu 2.2 ms, results 1.7 ms (150 rows)` and `agg task: ...`. |
| `PGSTROM_GPUSERV_NO_MAP=1` | Read pages with `pread` instead of mapping the file into Vulkan; separates mapping problems from kernel problems. |
| `PGSTROM_GPUSERV_TIMEOUT_MS=<n>` | Dispatch time limit; raise it on slow CPU Vulkan devices. |

Also: `pg_strom.debug_level = 1` or `2` (more module messages), `client_min_messages = debug1` (prints
`CPU re-evaluation of N tuples` for rows re-run on the CPU), `pg_strom.explain_developer_mode`.

## 8. What is known to be missing

See [features.md](features.md) for the supported surface and `.claude/hw-acceleration.md` for the open
items: no CPU fallback for kernel errors, GROUP BY speed for many groups, the tier abstraction not yet
used by kernels, the upstream regression expected files not yet adapted (their `EXPLAIN` text describes
the CUDA product), and no testing on any GPU other than the Apple M1 Pro.
