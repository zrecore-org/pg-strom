# Installing PG-Strom (Vulkan build)

This describes the `vulkan-gpu` branch: PG-Strom with the CUDA code removed, running GPU work
through Vulkan compute. The CUDA product on `master` is documented by the generated site in
`docs/`, which has **not** been updated for this branch.

State of the facts below: the source at the time this file was written (branch `vulkan-gpu`,
version `7.0` in `src/pg_strom.control`). Statements about what was and was not tested come from
`.claude/hw-acceleration.md` (the plan; its "Validation matrix" and per-milestone status).

## Platforms and capability tiers

| Platform | Status |
|---|---|
| macOS on Apple Silicon (M1 Pro, MoltenVK, PostgreSQL 17) | **Validated**: all development and the test suites in `test/gpuserv/` run here. |
| arm64 Linux with lavapipe (Mesa's Vulkan implementation on the CPU) | Used to check the host code and the kernels' logic. It is a CPU device: refused unless `pg_strom.gpu_allow_cpu_device = on`, and never faster than the CPU. |
| NVIDIA, AMD or Intel GPUs (Linux, Windows) | **Unvalidated.** The code is written against Vulkan, not against a vendor; behavior and speed on real discrete or desktop GPUs are unknown. No Windows build exists. |
| Other PostgreSQL major versions | The code requires PostgreSQL 15 or later (`src/pg_strom.h`). Only 17 is exercised by the Vulkan tests; the upstream regression expected files for 15-19 describe the CUDA product (see [features.md](features.md)). |

At start-up a probe program runs small test kernels on every Vulkan device and records what really
works next to what the driver claims (`pgstrom.gpu_devcaps()` shows both: `caps` and
`reported_caps`; MoltenVK, for example, claims 64-bit subgroup arithmetic that the Metal shader
compiler rejects). The result is a tier (`src/gpuserv/devcaps.c`, `gpuserv_devcaps_compute_tier`):

| Tier | Meaning |
|---|---|
| `none` | Not usable. Reasons are in `tier_reason`: a CPU device without the allow flag, a missing capability (int64, 8-bit storage, subgroup ballot and 32-bit subgroup arithmetic, host-memory import or unified memory), shared memory below 16 KiB, fewer than 128 invocations per workgroup, or a storage buffer limit below 1 GiB without buffer device address. |
| `base` | Usable. This is what Apple Silicon gets. No fp64 and no 64-bit buffer atomics are required or used. |
| `desktop` | `base` plus 64-bit buffer atomics, fp64 and the Vulkan memory model. lavapipe reports this tier. **In this branch the tier only decides whether a device is usable and is shown to the user; no kernel changes behavior by tier** (`grep` of `src/` finds no use of the tier beyond reporting). The planned fp64 / 64-bit-atomic kernels (plan item A3b) are not implemented. |

Exactly one device is used: the first one (in Vulkan enumeration order) that is not tier `none` and is
allowed by `pg_strom.gpu_visible_devices`. Multi-GPU is not implemented.

## Prerequisites

| Need | Details |
|---|---|
| PostgreSQL 15+ with development files | Builds go through PGXS against the `pg_config` you name. |
| C compiler, C++20 compiler, GNU make | GNU make 3.81 (the one macOS ships) works. |
| Apache Arrow and Parquet | `libarrow` and `libparquet`, found with `pkgconf` (package names `arrow` and `parquet`). Version 26.0.0 was used for development; the RPM spec asks for 23 or newer, the Makefile checks nothing. |
| Vulkan loader and headers | Linux: `pkg-config vulkan` is used. macOS: see below. |
| A Vulkan driver | MoltenVK on macOS; on Linux a Mesa or vendor driver. Without a usable device the extension still loads and PostgreSQL runs without GPU offload. |
| `slangc` (the Slang compiler) | Needed at **build time only**: it compiles the kernels to SPIR-V, which are embedded in the `pgstrom-gpuserv` executable (no `.spv` file is installed). Developed with **Slang 2026.20**; other versions are untested. Slang is not in distribution repositories: download the release archive for your platform from the Slang project's GitHub releases page (for example `slang-2026.20-macos-aarch64.tar.gz`, `slang-2026.20-linux-x86_64.tar.gz`, `slang-2026.20-linux-aarch64.tar.gz`). |
| `libnuma` | Optional. |
| Linux only: `libkrb5-dev` (Debian/Ubuntu) | PostgreSQL's `libpq/libpq.h` includes `gssapi/gssapi.h` (found while building on Ubuntu, plan A1). |

### How the build finds `slangc`

`src/gpuserv/Makefile` takes, in this order: the `SLANGC` variable (command line or environment),
`slangc` in `PATH`, `$HOME/tools/slang-2026.20/bin/slangc`, `/opt/slang/bin/slangc`.

```
make -C src PG_CONFIG=... SLANGC=/opt/slang/bin/slangc install
```

### How the build finds Vulkan

On Linux, `pkg-config --cflags --libs vulkan` (fallback `-lvulkan`). On macOS the Makefile defaults to
`-I/usr/local/include` and `-L/usr/local/lib -lvulkan`, i.e. the layout of the LunarG Vulkan SDK
system install (which is what the validated machine uses: `libvulkan` and `libMoltenVK.dylib` in
`/usr/local/lib`, the MoltenVK ICD file in `/usr/local/share/vulkan/icd.d`). With Homebrew's
`molten-vk`, `vulkan-loader` and `vulkan-headers` the paths differ; name them yourself (this
combination was not tried by the author; it is what `.github/workflows/vulkan-check.yml` does):

```
make -C src PG_CONFIG=... \
  VK_CFLAGS="-I$(brew --prefix vulkan-headers)/include" \
  VK_LIBS="-L$(brew --prefix vulkan-loader)/lib -lvulkan -Wl,-rpath,$(brew --prefix vulkan-loader)/lib" install
```

If the loader cannot find MoltenVK at run time, the probe finds no device (see
[troubleshooting.md](troubleshooting.md)).

## Build and install

Always pass `PG_CONFIG`:

```
# macOS (Homebrew PostgreSQL 17)
brew install postgresql@17 apache-arrow pkgconf
make -C src PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config install
```

```
# Debian / Ubuntu (the package list the CI job uses; the CI job has not been run yet)
sudo apt-get install build-essential pkgconf libvulkan-dev mesa-vulkan-drivers \
     libnuma-dev libkrb5-dev libicu-dev postgresql-17 postgresql-server-dev-17
# libarrow-dev and libparquet-dev: from the Apache Arrow apt repository
make -C src PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config install
```

Variables:

| Variable | Effect |
|---|---|
| `WITH_GPUSERV=auto` (default) | Build `pgstrom-gpuserv` when `slangc` and the Vulkan headers are found; otherwise print a warning and continue, so the module still builds and runs without GPU. |
| `WITH_GPUSERV=1` | A missing prerequisite is an error (the RPM spec uses this). |
| `WITH_GPUSERV=0` | Do not build the service. |
| `SLANGC`, `VK_CFLAGS`, `VK_LIBS` | See above. |
| `PGSTROM_DEBUG=1` | `-g -O0 -DPGSTROM_DEBUG_BUILD`. |
| `PGSTROM_FLAGS_CUSTOM` | Extra host compiler flags. |

What is installed: `pg_strom.so` and `pgstrom-gpuserv` in `pg_config --pkglibdir`, the extension
control file and SQL scripts in the PostgreSQL share directory. `make -C src/gpuserv` builds the
service alone. `arrow-tools/` (`pg2arrow`, ...) does not build on macOS yet.

Packages: `make rpm` and `make deb` exist (`files/`); they have not been built since the CUDA removal
(see "Packaging" below).

## Configure and start

The module must be loaded when the postmaster starts. In `postgresql.conf`:

```
shared_preload_libraries = 'pg_strom'
```

Restart PostgreSQL. During start-up (module initialization, in the postmaster) PG-Strom runs
`pgstrom-gpuserv --probe` and waits for it, so **start-up can take longer** (up to
`pg_strom.gpuserv_probe_timeout`, 60 s by default; the probe is killed after that and PG-Strom runs
without GPU). It then registers one background worker, the launcher, which starts the service
(`pgstrom-gpuserv --serve`) and restarts it when it dies, so `max_worker_processes` needs one free
slot. A crash or hang of the GPU driver kills only the service, not PostgreSQL.

The server log shows the probe result, for example (macOS):

```
LOG:  PG-Strom: Vulkan device 0: Apple M1 Pro (MoltenVK), tier base - base tier; missing capability atomic_i64_add for desktop
LOG:  PG-Strom: GPU0 Apple M1 Pro (Vulkan, tier base), RAM 16384MB
LOG:  PG-Strom: GPU service started (pid 39320, generation 1)
```

Then, in each database that should use it:

```sql
CREATE EXTENSION pg_strom;
```

Settings from the CUDA product (`pg_strom.cuda_visible_devices`, `pg_strom.gpudirect_*`, ...) no
longer exist; see [parameters.md](parameters.md). PG-Strom does not reserve its GUC prefix, so
PostgreSQL treats an unknown `pg_strom.*` name in `postgresql.conf` as a placeholder and the
setting silently does nothing. Remove them.

## First checks

```sql
SELECT * FROM pgstrom.gpu_service_state();   -- state = 'ready'
SELECT * FROM pgstrom.gpu_devcaps();         -- one row per Vulkan device; selected = true for the one in use
SELECT pgstrom.gpu_service_ping();           -- 'ok pid=... generation=... device=... tier=...'
SELECT * FROM pgstrom.gpu_device_info;       -- the same facts, one row per attribute
```

`gpu_service_state()` returns `state` (`starting`, `ready`, `failed`, `stopped`, or `unavailable` when no
usable device was found), `pid`, `generation` (it increments with every service start),
`restart_count`, `selected_device`, `socket_path` and `last_error`.

Then look at a plan on a table large enough for the GPU to pay off (or set
`pg_strom.gpu_setup_cost = 0`, as the tests do, for a small one):

```sql
EXPLAIN SELECT id FROM big WHERE v < 10 AND w + 1 > 5;
-- Custom Scan (GpuScan) on big
--   GPU Scan Quals: ...
--   Scan-Engine: VFS with GPU0
```

How to read these labels is in [troubleshooting.md](troubleshooting.md); what runs on the GPU is in
[features.md](features.md).

## Tests

```
make -C test/port && test/port/port_test                                  # portability layer, no PostgreSQL needed
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/service_test.sh   # service lifecycle (throwaway cluster)
GPUSERV_BIN=<installed pgstrom-gpuserv> sh test/gpuserv/scan_test.sh      # GPU vs CPU result comparison
```

Both shell tests build a throwaway cluster and need the module and the service installed first;
`PGBIN` names the PostgreSQL `bin` directory (default: Homebrew PostgreSQL 17). `ALLOW_CPU=on` runs
them on lavapipe. The upstream regression suite (`test/`, `installcheck`) mostly does not pass on this
branch yet: its expected output asserts CUDA-era plan shapes.

## Packaging

`make rpm` (RPM, per PostgreSQL major version) and `make deb` use the templates in `files/`. After the
Vulkan port they: require `vulkan-loader` (RPM) / `libvulkan1` (DEB) and the Arrow and Parquet
runtime libraries, build with `WITH_GPUSERV=1` (a build machine without `slangc` fails), install
`pgstrom-gpuserv` next to `pg_strom.so`, and contain no CUDA, `heterodb-extra` or GPUDirect
dependency. `slangc` cannot be a package requirement; give it with `SLANGC=/path/to/slangc make rpm`.
**None of this has been built:** the templates were reviewed and syntax-checked only, because the
development machine has neither `rpmbuild` nor `dpkg`. The systemd drop-in (`files/systemd-pg_strom.conf`)
keeps the file-descriptor and core-file limits and carries a commented `SupplementaryGroups=render`
hint (a Vulkan driver opens `/dev/dri/renderD*`; general practice, not tested here).
