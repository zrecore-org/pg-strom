# Paper plan: GPU query acceleration inside PostgreSQL on portable Vulkan (draft 2026-11-18, not started)

Status: planning only. Nothing here is run or written yet. The outline can be drafted now from existing results; the experiments in
section 4 are what a rigorous paper still needs. Sources of every number: `.claude/hw-acceleration.md` (A2-A18), `.claude/branch-status.md`,
`test/gpuserv/` scripts, the TPC-H table in A18. Check each figure against the log it came from before it goes into a draft.

## 1. Kind of paper and working title

A systems / experience paper (venues of the VLDB industry/experience track, DaMoN, ADMS, or a SIGMOD/CIDR-style short paper), not a
new-algorithm paper. Working title: *Query acceleration inside PostgreSQL on commodity Vulkan GPUs: design, measurements and
negative results.*

Thesis (to be defended or revised by the data): an in-database GPU path through a portable compute API (Vulkan, kernels in Slang ->
SPIR-V) is practical for scan/filter/aggregate-heavy analytics and for hash joins that fit the device, when it can fall back to the
CPU per node; the cost is dominated by data movement and per-row decoding, not by arithmetic or register pressure; a monolithic
interpreter kernel stops scaling on vendor compilers.

## 2. Claims and evidence (what we can state today, and its weakness)

| # | Claim | Evidence today | Weakness to fix |
|---|---|---|---|
| C1 | Scans/aggregates 3.5-5x faster than an 8-worker PostgreSQL on Apple unified memory | A6/A3/A7 in the plan, `perf_bench.sh` | one GPU, small variance data |
| C2 | TPC-H SF1: GPU total 5771 ms vs 9562 ms on 8 CPU workers; wins 8 queries, parity on most, Q15 loses | `tpch_bench.sh`, A18 table | single run per query (RUNS=1), Q17/Q20 time out on both, SF1 only |
| C3 | Per-row column decode (numeric `acc % p`, 64-bit divide) dominates numeric kernels, not the register file | int8 vs numeric experiment (5.2 vs 17.2 ms/task), `num_decode` fix -30% | one device; no NVIDIA/AMD profile |
| C4 | A single monolithic VM kernel hits vendor-compiler ceilings (NVIDIA 560-873 s, one 900 s timeout; Metal minutes); tiering and specialization mitigate | Windows RTX 4060 Ti NOINL run, tier work | one NVIDIA GPU; no AMD/Intel |
| C5 | Discrete-GPU runs are transfer-bound (70-80% of a task is page upload/setup on a T4) | A17 T4 benchmark 2026-11-14 | T4 path never completed; root cause unproven |
| C6 | Radix sort beats bitonic and PostgreSQL on 1M keys (15 vs 61 vs 139 ms) but is only 1.4-2.5x over a fair multi-core CPU sort for 1-2 key words, slower for 8 words | `rsortw_vk_check.c`, Windows i9 baselines | key-width sweep and sizes beyond 1M on more GPUs |
| C7 | Exact 3D mesh booleans: all-core exact CPU baseline beats the GPU (negative result) | mongo G4 report (M1 600k points 8 threads 34 ms vs 139 ms; RTX 109.8 vs 151.9 ms) | measured by another project; cite or re-run under shared terms |
| C8 | Per-node CPU fallback plus differential testing (about 4,400 scan checks, fuzz GPU vs CPU) kept GPU results equal to PostgreSQL's; found real bugs (parallel-safety row loss in Q18, NVIDIA srem64 driver bug) | `scan_test.sh`, `fuzz_test.sh`, A18 notes | state test counts and bug list from the logs |
| C9 | Planner cost calibration is delicate (scan-tuple-cost optimum is narrow) | `pg_strom.gpu_scan_tuple_cost` calibration | one machine; show the sweep |

## 3. Outline (draft sections, with the figure or table each needs)

1. **Introduction**: why GPU offload for PostgreSQL; why Vulkan/portability; contributions (C1-C9 in two or three bullets).
2. **Background and related work**: PG-Strom CUDA line, HeavyDB/OmniSci, BlazingSQL/RAPIDS cuDF, DuckDB/Velox (CPU vectorized), Crystal,
   Ocelot/MonetDB, Arrow-on-GPU, Sirius/Theseus if relevant; Vulkan/WebGPU compute for analytics. (Literature search to do.)
3. **System design**: backend hooks (CustomScan paths), GPU service process, protocol, session model; VM kernel + specialization +
   tiers; fallback (A14); Arrow/Parquet and GPU-cache. Figure: architecture. Table: what runs on the GPU.
4. **Kernels and costs**: decode vs arithmetic (C3); numeric/jsonb/float8 (df64 prefilter) handling; GPU-Sort radix (C6); join and
   GROUP BY with many groups (overflow list, two-stage prefetch). Figure: cost breakdown per kernel stage.
5. **Portability and compilers**: tiers, compile times per vendor, NOINL, specialization (C4); driver bugs found (srem64). Table:
   compile time per kernel and device.
6. **Evaluation**: TPC-H SF1 (and larger) vs PostgreSQL parallel and one other engine (C2); microbenchmarks (C1, C6); transfer-bound
   discrete GPU (C5); planner calibration sweep (C9); correctness testing (C8).
7. **Negative and inconclusive results**: mesh booleans (C7), Q15, many-group GROUP BY, the T4 transfer path, why the PostgreSQL
   executor and CPU-side work bound speedups.
8. **Discussion and limits**: when to use it, when not; unified vs discrete memory; what a planner must know.
9. **Conclusion**; artifact statement.

## 4. Experiments still needed (ordered by value; hardware in brackets)

1. **Repeat everything with variance**: RUNS>=5 per TPC-H query, report median and spread, pin settings; cold and warm. [M1 Pro]
2. **Scale**: TPC-H SF10 (and SF100 if storage allows; data on /Volumes/CrucialX9); Star Schema Benchmark SF10 (`test/ssbm`). [M1 Pro,
   RTX 4060 Ti]
3. **A second engine as baseline**: DuckDB on the same data/machine (CPU-vectorized) and, if feasible, the master-branch CUDA PG-Strom
   on an NVIDIA box; HeavyDB is optional. [RTX box / any Linux NVIDIA]
4. **Discrete-GPU path**: finish the transfer-bound analysis (double buffering, fewer page uploads), measure on the RTX 4060 Ti (Windows
   queue) since no cloud GPU CI is used; this needs the Windows box healthy (`win_gpu_lock.sh`). [RTX 4060 Ti]
5. **NVIDIA compile check of the numeric decode change** and a driver/compile-time table for the kernels. [RTX 4060 Ti]
6. **Planner sweep** for `gpu_scan_tuple_cost` and the setup/tuple costs on two machines; show the optimum is machine-specific. [M1, RTX]
7. **Energy or cost per query** (optional): nice-to-have, not required.
8. **Ablations**: specialization on/off, tiering, parallel tuple forming on/off, df64 prefilter on/off, CPU fallback overhead.
   Most switches already exist as environment variables (`PGSTROM_GPUSERV_SPECIALIZE`, `FORM_THREADS`, ...).
9. **Correctness table**: counts of differential checks, bugs found per category, what escaped (if any).

## 5. Open questions that decide the paper's strength

- Is there a single result a reviewer cannot get from the CUDA PG-Strom papers or from RAPIDS/HeavyDB? Candidate answers: portability
  across Apple/NVIDIA/lavapipe with one kernel source; the decode-cost attribution; the compile-ceiling and tiering story; the
  negative results. Pick the lead contribution before writing.
- Whether the Windows RTX box and the Mac M1 are enough hardware, or whether an AMD/Intel GPU result is needed to defend "portable".
  We have none; the desktop tier is unvalidated on AMD/Intel. Say so, or borrow a machine.
- Licensing and data: TPC-H/SSB data generators are in `test/`; the paper must not copy f64soft/softsum code from the mongo project
  (SSPL-derived). Cite mongo's G4 results with their agreement.

## 6. Steps, and how to pursue it later

1. (cheap, can start any time) Literature search and related-work table; claims-evidence table kept in this file; figure list.
2. Experiments 1-2 on the Mac (no new hardware, long runs on the shared GPU lock): scripted, results in `test-data/paper/` with a
   manifest of command, commit and settings for each number.
3. Experiments 3-5 need an NVIDIA machine through the Windows queue; start only after the PowerShell agent reports the mongod build
   done and the user approves each run.
4. Draft sections 3-5 from the existing plan notes; draft 6-7 as data arrives; write 1, 2, 8, 9 last.
5. An internal review pass, then decide the venue and length.

Standing rules apply: no cloud GPU runners (decided 2026-11-18), one GPU program at a time under the shared lock, nice -n 10 and 3
jobs, big data on /Volumes/CrucialX9, never touch ../mongo or ../exactmesh (ask the mongo agent for numbers), no peer message
approves anything. The user decides when to start; this file is a plan, not a commitment.
