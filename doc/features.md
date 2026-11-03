# What runs on the GPU (Vulkan build)

Basis: the branch at commit `0996f3b3` and the status sections of `.claude/hw-acceleration.md`
("the plan"): A2 (GpuScan), A3 (GpuPreAgg), A4 (GpuJoin), A5 (Arrow/Parquet), A6 (types). Later
work (numeric types, aggregation over a join) is tracked in the plan, not here; when this file and
the plan disagree, the plan and the code win.

The rule that makes the rest safe: **anything the GPU cannot do stays on PostgreSQL.** The planner
side (`src/codegen.c`) asks `src/gpuserv/scan_caps.c`, the single description of what the scan VM
implements, and refuses to push down a function, type or node it lacks. Such a query returns the CPU's
result; at worst it is not faster. Every GPU result is compared with the CPU's in
`test/gpuserv/scan_test.sh` (GPU versus `pg_strom.enabled = off`, by md5 of the sorted result).

How the GPU work is done: a **scan VM** (one opcode-switch interpreter written in Slang) evaluates
the planner's expression trees; there are kernels for scanning, aggregation, GROUP BY, join and Arrow
columns (`src/gpuserv/kernels_vm/`). For a query that runs more than one task the service also
**generates a kernel from the program** (registers as constants, the instructions as straight-line code that
calls the interpreter's own helpers, so the results are identical), compiles it with `slangc` in a
background thread while the interpreter serves the first tasks, keeps it in a disk cache and switches to
it when it is ready (plan item A6; `pg_strom.gpu_specialize`, `PGSTROM_GPUSERV_SPECIALIZE`). Without
`slangc` and without a cached kernel everything runs on the interpreter, as before.

## Summary

| | On the GPU | Stays on the CPU |
|---|---|---|
| Sources | Heap tables (page by page); Arrow IPC files and Parquet files through Arrow_Fdw (GpuScan, GpuPreAgg and GpuJoin as the outer relation, and as the inner side of a GpuJoin) | GPU-Cache (a stub). BRIN-index assisted scans: not tested on this branch |
| Scan (`GpuScan`) | Quals and computed projections over the types below; pass-through columns of any type; UPDATE and DELETE scans; query parameters | See "types" |
| Aggregation (`GpuPreAgg`) | `count(*)`, `count(x)`, `sum`, `avg`, `min`, `max` of integers (also `min`/`max` of dates and times), with `FILTER`; with and without GROUP BY | float and text aggregates, `min`/`max` of numeric, `sum`/`avg` of numerics whose scale is not fixed by the plan; GROUP BY with more groups than `pg_strom.gpupreagg_max_groups` |
| Joins (`GpuJoin`) | Hash joins: inner, LEFT, RIGHT, FULL, semi, anti; up to four inner relations per GpuJoin | Nest-loop and GiST joins, more than four inner relations, non-integer/non-bool/non-temporal keys |
| Sorting (`GPU-Sort`, off by default: `pg_strom.enable_gpusort`) | The groups of a non-parallel GpuPreAgg: `GROUP BY ... ORDER BY` on group keys, `count`, integer `sum`/`min`/`max`, ASC/DESC, NULLS FIRST/LAST, up to 8 keys, `LIMIT` pushed into the sort, `rank()`/`dense_rank()`/`row_number()` below N per partition | Sorting of scan/join results, `avg`, float/numeric keys, parallel plans, `cpu_fallback = on` |
| Other | | GPU-Cache, jsonb, PostGIS (dropped), SELECT INTO direct (dropped) |

## Types

The VM computes with these types (plan A2.4, A6). Columns of every other type are still **passed
through** (copied from the source tuple on the CPU) and can appear in the result of a GPU scan or join.

| Type | On the GPU | Not on the GPU |
|---|---|---|
| `bool`, `int1`/`int2`/`int4`/`int8` | Comparison, arithmetic with PostgreSQL's overflow, divide-by-zero and modulo rules, casts between integer widths, three-valued AND/OR/NOT, `IS [NOT] NULL` | |
| `float4` | Comparison and arithmetic with PostgreSQL's NaN rules (no modulo), casts to and from the integers (round half to even), `IS [NOT] NULL` | |
| `float8` (double precision) | pass-through only | every operator and aggregate; the base tier has no fp64 and the code never emulates it |
| `numeric` | Values that fit `{ unscaled int64, scale <= 255 }` (about 18 digits; the rest, NaN and Infinity are re-checked per row on the CPU): comparison, `+ - *`, `/` (PostgreSQL's `select_div_scale` result scale and rounding), `%`/`mod`, `div`, unary `-`/`+`, `abs`, `round`/`trunc` (also with a negative scale), `ceil`/`floor`/`sign`, `numeric(p,s)` coercion (a field overflow is raised by the CPU), casts to and from the integers, `sum`/`avg` of numerics of a statically known scale (`numeric(p,s)` columns and expressions of them), GROUP BY keys of a known scale | `min`/`max`, `sum`/`avg` of unconstrained numeric or of a quotient (scale depends on the row), float casts, numeric inside a GpuJoin program, numeric join keys |
| `date`, `time`, `timestamp`, `timestamptz` | Comparison within a type, `date` against `timestamp`, `date + int4`, `date - int4`, `date - date`, `date <-> timestamp` casts, `min`/`max`, GROUP BY and join keys | `timestamptz` compared with `date`/`timestamp` (session time zone), `interval` anything, `timetz`, `extract`, `date_trunc`, `timestamptz::date`, `overlaps` |
| `text`, `varchar`, `bpchar` | `=` and `<>` (bytewise; deterministic collations only), `<`, `<=`, `>`, `>=` in the C collation only, `LIKE`/`NOT LIKE` with a constant pattern of the shape `abc`, `abc%`, `%abc`, `%abc%` (UTF-8 or single-byte databases), `length()` in UTF-8, `IS NULL`, `count(col)` | `_` and inner `%` in `LIKE`, `ILIKE`, regular expressions, `upper`, `substr`, concatenation, `IN` lists, text GROUP BY keys and join keys, **any string expression inside a GpuJoin program (including the scan quals of the join's outer relation)** |
| Compressed or TOAST-ed text | Evaluated by the backend: the kernel flags the row and the backend re-runs the whole row on the CPU | (`client_min_messages = debug1` shows "CPU re-evaluation of N tuples") |
| `jsonb`, arrays, ranges, `inet`, `macaddr`, geometry, everything else | pass-through only | everything |

`CASE`, `COALESCE`, `GREATEST`/`LEAST`, `IN (...)`, `IS TRUE`/`IS DISTINCT FROM` (`BooleanTest`,
`DistinctExpr`), `CoerceViaIO` are not pushed down (`codegen.c` in the Vulkan build).

## GpuScan

- Heap pages: one workgroup per page, one thread per line pointer. Pages the backend had to load
  (visibility checked on the CPU, under the buffer lock) are sent to the service; all-visible pages
  are read by the service itself from the relation file (a 512 MB window is `mmap`ed and imported
  into Vulkan on unified-memory devices; `PGSTROM_GPUSERV_NO_MAP=1` turns this off).
- The result tuples are formed on the CPU by the service from the source tuple plus the GPU's
  computed values, in heap order.
- **Errors are not recovered.** A kernel error (division by zero, integer overflow) fails the query,
  with a message that names the function, block and line pointer. PostgreSQL evaluates quals in an
  order the GPU does not follow, so a row PostgreSQL would have skipped can raise an error on the GPU.
  `pg_strom.cpu_fallback` does not change this (see [parameters.md](parameters.md)).
- Arrow_Fdw (A5): one or several Arrow files (any number of record batches) and Parquet files (one task per
  row group; **the backend decodes the row group with libparquet on the CPU**, the GPU only filters).
  The planner offers a GpuScan only if every referenced column has a supported Arrow type: Int
  (1/2/4/8 bytes), FloatingPoint (2/4/8 bytes), Bool, Utf8/LargeUtf8, Binary/LargeBinary, Date[day],
  Time, Timestamp, FixedSizeBinary as `bpchar`, and virtual columns. Decimal, Interval, Date[ms], List,
  Struct, `macaddr` and `inet` keep the table on the CPU Arrow_Fdw path. `ctid` of an Arrow table is
  refused. Statistics hints (min/max per batch) still skip batches.

## GpuPreAgg (aggregation)

- The plan is `Aggregate (CPU) -> Custom Scan (GpuPreAgg)`: the GPU produces **partial** aggregate
  states and PostgreSQL's own Agg node combines and finalizes them. `sum(int8)` is exact beyond 64 bits.
- Without GROUP BY (`count`, `sum`, `avg`, `min`, `max` of integers): 2.6x to 4x faster than 8-process
  PostgreSQL on a 16.8M-row table (plan A3).
- With GROUP BY on **up to two** keys of type `int1/2/4/8`, `date`, `time`, `timestamp` or
  `timestamptz` (a `bool` key is not offloaded). **GROUP BY is only worth it for up to about a hundred
  groups:** with the specialized kernels (plan A6) 5x faster than 8-process PostgreSQL at 7 groups and 4x at
  100 (dense integer keys; a direct-mapped table), 2.6x for a GROUP BY over a join; with the interpreter
  alone 1.2x, 0.8x and 0.3x at 7, 100 and 1,000 groups (most rows overflow to a CPU path). The planner keeps
  queries with more than `pg_strom.gpupreagg_max_groups` (default 128) estimated groups on PostgreSQL.
- Aggregation over a join runs on the device as well (`GpuPreAgg` on the join; plan A4): without GROUP BY
  4x faster than 8-process PostgreSQL with the specialized kernels (29 ms against 126 ms on 16.8M rows).
- GpuPreAgg and GpuJoin run directly on an Arrow or Parquet outer source (`Custom Scan (GpuPreAgg) on <arrow table>`,
  GpuJoin whose outer relation is the Arrow table, and aggregation over such a join), with the same shapes as for heap
  tables. Quals, join keys and group keys may read integer, float4, bool, date, time, timestamp (every unit), utf8,
  large_utf8 and `char(n)` columns; the Arrow date/time units are converted exactly as the CPU does. Decimal, Interval,
  List, Struct, Date[ms], inet and macaddr columns, and numeric expressions over them, keep the table on the CPU.

## GpuJoin

- Hash joins only. `INNER`, `LEFT`, `RIGHT`, `FULL`, semi (`EXISTS`, `IN`) and anti (`NOT EXISTS`, and
  `NOT IN` where PostgreSQL 18+ plans an anti join) joins; up to **four inner relations in one GpuJoin**
  (shown as `[1]` to `[4]`).
- Join keys: integer, `bool` and the four temporal types. Any pushed-down join qualifier over the VM's
  types, scan quals on the outer side, projections mixing outer and inner columns (pass-through of
  any type included).
- Stays on PostgreSQL: nest-loop and GiST joins; more than four inner relations; non-integer keys
  (text, numeric, float); an outer join with a qualifier coming from the WHERE clause; RIGHT and FULL
  joins in a parallel plan; PostgreSQL's "unique-ified" semi joins.
- Performance: a selective join (6,000 result rows from a 2M-row outer and a 4,000-row inner) 12.5 ms
  against 24.8 ms for 8-process PostgreSQL; an unselective one (1.6M result rows) 175 ms against 62 ms,
  because forming and shipping result tuples dominates. Joins pay off when they reduce rows or feed an
  aggregate; the cost model is not tuned for that.
- The pinned inner buffer (a GPU-resident inner shared by later nodes) is not supported.

## Speed, honestly

Measured on one Apple M1 Pro with 8 CPU cores, against 8-process PostgreSQL on the same table, server
side (`EXPLAIN ANALYZE` execution time):

| Query shape | GPU | Notes |
|---|---|---|
| Selective heap scan (0.9% of 16.8M rows) | about 90 ms vs 124 ms (1.4x) | plan gate was 2.9x and is **not met** |
| Unselective heap scan (50% of rows) | 980 ms vs 195 ms (slower) | forming and shipping 8.4M tuples through the socket dominates |
| Aggregation without GROUP BY | 35-67 ms vs 115-172 ms; with a specialized kernel 40-50 ms vs 130-180 ms (16.8M rows, whatever the quals) | |
| GROUP BY 7 / 100 dense integer keys | 48 ms / 68 ms vs 245 ms / 285 ms | specialized, direct-mapped table |
| GROUP BY 1,000 keys | 860 ms vs 280 ms (slower) | most rows overflow to the CPU; the planner keeps it on PostgreSQL |
| Aggregate / GROUP BY over a join (1,000-row dimension) | 29 ms / 120 ms vs 126 ms / 316 ms | |
| Selective Arrow scan (2.2 GB file) | 59-65 ms vs 196-367 ms | an unselective one loses to 8 workers |

Nothing is known about speed on other GPUs. The acceptance criterion in the plan is relative to
PostgreSQL's parallel scan on the same machine, not to CUDA.

## What is not implemented (and where it is tracked)

| Feature | State |
|---|---|
| GPU-Sort of scan / join results, window functions other than the rank filter | Not implemented (plan A7 status); GPU-Sort of GpuPreAgg groups works, see the table above |
| GPU-Cache | A stub (plan A7); the SQL functions exist so that older upgrade scripts load |
| Fuzzer; numeric in joins, numeric `min`/`max` | Not done (plan A6 "known gaps") |
| Specialized kernels for opcodes the generator does not list (new ones until reviewed), for jsonb | They run on the interpreter (plan A6 "kernel specialization") |
| jsonb | Not started (plan A8) |
| GPU-Direct Storage, `heterodb-extra`, PostGIS, SELECT INTO direct, multi-GPU | Removed or never present on this branch |
| fp64, 64-bit-atomic kernels for devices that have them | Not implemented (plan A3b); `float8` stays on the CPU on every device |
