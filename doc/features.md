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
| Sources | Heap tables (page by page); a table's GPU-Cache (see below); Arrow IPC files and Parquet files through Arrow_Fdw (GpuScan, GpuPreAgg and GpuJoin as the outer relation, and as the inner side of a GpuJoin) | BRIN-index assisted scans: not tested on this branch |
| Scan (`GpuScan`) | Quals and computed projections over the types below; pass-through columns of any type; UPDATE and DELETE scans; query parameters | See "types" |
| Aggregation (`GpuPreAgg`) | `count(*)`, `count(x)`, `sum`, `avg`, `min`, `max` of integers and of `float4` / `float8` (`sum(float4)` adds in float4 like PostgreSQL; also `min`/`max` of dates and times), with `FILTER`; with and without GROUP BY. A float sum or average adds in the row order of a workgroup and then the workgroups' partial sums, not in the order of a serial scan: it is deterministic but can differ from PostgreSQL's in the last bits (as PostgreSQL's own parallel plans do); `min`/`max` and sums of exactly representable values are equal | `min`/`max` of text, bool aggregates (`bool_and`, `bool_or`, `every`), `bit_*`, variance / stddev / corr / regr_*, `min`/`max` of numeric, `sum`/`avg` of a numeric whose scale is not fixed by the plan, a CASE expression as an aggregate argument; DISTINCT, ordered-set and `ORDER BY` aggregates, `string_agg` / `array_agg` (by design); GROUP BY with more groups than `pg_strom.gpupreagg_max_groups`, with three or more keys, or with a key of type float, text, bool or uuid, GROUPING SETS (measured by `coverage_test.sh`) |
| Joins (`GpuJoin`) | Hash joins: inner, LEFT, RIGHT, FULL, semi, anti; up to four inner relations per GpuJoin | Nest-loop joins (non-equi, range, cross, LATERAL: no GpuNestLoop); keys of type float, numeric, text, varchar, bpchar, uuid and bool; `IS NOT DISTINCT FROM`; a hashed `NOT IN` SubPlan; GiST joins only in the forms of "GiST-index spatial join" below. More than four inner relations are planned as a chain of stacked GpuJoins (measured), not one GpuJoin |
| Sorting (`GPU-Sort`, off by default: `pg_strom.enable_gpusort`) | The groups of a non-parallel GpuPreAgg: `GROUP BY ... ORDER BY` on group keys, `count`, integer `sum`/`min`/`max`, ASC/DESC, NULLS FIRST/LAST, up to 8 keys, `LIMIT` pushed into the sort, `rank()`/`dense_rank()`/`row_number()` below N per partition | Sorting of scan/join results, `avg`, float/numeric keys, parallel plans, `cpu_fallback = on` |
| Other | GPU-Cache (below) | PostGIS beyond the box operators, point-in-polygon, `ST_DWithin`, `ST_Z` / `ST_M` and `ST_3DDWithin` of two points (not yet), SELECT INTO direct (dropped) |

## GPU-Cache

A table with a row trigger `CREATE TRIGGER ... AFTER INSERT OR UPDATE OR DELETE ON t FOR EACH ROW EXECUTE FUNCTION pgstrom.gpucache_sync_trigger('options')` has a copy that the GPU service keeps in device memory and current, so that a scan, aggregate or join of the table needs no page transfer (plan A7 part 2; design in the plan). The cache is built the first time a backend writes or reads the table (`pgstrom.gpucache_info('t')` shows the phase: `not_built`, `loading`, `ready`, `corrupted`), and `EXPLAIN` shows `GPU Cache: GPU0 [phase: ready, max_num_rows: N]` under the node that uses it.

| | |
|---|---|
| Options of the trigger | `max_num_rows` (default 10485760: row versions the cache holds, including versions of transactions in flight), `cache_size` (default 512 MB, minimum 64 MB: the device buffer; `max_num_rows` is clamped to what fits), `gpu_device_id` (only 0), `gpumask` (compatibility). `redo_buffer_size`, `gpu_sync_interval` and `gpu_sync_threshold` are validated and shown but have no effect |
| What a scan sees | **The latest committed state plus the session's own changes**, as in the CUDA product (not a snapshot: a REPEATABLE READ transaction sees rows committed after its snapshot). A transaction's changes become visible to others atomically at its commit |
| Types | Every column type is cached. The GPU computes with bool, int1-8, float4, float8, date/time/timestamp(tz), text/varchar/bpchar and (PostGIS installed) `geometry`, read as a string like on a heap table; **numeric** and **jsonb** are read from the cache too (a numeric as `{ int64, scale }`, a value beyond that, NaN and Infinity are re-checked by the backend from a heap tuple built from the cache slot; jsonb as on a heap table); the others (float2, uuid, arrays, ...) are passed through. A query whose program needs the VM to read a passed-through column reads the table instead (`gpucache_info()` counts the scans the cache served: `scans`); `cstring` and fixed types longer than 1024 bytes cannot be cached (an error at the initial load; the table is read) |
| Changes | The trigger logs every inserted and removed row version in the backend; the log is sent to the service (synchronously) before the transaction commits and before the same backend scans the cache, so a commit of a transaction that wrote a cached table costs one round trip to the service. INSERT, UPDATE, DELETE, `COPY`, `ON CONFLICT`, subtransactions (SAVEPOINT) and rollbacks are handled; `TRUNCATE`, `ALTER TABLE`, `VACUUM FULL`, `DROP` and trigger changes drop the cache, which is rebuilt on next use. `PREPARE TRANSACTION` of a transaction that wrote a cached table is refused |
| Failures | If the cache may not mirror the table (the GPU service restarted, a request failed) it is rebuilt from the table at the next use, and queries read the table meanwhile. If the row versions exceed `max_num_rows` (or the table is larger at the initial load) the cache is `corrupted`: a WARNING, queries read the table, and `SELECT pgstrom.gpucache_recovery('t')` rebuilds it |
| SQL | `pgstrom.gpucache_info()` and `pgstrom.gpucache_info(regclass)` (json), `pgstrom.gpucache_compaction(regclass)` (compact the string regions), `pgstrom.gpucache_recovery(regclass)`, `pgstrom.gpucache_apply_redo()` (sends the backend's pending log; nothing else is pending). Parameters: `pg_strom.enable_gpucache`, `pg_strom.gpucache_max_relation_entries` |
| Not done | Applying the log with GPU kernels (the service applies it on the CPU into the shared buffer), a shared-memory redo buffer, several GPUs, replication slaves (`ALWAYS` triggers are honoured but untested), caches larger than the device's storage buffer limit |

## Types

The VM computes with these types (plan A2.4, A6). Columns of every other type are still **passed
through** (copied from the source tuple on the CPU) and can appear in the result of a GPU scan or join.

| Type | On the GPU | Not on the GPU |
|---|---|---|
| `bool`, `int1`/`int2`/`int4`/`int8` | Comparison, arithmetic with PostgreSQL's overflow, divide-by-zero and modulo rules, casts between integer widths, three-valued AND/OR/NOT, `IS [NOT] NULL` | |
| `float4` | Comparison and arithmetic with PostgreSQL's NaN rules (no modulo), casts to and from the integers (round half to even), `IS [NOT] NULL` | |
| `float8` (double precision) | Comparison (PostgreSQL's order: NaN equals NaN and is the largest, -0 = +0), `+ - * /` with PostgreSQL's overflow, underflow and division-by-zero errors, unary `-`/`+`, `abs`, `sqrt`, `round`, `ceil`/`ceiling`, `floor`, `trunc`, `sign` (a negative `sqrt` raises PostgreSQL's error), casts to and from the integers (round half to even, range checked), `float4` and `float8` mixed, `IS [NOT] NULL`, the aggregates `min`, `max`, `sum`, `avg` of `float8` and `min`, `max`, `sum`, `avg` of `float4`, with and without GROUP BY (also over a join, Arrow doubles and the GPU-Cache; a sum adds in another order than a serial scan, so its last bits can differ from PostgreSQL's); also from Arrow double columns and the GPU-Cache. Computed in integer arithmetic (`kernels_vm/f64_soft.inc`, bit-identical to the CPU's double), so it runs on devices without fp64 | Group keys, join keys, sort keys and the other math functions (`exp`, `ln`, `pow`, trigonometry, ...) are not on the GPU yet |
| `numeric` | Values that fit `{ unscaled int64, scale <= 255 }` (about 18 digits; the rest, NaN and Infinity are re-checked per row on the CPU): comparison, `+ - *`, `/` (PostgreSQL's `select_div_scale` result scale and rounding), `%`/`mod`, `div`, unary `-`/`+`, `abs`, `round`/`trunc` (also with a negative scale), `ceil`/`floor`/`sign`, `numeric(p,s)` coercion (a field overflow is raised by the CPU), casts to and from the integers, `sum`/`avg` of numerics of a statically known scale (`numeric(p,s)` columns and expressions of them), GROUP BY keys of a known scale | `min`/`max`, `sum`/`avg` of unconstrained numeric or of a quotient (scale depends on the row), float casts, numeric inside a GpuJoin program, numeric join keys |
| `date`, `time`, `timestamp`, `timestamptz` | Comparison within a type, `date` against `timestamp`, `date + int4`, `date - int4`, `date - date`, `date <-> timestamp` casts, `min`/`max`, GROUP BY and join keys | `timestamptz` compared with `date`/`timestamp` (session time zone), `interval` anything, `timetz`, `extract`, `date_trunc`, `timestamptz::date`, `overlaps` |
| `text`, `varchar`, `bpchar` | `=` and `<>` (bytewise; deterministic collations only), `<`, `<=`, `>`, `>=` in the C collation only, `LIKE`/`NOT LIKE` with a constant pattern of the shape `abc`, `abc%`, `%abc`, `%abc%` (UTF-8 or single-byte databases), `length()` in UTF-8, `IS NULL`, `count(col)` | `_` and inner `%` in `LIKE`, `ILIKE`, regular expressions, `upper`, `substr`, concatenation, `IN` lists, text GROUP BY keys and join keys, **any string expression inside a GpuJoin program (including the scan quals of the join's outer relation)** |
| Compressed or TOAST-ed text | Evaluated by the backend: the kernel flags the row and the backend re-runs the whole row on the CPU | (`client_min_messages = debug1` shows "CPU re-evaluation of N tuples") |
| `jsonb` | Quals and aggregate arguments over constant paths: `->` / `->>` with a constant (or column) text key or an integer index (negative from the end), chains of them, `(x ->> 'k')::numeric / int2 / int4 / int8`, `?` (`jsonb_exists`), `jsonb_typeof`, `@>` against a constant object / array of scalars (nested objects, arrays of scalars, numbers compared by value), `IS [NOT] NULL` of any of these. Numbers that do not fit `{ int64, scale }`, the text of a number or of a container, a string cast to a number, and compressed or out-of-line documents are evaluated by the backend (per row, exactly) | A jsonb or text value *computed* from a document in the target list (`SELECT d->>'k'`, `d->'g'`: the column is read and the CPU evaluates it), float casts, `=` and ordering of jsonb, `#>`, `?|`, `?&`, `<@`, `@>` with a pattern containing arrays of arrays / raw scalars / the empty array, jsonb parameters, jsonb in a GpuJoin program (a jsonb column still passes through a join), jsonb in Arrow tables (a jsonb column of a GPU-Cache is read, see GPU-Cache) |
| arrays, ranges, `inet`, `macaddr`, geometry, everything else | pass-through only | everything |

`CASE`, `COALESCE`, `GREATEST`/`LEAST`, `IN (...)`, `IS TRUE`/`IS DISTINCT FROM` (`BooleanTest`,
`DistinctExpr`), and `CoerceViaIO` other than `(jsonb ->> key)::numeric/int` are not pushed down (`codegen.c` in the Vulkan build).

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
  `timestamptz`, or `numeric` (keys of type bool, float, text, varchar, bpchar and uuid, and three or more keys, are not offloaded). **GROUP BY is only worth it for up to about a hundred
  groups:** with the specialized kernels (plan A6) 5.7x faster than 8-process PostgreSQL at 7 groups and 3.9x at
  100 (dense integer keys; a direct-mapped table), 3.5x for a GROUP BY over a join; with the interpreter
  alone 1.2x, 0.8x and 0.3x at 7, 100 and 1,000 groups (most rows overflow to a CPU path). The planner keeps
  queries with more than `pg_strom.gpupreagg_max_groups` (default 128) estimated groups on PostgreSQL.
- Aggregation over a join runs on the device as well (`GpuPreAgg` on the join; plan A4): without GROUP BY
  4.8x faster than 8-process PostgreSQL with the specialized kernels (28 ms against 136 ms on 16.8M rows).
  This includes RIGHT and FULL joins: the matched rows are aggregated on the device and the inner rows
  without a partner are joined with NULL outer columns by the backend, which adds them to the partial
  states (a RIGHT/FULL join combined with a GiST join, and GPU-Sort over a RIGHT/FULL join, stay on the CPU).
- GpuPreAgg and GpuJoin run directly on an Arrow or Parquet outer source (`Custom Scan (GpuPreAgg) on <arrow table>`,
  GpuJoin whose outer relation is the Arrow table, and aggregation over such a join), with the same shapes as for heap
  tables. Quals, join keys and group keys may read integer, float4, bool, date, time, timestamp (every unit), utf8,
  large_utf8 and `char(n)` columns; the Arrow date/time units are converted exactly as the CPU does. Decimal, Interval,
  List, Struct, Date[ms], inet and macaddr columns keep the table on the CPU. Numeric *expressions* over integer or float columns (quals, projections, aggregate arguments) run on the GPU for a scan and an aggregation (not inside a join program); a row the kernel cannot decide (numeric beyond 64 bits, division by zero, float4 underflow) is re-evaluated by the backend from the file (A12).

## GpuJoin

- Hash joins only. `INNER`, `LEFT`, `RIGHT`, `FULL`, semi (`EXISTS`, `IN`) and anti (`NOT EXISTS`, and
  `NOT IN` where PostgreSQL 18+ plans an anti join) joins; up to **four inner relations in one GpuJoin**
  (shown as `[1]` to `[4]`).
- Join keys: integers and the four temporal types (a `bool` key stays on the CPU: `booleq` has no device function; measured by `test/gpuserv/coverage_test.sh`). Any pushed-down join qualifier over the VM's
  types, scan quals on the outer side, projections mixing outer and inner columns (pass-through of
  any type included).
- Stays on PostgreSQL: nest-loop joins; non-integer keys (float, numeric, text, varchar, bpchar, uuid, bool); an outer join with a qualifier coming from the WHERE clause; RIGHT and FULL
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
| Aggregation without GROUP BY | 35-67 ms vs 115-172 ms; with a specialized kernel 37-41 ms vs 132-183 ms (16.8M rows, whatever the quals) | |
| GROUP BY 7 / 100 dense integer keys | 44 ms / 73 ms vs 249 ms / 288 ms | specialized, direct-mapped table |
| GROUP BY 1,000 keys | 855 ms vs 285 ms (slower) | most rows overflow to the CPU; the planner keeps it on PostgreSQL |
| Aggregate / GROUP BY over a join (1,000-row dimension) | 28 ms / 91 ms vs 136 ms / 318 ms | |
| Selective Arrow scan (2.2 GB file) | 59-65 ms vs 196-367 ms | an unselective one loses to 8 workers |

Nothing is known about speed on other GPUs. The acceptance criterion in the plan is relative to
PostgreSQL's parallel scan on the same machine, not to CUDA.

## What is not implemented (and where it is tracked)

| Feature | State |
|---|---|
| GPU-Sort of scan / join results, window functions other than the rank filter | Not implemented (plan A7 status); GPU-Sort of GpuPreAgg groups works, see the table above |
| GPU-Cache | Works, see the section above; differences from the CUDA product are listed there |
| Numeric in joins, numeric `min`/`max` | Not done (plan A6 "known gaps") |
| Specialized kernels for opcodes the generator does not list (new ones until reviewed), for jsonb | They run on the interpreter (plan A6 "kernel specialization") |
| PostGIS `geometry` | The 2D bounding-box operators `&&`, `~`, `@` (also with an empty, infinite or huge-coordinate operand, exactly as PostGIS 3.7 answers them), and `ST_Contains`, `ST_Covers`, `ST_Intersects`, `ST_Within`, `ST_CoveredBy` of a polygon or multipolygon (2D, same SRID) and a point, a point column or the fused `ST_MakePoint(x, y)` / `ST_Point(x, y)` (optionally under `ST_SetSRID` of a constant), and `ST_DWithin(point, point, tolerance)` of two points (a column, a constant or a fused `ST_MakePoint`, any `float8` tolerance; Z and M are ignored, as PostGIS does), `ST_Z` and `ST_M` of a point column, constant or parameter (NULL for a point without that dimension and for an empty point; NaN, Infinity and -0 as stored), the 3- and 4-argument `ST_MakePoint(x, y, z[, m])` (fused like the 2-argument form; the point is NULL when any argument is, and the 2D functions ignore Z and M), `ST_3DDWithin(point, point, tolerance)` of two points that both have a Z (a column, a constant or a fused `ST_MakePoint(x, y, z[, m])`) and `ST_3DIntersects(point, point)` (the same with the tolerance 0), the n-D box operator `&&&` (the 2D boxes, plus the Z ranges when both operands have a Z and the M ranges when both have an M; a Z against an M is not compared; a point with a coordinate beyond the float4 range is re-checked), and the header functions `ST_HasZ`, `ST_HasM`, `ST_NDims`, `ST_Zmflag` and `ST_CoordDim` of a geometry of any type, `ST_Distance`, `ST_3DDistance` and `ST_3DMaxDistance` of two points and `ST_Length`, `ST_Length2D`, `ST_3DLength`, `ST_Perimeter`, `ST_Perimeter2D` and `ST_3DPerimeter` of a line, polygon, multi-geometry or a collection of those (a member adds its own total to its container's running total, as PostGIS does), `ST_Distance` of a point and a line, polygon (with holes), multi-geometry or collection (2D; a point near an edge of a polygon, where the winding test cannot tell, is re-checked), `ST_DWithin` of a point and a line, polygon, multi-geometry or collection (decided as `ST_Distance <= tolerance` with the same exact distance, either argument order, Z and M of the geometry ignored; an empty geometry is false; the planner first calls the installed PostGIS on fixed lines and polygons at the distance and its float neighbours and keeps the function on the CPU if one answer differs), `ST_3DDistance` and `ST_3DDWithin` of a point with a Z and a line, point, multi-line, multi-point or collection of those that have a Z (PostGIS' `lw_dist3d_pt_seg`: `r = num / den` with fused sums, the end for `r < 0` or `r > 1`, else the foot `fma(r, B - A, A)`; 0 differences in 213,000 probe rows, `test/gpuserv/postgis/seg3d_probe.c`; a 3D polygon, an M-only or 2D line and a point without Z are the CPU's), a distance beyond `FLT_MAX` (PostGIS starts its minimum there: `ST_Distance` is then NULL and `ST_DWithin` true for a tolerance of `FLT_MAX` or more, `postgis/fltmax_probe.sql`) is re-checked in all of these (two non-points always are), and `ST_DWithin` / `ST_3DDWithin` of two points decided exactly as "distance <= tolerance" (the float8 values are PostGIS' own bits: it evaluates `sqrt(fma(dz, dz, fma(dx, dx, dy*dy)))` per segment and adds them left to right, which the kernel reproduces with a software fused multiply-add; the planner first calls the installed PostGIS on fixed pairs and lines and keeps the functions on the CPU if one bit differs, as a build without floating-point contraction would; an empty point gives NULL, a SRID mismatch, a 3D distance with an operand without Z, a non-finite coordinate, an overflow and every other geometry type go back to the CPU), in quals and projections of heap tables, Arrow_Fdw / Parquet tables and tables with a GPU-Cache (a row the kernel cannot decide is handed to the backend as its Arrow row or as a heap tuple built from the cache slot, see the section on Arrow and GPU-Cache below). The kernel decides a point that is clearly inside or outside (non-zero winding number, holes, multipolygons in PostGIS's order); a point on an edge or a vertex, too close to an edge to be sure, a Z/M or geodetic polygon, another SRID (PostGIS raises an error), another geometry type, or non-finite or extreme coordinates is re-checked on the CPU; `ST_DWithin` and `ST_3DDWithin` compare the exact distance (the value `ST_Distance` returns) with the tolerance when the installed PostGIS decides that way (checked by the planner at its first use), and otherwise only decide a pair that is beyond a relative 2^-47 band around the tolerance; a negative tolerance (PostGIS raises) or a NaN or infinite one is re-checked, as is every row of `ST_3DDWithin` with an operand without Z (PostGIS then measures in 2D and raises a NOTICE), an empty point and a different SRID, and `ST_Z` / `ST_M` re-check anything that is not a plain point (PostGIS raises "Argument to ST_Z() must have type POINT"). The geometry value passes through | `ST_Distance` / `ST_DWithin` of two non-points, `ST_3DDistance` / `ST_3DDWithin` of a point and a polygon, the other 3D-only functions (`ST_3DClosestPoint`, `ST_3DDWithin`, `ST_3DIntersects` and `ST_3DDistance` of lines and polygons), `ST_Z(ST_MakePoint(...))` (a fused point has no register), geometry in a join program (except the GiST join below) |
| jsonb | Quals, aggregate arguments and number extraction run on the GPU; computed jsonb/text results in the target list, equality, `#>` and the other operators stay on the CPU (plan A8 status) |
| GPU-Direct Storage, `heterodb-extra`, SELECT INTO direct, multi-GPU | Removed or never present on this branch |
| GiST-index spatial join (`pg_strom.enable_gpugistindex`, off by default) | `inner.geom && ST_MakePoint(x, y)` against a `gist_geometry_ops_2d` index of a plain heap table (inner, LEFT, RIGHT, FULL, semi and anti joins; inner quals run on the CPU): the service walks a copy of the index pages for each outer row. Also `ST_Contains(inner.geom, point)`, `ST_Within(point, inner.geom)` and `ST_Intersects` in either order, the point being `ST_MakePoint(x, y[, z[, m]])` of outer columns (optionally under `ST_SetSRID`): the index answers `geom && point` and the join program runs the function on the inner geometry; a row it cannot decide (Z/M polygon, line, mixed SRID; inner values are decompressed and fetched from the TOAST table when the inner relation is loaded) is joined by the backend on the CPU, so the answer is PostGIS's. Also `ST_DWithin` and `ST_3DDWithin` of an inner point column and such a point, with the tolerance an expression of the outer row: the walk grows the point's box by the tolerance and the join program decides each pair with the exact distance; inner lines, polygons and empty geometries, and a negative, NaN or infinite tolerance, are joined by the backend on the CPU (the same functions on a line or polygon run on the GPU as scan quals but not inside a join kernel: the point-to-geometry walker made the join kernel 4.3 MB larger and Metal's shader compiler, which needs 294 s for the join kernel today, gave up or ran past 900 s; geometry is 8.8 of the join kernel's 9.6 MB, so the way out is moving geometry out of the default join kernel, plan A13 slice 4; until then keep `pg_strom.enable_gpugistindex` off for relations with lines or polygons: measured 34 s against 0.23 s in PostgreSQL for 20,000 points against 20,000 lines/polygons; the scan quals of the same functions are correct on the GPU but not faster than PostgreSQL either, 0.3-0.5 s per 1M rows against 0.14-0.3 s on one core). Aggregation directly on top of such a join (`GpuPreAgg`, ungrouped or `GROUP BY` on up to two integer/date-time keys, inner, LEFT, semi and anti joins) runs on the GPU too: a first (scout) pass finds the outer rows the kernel cannot decide, the aggregating kernels skip them, and the backend joins and aggregates those rows itself, so the result is PostGIS's (RIGHT and FULL joins and GPU-Sort stay on the CPU; an Arrow, Parquet or GPU-Cache outer relation works the same way, the undecided rows being named by record batch and row or sent as cache tuples). A walk that exceeds `pg_strom.gpugist_max_steps`, a non-finite probe point and an index that keeps changing while it is copied are errors, never short results (plan A13) |
| PostGIS functions other than the ones above | Not on the GPU yet (plans A11, A13) |
| Row-level CPU recheck on Arrow_Fdw, Parquet and GPU-Cache sources | An instruction that cannot decide a row (a geometry point on an edge, a numeric that does not fit 64 bits, ...) hands that row to the backend on every source, as for heap tables: the service names an Arrow / Parquet row by (record batch, row) and the backend decodes it from the file; a GPU-Cache row is sent as a heap tuple built from the cache slot (the cache is not a snapshot, so the backend cannot fetch the version by ctid). Scan, `GpuPreAgg` (with and without GROUP BY), `GpuJoin` and aggregation over a GiST join; `EXPLAIN ANALYZE` shows the rows in `Fallback-stats`. In an *aggregation over a join* (`GpuPreAgg` with the join folded in) the same holds on every source: an outer row whose scan quals cannot be decided on the GPU (float4 results that are denormal or may underflow, a numeric division by zero or beyond 64 bits, ...) is flagged by the aggregating kernel and handed to the backend, which joins and aggregates it on the CPU; when the join walk or the projection can recheck, a first (scout) pass names those rows and the aggregating kernel skips them, so nothing is counted twice (plan A14). A plain `GpuJoin` hands such rows to the backend too. |
| fp64 hardware, 64-bit-atomic kernels for devices that have them | Not used (plan A3b); `float8` is computed in integer arithmetic on every device |
