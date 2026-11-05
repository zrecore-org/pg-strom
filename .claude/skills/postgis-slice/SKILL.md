---
name: postgis-slice
description: Checklist for adding or changing PostGIS functionality on the PG-Strom GPU path, including the user's standing rule that every slice also covers Z, M and ZM geometries and the 3D function variants. Use when adding a geometry function, operator or index feature.
---

# Adding a PostGIS function to the GPU path

The user's standing rule: **as PostGIS support is added, the 3D functions must work too** - every slice gets Z/M/ZM tests
and the 3D variants of the function, not just the 2D case. State explicitly what is left on the CPU.

## Steps

1. **Probe PostGIS first.** Write `test/gpuserv/postgis/<name>_probe.sql` with the edge cases (empty, NULL, missing
   dimension, SRID mismatch, NaN/Inf, boundary within 2^-47 relative on squared distances), run it on PostGIS 3.7 and save the
   filtered output as `<name>_probe.out` (drop NOTICE spam). The GPU must reproduce these answers; where PostGIS raises a
   NOTICE or an error the GPU path must hand the row to the CPU (`SCANVM_RECHECK_CODE`, heap scans only).
2. **Registration chain** (in this order):
   - `src/xpu_opcodes.h`: `FUNC_OPCODE` / `__FUNC_OPCODE` with the argument types.
   - `src/gpuserv/scan_caps.c`: classify it (`classify_geometry`) into a `SOP_GEOM*` kind.
   - `src/codegen.c`: operand shapes (`vulkan_geom_operand_shape`, fused `st_makepoint` with 2-4 args), `vulkan_text_func_allowed`
     (refuse in join programs or on Arrow/GPU-Cache sources that have no recheck), `geometry_func_walker`.
   - `src/gpuserv/scan_prog.c`: `emit_*` (use `fused_y_reg` for fused points; registers must be <= 0x1f where packed).
   - `src/gpuserv/kernels_vm/scan_vm_ops.inc` (new `SCANVM_OP`), `scan_common.slang` (the op), `cpu_vm.c` (the C twin),
     `specialize.c` (`op_known`).
3. **Tests** in `test/gpuserv/postgis_tests.inc` / `postgis3d_tests.inc` (helpers `run`, `run_any`, `refused`, `pushed`;
   `pushed` asserts the function name appears under "GPU Scan Quals"). Cover: 2D, Z, M, ZM columns, fused 3-/4-argument points,
   NULL and empty, boundary tables (`pg_3dn`, `pg_3dq`), a GPU-Cache table (run_any), a join qual, the ladder, GiST join over Z/M/ZM.
4. **Run** `postgis3d_dev.sh` and `postgis_dev.sh` (see the vulkan-verify skill), `spec_compile_check.sh`, then the full verification.
5. **Docs**: `doc/features.md` (function table, "not yet" cells) and the status section of `.claude/hw-acceleration.md`.

## Known limits (as of 2026-11-05)

On the GPU: ST_Z, ST_M, 3-/4-argument ST_MakePoint (fused), ST_3DDWithin and ST_3DIntersects of two points, ST_DWithin, ST_Contains/
ST_Intersects point-in-polygon, `&&`, `&&&` (n-D boxes), the header functions ST_HasZ/ST_HasM/ST_NDims/ST_Zmflag/ST_CoordDim, GiST
join on `&&` (2D opfamily). Still on the CPU, with the reason: ST_3DDistance (PostGIS differs from plain binary64 in 10.9% of random
pairs: `postgis/threed_probe.out`), ST_3DLength (line parsing, exact summation order), GiST join with ST_Contains/ST_DWithin (A13 slices
2-3, needs inner geometry registers in join programs).

Pitfalls: a new geometry opcode must be in `emit()`'s feature classification in `scan_prog.c` (explicit list, not a range: NULLMRG sits
at 136) and inside the `VM_NO_GEOM` guard of `scan_common.slang`, or the program runs on a lite kernel and fails with "malformed
program". PostGIS' n-D box code (`&&&`) treats coordinates beyond the float4 range differently from the 2D box: send them to the CPU.
