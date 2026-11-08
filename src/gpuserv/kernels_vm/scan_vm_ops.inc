/*
 * Instruction numbers of the scan VM, shared by the program builder (scan_prog.c) and
 * the kernel (scan.slang) through SCANVM_OP(name, value).
 *
 * An instruction is 6 words { op, dst, a, b, k0, k1 }; registers hold sign-extended
 * int64 (bool, int1/2/4/8) or the zero-extended bits of a float4.
 *
 *   CONST    dst = (k0 | k1 << 32), NULL if a != 0
 *   AND, OR  accumulate-and-jump: dst = accumulator, a = next operand, b = jump target
 *   NOT, ISNULL, ISNOTNULL   dst = f(a)
 *   MOV      dst = a
 *   CAST     dst = a narrowed to k0 bits (range checked, error id k1); k0 == 1 means ::bool
 *   CMPI     dst = cmp(a, b) on integers;  k0 = SCANVM_CMP_*
 *   CMPF     dst = cmp(a, b) on float4 (NaN sorts last, as in PostgreSQL)
 *   ARITH    dst = a op b on integers;  k0 = SCANVM_ARITH_* | width << 8; k1 = error id
 *   FARITH   dst = a op b on float4;    k0 = SCANVM_ARITH_*; k1 = error id

 *   HASHINIT dst = 0xffffffff;  HASHMERGE dst = merge(dst, hash_any(a, k0 bytes)) unless a is NULL;
 *   HASHFIN  dst ^= 0xffffffff   (the hash of a join key, bit-identical to PostgreSQL's hash_any)
 *   I2F      dst = (float4) a;  F2I  dst = rint(a) as an integer of k0 bits (range checked, error id k1)
 *
 *   D2TS     dst = timestamp of the date a (k0 = 0: saturating, for comparisons; k0 = 1: the ::timestamp cast,
 *            error id k1 when out of range)
 *   TS2D     dst = date of the timestamp a (floor of the days; infinities map to the date infinities)
 *   DATEOP   dst = date arithmetic of a and b; k0 = SCANVM_DATEOP_*, error id k1
 *
 * Strings (A6): a text/varchar/bpchar register holds { byte offset | length << 32 | source << 60 } of the
 * varlena payload, the source being the buffer it lives in (0: heap pages of the file, 1: pages the backend
 * loaded, 2: the join inner buffer, 3: the program's constant pool). A value that is compressed or stored
 * out of line (TOAST) cannot be read: the row is marked for re-evaluation by the backend.
 *
 *   CONSTSTR dst = constant string; k0 = byte offset in the pool (words from header word 20), k1 = length
 *   STRCMP   dst = cmp(a, b) as a string comparison (bytewise); k0 = SCANVM_CMP_*, k1 = 1: bpchar (trailing blanks
 *            do not count)
 *   STRLIKE  dst = a matches the literal b; k0 = SCANVM_LIKE_* | negate << 8 | bpchar << 9
 *   STRLEN   dst = number of characters of a (UTF-8; k0 = 1) or bytes (k0 = 0)
 *
 * numeric (A6): a numeric register is an exact decimal { unscaled int64 v, scale sc } (value = v / 10^sc,
 * |v| <= INT64_MAX, sc <= 255; sc is the PostgreSQL dscale, so a result prints as PostgreSQL would print
 * it). NaN, +-Infinity and values that do not fit (or an intermediate result that overflows) mark the row
 * for re-evaluation by the backend (the instruction returns SCANVM_RECHECK_CODE).
 *
 *   NUMCONST dst = constant numeric; a = isnull, b = scale, k0 | k1 << 32 = unscaled value
 *   NUMCMP   dst = cmp(a, b) of two numerics; k0 = SCANVM_CMP_*
 *   NUMARITH dst = a op b (k0 = SCANVM_ARITH_PL / MI / MUL) or, with k0 = SCANVM_NUMOP_NEG / ABS, op a
 *   I2N      dst = (numeric) a, scale 0
 *   N2I      dst = a rounded half away from zero to an integer of k0 bits (a row that does not fit is rechecked)
 *   NUMDIV   dst = a / b (k0 = SCANVM_NUMDIV_*): numeric_div with PostgreSQL's select_div_scale() result scale and
 *            rounding, numeric_mod, div(); division by zero and a quotient that does not fit are rechecked
 *   NUMRND   dst = round / trunc of numeric a to the scale given by the int register b (k0 = SCANVM_NUMRND_*; a
 *            negative scale rounds to tens, ...) or the numeric(p,s) coercion with typmod register b
 *   NUMARITH with k0 = NUMOP_CEIL / FLOOR / SIGN: ceil, floor, sign (scale 0)
 *
 * jsonb (A8): a jsonb register is { byte offset | length << 32 | source << 60 } of a JsonbContainer, or of a scalar
 * inside one, and Reg.sc says which (the JEntry type: 0 string payload, 1 numeric varlena, 2 false, 3 true, 4 JSON
 * null, 5 container). A column loads as kind 5; a compressed or external value is rechecked like a string. A
 * text register with bit 62 of v set is a *deferred* text: the value of ->> that cannot be shown without
 * formatting (a numeric or a container). It can be tested for NULL; every other use hands the row to the backend.
 *
 *   JSONGET    dst = a -> b: k0 = 0: b is a string register (object key), k0 = 1: b is an int register (array
 *              element, negative from the end); NULL if absent or of the wrong shape
 *   JSONTXT    dst = a ->> ...: the text of a jsonb scalar; k0 = pool offset of the literal "truefalse"
 *   JSONEXISTS dst = jsonb_exists(a, b) (the ? operator: key of an object, string element of an array)
 *   JSONTYPE   dst = jsonb_typeof(a); k0 = pool offset of "objectarraystringnumberbooleannull"
 *   JSONNUM    dst = (a ->> ...)::numeric: a numeric scalar as { unscaled, scale }; a JSON null is NULL
 *   JSONINT    dst = (a ->> ...)::intN (k0 = bits): a numeric scalar of scale 0 in range
 *   JSONCONT   dst = a @> pattern (k0 = pool word offset of the compiled pattern, scan_caps.c)
 * Whatever these cannot decide exactly returns SCANVM_RECHECK_CODE.
 *
 * float8 (A10): a float8 register holds the 64 bits of the double. There is no float arithmetic in the kernel: the
 * operations are the integer software binary64 of f64_soft.inc (bit-identical to a hardware double, PostgreSQL's
 * float8 order and error rules), so they run on devices without fp64.
 *
 *   CMPD     dst = cmp(a, b) on float8; k0 = SCANVM_CMP_* (NaN equals NaN and sorts last, -0 = +0)
 *   DARITH   dst = a op b on float8; k0 = SCANVM_ARITH_PL/MI/MUL/DIV, k1 = error id (overflow, underflow, division by zero)
 *   DUNARY   dst = a float8 function of one argument, k0 = SCANVM_DUN_*: -a, abs, sqrt, round (rint: ties to even), ceil,
 *            floor, trunc, sign (PostgreSQL's float.c: -0 stays -0 except in sign, which gives 0 for zero and NaN).
 *            sqrt of a negative number (-infinity included, not -0 or NaN) is the error SCANVM_ERR_SQRTNEG, k1 = error id
 *   I2D      dst = (float8) a (an integer)
 *   D2I      dst = rint(a) as an integer of k0 bits (range checked, error id k1)
 *   F2D      dst = (float8) a (a float4, exact)
 *   D2F      dst = (float4) a (nearest even; overflow and underflow errors, error id k1)
 *
 * PostGIS (A11): a geometry is a string register (the payload of the varlena: GSERIALIZED v1/v2, xpu_postgis.h), so the
 * loaders, constants and the recheck of compressed or external values are the strings'.
 *
 *   GEOMBOX  dst = a && b (k0 = 0), a ~ b (a contains b: 1), a @ b (a within b: 2) on the 2D float bounding boxes
 *            (gserialized_overlaps_2d ...): the cached box, or for a point the box of its coordinates rounded outward
 *            to float4 (a value beyond the float range becomes +-FLT_MAX, as PostGIS stores it); an empty operand
 *            decides: && is false, ~ and @ are true; whatever else the VM cannot box (geodetic, a small line or
 *            polygon without a cached box) returns SCANVM_RECHECK_CODE.
 *
 *   GEOMPIP  dst = a point lies in a (multi)polygon: ST_Contains / ST_Covers / ST_Intersects / ST_Within / ST_CoveredBy
 *            of a polygon and a point. a is the polygon (a string register), b the point: a string register, or with
 *            k0 & SOP_GPIP_FUSED the float8 register of x and the float8 register (k1 & 0xff) of y, the point of a
 *            fused st_makepoint(x, y), whose SRID is k1 >> 8. k0 & SOP_GPIP_SYMM: either operand may be the polygon.
 *            Non-zero winding number over the rings (a hole excludes), the orientation of every edge that spans the
 *            point's y decided on the software binary64 only when its sign is certain (Shewchuk's error bound with a
 *            safety factor); PostGIS evaluates the same test with fused multiply-add in places, so a point on an edge
 *            or too close to decide, an edge that overflows or underflows, non-finite coordinates, a different SRID, a
 *            geodetic, Z or M polygon and every other geometry type return SCANVM_RECHECK_CODE: the three predicates
 *            differ only on the boundary, which the backend always decides. An empty operand (same SRID) is false.
 *
 *   GEOMDWITHIN  dst = ST_DWithin(point, point, tolerance) (2D; Z and M are ignored, as PostGIS does). a and b are the
 *            points: a string register (a column, constant or parameter), or with k0 & SOP_GDW_FUSED_A / _B the float8
 *            register of x (a / b) and of y (byte 1 / byte 2 of k0, bits 8..15 and 16..23) of a fused st_makepoint(x, y);
 *            the tolerance is the float8 register in bits 24..31 of k0; k1 is the SRID of the fused point(s). PostGIS
 *            tests sqrt(dx*dx + dy*dy) <= r with a distance that differs from plain binary64 in 8% of random pairs (it
 *            fuses in places), so the instruction decides only when dx*dx + dy*dy is beyond r*r by a relative 2^-47 and
 *            otherwise (the boundary), for a SRID mismatch (PostGIS raises), a negative, NaN or infinite tolerance, a
 *            non-finite coordinate or intermediate, an underflow, a multipoint, a line or a geodetic operand returns
 *            SCANVM_RECHECK_CODE. An empty point is false.
 *
 *   GEOMZM   dst = ST_Z(point) (k0 = 0), ST_M(point) (k0 = 1), ST_X(point) (2) or ST_Y(point) (3): the float8 bits of the coordinate. NULL (as PostGIS) for a point
 *            without that dimension, an empty point and a NULL geometry; NaN, Infinity and -0 pass through. Every other
 *            geometry type (PostGIS raises "Argument to ST_Z() must have type POINT"), a geodetic point and a malformed value
 *            return SCANVM_RECHECK_CODE: the backend evaluates the row and raises the error in its own order.
 *
 *   GEOMDIST dst = ST_Distance / ST_3DDistance / ST_3DMaxDistance of two points, the float8 bits of PostGIS' own result:
 *            sqrt(fma(dz, dz, fma(dx, dx, dy * dy))) (3D) or sqrt(fma(dx, dx, dy * dy)) (2D), the order the installed PostGIS
 *            computes (200,000 pairs, 0 differences: postgis/dist_probe.c) with the software binary64 (f64_sqr_add is the fused
 *            step). The operands are those of GEOM3DDW (a column, constant, parameter or a fused ST_MakePoint; k0 the flags and Y
 *            registers, k1 the SRID, the Z registers and GDIST_3D). An empty operand gives NULL (as PostGIS); a SRID mismatch
 *            (PostGIS raises), a 3D distance with an operand without Z, non-finite coordinates or an overflow, a line or any
 *            other type, a geodetic point return SCANVM_RECHECK_CODE. The planner calibrates the order against the installed
 *            PostGIS first (codegen.c): a build that computes it differently keeps the functions on the CPU.
 *
 *   GEOMLEN  dst = ST_Length / ST_Length2D (k0 = 0), ST_3DLength (1), ST_Perimeter / ST_Perimeter2D (2) or ST_3DPerimeter (3) of a
 *            geometry: the float8 bits of PostGIS' own result. A line's length is the sum of its segment lengths
 *            sqrt(fma(dz, dz, fma(dx, dx, dy * dy))) (the Z term only for the 3D forms of a geometry with Z) added left to right
 *            (60,000 lines, 0 differences: postgis/len_probe.c); a polygon's perimeter adds its rings; a multi-geometry or a
 *            collection adds what its members return (each member's total first). Points, lines, polygons, their multi-geometries and
 *            a collection of those are read; a collection in a collection, curves, a geodetic value, a malformed one, non-finite
 *            coordinates and an overflow return SCANVM_RECHECK_CODE. Calibrated by the planner like GEOMDIST.
 *            k0 = 4 is ST_Area (R3): lwgeom_area adds the members' areas, lwpoly_area the exterior ring's |signed area| minus the holes'
 *            (a ring of fewer than 3 points is skipped), and PostGIS 3.7's ptarray_signed_area is an adaptive determinant sum that
 *            area_ring.inc reproduces (postgis/area_probe*.sql: 80,000 rings, 0 differences); a ring that needs PostGIS' exact path
 *            (cancellation), rescaling (products beyond 2^990) or has a non-finite value returns SCANVM_RECHECK_CODE.
 *
 *   GEOMFLAG (k0 5..11, R3/R4: ST_SRID from the header; ST_NPoints, ST_NumPoints, ST_NumGeometries, ST_NRings, ST_NumInteriorRings, ST_IsEmpty
 *            by a walk over the members, NULL where PostGIS returns NULL; curves, a collection in a collection: SCANVM_RECHECK_CODE)
 *   GEOMFLAG dst = the flags of a geometry of any type (empty or not): k0 0 ST_HasZ, 1 ST_HasM (boolean), 2 ST_NDims (2 + Z + M),
 *            3 ST_Zmflag (2 * Z + M), 4 ST_CoordDim (= ST_NDims) as smallint. Only the header is read; NULL in, NULL out; a value
 *            shorter than a header returns SCANVM_RECHECK_CODE.
 *
 *   GEOM3DDW dst = ST_3DDWithin(point, point, tolerance), the layout of GEOMDWITHIN with the Z of a fused point in a register
 *            too: bits 0..19 of k1 are the SRID, bits 20..24 the register of the Z of a (fused), bits 25..29 that of b. PostGIS
 *            raises a NOTICE when an operand has no Z value ("the unknown z-value will be regarded as any value") and the
 *            distance is then 2D: such a row, an empty point, a SRID mismatch, a negative, NaN or infinite tolerance,
 *            non-finite coordinates, an overflow or underflow and the boundary (the same 2^-47 band as GEOMDWITHIN, measured on
 *            300,000 pairs: PostGIS' 3D distance is within 3e-16 of sqrt(dx*dx + dy*dy + dz*dz)) return SCANVM_RECHECK_CODE.
 *
 *   NULLMRG  dst = a with the null flag of a or b: st_makepoint(x, y, z[, m]) is NULL when any argument is; the 2D functions
 *            that take the fused point get its y register through this, so the Z and M expressions are computed too (their
 *            errors reach the user).
 *
 * A failing instruction returns k1 | reason << 16 (SCANVM_ERR_*).
 */
SCANVM_OP(CONST,    1)
SCANVM_OP(AND,      2)
SCANVM_OP(OR,       3)
SCANVM_OP(NOT,      4)
SCANVM_OP(ISNULL,   5)
SCANVM_OP(ISNOTNULL,6)
SCANVM_OP(MOV,      7)
SCANVM_OP(CAST,     8)
SCANVM_OP(CMPI,     9)
SCANVM_OP(CMPF,     10)
SCANVM_OP(ARITH,    11)
SCANVM_OP(FARITH,   12)
SCANVM_OP(I2F,      13)
SCANVM_OP(F2I,      14)
SCANVM_OP(HASHINIT, 15)
SCANVM_OP(HASHMERGE,16)
SCANVM_OP(HASHFIN,  17)
SCANVM_OP(D2TS,     40)
SCANVM_OP(TS2D,     41)
SCANVM_OP(DATEOP,   42)
SCANVM_OP(CONSTSTR, 60)
SCANVM_OP(STRCMP,   61)
SCANVM_OP(STRLIKE,  62)
SCANVM_OP(STRLEN,   63)
SCANVM_OP(NUMCONST, 80)
SCANVM_OP(NUMCMP,   81)
SCANVM_OP(NUMARITH, 82)
SCANVM_OP(I2N,      83)
SCANVM_OP(N2I,      84)
SCANVM_OP(NUMDIV,   85)
SCANVM_OP(NUMRND,   86)
SCANVM_OP(JSONGET,    100)
SCANVM_OP(JSONTXT,    101)
SCANVM_OP(JSONEXISTS, 102)
SCANVM_OP(JSONTYPE,   103)
SCANVM_OP(JSONNUM,    104)
SCANVM_OP(JSONINT,    105)
SCANVM_OP(JSONCONT,   106)
SCANVM_OP(CMPD,       120)
SCANVM_OP(DARITH,     121)
SCANVM_OP(DUNARY,     122)
SCANVM_OP(I2D,        123)
SCANVM_OP(D2I,        124)
SCANVM_OP(F2D,        125)
SCANVM_OP(D2F,        126)
SCANVM_OP(GEOMBOX,    130)
SCANVM_OP(GEOMPIP,    131)
/* never emitted: the class of st_makepoint / st_point (kind 0) and st_setsrid (kind 1), which only a GEOMPIP argument may use */
SCANVM_OP(GEOMMK,     132)
SCANVM_OP(GEOMDWITHIN, 133)
SCANVM_OP(GEOMZM,     134)
SCANVM_OP(GEOM3DDW,   135)
SCANVM_OP(NULLMRG,    136)
SCANVM_OP(GEOMFLAG,   137)
SCANVM_OP(GEOMDIST,   138)
SCANVM_OP(GEOMLEN,    139)
/* GEOMPIP: the operands of the function (kind) and the flags of k0 */
SCANVM_OP(GPIP_POLY_POINT,  0)
SCANVM_OP(GPIP_POINT_POLY,  1)
SCANVM_OP(GPIP_EITHER,      2)
SCANVM_OP(GPIP_FUSED,       1)
SCANVM_OP(GPIP_SYMM,        2)
/* GEOMDWITHIN: the flags of k0 */
SCANVM_OP(GDW_FUSED_A,      1)
SCANVM_OP(GDW_FUSED_B,      2)
/* DUNARY kinds (k0) */
SCANVM_OP(DUN_NEG,   0)
SCANVM_OP(DUN_ABS,   1)
SCANVM_OP(DUN_SQRT,  2)
SCANVM_OP(DUN_ROUND, 3)
SCANVM_OP(DUN_CEIL,  4)
SCANVM_OP(DUN_FLOOR, 5)
SCANVM_OP(DUN_TRUNC, 6)
SCANVM_OP(DUN_SIGN,  7)
/* GEOMBOX kinds */
SCANVM_OP(GBOX_OVERLAPS, 0)
SCANVM_OP(GBOX_CONTAINS, 1)
SCANVM_OP(GBOX_WITHIN,   2)
/* &&& (geometry_overlaps_nd): the 2D boxes, and the Z ranges when both operands have a Z, the M ranges when both have an M */
SCANVM_OP(GBOX_OVERLAPS_ND, 3)
/* GEOM3DDW kind 1: ST_3DIntersects(point, point) = ST_3DDWithin with the constant tolerance 0 (the program builder adds it) */
SCANVM_OP(G3DDW_INTERSECTS, 1)
/* GEOMFLAG kinds: ST_HasZ, ST_HasM (boolean), ST_NDims, ST_Zmflag, ST_CoordDim (smallint) */
/* GEOMDIST: bit 30 of k1 is set for the 3D forms (ST_3DDistance, ST_3DMaxDistance of two points); ST_Distance is the 2D one */
SCANVM_OP(GDIST_3D,       0x40000000)
/* GEOMDWITHIN / GEOM3DDW: bit 30 of k1 (the bits above the SRID) says the decision is the exact "distance <= tolerance" (KEXP_FLAG__GEOM_EXACT);
 * without it the kernel decides outside a 2^-47 band around the boundary and re-checks inside it */
SCANVM_OP(GDW_EXACT,      0x40000000)
SCANVM_OP(GFLAG_HASZ,     0)
SCANVM_OP(GFLAG_HASM,     1)
SCANVM_OP(GFLAG_NDIMS,    2)
SCANVM_OP(GFLAG_ZMFLAG,   3)
SCANVM_OP(GFLAG_COORDDIM, 4)
/* GEOMFLAG kinds 5..11 (R3/R4): ST_SRID (header), then the walks of geom_cnt(): ST_NPoints, ST_NumPoints, ST_NumGeometries, ST_NRings,
 * ST_NumInteriorRings, ST_IsEmpty (int4 / boolean) */
SCANVM_OP(GFLAG_SRID,     5)
SCANVM_OP(GFLAG_NPOINTS,  6)
SCANVM_OP(GFLAG_NUMPOINTS, 7)
SCANVM_OP(GFLAG_NUMGEOMS, 8)
SCANVM_OP(GFLAG_NRINGS,   9)
SCANVM_OP(GFLAG_NUMINTRINGS, 10)
SCANVM_OP(GFLAG_ISEMPTY,  11)
/* returned by an instruction that cannot evaluate this row: the backend evaluates it (not an error) */
SCANVM_OP(RECHECK_CODE, 0x7ffffffe)
SCANVM_OP(GIST_NONFINITE_CODE, 0x7ffffffd)  /* A13: a GiST join probe point has a non-finite coordinate (an error, not a recheck) */
SCANVM_OP(GIST_STEPS_CODE, 0x7ffffffc)       /* A13: the index walk of one outer row exceeded the step limit */
/* unary numeric operations (NUMARITH k0) */
SCANVM_OP(NUMOP_NEG, 16)
SCANVM_OP(NUMOP_ABS, 17)
SCANVM_OP(NUMOP_CEIL, 18)
SCANVM_OP(NUMOP_FLOOR, 19)
SCANVM_OP(NUMOP_SIGN, 20)
/* NUMDIV k0: numeric_div, numeric_mod, numeric_div_trunc (div()) */
SCANVM_OP(NUMDIV_DIV, 0)
SCANVM_OP(NUMDIV_MOD, 1)
SCANVM_OP(NUMDIV_TRUNC, 2)
/* NUMRND k0: round(numeric, int4), trunc(numeric, int4), numeric(p,s) typmod coercion */
SCANVM_OP(NUMRND_ROUND, 0)
SCANVM_OP(NUMRND_TRUNC, 1)
SCANVM_OP(NUMRND_TYPMOD, 2)
/* reasons of an error */
SCANVM_OP(ERR_ZERODIV,  1)
SCANVM_OP(ERR_INTRANGE, 2)
SCANVM_OP(ERR_FLTOVER,  3)
SCANVM_OP(ERR_FLTUNDER, 4)
SCANVM_OP(ERR_DATERANGE, 20)    /* date out of range */
SCANVM_OP(ERR_TSRANGE,   21)    /* date out of range for timestamp */
SCANVM_OP(ERR_INFSUB,    22)    /* cannot subtract infinite dates */
SCANVM_OP(ERR_SQRTNEG,   23)    /* cannot take square root of a negative number */
/* date arithmetic kinds */
SCANVM_OP(DATEOP_PLI, 0)        /* date + int4 */
SCANVM_OP(DATEOP_MII, 1)        /* date - int4 */
SCANVM_OP(DATEOP_MI,  2)        /* date - date -> int4 */
/* LIKE shapes of a pattern that is a literal with optional leading/trailing % */
SCANVM_OP(LIKE_EXACT,    0)
SCANVM_OP(LIKE_PREFIX,   1)
SCANVM_OP(LIKE_SUFFIX,   2)
SCANVM_OP(LIKE_CONTAINS, 3)
/* comparison kinds */
SCANVM_OP(CMP_EQ,   0)
SCANVM_OP(CMP_NE,   1)
SCANVM_OP(CMP_LT,   2)
SCANVM_OP(CMP_LE,   3)
SCANVM_OP(CMP_GT,   4)
SCANVM_OP(CMP_GE,   5)
/* arithmetic kinds */
SCANVM_OP(ARITH_PL, 0)
SCANVM_OP(ARITH_MI, 1)
SCANVM_OP(ARITH_MUL,2)
SCANVM_OP(ARITH_DIV,3)
SCANVM_OP(ARITH_MOD,4)
/* value types of a loaded column */
SCANVM_OP(VT_BOOL,  1)
SCANVM_OP(VT_INT,   2)
SCANVM_OP(VT_FLOAT4,3)
SCANVM_OP(VT_NULLONLY,4)   /* any type: only NULL-ness is loaded (IS [NOT] NULL) */
SCANVM_OP(VT_STR,   5)     /* text/varchar/bpchar: a string register (see above) */
SCANVM_OP(VT_NUM,   6)     /* numeric: { unscaled, scale } (see above); not readable -> row rechecked */
SCANVM_OP(VT_JSON,  7)     /* jsonb: a register pointing at the container (see above); not readable -> row rechecked */
SCANVM_OP(VT_FLOAT8,8)     /* float8: the 64 bits of the double (see above) */
