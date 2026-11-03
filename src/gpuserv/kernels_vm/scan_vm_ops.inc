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
/* returned by an instruction that cannot evaluate this row: the backend evaluates it (not an error) */
SCANVM_OP(RECHECK_CODE, 0x7ffffffe)
/* unary numeric operations (NUMARITH k0) */
SCANVM_OP(NUMOP_NEG, 16)
SCANVM_OP(NUMOP_ABS, 17)
/* reasons of an error */
SCANVM_OP(ERR_ZERODIV,  1)
SCANVM_OP(ERR_INTRANGE, 2)
SCANVM_OP(ERR_FLTOVER,  3)
SCANVM_OP(ERR_FLTUNDER, 4)
SCANVM_OP(ERR_DATERANGE, 20)    /* date out of range */
SCANVM_OP(ERR_TSRANGE,   21)    /* date out of range for timestamp */
SCANVM_OP(ERR_INFSUB,    22)    /* cannot subtract infinite dates */
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
