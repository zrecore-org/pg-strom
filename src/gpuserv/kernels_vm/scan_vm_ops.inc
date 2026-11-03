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
/* reasons of an error */
SCANVM_OP(ERR_ZERODIV,  1)
SCANVM_OP(ERR_INTRANGE, 2)
SCANVM_OP(ERR_FLTOVER,  3)
SCANVM_OP(ERR_FLTUNDER, 4)
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
