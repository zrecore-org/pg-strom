// Opcode-switch VM (Slang). Moved from the spike (spike/vulkan-apple) into the service in A1;
// the heap-page mode (VM_HEAP, needs heap_common.slang) arrives with the scan milestone.
//
//
// Replaces the CUDA interpreter's function-pointer vtables and recursive
// EXEC_KERN_EXPRESSION with a flat register-machine program evaluated by a
// switch loop. Opcode numbers come from the real xpu_opcodes.h via the
// extracted enums in opcodes.slang (see gen_opcodes.sh).
//
// Program buffer layout (uint words):
//   [0] ninsn   [1] result register   then ninsn * 6 words:
//   { op (FuncOpCode), dst, a, b, imm_lo, imm_hi }
//
//   ConstExpr   : a = TypeOpCode, b = 1 for NULL, imm = value (float4: bits)
//   VarExpr     : a = TypeOpCode, b = byte offset in row, imm_lo = column
//                 (null-bit index), imm_hi = byte offset of the null bitmap
//   BoolExpr_*  : accumulate-and-jump step (see below); b = jump target
//   functions   : dst = f(reg[a], reg[b])
//
// Registers hold sign-extended int64 (bool/int4/int8) or the raw float4 bits.

#include "opcodes.slang"
#ifdef VM_HEAP
// heap-page mode: tuples arrive already deformed (the LoadVars step)
#include "heap_common.slang"
#else
#include "common.slang"
#endif

#ifdef VM_PROG_UBO
// program in a uniform buffer: Metal `constant` address space, uniform cached path
struct ProgBlock { uint4 w[1024]; };
[[vk::binding(2, 0)]] ConstantBuffer<ProgBlock> progc;
uint pw(uint i) { return progc.w[i >> 2][i & 3]; }
#else
[[vk::binding(2, 0)]] StructuredBuffer<uint>    prog;
uint pw(uint i) { return prog[i]; }
#endif

#ifndef VM_MAX_REGS
#define VM_MAX_REGS 16
#endif
static const uint MAX_REGS   = VM_MAX_REGS;
static const uint INSN_WORDS = 6;
static const uint PROG_HDR   = 2;

struct Reg
{
    int64_t v;
    uint    isnull;
};

struct VmResult
{
    uint    err;        // FuncOpCode of the failing function, 0 if none
    uint    isnull;
    int64_t v;
};

int64_t b2l(bool c)
{
    return c ? 1 : 0;
}

// NULL-strict binary operator: result is NULL if either input is NULL
#define STRICT2(OP, EXPR)                                           \
    case OP:                                                        \
    {                                                               \
        Reg x = regs[ra];                                           \
        Reg y = regs[rb];                                           \
        if ((x.isnull | y.isnull) != 0)                             \
            regs[dst].isnull = 1;                                   \
        else                                                        \
        {                                                           \
            regs[dst].isnull = 0;                                   \
            regs[dst].v = (EXPR);                                   \
        }                                                           \
        break;                                                      \
    }

#ifdef VM_HEAP
VmResult vm_eval(Tup tup)
#else
VmResult vm_eval(uint rowbase)
#endif
{
    Reg      regs[MAX_REGS];
    VmResult res;
    uint     ninsn  = pw(0);
    uint     resreg = pw(1);
    uint     ip     = 0;

    res.err = 0;
    res.isnull = 1;
    res.v = 0;
    // The linearizer never reads a register before writing it, but the
    // compiler cannot prove that and a malformed program must not read
    // garbage, so start from NULL.
#ifndef VM_NO_ZERO_INIT
    [unroll]
    for (uint i = 0; i < MAX_REGS; i++)
    {
        regs[i].v = 0;
        regs[i].isnull = 1;
    }
#endif
    while (ip < ninsn)
    {
        uint ib  = PROG_HDR + ip * INSN_WORDS;
        uint op  = pw(ib + 0);
        uint dst = pw(ib + 1);
        uint ra  = pw(ib + 2);
        uint rb  = pw(ib + 3);
        uint ilo = pw(ib + 4);
        uint ihi = pw(ib + 5);

        ip++;
        switch (op)
        {
        case FuncOpCode__ConstExpr:
            regs[dst].v = int64_t(uint64_t(ilo) | (uint64_t(ihi) << 32));
            regs[dst].isnull = rb;
            break;

        case FuncOpCode__VarExpr:
        {
#ifdef VM_HEAP
            // ilo = column index; the value was deformed from the heap tuple already
            regs[dst].isnull = (tup.nulls >> ilo) & 1u;
            switch (ilo)
            {
            case 0:  regs[dst].v = int64_t(tup.a); break;
            case 1:  regs[dst].v = tup.b; break;
            case 2:  regs[dst].v = int64_t(asuint(tup.f)); break;
            default: regs[dst].v = int64_t(tup.d); break;
            }
#else
            // replaces xpu_datum_operators->xpu_datum_heap_read
            uint isnull = (uint(rows[rowbase + ihi]) >> ilo) & 1u;
            regs[dst].isnull = isnull;
            if (isnull == 0)
            {
                switch (ra)
                {
                case TypeOpCode__bool:
                    regs[dst].v = b2l(rows[rowbase + rb] != 0);
                    break;
                case TypeOpCode__int4:
                    regs[dst].v = int64_t(int(ld32(rowbase + rb)));
                    break;
                case TypeOpCode__int8:
                    regs[dst].v = ld64(rowbase + rb);
                    break;
                case TypeOpCode__float4:
                    regs[dst].v = int64_t(ld32(rowbase + rb));
                    break;
                default:
                    res.err = op;
                    return res;
                }
            }
#endif
            break;
        }

        // AND / OR: three-valued logic, evaluated as accumulate-and-jump.
        // dst holds the accumulator (initialised by a preceding ConstExpr),
        // reg[ra] is the next operand, rb is the end-of-expression target.
        case FuncOpCode__BoolExpr_And:
            if (regs[ra].isnull != 0)
                regs[dst].isnull = 1;
            else if (regs[ra].v == 0)
            {
                regs[dst].v = 0;
                regs[dst].isnull = 0;
                ip = rb;
            }
            break;

        case FuncOpCode__BoolExpr_Or:
            if (regs[ra].isnull != 0)
                regs[dst].isnull = 1;
            else if (regs[ra].v != 0)
            {
                regs[dst].v = 1;
                regs[dst].isnull = 0;
                ip = rb;
            }
            break;

        case FuncOpCode__BoolExpr_Not:
            regs[dst].isnull = regs[ra].isnull;
            regs[dst].v = b2l(regs[ra].v == 0);
            break;

        case FuncOpCode__NullTestExpr_IsNull:
            regs[dst].v = b2l(regs[ra].isnull != 0);
            regs[dst].isnull = 0;
            break;

        case FuncOpCode__NullTestExpr_IsNotNull:
            regs[dst].v = b2l(regs[ra].isnull == 0);
            regs[dst].isnull = 0;
            break;

        // int4 -> int8: registers are already sign-extended
        case FuncOpCode__int4_to_int8:
            regs[dst] = regs[ra];
            break;

        // comparison operators
        STRICT2(FuncOpCode__int4eq, b2l(x.v == y.v))
        STRICT2(FuncOpCode__int4ne, b2l(x.v != y.v))
        STRICT2(FuncOpCode__int4lt, b2l(x.v <  y.v))
        STRICT2(FuncOpCode__int4gt, b2l(x.v >  y.v))
        STRICT2(FuncOpCode__int8eq, b2l(x.v == y.v))
        STRICT2(FuncOpCode__int8lt, b2l(x.v <  y.v))
        STRICT2(FuncOpCode__int8gt, b2l(x.v >  y.v))
        // float4: no NaN handling in the prototype (PostgreSQL sorts NaN last)
        STRICT2(FuncOpCode__float4lt, b2l(asfloat(uint(x.v)) < asfloat(uint(y.v))))
        STRICT2(FuncOpCode__float4gt, b2l(asfloat(uint(x.v)) > asfloat(uint(y.v))))

        // arithmetic with PostgreSQL overflow / zero-divide errors. The CUDA
        // code checks int8 overflow with int128; int64 sign tests suffice.
        case FuncOpCode__int4pl:
        case FuncOpCode__int4mul:
        case FuncOpCode__int4div:
        case FuncOpCode__int8pl:
        {
            Reg x = regs[ra];
            Reg y = regs[rb];

            if ((x.isnull | y.isnull) != 0)
            {
                regs[dst].isnull = 1;
                break;
            }
            int64_t r = 0;
            bool    bad = false;
            if (op == FuncOpCode__int4pl)
            {
                r = x.v + y.v;
                bad = (r < -2147483648 || r > 2147483647);
            }
            else if (op == FuncOpCode__int4mul)
            {
                r = x.v * y.v;      // |x|,|y| < 2^31 so the product fits
                bad = (r < -2147483648 || r > 2147483647);
            }
            else if (op == FuncOpCode__int4div)
            {
                if (y.v == 0)
                    bad = true;                         // division by zero
                else if (y.v == -1)
                {
                    r = -x.v;
                    bad = (r > 2147483647);             // INT_MIN / -1
                }
                else
                    r = int64_t(int(x.v) / int(y.v));   // 32-bit divide
            }
            else    // int8pl
            {
                r = x.v + y.v;
                bad = (((x.v ^ r) & (y.v ^ r)) < 0);
            }
            if (bad)
            {
                res.err = op;
                return res;
            }
            regs[dst].v = r;
            regs[dst].isnull = 0;
            break;
        }

        default:
            res.err = op;       // opcode not implemented in this prototype
            return res;
        }
    }
    res.isnull = regs[resreg].isnull;
    res.v = regs[resreg].v;
    return res;
}

#ifdef VM_HEAP
struct VmEval : ITupEval
{
    bool test(Tup t, out uint err)
    {
        VmResult r = vm_eval(t);
        err = r.err;
        return r.err == 0 && r.isnull == 0 && r.v != 0;
    }
}

[shader("compute")]
[numthreads(HEAP_WG, 1, 1)]
void heap_vm(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    VmEval ev;
    heap_scan<VmEval>(ev, gid, lid);
}
#else
[shader("compute")]
[numthreads(256, 1, 1)]
void vm_main(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    bool hit = false;

    if (gid < kparam.nrows)
    {
        VmResult r = vm_eval(gid * kparam.stride);
        if (r.err != 0)
            report_error(gid, r.err);
        else
            hit = (r.isnull == 0 && r.v != 0);
    }
    emit_matches(gid, hit);
}

#ifndef FIXED_T
#define FIXED_T 1073741824
#endif
// Hand-written kernel for the same predicate as test E1, as the cost baseline
// for the VM:  (b + (int8)a) > 2^30 AND f > 500.0 AND d < 25 AND a > 1000
[shader("compute")]
[numthreads(256, 1, 1)]
void fixed_main(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    bool hit = false;

    if (gid < kparam.nrows)
    {
        uint    base = gid * kparam.stride;
        int     a = int(ld32(base + 0));
        int64_t b = ld64(base + 4);
        float   f = asfloat(ld32(base + 12));
        int     d = int(ld32(base + 16));
        uint    nb = uint(rows[base + 20]);

        hit = ((nb & 0xFu) == 0) && (b + int64_t(a) > FIXED_T) &&
              (f > 500.0f) && (d < 25) && (a > 1000);
    }
    emit_matches(gid, hit);
}
#endif
