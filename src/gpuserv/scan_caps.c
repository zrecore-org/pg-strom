/*
 * scan_caps.c
 *
 * What the scan VM can execute: classification of PostgreSQL function opcodes by name.
 * Shared by the planner side (codegen.c refuses to push down anything else) and by the
 * service's program builder (scan_prog.c), so the two cannot disagree.
 */
#include <string.h>
#include "xpu_common.h"
#include "scan_caps.h"

/* ----------------------------------------------------------------
 * opcode names (to classify the many intNNxx operators by name)
 * ----------------------------------------------------------------
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			[FuncOpCode__##NAME] = #NAME,
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	[FuncOpCode__##NAME] = #NAME,
static const char *const func_names[FuncOpCode__LoadVars] = {
#include "xpu_opcodes.h"
};
#undef FUNC_OPCODE
#undef DEVONLY_FUNC_OPCODE

const char *
scan_vm_opcode_name(uint32_t opcode)
{
	switch (opcode)
	{
		case FuncOpCode__ConstExpr:		return "ConstExpr";
		case FuncOpCode__ParamExpr:		return "ParamExpr";
		case FuncOpCode__VarExpr:		return "VarExpr";
		case FuncOpCode__BoolExpr_And:	return "BoolExpr_And";
		case FuncOpCode__BoolExpr_Or:	return "BoolExpr_Or";
		case FuncOpCode__BoolExpr_Not:	return "BoolExpr_Not";
		default:
			if (opcode < FuncOpCode__LoadVars && func_names[opcode])
				return func_names[opcode];
	}
	return "?";
}


static int
int_width_of_digit(char c)
{
	switch (c)
	{
		case '1': return 8;
		case '2': return 16;
		case '4': return 32;
		case '8': return 64;
	}
	return 0;
}

static int
cmp_kind_of(const char *s)
{
	if (!strcmp(s, "eq")) return SOP_CMP_EQ;
	if (!strcmp(s, "ne")) return SOP_CMP_NE;
	if (!strcmp(s, "lt")) return SOP_CMP_LT;
	if (!strcmp(s, "le")) return SOP_CMP_LE;
	if (!strcmp(s, "gt")) return SOP_CMP_GT;
	if (!strcmp(s, "ge")) return SOP_CMP_GE;
	return -1;
}

static int
arith_kind_of(const char *s)
{
	if (!strcmp(s, "pl"))  return SOP_ARITH_PL;
	if (!strcmp(s, "mi"))  return SOP_ARITH_MI;
	if (!strcmp(s, "mul")) return SOP_ARITH_MUL;
	if (!strcmp(s, "div")) return SOP_ARITH_DIV;
	if (!strcmp(s, "mod")) return SOP_ARITH_MOD;
	return -1;
}

/* "int4lt", "int48pl", "float4mul", "int4_to_int8", "booleq", "int4um" ... */
scan_func_class
scan_vm_classify_func(uint32_t opcode)
{
	scan_func_class	fc = {0, 0, 0};
	const char *name = (opcode < FuncOpCode__LoadVars ? func_names[opcode] : NULL);
	int			w1, w2, k;

	if (!name)
		return fc;
	if (!strcmp(name, "booleq")) { fc.cls = SOP_CMPI; fc.kind = SOP_CMP_EQ; return fc; }
	if (!strcmp(name, "boolne")) { fc.cls = SOP_CMPI; fc.kind = SOP_CMP_NE; return fc; }
	if (!strcmp(name, "int4_to_bool")) { fc.cls = SOP_CAST; fc.width = 1; return fc; }
	if (!strncmp(name, "float4", 6))
	{
		const char *rest = name + 6;

		/* float4_to_intN */
		if (!strncmp(rest, "_to_int", 7) && (w1 = int_width_of_digit(rest[7])) != 0 && rest[8] == '\0')
		{
			fc.cls = SOP_F2I;
			fc.width = w1;
			return fc;
		}

		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPF;
			fc.kind = k;
		}
		else if ((k = arith_kind_of(rest)) >= 0 && k != SOP_ARITH_MOD)
		{
			fc.cls = SOP_FARITH;
			fc.kind = k;
		}
		else if (!strcmp(rest, "up"))
		{
			fc.cls = SOP_MOV;
		}
		return fc;
	}
	if (strncmp(name, "int", 3) != 0 || (w1 = int_width_of_digit(name[3])) == 0)
		return fc;
	{
		const char *rest = name + 4;

		/* intA_to_float4 */
		if (!strcmp(rest, "_to_float4"))
		{
			fc.cls = SOP_I2F;
			return fc;
		}
		/* intA_to_intB */
		if (!strncmp(rest, "_to_int", 7) && (w2 = int_width_of_digit(rest[7])) != 0 && rest[8] == '\0')
		{
			fc.cls = SOP_CAST;
			fc.width = w2;
			return fc;
		}
		/* intAB<op>: the result has the wider width */
		w2 = int_width_of_digit(rest[0]);
		if (w2 != 0)
			rest++;
		else
			w2 = w1;
		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPI;
			fc.kind = k;
		}
		else if ((k = arith_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_ARITH;
			fc.kind = k;
			fc.width = (w1 > w2 ? w1 : w2);
		}
		else if (!strcmp(rest, "up") && w2 == w1)
		{
			fc.cls = SOP_MOV;
		}
		else if (!strcmp(rest, "um") && w2 == w1)
		{
			fc.cls = SOP_ARITH;			/* 0 - x */
			fc.kind = SOP_ARITH_MI;
			fc.width = w1;
			fc.cls = -SOP_ARITH;		/* marks the unary minus */
		}
	}
	return fc;
}


bool
scan_vm_func_supported(uint32_t opcode)
{
	return scan_vm_classify_func(opcode).cls != 0;
}
