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

/* ----------------------------------------------------------------
 * A6: temporal types and strings
 * ----------------------------------------------------------------
 */

/*
 * date_eq, time_lt, timestamp_ge, timestamptz_ne: plain integer comparison of the stored values
 * (days / microseconds; infinity is INT_MAX/INT_MIN, so the order is the integers' order).
 * date_lt_timestamp and timestamp_lt_date compare after converting the date to a timestamp (width 1:
 * the left side is the date, 2: the right side), which is exact; the comparisons that involve a
 * timestamptz and a date or a timestamp depend on the session time zone and are not implemented.
 * date_pli, date_mii, integer_pl_date, date_mi: date arithmetic. date_to_timestamp, timestamp_date.
 */
static scan_func_class
classify_temporal(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	static const char *const tnames[] = {"timestamptz", "timestamp", "date", "time"};
	int			k;

	if (!strcmp(name, "date_pli"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_PLI; return fc; }
	if (!strcmp(name, "date_mii"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_MII; return fc; }
	if (!strcmp(name, "date_mi"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_MI; return fc; }
	if (!strcmp(name, "integer_pl_date"))
		{ fc.cls = SOP_DATEOP; fc.kind = SOP_DATEOP_PLI; fc.width = 1; return fc; }	/* arguments swapped */
	if (!strcmp(name, "date_to_timestamp"))
		{ fc.cls = SOP_D2TS; fc.width = 1; return fc; }		/* the cast: out of range is an error */
	if (!strcmp(name, "timestamp_date"))
		{ fc.cls = SOP_TS2D; return fc; }
	for (size_t i = 0; i < sizeof(tnames) / sizeof(tnames[0]); i++)
	{
		size_t		n = strlen(tnames[i]);
		const char *rest;

		if (strncmp(name, tnames[i], n) != 0 || name[n] != '_')
			continue;
		rest = name + n + 1;
		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_CMPI;
			fc.kind = k;
			return fc;
		}
		/* date <op> timestamp, timestamp <op> date */
		if (!strcmp(tnames[i], "date") || !strcmp(tnames[i], "timestamp"))
		{
			int			left_is_date = !strcmp(tnames[i], "date");
			const char *other = (left_is_date ? "timestamp" : "date");
			char		op[3];

			if (strlen(rest) > 3 && rest[2] == '_' && strcmp(rest + 3, other) == 0)
			{
				memcpy(op, rest, 2);
				op[2] = '\0';
				if ((k = cmp_kind_of(op)) >= 0)
				{
					fc.cls = SOP_CMPI;
					fc.kind = k;
					fc.width = (left_is_date ? 1 : 2);		/* which side is the date */
					return fc;
				}
			}
		}
		break;
	}
	return fc;
}

/* text_eq, bpcharne, text_lt (C collation only: the planner checks), textlen, textlike ... */
static scan_func_class
classify_string(const char *name)
{
	scan_func_class fc = {0, 0, 0};
	int			k;

	if (!strncmp(name, "bpchar", 6))
	{
		const char *rest = name + 6;

		if ((k = cmp_kind_of(rest)) >= 0)
		{
			fc.cls = SOP_STRCMP; fc.kind = k; fc.width = 1;
		}
		else if (!strcmp(rest, "like") || !strcmp(rest, "nlike"))
		{
			fc.cls = SOP_STRLIKE; fc.kind = (rest[0] == 'n'); fc.width = 1;
		}
		return fc;
	}
	if (!strncmp(name, "text_", 5) && (k = cmp_kind_of(name + 5)) >= 0)
	{
		fc.cls = SOP_STRCMP; fc.kind = k;
	}
	else if (!strcmp(name, "like") || !strcmp(name, "textlike"))
	{
		fc.cls = SOP_STRLIKE; fc.kind = 0;
	}
	else if (!strcmp(name, "notlike") || !strcmp(name, "textnlike"))
	{
		fc.cls = SOP_STRLIKE; fc.kind = 1;
	}
	else if (!strcmp(name, "textlen"))
	{
		fc.cls = SOP_STRLEN;
		fc.width = 1;				/* characters of a UTF-8 string (the planner allows it for UTF8 only) */
	}
	return fc;
}

/*
 * The shape of a LIKE pattern the VM can match bytewise: a literal, optionally with % before and
 * after it (the backslash escapes a character). The unescaped literal is stored in lit (at least
 * plen bytes). Returns SOP_LIKE_*, or -1 if the pattern has '_' or an inner '%'.
 */
int
scan_vm_like_shape(const unsigned char *pat, int plen, unsigned char *lit, int *p_litlen)
{
	int			i = 0, n = 0, lead = 0, trail = 0;

	while (i < plen && pat[i] == '%')
	{
		lead = 1;
		i++;
	}
	while (i < plen)
	{
		unsigned char c = pat[i];

		if (c == '\\')
		{
			if (++i >= plen)
				return -1;			/* PostgreSQL raises an error */
			lit[n++] = pat[i++];
		}
		else if (c == '%')
			break;
		else if (c == '_')
			return -1;
		else
		{
			lit[n++] = c;
			i++;
		}
	}
	while (i < plen && pat[i] == '%')
	{
		trail = 1;
		i++;
	}
	if (i < plen)
		return -1;					/* a % between literals */
	*p_litlen = n;
	if (!lead)
		return trail ? SOP_LIKE_PREFIX : SOP_LIKE_EXACT;
	return trail ? SOP_LIKE_CONTAINS : SOP_LIKE_SUFFIX;
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
	if ((fc = classify_temporal(name)).cls != 0)
		return fc;
	if ((fc = classify_string(name)).cls != 0)
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

bool
scan_vm_agg_action_supported(int action)
{
	switch (action)
	{
		case KAGG_ACTION__NROWS_ANY:		/* count(*) */
		case KAGG_ACTION__NROWS_COND:		/* count(x) */
		case KAGG_ACTION__PMIN_INT32:		/* min(int1/2/4) */
		case KAGG_ACTION__PMIN_INT64:		/* min(int8) */
		case KAGG_ACTION__PMAX_INT32:
		case KAGG_ACTION__PMAX_INT64:
		case KAGG_ACTION__PSUM_INT:			/* sum(int1/2/4) */
		case KAGG_ACTION__PSUM_INT64:		/* sum(int8) */
		case KAGG_ACTION__PAVG_INT:			/* avg(int1/2/4) */
		case KAGG_ACTION__PAVG_INT64:		/* avg(int8) */
			return true;
		default:
			break;
	}
	return false;
}

bool
scan_vm_groupkey_type_supported(int type_code)
{
	switch (type_code)
	{
		case TypeOpCode__bool:
		case TypeOpCode__int1:
		case TypeOpCode__int2:
		case TypeOpCode__int4:
		case TypeOpCode__int8:
		/* A6: dates and times are integers; their device hash and equality are those of int4 / int8 */
		case TypeOpCode__date:
		case TypeOpCode__time:
		case TypeOpCode__timestamp:
		case TypeOpCode__timestamptz:
			return true;
		default:
			break;
	}
	return false;
}
