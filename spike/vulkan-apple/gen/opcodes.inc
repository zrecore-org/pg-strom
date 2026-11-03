#define TYPE_OPCODE(NAME,a,b)		TypeOpCode__##NAME,
typedef enum {
	TypeOpCode__Invalid = 0,
#include "xpu_opcodes.h"
	TypeOpCode__composite,
	TypeOpCode__array,
	TypeOpCode__internal,
	TypeOpCode__BuiltInMax,
} TypeOpCode;

#define FUNC_OPCODE(a,b,c,NAME,d,e)			FuncOpCode__##NAME,
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	FuncOpCode__##NAME,

#define WITH_DEVICE_ONLY_FUNCTIONS		1
typedef enum {
	FuncOpCode__Invalid = 0,
	FuncOpCode__ConstExpr,
	FuncOpCode__ParamExpr,
	FuncOpCode__VarExpr,
	FuncOpCode__BoolExpr_And,
	FuncOpCode__BoolExpr_Or,
	FuncOpCode__BoolExpr_Not,
	FuncOpCode__NullTestExpr_IsNull,
	FuncOpCode__NullTestExpr_IsNotNull,
	FuncOpCode__BoolTestExpr_IsTrue,
	FuncOpCode__BoolTestExpr_IsNotTrue,
	FuncOpCode__BoolTestExpr_IsFalse,
	FuncOpCode__BoolTestExpr_IsNotFalse,
	FuncOpCode__BoolTestExpr_IsUnknown,
	FuncOpCode__BoolTestExpr_IsNotUnknown,
	FuncOpCode__DistinctFrom,
	FuncOpCode__CoalesceExpr,
	FuncOpCode__LeastExpr,
	FuncOpCode__GreatestExpr,
	FuncOpCode__CaseWhenExpr,
	FuncOpCode__ScalarArrayOpAny,
	FuncOpCode__ScalarArrayOpAll,
	FuncOpCode__HashedArrayOp,
#include "xpu_opcodes.h"
	FuncOpCode__LoadVars = 9999,
	FuncOpCode__MoveVars,
	FuncOpCode__JoinQuals,
	FuncOpCode__HashValue,
	FuncOpCode__GiSTEval,
	FuncOpCode__SaveExpr,
	FuncOpCode__AggFuncs,
	FuncOpCode__Projection,
	FuncOpCode__SortKeys,
	FuncOpCode__Packed,		/* place-holder for the stacked expressions */
	FuncOpCode__BuiltInMax,
} FuncOpCode;
