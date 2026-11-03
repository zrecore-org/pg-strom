/*
 * exprlib.h - expression trees, linearizer and kernel generator shared by the
 * spike harnesses (vmspike.c: row format, heapspike.c: PostgreSQL heap pages).
 * Needs vkutil.h and the generated opcodes.inc to be included first.
 */
#ifndef EXPRLIB_H
#define EXPRLIB_H
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>

#define OFF_A		0		/* int4   */
#define OFF_B		4		/* int8   */
#define OFF_F		12		/* float4 */
#define OFF_D		16		/* int4   */
#define OFF_NB		20		/* null bitmap: bit0=a bit1=b bit2=f bit3=d */
#define MAX_REGS	16

/* ------------------------------------------------- expression trees */
typedef struct Node
{
	uint32_t	op;			/* FuncOpCode */
	uint32_t	type;		/* TypeOpCode (const / var) */
	int			nargs;
	struct Node *args[4];
	int64_t		imm;
	uint32_t	is_null;
	uint32_t	off, col;
} Node;

static Node pool[8192];
static int  npool;

static Node *mk(uint32_t op, int nargs, Node *a, Node *b, Node *c, Node *d)
{
	Node *n = &pool[npool++];

	n->op = op;
	n->nargs = nargs;
	n->args[0] = a; n->args[1] = b; n->args[2] = c; n->args[3] = d;
	return n;
}
static Node *K(uint32_t type, int64_t v)
{
	Node *n = mk(FuncOpCode__ConstExpr, 0, 0, 0, 0, 0);
	n->type = type;
	n->imm = v;
	return n;
}
static Node *KF(float f)
{
	uint32_t bits;
	memcpy(&bits, &f, 4);
	return K(TypeOpCode__float4, bits);
}
static Node *V(uint32_t type, uint32_t off, uint32_t col)
{
	Node *n = mk(FuncOpCode__VarExpr, 0, 0, 0, 0, 0);
	n->type = type; n->off = off; n->col = col;
	return n;
}
#define F1(op,a)		mk(FuncOpCode__##op, 1, a, 0, 0, 0)
#define F2(op,a,b)		mk(FuncOpCode__##op, 2, a, b, 0, 0)
#define AND2(a,b)		mk(FuncOpCode__BoolExpr_And, 2, a, b, 0, 0)
#define AND4(a,b,c,d)	mk(FuncOpCode__BoolExpr_And, 4, a, b, c, d)
#define OR2(a,b)		mk(FuncOpCode__BoolExpr_Or, 2, a, b, 0, 0)
#define K4(v)			K(TypeOpCode__int4, v)
#define K8(v)			K(TypeOpCode__int8, v)
#define VA()			V(TypeOpCode__int4, OFF_A, 0)
#define VB()			V(TypeOpCode__int8, OFF_B, 1)
#define VF()			V(TypeOpCode__float4, OFF_F, 2)
#define VD()			V(TypeOpCode__int4, OFF_D, 3)

/* ------------------------------------------------------ linearizer */
typedef struct { uint32_t op, dst, a, b, lo, hi; } Insn;
static Insn code[4096];
static int  ncode, maxreg;

static int ins(uint32_t op, uint32_t dst, uint32_t a, uint32_t b, uint32_t lo, uint32_t hi)
{
	if (ncode >= 4096 || dst + 2 > MAX_REGS)
	{
		fprintf(stderr, "program too large\n");
		exit(2);
	}
	code[ncode] = (Insn){op, dst, a, b, lo, hi};
	if ((int)dst + 1 > maxreg)
		maxreg = dst + 1;
	return ncode++;
}

/* tree -> flat program. AND/OR become accumulate-and-jump steps. */
static void emit(const Node *n, uint32_t dst)
{
	int steps[4], i;

	switch (n->op)
	{
		case FuncOpCode__ConstExpr:
			ins(n->op, dst, n->type, n->is_null,
				(uint32_t)n->imm, (uint32_t)((uint64_t)n->imm >> 32));
			break;
		case FuncOpCode__VarExpr:
			ins(n->op, dst, n->type, n->off, n->col, OFF_NB);
			break;
		case FuncOpCode__BoolExpr_And:
		case FuncOpCode__BoolExpr_Or:
			/* accumulator starts as TRUE for AND, FALSE for OR */
			ins(FuncOpCode__ConstExpr, dst, TypeOpCode__bool, 0,
				n->op == FuncOpCode__BoolExpr_And ? 1 : 0, 0);
			for (i = 0; i < n->nargs; i++)
			{
				emit(n->args[i], dst + 1);
				steps[i] = ins(n->op, dst, dst + 1, 0, 0, 0);
			}
			for (i = 0; i < n->nargs; i++)
				code[steps[i]].b = ncode;		/* jump target: end */
			break;
		default:
			emit(n->args[0], dst);
			if (n->nargs == 2)
				emit(n->args[1], dst + 1);
			ins(n->op, dst, dst, dst + 1, 0, 0);
			break;
	}
}

static void linearize(const Node *root, uint32_t *prog)
{
	ncode = 0;
	maxreg = 0;
	emit(root, 0);
	if ((2 + 6 * ncode) * 4 > 65536)
	{
		fprintf(stderr, "program does not fit the program buffer\n");
		exit(2);
	}
	prog[0] = ncode;
	prog[1] = 0;				/* result register */
	memcpy(prog + 2, code, sizeof(Insn) * ncode);
}


/* test expressions */
static Node *build1(void)
{
	return AND4(F2(int8gt, F2(int8pl, VB(), F1(int4_to_int8, VA())), K8(1LL << 30)),
				F2(float4gt, VF(), KF(500.0f)),
				F2(int4lt, VD(), K4(25)),
				F2(int4gt, VA(), K4(1000)));
}
static Node *build2(void)
{
	return AND2(OR2(F2(int4gt, VA(), K4(1000000000)), F1(NullTestExpr_IsNull, VB())),
				F1(BoolExpr_Not, OR2(F2(float4lt, VF(), KF(100.0f)),
									 F1(NullTestExpr_IsNull, VD()))));
}
static Node *build3(void) { return F2(int4gt, F2(int4div, VA(), VD()), K4(3)); }
static Node *build4(void) { return F2(int4gt, F2(int4pl, VA(), VA()), K4(0)); }
static Node *build5(void)
{
	return AND2(F1(BoolExpr_Not, F2(int4eq, VD(), K4(0))),
				F2(int4gt, F2(int4div, VA(), VD()), K4(3)));
}

/* scale test: balanced AND of n always-true terms (b + a > -(2^40) - 1 - i) */
static Node *build_scale(int lo, int hi)
{
	if (hi - lo == 1)
		return F2(int8gt, F2(int8pl, VB(), F1(int4_to_int8, VA())), K8(-(1LL << 40) - 1 - lo));
	int mid = (lo + hi) / 2;
	return AND2(build_scale(lo, mid), build_scale(mid, hi));
}

/* ------------------------------------------- specializer (M1c) */
/*
 * Walks the expression tree and emits one Slang kernel with typed locals and
 * no interpreter: AND/OR become do { ... break; } while (false) blocks, so a
 * short-circuit is a plain structured jump. NULL flags stay explicit (n<k>).
 */
#include <stdarg.h>
static char   gsrc[1 << 20];
static size_t glen;
static int    gnext;
static int    gen_heap;				/* 1: emit a heap-page kernel (ITupEval) instead of a row kernel */
static const char *gen_flags = "";		/* extra slangc flags, e.g. -DHEAP_WG=160 */

static void gp(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	glen += vsnprintf(gsrc + glen, sizeof gsrc - glen, fmt, ap);
	va_end(ap);
	if (glen >= sizeof gsrc - 1)
	{
		fprintf(stderr, "generated source too large\n");
		exit(2);
	}
}

static const char *ctype(uint32_t t)
{
	switch (t)
	{
		case TypeOpCode__bool:   return "bool";
		case TypeOpCode__int4:   return "int";
		case TypeOpCode__int8:   return "int64_t";
		case TypeOpCode__float4: return "float";
	}
	fprintf(stderr, "no device type %u in specializer\n", t);
	exit(2);
}

static uint32_t rtype(const Node *n)
{
	switch (n->op)
	{
		case FuncOpCode__ConstExpr:
		case FuncOpCode__VarExpr:
			return n->type;
		case FuncOpCode__int4pl:
		case FuncOpCode__int4mul:
		case FuncOpCode__int4div:
			return TypeOpCode__int4;
		case FuncOpCode__int8pl:
		case FuncOpCode__int4_to_int8:
			return TypeOpCode__int8;
		default:				/* And/Or/Not/IsNull/IsNotNull/comparisons */
			return TypeOpCode__bool;
	}
}

static const char *cmp_op(uint32_t op)
{
	switch (op)
	{
		case FuncOpCode__int4eq: case FuncOpCode__int8eq: return "==";
		case FuncOpCode__int4ne:                          return "!=";
		case FuncOpCode__int4lt: case FuncOpCode__int8lt:
		case FuncOpCode__float4lt:                        return "<";
		case FuncOpCode__int4gt: case FuncOpCode__int8gt:
		case FuncOpCode__float4gt:                        return ">";
	}
	return NULL;
}

/* emits statements computing v<id> (value) and n<id> (is-null); returns id */
static int gen(const Node *n)
{
	int id = ++gnext, ia, ib, i;
	const char *ty = ctype(rtype(n)), *cmp;
	uint32_t ty_id = rtype(n);

	switch (n->op)
	{
		case FuncOpCode__ConstExpr:
			if (ty_id == TypeOpCode__float4)
				gp("float v%d = asfloat(0x%08xu);\n", id, (uint32_t)n->imm);
			else if (ty_id == TypeOpCode__int8)
				gp("int64_t v%d = int64_t(%" PRId64 ");\n", id, n->imm);
			else if (ty_id == TypeOpCode__bool)
				gp("bool v%d = %s;\n", id, n->imm ? "true" : "false");
			else
				gp("int v%d = %d;\n", id, (int32_t)n->imm);
			gp("bool n%d = %s;\n", id, n->is_null ? "true" : "false");
			return id;

		case FuncOpCode__VarExpr:
		{
			const char *zero = ty_id == TypeOpCode__float4 ? "0.0f" :
							   ty_id == TypeOpCode__int8 ? "int64_t(0)" :
							   ty_id == TypeOpCode__bool ? "false" : "0";
			if (gen_heap)		/* the tuple is already deformed into t (a, b, f, d) */
			{
				gp("bool n%d = ((t.nulls >> %u) & 1u) != 0u;\n%s v%d = t.%c;\n",
				   id, n->col, ty, id, "abfd"[n->col]);
				return id;
			}
			gp("bool n%d = ((uint(rows[base + %u]) >> %u) & 1u) != 0u;\n", id, OFF_NB, n->col);
			gp("%s v%d = %s;\n", ty, id, zero);
			if (ty_id == TypeOpCode__float4)
				gp("if (!n%d) v%d = asfloat(ld32(base + %u));\n", id, id, n->off);
			else if (ty_id == TypeOpCode__int8)
				gp("if (!n%d) v%d = ld64(base + %u);\n", id, id, n->off);
			else if (ty_id == TypeOpCode__bool)
				gp("if (!n%d) v%d = rows[base + %u] != 0;\n", id, id, n->off);
			else
				gp("if (!n%d) v%d = int(ld32(base + %u));\n", id, id, n->off);
			return id;
		}

		case FuncOpCode__BoolExpr_And:
		case FuncOpCode__BoolExpr_Or:
		{
			int is_and = n->op == FuncOpCode__BoolExpr_And;

			gp("bool v%d = %s;\nbool n%d = false;\ndo {\n", id, is_and ? "true" : "false", id);
			for (i = 0; i < n->nargs; i++)
			{
				ia = gen(n->args[i]);
				gp("if (n%d) n%d = true;\n"
				   "else if (%sv%d) { v%d = %s; n%d = false; break; }\n",
				   ia, id, is_and ? "!" : "", ia, id, is_and ? "false" : "true", id);
			}
			gp("} while (false);\n");
			return id;
		}

		case FuncOpCode__BoolExpr_Not:
			ia = gen(n->args[0]);
			gp("bool v%d = !v%d;\nbool n%d = n%d;\n", id, ia, id, ia);
			return id;
		case FuncOpCode__NullTestExpr_IsNull:
			ia = gen(n->args[0]);
			gp("bool v%d = n%d;\nbool n%d = false;\n", id, ia, id);
			return id;
		case FuncOpCode__NullTestExpr_IsNotNull:
			ia = gen(n->args[0]);
			gp("bool v%d = !n%d;\nbool n%d = false;\n", id, ia, id);
			return id;
		case FuncOpCode__int4_to_int8:
			ia = gen(n->args[0]);
			gp("int64_t v%d = int64_t(v%d);\nbool n%d = n%d;\n", id, ia, id, ia);
			return id;

		case FuncOpCode__int4pl:
		case FuncOpCode__int4mul:
			ia = gen(n->args[0]);
			ib = gen(n->args[1]);
			gp("bool n%d = n%d || n%d;\nint v%d = 0;\n"
			   "if (!n%d) { int64_t r = int64_t(v%d) %c int64_t(v%d);\n"
			   "  if (r < int64_t(-2147483647) - 1 || r > int64_t(2147483647)) err = %u;\n"
			   "  else v%d = int(r); }\n",
			   id, ia, ib, id, id, ia, n->op == FuncOpCode__int4pl ? '+' : '*', ib,
			   n->op, id);
			return id;
		case FuncOpCode__int4div:
			ia = gen(n->args[0]);
			ib = gen(n->args[1]);
			gp("bool n%d = n%d || n%d;\nint v%d = 0;\n"
			   "if (!n%d) {\n"
			   "  if (v%d == 0) err = %u;\n"
			   "  else if (v%d == -1) { if (v%d == int(-2147483647) - 1) err = %u; else v%d = -v%d; }\n"
			   "  else v%d = v%d / v%d; }\n",
			   id, ia, ib, id, id, ib, n->op, ib, ia, n->op, id, ia, id, ia, ib);
			return id;
		case FuncOpCode__int8pl:
			ia = gen(n->args[0]);
			ib = gen(n->args[1]);
			gp("bool n%d = n%d || n%d;\nint64_t v%d = 0;\n"
			   "if (!n%d) { int64_t r = v%d + v%d;\n"
			   "  if ((((v%d ^ r) & (v%d ^ r)) < 0)) err = %u; else v%d = r; }\n",
			   id, ia, ib, id, id, ia, ib, ia, ib, n->op, id);
			return id;

		default:
			cmp = cmp_op(n->op);
			if (cmp)
			{
				ia = gen(n->args[0]);
				ib = gen(n->args[1]);
				gp("bool n%d = n%d || n%d;\nbool v%d = (v%d %s v%d);\n",
				   id, ia, ib, id, ia, cmp, ib);
				return id;
			}
	}
	fprintf(stderr, "specializer: opcode %u not supported\n", n->op);
	exit(2);
}

/* whole kernel source for one expression */
static void gen_kernel(const Node *root, const char *title, unsigned cold_salt)
{
	int r;

	glen = 0;
	gnext = 0;
	if (gen_heap)
	{
		gp("// generated by heapspike.c (M1c experiment, heap pages): %s\n#include \"heap_common.slang\"\n\n", title);
		gp("struct SpecEval : ITupEval\n{\nbool test(Tup t, out uint err)\n{\nerr = 0;\n");
		r = gen(root);
		if (cold_salt)	/* never true, but makes the text (and Metal's cache key) unique */
			gp("if (kparam.nrows == 0xFFFFFFFFu) outb[2] = %uu;\n", cold_salt);
		gp("if (err != 0) return false;\nreturn !n%d && v%d;\n}\n}\n\n", r, r);
		gp("[shader(\"compute\")]\n[numthreads(HEAP_WG, 1, 1)]\n"
		   "void spec_main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)\n"
		   "{\nSpecEval ev;\nheap_scan<SpecEval>(ev, gid, lid);\n}\n");
		return;
	}
	gp("// generated by vmspike.c (M1c experiment): %s\n#include \"common.slang\"\n\n", title);
	gp("[shader(\"compute\")]\n[numthreads(256, 1, 1)]\nvoid spec_main(uint3 tid : SV_DispatchThreadID)\n{\n");
	gp("uint gid = tid.x;\nbool hit = false;\nif (gid < kparam.nrows)\n{\nuint base = gid * kparam.stride;\nuint err = 0;\n");
	r = gen(root);
	gp("if (err != 0) report_error(gid, err);\nelse hit = !n%d && v%d;\n", r, r);
	if (cold_salt)		/* never true, but makes the text (and Metal's cache key) unique */
		gp("if (gid == 0xFFFFFFFFu) outb[2] = %uu;\n", cold_salt);
	gp("}\nemit_matches(gid, hit);\n}\n");
}

static uint64_t fnv1a(const char *s, size_t n)
{
	uint64_t h = 1469598103934665603ULL;

	while (n--)
		h = (h ^ (uint8_t)*s++) * 1099511628211ULL;
	return h;
}

/* source -> spv via slangc (cached by hash) -> pipeline. times in ms. */
static VkPipeline specialize(const char *title, double *t_slangc, double *t_pipe, int *cached)
{
	char src[256], spv[256], cmd[1024];
	const char *slangc = getenv("SLANGC");
	uint64_t h = fnv1a(gsrc, glen) ^ fnv1a(gen_flags, strlen(gen_flags));
	FILE *f;
	double t0;

	snprintf(src, sizeof src, "gen/spec_%016" PRIx64 ".slang", h);
	snprintf(spv, sizeof spv, "gen/spec_%016" PRIx64 ".spv", h);
	*cached = (access(spv, R_OK) == 0);
	*t_slangc = 0;
	if (!*cached)
	{
		f = fopen(src, "w");
		if (!f) { perror(src); exit(2); }
		fwrite(gsrc, 1, glen, f);
		fclose(f);
		snprintf(cmd, sizeof cmd,
				 "'%s' %s -I gen -I shaders %s -target spirv -entry spec_main -stage compute -o %s 2>gen/slangc.log",
				 slangc ? slangc : "slangc", src, gen_flags, spv);
		t0 = vk_now_ms();
		if (system(cmd) != 0)
		{
			fprintf(stderr, "slangc failed for %s, see gen/slangc.log\n", title);
			exit(2);
		}
		*t_slangc = vk_now_ms() - t0;
	}
	t0 = vk_now_ms();
	VkPipeline pl = vk_pipeline(spv);
	*t_pipe = vk_now_ms() - t0;
	return pl;
}


#endif /* EXPRLIB_H */
