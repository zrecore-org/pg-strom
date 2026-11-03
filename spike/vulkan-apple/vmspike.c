/*
 * vmspike.c - harness for the Slang opcode-switch VM prototype (shaders/vm.slang)
 *
 * - FuncOpCode / TypeOpCode come from the real xpu_common.h + xpu_opcodes.h
 *   (extracted by gen_opcodes.sh), i.e. the numbers codegen.c would emit.
 * - Expression trees are linearized into the flat register program the shader
 *   executes (recursion lives on the host only).
 * - Every result is checked against an independently written CPU reference
 *   with SQL three-valued logic; two tests must report the right error code,
 *   one test must be saved from a zero-divide by AND short-circuiting.
 *
 * Throwaway code. See Makefile: make run-vm
 */
#include "vkutil.h"
#include "opcodes.inc"		/* TypeOpCode, FuncOpCode: generated, see gen_opcodes.sh */
#include "exprlib.h"

#define NROWS		(16u * 1024 * 1024)
#define STRIDE		21u		/* every field unaligned */

/* ---------------------------------------------------------------- rows */
typedef struct
{
	int32_t		a;
	int64_t		b;
	float		f;
	int32_t		d;
	int			na, nb, nf, nd;
} RowC;

static void decode(const uint8_t *r, RowC *o)
{
	uint8_t nb = r[OFF_NB];

	memcpy(&o->a, r + OFF_A, 4);
	memcpy(&o->b, r + OFF_B, 8);
	memcpy(&o->f, r + OFF_F, 4);
	memcpy(&o->d, r + OFF_D, 4);
	o->na = nb & 1;
	o->nb = (nb >> 1) & 1;
	o->nf = (nb >> 2) & 1;
	o->nd = (nb >> 3) & 1;
}

static uint64_t xs = 88172645463325252ULL;
static uint64_t rnd(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs; }

/* ---------------------------------------- CPU references (3VL) */
enum { T_F = 0, T_T = 1, T_N = 2 };
static int t_and(int x, int y) { return (x == T_F || y == T_F) ? T_F : (x == T_N || y == T_N) ? T_N : T_T; }
static int t_or(int x, int y)  { return (x == T_T || y == T_T) ? T_T : (x == T_N || y == T_N) ? T_N : T_F; }
static int t_not(int x)        { return x == T_N ? T_N : !x; }

/* E1: (b + (int8)a) > 2^30 AND f > 500.0 AND d < 25 AND a > 1000 */
static int ref1(const RowC *r)
{
	int t1 = (r->na || r->nb) ? T_N : ((r->b + (int64_t)r->a) > (1LL << 30));
	int t2 = r->nf ? T_N : (r->f > 500.0f);
	int t3 = r->nd ? T_N : (r->d < 25);
	int t4 = r->na ? T_N : (r->a > 1000);
	return t_and(t_and(t_and(t1, t2), t3), t4);
}
/* E2: (a > 1e9 OR b IS NULL) AND NOT (f < 100.0 OR d IS NULL) */
static int ref2(const RowC *r)
{
	int o1 = t_or(r->na ? T_N : (r->a > 1000000000), r->nb ? T_T : T_F);
	int o2 = t_or(r->nf ? T_N : (r->f < 100.0f), r->nd ? T_T : T_F);
	return t_and(o1, t_not(o2));
}
/* E5: NOT (d = 0) AND a / d > 3    (the zero-divide must be short-circuited) */
static int ref5(const RowC *r)
{
	int t1 = r->nd ? T_N : (r->d == 0 ? T_F : T_T);
	int t2;

	if (t1 == T_F)
		return T_F;
	t2 = (r->na || r->nd) ? T_N : ((r->a / r->d) > 3);
	return t_and(t1, t2);
}
/* E3: a / d > 3        errors on zero-divide */
static int err3(const RowC *r) { return !r->na && !r->nd && r->d == 0; }
/* E4: a + a > 0        errors on int4 overflow */
static int err4(const RowC *r) { return !r->na && ((int64_t)r->a + r->a) > INT32_MAX; }

static int ref_scale(const RowC *r) { return (r->na || r->nb) ? T_N : T_T; }

/* -------------------------------------------------------- tests */
typedef struct
{
	const char *name;
	Node	  *(*build)(void);
	int		  (*ref)(const RowC *);		/* match reference, or NULL */
	int		  (*errpred)(const RowC *);	/* rows that must raise an error */
	uint32_t	expect_err;
} Test;


static const Test tests[] = {
	{"E1 arithmetic+compare AND",        build1, ref1, NULL, 0},
	{"E2 NULL tests, OR, NOT (3VL)",     build2, ref2, NULL, 0},
	{"E5 short-circuit saves a/0",       build5, ref5, NULL, 0},
	{"E3 int4div by zero -> error",      build3, NULL, err3, FuncOpCode__int4div},
	{"E4 int4pl overflow -> error",      build4, NULL, err4, FuncOpCode__int4pl},
};

/* -------------------------------------------------------------- main */
static int cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return x < y ? -1 : x > y;
}

static const uint8_t *rowsmem;
static VkBuf rowsb, outb, progb;
struct { uint32_t nrows, stride; } pc = {NROWS, STRIDE};
static const uint32_t groups = (NROWS + 255) / 256;

/* run one dispatch several times; returns best ms, leaves last result in outb */
static double run(VkPipeline pl, int iters)
{
	uint32_t *ob = outb.map;
	double best = 1e9;

	for (int i = 0; i < iters; i++)
	{
		ob[0] = ob[1] = ob[2] = 0;
		double ms = vk_dispatch(pl, &pc, sizeof pc, groups);
		if (ms < best)
			best = ms;
	}
	return best;
}

/* GPU match list must equal exactly the rows the reference says are TRUE */
static int check_matches(int (*ref)(const RowC *), uint64_t *nref)
{
	uint32_t *ob = outb.map, *idx = ob + 3, k = 0;
	int ok = 1;

	qsort(idx, ob[0], 4, cmp_u32);
	*nref = 0;
	for (uint32_t i = 0; i < NROWS; i++)
	{
		RowC r;
		decode(rowsmem + (size_t)i * STRIDE, &r);
		if (ref(&r) == T_T)
		{
			(*nref)++;
			if (k >= ob[0] || idx[k] != i)
				ok = 0;
			k++;
		}
	}
	return ok && k == ob[0];
}

/* check the result sitting in outb against the reference / error expectation */
static int verify(const Test *tc, const char *tag, const char *extra, double ms)
{
	uint32_t *ob = outb.map;
	int ok;

	if (tc->ref)
	{
		uint64_t nref;
		ok = (ob[1] == 0) && check_matches(tc->ref, &nref);
		printf("[%s] %s %-32s %smatches gpu=%u cpu=%llu err=%u  best=%.1f ms\n",
			   ok ? "PASS" : "FAIL", tag, tc->name, extra, ob[0],
			   (unsigned long long)nref, ob[1], ms);
	}
	else
	{
		RowC r;
		uint64_t nerr = 0;

		for (uint32_t i = 0; i < NROWS; i++)
		{
			decode(rowsmem + (size_t)i * STRIDE, &r);
			nerr += tc->errpred(&r);
		}
		ok = 0;
		if (nerr > 0 && ob[1] == tc->expect_err && ob[2] < NROWS)
		{
			decode(rowsmem + (size_t)ob[2] * STRIDE, &r);
			ok = tc->errpred(&r);
		}
		printf("[%s] %s %-32s %serr=%u (want %u) at row %u, %llu rows would error  best=%.1f ms\n",
			   ok ? "PASS" : "FAIL", tag, tc->name, extra, ob[1], tc->expect_err,
			   ob[2], (unsigned long long)nerr, ms);
	}
	return ok;
}

int main(int argc, char **argv)
{
	const char *dir = argc > 1 ? argv[1] : "shaders";
	char path[512];
	int failures = 0;

	vk_init();
	rowsb = vk_buf((size_t)NROWS * STRIDE + 16);
	outb = vk_buf(sizeof(uint32_t) * (NROWS + 8));
	progb = vk_buf(65536);
	VkBuf bufs[3] = {rowsb, outb, progb};
	vk_bind(bufs, 3, sizeof pc, getenv("VM_UBO") ? 4u : 0u);
	if (getenv("VM_UBO"))
		printf("program bound as UNIFORM buffer\n");

	/* ---- data ---- */
	uint8_t *p = rowsb.map;
	rowsmem = p;
	for (uint32_t i = 0; i < NROWS; i++)
	{
		int32_t a = (int32_t)(rnd() & 0x7fffffff);
		int64_t b = (int64_t)(rnd() % (1ULL << 41)) - (1LL << 40);
		float f = (float)(rnd() % 1000000) / 1000.0f;
		int32_t d = (int32_t)(rnd() % 101) - 50;
		uint8_t nb = 0, *r = p + (size_t)i * STRIDE;

		nb |= (rnd() % 100 < 5) ? 1 : 0;
		nb |= (rnd() % 100 < 5) ? 2 : 0;
		nb |= (rnd() % 100 < 5) ? 4 : 0;
		nb |= (rnd() % 100 < 3) ? 8 : 0;
		memcpy(r + OFF_A, &a, 4);
		memcpy(r + OFF_B, &b, 8);
		memcpy(r + OFF_F, &f, 4);
		memcpy(r + OFF_D, &d, 4);
		r[OFF_NB] = nb;
	}
	printf("rows=%u stride=%u (%.0f MB)\n", NROWS, STRIDE, (double)NROWS * STRIDE / 1e6);
	printf("opcodes from real headers: FuncOpCode__int4pl=%d int4div=%d int8gt=%d "
		   "BoolExpr_And=%d, TypeOpCode__int8=%d\n",
		   FuncOpCode__int4pl, FuncOpCode__int4div, FuncOpCode__int8gt,
		   FuncOpCode__BoolExpr_And, TypeOpCode__int8);

	snprintf(path, sizeof path, "%s/vm_main.spv", dir);
	double tp0 = vk_now_ms();
	VkPipeline vm = vk_pipeline(path);
	double tp1 = vk_now_ms();
	snprintf(path, sizeof path, "%s/fixed_main.spv", dir);
	VkPipeline fixed = vk_pipeline(path);
	printf("pipeline creation (SPIR-V -> MSL -> Metal): vm_main %.1f ms, fixed_main %.1f ms\n",
		   tp1 - tp0, vk_now_ms() - tp1);
	if (!vm || !fixed)
	{
		printf("[FAIL] Metal compiler rejected the VM or fixed kernel\n");
		return 1;
	}

	/* ---- VM tests ---- */
	double vm_ms[16] = {0};
	uint32_t *ob = outb.map;
	for (size_t t = 0; t < sizeof(tests) / sizeof(tests[0]); t++)
	{
		const Test *tc = &tests[t];
		char extra[64];

		npool = 0;
		linearize(tc->build(), progb.map);
		snprintf(extra, sizeof extra, "insns=%2d regs=%d  ", ncode, maxreg);
		vm_ms[t] = run(vm, 5);
		failures += !verify(tc, "VM  ", extra, vm_ms[t]);
	}
	double vm_e1_ms = vm_ms[0], hand_ms = 0;

	/* ---- cost of the VM vs a hand-written kernel, same predicate as E1 ---- */
	{
		uint64_t nref;
		double fx = run(fixed, 5);
		hand_ms = fx;
		int ok = check_matches(ref1, &nref);
		RowC r;
		volatile uint64_t cnt = 0;
		double c0 = vk_now_ms();

		for (uint32_t i = 0; i < NROWS; i++)
		{
			decode(rowsmem + (size_t)i * STRIDE, &r);
			cnt += (ref1(&r) == T_T);
		}
		printf("[%s] hand-written kernel for E1: matches=%u best=%.1f ms\n",
			   ok ? "PASS" : "FAIL", ob[0], fx);
		printf("      VM E1 %.1f ms vs fixed %.1f ms => VM overhead %.2fx;  cpu 1-thread %.1f ms\n",
			   vm_e1_ms, fx, vm_e1_ms / fx, vk_now_ms() - c0);
		failures += !ok;
	}

	/* ---- M1c: specialized kernels generated from the same trees ---- */
	{
		unsigned salt = getenv("SPEC_COLD") ? (unsigned)time(NULL) : 0;
		double gen_total = 0;

		mkdir("gen", 0755);
		printf("-- specialized kernels (slangc=%s%s)\n",
			   getenv("SLANGC") ? getenv("SLANGC") : "slangc", salt ? ", forced cold" : "");
		for (size_t t = 0; t < sizeof(tests) / sizeof(tests[0]); t++)
		{
			const Test *tc = &tests[t];
			double t_gen, t_slangc, t_pipe, ms;
			int cached;
			char extra[96];

			npool = 0;
			t_gen = vk_now_ms();
			gen_kernel(tc->build(), tc->name, salt ? salt + t : 0);
			t_gen = vk_now_ms() - t_gen;
			VkPipeline sp = specialize(tc->name, &t_slangc, &t_pipe, &cached);
			if (!sp)
			{
				printf("[FAIL] SPEC %s: Metal compiler rejected the generated kernel\n", tc->name);
				failures++;
				continue;
			}
			ms = run(sp, 5);
			snprintf(extra, sizeof extra, "gen=%.2fms slangc=%.0fms%s pipe=%.1fms  ",
					 t_gen, t_slangc, cached ? "(cached)" : "", t_pipe);
			failures += !verify(tc, "SPEC", extra, ms);
			if (tc->ref)
				printf("      %s: VM %.1f ms -> specialized %.1f ms (%.1fx faster)\n",
					   tc->name, vm_ms[t], ms, vm_ms[t] / ms);
			if (t == 0)
				printf("      E1 specialized vs hand-written kernel: %.1f ms vs %.1f ms (%.2fx)\n",
					   ms, hand_ms, ms / hand_ms);
			gen_total += t_gen;
		}
	}


	/* ---- M1c: how does specialization cost scale with expression size? ---- */
	{
		static const int sizes[] = {4, 16, 64, 256};
		unsigned salt = (unsigned)time(NULL);

		printf("-- compile-cost scaling: AND of N terms, each (b + (int8)a) > const, always true when non-NULL\n");
		for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++)
		{
			double t_gen, t_slangc, t_pipe, ms;
			int cached;
			char extra[96], nm[48];
			Test tc = {nm, NULL, ref_scale, NULL, 0};

			snprintf(nm, sizeof nm, "N=%d", sizes[k]);
			npool = 0;
			Node *root = build_scale(0, sizes[k]);
			linearize(root, progb.map);
			snprintf(extra, sizeof extra, "insns=%d  ", ncode);
			failures += !verify(&tc, "VM   ", extra, run(vm, 2));
			t_gen = vk_now_ms();
			gen_kernel(root, nm, getenv("SPEC_COLD") ? salt + 100 + k : 0);
			t_gen = vk_now_ms() - t_gen;
			VkPipeline sp = specialize(nm, &t_slangc, &t_pipe, &cached);
			if (!sp)
			{
				printf("[FAIL] SCALE %s: Metal compiler rejected the kernel\\n", nm);
				failures++;
				continue;
			}
			ms = run(sp, 5);
			snprintf(extra, sizeof extra, "%d nodes, %zu KB src, gen=%.1fms slangc=%.0fms%s pipe=%.0fms  ",
					 gnext, glen / 1024, t_gen, t_slangc, cached ? "(cached)" : "", t_pipe);
			failures += !verify(&tc, "SCALE", extra, ms);
		}
	}

	printf("%s\n", failures ? "VM SPIKE: FAILURES" : "VM SPIKE: ALL PASSED");
	return failures ? 1 : 0;
}
