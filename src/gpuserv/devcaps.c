/*
 * devcaps.c
 *
 * Probe kernels + tier computation. A capability goes into gpuserv_devcaps.caps only if
 * the API reported it (reported_caps) AND the probe kernel ran and verified exactly.
 */
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include "gpu_vk.h"
#include "gpuserv_probe.h"
#include "kernels_spv.h"

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static char last_error[GPU_VK_ERRLEN];

const char *
gpuserv_probe_last_error(void)
{
	return last_error;
}

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ------------------------------------------------------------------------ */
/* probe plumbing                                                            */
/* ------------------------------------------------------------------------ */
typedef struct
{
	uint32_t	subgroup_observed;		/* set by the subgroup probes */
} probe_state;

typedef int (*probe_fn)(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen);

#define WHY(...)	snprintf(why, whylen, __VA_ARGS__)

#define KSPV(name)	kspv_##name, kspv_##name##_len

/* compile + run one kernel; the pipeline is destroyed afterwards. 0 on success */
static int
run_kernel(gpu_vk_ctx *ctx, const unsigned char *spv, unsigned int len,
		   uint32_t nbind, gpu_vk_buffer *const *bufs,
		   const void *pc, uint32_t pcsz, uint32_t groups, char *why, size_t whylen)
{
	gpu_vk_pipeline pl;
	int			rc;

	if (gpu_vk_pipeline_create(ctx, &pl, spv, len, nbind, pcsz) != 0)
	{
		WHY("%s", ctx->err);
		return -1;
	}
	rc = gpu_vk_dispatch(ctx, &pl, bufs, pc, groups, 1, 1, NULL);
	if (rc != 0)
		WHY("%s", ctx->err);
	gpu_vk_pipeline_destroy(ctx, &pl);
	return rc;
}

#define NB 3
#define MKBUF(i, sz, bda)											\
	do {															\
		if (gpu_vk_buffer_create(ctx, &b[i], (sz), (bda)) != 0)		\
		{															\
			WHY("%s", ctx->err);									\
			rc = -1;												\
			goto out;												\
		}															\
		memset(b[i].map, 0, (sz));									\
	} while (0)

static void
free_bufs(gpu_vk_ctx *ctx, gpu_vk_buffer *b, int n)
{
	for (int i = 0; i < n; i++)
		gpu_vk_buffer_destroy(ctx, &b[i]);
}

/* ------------------------------------------------------------------------ */
/* int64 arithmetic                                                          */
/* ------------------------------------------------------------------------ */
static int
probe_int64(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 256;
	uint32_t	pc = n;
	int			rc = 0;
	const uint64_t *out;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, n * 8 * 8, 0);
	if ((rc = run_kernel(ctx, KSPV(i64_arith), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	for (uint32_t i = 0; i < n; i++)
	{
		uint64_t	u = i;
		int64_t		s = (int64_t) i - 128;
		uint64_t	big = 0xFFFFFFFFFFFFFFF0ULL - u * 1000003ULL;
		uint64_t	e[8];

		e[0] = (u << 33) | 5;
		e[1] = (u + 1) * 4294967311ULL;
		e[2] = (uint64_t) (s * 3000000000LL);
		e[3] = (uint64_t) ((-(int64_t) (i + 1) * 1000003LL) >> 3);
		e[4] = big / 7ULL;
		e[5] = big % 7ULL;
		e[6] = ((int64_t) ((uint64_t) s << 32) > ((int64_t) 10 << 32)) ? 1 : 0;
		e[7] = (big > 0xFFFFFFFF00000000ULL + (u << 20)) ? 1 : 0;
		for (int k = 0; k < 8; k++)
			if (out[i * 8 + k] != e[k])
			{
				WHY("thread %u value %d: got 0x%llx expected 0x%llx", i, k,
					(unsigned long long) out[i * 8 + k], (unsigned long long) e[k]);
				rc = -1;
				goto out;
			}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* 8-bit storage                                                             */
/* ------------------------------------------------------------------------ */
static int
probe_storage8(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	uint32_t	n = 256, pc = n;
	uint8_t	   *src;
	const uint32_t *out;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 2048, 0);
	MKBUF(1, n * 4, 0);
	src = b[0].map;
	for (int k = 0; k < 2048; k++)
		src[k] = (uint8_t) (k * 131 + 7);
	if ((rc = run_kernel(ctx, KSPV(storage8), 2, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[1].map;
	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t	e = (uint32_t) src[3 * i + 1] | ((uint32_t) src[3 * i + 2] << 8) |
						((uint32_t) src[3 * i + 5] << 16) | ((uint32_t) src[7 * i + 3] << 24);

		if (out[i] != e)
		{
			WHY("thread %u: got 0x%08x expected 0x%08x", i, out[i], e);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* subgroups                                                                 */
/* ------------------------------------------------------------------------ */
/*
 * Workgroups are 64 threads. Verify the geometry that the kernels report: all threads
 * see the same lane count S (1..64, dividing 64) and lane == i % S. Returns S or 0.
 */
static uint32_t
check_geometry(const uint32_t *lane, const uint32_t *count, uint32_t stride, uint32_t n,
			   char *why, size_t whylen)
{
	uint32_t	S = count[0];

	if (S == 0 || S > 64 || (64 % S) != 0)
	{
		WHY("unusable subgroup size %u", S);
		return 0;
	}
	for (uint32_t i = 0; i < n; i++)
	{
		if (count[i * stride] != S || lane[i * stride] != i % S)
		{
			WHY("thread %u: lane %u of %u does not match i %% %u", i, lane[i * stride],
				count[i * stride], S);
			return 0;
		}
	}
	return S;
}

static int
probe_ballot(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	uint32_t	n = 256, pc = n, S;
	const uint32_t *info, *comp;
	uint32_t	expc[256];
	int			rc = 0;

	memset(b, 0, sizeof(b));
	MKBUF(0, n * 10 * 4, 0);
	MKBUF(1, n * 4, 0);
	memset(b[1].map, 0xFF, n * 4);
	if ((rc = run_kernel(ctx, KSPV(subgroup_ballot), 2, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	info = b[0].map;
	comp = b[1].map;
	if (!(S = check_geometry(info + 8, info + 9, 10, n, why, whylen)))
	{
		rc = -1;
		goto out;
	}
	ps->subgroup_observed = S;
	memset(expc, 0xFF, sizeof(expc));
	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t	lane = i % S, base = i - lane;
		uint32_t	words[4] = {0, 0, 0, 0}, cnt = 0, pfx = 0;
		const uint32_t *r = info + i * 10;

		for (uint32_t j = 0; j < S; j++)
		{
			uint32_t	t = base + j;
			int			pred = (t % 3 == 0) || (t % 7 == 2);

			if (pred)
			{
				words[j / 32] |= 1u << (j % 32);
				if (j < lane)
					pfx++;
				cnt++;
			}
		}
		if (((i % 3 == 0) || (i % 7 == 2)))
			expc[base + pfx] = i;
		if (r[0] != words[0] || r[1] != words[1] || r[2] != words[2] || r[3] != words[3])
		{
			WHY("thread %u: ballot %08x%08x%08x%08x expected %08x%08x%08x%08x", i,
				r[3], r[2], r[1], r[0], words[3], words[2], words[1], words[0]);
			rc = -1;
			goto out;
		}
		if (r[4] != cnt || r[5] != pfx || r[6] != (lane == 0 ? 1u : 0u) || r[7] != base)
		{
			WHY("thread %u: count/prefix/first/readfirst %u/%u/%u/%u expected %u/%u/%u/%u",
				i, r[4], r[5], r[6], r[7], cnt, pfx, lane == 0 ? 1u : 0u, base);
			rc = -1;
			goto out;
		}
	}
	for (uint32_t i = 0; i < n; i++)
		if (comp[i] != expc[i])
		{
			WHY("compaction slot %u: got 0x%x expected 0x%x", i, comp[i], expc[i]);
			rc = -1;
			goto out;
		}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_arith32(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 256, pc = n, S;
	const uint32_t *out;
	int			rc = 0;

	memset(b, 0, sizeof(b));
	MKBUF(0, n * 5 * 4, 0);
	if ((rc = run_kernel(ctx, KSPV(subgroup_arith32), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	if (!(S = check_geometry(out + 3, out + 4, 5, n, why, whylen)))
	{
		rc = -1;
		goto out;
	}
	if (!ps->subgroup_observed)
		ps->subgroup_observed = S;
	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t	base = i - i % S, sum = 0, mn = 0xFFFFFFFFu, mx = 0;

		for (uint32_t j = 0; j < S; j++)
		{
			uint32_t	t = base + j;
			uint32_t	v = ((t * 2654435761u) >> 7) % 1000 + 1;

			sum += v;
			if (v < mn) mn = v;
			if (v > mx) mx = v;
		}
		if (out[i * 5] != sum || out[i * 5 + 1] != mn || out[i * 5 + 2] != mx)
		{
			WHY("thread %u: sum/min/max %u/%u/%u expected %u/%u/%u", i,
				out[i * 5], out[i * 5 + 1], out[i * 5 + 2], sum, mn, mx);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_arith64(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	uint32_t	n = 256, pc = n, S;
	const uint64_t *out;
	const uint32_t *lane;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, n * 8, 0);
	MKBUF(1, n * 2 * 4, 0);
	if ((rc = run_kernel(ctx, KSPV(subgroup_arith64), 2, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	lane = b[1].map;
	if (!(S = check_geometry(lane, lane + 1, 2, n, why, whylen)))
	{
		rc = -1;
		goto out;
	}
	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t	base = i - i % S;
		uint64_t	sum = 0;

		for (uint32_t j = 0; j < S; j++)
			sum += (uint64_t) (((int64_t) (base + j + 1) * 4294967297LL) - 1000000000000LL);
		if (out[i] != sum)
		{
			WHY("thread %u: sum 0x%llx expected 0x%llx", i,
				(unsigned long long) out[i], (unsigned long long) sum);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* 64-bit atomics                                                            */
/* ------------------------------------------------------------------------ */
static int
probe_i64_add(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 4096, pc = n;
	int64_t	   *out;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 8 * 8, 0);
	if ((rc = run_kernel(ctx, KSPV(atomic_i64_add), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	if ((uint64_t) out[0] != (uint64_t) n * 4294967297ULL)
	{
		WHY("out[0]=0x%llx expected 0x%llx", (unsigned long long) out[0],
			(unsigned long long) ((uint64_t) n * 4294967297ULL));
		rc = -1;
		goto out;
	}
	for (int k = 1; k <= 4; k++)
		if (out[k] != (int64_t) (n / 4))
		{
			WHY("out[%d]=%lld expected %u", k, (long long) out[k], n / 4);
			rc = -1;
			goto out;
		}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_i64_minmax(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 1024, pc = n;
	int64_t	   *out, mn = INT64_MAX, mx = INT64_MIN;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 2 * 8, 0);
	out = b[0].map;
	out[0] = INT64_MAX;
	out[1] = INT64_MIN;
	if ((rc = run_kernel(ctx, KSPV(atomic_i64_minmax), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	for (uint32_t i = 0; i < n; i++)
	{
		int64_t		v = ((int64_t) ((i * 37 + 11) % 4001) - 2000) * 4294967297LL;

		if (v < mn) mn = v;
		if (v > mx) mx = v;
	}
	if (out[0] != mn || out[1] != mx)
	{
		WHY("min/max %lld/%lld expected %lld/%lld", (long long) out[0], (long long) out[1],
			(long long) mn, (long long) mx);
		rc = -1;
		goto out;
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_i64_cas(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 1024, pc = n;
	int64_t	   *out, expect = 0;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 8, 0);
	out = b[0].map;
	if ((rc = run_kernel(ctx, KSPV(atomic_i64_cas), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	for (uint32_t i = 0; i < n; i++)
		expect += (((int64_t) (i + 1)) << 32) + 1;
	if (out[0] != expect)
	{
		WHY("counter 0x%llx expected 0x%llx", (unsigned long long) out[0],
			(unsigned long long) expect);
		rc = -1;
		goto out;
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* fp64, fp32 atomic add                                                     */
/* ------------------------------------------------------------------------ */
static uint64_t
dbits(double d)
{
	uint64_t	u;

	memcpy(&u, &d, sizeof(u));
	return u;
}

static int
probe_fp64(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	uint32_t	n = 64, pc = n;
	double	   *src;
	const double *out;
	int			rc = 0;
	volatile double va, vb, vc, vd, ve, vf, vg, vh, vk;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 16 * 8, 0);
	MKBUF(1, n * 8 * 8, 0);
	src = b[0].map;
	src[0] = 0.1; src[1] = 0.2; src[2] = 1.1; src[3] = 1.0;
	src[4] = 3.0; src[5] = 8.0; src[6] = 4.0; src[7] = 10.0; src[8] = 1e15;
	if ((rc = run_kernel(ctx, KSPV(fp64), 2, bl, &pc, 4, 1, why, whylen)) != 0)
		goto out;
	out = b[1].map;
	va = src[0]; vb = src[1]; vc = src[2]; vd = src[3];
	ve = src[4]; vf = src[5]; vg = src[6]; vh = src[7]; vk = src[8];
	for (uint32_t i = 0; i < n; i++)
	{
		double		acc = 0.0, e[8];

		for (int k = 0; k < 10; k++)
			acc += va;
		e[0] = va + vb;
		e[1] = vc * vc;
		e[2] = vd / ve;
		e[3] = vd / vf;
		e[4] = vh / vg;
		e[5] = acc;
		e[6] = (double) (int64_t) (9007199254740993LL + (int64_t) i);
		e[7] = (double) (int64_t) (vc * vk) - (double) i;
		for (int k = 0; k < 8; k++)
		{
			uint64_t	g = dbits(out[i * 8 + k]), x = dbits(e[k]);

			if (g == x)
				continue;
			/* 1.0/3.0 is the only inexact division: tolerate a 1 ulp difference */
			if (k == 2 && (g == x + 1 || g + 1 == x))
				continue;
			WHY("thread %u value %d: got %.17g (0x%llx) expected %.17g (0x%llx)", i, k,
				out[i * 8 + k], (unsigned long long) g, e[k], (unsigned long long) x);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_f32_add(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 256, pc = n;
	const float *out;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 16, 0);
	if ((rc = run_kernel(ctx, KSPV(atomic_f32_add), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	if (out[0] != (float) n)
	{
		WHY("sum %.1f expected %u.0", out[0], n);
		rc = -1;
		goto out;
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* df64: the error-free transformations (A10 part 6)                          */
/* ------------------------------------------------------------------------ */
#define DF64_C 1
#include "kernels_vm/df64.inc"

static uint32_t df_rng_state = 0x2545F491u;
static uint32_t
df_rnd(void)
{
	df_rng_state ^= df_rng_state << 13;
	df_rng_state ^= df_rng_state >> 17;
	df_rng_state ^= df_rng_state << 5;
	return df_rng_state;
}
static uint32_t df_fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float df_bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/*
 * Does this device (with MoltenVK's fast math switched off, gpu_vk.c) compute TwoSum, TwoProduct, the df64 add and multiply and the
 * binary64 -> df64 conversion bit for bit as the host's float32 does? The kernel runs 2,048 operand sets (exponents within +-30,
 * cancellations, wide gaps, ties) and every output must equal kernels_vm/df64.inc compiled as C. 0: yes, -1: no (why says which).
 */
int
gpuserv_df64_selfcheck(gpu_vk_ctx *ctx, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	const uint32_t n = 2048;
	uint32_t	pc = n;
	uint32_t   *in;
	const uint32_t *out;
	int			rc = 0;

	memset(b, 0, sizeof(b));
	MKBUF(0, (size_t) n * 6 * 4, 0);
	MKBUF(1, (size_t) n * 12 * 4, 0);
	in = b[0].map;
	for (uint32_t i = 0; i < n; i++)
	{
		float		a = df_bitsf((df_rnd() & 0x807FFFFFu) | ((uint32_t)(127 - 30 + df_rnd() % 61) << 23));
		float		bb = df_bitsf((df_rnd() & 0x807FFFFFu) | ((uint32_t)(127 - 30 + df_rnd() % 61) << 23));
		uint64_t	x = ((uint64_t)(df_rnd() & 0x800FFFFFu) << 32) | df_rnd();

		x |= (uint64_t)(1023 - 40 + df_rnd() % 81) << 52;
		switch (i % 4)
		{
			case 1: bb = -df_bitsf(df_fbits(a) + (df_rnd() % 7)); break;						/* cancellation */
			case 2: bb = df_bitsf((df_fbits(bb) & 0x807FFFFFu) | ((uint32_t)(127 - 40 + df_rnd() % 20) << 23)); break;
			case 3: a = df_bitsf(df_fbits(a) & 0xFFFF0000u); bb = df_bitsf(df_fbits(bb) & 0xFFFF0000u); break;	/* ties */
		}
		in[i * 6 + 0] = df_fbits(a);
		in[i * 6 + 1] = df_fbits(bb);
		in[i * 6 + 2] = df_fbits(a * df_bitsf(0x33800000u) * (float)((int)(df_rnd() % 2001) - 1000) / 1000.0f);	/* |lo| <= 2^-24 |a| */
		in[i * 6 + 3] = df_fbits(bb * df_bitsf(0x33800000u) * (float)((int)(df_rnd() % 2001) - 1000) / 1000.0f);
		in[i * 6 + 4] = (uint32_t) x;
		in[i * 6 + 5] = (uint32_t)(x >> 32);
	}
	if ((rc = run_kernel(ctx, KSPV(df64_exact), 2, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[1].map;
	for (uint32_t i = 0; i < n; i++)
	{
		float		a = df_bitsf(in[i * 6]), bb = df_bitsf(in[i * 6 + 1]);
		DF			A = df_mk(a, df_bitsf(in[i * 6 + 2])), B = df_mk(bb, df_bitsf(in[i * 6 + 3]));
		DF			ts = df_two_sum(a, bb), tp = df_two_prod(a, bb), ad = df_add(A, B), mu = df_mul(A, B), cv;
		uint64_t	xb = (uint64_t) in[i * 6 + 4] | ((uint64_t) in[i * 6 + 5] << 32);
		double		x;
		uint32_t	want[10];

		memcpy(&x, &xb, 8);
		cv = df_mk((float) x, (float)(x - (double)(float) x));
		want[0] = df_fbits(ts.hi); want[1] = df_fbits(ts.lo); want[2] = df_fbits(tp.hi); want[3] = df_fbits(tp.lo);
		want[4] = df_fbits(ad.hi); want[5] = df_fbits(ad.lo); want[6] = df_fbits(mu.hi); want[7] = df_fbits(mu.lo);
		want[8] = df_fbits(cv.hi); want[9] = df_fbits(cv.lo);
		for (int k = 0; k < 10; k++)
			if (out[i * 12 + k] != want[k])
			{
				WHY("case %u output %d: device %08x, host %08x (a=%a b=%a)", i, k, out[i * 12 + k], want[k], a, bb);
				rc = -1;
				goto out;
			}
		if (out[i * 12 + 10] != 1u)
		{
			WHY("case %u: the conversion refused an operand in range", i);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* buffer device address, memory model, unified memory, host import          */
/* ------------------------------------------------------------------------ */
static int
probe_bda(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	struct { uint64_t addr; uint32_t n; uint32_t pad; } pc;
	uint32_t	n = 1024;
	uint32_t   *p;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, n * 4, 1);
	if (b[0].addr == 0)
	{
		WHY("vkGetBufferDeviceAddress returned 0");
		rc = -1;
		goto out;
	}
	p = b[0].map;
	for (uint32_t i = 0; i < n; i++)
		p[i] = i;
	pc.addr = b[0].addr;
	pc.n = n;
	pc.pad = 0;
	if ((rc = run_kernel(ctx, KSPV(bda), 0, NULL, &pc, sizeof(pc), n / 64, why, whylen)) != 0)
		goto out;
	for (uint32_t i = 0; i < n; i++)
		if (p[i] != i * 3 + 1)
		{
			WHY("p[%u]=%u expected %u", i, p[i], i * 3 + 1);
			rc = -1;
			goto out;
		}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_memmodel(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 256, pc = n;
	const uint32_t *out;
	uint32_t	e[5] = {0, 0, 0, 0, 0};
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, 5 * 4, 0);
	if ((rc = run_kernel(ctx, KSPV(memmodel), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	e[0] = n;
	for (uint32_t t = 0; t < n; t++)
		e[1 + t % 4] += t;
	for (int k = 0; k < 5; k++)
		if (out[k] != e[k])
		{
			WHY("out[%d]=%u expected %u", k, out[k], e[k]);
			rc = -1;
			goto out;
		}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_unified(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[1] = { &b[0] };
	uint32_t	n = 256, pc = n;
	const uint32_t *out;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	MKBUF(0, n * 4, 0);
	if (!b[0].unified)
	{
		WHY("buffer was not placed in a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type");
		rc = -1;
		goto out;
	}
	if ((rc = run_kernel(ctx, KSPV(fill_u32), 1, bl, &pc, 4, n / 64, why, whylen)) != 0)
		goto out;
	out = b[0].map;
	for (uint32_t i = 0; i < n; i++)
		if (out[i] != i * 2654435761u + 7u)
		{
			WHY("out[%u]=0x%x expected 0x%x", i, out[i], i * 2654435761u + 7u);
			rc = -1;
			goto out;
		}
out:
	free_bufs(ctx, b, NB);
	return rc;
}

static int
probe_host_import(gpu_vk_ctx *ctx, probe_state *ps, char *why, size_t whylen)
{
	gpu_vk_buffer b[NB];
	gpu_vk_buffer *bl[2] = { &b[0], &b[1] };
	size_t		align = ctx->host_import_alignment;
	size_t		size = align * 4;
	size_t		maplen = size + align;
	uint8_t	   *base = MAP_FAILED, *ptr;
	uint32_t   *w, nwords = (uint32_t) (size / 4), pc = nwords;
	const uint32_t *out;
	uint32_t	x = 0x12345678u;
	int			rc = 0;

	(void) ps;
	memset(b, 0, sizeof(b));
	if (align == 0)
	{
		WHY("no host import alignment");
		return -1;
	}
	base = mmap(NULL, maplen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (base == MAP_FAILED)
	{
		WHY("mmap failed: %s", strerror(errno));
		return -1;
	}
	ptr = (uint8_t *) (((uintptr_t) base + align - 1) / align * align);
	w = (uint32_t *) ptr;
	for (uint32_t k = 0; k < nwords; k++)
	{
		x ^= x << 13; x ^= x >> 17; x ^= x << 5;
		w[k] = x;
	}
	if (gpu_vk_buffer_import_host(ctx, &b[0], ptr, size) != 0)
	{
		WHY("%s", ctx->err);
		rc = -1;
		goto out;
	}
	MKBUF(1, 64 * 4, 0);
	if ((rc = run_kernel(ctx, KSPV(host_import), 2, bl, &pc, 4, 1, why, whylen)) != 0)
		goto out;
	out = b[1].map;
	for (uint32_t t = 0; t < 64; t++)
	{
		uint32_t	sum = 0;

		for (uint32_t k = t; k < nwords; k += 64)
			sum += w[k] ^ (k * 31u);
		if (out[t] != sum)
		{
			WHY("thread %u: checksum 0x%x expected 0x%x", t, out[t], sum);
			rc = -1;
			goto out;
		}
	}
out:
	free_bufs(ctx, b, NB);		/* before munmap: the import still references it */
	if (base != MAP_FAILED)
		munmap(base, maplen);
	return rc;
}

/* ------------------------------------------------------------------------ */
/* the table                                                                 */
/* ------------------------------------------------------------------------ */
static uint64_t
compute_reported(const gpu_vk_ctx *ctx)
{
	const gpu_vk_feat *f = &ctx->feat;
	uint64_t	r = 0;
	int			sg_ballot = f->subgroup_compute &&
		(f->subgroup_ops & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
		(f->subgroup_ops & VK_SUBGROUP_FEATURE_BALLOT_BIT);
	int			sg_arith = f->subgroup_compute &&
		(f->subgroup_ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);

	if (f->int64)
		r |= GPUCAP_INT64;
	if (f->storage8 && f->int8)
		r |= GPUCAP_STORAGE8;
	if (sg_ballot)
		r |= GPUCAP_SUBGROUP_BALLOT;
	if (sg_arith)
		r |= GPUCAP_SUBGROUP_ARITH32;
	if (sg_arith && f->subgroup_extended_types && f->int64)
		r |= GPUCAP_SUBGROUP_ARITH64;
	if (f->buffer_int64_atomics && f->int64)
		r |= GPUCAP_ATOMIC_I64_ADD | GPUCAP_ATOMIC_I64_MINMAX | GPUCAP_ATOMIC_I64_CAS;
	if (f->float64)
		r |= GPUCAP_FP64;
	if (f->atomic_float32_add)
		r |= GPUCAP_ATOMIC_F32_ADD;
	if (f->bda && f->int64 && ctx->fn_bda)		/* the probe needs a 64-bit push constant */
		r |= GPUCAP_BDA;
	if (f->memory_model && f->memory_model_device_scope)
		r |= GPUCAP_MEMORY_MODEL;
	if (f->ext_host_import && ctx->host_import_alignment)
		r |= GPUCAP_HOST_IMPORT;
	if (ctx->unified_memtype >= 0)
		r |= GPUCAP_UNIFIED_MEMORY;
	if (f->float16)
		r |= GPUCAP_FP16;
	return r;
}

typedef struct
{
	const char *name;
	uint64_t	cap;
	probe_fn	fn;		/* NULL: no kernel, the reported bit is taken as is */
} probe_def;

static const probe_def probes[] = {
	{ "int64",             GPUCAP_INT64,             probe_int64 },
	{ "storage8",          GPUCAP_STORAGE8,          probe_storage8 },
	{ "subgroup_ballot",   GPUCAP_SUBGROUP_BALLOT,   probe_ballot },
	{ "subgroup_arith32",  GPUCAP_SUBGROUP_ARITH32,  probe_arith32 },
	{ "subgroup_arith64",  GPUCAP_SUBGROUP_ARITH64,  probe_arith64 },
	{ "atomic_i64_add",    GPUCAP_ATOMIC_I64_ADD,    probe_i64_add },
	{ "atomic_i64_minmax", GPUCAP_ATOMIC_I64_MINMAX, probe_i64_minmax },
	{ "atomic_i64_cas",    GPUCAP_ATOMIC_I64_CAS,    probe_i64_cas },
	{ "fp64",              GPUCAP_FP64,              probe_fp64 },
	{ "atomic_f32_add",    GPUCAP_ATOMIC_F32_ADD,    probe_f32_add },
	{ "bda",               GPUCAP_BDA,               probe_bda },
	{ "memory_model",      GPUCAP_MEMORY_MODEL,      probe_memmodel },
	{ "unified_memory",    GPUCAP_UNIFIED_MEMORY,    probe_unified },
	{ "host_import",       GPUCAP_HOST_IMPORT,       probe_host_import },
	{ "fp16",              GPUCAP_FP16,              NULL },
};

static void
copy_str(char *dst, size_t dstsz, const char *src)
{
	snprintf(dst, dstsz, "%s", src);
}

/* ------------------------------------------------------------------------ */
/* tier                                                                      */
/* ------------------------------------------------------------------------ */
static const char *
first_missing(uint64_t caps, uint64_t need)
{
	for (int i = 0; i < GPUCAP_NUM; i++)
		if ((need & (1ULL << i)) && !(caps & (1ULL << i)))
			return gpuserv_cap_name(i);
	return NULL;
}

void
gpuserv_devcaps_compute_tier(gpuserv_devcaps *dc, int allow_cpu)
{
	const uint64_t c = dc->caps;
	const char *m;

	dc->tier = GPUSERV_TIER_NONE;
	dc->tier_reason[0] = '\0';
	if (dc->device_type == 4 /* VK_PHYSICAL_DEVICE_TYPE_CPU */ && !allow_cpu)
	{
		copy_str(dc->tier_reason, sizeof(dc->tier_reason), "CPU device (use --allow-cpu)");
		return;
	}
	/*
	 * Note: the Vulkan memory model is NOT a base-tier requirement. MoltenVK on Apple
	 * Silicon reports vulkanMemoryModel = false (measured), and Apple is the validated
	 * platform, so requiring it would classify that device as unusable. Kernels on the base
	 * tier must therefore order memory with explicit barriers, not acquire/release
	 * atomics. The desktop tier requires it, below.
	 */
	m = first_missing(c, GPUCAP_INT64 | GPUCAP_STORAGE8 | GPUCAP_SUBGROUP_BALLOT |
					  GPUCAP_SUBGROUP_ARITH32);
	if (m)
	{
		snprintf(dc->tier_reason, sizeof(dc->tier_reason), "missing capability %s", m);
		return;
	}
	if (!(c & (GPUCAP_HOST_IMPORT | GPUCAP_UNIFIED_MEMORY)))
	{
		copy_str(dc->tier_reason, sizeof(dc->tier_reason),
				 "missing capability host_import or unified_memory");
		return;
	}
	if (dc->max_storage_buffer_range < (1ULL << 30) && !(c & GPUCAP_BDA))
	{
		snprintf(dc->tier_reason, sizeof(dc->tier_reason),
				 "max_storage_buffer_range %llu < 1 GiB and no bda",
				 (unsigned long long) dc->max_storage_buffer_range);
		return;
	}
	if (dc->max_shared_memory < 16384)
	{
		snprintf(dc->tier_reason, sizeof(dc->tier_reason),
				 "max_shared_memory %u < 16384", dc->max_shared_memory);
		return;
	}
	if (dc->max_wg_invocations < 128)
	{
		snprintf(dc->tier_reason, sizeof(dc->tier_reason),
				 "max_wg_invocations %u < 128", dc->max_wg_invocations);
		return;
	}
	dc->tier = GPUSERV_TIER_BASE;
	m = first_missing(c, GPUCAP_ATOMIC_I64_ADD | GPUCAP_ATOMIC_I64_MINMAX |
					  GPUCAP_ATOMIC_I64_CAS | GPUCAP_FP64 | GPUCAP_MEMORY_MODEL);
	if (m)
	{
		snprintf(dc->tier_reason, sizeof(dc->tier_reason),
				 "base tier; missing capability %s for desktop", m);
		return;
	}
	dc->tier = GPUSERV_TIER_DESKTOP;
}

/* ------------------------------------------------------------------------ */
/* driver                                                                    */
/* ------------------------------------------------------------------------ */
static void
fill_from_info(gpuserv_devcaps *dc, const gpu_vk_devinfo *di)
{
	copy_str(dc->name, sizeof(dc->name), di->name);
	copy_str(dc->driver_name, sizeof(dc->driver_name), di->driver_name);
	copy_str(dc->driver_info, sizeof(dc->driver_info), di->driver_info);
	copy_str(dc->uuid, sizeof(dc->uuid), di->uuid);
	dc->vendor_id = di->vendor_id;
	dc->device_id = di->device_id;
	dc->device_type = di->device_type;
	dc->api_version = di->api_version;
	dc->driver_version = di->driver_version;
	dc->max_storage_buffer_range = di->max_storage_buffer_range;
	dc->max_shared_memory = di->max_shared_memory;
	dc->max_wg_invocations = di->max_wg_invocations;
	dc->max_push_constants = di->max_push_constants;
}

static void
probe_one(gpuserv_devcaps *dc, const gpu_vk_devinfo *di, int allow_cpu, FILE *log)
{
	gpu_vk_ctx	ctx;
	probe_state	ps;
	uint64_t	reported;
	int			stop = 0;
	char		desc[1024];

	memset(dc, 0, sizeof(*dc));
	memset(&ps, 0, sizeof(ps));
	fill_from_info(dc, di);

	if (gpu_vk_open(&ctx, di->index) != 0)
	{
		if (log)
			fprintf(log, "[dev%d %s] open failed: %s\n", di->index, di->name, ctx.err);
		snprintf(dc->tier_reason, sizeof(dc->tier_reason), "device open failed: %.130s", ctx.err);
		dc->tier = GPUSERV_TIER_NONE;
		return;
	}
	dc->subgroup_size_min = ctx.feat.subgroup_size_min;
	dc->subgroup_size_max = ctx.feat.subgroup_size_max;
	dc->host_import_alignment = ctx.host_import_alignment;
	dc->max_buffer_size = ctx.max_buffer_size;
	dc->device_memory = ctx.device_memory;
	dc->max_storage_buffer_range = ctx.limits.maxStorageBufferRange;
	dc->max_shared_memory = ctx.limits.maxComputeSharedMemorySize;
	dc->max_wg_invocations = ctx.limits.maxComputeWorkGroupInvocations;
	dc->max_push_constants = ctx.limits.maxPushConstantsSize;
	reported = dc->reported_caps = compute_reported(&ctx);
	if (log)
	{
		gpu_vk_describe(&ctx, desc, sizeof(desc));
		fprintf(log, "[dev%d %s] api %u.%u.%u driver '%s' %s\n", di->index, di->name,
				VK_API_VERSION_MAJOR(di->api_version), VK_API_VERSION_MINOR(di->api_version),
				VK_API_VERSION_PATCH(di->api_version), di->driver_name, di->driver_info);
		fprintf(log, "[dev%d] %s\n", di->index, desc);
	}

	for (size_t k = 0; k < sizeof(probes) / sizeof(probes[0]); k++)
	{
		const probe_def *pd = &probes[k];
		char		why[GPU_VK_ERRLEN];
		double		t0;
		int			rc;

		why[0] = '\0';
		if (!(reported & pd->cap))
		{
			if (log)
				fprintf(log, "[dev%d] probe %-18s skip (not reported by the device)\n",
						di->index, pd->name);
			continue;
		}
		if (!pd->fn)
		{
			dc->caps |= pd->cap;
			if (log)
				fprintf(log, "[dev%d] probe %-18s set from the feature flag (no kernel)\n",
						di->index, pd->name);
			continue;
		}
		if (stop)
		{
			dc->probes_failed++;
			if (log)
				fprintf(log, "[dev%d] probe %-18s FAIL (not run: device lost)\n",
						di->index, pd->name);
			continue;
		}
		t0 = now_ms();
		rc = pd->fn(&ctx, &ps, why, sizeof(why));
		if (rc == 0)
		{
			dc->caps |= pd->cap;
			dc->probes_ok++;
			if (log)
				fprintf(log, "[dev%d] probe %-18s ok (%.1f ms)\n", di->index, pd->name,
						now_ms() - t0);
		}
		else
		{
			dc->probes_failed++;
			if (log)
				fprintf(log, "[dev%d] probe %-18s FAIL (%.1f ms): %s\n", di->index,
						pd->name, now_ms() - t0, why);
		}
		if (ctx.lost)
			stop = 1;
	}
	dc->subgroup_size_observed = ps.subgroup_observed;
	gpu_vk_close(&ctx);
	gpuserv_devcaps_compute_tier(dc, allow_cpu);
}

int
gpuserv_probe_devices(gpuserv_devcaps *out, int max, int allow_cpu, FILE *log)
{
	gpu_vk_devinfo infos[GPU_VK_MAX_DEVICES];
	char		err[GPU_VK_ERRLEN];
	int			n;

	last_error[0] = '\0';
	if (max > GPU_VK_MAX_DEVICES)
		max = GPU_VK_MAX_DEVICES;
	if (max <= 0)
		return 0;
	n = gpu_vk_enumerate(infos, max, err, sizeof(err));
	if (n < 0)
	{
		snprintf(last_error, sizeof(last_error), "%s", err);
		return -1;
	}
	for (int i = 0; i < n; i++)
		probe_one(&out[i], &infos[i], allow_cpu, log);
	return n;
}
