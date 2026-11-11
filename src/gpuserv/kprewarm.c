/*
 * kprewarm.c
 *
 * First-use kernel compile (A9 part 3): kernel registry, persisted readiness manifest, background prewarm thread,
 * readiness gate. See kprewarm.h.
 *
 * Why a manifest: the driver's pipeline cache cannot be asked whether it holds a kernel, and creating the pipeline of
 * a cold kernel takes minutes. So the service records every kernel it has seen compiled (name, hash of its SPIR-V) in
 * <dir>/kernels-<same suffix as the pipeline cache file>.txt, together with the size of the pipeline cache file at that
 * time. The record is trusted only if the pipeline cache file loaded at start is at least that big (a deleted,
 * damaged, foreign or reset cache file loads as 0 bytes, and every kernel is then compiled again), and only for a kernel
 * whose SPIR-V hash is unchanged (a rebuilt service with other kernels compiles those again).
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <pthread/qos.h>
#else
#include <sys/syscall.h>
#endif
#include "kprewarm.h"
#include "kernels_spv.h"

typedef struct kent
{
	const char *name;
	const unsigned char *spv;
	size_t		len;
	uint32_t	nbind, pcsz;
	int			tier;
	int			common;				/* in the common set (the kernels a typical program uses) */
	int			rank;				/* order of the prewarm: tier, then common first */
	int			demand;				/* a task asked for it */
	int			hashed;
	uint64_t	hash;
	int			has_mrec;			/* the manifest has a record of this name */
	uint64_t	mhash;
	float		msecs;
	int			persist;			/* ready and (as far as we know) in the pipeline cache file */
	int			refused;			/* a task was refused for it: later ones are refused at once until it is ready */
} kent;

static const struct { const char *name; const unsigned char *spv; const unsigned int *len; } kreg[] = {
#include "kernels_reg.inc"
};
#define KREG_N		((int) (sizeof(kreg) / sizeof(kreg[0])))

/* the base names (without the tier suffix) a typical program uses: the smallest register file of each family */
static const char *const common_bases[] = {
	"gpuscan8", "gpuagg8_4", "gagg16_4", "gtab16", "gpujoin8", "jagg16_4", "jgagg16_4",
	"gpuarrow8", "aagg16_4", "agagg16_4", "agtab16", "ajoin16", "ajagg16_4", "ajgagg16_4",
	"jagg32_4", "jgagg32_4", "ajagg32_4", "ajgagg32_4", NULL
};
/* and the geometry tiers of the join families (the register-8 join kernel has none) */
static const char *const common_geom_bases[] = {
	"gpujoin16", "jagg16_4", "jgagg16_4", "ajoin16", "ajagg16_4", "ajgagg16_4",
	"jagg32_4", "jgagg32_4", "ajagg32_4", "ajgagg32_4", NULL
};

static struct
{
	gpu_vk_ctx *vk;
	gpuserv_prewarm_state *st;
	kent		ents[GPUSERV_KERNEL_MAX];
	int			n;
	int			mode;
	char		manifest_path[800];
	size_t		pcache_loaded;
	pthread_t	thr;
	int			thr_started;
	pthread_mutex_t mu;
	pthread_cond_t cv;				/* wakes the thread: a demand, stop */
	pthread_cond_t done_cv;			/* an entry became ready or failed */
	int			stop;
	int			compiling;			/* the thread is inside a pipeline creation */
	int			busy;				/* tasks running */
	double		last_active;		/* monotonic seconds of the last task end */
	uint32_t	sess_wait;			/* the session's field */
	int			force_wait;			/* PGSTROM_GPUSERV_KERNEL_WAIT_MS=-1 */
	long		idle_ms, delay_ms;
} K = { .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER, .done_cv = PTHREAD_COND_INITIALIZER };

static double
mono_now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static double
real_now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_REALTIME, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static uint64_t
spv_hash(const void *data, size_t size)
{
	const unsigned char *p = data;
	uint64_t	h = 1469598103934665603ULL;
	size_t		i = 0;

	for (; i + 8 <= size; i += 8)
	{
		uint64_t	w;

		memcpy(&w, p + i, 8);
		h = (h ^ w) * 1099511628211ULL;
		h ^= h >> 29;
	}
	for (; i < size; i++)
		h = (h ^ p[i]) * 1099511628211ULL;
	return h ^ size;
}

int
kpw_parse_mode(const char *s)
{
	if (!s)
		return -1;
	if (!strcmp(s, "off") || !strcmp(s, "0") || !strcmp(s, "false"))
		return GPUSERV_PREWARM_OFF;
	if (!strcmp(s, "lite") || !strcmp(s, "on") || !strcmp(s, "auto"))
		return GPUSERV_PREWARM_LITE;
	if (!strcmp(s, "geometry"))
		return GPUSERV_PREWARM_GEOMETRY;
	if (!strcmp(s, "all"))
		return GPUSERV_PREWARM_ALL;
	return -1;
}

static int
in_list(const char *const *list, const char *s)
{
	for (; *list; list++)
		if (!strcmp(*list, s))
			return 1;
	return 0;
}

/* tier and base name of a kernel name */
static int
split_name(const char *name, char *base, size_t blen)
{
	static const struct { const char *suffix; int tier; } sfx[] = {
		{"_lite", GPUSERV_KTIER_LITE}, {"_nogeom", GPUSERV_KTIER_NOGEOM},
		{"_gpip", GPUSERV_KTIER_GEOM}, {"_gdw", GPUSERV_KTIER_GEOM}, {"_gwalk", GPUSERV_KTIER_GEOM},
	};
	size_t		n = strlen(name);

	for (size_t i = 0; i < sizeof(sfx) / sizeof(sfx[0]); i++)
	{
		size_t	sl = strlen(sfx[i].suffix);

		if (n > sl && !strcmp(name + n - sl, sfx[i].suffix))
		{
			snprintf(base, blen, "%.*s", (int)(n - sl), name);
			return sfx[i].tier;
		}
	}
	snprintf(base, blen, "%s", name);
	return GPUSERV_KTIER_FULL;
}

/* the layout of the pipeline, as the callers in scan_exec.c / arrow_exec.c create it */
static void
layout_of(const char *base, uint32_t *nbind, uint32_t *pcsz)
{
	if (!strncmp(base, "gpuscan", 7) || !strncmp(base, "gpuagg", 6) || !strncmp(base, "gagg", 4) ||
		!strncmp(base, "gpuarrow", 8))
		*nbind = 5, *pcsz = 16;
	else if (!strncmp(base, "aagg", 4) || !strncmp(base, "agagg", 5))
		*nbind = 5, *pcsz = 20;
	else if (!strncmp(base, "gtab", 4))
		*nbind = 6, *pcsz = 16;
	else
		*nbind = 6, *pcsz = 20;			/* agtab, the join families and their Arrow twins */
}

static int
rank_cmp(const void *a, const void *b)
{
	const kent *x = *(const kent *const *) a, *y = *(const kent *const *) b;

	if (x->rank != y->rank)
		return x->rank - y->rank;
	return strcmp(x->name, y->name);
}

static int
in_set(const kent *e, int mode)
{
	switch (mode)
	{
		case GPUSERV_PREWARM_LITE:		return e->common && e->tier <= GPUSERV_KTIER_NOGEOM;
		case GPUSERV_PREWARM_GEOMETRY:	return e->common && e->tier <= GPUSERV_KTIER_GEOM;
		case GPUSERV_PREWARM_ALL:		return 1;
		default:						return 0;
	}
}

/* the manifest next to the pipeline cache file: ".../pipeline-<x>.bin" -> ".../kernels-<x>.txt" */
static void
manifest_path_of(const char *pcpath, char *out, size_t outlen)
{
	const char *sl = strrchr(pcpath, '/');
	const char *base = sl ? sl + 1 : pcpath;

	out[0] = '\0';
	if (strncmp(base, "pipeline-", 9) != 0)
		return;
	snprintf(out, outlen, "%.*skernels-%s", (int)(base - pcpath), pcpath, base + 9);
	{
		size_t	n = strlen(out);

		if (n > 4 && !strcmp(out + n - 4, ".bin"))
			strcpy(out + n - 4, ".txt");
	}
}

static void
manifest_load(void)
{
	FILE	   *fp;
	char		line[256];
	unsigned long long msize = 0;
	int			ok = 0;

	if (!K.manifest_path[0] || !(fp = fopen(K.manifest_path, "r")))
		return;
	if (fgets(line, sizeof(line), fp) && !strncmp(line, "pgstrom-kernel-state 1", 22) &&
		fgets(line, sizeof(line), fp) && sscanf(line, "pcache %llu", &msize) == 1)
		ok = (K.pcache_loaded > 0 && K.pcache_loaded >= msize);
	while (ok && fgets(line, sizeof(line), fp))
	{
		char		name[64];
		unsigned long long h;
		double		secs;

		if (sscanf(line, "%63s %llx %lf", name, &h, &secs) != 3)
			continue;
		for (int i = 0; i < K.n; i++)
			if (!strcmp(K.ents[i].name, name))
			{
				K.ents[i].has_mrec = 1;
				K.ents[i].mhash = h;
				K.ents[i].msecs = (float) secs;
			}
	}
	fclose(fp);
}

/* caller holds K.mu */
static void
manifest_save_locked(void)
{
	char		tmp[sizeof(K.manifest_path) + 16];
	FILE	   *fp;

	if (!K.manifest_path[0])
		return;
	snprintf(tmp, sizeof(tmp), "%s.%d.tmp", K.manifest_path, (int) getpid());
	fp = fopen(tmp, "w");
	if (!fp)
		return;
	fprintf(fp, "pgstrom-kernel-state 1\npcache %llu\n", (unsigned long long) gpu_vk_pcache_saved(K.vk));
	for (int i = 0; i < K.n; i++)
	{
		if (K.ents[i].persist && K.ents[i].hashed)
			fprintf(fp, "%s %llx %.3f\n", K.ents[i].name, (unsigned long long) K.ents[i].hash,
					(double) K.st->k[i].compile_secs);
		else if (K.ents[i].has_mrec && !K.ents[i].hashed)		/* not looked at (outside the set): keep the record */
			fprintf(fp, "%s %llx %.3f\n", K.ents[i].name, (unsigned long long) K.ents[i].mhash, (double) K.ents[i].msecs);
	}
	if (fclose(fp) != 0 || rename(tmp, K.manifest_path) != 0)
		unlink(tmp);
}

/* caller holds K.mu: recount */
static void
recount_locked(void)
{
	uint32_t	target = 0, ready = 0, failed = 0;

	for (int i = 0; i < K.n; i++)
	{
		if (K.st->k[i].in_set)
		{
			target++;
			if (K.st->k[i].state == GPUSERV_KSTATE_READY)
				ready++;
		}
		if (K.st->k[i].state == GPUSERV_KSTATE_FAILED)
			failed++;
	}
	K.st->target = target;
	K.st->target_ready = ready;
	K.st->nfailed = failed;
}

static void
set_state_locked(kent *e, int state)
{
	int			i = (int)(e - K.ents);

	__atomic_store_n(&K.st->k[i].state, (uint8_t) state, __ATOMIC_SEQ_CST);
	recount_locked();
	pthread_cond_broadcast(&K.done_cv);
}

/* caller holds K.mu, e->hash is set: take the manifest's word for a kernel with the same hash */
static void
apply_manifest_locked(kent *e)
{
	int			i = (int)(e - K.ents);

	e->hashed = 1;
	if (e->has_mrec && e->mhash == e->hash && K.st->k[i].state == GPUSERV_KSTATE_PENDING)
	{
		K.st->k[i].from_disk = 1;
		K.st->k[i].compile_secs = e->msecs;
		e->persist = 1;
		K.st->persisted++;
		set_state_locked(e, GPUSERV_KSTATE_READY);
	}
}

/* caller holds K.mu: hash the SPIR-V (only a kernel with a manifest record is worth it) */
static void
verify_locked(kent *e)
{
	if (e->hashed || !e->has_mrec)
		return;
	e->hash = spv_hash(e->spv, e->len);
	apply_manifest_locked(e);
}

static int
kreg_init(gpu_vk_ctx *vk, gpuserv_prewarm_state *st, int mode)
{
	int			n = 0;

	memset(&K.ents, 0, sizeof(K.ents));
	K.vk = vk;
	K.st = st;
	K.mode = mode;
	for (int i = 0; i < KREG_N && n < GPUSERV_KERNEL_MAX; i++)
	{
		char		base[64];
		kent	   *e = &K.ents[n];

		e->name = kreg[i].name;
		e->spv = kreg[i].spv;
		e->len = *kreg[i].len;
		e->tier = split_name(e->name, base, sizeof(base));
		layout_of(base, &e->nbind, &e->pcsz);
		e->common = (e->tier == GPUSERV_KTIER_GEOM ? in_list(common_geom_bases, base) : in_list(common_bases, base));
		e->rank = e->tier * 2 + (e->common ? 0 : 1);
		n++;
	}
	K.n = n;
	memset(st, 0, sizeof(*st));
	st->nkernels = (uint32_t) n;
	st->mode = mode;
	for (int i = 0; i < n; i++)
	{
		snprintf(st->k[i].name, sizeof(st->k[i].name), "%s", K.ents[i].name);
		st->k[i].tier = (uint8_t) K.ents[i].tier;
		st->k[i].in_set = (uint8_t) in_set(&K.ents[i], mode);
		st->k[i].spirv_kb = (uint32_t)(K.ents[i].len / 1024);
	}
	K.pcache_loaded = vk->pcache_loaded;
	manifest_path_of(gpu_vk_pcache_path(vk), K.manifest_path, sizeof(K.manifest_path));
	manifest_load();
	recount_locked();
	return n;
}

/* the entry whose SPIR-V array is 'spv' (the callers pass the embedded arrays) */
static kent *
find_ent(const void *spv)
{
	for (int i = 0; i < K.n; i++)
		if ((const void *) K.ents[i].spv == spv)
			return &K.ents[i];
	return NULL;
}

/* compile one kernel in the calling thread: create the pipeline (the driver compiles), end it with an empty dispatch */
static int
compile_ent(kent *e, double *secs_out)
{
	gpu_vk_pipeline pl;
	char		err[256];
	double		t0 = mono_now();
	size_t		before = gpu_vk_pcache_saved(K.vk);
	int			rc;
	int			i = (int)(e - K.ents);

	memset(&pl, 0, sizeof(pl));
	if (!e->hashed)
	{
		uint64_t	h = spv_hash(e->spv, e->len);

		pthread_mutex_lock(&K.mu);
		e->hash = h;
		e->hashed = 1;
		pthread_mutex_unlock(&K.mu);
	}
	pthread_mutex_lock(&K.mu);
	if (K.stop || K.st->k[i].state != GPUSERV_KSTATE_PENDING)
	{
		pthread_mutex_unlock(&K.mu);			/* stopping, or another path made it ready meanwhile */
		*secs_out = 0;
		return -2;
	}
	set_state_locked(e, GPUSERV_KSTATE_COMPILING);
	snprintf(K.st->current, sizeof(K.st->current), "%s", e->name);
	K.st->current_since = real_now();
	K.compiling = 1;
	pthread_mutex_unlock(&K.mu);

	err[0] = '\0';
	rc = gpu_vk_pipeline_create_ex(K.vk, &pl, e->spv, e->len, e->nbind, e->pcsz, err, sizeof(err));
	if (rc == 0)
	{
		rc = gpu_vk_warm_dispatch(K.vk, &pl, err, sizeof(err));
		gpu_vk_pipeline_destroy(K.vk, &pl);
	}
	*secs_out = mono_now() - t0;

	pthread_mutex_lock(&K.mu);
	K.compiling = 0;
	K.st->current[0] = '\0';
	K.st->current_since = 0;
	e->demand = 0;
	if (rc == 0)
	{
		/*
		 * A compile that took long must have grown the cache file; if the file did not grow (the size bound, a full disk) the
		 * kernel is ready for this process only and the next start compiles it again.
		 */
		e->persist = (*secs_out < 0.05 || gpu_vk_pcache_saved(K.vk) > before);
		K.st->k[i].compile_secs = (float) *secs_out;
		set_state_locked(e, GPUSERV_KSTATE_READY);
		if (e->persist)
			manifest_save_locked();
		else
			fprintf(stderr, "pgstrom-gpuserv[%d]: prewarm: %s compiled in %.1f s but the pipeline cache file did not grow "
					"(size bound or write failure): it will be compiled again at the next start\n",
					(int) getpid(), e->name, *secs_out);
	}
	else
	{
		set_state_locked(e, GPUSERV_KSTATE_FAILED);
		fprintf(stderr, "pgstrom-gpuserv[%d]: prewarm: %s: %s\n", (int) getpid(), e->name, err);
	}
	pthread_mutex_unlock(&K.mu);
	return rc;
}

/* ----------------------------------------------------------------
 * the gate (main thread)
 * ----------------------------------------------------------------
 */
static int
kpw_gate(void *arg, const void *spirv, size_t spirv_bytes, uint32_t nbind, uint32_t pc_size, char *err, size_t errlen)
{
	kent	   *e = find_ent(spirv);
	uint32_t	field = K.sess_wait;
	long		budget;				/* -1: none */
	struct timespec dl;

	(void) arg;
	(void) spirv_bytes;
	if (!e || nbind != e->nbind || pc_size != e->pcsz)
		return 0;
	budget = (K.force_wait || field == 0 || field == KPW_WAIT_NONE) ? -1 : (long) field - 1;
	pthread_mutex_lock(&K.mu);
	verify_locked(e);
	if (K.st->k[e - K.ents].state == GPUSERV_KSTATE_READY || K.st->k[e - K.ents].state == GPUSERV_KSTATE_FAILED)
	{
		pthread_mutex_unlock(&K.mu);
		return 0;
	}
	if (budget < 0)
	{
		/*
		 * No gate (the backend cannot continue on the CPU, or the old behavior was asked for): the caller creates the pipeline
		 * itself and waits as long as it takes. Nothing is recorded here (the prewarm thread finds it in the pipeline cache and
		 * records it, in milliseconds). A kernel the prewarm thread is compiling right now is waited for, not compiled twice.
		 */
		while (K.st->k[e - K.ents].state == GPUSERV_KSTATE_COMPILING && !K.stop)
			pthread_cond_wait(&K.done_cv, &K.mu);
		pthread_mutex_unlock(&K.mu);
		return 0;
	}
	if (e->refused)
		budget = 0;				/* one task already waited for it: do not stall the service again */
	if (!e->demand)
	{
		e->demand = 1;
		K.st->demanded++;
		pthread_cond_broadcast(&K.cv);
	}
	clock_gettime(CLOCK_REALTIME, &dl);
	dl.tv_sec += budget / 1000;
	dl.tv_nsec += (budget % 1000) * 1000000L;
	if (dl.tv_nsec >= 1000000000L)
	{
		dl.tv_sec++;
		dl.tv_nsec -= 1000000000L;
	}
	while (K.st->k[e - K.ents].state != GPUSERV_KSTATE_READY && K.st->k[e - K.ents].state != GPUSERV_KSTATE_FAILED && !K.stop)
	{
		if (budget == 0 || pthread_cond_timedwait(&K.done_cv, &K.mu, &dl) == ETIMEDOUT)
			break;
	}
	if (K.st->k[e - K.ents].state == GPUSERV_KSTATE_READY || K.st->k[e - K.ents].state == GPUSERV_KSTATE_FAILED)
	{
		K.st->waited++;
		pthread_mutex_unlock(&K.mu);
		return 0;
	}
	K.st->refused++;
	e->refused = 1;
	snprintf(err, errlen, "kernel %s is not compiled yet: the GPU driver is compiling it in the background (see pgstrom.gpu_kernel_state())",
			 e->name);
	pthread_mutex_unlock(&K.mu);
	return -1;
}

void
kpw_set_session_wait(uint32_t session_field)
{
	K.sess_wait = session_field;
}

void
kpw_task_begin(void)
{
	__atomic_add_fetch(&K.busy, 1, __ATOMIC_SEQ_CST);
}

void
kpw_task_end(void)
{
	K.last_active = mono_now();
	__atomic_sub_fetch(&K.busy, 1, __ATOMIC_SEQ_CST);
}

/* ----------------------------------------------------------------
 * the thread
 * ----------------------------------------------------------------
 */
/* caller holds K.mu: the next kernel to compile, demanded ones first; NULL if none. *demanded says which kind. */
static kent *
pick_next_locked(int *demanded)
{
	kent	   *best = NULL;

	for (int i = 0; i < K.n; i++)
	{
		kent   *e = &K.ents[i];
		int		state = K.st->k[i].state;

		if (state != GPUSERV_KSTATE_PENDING)
			continue;
		if (e->demand)
		{
			*demanded = 1;
			return e;
		}
	}
	*demanded = 0;
	for (int i = 0; i < K.n; i++)
	{
		kent   *e = &K.ents[i];

		if (K.st->k[i].state == GPUSERV_KSTATE_PENDING && K.st->k[i].in_set &&
			(!best || e->rank < best->rank))
			best = e;
	}
	return best;
}

static void
lower_priority(void)
{
#ifdef __APPLE__
	pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#elif defined(SYS_gettid)
	(void) setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#endif
}

/* wait until the service is quiet (or a demand / stop arrives); returns 1 to go on compiling, 0 if stopping */
static int
wait_quiet(void)
{
	pthread_mutex_lock(&K.mu);
	for (;;)
	{
		struct timespec dl;
		int			dummy;
		double		quiet = mono_now() - K.last_active;

		if (K.stop)
		{
			pthread_mutex_unlock(&K.mu);
			return 0;
		}
		if (__atomic_load_n(&K.busy, __ATOMIC_SEQ_CST) == 0 && quiet * 1000.0 >= (double) K.idle_ms)
			break;
		{
			kent   *d = pick_next_locked(&dummy);

			if (d && dummy)			/* a task asked for a kernel: no waiting */
				break;
		}
		clock_gettime(CLOCK_REALTIME, &dl);
		dl.tv_nsec += 200 * 1000000L;
		if (dl.tv_nsec >= 1000000000L)
		{
			dl.tv_sec++;
			dl.tv_nsec -= 1000000000L;
		}
		pthread_cond_timedwait(&K.cv, &K.mu, &dl);
	}
	pthread_mutex_unlock(&K.mu);
	return 1;
}

static void *
prewarm_main(void *arg)
{
	sigset_t	ss;
	int			announced = 0;

	(void) arg;
	sigfillset(&ss);
	pthread_sigmask(SIG_BLOCK, &ss, NULL);
	lower_priority();

	/* the first look: kernels the manifest vouches for become ready without a compile */
	for (int i = 0; i < K.n; i++)
		if (K.ents[i].has_mrec)
		{
			uint64_t	h = spv_hash(K.ents[i].spv, K.ents[i].len);		/* not under the lock: 10 MB of SPIR-V */

			pthread_mutex_lock(&K.mu);
			if (!K.ents[i].hashed)
			{
				K.ents[i].hash = h;
				apply_manifest_locked(&K.ents[i]);
			}
			pthread_mutex_unlock(&K.mu);
			if (K.stop)
				break;
		}

	/* give the service time to start serving */
	{
		struct timespec dl;

		clock_gettime(CLOCK_REALTIME, &dl);
		dl.tv_sec += K.delay_ms / 1000;
		dl.tv_nsec += (K.delay_ms % 1000) * 1000000L;
		if (dl.tv_nsec >= 1000000000L)
		{
			dl.tv_sec++;
			dl.tv_nsec -= 1000000000L;
		}
		pthread_mutex_lock(&K.mu);
		while (!K.stop)
		{
			int			dem = 0;

			(void) pick_next_locked(&dem);
			if (dem)					/* a task already asked for a kernel */
				break;
			if (pthread_cond_timedwait(&K.cv, &K.mu, &dl) == ETIMEDOUT)
				break;
		}
		pthread_mutex_unlock(&K.mu);
	}

	for (;;)
	{
		kent	   *e;
		int			demanded = 0;
		double		secs;

		pthread_mutex_lock(&K.mu);
		if (K.stop || K.vk->lost)
		{
			pthread_mutex_unlock(&K.mu);
			break;
		}
		e = pick_next_locked(&demanded);
		if (!e)
		{
			if (K.st->phase != GPUSERV_PHASE_DONE && K.st->phase == GPUSERV_PHASE_RUNNING)
			{
				K.st->phase = GPUSERV_PHASE_DONE;
				K.st->finished_at = real_now();
				if (K.mode != GPUSERV_PREWARM_OFF)
				fprintf(stderr, "pgstrom-gpuserv[%d]: prewarm: the %s set is compiled (%u kernels, %u failed)\n", (int) getpid(),
						K.mode == GPUSERV_PREWARM_LITE ? "lite" : K.mode == GPUSERV_PREWARM_GEOMETRY ? "geometry" : "all",
						K.st->target, K.st->nfailed);
			}
			pthread_cond_wait(&K.cv, &K.mu);		/* a demand or stop */
			pthread_mutex_unlock(&K.mu);
			continue;
		}
		if (K.st->phase == GPUSERV_PHASE_NONE || (K.st->phase == GPUSERV_PHASE_DONE && !demanded))
		{
			K.st->phase = GPUSERV_PHASE_RUNNING;
			K.st->started_at = real_now();
			K.st->finished_at = 0;
		}
		pthread_mutex_unlock(&K.mu);
		if (!announced && K.st->target > 0 && K.st->target_ready < K.st->target)
		{
			announced = 1;
			fprintf(stderr, "pgstrom-gpuserv[%d]: prewarm: %u of %u kernels of the set are compiled, compiling the rest in the "
					"background (one at a time, while no task runs)\n", (int) getpid(), K.st->target_ready, K.st->target);
		}
		if (!demanded)
		{
			if (!wait_quiet())
				break;
			/* a task may have asked for another kernel while we waited */
			pthread_mutex_lock(&K.mu);
			e = pick_next_locked(&demanded);
			pthread_mutex_unlock(&K.mu);
			if (!e)
				continue;
		}
		compile_ent(e, &secs);
		if (getenv("PGSTROM_GPUSERV_PIPELINE_LOG"))
			fprintf(stderr, "pgstrom-gpuserv[%d]: prewarm: %s %s in %.2f s\n", (int) getpid(), e->name,
					K.st->k[e - K.ents].state == GPUSERV_KSTATE_READY ? "ready" : "FAILED", secs);
	}
	return NULL;
}

int
kpw_start(gpu_vk_ctx *vk, gpuserv_prewarm_state *st, int mode)
{
	const char *e;

	if (!vk->pcache)
		return 0;						/* nothing could be kept: the old behavior */
	e = getenv("PGSTROM_GPUSERV_PREWARM");
	if (e && e[0] && kpw_parse_mode(e) >= 0)
		mode = kpw_parse_mode(e);
	if (mode < 0)
		mode = GPUSERV_PREWARM_LITE;
	K.idle_ms = getenv("PGSTROM_GPUSERV_PREWARM_IDLE_MS") ? atol(getenv("PGSTROM_GPUSERV_PREWARM_IDLE_MS")) : 2000;
	K.delay_ms = getenv("PGSTROM_GPUSERV_PREWARM_DELAY_MS") ? atol(getenv("PGSTROM_GPUSERV_PREWARM_DELAY_MS")) : 3000;
	e = getenv("PGSTROM_GPUSERV_KERNEL_WAIT_MS");
	K.force_wait = (e && atol(e) < 0 && e[0] == '-');
	K.last_active = mono_now();
	if (mode == GPUSERV_PREWARM_OFF && K.force_wait)
		return 0;						/* nothing to prewarm and no gate: do not even look at the kernels (the tests) */
	kreg_init(vk, st, mode);
	if (pthread_create(&K.thr, NULL, prewarm_main, NULL) == 0)
	{
		K.thr_started = 1;
		vk->create_gate = kpw_gate;		/* only with a thread that can compile what the gate queues */
		vk->create_gate_arg = NULL;
	}
	return K.thr_started;
}

int
kpw_stop(int wait_ms)
{
	double		t0 = mono_now();

	if (!K.thr_started)
		return 1;
	pthread_mutex_lock(&K.mu);
	K.stop = 1;
	pthread_cond_broadcast(&K.cv);
	pthread_cond_broadcast(&K.done_cv);
	pthread_mutex_unlock(&K.mu);
	for (;;)
	{
		pthread_mutex_lock(&K.mu);
		if (!K.compiling)
		{
			pthread_mutex_unlock(&K.mu);
			pthread_join(K.thr, NULL);
			K.thr_started = 0;
			return 1;
		}
		pthread_mutex_unlock(&K.mu);
		if ((mono_now() - t0) * 1000.0 > wait_ms)
			return 0;
		usleep(20000);
	}
}

/* ----------------------------------------------------------------
 * command line
 * ----------------------------------------------------------------
 */
int
kpw_cli(gpu_vk_ctx *vk, int mode, int list_only)
{
	static gpuserv_prewarm_state st;
	kent	   *order[GPUSERV_KERNEL_MAX];
	int			n = 0, nfail = 0, ntodo = 0;
	double		t0 = mono_now();

	kreg_init(vk, &st, mode);
	K.force_wait = 1;
	for (int i = 0; i < K.n; i++)
		if (st.k[i].in_set)
			order[n++] = &K.ents[i];
	qsort(order, n, sizeof(order[0]), rank_cmp);
	pthread_mutex_lock(&K.mu);
	for (int i = 0; i < n; i++)
		verify_locked(order[i]);
	pthread_mutex_unlock(&K.mu);
	for (int i = 0; i < n; i++)
		if (st.k[order[i] - K.ents].state != GPUSERV_KSTATE_READY)
			ntodo++;
	printf("prewarm: %d kernels in the set, %d to compile (pipeline cache %s)\n", n, ntodo,
		   gpu_vk_pcache_path(vk)[0] ? gpu_vk_pcache_path(vk) : "OFF: nothing would be kept");
	fflush(stdout);
	for (int i = 0; i < n; i++)
	{
		kent	   *e = order[i];
		double		secs = 0;
		int			ready = (st.k[e - K.ents].state == GPUSERV_KSTATE_READY);

		if (list_only)
		{
			printf("  %-20s %-9s %6u KB  %s\n", e->name,
				   e->tier == GPUSERV_KTIER_LITE ? "lite" : e->tier == GPUSERV_KTIER_NOGEOM ? "nogeom" :
				   e->tier == GPUSERV_KTIER_GEOM ? "geometry" : "full", (unsigned) (e->len / 1024),
				   ready ? "ready" : "to compile");
			continue;
		}
		if (ready)
			continue;
		if (compile_ent(e, &secs) != 0)
		{
			nfail++;
			printf("  %-20s FAILED after %.1f s\n", e->name, secs);
		}
		else
			printf("  %-20s compiled in %7.1f s   (%d of %d)\n", e->name, secs, i + 1, n);
		fflush(stdout);
	}
	if (!list_only)
		printf("prewarm: done in %.1f s, %d failed\n", mono_now() - t0, nfail);
	return list_only ? 0 : (nfail ? 1 : 0);
}
