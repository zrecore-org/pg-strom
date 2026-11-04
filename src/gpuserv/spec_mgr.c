/*
 * spec_mgr.c
 *
 * A6 kernel specialization, the manager: compiles the generated kernels (specialize.c) with slangc in a
 * worker thread, caches the SPIR-V on disk, creates the pipelines, and hands them to the tasks.
 *
 * Thread rules: the service is single-threaded (poll loop); the only other thread is this worker.
 * The main thread queues entries and reads their state (atomic); the worker owns an entry from the
 * moment it is queued until it publishes READY or FAILED, and creates its pipeline (Vulkan pipeline
 * creation needs no external synchronization; a private error buffer is used). Nothing else of the
 * engine is touched by the worker.
 *
 * Options (environment of the service, which inherits the postmaster's):
 *   PGSTROM_GPUSERV_SPECIALIZE = off | auto (default) | force | check
 *       auto:  a session's second task queues the compilation; the VM serves tasks meanwhile
 *       force: the first task queues it and waits for it (tests)
 *       check: force, and every dispatch also runs the VM and compares (scan_exec.c)
 *   PGSTROM_GPUSERV_SPECIALIZE_TASKS = N   the task of a session that queues the compilation (auto, default 2)
 *   PGSTROM_GPUSERV_CACHE_DIR = dir | none  the SPIR-V cache (default $XDG_CACHE_HOME/pgstrom or ~/.cache/pgstrom)
 *   PGSTROM_GPUSERV_SLANGC = path           the compiler (default: the one found at build time, then PATH)
 *   PGSTROM_GPUSERV_SPEC_KEEP               keep the scratch directories of the compiler
 *   PGSTROM_GPUSERV_SPEC_LOG                log what the specializer does (also with PGSTROM_GPUSERV_TRACE)
 *   PGSTROM_GPUSERV_CACHE_MAX_MB = N        size limit of the cache directory (default 256)
 */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xpu_common.h"
#include "scan_exec.h"
#include "scan_prog.h"
#include "scan_caps.h"
#include "specialize.h"
#include "kernels_src.h"

extern char **environ;

enum { SPEC_Q = 0, SPEC_READY = 1, SPEC_FAILED = 2 };
enum { SPECM_OFF = 0, SPECM_AUTO = 1, SPECM_FORCE = 2, SPECM_CHECK = 3 };

typedef struct spec_entry
{
	struct spec_entry *next;		/* all entries */
	struct spec_entry *qnext;		/* the job queue */
	uint64_t	key[2];
	char		keyhex[33];
	int			kind;
	int			arrow;
	int			agg_max, ga_c;
	spec_shape	shape;
	char	   *defs, *body;
	int			refs;				/* programs using it (main thread only) */
	int			state;				/* SPEC_Q / READY / FAILED: set by the worker (atomic) */
	int			announced;			/* the main thread told the trace that it switched */
	gpu_vk_pipeline pipe;
	char		err[256];
	uint64_t	last_used;
} spec_entry;

struct spec_mgr
{
	gpu_vk_ctx *vk;
	int			trace;
	int			mode;
	int			threshold;			/* the task of a session that queues the compilation (auto) */
	char		cache_dir[1024];	/* "" = no disk cache */
	char		tmp_base[1024];		/* where the scratch directories go */
	char		slangc[1024];
	char		slang_version[128];
	char		srcdir[1100];		/* the extracted kernel sources */
	char		src_hash[17];
	uint64_t	src_fp;				/* hash of the embedded kernel sources */
	int			keep;
	int			max_entries;

	pthread_t	thr;
	int			thr_started;
	pthread_mutex_t mu;
	pthread_cond_t cv;				/* a job is queued / stop */
	pthread_cond_t done_cv;			/* a job finished */
	spec_entry *qhead, *qtail;
	int			stop;
	volatile pid_t child;

	spec_entry *entries;
	int			nentries;
	uint64_t	clock;
	unsigned long n_hits_disk, n_compiled, n_failed;
};

static double
now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static uint64_t
fnv64(uint64_t h, const void *data, size_t len)
{
	const unsigned char *p = data;

	for (size_t i = 0; i < len; i++)
	{
		h ^= p[i];
		h *= 0x100000001b3ULL;
	}
	return h;
}

static void
spec_log(spec_mgr *m, int always, const char *fmt, ...)
{
	va_list		ap;
	char		buf[700];

	if (!always && !m->trace)
		return;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	fprintf(stderr, "pgstrom-gpuserv: specialize: %s\n", buf);
}

static const char *
kind_name(int kind)
{
	if (kind & SPEC_ARROW)
	{
		static char names[8][24];
		static int	next = 0;
		char	   *p = names[next++ & 7];

		snprintf(p, 24, "arrow %s", kind_name(kind & ~SPEC_ARROW));
		return p;
	}
	switch (kind)
	{
		case SPEC_SCAN:		return "scan";
		case SPEC_AGG:		return "agg";
		case SPEC_GAGG:		return "group";
		case SPEC_JOIN:		return "join";
		case SPEC_JAGG:		return "join agg";
		case SPEC_JGAGG:	return "join group";
	}
	return "?";
}

static const char *
kind_entry(int kind)
{
	switch (kind)
	{
		case SPEC_SCAN:		return "scan_main";
		case SPEC_AGG:		return "agg_main";
		case SPEC_GAGG:		return "gagg_main";
		case SPEC_JOIN:		return "join_main";
		case SPEC_JAGG:		return "jagg_main";
		case SPEC_JGAGG:	return "jgagg_main";
	}
	return "main";
}

static const char *
kind_file(int kind)
{
	switch (kind)
	{
		case SPEC_SCAN:		return "scan.slang";
		case SPEC_AGG:		return "agg.slang";
		case SPEC_GAGG:		return "gagg.slang";
		case SPEC_JOIN:		return "join.slang";
		case SPEC_JAGG:		return "jagg.slang";
		case SPEC_JGAGG:	return "jgagg.slang";
	}
	return "scan.slang";
}

/* ---- small file helpers ---- */
static int
mkdir_p1(const char *path)
{
	if (mkdir(path, 0700) == 0 || errno == EEXIST)
		return 0;
	return -1;
}

/* creates the directory (and one missing parent level, like ~/.cache) with mode 0700; 0 if it is usable */
static int
ensure_dir(const char *path)
{
	char		parent[1024];
	char	   *slash;

	if (access(path, W_OK | X_OK) == 0)
		return 0;
	snprintf(parent, sizeof(parent), "%s", path);
	slash = strrchr(parent, '/');
	if (slash && slash != parent)
	{
		*slash = '\0';
		if (access(parent, F_OK) != 0)
			(void) mkdir_p1(parent);
	}
	if (mkdir_p1(path) != 0)
		return -1;
	return access(path, W_OK | X_OK);
}

static char *
read_file(const char *path, size_t *p_len)
{
	FILE	   *f = fopen(path, "rb");
	char	   *buf;
	long		n;

	if (!f)
		return NULL;
	if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0)
	{
		fclose(f);
		return NULL;
	}
	buf = malloc(n + 1);
	if (!buf || (long) fread(buf, 1, n, f) != n)
	{
		free(buf);
		fclose(f);
		return NULL;
	}
	fclose(f);
	buf[n] = '\0';
	*p_len = n;
	return buf;
}

static int
write_file(const char *path, const void *data, size_t len)
{
	FILE	   *f = fopen(path, "wb");

	if (!f)
		return -1;
	if (len > 0 && fwrite(data, 1, len, f) != len)
	{
		fclose(f);
		return -1;
	}
	return fclose(f);
}

/* write to a temporary name and rename: concurrent services and a crash never leave a half file */
static int
write_file_atomic(const char *path, const void *data, size_t len)
{
	char		tmp[1200];

	snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path, (int) getpid());
	if (write_file(tmp, data, len) != 0 || rename(tmp, path) != 0)
	{
		unlink(tmp);
		return -1;
	}
	return 0;
}

static void
remove_tree(const char *path)
{
	/* the scratch directories hold a handful of plain files */
	pid_t		pid;
	char	   *argv[] = {"rm", "-rf", (char *) path, NULL};

	if (posix_spawnp(&pid, "rm", NULL, NULL, argv, environ) == 0)
	{
		int			st;

		(void) waitpid(pid, &st, 0);
	}
}

/* run argv with stdout and stderr in logpath; returns the exit status or -1 */
static int
run_cmd(spec_mgr *m, char *const argv[], const char *logpath)
{
	posix_spawn_file_actions_t fa;
	pid_t		pid;
	int			st = 0, rc;

	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
	posix_spawn_file_actions_addopen(&fa, 1, logpath, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	posix_spawn_file_actions_adddup2(&fa, 1, 2);
	rc = posix_spawn(&pid, argv[0], &fa, NULL, argv, environ);
	posix_spawn_file_actions_destroy(&fa);
	if (rc != 0)
		return -1;
	m->child = pid;
	while (waitpid(pid, &st, 0) < 0)
	{
		if (errno != EINTR)
		{
			m->child = 0;
			return -1;
		}
	}
	m->child = 0;
	return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

/* ---- the compiler and the kernel sources ---- */
static int
find_slangc(spec_mgr *m)
{
	const char *env = getenv("PGSTROM_GPUSERV_SLANGC");
	const char *path;

	m->slangc[0] = '\0';
	if (env && env[0])
	{
		if (access(env, X_OK) == 0)
			snprintf(m->slangc, sizeof(m->slangc), "%s", env);
		return m->slangc[0] ? 0 : -1;
	}
#ifdef SPEC_SLANGC_PATH
	if (access(SPEC_SLANGC_PATH, X_OK) == 0)
	{
		snprintf(m->slangc, sizeof(m->slangc), "%s", SPEC_SLANGC_PATH);
		return 0;
	}
#endif
	path = getenv("PATH");
	if (path)
	{
		char	   *copy = strdup(path), *save = NULL, *dir;

		for (dir = strtok_r(copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save))
		{
			char		cand[1200];

			snprintf(cand, sizeof(cand), "%s/slangc", dir);
			if (access(cand, X_OK) == 0)
			{
				snprintf(m->slangc, sizeof(m->slangc), "%s", cand);
				break;
			}
		}
		free(copy);
	}
	return m->slangc[0] ? 0 : -1;
}

static uint64_t
sources_fingerprint(void)
{
	uint64_t	h = 0xcbf29ce484222325ULL;

	for (unsigned int i = 0; i < ksrc_nfiles; i++)
	{
		h = fnv64(h, ksrc_files[i].name, strlen(ksrc_files[i].name) + 1);
		h = fnv64(h, ksrc_files[i].data, ksrc_files[i].len);
	}
	return h;
}

/* extract the embedded kernel sources into a directory named after their hash; 0 on success */
static int
extract_sources(spec_mgr *m)
{
	char		marker[1200], tmpdir[1200];
	const char *base = m->cache_dir[0] ? m->cache_dir : m->tmp_base;

	snprintf(m->src_hash, sizeof(m->src_hash), "%016llx", (unsigned long long) m->src_fp);
	snprintf(m->srcdir, sizeof(m->srcdir), "%s/src-%s", base, m->src_hash);
	snprintf(marker, sizeof(marker), "%s/.complete", m->srcdir);
	if (access(marker, F_OK) == 0)
		return 0;
	snprintf(tmpdir, sizeof(tmpdir), "%s/src-%s.%d.tmp", base, m->src_hash, (int) getpid());
	if (mkdir_p1(tmpdir) != 0)
		return -1;
	for (unsigned int i = 0; i < ksrc_nfiles; i++)
	{
		char		fp[1400];

		snprintf(fp, sizeof(fp), "%s/%s", tmpdir, ksrc_files[i].name);
		if (write_file(fp, ksrc_files[i].data, ksrc_files[i].len) != 0)
		{
			remove_tree(tmpdir);
			m->srcdir[0] = '\0';
			return -1;
		}
	}
	{
		char		fp[1400];

		snprintf(fp, sizeof(fp), "%s/.complete", tmpdir);
		if (write_file(fp, "", 0) != 0)
		{
			remove_tree(tmpdir);
			m->srcdir[0] = '\0';
			return -1;
		}
	}
	if (rename(tmpdir, m->srcdir) != 0)
	{
		/* another service won the race: its copy is as good */
		remove_tree(tmpdir);
		if (access(marker, F_OK) != 0)
		{
			m->srcdir[0] = '\0';
			return -1;
		}
	}
	return 0;
}

static void
slang_version(spec_mgr *m)
{
	char		logp[1300];
	char	   *argv[] = {m->slangc, "-v", NULL};
	char	   *txt;
	size_t		n;

	snprintf(m->slang_version, sizeof(m->slang_version), "unknown");
	snprintf(logp, sizeof(logp), "%s/slangc-version.%d", m->tmp_base, (int) getpid());
	if (run_cmd(m, argv, logp) == 0 && (txt = read_file(logp, &n)) != NULL)
	{
		char	   *nl = strchr(txt, '\n');

		if (nl)
			*nl = '\0';
		snprintf(m->slang_version, sizeof(m->slang_version), "%s", txt);
		free(txt);
	}
	unlink(logp);
}

/* the part of the generated text that identifies it: defs, then body */
static char *
entry_text(const spec_entry *e, size_t *p_len)
{
	size_t		ld = strlen(e->defs), lb = strlen(e->body);
	char	   *t = malloc(ld + lb + 32);

	if (!t)
		return NULL;
	memcpy(t, e->defs, ld);
	memcpy(t + ld, "\n//----\n", 8);
	memcpy(t + ld + 8, e->body, lb + 1);
	*p_len = ld + 8 + lb;
	return t;
}

/* compile the entry's source into SPIR-V (malloc'ed); NULL with e->err set */
static void *
compile_entry(spec_mgr *m, spec_entry *e, size_t *p_len, double *p_ms)
{
	char		jobdir[1300], fp[1400], spv[1400], logp[1400], mainp[1400];
	char		tmpl[1300];
	char		text[160];
	size_t		len;
	void	   *out = NULL;
	double		t0 = now_ms();

	if (!m->slangc[0])
	{
		snprintf(e->err, sizeof(e->err), "slangc not found");
		return NULL;
	}
	if (!m->srcdir[0])
	{
		snprintf(e->err, sizeof(e->err), "the kernel sources are not available");
		return NULL;
	}
	snprintf(tmpl, sizeof(tmpl), "%s/job.XXXXXX", m->tmp_base);
	if (!mkdtemp(tmpl))
	{
		snprintf(e->err, sizeof(e->err), "cannot create a scratch directory in %s: %s", m->tmp_base, strerror(errno));
		return NULL;
	}
	snprintf(jobdir, sizeof(jobdir), "%s", tmpl);
	snprintf(fp, sizeof(fp), "%s/spec_defs.inc", jobdir);
	if (write_file(fp, e->defs, strlen(e->defs)) != 0)
		goto fail_write;
	snprintf(fp, sizeof(fp), "%s/spec_gen.inc", jobdir);
	if (write_file(fp, e->body, strlen(e->body)) != 0)
		goto fail_write;
	snprintf(text, sizeof(text), "#include \"spec_defs.inc\"\n#include \"%s\"\n",
			 (e->arrow && e->kind == SPEC_SCAN) ? "arrow.slang" :
			 e->shape.dm ? (e->kind == SPEC_GAGG ? "gagg_dm.slang" : "jgagg_dm.slang") : kind_file(e->kind));
	snprintf(mainp, sizeof(mainp), "%s/main.slang", jobdir);
	if (write_file(mainp, text, strlen(text)) != 0)
		goto fail_write;
	snprintf(spv, sizeof(spv), "%s/out.spv", jobdir);
	snprintf(logp, sizeof(logp), "%s/slangc.log", jobdir);
	{
		char	   *argv[] = {m->slangc, mainp, "-I", jobdir, "-I", m->srcdir, "-target", "spirv", "-entry",
							  (char *) ((e->arrow && e->kind == SPEC_SCAN) ? "arrow_main" : kind_entry(e->kind)),
							  "-stage", "compute", "-o", spv, NULL};
		int			rc = run_cmd(m, argv, logp);

		if (rc != 0)
		{
			char	   *log = read_file(logp, &len);

			snprintf(e->err, sizeof(e->err), "slangc failed (%d): %.180s", rc, log ? log : "");
			free(log);
			goto done;
		}
	}
	out = read_file(spv, p_len);
	if (!out || *p_len < 20 || (*p_len % 4) != 0)
	{
		free(out);
		out = NULL;
		snprintf(e->err, sizeof(e->err), "slangc produced no SPIR-V");
	}
	goto done;
fail_write:
	snprintf(e->err, sizeof(e->err), "cannot write the generated source: %s", strerror(errno));
done:
	if (!m->keep)
		remove_tree(jobdir);
	else
		spec_log(m, 1, "kept %s", jobdir);
	*p_ms = now_ms() - t0;
	return out;
}

static void
cache_paths(spec_mgr *m, const spec_entry *e, char *spvp, char *srcp, size_t n)
{
	snprintf(spvp, n, "%s/%s.spv", m->cache_dir, e->keyhex);
	snprintf(srcp, n, "%s/%s.src", m->cache_dir, e->keyhex);
}

/* a cached SPIR-V whose recorded source equals this entry's (not just the same hash) */
static void *
cache_load(spec_mgr *m, spec_entry *e, size_t *p_len)
{
	char		spvp[1300], srcp[1300];
	size_t		l1, l2, tl;
	char	   *saved, *text;
	void	   *spv;

	if (!m->cache_dir[0])
		return NULL;
	cache_paths(m, e, spvp, srcp, sizeof(spvp));
	saved = read_file(srcp, &l1);
	if (!saved)
		return NULL;
	text = entry_text(e, &tl);
	if (!text || tl != l1 || memcmp(text, saved, tl) != 0)
	{
		free(saved);
		free(text);
		return NULL;
	}
	free(saved);
	free(text);
	spv = read_file(spvp, &l2);
	if (!spv || l2 < 20 || (l2 % 4) != 0)
	{
		free(spv);
		return NULL;
	}
	*p_len = l2;
	utimes(spvp, NULL);				/* least recently used first when the cache is pruned */
	return spv;
}

static void
cache_store(spec_mgr *m, spec_entry *e, const void *spv, size_t len)
{
	char		spvp[1300], srcp[1300];
	size_t		tl;
	char	   *text;

	if (!m->cache_dir[0])
		return;
	text = entry_text(e, &tl);
	if (!text)
		return;
	cache_paths(m, e, spvp, srcp, sizeof(spvp));
	/* the source first: a reader that sees the .spv finds the .src that goes with it */
	if (write_file_atomic(srcp, text, tl) == 0)
		(void) write_file_atomic(spvp, spv, len);
	free(text);
}

static void
build_entry(spec_mgr *m, spec_entry *e)
{
	size_t		len = 0;
	void	   *spv;
	double		t0 = now_ms(), t_compile = 0, t1;
	int			from_cache = 0;
	/* the bindings and the push constants of the kernel: the join kernels have the inner buffer and a fifth word, so do
	 * the aggregations of an Arrow record batch (arrow_exec.c) */
	uint32_t	nbind = (e->kind >= SPEC_JOIN ? 6 : 5);
	uint32_t	pcs = (e->kind >= SPEC_JOIN || (e->arrow && (e->kind == SPEC_AGG || e->kind == SPEC_GAGG))) ? 20 : 16;
	char		perr[256];

	spv = cache_load(m, e, &len);
	if (spv)
		from_cache = 1;
	else
	{
		if (m->slangc[0] && !m->srcdir[0])
			(void) extract_sources(m);
		spv = compile_entry(m, e, &len, &t_compile);
		if (!spv)
			return;
		cache_store(m, e, spv, len);
	}
	t1 = now_ms();
	perr[0] = '\0';
	if (gpu_vk_pipeline_create_ex(m->vk, &e->pipe, spv, len, nbind, pcs, perr, sizeof(perr)) != 0)
	{
		snprintf(e->err, sizeof(e->err), "pipeline: %s", perr);
		free(spv);
		return;
	}
	free(spv);
	spec_log(m, 0, "%s kernel %.8s ready: %s %.0f ms, pipeline %.0f ms, %.0f ms in all",
			 kind_name(e->kind | (e->arrow ? SPEC_ARROW : 0)), e->keyhex, from_cache ? "disk cache" : "slangc",
			 from_cache ? t1 - t0 : t_compile, now_ms() - t1, now_ms() - t0);
	if (from_cache)
		m->n_hits_disk++;
	else
		m->n_compiled++;
	e->err[0] = '\0';
	__atomic_store_n(&e->state, SPEC_READY, __ATOMIC_RELEASE);
}

/*
 * Keep the cache directory bounded: kernels are 20-100 KB of SPIR-V plus their source. When the files of
 * *.spv / *.src exceed the limit (PGSTROM_GPUSERV_CACHE_MAX_MB, default 256) the least recently used are
 * removed (a cache hit touches the file); source directories of other builds and temporary files that no
 * service is writing any more (a day old) are removed.
 */
typedef struct
{
	char		name[64];
	time_t		mtime;
	off_t		size;
} cache_file;

static int
cmp_cache_file(const void *a, const void *b)
{
	const cache_file *x = a, *y = b;

	return (x->mtime < y->mtime) ? -1 : (x->mtime > y->mtime) ? 1 : 0;
}

static void
prune_cache(spec_mgr *m)
{
	DIR		   *d;
	struct dirent *de;
	cache_file *files = NULL;
	size_t		nfiles = 0, cap = 0;
	off_t		total = 0, limit;
	const char *s = getenv("PGSTROM_GPUSERV_CACHE_MAX_MB");
	time_t		now = time(NULL);

	if (!m->cache_dir[0])
		return;
	limit = (off_t) ((s && atoi(s) > 0) ? atoi(s) : 256) * 1024 * 1024;
	d = opendir(m->cache_dir);
	if (!d)
		return;
	while ((de = readdir(d)) != NULL)
	{
		char		path[1400];
		struct stat st;
		size_t		nl = strlen(de->d_name);

		if (de->d_name[0] == '.' || nl >= sizeof(files->name))
			continue;
		snprintf(path, sizeof(path), "%s/%s", m->cache_dir, de->d_name);
		if (stat(path, &st) != 0)
			continue;
		if (S_ISDIR(st.st_mode))
		{
			/* kernel sources of another build: gone after a month without a start of that build */
			if (strncmp(de->d_name, "src-", 4) == 0 && now - st.st_mtime > 30 * 86400)
				remove_tree(path);
			continue;
		}
		if (nl > 4 && strcmp(de->d_name + nl - 4, ".tmp") == 0)
		{
			if (now - st.st_mtime > 86400)
				unlink(path);
			continue;
		}
		if (!(nl > 4 && (strcmp(de->d_name + nl - 4, ".spv") == 0 || strcmp(de->d_name + nl - 4, ".src") == 0)))
			continue;
		if (nfiles == cap)
		{
			cache_file *nf = realloc(files, sizeof(cache_file) * (cap ? cap * 2 : 256));

			if (!nf)
				break;
			files = nf;
			cap = cap ? cap * 2 : 256;
		}
		snprintf(files[nfiles].name, sizeof(files[nfiles].name), "%s", de->d_name);
		files[nfiles].mtime = st.st_mtime;
		files[nfiles].size = st.st_size;
		total += st.st_size;
		nfiles++;
	}
	closedir(d);
	if (total > limit && nfiles > 0)
	{
		qsort(files, nfiles, sizeof(cache_file), cmp_cache_file);
		for (size_t i = 0; i < nfiles && total > limit - limit / 4; i++)
		{
			char		path[1400];

			snprintf(path, sizeof(path), "%s/%s", m->cache_dir, files[i].name);
			if (unlink(path) == 0)
				total -= files[i].size;
		}
		spec_log(m, 1, "cache %s pruned to %lld MB", m->cache_dir, (long long) (total / (1024 * 1024)));
	}
	free(files);
}

static void *
worker_main(void *arg)
{
	spec_mgr   *m = arg;
	sigset_t	ss;

	/* signals are for the main thread */
	sigfillset(&ss);
	pthread_sigmask(SIG_BLOCK, &ss, NULL);
	prune_cache(m);
	pthread_mutex_lock(&m->mu);
	for (;;)
	{
		spec_entry *e;

		while (!m->qhead && !m->stop)
			pthread_cond_wait(&m->cv, &m->mu);
		if (m->stop)
			break;
		e = m->qhead;
		m->qhead = e->qnext;
		if (!m->qhead)
			m->qtail = NULL;
		pthread_mutex_unlock(&m->mu);

		e->err[0] = '\0';
		build_entry(m, e);
		pthread_mutex_lock(&m->mu);
		if (__atomic_load_n(&e->state, __ATOMIC_ACQUIRE) != SPEC_READY)
		{
			m->n_failed++;
			spec_log(m, 1, "%s kernel %.8s not used: %s", kind_name(e->kind | (e->arrow ? SPEC_ARROW : 0)), e->keyhex,
					 e->err[0] ? e->err : "failed");
			__atomic_store_n(&e->state, SPEC_FAILED, __ATOMIC_RELEASE);
		}
		pthread_cond_broadcast(&m->done_cv);
	}
	pthread_mutex_unlock(&m->mu);
	return NULL;
}

spec_mgr *
spec_mgr_create(gpu_vk_ctx *vk, int trace)
{
	spec_mgr   *m;
	const char *env = getenv("PGSTROM_GPUSERV_SPECIALIZE");
	const char *s;

	m = calloc(1, sizeof(*m));
	if (!m)
		return NULL;
	m->vk = vk;
	m->trace = trace || getenv("PGSTROM_GPUSERV_SPEC_LOG") != NULL;
	m->mode = SPECM_AUTO;
	if (env && env[0])
	{
		if (!strcmp(env, "off") || !strcmp(env, "0") || !strcmp(env, "no") || !strcmp(env, "false"))
			m->mode = SPECM_OFF;
		else if (!strcmp(env, "force"))
			m->mode = SPECM_FORCE;
		else if (!strcmp(env, "check"))
			m->mode = SPECM_CHECK;
	}
	m->threshold = 2;
	if ((s = getenv("PGSTROM_GPUSERV_SPECIALIZE_TASKS")) != NULL && atoi(s) > 0)
		m->threshold = atoi(s);
	m->max_entries = 64;
	m->keep = (getenv("PGSTROM_GPUSERV_SPEC_KEEP") != NULL);
	m->src_fp = sources_fingerprint();
	if (m->mode == SPECM_OFF)
		return m;

	/* the cache directory: the option, XDG_CACHE_HOME or ~/.cache; mode 0700 */
	{
		const char *dir = getenv("PGSTROM_GPUSERV_CACHE_DIR");
		const char *xdg = getenv("XDG_CACHE_HOME");
		const char *home = getenv("HOME");
		char		cand[1024];

		cand[0] = '\0';
		if (dir && !strcmp(dir, "none"))
			;
		else if (dir && dir[0])
			snprintf(cand, sizeof(cand), "%s", dir);
		else if (xdg && xdg[0])
			snprintf(cand, sizeof(cand), "%s/pgstrom", xdg);
		else if (home && home[0])
			snprintf(cand, sizeof(cand), "%s/.cache/pgstrom", home);
		if (cand[0] && ensure_dir(cand) == 0)
			snprintf(m->cache_dir, sizeof(m->cache_dir), "%s", cand);
		else if (cand[0])
			spec_log(m, 1, "cache directory %s is not usable: kernels are compiled again when this service restarts", cand);
	}
	/* scratch space for the compiler's files */
	{
		const char *tmp = getenv("TMPDIR");
		char		cand[1024];

		snprintf(cand, sizeof(cand), "%s/pgstrom-spec.%d", (tmp && tmp[0]) ? tmp : "/tmp", (int) getpid());
		if (mkdir(cand, 0700) != 0 && errno != EEXIST)
			snprintf(cand, sizeof(cand), "/tmp");
		snprintf(m->tmp_base, sizeof(m->tmp_base), "%s", cand);
	}
	pthread_mutex_init(&m->mu, NULL);
	pthread_cond_init(&m->cv, NULL);
	pthread_cond_init(&m->done_cv, NULL);
	if (find_slangc(m) != 0)
	{
		spec_log(m, 0, "slangc not found: kernels run on the VM unless the disk cache has them");
		m->slangc[0] = '\0';
		snprintf(m->slang_version, sizeof(m->slang_version), "none");
	}
	else
		slang_version(m);
	if (m->slangc[0] || m->cache_dir[0])
	{
		if (pthread_create(&m->thr, NULL, worker_main, m) == 0)
			m->thr_started = 1;
		else
			m->mode = SPECM_OFF;
	}
	else
		m->mode = SPECM_OFF;
	spec_log(m, 0, "mode %s, compiler %s (%s), cache %s",
			 m->mode == SPECM_AUTO ? "auto" : m->mode == SPECM_FORCE ? "force" : m->mode == SPECM_CHECK ? "check" : "off",
			 m->slangc[0] ? m->slangc : "(none)", m->slang_version, m->cache_dir[0] ? m->cache_dir : "(none)");
	return m;
}

static void
entry_free(spec_mgr *m, spec_entry *e)
{
	if (__atomic_load_n(&e->state, __ATOMIC_ACQUIRE) == SPEC_READY)
		gpu_vk_pipeline_destroy(m->vk, &e->pipe);
	free(e->defs);
	free(e->body);
	free(e);
}

void
spec_mgr_destroy(spec_mgr *m)
{
	spec_entry *e, *next;

	if (!m)
		return;
	if (m->thr_started)
	{
		pthread_mutex_lock(&m->mu);
		m->stop = 1;
		if (m->child > 0)
			kill(m->child, SIGTERM);
		pthread_cond_broadcast(&m->cv);
		pthread_mutex_unlock(&m->mu);
		pthread_join(m->thr, NULL);
	}
	for (e = m->entries; e; e = next)
	{
		next = e->next;
		entry_free(m, e);
	}
	if (m->mode != SPECM_OFF && strstr(m->tmp_base, "/pgstrom-spec."))
		rmdir(m->tmp_base);
	free(m);
}

static void
entry_key(const spec_mgr *m, int kind, const char *defs, const char *body, uint64_t *p_k0, uint64_t *p_k1)
{
	uint64_t	k0 = 0xcbf29ce484222325ULL, k1 = 0x84222325cbf29ce4ULL;

	k0 = fnv64(k0, &kind, sizeof(kind));
	k1 = fnv64(k1, &kind, sizeof(kind));
	k0 = fnv64(k0, m->slang_version, strlen(m->slang_version));
	k1 = fnv64(k1, m->slang_version, strlen(m->slang_version));
	k0 = fnv64(k0, &m->src_fp, sizeof(m->src_fp));		/* the kernel sources decide what the text means */
	k1 = fnv64(k1, &m->src_fp, sizeof(m->src_fp));
	k0 = fnv64(k0, defs, strlen(defs));
	k1 = fnv64(k1, defs, strlen(defs));
	k0 = fnv64(k0, "\n", 1);
	k1 = fnv64(k1, "\n", 1);
	k0 = fnv64(k0, body, strlen(body));
	k1 = fnv64(k1, body, strlen(body));
	*p_k0 = k0;
	*p_k1 = k1;
}

/* is the kernel of this generated program in the disk cache? (the first task then does not need the VM kernel) */
static int
entry_on_disk(const spec_mgr *m, int kind, const char *defs, const char *body)
{
	uint64_t	k0, k1;
	char		p1[1300], p2[1300];
	struct stat sb;

	if (!m->cache_dir[0])
		return 0;
	entry_key(m, kind, defs, body, &k0, &k1);
	snprintf(p1, sizeof(p1), "%s/%016llx%016llx.spv", m->cache_dir, (unsigned long long) k0, (unsigned long long) k1);
	snprintf(p2, sizeof(p2), "%s/%016llx%016llx.src", m->cache_dir, (unsigned long long) k0, (unsigned long long) k1);
	return stat(p1, &sb) == 0 && stat(p2, &sb) == 0;
}

/* the cache entry of a generated program; takes ownership of defs and body */
static spec_entry *
entry_find_or_add(spec_mgr *m, const spec_shape *sh, char *defs, char *body)
{
	spec_entry *e;
	uint64_t	k0, k1;
	int			kind = sh->kind;

	entry_key(m, kind, defs, body, &k0, &k1);
	for (e = m->entries; e; e = e->next)
		if (e->key[0] == k0 && e->key[1] == k1 && e->kind == kind &&
			!strcmp(e->defs, defs) && !strcmp(e->body, body))
		{
			free(defs);
			free(body);
			return e;
		}
	/* evict an entry nobody uses when the table is full */
	if (m->nentries >= m->max_entries)
	{
		spec_entry **pp, **victim = NULL;

		for (pp = &m->entries; *pp; pp = &(*pp)->next)
		{
			spec_entry *c = *pp;

			if (c->refs == 0 && __atomic_load_n(&c->state, __ATOMIC_ACQUIRE) != SPEC_Q &&
				(!victim || c->last_used < (*victim)->last_used))
				victim = pp;
		}
		if (victim)
		{
			spec_entry *v = *victim;

			*victim = v->next;
			m->nentries--;
			entry_free(m, v);
		}
	}
	e = calloc(1, sizeof(*e));
	if (!e)
	{
		free(defs);
		free(body);
		return NULL;
	}
	e->key[0] = k0;
	e->key[1] = k1;
	snprintf(e->keyhex, sizeof(e->keyhex), "%016llx%016llx", (unsigned long long) k0, (unsigned long long) k1);
	e->kind = kind;
	e->arrow = sh->arrow;
	e->agg_max = sh->agg_max;
	e->ga_c = sh->ga_c;
	e->shape = *sh;
	e->defs = defs;
	e->body = body;
	e->state = SPEC_Q;
	e->next = m->entries;
	m->entries = e;
	m->nentries++;
	pthread_mutex_lock(&m->mu);
	if (m->qtail)
		m->qtail->qnext = e;
	else
		m->qhead = e;
	m->qtail = e;
	pthread_cond_signal(&m->cv);
	pthread_mutex_unlock(&m->mu);
	spec_log(m, 0, "%s kernel %.8s queued (%zu bytes of generated code)", kind_name(kind | (sh->arrow ? SPEC_ARROW : 0)), e->keyhex, strlen(body));
	return e;
}

/*
 * The shape of a specialized kernel. The aggregation kernels without GROUP BY keep their output layout whatever
 * AGG_MAX is, so it is the exact number of aggregates (less shared memory). The GROUP BY kernels' layout depends
 * on the aggregate count and the table capacity: the exact aggregate count, and a table sized for the groups the
 * session has seen so far (a small table leaves room for more workgroups on a core; the VM variants have 128 or
 * 64 slots whatever the query). In check mode the shape stays the VM variant's, so the outputs can be compared.
 */
static int
shape_slot_bytes(int na)
{
	return 4 + 4 + 8 * 2 + 20 * na;		/* owner, null mask, two keys, count and four limbs per aggregate */
}

static int
shape_stage_bytes(int na)
{
	return 64 * (12 + 16 + 12 * na);	/* the staging arrays of the 64 threads of a workgroup */
}

/*
 * A direct-mapped table for the keys the session has seen (spec_shape.dm): one key or two, integers in a window
 * (a little wider than what has been seen, so that groups that show up later still fit); the table has one
 * slot per cell of the windows, plus the NULL cell of each key. Not worth it for sparse keys (a hash table with
 * few slots is smaller) or when the cells do not fit shared memory. PGSTROM_GPUSERV_SPEC_DM=off disables it;
 * =force:<base>:<width>[,<base>:<width>] fixes the windows (tests: also without keys seen).
 */
static int
pick_dm(spec_mgr *m, const scan_prog *prog, const spec_keystats *ks, const spec_shape *in, spec_shape *out)
{
	const char *env = getenv("PGSTROM_GPUSERV_SPEC_DM");
	int			na = prog->nagg > 0 ? prog->nagg : 1;
	int			nkeys = prog->nkeys;
	int			limit = (int) ((m->vk->info.max_shared_memory ? m->vk->info.max_shared_memory : 32768) * 3 / 4);
	int			cmax = limit / (4 + 20 * na);
	long long	cells = 1;
	int64_t		base[2] = {0, 0};
	int			width[2] = {1, 1};

	(void) in;
	if (nkeys < 1 || nkeys > 2 || (env && !strcmp(env, "off")))
		return 0;
	if (env && !strncmp(env, "force:", 6))
	{
		const char *p = env + 6;

		for (int k = 0; k < nkeys; k++)
		{
			char	   *end;

			base[k] = strtoll(p, &end, 10);
			if (*end != ':')
				return 0;
			width[k] = (int) strtol(end + 1, &end, 10);
			p = (*end == ',') ? end + 1 : end;
		}
	}
	else
	{
		if (!ks || ks->ngroups == 0 || ks->nkeys != nkeys)
			return 0;
		for (int k = 0; k < nkeys; k++)
		{
			int64_t		lo = ks->kmin[k], hi = ks->kmax[k];
			int64_t		span, margin;

			if (!ks->has_value[k])
			{
				base[k] = 0;
				width[k] = 1;
				continue;
			}
			if (lo < -((int64_t) 1 << 60) || hi > ((int64_t) 1 << 60))
				return 0;
			span = hi - lo + 1;
			if (span > 4096)
				return 0;
			margin = span / 16 + 1;
			base[k] = lo - margin;
			width[k] = (int) (span + 2 * margin);
		}
	}
	for (int k = 0; k < nkeys; k++)
	{
		if (width[k] < 1 || width[k] > 4096)
			return 0;
		cells *= (long long) width[k] + 1;
		if (cells > 100000)
			return 0;
	}
	if (cells > cmax || cells < 2)
		return 0;
	if (!(env && !strncmp(env, "force:", 6)) && cells > 4LL * ks->ngroups + 16)
		return 0;						/* sparse: a small hash table is better */
	*out = *in;
	out->agg_max = na;
	out->ga_c = (int) cells;
	out->dm = 1;
	out->dm_nkeys = nkeys;
	for (int k = 0; k < 2; k++)
	{
		out->dm_base[k] = base[k];
		out->dm_w[k] = width[k];
	}
	return 1;
}

static void
pick_shape(spec_mgr *m, const scan_prog *prog, int kind, const spec_shape *in, const spec_keystats *ks, spec_shape *out)
{
	int			na = prog->nagg > 0 ? prog->nagg : 1;
	uint32_t	groups_seen = ks ? ks->ngroups : 0;

	*out = *in;
	out->kind = kind;
	out->dm = 0;
	out->stats = (ks && ks->ngroups > 0);
	if (kind == SPEC_AGG || kind == SPEC_JAGG)
	{
		out->agg_max = na;
		return;
	}
	if (kind != SPEC_GAGG && kind != SPEC_JGAGG)
		return;
	if (m->mode == SPECM_CHECK)
		return;
	if (pick_dm(m, prog, ks, in, out))
	{
		out->kind = kind;
		out->stats = (ks && ks->ngroups > 0);
		return;
	}
	{
		const char *env = getenv("PGSTROM_GPUSERV_SPEC_GA_C");
		int			limit = (int) ((m->vk->info.max_shared_memory ? m->vk->info.max_shared_memory : 32768) * 3 / 4);
		int			cmax = (limit - shape_stage_bytes(na)) / shape_slot_bytes(na);
		int			c;

		out->agg_max = na;
		if (cmax < 16)
		{
			/* the aggregates do not leave room for a useful table: the VM's layout */
			*out = *in;
			out->kind = kind;
			out->stats = (ks && ks->ngroups > 0);
			return;
		}
		if (env && atoi(env) > 0)
			c = atoi(env);
		else if (groups_seen == 0)
			c = in->ga_c;
		else
		{
			c = 16;
			while (c < (int) (2 * groups_seen) && c < 4096)
				c *= 2;
		}
		out->ga_c = c > cmax ? cmax : c;
	}
}

/* keys the session has seen outside the window a direct-mapped kernel was made for */
static int
spec_window_exceeded(const spec_shape *sh, const spec_keystats *ks)
{
	for (int k = 0; k < sh->dm_nkeys && k < 2; k++)
		if (ks->has_value[k] && (ks->kmin[k] < sh->dm_base[k] || ks->kmax[k] >= sh->dm_base[k] + sh->dm_w[k]))
			return 1;
	return 0;
}

gpu_vk_pipeline *
spec_pipe(scan_engine *eng, scan_prog *prog, int kind, int agg_max, int ga_c, const spec_keystats *ks,
		  spec_shape *out_shape, gpu_vk_pipeline *vm_pipe)
{
	spec_mgr   *m = eng->spec;
	spec_entry *e;
	spec_shape	in, sh;
	int			want_wait = 0;
	int			arrow = ((kind & SPEC_ARROW) != 0);

	kind &= ~SPEC_ARROW;
	memset(&in, 0, sizeof(in));
	memset(&sh, 0, sizeof(sh));
	in.arrow = arrow;
	in.kind = kind;
	in.agg_max = agg_max;
	in.ga_c = ga_c;
	if (out_shape)
		*out_shape = in;
	if (!m || m->mode == SPECM_OFF || !m->thr_started || prog->spec_state == 2)
		return vm_pipe;
	if (!prog->spec)
	{
		char	   *defs = NULL, *body = NULL, err[256];

		prog->spec_tasks++;
		if (m->mode == SPECM_AUTO && prog->spec_tasks < m->threshold && prog->spec_state == 1)
			return vm_pipe;				/* looked at in the first task: not on the disk */
		pick_shape(m, prog, kind, &in, ks, &sh);
		if (spec_generate(&sh, prog, &defs, &body, err, sizeof(err)) != 0)
		{
			spec_log(m, 0, "%s program stays on the VM: %s", kind_name(kind | (arrow ? SPEC_ARROW : 0)), err);
			prog->spec_state = 2;
			return vm_pipe;
		}
		if (m->mode == SPECM_AUTO && prog->spec_tasks < m->threshold)
		{
			/*
			 * A kernel the disk cache already has is as good as a compiled one: use it from the first task (the
			 * task waits for the pipeline, about 50 ms, instead of the VM kernel's, up to seconds the first time);
			 * otherwise the compilation starts with the second task.
			 */
			if (!entry_on_disk(m, kind, defs, body))
			{
				free(defs);
				free(body);
				prog->spec_state = 1;
				return vm_pipe;
			}
			want_wait = 1;
		}
		e = entry_find_or_add(m, &sh, defs, body);
		if (!e)
		{
			prog->spec_state = 2;
			return vm_pipe;
		}
		e->refs++;
		prog->spec = e;
		prog->spec_owner = m;
	}
	else if (kind == SPEC_GAGG || kind == SPEC_JGAGG)
	{
		/*
		 * A GROUP BY kernel sized for the groups seen at the time may be too small now: a bigger one is
		 * compiled while the current one (rows that find no slot are aggregated on the CPU) keeps serving.
		 */
		spec_entry *cur = prog->spec, *nx = prog->spec_next;

		if (nx)
		{
			int			st = __atomic_load_n(&nx->state, __ATOMIC_ACQUIRE);

			if (st == SPEC_READY)
			{
				cur->refs--;
				prog->spec = nx;
				prog->spec_next = NULL;
			}
			else if (st == SPEC_FAILED)
			{
				nx->refs--;
				prog->spec_next = NULL;
				prog->spec_grow_failed = 1;
			}
		}
		else if (!prog->spec_grow_failed && m->mode != SPECM_CHECK && ks &&
				 __atomic_load_n(&cur->state, __ATOMIC_ACQUIRE) == SPEC_READY &&
				 ((cur->shape.dm ? spec_window_exceeded(&cur->shape, ks)
				   : ks->ngroups * 3 / 2 > (uint32_t) cur->ga_c) ||
				  (!cur->shape.stats && ks->ngroups > 0)))
		{
			/*
			 * Too small a table or a window the groups left, or a kernel that was chosen before the session had
			 * any group (a kernel from the disk cache at the first task): look again with what has been seen.
			 */
			char	   *defs = NULL, *body = NULL, err[256];

			pick_shape(m, prog, kind, &in, ks, &sh);
			if ((sh.dm != cur->shape.dm || sh.ga_c != cur->ga_c ||
				 (sh.dm && memcmp(sh.dm_base, cur->shape.dm_base, sizeof(sh.dm_base)) != 0)) &&
				prog->spec_regens++ < 4 &&
				spec_generate(&sh, prog, &defs, &body, err, sizeof(err)) == 0)
			{
				spec_entry *ne = entry_find_or_add(m, &sh, defs, body);

				if (ne && ne != cur)
				{
					ne->refs++;
					prog->spec_next = ne;
				}
			}
			else
				prog->spec_grow_failed = 1;
		}
	}
	e = prog->spec;
	if (__atomic_load_n(&e->state, __ATOMIC_ACQUIRE) == SPEC_Q && (m->mode >= SPECM_FORCE || want_wait))
	{
		/* force / check: the first task waits for the compilation (tests, measurements); a cached kernel, for its pipeline */
		pthread_mutex_lock(&m->mu);
		while (__atomic_load_n(&e->state, __ATOMIC_ACQUIRE) == SPEC_Q)
			pthread_cond_wait(&m->done_cv, &m->mu);
		pthread_mutex_unlock(&m->mu);
	}
	switch (__atomic_load_n(&e->state, __ATOMIC_ACQUIRE))
	{
		case SPEC_READY:
			e->last_used = ++m->clock;
			if (!e->announced)
			{
				e->announced = 1;
				spec_log(m, 0, "%s kernel %.8s in use (%d aggregates, %d slots%s)", kind_name(e->kind | (e->arrow ? SPEC_ARROW : 0)), e->keyhex,
						 e->agg_max, e->ga_c, e->shape.dm ? ", direct-mapped" : "");
			}
			if (out_shape)
			{
				out_shape->agg_max = e->agg_max;
				out_shape->ga_c = e->ga_c;
			}
			return &e->pipe;
		case SPEC_FAILED:
			prog->spec_state = 2;
			break;
	}
	return vm_pipe;
}

void
spec_release(scan_prog *prog)
{
	if (prog->spec)
		((spec_entry *) prog->spec)->refs--;
	if (prog->spec_next)
		((spec_entry *) prog->spec_next)->refs--;
	prog->spec = NULL;
	prog->spec_next = NULL;
	prog->spec_owner = NULL;
}

int
spec_check_enabled(scan_engine *eng)
{
	return eng->spec && eng->spec->mode == SPECM_CHECK;
}
