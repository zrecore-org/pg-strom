/*
 * heapspike.c - GPU scan over REAL PostgreSQL heap pages (the "real scan path"
 * measurement): the relation file of table t(a int4, b int8, f float4, d int4)
 * from the throwaway cluster is copied into a GPU-visible buffer and scanned by
 *   - a hand-written kernel (heap_fixed)
 *   - the opcode-switch VM in heap mode (heap_vm)
 *   - a kernel generated from the expression tree (spec)
 * Each result is checked against a SQL oracle: match count and the sum of
 * blkno * 512 + offnum over all matches (derived from ctid on the SQL side).
 *
 * usage: heapspike <shaderdir> <relation-file> <E1 count> <E1 idsum> <E2 count> <E2 idsum>
 * env:   SLANGC (path to slangc), HEAP_GROUPS (workgroups, default 16384),
 *        HEAP_WG (workgroup size the shaders were built with, default 128)
 * Throwaway code. See Makefile: make run-heap
 */
#include "vkutil.h"
#include "opcodes.inc"
#include "exprlib.h"
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define PAGE_SIZE_BYTES 8192u

static VkBuf pagesb, outb, progb;
static struct { uint32_t npages, ngroups; } pc;
static double g_first_ms;					/* time of the first dispatch of the last run() */

static void residency(const char *when, void *host, size_t len)
{
	size_t nos = len / 16384, resident = 0;
	char *vec = malloc(nos);

	if (mincore(host, len, vec) == 0)
	{
		for (size_t i = 0; i < nos; i++)
			resident += vec[i] & 1;
		printf("OS page cache residency %s: %zu of %zu pages (%.0f%%)\n", when, resident, nos, 100.0 * resident / nos);
	}
	free(vec);
}

static double run(VkPipeline pl, int iters)
{
	uint32_t *ob = outb.map;
	double best = 1e9;

	for (int i = 0; i < iters; i++)
	{
		ob[0] = ob[1] = ob[2] = 0;
		double ms = vk_dispatch(pl, &pc, sizeof pc, pc.ngroups);
		if (i == 0)
			g_first_ms = ms;
		if (ms < best)
			best = ms;
	}
	return best;
}

/* compare the GPU result in outb with the SQL oracle */
static int verify(const char *tag, const char *what, double ms, uint64_t o_count, uint64_t o_sum)
{
	uint32_t *ob = outb.map;
	uint64_t sum = 0;
	int ok;

	for (uint32_t i = 0; i < ob[0]; i++)
		sum += ob[3 + i];
	ok = ob[1] == 0 && ob[0] == o_count && sum == o_sum;
	printf("[%s] %-5s %-22s matches gpu=%u sql=%llu  idsum %s  err=%u  best=%.1f ms\n",
		   ok ? "PASS" : "FAIL", tag, what, ob[0], (unsigned long long)o_count,
		   sum == o_sum ? "equal" : "DIFFERENT", ob[1], ms);
	return ok;
}

int main(int argc, char **argv)
{
	if (argc < 7)
	{
		fprintf(stderr, "usage: %s <shaderdir> <relfile> <E1 count> <E1 idsum> <E2 count> <E2 idsum>\n", argv[0]);
		return 2;
	}
	const char *dir = argv[1], *relfile = argv[2];
	uint64_t oc[2] = {strtoull(argv[3], 0, 10), strtoull(argv[5], 0, 10)};
	uint64_t os[2] = {strtoull(argv[4], 0, 10), strtoull(argv[6], 0, 10)};
	const char *wg_env = getenv("HEAP_WG");
	int failures = 0;
	char path[512];

	/* ---- where do the pages the GPU reads come from? ----
	 *   copy     control: read into ordinary memory, memcpy into a GPU buffer (the old path)
	 *   anon     anonymous mmap, filled by pread, imported into Vulkan (zero copy)
	 *   shm      POSIX shared memory (shm_open) filled by a separate "load" process
	 *            (HEAP_SHM_ROLE=load), then attached by name and imported (role scan)
	 *   file     the relation file itself, MAP_SHARED: the OS page cache, imported
	 *   filepriv same, MAP_PRIVATE (copy-on-write)
	 * For file/filepriv pass a COPY of the relation file: the coherence test writes to it. */
	const char *src = getenv("HEAP_SRC") ? getenv("HEAP_SRC") : "copy";
	const char *role = getenv("HEAP_SHM_ROLE") ? getenv("HEAP_SHM_ROLE") : "scan";
	const char *shmname = "/pgscan_zc";
	int fd = open(relfile, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st) < 0) { perror(relfile); return 2; }
	if (st.st_size % PAGE_SIZE_BYTES)
	{
		fprintf(stderr, "file size is not a multiple of the page size\n");
		return 2;
	}
	uint32_t npages = st.st_size / PAGE_SIZE_BYTES;
	size_t alloc_len = (st.st_size + 16383) & ~(size_t)16383;	/* import needs 16 KB multiples */
	uint8_t *host = NULL;
	double t0 = vk_now_ms(), t_prep, t_import = 0;

	if (!strcmp(src, "copy"))
	{
		host = aligned_alloc(16384, alloc_len);
		if (host && pread(fd, host, st.st_size, 0) != st.st_size) { perror("pread"); return 2; }
	}
	else if (!strcmp(src, "anon"))
	{
		host = mmap(NULL, alloc_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
		if (host != MAP_FAILED && pread(fd, host, st.st_size, 0) != st.st_size) { perror("pread"); return 2; }
	}
	else if (!strcmp(src, "shm"))
	{
		if (!strcmp(role, "unlink"))
		{
			printf("shm_unlink(%s): %s\n", shmname, shm_unlink(shmname) == 0 ? "ok" : strerror(errno));
			return 0;
		}
		int sfd = shm_open(shmname, !strcmp(role, "load") ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR, 0600);
		if (sfd < 0) { perror("shm_open"); return 2; }
		if (!strcmp(role, "load") && ftruncate(sfd, alloc_len) < 0) { perror("ftruncate"); return 2; }
		host = mmap(NULL, alloc_len, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
		if (host != MAP_FAILED && !strcmp(role, "load"))
		{
			if (pread(fd, host, st.st_size, 0) != st.st_size) { perror("pread"); return 2; }
			printf("loaded %.0f MB into %s in %.0f ms (this process now exits; the region stays)\n",
				   st.st_size / 1e6, shmname, vk_now_ms() - t0);
			return 0;
		}
	}
	else if (!strcmp(src, "file") || !strcmp(src, "filepriv"))
	{
		close(fd);
		fd = open(relfile, O_RDWR);
		if (fd < 0) { perror(relfile); return 2; }
		host = mmap(NULL, alloc_len, PROT_READ | PROT_WRITE,
					!strcmp(src, "file") ? MAP_SHARED : MAP_PRIVATE, fd, 0);
	}
	else
	{
		fprintf(stderr, "unknown HEAP_SRC %s\n", src);
		return 2;
	}
	if (!host || host == MAP_FAILED) { perror("host memory"); return 2; }
	t_prep = vk_now_ms() - t0;
	close(fd);

	/* ---- residency of the file pages, if the source is file-backed ---- */
	if (!strncmp(src, "file", 4))
	{
		size_t nos = alloc_len / 16384, resident = 0;
		char *vec = malloc(nos);
		if (mincore(host, alloc_len, vec) == 0)
		{
			for (size_t i = 0; i < nos; i++)
				resident += vec[i] & 1;
			printf("OS page cache residency of the mapping: %zu of %zu pages (%.0f%%) before any access\n",
				   resident, nos, 100.0 * resident / nos);
		}
		free(vec);
	}

	/* ---- sanity: page headers and tuple count (reads every page through the CPU mapping;
	 *      HEAP_SKIP_SANITY=1 leaves the pages untouched so the GPU is the first reader) ---- */
	uint64_t ntup = 0, nbad = 0;
	if (getenv("HEAP_SKIP_SANITY"))
		printf("source=%s: %u pages, CPU sanity pass SKIPPED, host memory ready in %.0f ms\n",
			   src, npages, t_prep);
	else
	{
		for (uint32_t p = 0; p < npages; p++)
		{
			const uint8_t *pg = host + (size_t)p * PAGE_SIZE_BYTES;
			uint16_t lower, upper, ver;
			memcpy(&lower, pg + 12, 2);
			memcpy(&upper, pg + 14, 2);
			memcpy(&ver, pg + 18, 2);
			if (ver != (PAGE_SIZE_BYTES | 4) || lower < 24 || upper > PAGE_SIZE_BYTES || lower > upper)
				nbad++;
			else
				ntup += (lower - 24) / 4;
		}
		printf("source=%s: %u pages (%.0f MB), %llu line pointers, %llu pages with an unexpected header; host memory ready in %.0f ms\n",
			   src, npages, st.st_size / 1e6, (unsigned long long)ntup, (unsigned long long)nbad, t_prep);
		if (nbad)
			return 2;
	}

	/* ---- GPU buffers ---- */
	vk_init();
	printf("VK_EXT_external_memory_host: %s (minImportedHostPointerAlignment %llu)\n",
		   vk_host_import_ok ? "available" : "NOT available", (unsigned long long)vk_host_align);
	double t_copy = 0;
	if (!strcmp(src, "copy"))
	{
		pagesb = vk_buf(st.st_size);
		t_copy = 1e9;
		for (int i = 0; i < 5; i++)
		{
			t0 = vk_now_ms();
			memcpy(pagesb.map, host, st.st_size);
			double t = vk_now_ms() - t0;
			if (t < t_copy)
				t_copy = t;
		}
		printf("page transfer: memcpy into the GPU buffer %.0f ms best of 5 (%.1f GB/s)\n",
			   t_copy, st.st_size / 1e9 / (t_copy / 1e3));
	}
	else
	{
		const char *why = "";
		t0 = vk_now_ms();
		VkResult rc = vk_buf_import(host, alloc_len, &pagesb, &why);
		t_import = vk_now_ms() - t0;
		if (rc != VK_SUCCESS)
		{
			printf("[FAIL] import of %s memory rejected: %s (VkResult %d)\n", src, why, rc);
			return 1;
		}
		printf("zero-copy import of %.0f MB took %.1f ms; no page copy\n", st.st_size / 1e6, t_import);
	}
	outb = vk_buf(sizeof(uint32_t) * ((size_t)npages * 291 + 16));	/* MaxHeapTuplesPerPage = 291 */
	progb = vk_buf(65536);
	VkBuf bufs[3] = {pagesb, outb, progb};
	vk_bind(bufs, 3, sizeof pc, 0);

	pc.npages = npages;
	pc.ngroups = getenv("HEAP_GROUPS") ? (uint32_t)atoi(getenv("HEAP_GROUPS")) : 16384;
	if (pc.ngroups > npages)
		pc.ngroups = npages;
	printf("dispatch: %u workgroups x %s threads, %.1f pages per workgroup\n",
		   pc.ngroups, wg_env ? wg_env : "128", (double)npages / pc.ngroups);

	/* ---- hand-written kernel (E1 only) ---- */
	double hand_ms;
	VkPipeline hand_pl;
	{
		snprintf(path, sizeof path, "%s/heap_fixed.spv", dir);
		VkPipeline pl = hand_pl = vk_pipeline(path);
		if (!pl) { printf("[FAIL] Metal rejected heap_fixed\n"); return 1; }
		hand_ms = run(pl, 5);
		failures += !verify("HAND", "E1 hand-written", hand_ms, oc[0], os[0]);
		printf("      first dispatch of the hand-written kernel: %.1f ms (includes any page-in)\n", g_first_ms);
		if (!strncmp(src, "file", 4))
			residency("after the first GPU scan", host, alloc_len);
	}

	/* ---- VM and generated kernels for E1 and E2 ---- */
	static const struct { const char *name; Node *(*build)(void); } exprs[2] = {
		{"E1 arithmetic+compare", build1},
		{"E2 NULL tests, OR, NOT", build2},
	};
	double vm_ms[2], spec_ms[2], spec_compile[2];
	gen_heap = 1;
	if (wg_env)
	{
		static char fl[64];
		snprintf(fl, sizeof fl, "-DHEAP_WG=%s", wg_env);
		gen_flags = fl;
	}
	mkdir("gen", 0755);
	for (int e = 0; e < 2; e++)
	{
		/* VM */
		snprintf(path, sizeof path, "%s/heap_vm.spv", dir);
		VkPipeline vm = vk_pipeline(path);
		if (!vm) { printf("[FAIL] Metal rejected heap_vm\n"); return 1; }
		npool = 0;
		linearize(exprs[e].build(), progb.map);
		vm_ms[e] = run(vm, 5);
		failures += !verify("VM", exprs[e].name, vm_ms[e], oc[e], os[e]);

		/* generated kernel */
		double t_slangc, t_pipe;
		int cached;
		npool = 0;
		gen_kernel(exprs[e].build(), exprs[e].name, getenv("SPEC_COLD") ? (unsigned)time(NULL) + e : 0);
		VkPipeline sp = specialize(exprs[e].name, &t_slangc, &t_pipe, &cached);
		if (!sp) { printf("[FAIL] Metal rejected the generated kernel\n"); return 1; }
		spec_ms[e] = run(sp, 5);
		spec_compile[e] = t_slangc + t_pipe;
		failures += !verify("SPEC", exprs[e].name, spec_ms[e], oc[e], os[e]);
		printf("      generated kernel compile: slangc %.0f ms%s + pipeline %.0f ms\n",
			   t_slangc, cached ? " (cached)" : "", t_pipe);
	}

	/* ---- summary ---- */
	printf("\n== summary, source=%s, %u pages / %llu tuples (kernel only; %s) ==\n",
		   src, npages, (unsigned long long)ntup,
		   t_copy > 0 ? "add the per-scan page copy printed above" : "no page copy needed");
	printf("E1:  hand-written %6.1f ms   specialized %6.1f ms   VM %6.1f ms\n", hand_ms, spec_ms[0], vm_ms[0]);
	printf("E2:                         specialized %6.1f ms   VM %6.1f ms\n", spec_ms[1], vm_ms[1]);
	/* ---- coherence: write through the CPU mapping, rescan, no copy in between ---- */
	if (getenv("HEAP_COHERENCE"))
	{
		enum { K = 1000 };
		uint32_t *ob = outb.map, ids[K], before;
		uint8_t *wr = host;			/* default: write through the mapping we imported */
		int two_maps = 0;

		run(hand_pl, 1);
		before = ob[0];
		for (int i = 0; i < K; i++)
			ids[i] = ob[3 + i];
		if (!strcmp(src, "shm"))	/* write through a SECOND mapping of the same shared memory */
		{
			int sfd = shm_open(shmname, O_RDWR, 0600);
			wr = mmap(NULL, alloc_len, PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
			if (wr == MAP_FAILED) { perror("second mapping"); return 2; }
			two_maps = 1;
		}
		for (int i = 0; i < K; i++)		/* set column a of K matching tuples to 0 so a > 1000 fails */
		{
			uint32_t page = ids[i] / 512, lp = ids[i] % 512 - 1, w;
			uint8_t *pg = wr + (size_t)page * PAGE_SIZE_BYTES;
			memcpy(&w, pg + 24 + 4 * lp, 4);
			uint8_t *tup = pg + (w & 0x7FFF);
			int32_t zero = 0;
			memcpy(tup + tup[22], &zero, 4);	/* a is the first column; matches have a non-NULL */
		}
		double ms = run(hand_pl, 3);
		int ok = ob[1] == 0 && ob[0] == before - K;
		printf("[%s] coherence (%s%s): GPU matches before=%u, after zeroing a in %d tuples=%u (expected %u) -> %s\n",
			   ok ? "SEES" : "STALE", src, two_maps ? ", written through a 2nd mapping" : "",
			   before, K, ob[0], before - K,
			   ok ? "the GPU read the CPU's writes in place" : "the GPU did NOT see the writes");
		(void)ms;
	}

	printf("%s\n", failures ? "HEAP SPIKE: FAILURES" : "HEAP SPIKE: ALL PASSED");
	return failures ? 1 : 0;
}
