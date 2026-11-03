/*
 * zcscan.c - scan LIVE shared_buffers on the GPU, in place, while writers run.
 *
 *   SELECT zc_scan('u', 'inplace', 10, '/path/to/shaders');
 *
 * modes:
 *   inplace  pin every buffer of the table, import the whole shared_buffers array into
 *            Vulkan (zero copy) and scan the pinned buffers in place. No content lock.
 *   nopin    like inplace but drop the pins before scanning (stress mode: lets the
 *            writers prune and defragment pages under the GPU); buffer tags are
 *            re-checked after each scan to detect a remapped buffer.
 *   locked   for each buffer: shared content lock, memcpy the page into a GPU-visible
 *            staging buffer, unlock; then scan the copy.
 *
 * Throwaway spike code: Vulkan failures call exit() through vkutil.h.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "access/relation.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/bufpage.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "vkutil.h"

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1(zc_scan);

#define ZC_STATS_WORDS	64
#define ZC_SAMPLES		8

static bool			zc_ready;
static VkBuf		sb_buf, stats_buf, plist_in_buf, plist_st_buf, stage_buf;
static VkPipeline	zc_pl;
static VkDescriptorSet dset_in, dset_st;
static uintptr_t	sb_base;
static size_t		sb_len;
static uint32		zc_cap;			/* max list entries (blocks) */
static bool			zc_stage_ready;

/*
 * macOS: Metal's shader compiler is a separate service reached over XPC. A process forked
 * WITHOUT exec from a parent that never used Metal (every PostgreSQL backend) cannot connect:
 * "Unable to reach MTLCompilerService ... Connection init failed at lookup with error 3".
 * Initialising Metal once in the postmaster (shared_preload_libraries, ZC_SHADERDIR set) lets
 * the backends it forks compile shaders. Reproduced and confirmed outside PostgreSQL by
 * ../forktest.c.
 */
void		_PG_init(void);

void
_PG_init(void)
{
	const char *dir = getenv("ZC_SHADERDIR");
	char		path[512];
	VkBuf		b[3];

	if (!process_shared_preload_libraries_in_progress || !dir)
		return;
	vk_init();
	b[0] = vk_buf(4096);
	b[1] = vk_buf(4096);
	b[2] = vk_buf(4096);
	vk_bind(b, 3, 8, 0);
	snprintf(path, sizeof path, "%s/heap_live.spv", dir);
	if (!vk_pipeline(path))
		elog(WARNING, "zcscan: warm-up compile in the postmaster failed");
	else
		elog(LOG, "zcscan: Metal initialised in the postmaster (compile warm-up ok)");
}

static void
zc_init(const char *shaderdir, BlockNumber nblocks)
{
	const char *why = "";
	char		path[512];
	uintptr_t	bb = (uintptr_t) BufferBlocks;
	uintptr_t	end = bb + (size_t) NBuffers * BLCKSZ;

	if (zc_ready)
	{
		if (nblocks > zc_cap)
			elog(ERROR, "relation grew beyond the capacity allocated at init (%u > %u); reconnect", nblocks, zc_cap);
		return;
	}
	vk_init();
	if (!vk_host_import_ok)
		elog(ERROR, "VK_EXT_external_memory_host not available");

	/* import the shared_buffers array: start/size must be multiples of 16 KB */
	sb_base = (bb + 16383) & ~(uintptr_t) 16383;
	sb_len = ((end - sb_base) / 16384) * 16384;
	elog(NOTICE, "BufferBlocks=%p NBuffers=%d: importing %zu MB from %p (skipping %zu leading bytes)",
		 (void *) bb, NBuffers, sb_len >> 20, (void *) sb_base, (size_t) (sb_base - bb));
	if (vk_buf_import((void *) sb_base, sb_len, &sb_buf, &why) != VK_SUCCESS)
		elog(ERROR, "import of shared_buffers failed: %s", why);

	zc_cap = nblocks + nblocks / 4 + 64;
	stats_buf = vk_buf(ZC_STATS_WORDS * 4 * 4);
	plist_in_buf = vk_buf((size_t) zc_cap * 8);
	plist_st_buf = vk_buf((size_t) zc_cap * 8);
	{
		VkBuf b[3] = {sb_buf, stats_buf, plist_in_buf};
		vk_bind(b, 3, 8, 0);
		dset_in = vk_dset;
	}
	snprintf(path, sizeof path, "%s/heap_live.spv", shaderdir);
	zc_pl = vk_pipeline(path);
	if (!zc_pl)
		elog(ERROR, "the Metal compiler rejected heap_live.spv");
	zc_ready = true;
}

static void
zc_init_stage(void)
{
	if (zc_stage_ready)
		return;
	stage_buf = vk_buf((size_t) zc_cap * BLCKSZ);
	{
		VkBuf b[3] = {stage_buf, stats_buf, plist_st_buf};
		vk_bind(b, 3, 8, 0);		/* new layout, compatible with the pipeline's */
		dset_st = vk_dset;
	}
	zc_stage_ready = true;
}

Datum
zc_scan(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	char	   *mode = text_to_cstring(PG_GETARG_TEXT_PP(1));
	double		seconds = PG_GETARG_FLOAT8(2);
	char	   *dir = text_to_cstring(PG_GETARG_TEXT_PP(3));
	int			kind;
	Relation	rel = relation_open(relid, AccessShareLock);
	BlockNumber nblocks = RelationGetNumberOfBlocks(rel);
	Buffer	   *bufs;
	uint32	   *plist_in;
	uint32	   *plist_st;
	uint32		nin = 0, nexcluded = 0;
	BlockNumber	blk;
	uint32	   *st = NULL;

	/* accumulators */
	uint64		scans = 0, scans_anom = 0, remapped_scans = 0, remapped_bufs = 0;
	uint64		viol = 0, badlp = 0, badhdr = 0, badpg = 0, stale_sum = 0;
	uint32		last_tuples = 0, last_match = 0, last_idsum = 0, last_pages = 0;
	double		kern_sum = 0, copy_sum = 0, kern_min = 1e18, kern_max = 0, copy_max = 0;
	double		t_end;
	StringInfoData samples;

	if (!strcmp(mode, "inplace"))
		kind = 0;
	else if (!strcmp(mode, "nopin"))
		kind = 1;
	else if (!strcmp(mode, "locked"))
		kind = 2;
	else
		elog(ERROR, "unknown mode \"%s\" (inplace | nopin | locked)", mode);

	zc_init(dir, nblocks);
	if (kind == 2)
		zc_init_stage();
	initStringInfo(&samples);

	/* pin every block (also makes sure it is resident) and build the page lists */
	bufs = palloc(sizeof(Buffer) * nblocks);
	plist_in = (uint32 *) plist_in_buf.map;
	plist_st = (uint32 *) plist_st_buf.map;
	for (blk = 0; blk < nblocks; blk++)
	{
		Buffer		buf = ReadBufferExtended(rel, MAIN_FORKNUM, blk, RBM_NORMAL, NULL);
		uintptr_t	addr = (uintptr_t) BufferGetBlock(buf);

		bufs[blk] = buf;
		if (addr >= sb_base && addr + BLCKSZ <= sb_base + sb_len)
		{
			plist_in[2 * nin] = (uint32) ((addr - sb_base) / 4);	/* word offset */
			plist_in[2 * nin + 1] = blk;
			nin++;
		}
		else
			nexcluded++;		/* in the 16 KB-alignment slack at the start of the array */
		plist_st[2 * blk] = blk * (BLCKSZ / 4);
		plist_st[2 * blk + 1] = blk;
	}
	if (kind == 1)				/* nopin: release the pins, keep the buffer ids */
		for (blk = 0; blk < nblocks; blk++)
			ReleaseBuffer(bufs[blk]);

	t_end = vk_now_ms() + seconds * 1000.0;
	do
	{
		uint32		nlist = (kind == 2) ? nblocks : nin;
		uint32		groups = Min(nlist, 16384);
		struct { uint32 nlist, groups; } pc = {nlist, groups};
		double		t_copy = 0, t_kern;
		bool		anom;

		CHECK_FOR_INTERRUPTS();
		if (kind == 2)			/* shared content lock + memcpy: a consistent copy of each page */
		{
			double		t0 = vk_now_ms();

			for (blk = 0; blk < nblocks; blk++)
			{
				LockBuffer(bufs[blk], BUFFER_LOCK_SHARE);
				memcpy((char *) stage_buf.map + (size_t) blk * BLCKSZ, BufferGetPage(bufs[blk]), BLCKSZ);
				LockBuffer(bufs[blk], BUFFER_LOCK_UNLOCK);
			}
			t_copy = vk_now_ms() - t0;
		}
		st = (uint32 *) stats_buf.map;
		memset(st, 0, ZC_STATS_WORDS * 4 * 4);
		vk_dset = (kind == 2) ? dset_st : dset_in;
		t_kern = vk_dispatch(zc_pl, &pc, sizeof pc, groups);

		scans++;
		kern_sum += t_kern;
		copy_sum += t_copy;
		kern_min = Min(kern_min, t_kern);
		kern_max = Max(kern_max, t_kern);
		copy_max = Max(copy_max, t_copy);
		last_tuples = st[0]; last_match = st[1]; last_idsum = st[2]; last_pages = st[7];
		viol += st[3]; badlp += st[4]; badhdr += st[5]; badpg += st[6]; stale_sum += st[8];
		anom = (st[3] + st[4] + st[5] + st[6]) != 0;
		if (anom)
		{
			scans_anom++;
			if (samples.len < 900)		/* keep the first few samples */
			{
				uint32		ns = Min(st[9], ZC_SAMPLES), k;

				for (k = 0; k < ns && k < 3; k++)
				{
					uint32	   *s = st + 16 + k * 8;
					int64		b = (int64) (((uint64) s[6] << 32) | s[5]);

					appendStringInfo(&samples, "\n    scan %lu: kind=%u (1 invariant, 2 line pointer, 3 tuple header, 4 page header) "
									 "blk=%u lp=%u lpword=0x%08x a=%d b=%ld d=%d",
									 (unsigned long) scans, s[0], s[1], s[2], s[3], (int32) s[4], (long) b, (int32) s[7]);
				}
			}
		}
		if (kind == 1)			/* nopin: did any buffer get a different page meanwhile? */
		{
			bool		any = false;

			for (blk = 0; blk < nblocks; blk++)
			{
				BufferDesc *bd = GetBufferDescriptor(bufs[blk] - 1);
				BufferTag	tag = bd->tag;

				if (tag.blockNum != blk || tag.relNumber != rel->rd_locator.relNumber ||
					tag.forkNum != MAIN_FORKNUM)
				{
					remapped_bufs++;
					any = true;
				}
			}
			remapped_scans += any;
		}
	} while (vk_now_ms() < t_end);

	if (kind != 1)
		for (blk = 0; blk < nblocks; blk++)
			ReleaseBuffer(bufs[blk]);
	relation_close(rel, AccessShareLock);

	{
		StringInfoData out;

		initStringInfo(&out);
		appendStringInfo(&out,
						 "mode=%s  relation blocks=%u  listed=%u (excluded by 16KB alignment: %u)\n"
						 "scans=%lu  kernel ms avg=%.1f min=%.1f max=%.1f  %s\n"
						 "last scan: pages=%u tuples=%u matches=%u idsum_mod2^32=%u\n"
						 "over all scans: scans with an anomaly=%lu  invariant violations=%lu  bad line pointers=%lu  "
						 "bad tuple headers=%lu  bad page headers=%lu\n"
						 "superseded/deleted versions counted as live (xmax set), summed over scans=%lu (avg %.0f per scan)",
						 mode, nblocks, (kind == 2) ? nblocks : nin, nexcluded,
						 (unsigned long) scans, kern_sum / scans, kern_min, kern_max,
						 kind == 2 ? psprintf("copy-under-lock ms avg=%.1f max=%.1f", copy_sum / scans, copy_max) : "no copy",
						 last_pages, last_tuples, last_match, last_idsum,
						 (unsigned long) scans_anom, (unsigned long) viol, (unsigned long) badlp,
						 (unsigned long) badhdr, (unsigned long) badpg,
						 (unsigned long) stale_sum, (double) stale_sum / scans);
		if (kind == 1)
			appendStringInfo(&out, "\nbuffer remapped during a scan: scans=%lu buffers=%lu",
							 (unsigned long) remapped_scans, (unsigned long) remapped_bufs);
		if (samples.len > 0)
			appendStringInfo(&out, "\nfirst anomalies:%s", samples.data);
		PG_RETURN_TEXT_P(cstring_to_text(out.data));
	}
}
