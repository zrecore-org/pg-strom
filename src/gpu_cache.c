/*
 * gpu_cache.c
 *
 * GPU-Cache: a copy of a table that the GPU service keeps resident and current, so that a scan of the
 * table needs no page transfer. The backend's side: the AFTER ROW trigger that logs the changes of a
 * transaction, the transaction callbacks that finalize them, the initial load, the planner / executor
 * hooks and the SQL functions. The service's side (the cache itself) is src/gpuserv/gcache.c; the
 * protocol is src/gpuserv/gcache_proto.h. The design and the semantics are in the plan
 * (.claude/hw-acceleration.md, "A7 part 2 status").
 *
 * In short
 *   - A table has a cache if it has an AFTER INSERT OR UPDATE OR DELETE ... FOR EACH ROW trigger that
 *     executes pgstrom.gpucache_sync_trigger('options'). The cache is built on first use (by the first
 *     backend that writes the table or scans it): the builder sends every visible tuple to the service.
 *   - The trigger appends a record per inserted / deleted row version to a buffer of the backend, which
 *     is sent to the service synchronously: before the transaction commits, and before the backend scans
 *     a cache (so that it sees its own changes). At the end of the transaction the service is told which
 *     transaction ids committed or aborted.
 *   - A cache scan runs as an XpuTaskExecGpuCache task: the service reads the cache like a record batch
 *     and applies the visibility of the session (committed rows and the session's own).
 *   - Whenever the cache may no longer mirror the table (the service restarted, a log record failed to
 *     apply, a transaction could not log), it is invalidated: the next user rebuilds it from the table.
 *     Only an overflow (more row versions than max_num_rows, a full data region) marks it corrupted,
 *     which needs pgstrom.gpucache_recovery().
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "access/tupmacs.h"
#include "access/transam.h"
#include "storage/spin.h"
#include "gpuserv/gpuserv_proto.h"
#include "gpuserv/gcache_proto.h"
#include "gpuserv/scan_prog.h"
#include <poll.h>

/* ----------------------------------------------------------------
 * the shared state, in the postmaster's shared memory
 * ----------------------------------------------------------------
 */
#define GPUCACHE_PHASE__NOT_BUILT		0	/* not built yet (or invalidated) */
#define GPUCACHE_PHASE__NOW_LOADING		1	/* a backend is loading it */
#define GPUCACHE_PHASE__IS_READY		2	/* usable */
#define GPUCACHE_PHASE__IS_CORRUPTED	3	/* overflow: needs gpucache_recovery() */

#define GPUCACHE_DEFAULT_MAX_ROWS		10485760L
#define GPUCACHE_DEFAULT_CACHE_SIZE		(512UL << 20)
#define GPUCACHE_DEFAULT_REDO_BUFSZ		(160UL << 20)
#define GPUCACHE_DEFAULT_SYNC_INTERVAL	5000000L		/* us */
#define GPUCACHE_MIN_REDO_BUFSZ			(16UL << 20)
#define GPUCACHE_STATE_HASH_NSLOTS		509
#define GPUCACHE_LOG_BATCH				(4UL << 20)		/* bytes of log records per request */

typedef struct
{
	int			gpu_device_id;
	int64_t		max_rows;
	size_t		cache_sz;
	size_t		redo_buffer_sz;			/* accepted for compatibility, not used */
	int64_t		sync_interval;			/* us; accepted for compatibility, not used */
	size_t		sync_threshold;			/* accepted for compatibility, not used */
} GpuCacheOptions;

typedef struct
{
	dlist_node	chain;
	Oid			database_oid;
	Oid			table_oid;
	uint32_t	table_sig;
	GpuCacheOptions opts;
	pg_atomic_uint32 phase;
	pg_atomic_uint32 built_generation;	/* generation of the service the cache was built on */
	pg_atomic_uint32 invalidate_seq;	/* bumped whenever the cache is invalidated */
	pg_atomic_uint32 loader_pid;
	pg_atomic_uint32 nrebuilds;
	pg_atomic_uint32 info_valid;
} GpuCacheSharedState;

typedef struct
{
	slock_t		lock;
	dlist_head	free_list;
	dlist_head	hash_slots[GPUCACHE_STATE_HASH_NSLOTS];
} GpuCacheSharedHead;

/* the signature of a table, kept per backend */
typedef struct
{
	Oid			table_oid;				/* hash key */
	uint32_t	table_sig;				/* 0: no GPU-Cache */
	GpuCacheOptions opts;
} GpuCacheRelSignature;

/* a transaction id the backend logged under */
typedef struct
{
	TransactionId		xid;
	SubTransactionId	subid;
} GpuCacheXid;

/*
 * GpuCacheDesc: the backend's view of a cache; its transaction-scoped fields are reset at the end of
 * each transaction.
 */
struct GpuCacheDesc
{
	Oid			table_oid;				/* hash key: table_oid, table_sig */
	uint32_t	table_sig;
	GpuCacheOptions opts;
	GpuCacheSharedState *gc_sstate;
	bool		drop_on_rollback;
	bool		drop_on_commit;
	bool		touched;				/* on the list of descs of this transaction */
	bool		unlogged;				/* this transaction changed the table without logging it */
	uint32_t	flush_generation;		/* the service generation at the first request of this transaction */
	uint32_t	flush_seq;				/* invalidate_seq at that moment */
	bool		flushed;				/* some record of this transaction reached the service */
	StringInfoData buf;					/* records not sent yet (CacheMemoryContext) */
	uint32_t	nrecords;
	int			nxids;
	int			maxxids;
	GpuCacheXid *xids;					/* transaction ids with records sent or buffered */
};

/* static variables */
static shmem_request_hook_type shmem_request_next = NULL;
static shmem_startup_hook_type shmem_startup_next = NULL;
static object_access_hook_type object_access_next = NULL;
static GpuCacheSharedHead *gpucache_shared_head = NULL;
static HTAB	   *gpucache_table_sig_htab = NULL;
static HTAB	   *gpucache_table_desc_htab = NULL;
static List	   *gpucache_touched_list = NIL;	/* descs of this transaction (TopMemoryContext) */
static bool		pgstrom_enable_gpucache;				/* GUC */
static int		pgstrom_gpucache_max_relation_entries;	/* GUC */
static int		pgstrom_gpucache_log_buffer_size_mb;	/* GUC (accepted, unused) */
static int		pgstrom_gpucache_sync_interval;			/* GUC (accepted, unused) */
static int		pgstrom_gpucache_sync_threshold;		/* GUC (accepted, unused) */

/* ----------------------------------------------------------------
 * the connection to the service for cache requests
 * ----------------------------------------------------------------
 */
static int			gcache_sockfd = -1;
static uint32_t		gcache_sock_generation = 0;
static bool			gcache_sock_busy = false;

static void
gcache_sock_close(void)
{
	if (gcache_sockfd >= 0)
		close(gcache_sockfd);
	gcache_sockfd = -1;
	gcache_sock_busy = false;
}

static uint32_t
gcache_service_generation(void)
{
	const gpuserv_control *ctl = pgstrom_gpuserv_control();

	return ctl ? __atomic_load_n(&ctl->generation, __ATOMIC_SEQ_CST) : 0;
}

static bool
gcache_sock_ensure(void)
{
	if (gcache_sockfd >= 0 && gcache_sock_busy)
		gcache_sock_close();			/* a request was interrupted: the stream is in an unknown state */
	if (gcache_sockfd >= 0 && gcache_sock_generation == gcache_service_generation() &&
		gpuserv_ready_accept())
		return true;
	gcache_sock_close();
	gcache_sockfd = gpuClientConnectControl(&gcache_sock_generation);
	return (gcache_sockfd >= 0);
}

static int64_t
gcache_now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (int64_t) tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

/* write or read exactly 'len' bytes, waiting at most until 'deadline'; 0 on success */
static int
gcache_sock_xfer(bool do_write, void *buf, size_t len, int64_t deadline, bool interruptible)
{
	char	   *p = buf;

	while (len > 0)
	{
		ssize_t		n;
		struct pollfd pfd;
		int64_t		remain;

		if (do_write)
			n = write(gcache_sockfd, p, len);
		else
			n = read(gcache_sockfd, p, len);
		if (n > 0)
		{
			p += n;
			len -= n;
			continue;
		}
		if (n == 0 && !do_write)
			return -1;					/* the service closed the connection */
		if (n < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
			return -1;
		if (interruptible)
			CHECK_FOR_INTERRUPTS();
		remain = deadline - gcache_now_ms();
		if (remain <= 0)
			return -1;
		pfd.fd = gcache_sockfd;
		pfd.events = (do_write ? POLLOUT : POLLIN);
		pfd.revents = 0;
		(void) poll(&pfd, 1, remain > 100 ? 100 : (int) remain);
	}
	return 0;
}

/*
 * One request. 0 and *p_status on a reply, -1 with a message in errbuf if the service could not be
 * reached or answered with an error. Never raises an error unless 'interruptible' and an interrupt
 * arrives.
 */
static int
gcache_rpc(uint32_t op, Oid database_oid, Oid table_oid, uint32_t table_sig,
		   const void *data, size_t datalen, uint32_t nrecords,
		   uint32_t *p_status, gcache_info *p_info, uint64_t *p_arg,
		   char *errbuf, size_t errlen, bool interruptible)
{
	size_t		hdrlen = offsetof(XpuCommand, u.gcop.data);
	XpuCommand	hdr;
	XpuCommand *resp = NULL;
	int64_t		deadline = gcache_now_ms() + 120 * 1000L;
	int			rv = -1;
	uint32_t	rhead[4];			/* magic, tag, length (8 bytes) */

	if (!gcache_sock_ensure())
	{
		snprintf(errbuf, errlen, "the GPU service is not available");
		return -1;
	}
	memset(&hdr, 0, sizeof(hdr));
	hdr.magic = XpuCommandMagicNumber;
	hdr.tag = XpuCommandTag__GpuCacheOp;
	hdr.length = hdrlen + datalen;
	hdr.u.gcop.op = op;
	hdr.u.gcop.database_oid = database_oid;
	hdr.u.gcop.table_oid = table_oid;
	hdr.u.gcop.table_sig = table_sig;
	hdr.u.gcop.nrecords = nrecords;
	gcache_sock_busy = true;
	if (gcache_sock_xfer(true, &hdr, hdrlen, deadline, interruptible) != 0 ||
		(datalen > 0 && gcache_sock_xfer(true, (void *) data, datalen, deadline, interruptible) != 0))
	{
		snprintf(errbuf, errlen, "failed to send a request to the GPU service");
		goto out;
	}
	if (gcache_sock_xfer(false, rhead, 16, deadline, interruptible) != 0)
	{
		snprintf(errbuf, errlen, "the GPU service did not answer");
		goto out;
	}
	{
		uint64_t	rlen;

		memcpy(&rlen, &rhead[2], 8);
		if (rhead[0] != XpuCommandMagicNumber || rlen < 16 || rlen > (64UL << 10))
		{
			snprintf(errbuf, errlen, "the GPU service sent a corrupted reply");
			goto out;
		}
		resp = malloc(rlen > sizeof(XpuCommand) ? rlen : sizeof(XpuCommand));
		if (!resp)
		{
			snprintf(errbuf, errlen, "out of memory");
			goto out;
		}
		memset(resp, 0, sizeof(XpuCommand));
		memcpy(resp, rhead, 16);
		if (rlen > 16 && gcache_sock_xfer(false, (char *) resp + 16, rlen - 16, deadline, interruptible) != 0)
		{
			snprintf(errbuf, errlen, "the GPU service did not answer");
			goto out;
		}
		if (resp->tag == XpuCommandTag__Error)
		{
			snprintf(errbuf, errlen, "%s", resp->u.error.message);
			gcache_sock_busy = false;
			free(resp);
			return -1;
		}
		if (resp->tag != XpuCommandTag__Success)
		{
			snprintf(errbuf, errlen, "the GPU service sent an unexpected reply");
			goto out;
		}
		*p_status = resp->u.gcop.status;
		if (p_arg)
			*p_arg = resp->u.gcop.arg;
		if (p_info)
		{
			memset(p_info, 0, sizeof(*p_info));
			if (rlen >= offsetof(XpuCommand, u.gcop.data) + sizeof(gcache_info))
				memcpy(p_info, resp->u.gcop.data, sizeof(*p_info));
		}
		rv = 0;
		gcache_sock_busy = false;
	}
out:
	if (rv != 0)
		gcache_sock_close();
	free(resp);
	return rv;
}

/* ----------------------------------------------------------------
 * trigger recognition and the options
 * ----------------------------------------------------------------
 */
static Oid	__gpucache_sync_trigger_function_oid = InvalidOid;

static Oid
gpucache_sync_trigger_function_oid(void)
{
	if (!OidIsValid(__gpucache_sync_trigger_function_oid))
	{
		Oid			namespace_oid;
		oidvector	argtypes;

		namespace_oid = get_namespace_oid("pgstrom", true);
		if (!OidIsValid(namespace_oid))
			return InvalidOid;
		memset(&argtypes, 0, sizeof(oidvector));
		SET_VARSIZE(&argtypes, offsetof(oidvector, values[0]));
		argtypes.ndim = 1;
		argtypes.dataoffset = 0;
		argtypes.elemtype = OIDOID;
		argtypes.dim1 = 0;
		argtypes.lbound1 = 0;
		__gpucache_sync_trigger_function_oid
			= GetSysCacheOid3(PROCNAMEARGSNSP,
							  Anum_pg_proc_oid,
							  CStringGetDatum("gpucache_sync_trigger"),
							  PointerGetDatum(&argtypes),
							  ObjectIdGetDatum(namespace_oid));
	}
	return __gpucache_sync_trigger_function_oid;
}

static inline bool
is_gpucache_sync_trigger(int16 trig_type, char trig_enabled, Oid trig_func_oid, int16 trig_nargs)
{
	return (trig_type == (TRIGGER_TYPE_ROW |
						  TRIGGER_TYPE_AFTER |
						  TRIGGER_TYPE_INSERT |
						  TRIGGER_TYPE_DELETE |
						  TRIGGER_TYPE_UPDATE) &&
			(trig_enabled == TRIGGER_FIRES_ON_ORIGIN ||
			 trig_enabled == TRIGGER_FIRES_ALWAYS) &&
			trig_func_oid == gpucache_sync_trigger_function_oid() &&
			(trig_nargs == 0 || trig_nargs == 1));
}

static bool
__parse_size(const char *val, size_t *p_size)
{
	char	   *end;
	long long	v = strtoll(val, &end, 10);

	if (end == val || v < 0)
		return false;
	if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0)
		v <<= 30;
	else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0)
		v <<= 20;
	else if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0)
		v <<= 10;
	else if (*end != '\0')
		return false;
	*p_size = (size_t) v;
	return true;
}

/*
 * Options of the sync trigger, 'KEY=VALUE,...': gpu_device_id, max_num_rows, cache_size, redo_buffer_size,
 * gpu_sync_interval (seconds), gpu_sync_threshold, and gpumask (of the older design). Only gpu_device_id
 * (must be 0), max_num_rows and cache_size change what the service does; the others are validated and shown
 * (the log is not buffered in shared memory in this design).
 */
static bool
__parseSyncTriggerOptions(int elevel, const char *trigger_config, GpuCacheOptions *opts)
{
	size_t		thr = 0;
	bool		has_thr = false;

	memset(opts, 0, sizeof(*opts));
	opts->max_rows = GPUCACHE_DEFAULT_MAX_ROWS;
	opts->cache_sz = GPUCACHE_DEFAULT_CACHE_SIZE;
	opts->redo_buffer_sz = GPUCACHE_DEFAULT_REDO_BUFSZ;
	opts->sync_interval = GPUCACHE_DEFAULT_SYNC_INTERVAL;
	if (trigger_config)
	{
		char   *config = pstrdup(trigger_config);
		char   *key, *val, *saved;

		for (key = strtok_r(config, ",", &saved);
			 key != NULL;
			 key = strtok_r(NULL, ",", &saved))
		{
			val = strchr(key, '=');
			if (!val)
			{
				ereport(elevel, (errmsg("gpucache: options syntax error [%s]", key)));
				return false;
			}
			*val++ = '\0';
			key = __trim(key);
			val = __trim(val);

			if (strcmp(key, "gpu_device_id") == 0)
			{
				char   *end;
				long	v = strtol(val, &end, 10);

				if (*end != '\0' || v < 0 || v >= Max(numGpuDevAttrs, 1))
				{
					ereport(elevel, (errmsg("gpucache: invalid gpu_device_id [%s]", val)));
					return false;
				}
				opts->gpu_device_id = (int) v;
			}
			else if (strcmp(key, "gpumask") == 0)
			{
				char   *end;
				long	v = strtol(val, &end, 10);

				if (*end != '\0' || v <= 0 || (v & ~1L) != 0)
				{
					ereport(elevel, (errmsg("gpucache: invalid gpumask [%s]", val)));
					return false;
				}
			}
			else if (strcmp(key, "max_num_rows") == 0)
			{
				char   *end;
				long long v = strtoll(val, &end, 10);

				if (*end != '\0' || v < 16 || v > 0x7fffffffLL)
				{
					ereport(elevel, (errmsg("gpucache: invalid max_num_rows [%s]", val)));
					return false;
				}
				opts->max_rows = v;
			}
			else if (strcmp(key, "cache_size") == 0)
			{
				size_t		v;

				if (!__parse_size(val, &v))
				{
					ereport(elevel, (errmsg("gpucache: invalid cache_size [%s]", val)));
					return false;
				}
				if (v < (64UL << 20))
				{
					ereport(elevel, (errmsg("gpucache: too small cache_size [%s], at least 64MB", val)));
					return false;
				}
				opts->cache_sz = v;
			}
			else if (strcmp(key, "redo_buffer_size") == 0)
			{
				size_t		v;

				if (!__parse_size(val, &v))
				{
					ereport(elevel, (errmsg("gpucache: invalid redo_buffer_size [%s]", val)));
					return false;
				}
				if (v < GPUCACHE_MIN_REDO_BUFSZ)
				{
					ereport(elevel, (errmsg("gpucache: 'redo_buffer_size' is too small (%zu)", v)));
					return false;
				}
				opts->redo_buffer_sz = v;
			}
			else if (strcmp(key, "gpu_sync_interval") == 0)
			{
				char   *end;
				long long v = strtoll(val, &end, 10);

				if (*end != '\0' || v < 0)
				{
					ereport(elevel, (errmsg("gpucache: invalid gpu_sync_interval [%s]", val)));
					return false;
				}
				opts->sync_interval = v * 1000000L;
			}
			else if (strcmp(key, "gpu_sync_threshold") == 0)
			{
				if (!__parse_size(val, &thr))
				{
					ereport(elevel, (errmsg("gpucache: invalid gpu_sync_threshold [%s]", val)));
					return false;
				}
				has_thr = true;
			}
			else
			{
				ereport(elevel, (errmsg("gpucache: unknown option [%s]=[%s]", key, val)));
				return false;
			}
		}
		pfree(config);
	}
	opts->sync_threshold = (has_thr ? thr : opts->redo_buffer_sz / 4);
	return true;
}

static char *
gpucache_config_options(const GpuCacheOptions *opts)
{
	return psprintf("gpu_device_id=%d,max_num_rows=%ld,redo_buffer_size=%zu,gpu_sync_interval=%ld,gpu_sync_threshold=%zu",
					opts->gpu_device_id, (long) opts->max_rows, opts->redo_buffer_sz,
					(long) opts->sync_interval, opts->sync_threshold);
}

/* ----------------------------------------------------------------
 * the signature of a table: anything that makes a cache obsolete changes it (a column added, the
 * table rewritten, other options)
 * ----------------------------------------------------------------
 */
typedef struct
{
	Oid			reltablespace;
	Oid			relfilenode;
	int			relnattrs;
	GpuCacheOptions opts;
	struct {
		Oid		atttypid;
		int32	atttypmod;
		int16	attlen;
		bool	attbyval;
		char	attalign;
		bool	attnotnull;
		bool	attisdropped;
	} attrs[1];
} GpuCacheTableSignatureBuffer;

static uint32_t
gpuCacheTableSignatureCommon(int elevel,
							 Form_pg_class pg_class,
							 TupleDesc pg_tupdesc,
							 const char *trigger_config,
							 GpuCacheOptions *p_opts)
{
	GpuCacheTableSignatureBuffer *sig;
	size_t		len = offsetof(GpuCacheTableSignatureBuffer, attrs[pg_class->relnatts]);
	GpuCacheOptions opts;

	if (!__parseSyncTriggerOptions(elevel, trigger_config, &opts))
		return 0U;
	sig = (GpuCacheTableSignatureBuffer *) palloc0(len);
	sig->reltablespace = pg_class->reltablespace;
	sig->relfilenode = pg_class->relfilenode;
	sig->relnattrs = pg_class->relnatts;
	sig->opts = opts;
	for (int j = 0; j < pg_class->relnatts; j++)
	{
		Form_pg_attribute att = TupleDescAttr(pg_tupdesc, j);

		sig->attrs[j].atttypid = att->atttypid;
		sig->attrs[j].atttypmod = att->atttypmod;
		sig->attrs[j].attlen = att->attlen;
		sig->attrs[j].attbyval = att->attbyval;
		sig->attrs[j].attalign = att->attalign;
		sig->attrs[j].attnotnull = att->attnotnull;
		sig->attrs[j].attisdropped = att->attisdropped;
	}
	if (p_opts)
		*p_opts = opts;
	{
		uint32_t	h = hash_bytes((unsigned char *) sig, len) | 0x80000000U;

		pfree(sig);
		return h;
	}
}

static uint32_t
gpuCacheTableSignature(Relation relation, GpuCacheOptions *p_opts)
{
	GpuCacheRelSignature *entry;
	Oid			table_oid = RelationGetRelid(relation);
	bool		found;

	if (!gpucache_table_sig_htab)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(Oid);
		hctl.entrysize = sizeof(GpuCacheRelSignature);
		hctl.hcxt = CacheMemoryContext;
		gpucache_table_sig_htab = hash_create("GPU-Cache Signature", 128, &hctl,
											  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}
	entry = hash_search(gpucache_table_sig_htab, &table_oid, HASH_ENTER, &found);
	if (!found)
	{
		Form_pg_class rel_form = RelationGetForm(relation);
		TupleDesc	tupdesc = RelationGetDescr(relation);

		memset((char *) entry + sizeof(Oid), 0, sizeof(GpuCacheRelSignature) - sizeof(Oid));
		if (rel_form->relkind == RELKIND_RELATION &&
			rel_form->relfilenode != 0 &&
			relation->trigdesc)
		{
			TriggerDesc *trigdesc = relation->trigdesc;
			bool		have = false;
			const char *trigger_config = NULL;

			for (int i = 0; i < trigdesc->numtriggers; i++)
			{
				Trigger *trig = &trigdesc->triggers[i];

				if (is_gpucache_sync_trigger(trig->tgtype, trig->tgenabled, trig->tgfoid, trig->tgnargs))
				{
					if (have)
					{
						elog(WARNING, "relation %s has multiple row-sync trigger for GPU-Cache",
							 RelationGetRelationName(relation));
						entry->table_sig = 0;
						goto out;
					}
					have = true;
					if (trig->tgnargs == 1)
						trigger_config = trig->tgargs[0];
				}
			}
			if (have)
				entry->table_sig = gpuCacheTableSignatureCommon(WARNING, rel_form, tupdesc,
																trigger_config, &entry->opts);
		}
	}
out:
	if (p_opts)
		*p_opts = entry->opts;
	return entry->table_sig;
}

/* the same from the catalogs as of a snapshot (DDL callbacks) */
static uint32_t
__gpuCacheTableSignatureSnapshot(Form_pg_class pg_class, Snapshot snapshot, GpuCacheOptions *p_opts)
{
	Oid			table_oid = pg_class->oid;
	const char *trigger_config = NULL;
	bool		have = false;
	Relation	srel;
	ScanKeyData	skey[2];
	SysScanDesc	sscan;
	HeapTuple	tuple;
	TupleDesc	tupdesc;
	uint32_t	table_sig = 0U;

	if (pg_class->relkind != RELKIND_RELATION || pg_class->relfilenode == 0 || !pg_class->relhastriggers)
		return 0U;
	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey[0], Anum_pg_trigger_tgrelid, BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	sscan = systable_beginscan(srel, TriggerRelidNameIndexId, true, snapshot, 1, skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_trigger pg_trig = (Form_pg_trigger) GETSTRUCT(tuple);

		if (is_gpucache_sync_trigger(pg_trig->tgtype, pg_trig->tgenabled, pg_trig->tgfoid, pg_trig->tgnargs))
		{
			if (pg_trig->tgnargs == 1)
			{
				Datum	datum;
				bool	isnull;

				datum = heap_getattr(tuple, Anum_pg_trigger_tgargs, RelationGetDescr(srel), &isnull);
				if (!isnull)
				{
					bytea  *arg = DatumGetByteaPP(datum);
					const char *p = VARDATA_ANY(arg);
					int		len = VARSIZE_ANY_EXHDR(arg);

					/* tgargs is "arg1\0arg2\0...": the first one */
					trigger_config = pnstrdup(p, strnlen(p, len));
				}
			}
			have = true;
			break;
		}
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	if (!have)
		return 0U;
	tupdesc = CreateTemplateTupleDesc(pg_class->relnatts);
	srel = table_open(AttributeRelationId, AccessShareLock);
	ScanKeyInit(&skey[0], Anum_pg_attribute_attrelid, BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(table_oid));
	ScanKeyInit(&skey[1], Anum_pg_attribute_attnum, BTGreaterStrategyNumber, F_INT2GT, Int16GetDatum(0));
	sscan = systable_beginscan(srel, AttributeRelidNumIndexId, true, snapshot, 2, skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_attribute attr = (Form_pg_attribute) GETSTRUCT(tuple);

		if (attr->attnum > 0 && attr->attnum <= pg_class->relnatts)
			memcpy(TupleDescAttr(tupdesc, attr->attnum - 1), attr, ATTRIBUTE_FIXED_PART_SIZE);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	table_sig = gpuCacheTableSignatureCommon(WARNING, pg_class, tupdesc, trigger_config, p_opts);
	FreeTupleDesc(tupdesc);
	return table_sig;
}

static uint32_t
gpuCacheTableSignatureSnapshot(Oid table_oid, Snapshot snapshot, GpuCacheOptions *p_opts)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;
	uint32_t	table_sig = 0U;

	srel = table_open(RelationRelationId, AccessShareLock);
	ScanKeyInit(&skey, Anum_pg_class_oid, BTEqualStrategyNumber, F_OIDEQ, ObjectIdGetDatum(table_oid));
	sscan = systable_beginscan(srel, ClassOidIndexId, true, snapshot, 1, &skey);
	tuple = systable_getnext(sscan);
	if (HeapTupleIsValid(tuple))
		table_sig = __gpuCacheTableSignatureSnapshot((Form_pg_class) GETSTRUCT(tuple), snapshot, p_opts);
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
	return table_sig;
}

static void
gpuCacheTableSignatureInvalidation(Oid table_oid)
{
	if (gpucache_table_sig_htab)
		hash_search(gpucache_table_sig_htab, &table_oid, HASH_REMOVE, NULL);
}

/* ----------------------------------------------------------------
 * the shared states and the descriptors
 * ----------------------------------------------------------------
 */
static uint32_t
gpuCacheHashValue(Oid database_oid, Oid table_oid, uint32_t table_sig)
{
	struct {
		Oid			database_oid;
		Oid			table_oid;
		uint32_t	table_sig;
	} hkey;

	memset(&hkey, 0, sizeof(hkey));
	hkey.database_oid = database_oid;
	hkey.table_oid = table_oid;
	hkey.table_sig = table_sig;
	return hash_bytes((unsigned char *) &hkey, sizeof(hkey));
}

static GpuCacheSharedState *
getGpuCacheSharedState(Oid database_oid, Oid table_oid, uint32_t table_sig, const GpuCacheOptions *opts)
{
	GpuCacheSharedState *gc_sstate = NULL;
	dlist_iter	iter;
	uint32_t	hindex = gpuCacheHashValue(database_oid, table_oid, table_sig) % GPUCACHE_STATE_HASH_NSLOTS;

	SpinLockAcquire(&gpucache_shared_head->lock);
	dlist_foreach(iter, &gpucache_shared_head->hash_slots[hindex])
	{
		GpuCacheSharedState *cur = dlist_container(GpuCacheSharedState, chain, iter.cur);

		if (cur->database_oid == database_oid && cur->table_oid == table_oid && cur->table_sig == table_sig)
		{
			gc_sstate = cur;
			goto found;
		}
	}
	if (!dlist_is_empty(&gpucache_shared_head->free_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gpucache_shared_head->free_list);

		gc_sstate = dlist_container(GpuCacheSharedState, chain, dnode);
		memset(gc_sstate, 0, sizeof(GpuCacheSharedState));
		gc_sstate->database_oid = database_oid;
		gc_sstate->table_oid = table_oid;
		gc_sstate->table_sig = table_sig;
		gc_sstate->opts = *opts;
		pg_atomic_init_u32(&gc_sstate->phase, GPUCACHE_PHASE__NOT_BUILT);
		pg_atomic_init_u32(&gc_sstate->built_generation, 0);
		pg_atomic_init_u32(&gc_sstate->invalidate_seq, 0);
		pg_atomic_init_u32(&gc_sstate->loader_pid, 0);
		pg_atomic_init_u32(&gc_sstate->nrebuilds, 0);
		pg_atomic_init_u32(&gc_sstate->info_valid, 0);
		dlist_push_tail(&gpucache_shared_head->hash_slots[hindex], &gc_sstate->chain);
	}
found:
	SpinLockRelease(&gpucache_shared_head->lock);
	return gc_sstate;
}

static void
releaseGpuCacheSharedState(GpuCacheSharedState *gc_sstate)
{
	SpinLockAcquire(&gpucache_shared_head->lock);
	dlist_delete(&gc_sstate->chain);
	gc_sstate->table_sig = 0;
	gc_sstate->table_oid = InvalidOid;
	dlist_push_tail(&gpucache_shared_head->free_list, &gc_sstate->chain);
	SpinLockRelease(&gpucache_shared_head->lock);
}

static GpuCacheDesc *
lookupGpuCacheDescRaw(Oid table_oid, uint32_t table_sig, const GpuCacheOptions *opts)
{
	struct { Oid table_oid; uint32_t table_sig; } hkey;
	GpuCacheDesc *entry;
	bool		found;

	if (!gpucache_shared_head)
		return NULL;
	if (!gpucache_table_desc_htab)
	{
		HASHCTL	hctl;

		memset(&hctl, 0, sizeof(hctl));
		hctl.keysize = sizeof(hkey);
		hctl.entrysize = sizeof(GpuCacheDesc);
		hctl.hcxt = CacheMemoryContext;
		gpucache_table_desc_htab = hash_create("GPU-Cache Descriptor", 128, &hctl,
											   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}
	memset(&hkey, 0, sizeof(hkey));
	hkey.table_oid = table_oid;
	hkey.table_sig = table_sig;
	entry = hash_search(gpucache_table_desc_htab, &hkey, HASH_ENTER, &found);
	if (found)
	{
		GpuCacheSharedState *ss = entry->gc_sstate;

		/* the shared slot may have been released and reused since */
		if (ss && ss->database_oid == MyDatabaseId && ss->table_oid == table_oid && ss->table_sig == table_sig)
			return entry;
	}
	else
	{
		memset((char *) entry + sizeof(hkey), 0, sizeof(GpuCacheDesc) - sizeof(hkey));
		initStringInfoCxt(CacheMemoryContext, &entry->buf);
	}
	entry->opts = *opts;
	entry->gc_sstate = getGpuCacheSharedState(MyDatabaseId, table_oid, table_sig, opts);
	if (!entry->gc_sstate)
	{
		elog(NOTICE, "too much GpuCacheSharedState entries are consumed. Please expand 'pg_strom.gpucache_max_relation_entries' then restart.");
		return NULL;
	}
	return entry;
}

static GpuCacheDesc *
lookupGpuCacheDesc(Relation relation)
{
	GpuCacheOptions opts;
	uint32_t	table_sig = gpuCacheTableSignature(relation, &opts);

	if (table_sig == 0U)
		return NULL;
	return lookupGpuCacheDescRaw(RelationGetRelid(relation), table_sig, &opts);
}

/* the cache no longer mirrors the table (as far as we know): the next user rebuilds it */
static void
gpuCacheInvalidate(GpuCacheSharedState *gc_sstate)
{
	uint32_t	expected = GPUCACHE_PHASE__IS_READY;

	pg_atomic_fetch_add_u32(&gc_sstate->invalidate_seq, 1);
	pg_atomic_compare_exchange_u32(&gc_sstate->phase, &expected, GPUCACHE_PHASE__NOT_BUILT);
}

static void
gpuCacheMarkCorrupted(GpuCacheSharedState *gc_sstate)
{
	pg_atomic_fetch_add_u32(&gc_sstate->invalidate_seq, 1);
	pg_atomic_write_u32(&gc_sstate->phase, GPUCACHE_PHASE__IS_CORRUPTED);
}

static void
gpuCacheTouch(GpuCacheDesc *gc_desc)
{
	if (!gc_desc->touched)
	{
		MemoryContext oldcxt = MemoryContextSwitchTo(TopMemoryContext);

		gpucache_touched_list = lappend(gpucache_touched_list, gc_desc);
		MemoryContextSwitchTo(oldcxt);
		gc_desc->touched = true;
		gc_desc->unlogged = false;
		gc_desc->flushed = false;
		gc_desc->flush_generation = 0;
		gc_desc->flush_seq = 0;
		gc_desc->nxids = 0;
		resetStringInfo(&gc_desc->buf);
		gc_desc->nrecords = 0;
	}
}

static void
gpuCacheAddXid(GpuCacheDesc *gc_desc, TransactionId xid)
{
	SubTransactionId subid = GetCurrentSubTransactionId();

	for (int i = gc_desc->nxids - 1; i >= 0; i--)
		if (gc_desc->xids[i].xid == xid)
			return;
	if (gc_desc->nxids == gc_desc->maxxids)
	{
		MemoryContext oldcxt = MemoryContextSwitchTo(CacheMemoryContext);
		int			ncap = (gc_desc->maxxids ? gc_desc->maxxids * 2 : 8);

		if (gc_desc->xids)
			gc_desc->xids = repalloc(gc_desc->xids, sizeof(GpuCacheXid) * ncap);
		else
			gc_desc->xids = palloc(sizeof(GpuCacheXid) * ncap);
		gc_desc->maxxids = ncap;
		MemoryContextSwitchTo(oldcxt);
	}
	gc_desc->xids[gc_desc->nxids].xid = xid;
	gc_desc->xids[gc_desc->nxids].subid = subid;
	gc_desc->nxids++;
}

/* ----------------------------------------------------------------
 * log records
 * ----------------------------------------------------------------
 */
static void
__gpuCacheAppendInsert(StringInfo buf, Relation rel, HeapTuple tuple, TransactionId xmin, TransactionId xmax)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	int			natts = tupdesc->natts;
	Datum	   *values = (Datum *) palloc(sizeof(Datum) * natts);
	bool	   *isnull = (bool *) palloc(sizeof(bool) * natts);
	size_t		tagsz = TYPEALIGN(8, natts);
	size_t		base = buf->len;
	gcache_rec_hdr hdr;
	gcache_rec_insert ri;
	char	   *tags;
	static const char zero[8] = {0};

	heap_deform_tuple(tuple, tupdesc, values, isnull);
	memset(&hdr, 0, sizeof(hdr));
	hdr.type = GCREC_INSERT;
	hdr.natts = natts;
	appendBinaryStringInfo(buf, (char *) &hdr, sizeof(hdr));
	memset(&ri, 0, sizeof(ri));
	ri.xmin = xmin;
	ri.xmax = xmax;
	ri.blkno = ItemPointerGetBlockNumber(&tuple->t_self);
	ri.offnum = ItemPointerGetOffsetNumber(&tuple->t_self);
	appendBinaryStringInfo(buf, (char *) &ri, sizeof(ri));
	enlargeStringInfo(buf, tagsz);
	tags = buf->data + buf->len;
	memset(tags, 0, tagsz);
	buf->len += tagsz;
	for (int j = 0; j < natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		if (attr->attisdropped || isnull[j])
			continue;
		if (attr->attlen > 0)
		{
			tags = buf->data + base + sizeof(hdr) + sizeof(ri);
			tags[j] = 1;
			enlargeStringInfo(buf, attr->attlen);
			if (attr->attbyval)
				store_att_byval(buf->data + buf->len, values[j], attr->attlen);
			else
				memcpy(buf->data + buf->len, DatumGetPointer(values[j]), attr->attlen);
			buf->len += attr->attlen;
		}
		else if (attr->attlen == -1)
		{
			struct varlena *datum = pg_detoast_datum_packed((struct varlena *) DatumGetPointer(values[j]));
			uint32_t	len = VARSIZE_ANY(datum);

			tags = buf->data + base + sizeof(hdr) + sizeof(ri);
			tags[j] = 2;
			appendBinaryStringInfo(buf, (char *) &len, sizeof(len));
			appendBinaryStringInfo(buf, (char *) datum, len);
			if ((Pointer) datum != DatumGetPointer(values[j]))
				pfree(datum);
		}
		else
			elog(ERROR, "gpucache: column '%s' has an unsupported length (%d)",
				 NameStr(attr->attname), (int) attr->attlen);
	}
	/* pad to 8 bytes and set the length */
	{
		size_t		len = buf->len - base;
		size_t		pad = TYPEALIGN(8, len) - len;
		uint32_t	rlen;

		if (pad > 0)
			appendBinaryStringInfo(buf, zero, pad);
		rlen = (uint32_t)(buf->len - base);
		memcpy(buf->data + base, &rlen, sizeof(rlen));
	}
	pfree(values);
	pfree(isnull);
}

static void
__gpuCacheAppendDelete(StringInfo buf, ItemPointer ctid, TransactionId xid)
{
	gcache_rec_hdr hdr;
	gcache_rec_delete rd;

	memset(&hdr, 0, sizeof(hdr));
	hdr.len = sizeof(hdr) + sizeof(rd);
	hdr.type = GCREC_DELETE;
	memset(&rd, 0, sizeof(rd));
	rd.xid = xid;
	rd.blkno = ItemPointerGetBlockNumber(ctid);
	rd.offnum = ItemPointerGetOffsetNumber(ctid);
	appendBinaryStringInfo(buf, (char *) &hdr, sizeof(hdr));
	appendBinaryStringInfo(buf, (char *) &rd, sizeof(rd));
}

static void
__gpuCacheAppendXact(StringInfo buf, TransactionId xid, bool commit)
{
	gcache_rec_hdr hdr;
	gcache_rec_xact rx;

	memset(&hdr, 0, sizeof(hdr));
	hdr.len = sizeof(hdr) + sizeof(rx);
	hdr.type = (commit ? GCREC_COMMIT : GCREC_ABORT);
	memset(&rx, 0, sizeof(rx));
	rx.xid = xid;
	appendBinaryStringInfo(buf, (char *) &hdr, sizeof(hdr));
	appendBinaryStringInfo(buf, (char *) &rx, sizeof(rx));
}

/*
 * Send records to the service. Returns the status of the request (GCST_*), or -1 if the service could
 * not be reached (the message is in errbuf).
 */
static int
gpuCacheSendRecords(GpuCacheDesc *gc_desc, StringInfo buf, uint32_t nrecords,
					char *errbuf, size_t errlen, bool interruptible)
{
	uint32_t	status = GCST_OK;

	if (nrecords == 0)
		return GCST_OK;
	if (gcache_rpc(GCOP_LOG, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
				   buf->data, buf->len, nrecords, &status, NULL, NULL,
				   errbuf, errlen, interruptible) != 0)
		return -1;
	return (int) status;
}

/*
 * A request failed: the cache does not (or may not) mirror the table any more. A rebuild fixes it;
 * an overflow needs the user.
 */
static void
gpuCacheHandleFailure(GpuCacheDesc *gc_desc, int status, const char *detail, int elevel)
{
	if (status == GCST_OVERFLOW)
	{
		gpuCacheMarkCorrupted(gc_desc->gc_sstate);
		ereport(elevel,
				(errmsg("gpucache: rowid exceeds max_num_rows (%ld), so it is now switched to 'corrupted' state",
						(long) gc_desc->opts.max_rows)));
	}
	else
	{
		gpuCacheInvalidate(gc_desc->gc_sstate);
		if (elevel >= WARNING || status == GCST_INCONSISTENT)
			ereport(status == GCST_INCONSISTENT ? DEBUG1 : elevel,
					(errmsg("gpucache: the GPU-Cache of '%s' is invalidated and will be rebuilt: %s",
							get_rel_name(gc_desc->table_oid),
							detail ? detail : (status == GCST_NOT_FOUND ? "the service lost it" : "inconsistent log"))));
	}
	gc_desc->unlogged = true;
	resetStringInfo(&gc_desc->buf);
	gc_desc->nrecords = 0;
}

/* send the buffered records of this transaction; false if the cache cannot be used any more */
static bool
gpuCacheFlush(GpuCacheDesc *gc_desc, bool interruptible)
{
	char		errbuf[256];
	int			status;

	if (gc_desc->nrecords == 0 || gc_desc->unlogged)
		return !gc_desc->unlogged;
	if (!gc_desc->flushed)
	{
		gc_desc->flush_generation = gcache_service_generation();
		gc_desc->flush_seq = pg_atomic_read_u32(&gc_desc->gc_sstate->invalidate_seq);
		gc_desc->flushed = true;
	}
	errbuf[0] = '\0';
	status = gpuCacheSendRecords(gc_desc, &gc_desc->buf, gc_desc->nrecords, errbuf, sizeof(errbuf), interruptible);
	if (status == GCST_OK)
	{
		resetStringInfo(&gc_desc->buf);
		gc_desc->nrecords = 0;
		return true;
	}
	gpuCacheHandleFailure(gc_desc, status, errbuf[0] ? errbuf : NULL, WARNING);
	return false;
}

/* ----------------------------------------------------------------
 * the initial load
 * ----------------------------------------------------------------
 */
static bool
__initialLoadVisibilityCheck(HeapTuple tuple, TransactionId *gcache_xmin, TransactionId *gcache_xmax)
{
	HeapTupleHeader	htup = tuple->t_data;
	TransactionId	xmin;
	TransactionId	xmax;

	if (!HeapTupleHeaderXminCommitted(htup))
	{
		if (HeapTupleHeaderXminInvalid(htup))
			return false;
		xmin = HeapTupleHeaderGetRawXmin(htup);
		if (TransactionIdIsCurrentTransactionId(xmin))
		{
			if (HeapTupleHeaderGetCmin(htup) >= GetCurrentCommandId(false))
			{
				/* written by this command: the trigger of the row logs it */
				return false;
			}
			*gcache_xmin = xmin;
			if (htup->t_infomask & HEAP_XMAX_INVALID)
			{
				*gcache_xmax = InvalidTransactionId;
				return true;
			}
			if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
			{
				*gcache_xmax = InvalidTransactionId;
				return true;
			}
			if (htup->t_infomask & HEAP_XMAX_IS_MULTI)
			{
				xmax = HeapTupleGetUpdateXid(htup);
				if (TransactionIdIsCurrentTransactionId(xmax))
				{
					*gcache_xmax = xmax;
					return true;
				}
				return false;
			}
			xmax = HeapTupleHeaderGetRawXmax(htup);
			if (TransactionIdIsCurrentTransactionId(xmax))
				*gcache_xmax = xmax;
			else
				*gcache_xmax = InvalidTransactionId;
			return true;
		}
		else if (TransactionIdIsInProgress(xmin))
		{
			/* another transaction's insert: its trigger logs it after the cache is ready */
			return false;
		}
		else if (!TransactionIdDidCommit(xmin))
		{
			return false;		/* aborted or crashed */
		}
	}
	/* the inserting transaction has committed */
	*gcache_xmin = FrozenTransactionId;
	if (htup->t_infomask & HEAP_XMAX_INVALID)
	{
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	if (htup->t_infomask & HEAP_XMAX_COMMITTED)
	{
		if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
		{
			*gcache_xmax = InvalidTransactionId;
			return true;
		}
		return false;
	}
	xmax = HeapTupleHeaderGetRawXmax(htup);
	if (TransactionIdIsCurrentTransactionId(xmax))
	{
		if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
			*gcache_xmax = InvalidTransactionId;
		else if (HeapTupleHeaderGetCmax(htup) >= GetCurrentCommandId(false))
			*gcache_xmax = InvalidTransactionId;	/* removed by this command: its trigger logs the delete */
		else
			*gcache_xmax = xmax;
		return true;
	}
	if (TransactionIdIsInProgress(xmax))
	{
		/* another transaction deletes it: its trigger logs the delete after the cache is ready */
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	if (!TransactionIdDidCommit(xmax))
	{
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	if (HEAP_XMAX_IS_LOCKED_ONLY(htup->t_infomask))
	{
		*gcache_xmax = InvalidTransactionId;
		return true;
	}
	return false;
}

static uint8_t
__gpuCacheColumnVt(Form_pg_attribute attr)
{
	devtype_info *dtype;

	if (attr->attisdropped)
		return GCVT_NONE;
	dtype = pgstrom_devtype_lookup(attr->atttypid);
	if (!dtype)
		return GCVT_NONE;
	switch (dtype->type_code)
	{
		case TypeOpCode__bool:
			return (attr->attlen == 1 && attr->attbyval) ? GCVT_BOOL : GCVT_NONE;
		case TypeOpCode__int1:
		case TypeOpCode__int2:
		case TypeOpCode__int4:
		case TypeOpCode__int8:
		case TypeOpCode__date:
		case TypeOpCode__time:
		case TypeOpCode__timestamp:
		case TypeOpCode__timestamptz:
			return (attr->attbyval && (attr->attlen == 1 || attr->attlen == 2 ||
									   attr->attlen == 4 || attr->attlen == 8)) ? GCVT_INT : GCVT_NONE;
		case TypeOpCode__float4:
			return (attr->attbyval && attr->attlen == 4) ? GCVT_FLOAT4 : GCVT_NONE;
		case TypeOpCode__text:
		case TypeOpCode__bpchar:
			return (attr->attlen == -1) ? GCVT_STR : GCVT_NONE;
		default:
			break;
	}
	return GCVT_NONE;
}

/* create (or reset) the cache in the service; returns the max_rows the service settled on */
static void
__gpuCacheSetupCache(GpuCacheDesc *gc_desc, Relation rel, int64_t *p_max_rows)
{
	TupleDesc	tupdesc = RelationGetDescr(rel);
	int			natts = tupdesc->natts;
	size_t		len = sizeof(gcache_setup_hdr) + sizeof(gcache_colspec) * natts;
	gcache_setup_hdr *sh;
	gcache_colspec *spec;
	uint32_t	status;
	uint64_t	arg = 0;
	char		errbuf[256];

	if (natts > GCACHE_MAXCOLS)
		elog(ERROR, "gpucache: table '%s' has too many columns (%d)", RelationGetRelationName(rel), natts);
	sh = (gcache_setup_hdr *) palloc0(len);
	sh->ncols = natts;
	sh->max_rows = gc_desc->opts.max_rows;
	sh->cache_size = gc_desc->opts.cache_sz;
	spec = (gcache_colspec *)(sh + 1);
	for (int j = 0; j < natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		spec[j].attlen = attr->attlen;
		spec[j].attbyval = attr->attbyval;
		spec[j].attalign = attr->attalign;
		spec[j].dropped = attr->attisdropped;
		spec[j].typid = attr->atttypid;
		if (attr->attisdropped)
			continue;
		if (attr->attlen != -1 && (attr->attlen < 1 || attr->attlen > 1024))
			elog(ERROR, "gpucache: column '%s' has a type that a GPU-Cache cannot hold (%s)",
				 NameStr(attr->attname), format_type_be(attr->atttypid));
		spec[j].vt = __gpuCacheColumnVt(attr);
		if (attr->attlen == -1)
		{
			int32_t		w = get_attavgwidth(RelationGetRelid(rel), attr->attnum);

			if (w <= 0)
				w = get_typavgwidth(attr->atttypid, attr->atttypmod);
			spec[j].avgwidth = Max(w, 8);
		}
	}
	errbuf[0] = '\0';
	if (gcache_rpc(GCOP_SETUP, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
				   sh, len, 0, &status, NULL, &arg, errbuf, sizeof(errbuf), true) != 0)
		elog(ERROR, "gpucache: failed on set up of the cache of '%s': %s",
			 RelationGetRelationName(rel), errbuf);
	pfree(sh);
	if (status != GCST_OK)
		elog(ERROR, "gpucache: failed on set up of the cache of '%s' (status %u)",
			 RelationGetRelationName(rel), status);
	if (p_max_rows)
		*p_max_rows = (int64_t) arg;
}

static void
__gpuCacheInitialLoadMain(GpuCacheDesc *gc_desc, Relation rel)
{
	TableScanDesc hscan;
	HeapTuple	tuple;
	StringInfoData lbuf;
	uint32_t	nrec = 0;
	char		errbuf[256];

	__gpuCacheSetupCache(gc_desc, rel, NULL);
	initStringInfo(&lbuf);
	hscan = table_beginscan(rel, SnapshotAny, 0, NULL, SO_NONE);
	while ((tuple = heap_getnext(hscan, ForwardScanDirection)) != NULL)
	{
		TransactionId	xmin, xmax;

		CHECK_FOR_INTERRUPTS();
		if (!__initialLoadVisibilityCheck(tuple, &xmin, &xmax))
			continue;
		__gpuCacheAppendInsert(&lbuf, rel, tuple, xmin, xmax);
		nrec++;
		if (xmin != FrozenTransactionId)
			gpuCacheAddXid(gc_desc, xmin);
		if (xmax != InvalidTransactionId)
			gpuCacheAddXid(gc_desc, xmax);
		if (lbuf.len >= GPUCACHE_LOG_BATCH)
		{
			int		status = gpuCacheSendRecords(gc_desc, &lbuf, nrec, errbuf, sizeof(errbuf), true);

			if (status != GCST_OK)
			{
				if (status == GCST_OVERFLOW)
				{
					gpuCacheMarkCorrupted(gc_desc->gc_sstate);
					elog(ERROR, "gpucache: table '%s' has more rows than max_num_rows (%ld), so its GPU-Cache is 'corrupted'",
						 RelationGetRelationName(rel), (long) gc_desc->opts.max_rows);
				}
				elog(ERROR, "gpucache: failed on the initial load of '%s': %s",
					 RelationGetRelationName(rel),
					 status < 0 ? errbuf : "the service rejected the rows");
			}
			resetStringInfo(&lbuf);
			nrec = 0;
		}
	}
	table_endscan(hscan);
	if (nrec > 0)
	{
		int		status = gpuCacheSendRecords(gc_desc, &lbuf, nrec, errbuf, sizeof(errbuf), true);

		if (status != GCST_OK)
		{
			if (status == GCST_OVERFLOW)
			{
				gpuCacheMarkCorrupted(gc_desc->gc_sstate);
				elog(ERROR, "gpucache: table '%s' has more rows than max_num_rows (%ld), so its GPU-Cache is 'corrupted'",
					 RelationGetRelationName(rel), (long) gc_desc->opts.max_rows);
			}
			elog(ERROR, "gpucache: failed on the initial load of '%s': %s",
				 RelationGetRelationName(rel), status < 0 ? errbuf : "the service rejected the rows");
		}
	}
	pfree(lbuf.data);
}

/*
 * Make sure the cache is built and usable. Returns false if it is not (corrupted, the service is not
 * available, or the load failed: then a WARNING says why; the caller reads the table instead).
 */
static bool
__gpuCacheInitialLoad(GpuCacheDesc *gc_desc, Relation rel)
{
	GpuCacheSharedState *gc_sstate = gc_desc->gc_sstate;
	uint32_t	phase;

	if (gc_desc->unlogged)
		return false;
	for (;;)
	{
		phase = pg_atomic_read_u32(&gc_sstate->phase);
		if (phase == GPUCACHE_PHASE__IS_READY)
		{
			uint32_t	gen = pg_atomic_read_u32(&gc_sstate->built_generation);

			if (!gpuserv_ready_accept())
				return false;
			if (gen == gcache_service_generation())
				return true;
			/* the service restarted: the cache is gone */
			gpuCacheInvalidate(gc_sstate);
			continue;
		}
		else if (phase == GPUCACHE_PHASE__NOT_BUILT)
		{
			uint32_t	expected = GPUCACHE_PHASE__NOT_BUILT;
			uint32_t	seq0, gen0;
			volatile bool ok = false;

			if (!gpuserv_ready_accept())
				return false;
			if (!pg_atomic_compare_exchange_u32(&gc_sstate->phase, &expected, GPUCACHE_PHASE__NOW_LOADING))
				continue;
			pg_atomic_write_u32(&gc_sstate->loader_pid, (uint32_t) MyProcPid);
			gpuCacheTouch(gc_desc);		/* the rows of this transaction that it loads are finalized at its end */
			seq0 = pg_atomic_read_u32(&gc_sstate->invalidate_seq);
			gen0 = gcache_service_generation();
			PG_TRY();
			{
				__gpuCacheInitialLoadMain(gc_desc, rel);
				ok = true;
			}
			PG_CATCH();
			{
				uint32_t	exp2 = GPUCACHE_PHASE__NOW_LOADING;

				pg_atomic_write_u32(&gc_sstate->loader_pid, 0);
				pg_atomic_compare_exchange_u32(&gc_sstate->phase, &exp2, GPUCACHE_PHASE__NOT_BUILT);
				PG_RE_THROW();
			}
			PG_END_TRY();
			pg_atomic_write_u32(&gc_sstate->loader_pid, 0);
			if (ok)
			{
				uint32_t	exp2 = GPUCACHE_PHASE__NOW_LOADING;

				if (pg_atomic_read_u32(&gc_sstate->invalidate_seq) != seq0 ||
					gcache_service_generation() != gen0)
				{
					/* invalidated while loading: start over */
					pg_atomic_compare_exchange_u32(&gc_sstate->phase, &exp2, GPUCACHE_PHASE__NOT_BUILT);
					continue;
				}
				pg_atomic_write_u32(&gc_sstate->built_generation, gen0);
				pg_atomic_fetch_add_u32(&gc_sstate->nrebuilds, 1);
				if (!pg_atomic_compare_exchange_u32(&gc_sstate->phase, &exp2, GPUCACHE_PHASE__IS_READY))
					continue;
				return true;
			}
		}
		else if (phase == GPUCACHE_PHASE__NOW_LOADING)
		{
			uint32_t	pid = pg_atomic_read_u32(&gc_sstate->loader_pid);

			CHECK_FOR_INTERRUPTS();
			if (pid != 0 && pid != (uint32_t) MyProcPid && kill((pid_t) pid, 0) != 0 && errno == ESRCH)
			{
				/* the loader died */
				uint32_t	exp2 = GPUCACHE_PHASE__NOW_LOADING;

				pg_atomic_compare_exchange_u32(&gc_sstate->phase, &exp2, GPUCACHE_PHASE__NOT_BUILT);
				continue;
			}
			pg_usleep(1000L);
		}
		else
			return false;			/* corrupted */
	}
}

/* wrap the load: a failure is a WARNING, and the transaction stops using the cache */
static bool
gpuCacheEnsureReady(GpuCacheDesc *gc_desc, Relation rel, bool mark_unlogged)
{
	volatile bool ready = false;
	MemoryContext oldcxt = CurrentMemoryContext;

	PG_TRY();
	{
		ready = __gpuCacheInitialLoad(gc_desc, rel);
	}
	PG_CATCH();
	{
		ErrorData  *edata;

		MemoryContextSwitchTo(oldcxt);
		edata = CopyErrorData();
		if (edata->sqlerrcode == ERRCODE_QUERY_CANCELED ||
			edata->sqlerrcode == ERRCODE_ADMIN_SHUTDOWN)
			PG_RE_THROW();
		FlushErrorState();
		ereport(WARNING,
				(errmsg("gpucache: %s", edata->message),
				 errhint("The table is read without the GPU-Cache.")));
		FreeErrorData(edata);
		if (mark_unlogged)
			gc_desc->unlogged = true;
		ready = false;
	}
	PG_END_TRY();
	return ready;
}

/* ----------------------------------------------------------------
 * the trigger
 * ----------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(pgstrom_gpucache_sync_trigger);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_sync_trigger(PG_FUNCTION_ARGS)
{
	TriggerData	   *trigdata = (TriggerData *) fcinfo->context;
	GpuCacheDesc   *gc_desc;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: must be called as trigger", get_func_name(fcinfo->flinfo->fn_oid));
	if (!TRIGGER_FIRED_FOR_ROW(trigdata->tg_event) || !TRIGGER_FIRED_AFTER(trigdata->tg_event))
		elog(ERROR, "%s: must be declared as AFTER ROW trigger", trigdata->tg_trigger->tgname);
	gc_desc = lookupGpuCacheDesc(trigdata->tg_relation);
	if (!gc_desc)
		goto bailout;
	gpuCacheTouch(gc_desc);
	if (gc_desc->unlogged)
		goto bailout;
	if (!gpuCacheEnsureReady(gc_desc, trigdata->tg_relation, true))
	{
		gc_desc->unlogged = true;		/* the cache is rebuilt after this transaction commits */
		goto bailout;
	}
	{
		TransactionId	xid = GetCurrentTransactionId();
		MemoryContext	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

		PG_TRY();
		{
			if (TRIGGER_FIRED_BY_INSERT(trigdata->tg_event))
			{
				__gpuCacheAppendInsert(&gc_desc->buf, trigdata->tg_relation,
									   trigdata->tg_trigtuple, xid, InvalidTransactionId);
				gc_desc->nrecords++;
			}
			else if (TRIGGER_FIRED_BY_UPDATE(trigdata->tg_event))
			{
				__gpuCacheAppendDelete(&gc_desc->buf, &trigdata->tg_trigtuple->t_self, xid);
				__gpuCacheAppendInsert(&gc_desc->buf, trigdata->tg_relation,
									   trigdata->tg_newtuple, xid, InvalidTransactionId);
				gc_desc->nrecords += 2;
			}
			else if (TRIGGER_FIRED_BY_DELETE(trigdata->tg_event))
			{
				__gpuCacheAppendDelete(&gc_desc->buf, &trigdata->tg_trigtuple->t_self, xid);
				gc_desc->nrecords++;
			}
			else
				elog(ERROR, "gpucache: unexpected trigger event type (%u)", trigdata->tg_event);
		}
		PG_CATCH();
		{
			MemoryContextSwitchTo(oldcxt);
			resetStringInfo(&gc_desc->buf);
			gc_desc->nrecords = 0;
			gc_desc->unlogged = true;
			PG_RE_THROW();
		}
		PG_END_TRY();
		MemoryContextSwitchTo(oldcxt);
		gpuCacheAddXid(gc_desc, xid);
		if (gc_desc->buf.len >= GPUCACHE_LOG_BATCH)
			(void) gpuCacheFlush(gc_desc, true);
	}
bailout:
	PG_RETURN_POINTER(NULL);
}

/* ----------------------------------------------------------------
 * SQL functions
 * ----------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(pgstrom_gpucache_apply_redo);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_apply_redo(PG_FUNCTION_ARGS)
{
	/* the log is applied as it is sent: only this transaction's own records can be waiting */
	ListCell   *lc;

	foreach (lc, gpucache_touched_list)
		(void) gpuCacheFlush((GpuCacheDesc *) lfirst(lc), true);
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_compaction);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_compaction(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel = relation_open(table_oid, AccessShareLock);
	GpuCacheDesc *gc_desc = lookupGpuCacheDesc(rel);

	if (gc_desc && pg_atomic_read_u32(&gc_desc->gc_sstate->phase) == GPUCACHE_PHASE__IS_READY)
	{
		uint32_t	status = 0;
		char		errbuf[256];

		(void) gpuCacheFlush(gc_desc, true);
		if (gcache_rpc(GCOP_COMPACT, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
					   NULL, 0, 0, &status, NULL, NULL, errbuf, sizeof(errbuf), true) != 0)
			elog(WARNING, "gpucache: compaction of '%s' failed: %s", RelationGetRelationName(rel), errbuf);
		else if (status == GCST_NOT_FOUND)
			gpuCacheInvalidate(gc_desc->gc_sstate);
		else if (status != GCST_OK)
			gpuCacheInvalidate(gc_desc->gc_sstate);
	}
	else
		elog(NOTICE, "GPU-Cache for '%s' is not built, so compaction is not necessary",
			 RelationGetRelationName(rel));
	relation_close(rel, AccessShareLock);
	PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_recovery);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_recovery(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	Relation	rel = relation_open(table_oid, AccessShareLock);
	GpuCacheDesc *gc_desc = lookupGpuCacheDesc(rel);

	if (gc_desc)
	{
		uint32_t	expected = GPUCACHE_PHASE__IS_CORRUPTED;

		if (pg_atomic_compare_exchange_u32(&gc_desc->gc_sstate->phase, &expected, GPUCACHE_PHASE__NOT_BUILT))
		{
			char		errbuf[256];
			uint32_t	status;

			pg_atomic_fetch_add_u32(&gc_desc->gc_sstate->invalidate_seq, 1);
			(void) gcache_rpc(GCOP_ERASE, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
							  NULL, 0, 0, &status, NULL, NULL, errbuf, sizeof(errbuf), true);
		}
		else
			elog(NOTICE, "GPU-Cache for '%s' is not corrupted, so recovery is not necessary",
				 RelationGetRelationName(rel));
	}
	relation_close(rel, AccessShareLock);
	PG_RETURN_VOID();
}

static const char *
gpucache_phase_label(uint32_t phase)
{
	switch (phase)
	{
		case GPUCACHE_PHASE__NOT_BUILT:		return "not_built";
		case GPUCACHE_PHASE__NOW_LOADING:	return "loading";
		case GPUCACHE_PHASE__IS_READY:		return "ready";
		case GPUCACHE_PHASE__IS_CORRUPTED:	return "corrupted";
		default:							break;
	}
	return "unknown";
}

static void
append_gpucache_info(StringInfo buf, GpuCacheSharedState *gc_sstate,
					 uint64_t *p_total_virtual_sz, uint64_t *p_total_physical_sz)
{
	uint32_t	phase = pg_atomic_read_u32(&gc_sstate->phase);
	char	   *config = gpucache_config_options(&gc_sstate->opts);

	appendStringInfo(buf,
					 "{ \"database_oid\" : %u"
					 ", \"table_oid\" : %u"
					 ", \"table_sig\" : %u",
					 gc_sstate->database_oid, gc_sstate->table_oid, gc_sstate->table_sig);
	if (gc_sstate->database_oid == MyDatabaseId)
	{
		const char *table_name = get_rel_name(gc_sstate->table_oid);

		if (table_name)
		{
			appendStringInfo(buf, ", \"table_name\" : ");
			escape_json(buf, table_name);
		}
	}
	appendStringInfo(buf,
					 ", \"phase\" : \"%s\""
					 ", \"gpu_mask\" : \"%08x\""
					 ", \"cache_sz\" : %zu"
					 ", \"max_num_rows\" : %ld"
					 ", \"rebuilds\" : %u"
					 ", \"config_options\" : ",
					 gpucache_phase_label(phase), 1U << gc_sstate->opts.gpu_device_id,
					 gc_sstate->opts.cache_sz, (long) gc_sstate->opts.max_rows,
					 pg_atomic_read_u32(&gc_sstate->nrebuilds));
	escape_json(buf, config);
	if (phase == GPUCACHE_PHASE__IS_READY)
	{
		gcache_info gi;
		uint32_t	status = 0;
		char		errbuf[256];

		if (gcache_rpc(GCOP_INFO, gc_sstate->database_oid, gc_sstate->table_oid, gc_sstate->table_sig,
					   NULL, 0, 0, &status, &gi, NULL, errbuf, sizeof(errbuf), true) == 0 &&
			status == GCST_OK)
		{
			appendStringInfo(buf,
							 ", \"nitems\" : %lu"
							 ", \"hwm\" : %lu"
							 ", \"free_nitems\" : %lu"
							 ", \"pending_nitems\" : %lu"
							 ", \"usage\" : %lu"
							 ", \"dead_space\" : %lu"
							 ", \"virtual_mem_sz\" : %lu"
							 ", \"physical_mem_sz\" : %lu"
							 ", \"nrecords\" : %lu",
							 (unsigned long) gi.nitems, (unsigned long) gi.hwm,
							 (unsigned long) gi.nfree, (unsigned long) gi.pending_rows,
							 (unsigned long) gi.usage, (unsigned long) gi.dead_space,
							 (unsigned long) gi.virtual_mem_sz, (unsigned long) gi.physical_mem_sz,
							 (unsigned long) gi.nrecords);
			if (p_total_virtual_sz)
				*p_total_virtual_sz += gi.virtual_mem_sz;
			if (p_total_physical_sz)
				*p_total_physical_sz += gi.physical_mem_sz;
		}
	}
	appendStringInfo(buf, " }");
	pfree(config);
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_info_one);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_info_one(PG_FUNCTION_ARGS)
{
	Oid			table_oid = PG_GETARG_OID(0);
	StringInfoData buf;
	Relation	rel;
	GpuCacheDesc *gc_desc;

	rel = relation_open(table_oid, AccessShareLock);
	gc_desc = lookupGpuCacheDesc(rel);
	if (!gc_desc)
	{
		relation_close(rel, AccessShareLock);
		PG_RETURN_NULL();
	}
	initStringInfo(&buf);
	append_gpucache_info(&buf, gc_desc->gc_sstate, NULL, NULL);
	relation_close(rel, AccessShareLock);
	PG_RETURN_TEXT_P(cstring_to_text_with_len(buf.data, buf.len));
}

PG_FUNCTION_INFO_V1(pgstrom_gpucache_info);
PUBLIC_FUNCTION(Datum)
pgstrom_gpucache_info(PG_FUNCTION_ARGS)
{
	StringInfoData	buf;
	GpuCacheSharedState **states;
	int			nstates = 0;
	uint64_t	total_virtual_sz = 0;
	uint64_t	total_physical_sz = 0;

	initStringInfo(&buf);
	if (!gpucache_shared_head)
		PG_RETURN_NULL();
	/* copy the list out of the lock: the requests to the service are slow */
	states = palloc(sizeof(GpuCacheSharedState *) * pgstrom_gpucache_max_relation_entries);
	SpinLockAcquire(&gpucache_shared_head->lock);
	for (int hindex = 0; hindex < GPUCACHE_STATE_HASH_NSLOTS; hindex++)
	{
		dlist_iter	iter;

		dlist_foreach(iter, &gpucache_shared_head->hash_slots[hindex])
		{
			if (nstates < pgstrom_gpucache_max_relation_entries)
				states[nstates++] = dlist_container(GpuCacheSharedState, chain, iter.cur);
		}
	}
	SpinLockRelease(&gpucache_shared_head->lock);
	appendStringInfo(&buf, "{ \"caches\" : [");
	for (int i = 0; i < nstates; i++)
	{
		if (i > 0)
			appendStringInfo(&buf, ", ");
		append_gpucache_info(&buf, states[i], &total_virtual_sz, &total_physical_sz);
	}
	appendStringInfo(&buf, "], \"num_caches\" : %d, \"total_virtual_sz\" : %lu, \"total_physical_sz\" : %lu }",
					 nstates, (unsigned long) total_virtual_sz, (unsigned long) total_physical_sz);
	PG_RETURN_TEXT_P(cstring_to_text_with_len(buf.data, buf.len));
}

/* ----------------------------------------------------------------
 * query planning
 * ----------------------------------------------------------------
 */
bool
baseRelHasGpuCache(PlannerInfo *root, RelOptInfo *baserel)
{
	RangeTblEntry *rte = root->simple_rte_array[baserel->relid];
	bool		retval = false;

	if (pgstrom_enable_gpucache &&
		gpucache_shared_head &&
		rte->rtekind == RTE_RELATION &&
		(baserel->reloptkind == RELOPT_BASEREL ||
		 baserel->reloptkind == RELOPT_OTHER_MEMBER_REL))
	{
		Relation	rel = table_open(rte->relid, NoLock);
		GpuCacheDesc *gc_desc = lookupGpuCacheDesc(rel);

		if (gc_desc && !gc_desc->unlogged)
		{
			uint32_t	phase = pg_atomic_read_u32(&gc_desc->gc_sstate->phase);

			if (phase != GPUCACHE_PHASE__IS_CORRUPTED)
				retval = true;
		}
		table_close(rel, NoLock);
	}
	return retval;
}

/* ----------------------------------------------------------------
 * query execution
 * ----------------------------------------------------------------
 */
bool
pgstromGpuCacheExecInit(pgstromTaskScanState *ptss)
{
	ptss->gpucache_scan_count = NULL;
	ptss->gpucache_desc = NULL;
	if (pgstrom_enable_gpucache && gpucache_shared_head && gpuserv_ready_accept())
	{
		GpuCacheDesc *gc_desc = lookupGpuCacheDesc(ptss->scan_rel);

		if (gc_desc && !gc_desc->unlogged)
		{
			/* our own changes must be in the cache before it is read */
			if (gc_desc->touched && !gpuCacheFlush(gc_desc, true))
				return false;
			if (gpuCacheEnsureReady(gc_desc, ptss->scan_rel, false))
			{
				ptss->gpucache_desc = gc_desc;
			}
		}
	}
	return (ptss->gpucache_desc != NULL);
}

/* the header of the KDS the cache task carries: the columns of the table */
static const kern_data_store *
__gpuCacheKdsHead(pgstromTaskState *pts, pgstromTaskScanState *ptss)
{
	if (!ptss->kds_head)
	{
		EState	   *estate = pts->css.ss.ps.state;
		TupleDesc	tupdesc = RelationGetDescr(ptss->scan_rel);
		size_t		head_sz = estimate_kern_data_store(tupdesc);
		kern_data_store *head = MemoryContextAllocZero(estate->es_query_cxt, head_sz);

		setup_kern_data_store(head, tupdesc, 0, KDS_FORMAT_COLUMN);
		head->table_oid = RelationGetRelid(ptss->scan_rel);
		ptss->kds_head = head;
	}
	return ptss->kds_head;
}

/*
 * Can the cache serve this session? A column whose value the program reads must be one the GPU can load
 * from the cache; a numeric (and any other type the cache only passes through) cannot. A join program
 * never reads such values (the planner does not offer them), so only scans and aggregations are checked.
 */
bool
pgstromGpuCacheProgramOk(pgstromTaskState *pts, const XpuCommand *session_cmd)
{
	const kern_session_info *session = &session_cmd->u.session;
	scan_prog	prog;
	char		err[256];
	bool		retval = true;

	if ((session->xpu_task_flags & DEVTASK__MASK) == DEVTASK__JOIN || session->join_inner_handle != 0)
		return true;
	for (int k=0; k < pts->num_scan_rels; k++)
	{
		pgstromTaskScanState *ptss = &pts->scan_rels[k];

		if (!ptss->gpucache_desc)
			continue;
		memset(&prog, 0, sizeof(prog));
		if (scan_prog_build(session, __gpuCacheKdsHead(pts, ptss), &prog, err, sizeof(err)) == 0)
		{
			uint32_t	coloff = prog.words[0];
			uint32_t	ncols = prog.words[3];

			for (uint32_t c = 0; c < ncols; c++)
			{
				uint32_t	desc = prog.words[coloff + c];

				if (((desc >> 16) & 0xff) != 0xff && ((desc >> 12) & 0xf) == SOP_VT_NUM)
					retval = false;
			}
			scan_prog_free(&prog);
		}
		/* a program the service cannot build fails the same way for a table: let it report */
		break;			/* every scan relation runs the same program */
	}
	return retval;
}

XpuCommand *
pgstromScanChunkGpuCache(pgstromTaskState *pts,
						 pgstromTaskScanState *ptss,
						 struct iovec *xcmd_iov,
						 int *xcmd_iovcnt)
{
	GpuCacheDesc *gc_desc = ptss->gpucache_desc;
	uint32_t	gpucache_count;

	if (!ptss->gpucache_scan_count)
		ptss->gpucache_scan_count = &ptss->__gpucache_count_data;
	gpucache_count = pg_atomic_fetch_add_u32(ptss->gpucache_scan_count, 1);
	if (gpucache_count == 0)
	{
		XpuCommand *xcmd;
		kern_data_store *kds;
		size_t		sz;

		__gpuCacheKdsHead(pts, ptss);
		sz = offsetof(XpuCommand, u.gc_task.data) + MAXALIGN(offsetof(kern_data_store, colmeta[ptss->kds_head->nr_colmeta]));
		resetStringInfo(&pts->xcmd_buf);
		enlargeStringInfo(&pts->xcmd_buf, sz);
		xcmd = (XpuCommand *) pts->xcmd_buf.data;
		memset(xcmd, 0, offsetof(XpuCommand, u.gc_task.data));
		xcmd->magic = XpuCommandMagicNumber;
		xcmd->tag = XpuCommandTag__XpuTaskExecGpuCache;
		xcmd->gpumask = 1UL;
		xcmd->u.gc_task.t.scan_relidx = ptss->scan_relidx;
		xcmd->u.gc_task.t.kds_src_offset = offsetof(XpuCommand, u.gc_task.data);
		xcmd->u.gc_task.database_oid = MyDatabaseId;
		xcmd->u.gc_task.table_oid = gc_desc->table_oid;
		xcmd->u.gc_task.table_sig = gc_desc->table_sig;
		kds = (kern_data_store *)((char *) xcmd + offsetof(XpuCommand, u.gc_task.data));
		memcpy(kds, ptss->kds_head, sz - offsetof(XpuCommand, u.gc_task.data));
		pts->xcmd_buf.len = sz;
		xcmd->length = sz;
		xcmd_iov->iov_base = pts->xcmd_buf.data;
		xcmd_iov->iov_len = pts->xcmd_buf.len;
		*xcmd_iovcnt = 1;
		return xcmd;
	}
	/* a single task reads the whole cache: move to the next relation */
	pts->curr_scan_rel++;
	return NULL;
}

void
pgstromGpuCacheExecReset(pgstromTaskScanState *ptss)
{
	ptss->gpucache_scan_count = NULL;
	pg_atomic_write_u32(&ptss->__gpucache_count_data, 0);
}

void
pgstromGpuCacheInitDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
	ptss->gpucache_scan_count = &psss->gpucache_count_data;
}

void
pgstromGpuCacheAttachDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
	ptss->gpucache_scan_count = &psss->gpucache_count_data;
}

void
pgstromGpuCacheShutdownDSM(pgstromTaskScanState *ptss, pgstromSharedScanState *psss)
{
	if (ptss->gpucache_scan_count)
	{
		uint32_t	ival = pg_atomic_read_u32(ptss->gpucache_scan_count);

		pg_atomic_write_u32(&ptss->__gpucache_count_data, ival);
		ptss->gpucache_scan_count = &ptss->__gpucache_count_data;
	}
}

void
pgstromGpuCacheExplain(pgstromTaskScanState *ptss, ExplainState *es, const char *prefix)
{
	GpuCacheDesc *gc_desc = ptss->gpucache_desc;
	char		label[100];
	char		buf[1024];

	if (gc_desc)
	{
		char	   *config = gpucache_config_options(&gc_desc->opts);

		if (!prefix)
			snprintf(label, sizeof(label), "GPU Cache");
		else
			snprintf(label, sizeof(label), "GPU Cache [%s]", prefix);
		snprintf(buf, sizeof(buf), "GPU%d [phase: %s, max_num_rows: %ld]",
				 gc_desc->opts.gpu_device_id,
				 gpucache_phase_label(pg_atomic_read_u32(&gc_desc->gc_sstate->phase)),
				 (long) gc_desc->opts.max_rows);
		ExplainPropertyText(label, buf, es);
		if (!prefix)
			snprintf(label, sizeof(label), "GPU Cache Options");
		else
			snprintf(label, sizeof(label), "GPU Cache Options [%s]", prefix);
		ExplainPropertyText(label, config, es);
		pfree(config);
	}
}

/* ----------------------------------------------------------------
 * DDL callbacks: when the trigger, the columns or the table change, the old cache is obsolete
 * ----------------------------------------------------------------
 */
static void
__gpuCacheDropOnCommit(Oid table_oid, uint32_t table_sig, const GpuCacheOptions *opts, bool on_commit)
{
	GpuCacheDesc *gc_desc;

	if (table_sig == 0U)
		return;
	gc_desc = lookupGpuCacheDescRaw(table_oid, table_sig, opts);
	if (gc_desc)
	{
		gpuCacheTouch(gc_desc);
		if (on_commit)
			gc_desc->drop_on_commit = true;
		else
			gc_desc->drop_on_rollback = true;
	}
}

static void
__gpuCacheCallbackOnAlterTable(Oid table_oid)
{
	GpuCacheOptions new_opts, old_opts;
	uint32_t	new_signature, old_signature;

	new_signature = gpuCacheTableSignatureSnapshot(table_oid, SnapshotSelf, &new_opts);
	old_signature = gpuCacheTableSignatureSnapshot(table_oid, NULL, &old_opts);
	if (old_signature != 0U && old_signature != new_signature)
		__gpuCacheDropOnCommit(table_oid, old_signature, &old_opts, true);
	if (new_signature != 0U && new_signature != old_signature)
		__gpuCacheDropOnCommit(table_oid, new_signature, &new_opts, false);
}

static void
__gpuCacheCallbackOnAlterTrigger(Oid trigger_oid, int elevel)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;

	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey, Anum_pg_trigger_oid, BTEqualStrategyNumber, F_OIDEQ, ObjectIdGetDatum(trigger_oid));
	sscan = systable_beginscan(srel, TriggerOidIndexId, true, SnapshotSelf, 1, &skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
	{
		Form_pg_trigger pg_trig = (Form_pg_trigger) GETSTRUCT(tuple);
		Oid			table_oid = pg_trig->tgrelid;
		Relation	__rel = table_open(table_oid, NoLock);

		/* validate the options now, and look for a second row-sync trigger */
		(void) gpuCacheTableSignatureSnapshot(table_oid, SnapshotSelf, NULL);
		if (is_gpucache_sync_trigger(pg_trig->tgtype, pg_trig->tgenabled, pg_trig->tgfoid, pg_trig->tgnargs) &&
			__rel->trigdesc)
		{
			TriggerDesc *trigdesc = __rel->trigdesc;

			for (int i = 0; i < trigdesc->numtriggers; i++)
			{
				Trigger *trigger = &trigdesc->triggers[i];

				if (trigger->tgoid != trigger_oid &&
					is_gpucache_sync_trigger(trigger->tgtype, trigger->tgenabled, trigger->tgfoid, trigger->tgnargs))
					ereport(elevel,
							(errmsg("gpucache: relation %s has multiple row-sync triggers",
									RelationGetRelationName(__rel))));
			}
		}
		table_close(__rel, NoLock);
		__gpuCacheCallbackOnAlterTable(table_oid);
	}
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
}

static void
__gpuCacheOnDropRelation(Oid table_oid)
{
	GpuCacheOptions opts;
	uint32_t	table_sig = gpuCacheTableSignatureSnapshot(table_oid, NULL, &opts);

	__gpuCacheDropOnCommit(table_oid, table_sig, &opts, true);
}

static void
__gpuCacheOnDropTrigger(Oid trigger_oid)
{
	Relation	srel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	HeapTuple	tuple;

	srel = table_open(TriggerRelationId, AccessShareLock);
	ScanKeyInit(&skey, Anum_pg_trigger_oid, BTEqualStrategyNumber, F_OIDEQ, ObjectIdGetDatum(trigger_oid));
	sscan = systable_beginscan(srel, TriggerOidIndexId, true, NULL, 1, &skey);
	while ((tuple = systable_getnext(sscan)) != NULL)
		__gpuCacheOnDropRelation(((Form_pg_trigger) GETSTRUCT(tuple))->tgrelid);
	systable_endscan(sscan);
	table_close(srel, AccessShareLock);
}

static void
gpuCacheObjectAccess(ObjectAccessType access, Oid classId, Oid objectId, int subId, void *arg)
{
	if (object_access_next)
		object_access_next(access, classId, objectId, subId, arg);
	if (!gpucache_shared_head)
		return;
	if (access == OAT_POST_CREATE)
	{
		if (classId == RelationRelationId && subId > 0)
			__gpuCacheCallbackOnAlterTable(objectId);		/* ADD COLUMN */
		else if (classId == TriggerRelationId)
			__gpuCacheCallbackOnAlterTrigger(objectId, ERROR);
	}
	else if (access == OAT_POST_ALTER)
	{
		if (classId == RelationRelationId)
			__gpuCacheCallbackOnAlterTable(objectId);
		else if (classId == TriggerRelationId)
			__gpuCacheCallbackOnAlterTrigger(objectId, WARNING);
	}
	else if (access == OAT_DROP)
	{
		if (classId == RelationRelationId)
			__gpuCacheOnDropRelation(objectId);
		else if (classId == TriggerRelationId)
			__gpuCacheOnDropTrigger(objectId);
	}
	else if (access == OAT_TRUNCATE)
		__gpuCacheOnDropRelation(objectId);
}

/* ----------------------------------------------------------------
 * transaction callbacks
 * ----------------------------------------------------------------
 */
/* the end of a transaction: finalize what the cache holds for it, then forget it */
static void
gpuCacheEndOfTransaction(bool normal_commit)
{
	ListCell   *lc;
	uint32_t	generation = gcache_service_generation();

	foreach (lc, gpucache_touched_list)
	{
		GpuCacheDesc *gc_desc = (GpuCacheDesc *) lfirst(lc);
		GpuCacheSharedState *ss = gc_desc->gc_sstate;
		bool		drop = (normal_commit ? gc_desc->drop_on_commit : gc_desc->drop_on_rollback);

		if (drop)
		{
			char		errbuf[256];
			uint32_t	status;

			/* the cache is obsolete: forget it in the service and in the shared memory */
			(void) gcache_rpc(GCOP_ERASE, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
							  NULL, 0, 0, &status, NULL, NULL, errbuf, sizeof(errbuf), false);
			if (ss && ss->database_oid == MyDatabaseId && ss->table_oid == gc_desc->table_oid &&
				ss->table_sig == gc_desc->table_sig)
				releaseGpuCacheSharedState(ss);
			gc_desc->gc_sstate = NULL;
		}
		else if (ss && (gc_desc->flushed || gc_desc->nrecords > 0 || gc_desc->unlogged || gc_desc->nxids > 0))
		{
			bool		affected = false;

			if (gc_desc->unlogged)
				affected = true;
			else if (gc_desc->flushed &&
					 (gc_desc->flush_generation != generation ||
					  pg_atomic_read_u32(&ss->invalidate_seq) != gc_desc->flush_seq))
				affected = true;
			if (!affected && gc_desc->nxids > 0 && pg_atomic_read_u32(&ss->phase) == GPUCACHE_PHASE__IS_READY)
			{
				/* tell the service which transactions committed */
				StringInfoData buf;
				char		errbuf[256];
				uint32_t	status = 0;

				initStringInfo(&buf);
				for (int i = 0; i < gc_desc->nxids; i++)
				{
					TransactionId xid = gc_desc->xids[i].xid;
					bool		committed = normal_commit && TransactionIdDidCommit(xid);

					__gpuCacheAppendXact(&buf, xid, committed);
				}
				if (gcache_rpc(GCOP_LOG, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
							   buf.data, buf.len, gc_desc->nxids, &status, NULL, NULL,
							   errbuf, sizeof(errbuf), false) != 0 ||
					status != GCST_OK)
					affected = true;
				pfree(buf.data);
			}
			if (affected)
				gpuCacheInvalidate(ss);
		}
		gc_desc->touched = false;
		gc_desc->drop_on_commit = false;
		gc_desc->drop_on_rollback = false;
		gc_desc->unlogged = false;
		gc_desc->flushed = false;
		gc_desc->nxids = 0;
		gc_desc->nrecords = 0;
		resetStringInfo(&gc_desc->buf);
	}
	list_free(gpucache_touched_list);
	gpucache_touched_list = NIL;
}

static void
gpuCacheXactCallback(XactEvent event, void *arg)
{
	if (gpucache_touched_list == NIL)
		return;
	switch (event)
	{
		case XACT_EVENT_PRE_PREPARE:
			{
				ListCell   *lc;

				foreach (lc, gpucache_touched_list)
				{
					GpuCacheDesc *gc_desc = (GpuCacheDesc *) lfirst(lc);

					if (gc_desc->nxids > 0 || gc_desc->nrecords > 0)
						ereport(ERROR,
								(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
								 errmsg("a transaction that modified the table \"%s\" with a GPU-Cache cannot be prepared",
										get_rel_name(gc_desc->table_oid))));
				}
			}
			break;
		case XACT_EVENT_PRE_COMMIT:
			{
				ListCell   *lc;

				/* the changes reach the service before the commit becomes visible */
				foreach (lc, gpucache_touched_list)
					(void) gpuCacheFlush((GpuCacheDesc *) lfirst(lc), false);
			}
			break;
		case XACT_EVENT_COMMIT:
		case XACT_EVENT_PARALLEL_COMMIT:
			gpuCacheEndOfTransaction(true);
			break;
		case XACT_EVENT_ABORT:
		case XACT_EVENT_PARALLEL_ABORT:
			gpuCacheEndOfTransaction(false);
			break;
		default:
			break;
	}
}

static void
gpuCacheSubXactCallback(SubXactEvent event, SubTransactionId mySubid, SubTransactionId parentSubid, void *arg)
{
	ListCell   *lc;

	if (event != SUBXACT_EVENT_ABORT_SUB || gpucache_touched_list == NIL)
		return;
	/* rows of the aborted subtransaction can go now (the final status of every id is checked at the end anyway) */
	foreach (lc, gpucache_touched_list)
	{
		GpuCacheDesc *gc_desc = (GpuCacheDesc *) lfirst(lc);
		StringInfoData buf;
		int			n = 0, k = 0;
		char		errbuf[256];
		uint32_t	status = 0;

		if (gc_desc->unlogged || gc_desc->nxids == 0 || !gc_desc->flushed)
			continue;
		if (!gpuCacheFlush(gc_desc, false))
			continue;
		initStringInfo(&buf);
		for (int i = 0; i < gc_desc->nxids; i++)
		{
			if (gc_desc->xids[i].subid >= mySubid)
			{
				__gpuCacheAppendXact(&buf, gc_desc->xids[i].xid, false);
				n++;
			}
			else
				gc_desc->xids[k++] = gc_desc->xids[i];
		}
		gc_desc->nxids = k;
		if (n > 0 &&
			(gcache_rpc(GCOP_LOG, MyDatabaseId, gc_desc->table_oid, gc_desc->table_sig,
						buf.data, buf.len, n, &status, NULL, NULL, errbuf, sizeof(errbuf), false) != 0 ||
			 status != GCST_OK))
			gc_desc->unlogged = true;
		pfree(buf.data);
	}
}

static void
gpuCacheRelcacheCallback(Datum arg, Oid relid)
{
	if (OidIsValid(relid))
		gpuCacheTableSignatureInvalidation(relid);
	else if (gpucache_table_sig_htab)
	{
		HASH_SEQ_STATUS hseq;
		GpuCacheRelSignature *entry;

		hash_seq_init(&hseq, gpucache_table_sig_htab);
		while ((entry = hash_seq_search(&hseq)) != NULL)
			hash_search(gpucache_table_sig_htab, &entry->table_oid, HASH_REMOVE, NULL);
	}
}

static void
gpuCacheSyscacheCallback(Datum arg, int cacheid, uint32 hashvalue)
{
	__gpucache_sync_trigger_function_oid = InvalidOid;
}

/* ----------------------------------------------------------------
 * start-up
 * ----------------------------------------------------------------
 */
static void
pgstrom_request_gpu_cache(void)
{
	size_t		len;

	if (shmem_request_next)
		shmem_request_next();
	len = MAXALIGN(sizeof(GpuCacheSharedHead)) +
		MAXALIGN(sizeof(GpuCacheSharedState)) * pgstrom_gpucache_max_relation_entries;
	RequestAddinShmemSpace(len);
}

static void
pgstrom_startup_gpu_cache(void)
{
	size_t		unitsz = MAXALIGN(sizeof(GpuCacheSharedState));
	size_t		len = MAXALIGN(sizeof(GpuCacheSharedHead)) + unitsz * pgstrom_gpucache_max_relation_entries;
	char	   *pos;
	bool		found;

	if (shmem_startup_next)
		shmem_startup_next();
	pos = ShmemInitStruct("GPU-Cache Shared Head", len, &found);
	if (found)
		elog(ERROR, "Bug? GpuCacheSharedHead already exist");
	memset(pos, 0, len);
	gpucache_shared_head = (GpuCacheSharedHead *) pos;
	SpinLockInit(&gpucache_shared_head->lock);
	dlist_init(&gpucache_shared_head->free_list);
	for (int i = 0; i < GPUCACHE_STATE_HASH_NSLOTS; i++)
		dlist_init(&gpucache_shared_head->hash_slots[i]);
	pos += MAXALIGN(sizeof(GpuCacheSharedHead));
	for (int k = 0; k < pgstrom_gpucache_max_relation_entries; k++)
	{
		GpuCacheSharedState *gc_sstate = (GpuCacheSharedState *) pos;

		dlist_push_tail(&gpucache_shared_head->free_list, &gc_sstate->chain);
		pos += unitsz;
	}
}

void
pgstrom_init_gpu_cache(void)
{
	DefineCustomBoolVariable("pg_strom.enable_gpucache",
							 "Enables GPU-Cache as data source for Scan",
							 NULL,
							 &pgstrom_enable_gpucache,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpucache_max_relation_entries",
							"max number of relations GPU-Cache can be configured. Usually, no need to change from the default.",
							NULL,
							&pgstrom_gpucache_max_relation_entries,
							1024,
							1024,
							INT_MAX / 4096,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* accepted for compatibility with the CUDA product; the log is applied as it is sent here */
	DefineCustomIntVariable("pg_strom.gpucache_log_buffer_size",
							"gpucache: accepted for compatibility (not used by the Vulkan service)",
							NULL,
							&pgstrom_gpucache_log_buffer_size_mb,
							512,
							64,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpucache_sync_interval",
							"gpucache: accepted for compatibility (not used by the Vulkan service)",
							NULL,
							&pgstrom_gpucache_sync_interval,
							3000,
							500,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.gpucache_sync_threshold",
							"gpucache: accepted for compatibility (not used by the Vulkan service)",
							NULL,
							&pgstrom_gpucache_sync_threshold,
							256,
							16,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	shmem_request_next = shmem_request_hook;
	shmem_request_hook = pgstrom_request_gpu_cache;
	shmem_startup_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_cache;
	object_access_next = object_access_hook;
	object_access_hook = gpuCacheObjectAccess;
	CacheRegisterRelcacheCallback(gpuCacheRelcacheCallback, 0);
	CacheRegisterSyscacheCallback(PROCOID, gpuCacheSyscacheCallback, 0);
	RegisterXactCallback(gpuCacheXactCallback, NULL);
	RegisterSubXactCallback(gpuCacheSubXactCallback, NULL);
}
