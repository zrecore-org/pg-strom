/*
 * fallback.c
 *
 * Portion of CPU Fallback operations
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/*
 * Routines to store/fetch fallback tuples
 */
static void
pgstromStoreFallbackTuple(pgstromTaskState *pts, MinimalTuple mtup)
{
	MemoryContext memcxt = pts->css.ss.ps.state->es_query_cxt;
	kern_tupitem *titem;
	uint32_t	rowid;
	size_t		sz;

	if (!pts->fallback_tuples)
	{
		pts->fallback_index = 0;
		pts->fallback_nitems = 0;
		pts->fallback_nrooms = 1000;
		pts->fallback_tuples =
			MemoryContextAlloc(memcxt, sizeof(off_t) * pts->fallback_nrooms);
	}
	if (!pts->fallback_buffer)
	{
		pts->fallback_usage = 0;
		pts->fallback_bufsz = 8 * BLCKSZ;
		pts->fallback_buffer =
			MemoryContextAlloc(memcxt, pts->fallback_bufsz);
	}
	sz = MAXALIGN(mtup->t_len + ROWID_SIZE);
	while (pts->fallback_usage + sz > pts->fallback_bufsz)
	{
		pts->fallback_bufsz = pts->fallback_bufsz * 2 + BLCKSZ;
		pts->fallback_buffer = repalloc_huge(pts->fallback_buffer,
											 pts->fallback_bufsz);
	}
	while (pts->fallback_nitems >= pts->fallback_nrooms)
	{
		pts->fallback_nrooms = pts->fallback_nrooms * 2 + 100;
		pts->fallback_tuples = repalloc_huge(pts->fallback_tuples,
											 sizeof(off_t) * pts->fallback_nrooms);
	}
	rowid = pts->fallback_nitems++;

	titem = (kern_tupitem *)(pts->fallback_buffer +
							 pts->fallback_usage);
	memcpy(titem, mtup, mtup->t_len);
	titem->t_len += ROWID_SIZE;
	KERN_TUPITEM_SET_ROWID(titem, rowid);

	pts->fallback_tuples[rowid] = pts->fallback_usage;
	pts->fallback_usage += sz;
}

TupleTableSlot *
pgstromFetchFallbackTuple(pgstromTaskState *pts)
{
	if (pts->fallback_tuples &&
		pts->fallback_buffer &&
		pts->fallback_index < pts->fallback_nitems)
	{
		TupleTableSlot *slot = pts->css.ss.ss_ScanTupleSlot;
		kern_tupitem   *titem = (kern_tupitem *)
			(pts->fallback_buffer +
			 pts->fallback_tuples[pts->fallback_index++]);
		ExecForceStoreMinimalTuple((MinimalTuple)titem, slot, false);
		/* reset the buffer if last one */
		if (pts->fallback_index == pts->fallback_nitems)
		{
			pts->fallback_index = 0;
			pts->fallback_nitems = 0;
			pts->fallback_usage = 0;
		}
		slot_getallattrs(slot);
		return slot;
	}
	return NULL;
}

/*
 * Core routines for CPU-Fallback SCAN/JOIN/PREAGG
 */
static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts,
							  int depth,
							  uint64_t l_state,
							  bool matched);

static void
__execFallbackLoadVarsSlot(TupleTableSlot *fallback_slot,
						   List *inner_load_src,
						   List *inner_load_dst,
						   const kern_data_store *kds,
						   const kern_tupitem *titem)
{
	const char *payload;
	uint32_t	ncols, h_off = 0;
	bool		heap_hasnull;
	ListCell   *lc1, *lc2;

	/* system attributes should not appear in the inner-buffer */
	forboth (lc1, inner_load_src,
			 lc2, inner_load_dst)
	{
		int		src = lfirst_int(lc1);
		int		dst = lfirst_int(lc2);

		if (!titem)
			goto fillup_by_null;
		if (src > 0)
			break;
		else
		{
			fallback_slot->tts_isnull[dst] = true;
			fallback_slot->tts_values[dst] = 0;
		}
	}
	/* extract the user data */
	payload = KERN_TUPITEM_GET_PAYLOAD(titem);
	ncols = Min(titem->t_infomask2 & HEAP_NATTS_MASK, kds->ncols);
	heap_hasnull = ((titem->t_infomask & HEAP_HASNULL) != 0);
	for (int j=0; j < ncols && lc1 && lc2; j++)
	{
		const kern_colmeta *cmeta = &kds->colmeta[j];
		const char *addr;
		Datum		datum;

		if (heap_hasnull && att_isnull(j, titem->t_bits))
		{
			addr = NULL;
			datum = 0;
		}
		else
		{
			if (cmeta->attlen > 0)
				h_off = TYPEALIGN(cmeta->attalign, h_off);
			else if (!VARATT_NOT_PAD_BYTE((char *)payload + h_off))
				h_off = TYPEALIGN(cmeta->attalign, h_off);
			addr = (payload + h_off);
			if (cmeta->attlen > 0)
				h_off += cmeta->attlen;
			else if (cmeta->attlen == -1)
				h_off += VARSIZE_ANY(addr);
			else
				elog(ERROR, "unknown typlen (%d)", cmeta->attlen);

			if (cmeta->attbyval)
			{
				switch (cmeta->attlen)
				{
					case 1:
						datum = CharGetDatum(*((char *)addr));		/* as fetch_att() */
						break;
					case 2:
						/* as fetch_att(): a negative int2/int4 is a sign-extended Datum, which partial min/max
						 * transition functions (PG_GETARG_INT64 of an int4 argument) rely on */
						datum = Int16GetDatum(*((int16_t *)addr));
						break;
					case 4:
						datum = Int32GetDatum(*((int32_t *)addr));
						break;
					case 8:
						datum = *((uint64_t *)addr);
						break;
					default:
						elog(ERROR, "invalid typlen (%d) of inline type",
							 cmeta->attlen);
				}
			}
			else
			{
				datum = PointerGetDatum(addr);
			}
		}
		if (lfirst_int(lc1) == j+1)
		{
			int		dst = lfirst_int(lc2) - 1;

			fallback_slot->tts_isnull[dst] = !addr;
			fallback_slot->tts_values[dst] = datum;

			lc1 = lnext(inner_load_src, lc1);
			lc2 = lnext(inner_load_dst, lc2);
		}
	}
fillup_by_null:
	/* fill-up by NULL for the remaining fields */
	while (lc1 && lc2)
	{
		int		dst = lfirst_int(lc2) - 1;

		fallback_slot->tts_isnull[dst] = true;
		fallback_slot->tts_values[dst] = 0;

		lc1 = lnext(inner_load_src, lc1);
		lc2 = lnext(inner_load_dst, lc2);
	}
}

static void
__execFallbackCpuNestLoop(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map,
						  int depth,
						  uint64_t l_state, bool matched)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;

	// MEMO: GiST-Join uses KDS_FORMAT_HASH but for ItemPointers.
	//       So, we deal with this buffer as KDS_FORMAT_ROW.
	// Assert(kds_in->format == KDS_FORMAT_ROW);
	for (uint32_t index=l_state; index < kds_in->nitems; index++)
	{
		kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds_in, index);

		if (!tupitem)
			continue;
		ResetExprContext(econtext);
		/* load inner variable */
		if (istate->inner_load_src != NIL &&
			istate->inner_load_dst != NIL)
		{
			__execFallbackLoadVarsSlot(scan_slot,
									   istate->inner_load_src,
									   istate->inner_load_dst,
									   kds_in,
									   tupitem);
		}
		/* check JOIN-clause */
		if (istate->join_quals == NULL ||
			ExecQual(istate->join_quals, econtext))
		{
			if (istate->join_type != JOIN_ANTI &&
				(istate->other_quals == NULL ||
				 ExecQual(istate->other_quals, econtext)))
			{
				/* Ok, go to the next depth */
				__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
			}
			/* mark outer-join map, if any */
			if (oj_map)
				oj_map[index] = true;
			/* mark as 'matched' in this depth */
			matched = true;
			/* semi and anti joins: the first match decides */
			if (istate->join_type == JOIN_SEMI ||
				istate->join_type == JOIN_ANTI)
				break;
		}
	}

	/* LEFT OUTER JOIN handling */
	if (!matched && (istate->join_type == JOIN_LEFT ||
					 istate->join_type == JOIN_FULL ||
					 istate->join_type == JOIN_ANTI))
	{
		__execFallbackLoadVarsSlot(scan_slot,
								   istate->inner_load_src,
								   istate->inner_load_dst,
								   kds_in,
								   NULL);
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

static void
__execFallbackCpuHashJoin(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map,
						  int depth,
						  uint64_t l_state, bool matched)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
	kern_hashitem  *hitem;
	uint32_t		hash;
	ListCell	   *lc1, *lc2;

	Assert(kds_in->format == KDS_FORMAT_HASH);

	/*
	 * Compute that hash-value
	 */
	if (l_state == 0)
	{
		hash = 0xffffffffU;
		forboth (lc1, istate->hash_outer_keys,
				 lc2, istate->hash_outer_funcs)
		{
			ExprState   *h_key = lfirst(lc1);
			devtype_hashfunc_f h_func = lfirst(lc2);
			Datum		datum;
			bool		isnull;

			datum = ExecEvalExprSwitchContext(h_key, econtext, &isnull);
			hash = pg_hash_merge(hash, h_func(isnull, datum));
		}
		hash ^= 0xffffffffU;
		hitem = KDS_HASH_FIRST_ITEM(kds_in, hash);
	}
	else
	{
		/* restart from the hash-chain */
		Assert(l_state <  kds_in->length &&
			   l_state >= kds_in->length - kds_in->usage);
		hitem = (kern_hashitem *)((char *)kds_in + l_state);
		hash = hitem->t.hash;
	}

	/*
	 * walks on the hash-join-table
	 */
	while (hitem != NULL)
	{
		if (hitem->t.hash == hash)
		{
			if (istate->inner_load_src != NIL &&
				istate->inner_load_dst != NIL)
			{
				__execFallbackLoadVarsSlot(scan_slot,
										   istate->inner_load_src,
										   istate->inner_load_dst,
										   kds_in,
										   &hitem->t);
			}
			/* check JOIN-clause */
			if (istate->join_quals == NULL ||
				ExecQual(istate->join_quals, econtext))
			{
				if (istate->join_type != JOIN_ANTI &&
					(istate->other_quals == NULL ||
					 ExecQual(istate->other_quals, econtext)))
				{
					/* Ok, go to the next depth */
					__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
				}
				/* mark outer-join map, if any */
				if (oj_map)
				{
					uint32_t	rowid = KERN_TUPITEM_GET_ROWID(&hitem->t);
					assert(rowid < kds_in->nitems);
					oj_map[rowid] = true;
				}
				/* mark as 'matched' in this depth */
				matched = true;
				/* semi and anti joins: the first match decides */
				if (istate->join_type == JOIN_SEMI ||
					istate->join_type == JOIN_ANTI)
					break;
			}
		}
		hitem = KDS_HASH_NEXT_ITEM(kds_in, hitem->next);
	}

	/* LEFT OUTER JOIN handling */
	if (!matched && (istate->join_type == JOIN_LEFT ||
					 istate->join_type == JOIN_FULL ||
					 istate->join_type == JOIN_ANTI))
	{
		__execFallbackLoadVarsSlot(scan_slot,
								   istate->inner_load_src,
								   istate->inner_load_dst,
								   kds_in,
								   NULL);
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts,
							  int depth,
							  uint64_t l_state,
							  bool matched)
{
	if (depth > pts->num_inner_rels)
	{
		ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
		TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
		MinimalTuple	mtup;

		/* apply projection if any */
		if (pts->fallback_proj)
		{
			ListCell   *lc;
			int			dst = 0;

			foreach (lc, pts->fallback_proj)
			{
				ExprState  *state = lfirst(lc);
				Datum	datum;
				bool	isnull;

				if (state)
				{
					datum = ExecEvalExpr(state, econtext, &isnull);

					if (isnull)
					{
						scan_slot->tts_isnull[dst] = true;
						scan_slot->tts_values[dst] = 0;
					}
					else
					{
						scan_slot->tts_isnull[dst] = false;
						scan_slot->tts_values[dst] = datum;
					}
				}
				dst++;
			}
		}
		mtup = heap_form_minimal_tuple(scan_slot->tts_tupleDescriptor,
									   scan_slot->tts_values,
									   scan_slot->tts_isnull,
									   0);
		pgstromStoreFallbackTuple(pts, mtup);
		pfree(mtup);
	}
	else
	{
		kern_multirels	   *h_kmrels = pts->h_kmrels;
		kern_data_store	   *kds_in;
		bool			   *oj_map;

		kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
		oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);
		if (h_kmrels->chunks[depth-1].is_nestloop ||
			h_kmrels->chunks[depth-1].gist_offset != 0)
		{
			if (pts->inners[depth-1].gist_clause_failed)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("%s: the CPU cannot run the GiST join at depth %d (its index clause has no CPU form)",
								pts->css.methods->CustomName, depth)));
			__execFallbackCpuNestLoop(pts, kds_in, oj_map, depth, l_state, matched);
		}
		else
		{
			__execFallbackCpuHashJoin(pts, kds_in, oj_map, depth, l_state, matched);
		}
	}
}

/*
 * execCpuFallbackOneTuple
 */
static void
execCpuFallbackOneTuple(pgstromTaskState *pts,
						int depth,
						uint64_t l_state,
						bool matched)
{
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;

	/*
	 * scan_slot must be set on the ecxt_scantuple prior to
	 * any ExecQual() because Var node is modified to reference
	 * INDEX_VAR.
	 */
	econtext->ecxt_scantuple = scan_slot;

	/* check WHERE-clause if any */
	if (depth == 0)
	{
		Assert(l_state == 0 && !matched);
		econtext->ecxt_scantuple = scan_slot;
		/* the quals refer to computed entries of the scan tuple (an expression shared with an aggregate argument) */
		if (pts->base_quals && pts->fallback_pre_expr != NIL)
		{
			ListCell   *lc1, *lc2;

			ResetExprContext(econtext);
			if (pts->fallback_pre_cxt)
				MemoryContextReset(pts->fallback_pre_cxt);
			forboth (lc1, pts->fallback_pre_dst,
					 lc2, pts->fallback_pre_expr)
			{
				int			dst = lfirst_int(lc1) - 1;
				bool		isnull;
				Datum		datum = ExecEvalExpr((ExprState *) lfirst(lc2), econtext, &isnull);

				if (!isnull)
				{
					Form_pg_attribute attr = TupleDescAttr(scan_slot->tts_tupleDescriptor, dst);

					/* the quals reset the expression context: keep a by-reference value in a context of our own */
					if (!attr->attbyval)
					{
						MemoryContext oldcxt;

						if (!pts->fallback_pre_cxt)
							pts->fallback_pre_cxt = AllocSetContextCreate(pts->css.ss.ps.state->es_query_cxt,
																		  "CPU fallback computed values",
																		  ALLOCSET_SMALL_SIZES);
						oldcxt = MemoryContextSwitchTo(pts->fallback_pre_cxt);
						datum = datumCopy(datum, attr->attbyval, attr->attlen);
						MemoryContextSwitchTo(oldcxt);
					}
				}
				scan_slot->tts_isnull[dst] = isnull;
				scan_slot->tts_values[dst] = isnull ? 0 : datum;
			}
		}
		if (pts->base_quals)
		{
			ResetExprContext(econtext);
			if (!ExecQual(pts->base_quals, econtext))
				return;
		}
		depth++;
	}
	/* Run JOIN, if any */
	__execFallbackCpuJoinOneDepth(pts, depth, l_state, matched);
}

/*
 * execCpuFallbackBaseTuple
 */
void
execCpuFallbackBaseTuple(pgstromTaskState *pts,
						 HeapTuple base_tuple)
{
	TupleTableSlot *base_slot = pts->base_slot;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
	ListCell	   *lc1, *lc2;

	/* a test tuple of EvalPlanQual may come without its table OID */
	if (!OidIsValid(base_tuple->t_tableOid))
		base_tuple->t_tableOid = RelationGetRelid(pts->css.ss.ss_currentRelation);
	ExecForceStoreHeapTuple(base_tuple, base_slot, false);
	slot_getallattrs(base_slot);
	ExecStoreAllNullTuple(scan_slot);
	forboth (lc1, pts->fallback_load_src,
			 lc2, pts->fallback_load_dst)
	{
		int		srcno = lfirst_int(lc1);
		int		dst = lfirst_int(lc2) - 1;

		if (srcno > 0)
		{
			scan_slot->tts_isnull[dst] = base_slot->tts_isnull[srcno - 1];
			scan_slot->tts_values[dst] = base_slot->tts_values[srcno - 1];
		}
		else if (srcno < 0)
		{
			/*
			 * A system column: the ctid that UPDATE and DELETE carry through the scan, tableoid, xmin, ... The old code
			 * subtracted 1 from the (negative) attribute number and read outside the slot's arrays, so a recheck tuple
			 * of a concurrently updated row (EvalPlanQual) made the backend crash in heap_form_minimal_tuple.
			 * The datum of ctid points into base_tuple, which outlives the minimal tuple formed below.
			 */
			bool		isnull;
			Datum		datum = heap_getsysattr(base_tuple, srcno,
												RelationGetDescr(pts->css.ss.ss_currentRelation),
												&isnull);

			scan_slot->tts_isnull[dst] = isnull;
			scan_slot->tts_values[dst] = datum;
		}
		/* srcno == 0 (a whole-row reference) stays NULL, as before */
	}
	if (pts->ps_state)
	{
		pgstromSharedState *ps_state = pts->ps_state;
		pg_atomic_fetch_add_u64(&ps_state->fallback_nitems, 1);
	}
	execCpuFallbackOneTuple(pts, 0, 0, false);
}

/*
 * execCpuFallbackOneChunk
 */
void
execCpuFallbackOneChunk(pgstromTaskState *pts)
{
	kern_data_store	   *kds = pts->curr_kds;

	if (kds->format == KDS_FORMAT_FALLBACK)
	{
		TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;

		elog(pgstrom_cpu_fallback_elevel, "%s: CPU fallback %u tuples (%s)",
			 pts->css.methods->CustomName, kds->nitems, format_bytesz(kds->usage));
		for (uint32_t i=0; i < kds->nitems; i++)
		{
			kern_fallbackitem *fb_item = KDS_GET_FALLBACK_ITEM(kds, i);

			ExecForceStoreMinimalTuple((MinimalTuple)&fb_item->t,
									   scan_slot, false);
			slot_getallattrs(scan_slot);
			if (pts->ps_state)
			{
				pgstromSharedState *ps_state = pts->ps_state;
				int		depth = fb_item->depth;

				if (depth == 0)
					pg_atomic_fetch_add_u64(&ps_state->fallback_nitems, 1);
				else if (depth <= pts->num_inner_rels)
					pg_atomic_fetch_add_u64(&pts->inners[depth-1].dsm->fallback_nitems, 1);
			}
			execCpuFallbackOneTuple(pts,
									fb_item->depth,
									fb_item->l_state,
									fb_item->matched);
		}
	}
}

/*
 * execCpuRecheckOneChunk
 *
 * A KDS_FORMAT_RECHECK chunk of the Vulkan service: rows of the base relation that the GPU could not
 * decide (a compressed or external string, a value or geometry the kernel cannot evaluate, ...).
 * The CPU evaluates them from the start (quals, joins, projection) like any other fallback tuple.
 * Items (kern_recheckitem) are of two kinds:
 *  - RECHECK_KIND_HEAP: the heap tuple itself. A GPU-Cache row arrives the same way (the service
 *    builds the heap tuple from the cache slot), so nothing here depends on visibility.
 *  - RECHECK_KIND_ARROW: (scan_relidx, record batch, row) of an Arrow_Fdw / Parquet scan; the row
 *    is decoded here from the file (pgstromArrowFdwRecheckTuple, which keeps the last batch).
 * Both end in execCpuFallbackBaseTuple.
 *
 * Risk: the base slot and fallback_load_src describe pts->css.ss.ss_currentRelation. For an Arrow
 * table of another scan relation (several scan_rels) the tuple is built with that relation's
 * descriptor, so its column order must be that of ss_currentRelation; the heap kind has the same
 * assumption. A different number of columns is refused below.
 */
void
execCpuRecheckOneChunk(pgstromTaskState *pts)
{
	kern_data_store *kds = pts->curr_kds;
	Relation		rel = pts->css.ss.ss_currentRelation;
	char		   *pos = KDS_BODY_ADDR(kds);
	MemoryContext	row_mcxt = NULL;

	Assert(kds->format == KDS_FORMAT_RECHECK);
	elog(DEBUG1, "%s: CPU re-evaluation of %u rows the GPU could not decide",
		 pts->css.methods->CustomName, kds->nitems);
	for (uint32_t i=0; i < kds->nitems; i++)
	{
		kern_recheckitem *item = (kern_recheckitem *)pos;

		if (item->kind == RECHECK_KIND_HEAP)
		{
			HeapTupleData	htup;

			memset(&htup, 0, sizeof(htup));
			htup.t_len = item->t_len;
			ItemPointerSet(&htup.t_self, item->blkno, item->offnum);
			htup.t_tableOid = RelationGetRelid(rel);
			htup.t_data = (HeapTupleHeader) item->t_data;
			execCpuFallbackBaseTuple(pts, &htup);
		}
		else if (item->kind == RECHECK_KIND_ARROW)
		{
			uint32_t	scan_relidx = item->offnum;
			pgstromTaskScanState *ptss;
			MemoryContext oldcxt;
			HeapTuple	tuple;

			if (scan_relidx >= pts->num_scan_rels)
				elog(ERROR, "%s: CPU re-evaluation names scan relation %u of %d",
					 pts->css.methods->CustomName, scan_relidx, pts->num_scan_rels);
			ptss = &pts->scan_rels[scan_relidx];
			if (!ptss->arrow_state)
				elog(ERROR, "%s: CPU re-evaluation of an Arrow row, but scan relation %u (%s) is not an Arrow_Fdw table",
					 pts->css.methods->CustomName, scan_relidx,
					 RelationGetRelationName(ptss->scan_rel));
			if (RelationGetDescr(ptss->scan_rel)->natts != RelationGetDescr(rel)->natts)
				elog(ERROR, "%s: CPU re-evaluation of '%s' needs the columns of '%s' in the same order",
					 pts->css.methods->CustomName,
					 RelationGetRelationName(ptss->scan_rel),
					 RelationGetRelationName(rel));
			/* the decoded row is garbage once the fallback tuple is stored: reset per row */
			if (!row_mcxt)
				row_mcxt = AllocSetContextCreate(CurrentMemoryContext,
												 "Arrow recheck row",
												 ALLOCSET_DEFAULT_SIZES);
			MemoryContextReset(row_mcxt);
			oldcxt = MemoryContextSwitchTo(row_mcxt);
			tuple = pgstromArrowFdwRecheckTuple(ptss->arrow_state,
												ptss->scan_rel,
												item->blkno,	/* rb_flat */
												item->arrow_row);
			execCpuFallbackBaseTuple(pts, tuple);
			MemoryContextSwitchTo(oldcxt);
		}
		else
			elog(ERROR, "%s: unknown kind %u of a CPU re-evaluation item",
				 pts->css.methods->CustomName, (unsigned int) item->kind);
		pos += MAXALIGN(offsetof(kern_recheckitem, t_data) + item->t_len);
	}
	if (row_mcxt)
		MemoryContextDelete(row_mcxt);
}

/*
 * ExecFallbackCpuJoinRightOuter
 */
static void
__execFallbackCpuJoinRightOuterOneDepth(pgstromTaskState *pts, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext		   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot	   *fallback_slot = pts->css.ss.ss_ScanTupleSlot;
	kern_multirels	   *h_kmrels = pts->h_kmrels;
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);

	Assert(oj_map != NULL);

	ExecStoreAllNullTuple(fallback_slot);
	econtext->ecxt_scantuple = fallback_slot;
	for (uint32_t i=0; i < kds_in->nitems; i++)
	{
		if (oj_map[i])
			continue;
		if (istate->inner_load_src != NIL &&
			istate->inner_load_dst != NIL)
		{
			kern_tupitem   *titem = KDS_GET_TUPITEM(kds_in, i);

			if (!titem)
				continue;
			__execFallbackLoadVarsSlot(fallback_slot,
									   istate->inner_load_src,
									   istate->inner_load_dst,
									   kds_in,
									   titem);
		}
		if (istate->other_quals && !ExecQual(istate->other_quals, econtext))
			continue;
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

void
ExecFallbackCpuJoinRightOuter(pgstromTaskState *pts)
{
	for (int depth=1; depth <= pts->num_inner_rels; depth++)
	{
		JoinType	join_type = pts->inners[depth-1].join_type;

		if (join_type == JOIN_RIGHT || join_type == JOIN_FULL)
			__execFallbackCpuJoinRightOuterOneDepth(pts, depth);
	}
}

/* ----------------------------------------------------------------
 *
 * A14: runtime CPU fallback of a whole node
 *
 * When the GPU service refuses a program, fails a task, is not ready, or disappears, the node
 * continues on the CPU instead of failing the query:
 *
 *  - every task sent to the service is recorded in a journal (the heap blocks, the Arrow record batch
 *    or the GPU-Cache table it reads).  The replies of one connection arrive in the order of the
 *    commands, so a consumed reply acknowledges the oldest journal entry;
 *  - on a failure the session is closed and the CPU runs the entries without an acknowledged reply
 *    again, then claims the remaining work of the scan from the same shared counters the GPU path
 *    uses.  Every visible row goes through execCpuFallbackBaseTuple() (base quals, joins against the
 *    inner buffers in host memory, projection or partial-aggregate state), the code that also
 *    re-evaluates the rows the GPU cannot decide;
 *  - GpuPreAgg keeps its partial aggregates inside the session, so all of its journal entries are
 *    replayed and the recheck tuples the lost session would have returned are held back until Final
 *    (defer_fallback), so that nothing is emitted twice;
 *  - RIGHT / FULL joins set the same outer-join map flags (idempotent), and the unmatched inner rows
 *    are emitted by the last worker as with a working session.
 *
 * Not recoverable (the original error is raised): GPU-Sort (the parent Sort node was removed), pinned
 * inner buffers and SELECT INTO direct (the rows never reach the host), and pg_strom.gpu_error_fallback
 * = off.
 * ----------------------------------------------------------------
 */
bool	pgstrom_gpu_error_fallback = true;
char   *pgstrom_gpuserv_fault = NULL;
bool	pgstrom_debug_no_limits = false;
int		pgstrom_scan_chunk_blocks = 0;

void
pgstromJournalAppend(pgstromTaskState *pts, const XpuCommand *xcmd)
{
	MemoryContext oldcxt;
	pgstromCpuJob *job;
	const kern_data_store *kds = NULL;

	if (!pts->cpu_fallback_ok)
		return;
	oldcxt = MemoryContextSwitchTo(pts->css.ss.ps.state->es_query_cxt);
	if (pts->journal_tail >= pts->journal_nrooms)
	{
		if (pts->journal_head > 0 && (pts->xpu_task_flags & DEVTASK__PREAGG) == 0)
		{
			memmove(pts->journal, pts->journal + pts->journal_head,
					sizeof(pgstromCpuJob) * (pts->journal_tail - pts->journal_head));
			pts->journal_tail -= pts->journal_head;
			pts->cpujob_next = (pts->cpujob_next > pts->journal_head
								? pts->cpujob_next - pts->journal_head : 0);
			pts->journal_head = 0;
		}
		if (pts->journal_tail >= pts->journal_nrooms)
		{
			uint32_t	nrooms = Max(pts->journal_nrooms * 2, 64);

			if (pts->journal)
				pts->journal = repalloc_huge(pts->journal, sizeof(pgstromCpuJob) * nrooms);
			else
				pts->journal = palloc(sizeof(pgstromCpuJob) * nrooms);
			pts->journal_nrooms = nrooms;
		}
	}
	job = &pts->journal[pts->journal_tail++];
	memset(job, 0, sizeof(*job));
	if (xcmd->tag == XpuCommandTag__XpuTaskExecGpuCache)
	{
		job->kind = CPUJOB_HEAP_ALL;
		job->scan_relidx = xcmd->u.gc_task.t.scan_relidx;
	}
	else
	{
		job->scan_relidx = xcmd->u.task.scan_relidx;
		if (xcmd->u.task.kds_src_offset != 0)
			kds = (const kern_data_store *)
				((const char *) xcmd + xcmd->u.task.kds_src_offset);
		if (kds && kds->format == KDS_FORMAT_BLOCK)
		{
			job->kind = CPUJOB_HEAP_BLOCKS;
			job->nblocks = kds->nitems;
			job->blocks = palloc(sizeof(BlockNumber) * Max(kds->nitems, 1));
			for (uint32_t i = 0; i < kds->nitems; i++)
				job->blocks[i] = KDS_BLOCK_BLCKNR(kds, i);
		}
		else if (kds && (kds->format == KDS_FORMAT_ARROW ||
						 kds->format == KDS_FORMAT_PARQUET))
		{
			job->kind = CPUJOB_ARROW_BATCH;
			job->rb_flat = xcmd->u.task.arrow_rb_flat;
		}
		/* kind 0: a task the CPU cannot repeat; replaying it raises an error */
	}
	MemoryContextSwitchTo(oldcxt);
}

/* a reply was consumed: the oldest task of the journal is done */
void
pgstromJournalAck(pgstromTaskState *pts)
{
	if (!pts->cpu_fallback_ok ||
		pts->journal_head >= pts->journal_tail ||
		(pts->xpu_task_flags & DEVTASK__PREAGG) != 0)
		return;
	if (pts->journal[pts->journal_head].blocks)
		pfree(pts->journal[pts->journal_head].blocks);
	pts->journal[pts->journal_head].blocks = NULL;
	pts->journal_head++;
	if (pts->journal_head == pts->journal_tail)
	{
		pts->journal_head = pts->journal_tail = 0;
		pts->cpujob_next = 0;
	}
}

void
pgstromCpuModeBegin(pgstromTaskState *pts, const char *reason)
{
	if (pts->conn)
	{
		xpuClientCloseSession(pts->conn);
		pts->conn = NULL;
	}
	elog(LOG, "%s: GPU execution failed (%s); the CPU runs the rest of the node",
		 pts->css.methods->CustomName, reason);
	pts->cpu_mode = true;
	pts->defer_fallback = false;
	pts->drain_fallback = false;
	pts->cpujob_active = false;
	if ((pts->xpu_task_flags & DEVTASK__PREAGG) != 0)
	{
		/* the partial aggregates of the session are lost: repeat every task, drop the held-back tuples */
		pts->cpujob_next = 0;
		pts->fallback_index = 0;
		pts->fallback_nitems = 0;
		pts->fallback_usage = 0;
	}
	else
		pts->cpujob_next = pts->journal_head;
	if (pts->ps_state)
		pg_atomic_fetch_add_u32(&pts->ps_state->cpu_mode_workers, 1);
}

void
pgstromCpuModeReset(pgstromTaskState *pts)
{
	if (pts->cpujob_scan)
	{
		table_endscan(pts->cpujob_scan);
		pts->cpujob_scan = NULL;
	}
	pts->cpu_mode = false;
	pts->cpu_final_done = false;
	pts->cpujob_active = false;
	pts->cpujob_next = 0;
	pts->journal_head = pts->journal_tail = 0;
	pts->drain_fallback = false;
	pts->final_is_last = false;
	pts->fallback_index = pts->fallback_nitems = pts->fallback_usage = 0;
}

/* CPU mode: take the next unit of work (a replayed journal entry, then fresh work of the scan) */
static bool
__cpuModeLoadJob(pgstromTaskState *pts)
{
	pgstromCpuJob *job = &pts->cpujob;

	memset(job, 0, sizeof(*job));
	if (pts->cpujob_next < pts->journal_tail)
	{
		*job = pts->journal[pts->cpujob_next++];
		if (job->kind == 0)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("%s: the GPU service failed and the CPU cannot repeat this task",
							pts->css.methods->CustomName)));
		goto found;
	}
	while (pts->curr_scan_rel < pts->num_scan_rels)
	{
		pgstromTaskScanState *ptss = &pts->scan_rels[pts->curr_scan_rel];

		job->scan_relidx = ptss->scan_relidx;
		if (ptss->gpucache_desc)
		{
			uint32_t	count;

			if (!ptss->gpucache_scan_count)
				ptss->gpucache_scan_count = &ptss->__gpucache_count_data;
			count = pg_atomic_fetch_add_u32(ptss->gpucache_scan_count, 1);
			if (count == 0)
			{
				job->kind = CPUJOB_HEAP_ALL;
				goto found;
			}
			pts->curr_scan_rel++;
		}
		else if (ptss->arrow_state)
		{
			uint32_t	rb_flat;

			if (pgstromArrowFdwClaimBatch(ptss->arrow_state, &rb_flat))
			{
				job->kind = CPUJOB_ARROW_BATCH;
				job->rb_flat = rb_flat;
				goto found;
			}
			pts->curr_scan_rel++;
		}
		else
		{
			uint32_t	n = 0;

			for (;;)
			{
				if (ptss->heap_block_pos < ptss->heap_block_end)
				{
					job->blocks = MemoryContextAlloc(pts->css.ss.ps.state->es_query_cxt,
													 sizeof(BlockNumber) * 64);
					while (ptss->heap_block_pos < ptss->heap_block_end && n < 64)
						job->blocks[n++] = (BlockNumber) ptss->heap_block_pos++;
					job->kind = CPUJOB_HEAP_BLOCKS;
					job->nblocks = n;
					goto found;
				}
				if (pts->br_state)
				{
					if (!pgstromBrinIndexNextChunk(pts, ptss))
					{
						pts->curr_scan_rel++;
						break;
					}
				}
				else
				{
					uint64_t	pos = pg_atomic_fetch_add_u64(&ptss->dsm->scan_block_count, 64);

					if (pos >= ptss->dsm->scan_block_nums)
					{
						pts->curr_scan_rel++;
						break;
					}
					ptss->heap_block_pos = pos;
					ptss->heap_block_end = Min(pos + 64, ptss->dsm->scan_block_nums);
				}
			}
		}
	}
	return false;
found:
	pts->cpujob_active = true;
	pts->cpujob_pos = 0;
	pts->cpujob_nrows = 0;
	if (pts->ps_state)
		pg_atomic_fetch_add_u64(&pts->ps_state->cpu_mode_chunks, 1);
	return true;
}

static void
__cpuJobCheckRel(pgstromTaskState *pts, pgstromTaskScanState *ptss)
{
	if (RelationGetDescr(ptss->scan_rel)->natts !=
		RelationGetDescr(pts->css.ss.ss_currentRelation)->natts)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s: the GPU service failed, and the CPU cannot continue because \"%s\" has other columns than \"%s\"",
						pts->css.methods->CustomName,
						RelationGetRelationName(ptss->scan_rel),
						RelationGetRelationName(pts->css.ss.ss_currentRelation))));
}

/* evaluate the visible rows of one heap block (a copy is taken, nothing runs under the buffer lock) */
static void
__cpuJobHeapBlock(pgstromTaskState *pts, pgstromTaskScanState *ptss, BlockNumber blkno)
{
	EState	   *estate = pts->css.ss.ps.state;
	Relation	relation = ptss->scan_rel;
	Snapshot	snapshot = estate->es_snapshot;
	Buffer		buffer;
	Page		page;
	OffsetNumber lines, lineoff;
	bool		check_visibility;
	HeapTupleData *htups;
	int			nhtups = 0;

	if (!pts->hscan_strategy)
	{
		MemoryContext oldcxt = MemoryContextSwitchTo(estate->es_query_cxt);

		pts->hscan_strategy = GetAccessStrategy(BAS_BULKREAD);
		MemoryContextSwitchTo(oldcxt);
	}
	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, blkno, RBM_NORMAL,
								pts->hscan_strategy);
	heap_page_prune_opt(relation, buffer, &ptss->heap_vm_buffer, true);
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buffer);
	lines = PageGetMaxOffsetNumber(page);
	check_visibility = (!PageIsAllVisible(page) || snapshot->takenDuringRecovery);
	htups = palloc(sizeof(HeapTupleData) * Max(lines, 1));
	for (lineoff = FirstOffsetNumber; lineoff <= lines; lineoff++)
	{
		ItemId		lpp = PageGetItemId(page, lineoff);
		HeapTupleData htup;
		bool		valid = true;

		if (!ItemIdIsNormal(lpp))
			continue;
		htup.t_tableOid = RelationGetRelid(relation);
		htup.t_data = (HeapTupleHeader) PageGetItem(page, lpp);
		htup.t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&htup.t_self, blkno, lineoff);
		if (check_visibility)
		{
			valid = HeapTupleSatisfiesVisibility(&htup, snapshot, buffer);
			HeapCheckForSerializableConflictOut(valid, relation, &htup,
												buffer, snapshot);
		}
		if (!valid)
			continue;
		htups[nhtups] = htup;
		htups[nhtups].t_data = palloc(htup.t_len);
		memcpy(htups[nhtups].t_data, htup.t_data, htup.t_len);
		nhtups++;
	}
	UnlockReleaseBuffer(buffer);
	for (int i = 0; i < nhtups; i++)
		execCpuFallbackBaseTuple(pts, &htups[i]);
}

/*
 * pgstromCpuModeStep - does one unit of CPU work (a heap block, up to 1000 rows of a record batch) and
 * leaves the resulting tuples in the fallback buffer.  False: nothing is left.
 */
bool
pgstromCpuModeStep(pgstromTaskState *pts)
{
	EState	   *estate = pts->css.ss.ps.state;
	MemoryContext step_cxt, oldcxt;
	pgstromCpuJob *job;
	pgstromTaskScanState *ptss;

	CHECK_FOR_INTERRUPTS();
	if (!pts->cpujob_active && !__cpuModeLoadJob(pts))
		return false;
	job = &pts->cpujob;
	if (job->scan_relidx < 0 || job->scan_relidx >= pts->num_scan_rels)
		elog(ERROR, "%s: CPU fallback names scan relation %d of %d",
			 pts->css.methods->CustomName, job->scan_relidx, pts->num_scan_rels);
	ptss = &pts->scan_rels[job->scan_relidx];
	__cpuJobCheckRel(pts, ptss);

	step_cxt = AllocSetContextCreate(estate->es_query_cxt,
									 "CPU fallback step",
									 ALLOCSET_DEFAULT_SIZES);
	oldcxt = MemoryContextSwitchTo(step_cxt);
	switch (job->kind)
	{
		case CPUJOB_HEAP_BLOCKS:
			__cpuJobHeapBlock(pts, ptss, job->blocks[pts->cpujob_pos++]);
			if (pts->cpujob_pos >= job->nblocks)
				pts->cpujob_active = false;
			break;

		case CPUJOB_HEAP_ALL:
			if (pts->cpujob_pos == 0)
				job->nblocks = RelationGetNumberOfBlocks(ptss->scan_rel);
			__cpuJobHeapBlock(pts, ptss, pts->cpujob_pos++);
			if (pts->cpujob_pos >= job->nblocks)
				pts->cpujob_active = false;
			break;

		case CPUJOB_ARROW_BATCH:
			{
				uint32_t	last;

				if (!ptss->arrow_state)
					elog(ERROR, "%s: CPU fallback of an Arrow task, but \"%s\" is not an Arrow_Fdw table",
						 pts->css.methods->CustomName,
						 RelationGetRelationName(ptss->scan_rel));
				if (pts->cpujob_pos == 0)
					pts->cpujob_nrows = pgstromArrowFdwBatchNrows(ptss->arrow_state,
																  ptss->scan_rel,
																  job->rb_flat);
				last = Min(pts->cpujob_pos + 1000, pts->cpujob_nrows);
				while (pts->cpujob_pos < last)
				{
					HeapTuple	tuple;

					/* the decoded row lives in the step context, which is deleted below */
					tuple = pgstromArrowFdwRecheckTuple(ptss->arrow_state,
														ptss->scan_rel,
														job->rb_flat,
														pts->cpujob_pos++);
					execCpuFallbackBaseTuple(pts, tuple);
				}
				if (pts->cpujob_pos >= pts->cpujob_nrows)
					pts->cpujob_active = false;
			}
			break;

		default:
			elog(ERROR, "%s: unknown CPU fallback job %d",
				 pts->css.methods->CustomName, job->kind);
	}
	MemoryContextSwitchTo(oldcxt);
	MemoryContextDelete(step_cxt);
	return true;
}
