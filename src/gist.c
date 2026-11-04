/*
 * gist.c
 *
 * Planner support for a join whose inner side is read through a GiST index (A13).
 * Slice 1: "inner.geom && ST_MakePoint(x, y)" against a gist_geometry_ops_2d index, where x and y are
 * computed from the outer relation. The service walks the index pages for each outer row and joins the
 * inner rows the leaf entries name (see src/gpuserv/kernels_vm/join_walk.inc).
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "catalog/pg_am.h"
#include "access/htup_details.h"
#include "catalog/pg_opfamily.h"
#include "utils/syscache.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"

/* the index must cover fewer blocks than a walker cursor can name (scan_prog.c / join_walk.inc) */
#define GIST_JOIN_MAX_BLOCKS	((1U << 21) - 2)

/* is the operator family the PostGIS 2D opclass of geometry? */
static bool
__gist_opfamily_is_2d(Oid opfamily)
{
	HeapTuple	htup = SearchSysCache1(OPFAMILYOID, ObjectIdGetDatum(opfamily));
	bool		result;

	if (!HeapTupleIsValid(htup))
		return false;
	result = (strcmp(NameStr(((Form_pg_opfamily) GETSTRUCT(htup))->opfname),
					 "gist_geometry_ops_2d") == 0);
	ReleaseSysCache(htup);
	return result;
}

/*
 * __gist_makepoint_args
 *
 * st_makepoint(x, y) of two float8 arguments from the postgis extension: returns the (x, y) pair
 */
static bool
__gist_makepoint_args(Node *node, Expr **p_x, Expr **p_y)
{
	FuncExpr   *func;
	char	   *extname;
	devtype_info *dtype;

	if (!node || !IsA(node, FuncExpr))
		return false;
	func = (FuncExpr *) node;
	if (list_length(func->args) != 2 ||
		exprType(linitial(func->args)) != FLOAT8OID ||
		exprType(lsecond(func->args)) != FLOAT8OID)
		return false;
	if (strcmp(get_func_name(func->funcid), "st_makepoint") != 0)
		return false;
	extname = get_func_extension_name(func->funcid);
	if (!extname || strcmp(extname, "postgis") != 0)
		return false;
	dtype = pgstrom_devtype_lookup(func->funcresulttype);
	if (!dtype || strcmp(dtype->type_name, "geometry") != 0)
		return false;
	*p_x = linitial(func->args);
	*p_y = lsecond(func->args);
	return true;
}

/*
 * __gist_match_clause
 *
 * "geom && st_makepoint(x, y)" in either argument order, geom being the key of the index column
 */
static bool
__gist_match_clause(PlannerInfo *root,
					RelOptInfo *outer_rel,
					RelOptInfo *inner_rel,
					IndexOptInfo *index,
					int indexcol,
					RestrictInfo *rinfo,
					Expr **p_x, Expr **p_y, Expr **p_clause)
{
	OpExpr	   *op;
	Node	   *left, *right, *keyarg, *ptarg;
	Relids		relids;

	if (rinfo->pseudoconstant || !rinfo->clause || !IsA(rinfo->clause, OpExpr))
		return false;
	op = (OpExpr *) rinfo->clause;
	if (list_length(op->args) != 2)
		return false;
	if (strcmp(get_opname(op->opno), "&&") != 0 ||
		!op_in_opfamily(op->opno, index->opfamily[indexcol]))
		return false;
	left = linitial(op->args);
	right = lsecond(op->args);
	if (match_index_to_operand(left, indexcol, index))
	{
		keyarg = left;
		ptarg = right;
	}
	else if (match_index_to_operand(right, indexcol, index))
	{
		keyarg = right;
		ptarg = left;
	}
	else
		return false;
	/* the key is the index column itself (a plain column of the inner relation) */
	if (!IsA(keyarg, Var) || ((Var *) keyarg)->varno != inner_rel->relid)
		return false;
	if (!__gist_makepoint_args(ptarg, p_x, p_y))
		return false;
	/* the point depends on the outer relation only */
	relids = pull_varnos(root, ptarg);
	if (!bms_is_subset(relids, outer_rel->relids))
		return false;
	if (contain_volatile_functions(ptarg))
		return false;
	*p_clause = (Expr *) op;
	return true;
}

/*
 * pgstromTryFindGistIndex
 *
 * Looks for a GiST index of the inner relation that one of the join clauses can use. On success it sets
 * the gist_* members and the outer keys (x, y) of pp_inner, returns the clause in *p_rinfo (the caller
 * leaves it out of the device join quals: the walk answers it) and returns the inner path, extended by the
 * ctid column the rows are matched with. NULL: no suitable index.
 */
Path *
pgstromTryFindGistIndex(PlannerInfo *root,
						RelOptInfo *outer_rel,
						Path *inner_path,
						List *restrict_clauses,
						uint32_t xpu_task_flags,
						Index base_scan_relid,
						List *inner_target_list,
						pgstromPlanInnerInfo *pp_inner,
						RestrictInfo **p_rinfo)
{
	RelOptInfo *inner_rel = inner_path->parent;
	PathTarget *inner_target = inner_path->pathtarget;
	ListCell   *lc;
	int			gist_ctid_resno = -1;

	Assert(pp_inner->hash_outer_keys == NIL &&
		   pp_inner->hash_inner_keys == NIL);
	/* a plain base relation read by PostgreSQL's own scan; no custom, partitioned or inherited inner */
	if (!IS_SIMPLE_REL(inner_rel) || inner_rel->rtekind != RTE_RELATION ||
		inner_rel->reloptkind != RELOPT_BASEREL)
		return NULL;
	if (inner_rel->part_scheme != NULL)
		return NULL;
	if (inner_path->param_info != NULL)
		return NULL;
	/*
	 * The inner rows are read by PostgreSQL's own sequential scan (its quals run on the CPU): a GpuScan or a
	 * parallel path does not carry the ctid the leaf entries are matched with.
	 */
	if (inner_path->pathtype != T_SeqScan || inner_path->parallel_aware)
		inner_path = (Path *) create_seqscan_path(root, inner_rel, NULL, 0);
	inner_target = inner_path->pathtarget;

	foreach (lc, inner_rel->indexlist)
	{
		IndexOptInfo *index = (IndexOptInfo *) lfirst(lc);
		ListCell   *cell;

		if (index->relam != GIST_AM_OID || index->indpred != NIL ||
			index->nkeycolumns < 1 || index->pages == 0 ||
			index->pages > GIST_JOIN_MAX_BLOCKS)
			continue;
		/* gist_geometry_ops_2d on the first column (its key is a box2df) */
		if (!__gist_opfamily_is_2d(index->opfamily[0]))
			continue;
		foreach (cell, restrict_clauses)
		{
			RestrictInfo *rinfo = lfirst(cell);
			Expr	   *x, *y, *clause;

			if (!__gist_match_clause(root, outer_rel, inner_rel, index, 0, rinfo, &x, &y, &clause))
				continue;
			/* x and y run on the device (in the join program) */
			if (!pgstrom_xpu_expression(x, (xpu_task_flags & ~DEVTASK__MASK) | DEVTASK__JOIN,
										base_scan_relid, inner_target_list, NULL) ||
				!pgstrom_xpu_expression(y, (xpu_task_flags & ~DEVTASK__MASK) | DEVTASK__JOIN,
										base_scan_relid, inner_target_list, NULL))
				continue;

			/* the inner rows carry their ctid (the leaf entries name rows by it) */
			{
				int		resno = 1;

				foreach (cell, inner_target->exprs)
				{
					Var	   *var = lfirst(cell);

					if (IsA(var, Var) && var->varno == inner_rel->relid &&
						var->varattno == SelfItemPointerAttributeNumber)
					{
						gist_ctid_resno = resno;
						break;
					}
					resno++;
				}
			}
			if (gist_ctid_resno < 0)
			{
				PathTarget *path_target = copy_pathtarget(inner_path->pathtarget);

				path_target->exprs = lappend(path_target->exprs,
											 makeVar(inner_rel->relid,
													 SelfItemPointerAttributeNumber,
													 TIDOID, -1, InvalidOid, 0));
				gist_ctid_resno = list_length(path_target->exprs);
				inner_path = (Path *) create_projection_path(root, inner_rel, inner_path, path_target);
			}
			pp_inner->hash_outer_keys = list_make2(x, y);
			pp_inner->gist_index_oid = index->indexoid;
			pp_inner->gist_index_col = 1;
			pp_inner->gist_ctid_resno = gist_ctid_resno;
			pp_inner->gist_strategy = 1;			/* && */
			pp_inner->gist_func_oid = InvalidOid;
			pp_inner->gist_slot_id = -1;
			pp_inner->gist_clause = clause;
			pp_inner->gist_selectivity = clauselist_selectivity(root, list_make1(rinfo),
															   inner_rel->relid, JOIN_INNER, NULL);
			pp_inner->gist_npages = index->pages;
			pp_inner->gist_height = index->tree_height;
			*p_rinfo = rinfo;
			return inner_path;
		}
	}
	return NULL;
}
