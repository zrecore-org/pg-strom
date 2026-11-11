/*
 * wire.c
 *
 * Serves the backend <-> GPU service protocol (XpuCommand frames) for the exec'd
 * pgstrom-gpuserv. One poll()-driven thread; connections are one per backend session.
 *
 *   OpenSession  keep the kern_session_info, answer Success
 *   XpuTaskExec  (scan execution is added by the A2 scan milestone; for now an Error)
 *   XpuTaskFinal answer Success with final_plan_task
 *
 * The same frames the CUDA service spoke, taken from the real xpu_common.h, so the
 * backend code is unchanged. Not a PostgreSQL process: libc only.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include "xpu_common.h"
#include "wire.h"
#include "kprewarm.h"
#include "gcache.h"

#define WIRE_MAX_FRAME		((uint64_t)1 << 31)

struct wire_client
{
	int				fd;
	unsigned char  *rbuf;
	size_t			rlen;
	size_t			rcap;
	XpuCommand	   *session;			/* copy of the OpenSession frame */
	scan_prog		prog;				/* the session's scan program, built on the first task */
	agg_state		agg;				/* GpuPreAgg: the running partial aggregates */
	join_inner		join;				/* GpuJoin: the inner relation */
	int				closing;
	int				fault_ntasks;		/* tasks seen (fault injection) */
	scan_engine	   *eng;				/* for releasing the join inner buffer */
};

static void
wlog(const wire_server *ws, int always, const char *fmt, ...)
{
	va_list		ap;

	if (!always && !ws->trace)
		return;
	fprintf(stderr, "pgstrom-gpuserv[%d]: ", (int) getpid());
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	fflush(stderr);
}

/* ----------------------------------------------------------------
 * opcode names, for tracing
 * ----------------------------------------------------------------
 */
#define TYPE_OPCODE(NAME,a,b)				[TypeOpCode__##NAME] = #NAME,
static const char *const type_opcode_names[TypeOpCode__BuiltInMax] = {
#include "xpu_opcodes.h"
};
#undef TYPE_OPCODE

#define FUNC_OPCODE(a,b,c,NAME,d,e)			case FuncOpCode__##NAME: return #NAME;
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	case FuncOpCode__##NAME: return #NAME;
static const char *
func_opcode_name(unsigned int opcode)
{
	switch (opcode)
	{
		case FuncOpCode__ConstExpr:			return "ConstExpr";
		case FuncOpCode__ParamExpr:			return "ParamExpr";
		case FuncOpCode__VarExpr:			return "VarExpr";
		case FuncOpCode__BoolExpr_And:		return "BoolExpr_And";
		case FuncOpCode__BoolExpr_Or:		return "BoolExpr_Or";
		case FuncOpCode__BoolExpr_Not:		return "BoolExpr_Not";
		case FuncOpCode__NullTestExpr_IsNull:	return "NullTestExpr_IsNull";
		case FuncOpCode__NullTestExpr_IsNotNull: return "NullTestExpr_IsNotNull";
		case FuncOpCode__LoadVars:			return "LoadVars";
		case FuncOpCode__MoveVars:			return "MoveVars";
		case FuncOpCode__JoinQuals:			return "JoinQuals";
		case FuncOpCode__HashValue:			return "HashValue";
		case FuncOpCode__SaveExpr:			return "SaveExpr";
		case FuncOpCode__AggFuncs:			return "AggFuncs";
		case FuncOpCode__Projection:		return "Projection";
		case FuncOpCode__Packed:			return "Packed";
#include "xpu_opcodes.h"
		default:
			break;
	}
	return "?";
}
#undef FUNC_OPCODE
#undef DEVONLY_FUNC_OPCODE

static void
dump_kexp(const kern_expression *kexp, int depth)
{
	const char *tname = (kexp->exptype < TypeOpCode__BuiltInMax
						 ? type_opcode_names[kexp->exptype] : NULL);

	fprintf(stderr, "  %*s%s -> %s (nr_args=%u, len=%u)",
			depth * 2, "", func_opcode_name(kexp->opcode),
			tname ? tname : "?", kexp->nr_args, kexp->len);
	switch (kexp->opcode)
	{
		case FuncOpCode__VarExpr:
			fprintf(stderr, " var_offset=%d slot=%u",
					kexp->u.v.var_offset, kexp->u.v.var_slot_id);
			break;
		case FuncOpCode__LoadVars:
			fprintf(stderr, " depth=%d nitems=%d [", kexp->u.load.depth, kexp->u.load.nitems);
			for (int i = 0; i < kexp->u.load.nitems; i++)
				fprintf(stderr, "%sresno%d->slot%d", i ? " " : "",
						kexp->u.load.desc[i].vl_resno, kexp->u.load.desc[i].vl_slot_id);
			fputc(']', stderr);
			break;
		case FuncOpCode__Projection:
			fprintf(stderr, " nattrs=%d", kexp->u.proj.nattrs);
			break;
		case FuncOpCode__AggFuncs:
			fprintf(stderr, " nattrs=%d [", kexp->u.pagg.nattrs);
			for (int i = 0; i < kexp->u.pagg.nattrs; i++)
				fprintf(stderr, "%saction=%u arg0=%d arg1=%d filter=%d", i ? " | " : "",
						kexp->u.pagg.desc[i].action, kexp->u.pagg.desc[i].arg0_slot_id,
						kexp->u.pagg.desc[i].arg1_slot_id, kexp->u.pagg.desc[i].filter_slot_id);
			fputc(']', stderr);
			break;
		case FuncOpCode__SaveExpr:
			fprintf(stderr, " -> slot%u", kexp->u.save.sv_slot_id);
			break;
		default:
			break;
	}
	fputc('\n', stderr);

	if (kexp->opcode == FuncOpCode__Packed)
	{
		for (uint32_t i = 0; i < kexp->u.pack.npacked; i++)
		{
			if (kexp->u.pack.offset[i] == 0)
				fprintf(stderr, "  %*s(null)\n", (depth + 1) * 2, "");
			else
				dump_kexp((const kern_expression *)
						  ((const char *)kexp + kexp->u.pack.offset[i]), depth + 1);
		}
	}
	else if (kexp->nr_args > 0 && kexp->args_offset > 0)
	{
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);

		for (uint32_t i = 0; i < kexp->nr_args; i++, karg = KEXP_NEXT_ARG(karg))
			dump_kexp(karg, depth + 1);
	}
}

static void
dump_session(const wire_server *ws, const kern_session_info *s)
{
	static const struct { const char *label; size_t off; } codes[] = {
		{"load_vars",     offsetof(kern_session_info, xpucode_load_vars_packed)},
		{"move_vars",     offsetof(kern_session_info, xpucode_move_vars_packed)},
		{"scan_quals",    offsetof(kern_session_info, xpucode_scan_quals)},
		{"join_quals",    offsetof(kern_session_info, xpucode_join_quals_packed)},
		{"hash_values",   offsetof(kern_session_info, xpucode_hash_values_packed)},
		{"projection",    offsetof(kern_session_info, xpucode_projection)},
		{"groupby_keyhash", offsetof(kern_session_info, xpucode_groupby_keyhash)},
		{"groupby_keyload", offsetof(kern_session_info, xpucode_groupby_keyload)},
		{"groupby_keycomp", offsetof(kern_session_info, xpucode_groupby_keycomp)},
		{"groupby_actions", offsetof(kern_session_info, xpucode_groupby_actions)},
		{"gpusort_keydesc", offsetof(kern_session_info, xpucode_gpusort_keydesc)},
	};

	wlog(ws, 0, "session: plan_id=%llu task_flags=%08x kvars=%u/%u kvecs_ndims=%u nparams=%u",
		 (unsigned long long) s->query_plan_id, s->xpu_task_flags,
		 s->kcxt_kvars_nslots, s->kcxt_kvars_nrooms, s->kcxt_kvecs_ndims, s->nparams);
	for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++)
	{
		uint32_t	off = *(const uint32_t *)((const char *)s + codes[i].off);

		if (off == 0)
			continue;
		fprintf(stderr, " %s:\n", codes[i].label);
		dump_kexp((const kern_expression *)((const char *)s + off), 0);
	}
	if (s->kcxt_kvars_nslots > 0 && s->kcxt_kvars_defs > 0)
	{
		const kern_varslot_desc *vs = (const kern_varslot_desc *)
			((const char *)s + s->kcxt_kvars_defs);

		for (uint32_t i = 0; i < s->kcxt_kvars_nslots; i++)
			fprintf(stderr, " kvar slot%u: type=%s len=%d byval=%d align=%d offset=%d\n",
					i, vs[i].vs_type_code < TypeOpCode__BuiltInMax
					   ? type_opcode_names[vs[i].vs_type_code] : "?",
					vs[i].vs_typlen, vs[i].vs_typbyval, vs[i].vs_typalign, vs[i].vs_offset);
	}
	fflush(stderr);
}

/* ----------------------------------------------------------------
 * replies
 * ----------------------------------------------------------------
 */
static int
write_all(int fd, const void *buf, size_t len)
{
	const char *p = buf;

	while (len > 0)
	{
		ssize_t		n = write(fd, p, len);

		if (n > 0)
		{
			p += n;
			len -= n;
		}
		else if (n < 0 && (errno == EINTR || errno == EAGAIN))
			continue;
		else
			return -1;
	}
	return 0;
}


/* ----------------------------------------------------------------
 * fault injection (tests of the backend's CPU fallback)
 *
 * The backend sends the superuser GUC pg_strom.gpuserv_fault in the session.  Syntax:
 *   KIND[=N][,node=scan|join|preagg][,src=heap|arrow|gcache]
 * KIND is one of
 *   open     the OpenSession command is answered with an error
 *   refuse   the N-th task (default 1) is answered with a translation error
 *   lost     the N-th task is answered with the error of a lost device
 *   drop     the connection is closed when the N-th task arrives (a service that disappeared)
 *   exit     the whole service exits when the N-th task arrives (a killed service)
 *   final    the XpuTaskFinal command is answered with an error
 * node= / src= restrict it to sessions of that kind / tasks reading that source.
 * ----------------------------------------------------------------
 */
typedef struct
{
	char		kind[16];
	int			nth;
	int			node;			/* DEVTASK__* or 0 */
	int			src;			/* 1 heap, 2 arrow, 3 gcache, 0 any */
} fault_spec_t;

static int
fault_parse(const XpuCommand *session, fault_spec_t *fs)
{
	const kern_session_info *s = &session->u.session;
	const char *p;
	char		buf[128];
	char	   *tok, *save = NULL;

	memset(fs, 0, sizeof(*fs));
	if (s->fault_spec == 0 ||
		s->fault_spec >= session->length - offsetof(XpuCommand, u.session))
		return 0;
	p = (const char *) s + s->fault_spec;
	snprintf(buf, sizeof(buf), "%s", p);
	fs->nth = 1;
	for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
	{
		char	   *eq = strchr(tok, '=');

		if (eq)
			*eq++ = '\0';
		if (strcmp(tok, "node") == 0 && eq)
			fs->node = (strcmp(eq, "scan") == 0 ? DEVTASK__SCAN :
						strcmp(eq, "join") == 0 ? DEVTASK__JOIN :
						strcmp(eq, "preagg") == 0 ? DEVTASK__PREAGG : 0);
		else if (strcmp(tok, "src") == 0 && eq)
			fs->src = (strcmp(eq, "heap") == 0 ? 1 :
					   strcmp(eq, "arrow") == 0 ? 2 :
					   strcmp(eq, "gcache") == 0 ? 3 : 0);
		else
		{
			snprintf(fs->kind, sizeof(fs->kind), "%s", tok);
			if (eq)
				fs->nth = atoi(eq);
		}
	}
	return fs->kind[0] != '\0';
}

static int
fault_match(const XpuCommand *session, const fault_spec_t *fs, const XpuCommand *task)
{
	if (fs->node != 0 &&
		(session->u.session.xpu_task_flags & DEVTASK__MASK) != (uint32_t) fs->node)
		return 0;
	if (task && fs->src != 0)
	{
		int		src = 1;

		if (task->tag == XpuCommandTag__XpuTaskExecGpuCache)
			src = 3;
		else if (task->u.task.kds_src_offset != 0 &&
				 task->u.task.kds_src_offset + sizeof(kern_data_store) <= task->length)
		{
			const kern_data_store *kds = (const kern_data_store *)
				((const char *) task + task->u.task.kds_src_offset);

			if (kds->format == KDS_FORMAT_ARROW || kds->format == KDS_FORMAT_PARQUET)
				src = 2;
		}
		if (src != fs->src)
			return 0;
	}
	return 1;
}

static void
reply_error(wire_client *wc, const char *func, const char *fmt, ...)
{
	XpuCommand	resp;
	va_list		ap;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Error;
	resp.length = offsetof(XpuCommand, u.error) + sizeof(kern_errorbuf);
	resp.u.error.errcode = ERRCODE_DEVICE_ERROR;
	snprintf(resp.u.error.filename, sizeof(resp.u.error.filename), "gpuserv");
	snprintf(resp.u.error.funcname, sizeof(resp.u.error.funcname), "%s", func);
	va_start(ap, fmt);
	vsnprintf(resp.u.error.message, sizeof(resp.u.error.message), fmt, ap);
	va_end(ap);
	if (write_all(wc->fd, &resp, resp.length) != 0)
		wc->closing = 1;
}

static void
reply_success(wire_client *wc, int final_plan_task)
{
	XpuCommand	resp;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u.results.stats);
	resp.u.results.scan_relidx = -1;
	resp.u.results.final_plan_task = final_plan_task ? true : false;
	if (write_all(wc->fd, &resp, resp.length) != 0)
		wc->closing = 1;
}

/*
 * The reply to XpuTaskFinal of a GpuJoin: the tasks have merged what the RIGHT/FULL joins matched into
 * the backend's outer-join map; the backend now emits the inner rows without a partner (it does
 * nothing for a join without such a relation).
 */
static void
reply_final_join(wire_client *wc)
{
	XpuCommand	resp;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u.results.stats);
	resp.u.results.scan_relidx = -1;
	resp.u.results.final_plan_task = true;
	resp.u.results.right_outer_join = true;
	if (write_all(wc->fd, &resp, resp.length) != 0)
		wc->closing = 1;
}

static void
reply_task_stats(wire_client *wc, int32_t scan_relidx, uint32_t raw, uint32_t in, uint32_t vfs)
{
	XpuCommand	resp;

	memset(&resp, 0, sizeof(resp));
	resp.magic = XpuCommandMagicNumber;
	resp.tag = XpuCommandTag__Success;
	resp.length = offsetof(XpuCommand, u.results.stats);
	resp.u.results.scan_relidx = scan_relidx;
	resp.u.results.nitems_raw = raw;
	resp.u.results.nitems_in = in;
	resp.u.results.nitems_out = in;
	resp.u.results.npages_vfs_read = vfs;
	if (write_all(wc->fd, &resp, resp.length) != 0)
		wc->closing = 1;
}

/* ----------------------------------------------------------------
 * command handlers
 * ----------------------------------------------------------------
 */
static void
handle_open_session(wire_server *ws, wire_client *wc, const XpuCommand *xcmd)
{
	const kern_session_info *s = &xcmd->u.session;

	if (wc->session)
	{
		reply_error(wc, "OpenSession", "OpenSession is called twice");
		return;
	}
	if (xcmd->length < offsetof(XpuCommand, u.session) + offsetof(kern_session_info, poffset))
	{
		reply_error(wc, "OpenSession", "session information is too short");
		return;
	}
	wc->session = malloc(xcmd->length);
	if (!wc->session)
	{
		reply_error(wc, "OpenSession", "out of memory");
		return;
	}
	memcpy(wc->session, xcmd, xcmd->length);
	dump_session(ws, s);
	{
		fault_spec_t fs;

		if (fault_parse(wc->session, &fs) && strcmp(fs.kind, "open") == 0 &&
			fault_match(wc->session, &fs, NULL))
		{
			reply_error(wc, "OpenSession", "injected fault: the session is refused");
			return;
		}
	}
	reply_success(wc, 0);
}

/* does the task read an Arrow record batch (Arrow_Fdw, Parquet row group)? */
static int
task_is_arrow(const XpuCommand *xcmd)
{
	const kern_data_store *kds;

	if (xcmd->tag == XpuCommandTag__XpuTaskExecGpuCache)
		return 1;			/* a GPU-Cache is read as a record batch of row slots */

	if (!xcmd->u.task.kds_src_offset || xcmd->u.task.kds_src_offset + sizeof(kern_data_store) > xcmd->length)
		return 0;
	kds = (const kern_data_store *)((const char *) xcmd + xcmd->u.task.kds_src_offset);
	return kds->format == KDS_FORMAT_ARROW;
}

static void
handle_task_exec(wire_server *ws, wire_client *wc, const XpuCommand *xcmd)
{
	XpuCommand *resp;
	size_t		resp_len = 0;
	char		msg[256];

	if (!wc->session)
	{
		reply_error(wc, "XpuTaskExec", "no session is opened");
		return;
	}
	wlog(ws, 0, "task: scan_relidx=%d kds_src_offset=%u pathname=%u iovec=%u length=%llu",
		 xcmd->u.task.scan_relidx, xcmd->u.task.kds_src_offset,
		 xcmd->u.task.kds_src_pathname, xcmd->u.task.kds_src_iovec,
		 (unsigned long long) xcmd->length);
	gcache_cur_session = &wc->session->u.session;
	wc->fault_ntasks++;
	{
		fault_spec_t fs;

		if (fault_parse(wc->session, &fs) && fs.nth == wc->fault_ntasks &&
			fault_match(wc->session, &fs, xcmd))
		{
			if (strcmp(fs.kind, "refuse") == 0)
			{
				reply_error(wc, "XpuTaskExec", "injected fault: the expression needs more than 32 registers");
				return;
			}
			if (strcmp(fs.kind, "lost") == 0)
			{
				reply_error(wc, "XpuTaskExec", "GPU dispatch failed: kernel did not finish within 10000 ms (device marked lost)");
				return;
			}
			if (strcmp(fs.kind, "drop") == 0)
			{
				wlog(ws, 1, "injected fault: dropping the connection at task %d", wc->fault_ntasks);
				wc->closing = 1;
				return;
			}
			if (strcmp(fs.kind, "exit") == 0)
			{
				wlog(ws, 1, "injected fault: exiting at task %d", wc->fault_ntasks);
				_exit(1);
			}
		}
	}
	if ((wc->session->u.session.xpu_task_flags & DEVTASK__MASK) == DEVTASK__PREAGG)
	{
		uint64_t	raw0 = wc->agg.nitems_raw, in0 = wc->agg.nitems_in;
		uint32_t	vfs0 = wc->agg.npages_vfs_read;
		int			failed;

		if (wc->session->u.session.join_inner_handle != 0)
		{
			/* aggregation over a join: the join kernel aggregates the joined rows */
			if (!wc->join.ready &&
				scan_exec_join_open(&ws->scan, wc->session, &wc->join, msg, sizeof(msg)) != 0)
			{
				wlog(ws, 1, "task failed: %s", msg);
				reply_error(wc, "XpuTaskExec", "%s", msg);
				return;
			}
			if (task_is_arrow(xcmd))
				failed = scan_exec_arrow_joinagg_task(&ws->scan, wc->session, &wc->prog, &wc->join, &wc->agg,
													  xcmd, msg, sizeof(msg));
			else
				failed = scan_exec_joinagg_task(&ws->scan, wc->session, &wc->prog, &wc->join, &wc->agg,
												xcmd, msg, sizeof(msg));
		}
		else if (task_is_arrow(xcmd))
			failed = scan_exec_arrow_agg_task(&ws->scan, wc->session, &wc->prog, &wc->agg, xcmd, msg, sizeof(msg));
		else if (wc->session->u.session.xpucode_groupby_keyhash)
			failed = scan_exec_group_task(&ws->scan, wc->session, &wc->prog, &wc->agg, xcmd, msg, sizeof(msg));
		else
			failed = scan_exec_agg_task(&ws->scan, wc->session, &wc->prog, &wc->agg, xcmd, msg, sizeof(msg));
		if (failed != 0)
		{
			wlog(ws, 1, "task failed: %s", msg);
			reply_error(wc, "XpuTaskExec", "%s", msg);
			return;
		}
		reply_task_stats(wc, xcmd->u.task.scan_relidx,
						 (uint32_t)(wc->agg.nitems_raw - raw0), (uint32_t)(wc->agg.nitems_in - in0),
						 wc->agg.npages_vfs_read - vfs0);
		return;
	}
	if ((wc->session->u.session.xpu_task_flags & DEVTASK__MASK) == DEVTASK__JOIN)
	{
		if (!wc->join.ready &&
			scan_exec_join_open(&ws->scan, wc->session, &wc->join, msg, sizeof(msg)) != 0)
		{
			wlog(ws, 1, "task failed: %s", msg);
			reply_error(wc, "XpuTaskExec", "%s", msg);
			return;
		}
		if (task_is_arrow(xcmd))
			resp = scan_exec_arrow_join_task(&ws->scan, wc->session, &wc->prog, &wc->join, xcmd, &resp_len,
											 msg, sizeof(msg));
		else
			resp = scan_exec_join_task(&ws->scan, wc->session, &wc->prog, &wc->join, xcmd, &resp_len,
									   msg, sizeof(msg));
	}
	else
		resp = scan_exec_task(&ws->scan, wc->session, &wc->prog, xcmd, &resp_len, msg, sizeof(msg));
	if (!resp)
	{
		wlog(ws, 1, "task failed: %s", msg);
		reply_error(wc, "XpuTaskExec", "%s", msg);
		return;
	}
	if (write_all(wc->fd, resp, resp_len) != 0)
		wc->closing = 1;
	free(resp);
}

static void
handle_command(wire_server *ws, wire_client *wc, const XpuCommand *xcmd)
{
	/* A9 part 3: the kernel readiness gate uses this session's wait budget; the prewarm thread waits for a quiet service */
	kpw_task_begin();
	kpw_set_session_wait(wc->session ? wc->session->u.session.kernel_wait_ms : 0);
	switch (xcmd->tag)
	{
		case XpuCommandTag__OpenSession:
			handle_open_session(ws, wc, xcmd);
			break;
		case XpuCommandTag__XpuTaskExec:
		case XpuCommandTag__XpuTaskExecGpuCache:
			handle_task_exec(ws, wc, xcmd);
			break;
		case XpuCommandTag__GpuCacheOp:
			{
				size_t		len = 0;
				XpuCommand *resp = gcache_handle_op(xcmd, &len);

				if (!resp)
				{
					reply_error(wc, "GpuCacheOp", "out of memory");
					break;
				}
				if (write_all(wc->fd, resp, len) != 0)
					wc->closing = 1;
				free(resp);
			}
			break;
		case XpuCommandTag__XpuTaskFinal:
			if (wc->session)
			{
				fault_spec_t fs;

				if (fault_parse(wc->session, &fs) && strcmp(fs.kind, "final") == 0 &&
					fault_match(wc->session, &fs, NULL))
				{
					reply_error(wc, "XpuTaskFinal", "injected fault: the final step failed");
					break;
				}
			}
			if (wc->session &&
				(wc->session->u.session.xpu_task_flags & DEVTASK__MASK) == DEVTASK__PREAGG)
			{
				char		msg[256];
				size_t		len = 0;
				XpuCommand *resp;

				if (scan_group_flush(&wc->prog, &wc->agg, msg, sizeof(msg)) != 0)
				{
					wlog(ws, 1, "final failed: %s", msg);
					reply_error(wc, "XpuTaskFinal", "%s", msg);
					break;
				}
				resp = (wc->session->u.session.xpucode_gpusort_keydesc
								   ? scan_exec_agg_final_sorted(&ws->scan, wc->session, &wc->prog, &wc->agg, &len,
																msg, sizeof(msg))
								   : scan_exec_agg_final(wc->session, &wc->prog, &wc->agg, &len, msg, sizeof(msg)));

				if (!resp)
				{
					wlog(ws, 1, "final failed: %s", msg);
					reply_error(wc, "XpuTaskFinal", "%s", msg);
					break;
				}
				wlog(ws, 0, "final: reply of %zu bytes", len);
				/*
				 * The kernels of an aggregation over a RIGHT/FULL join flagged the matched inner rows in the
				 * backend's outer-join map; the backend emits the others after it has handled this reply
				 * (a join without such a relation: nothing to do).
				 */
				resp->u.results.right_outer_join = true;
				if (write_all(wc->fd, resp, len) != 0)
					wc->closing = 1;
				free(resp);
			}
			else if (wc->session &&
					 (wc->session->u.session.xpu_task_flags & DEVTASK__MASK) == DEVTASK__JOIN)
				reply_final_join(wc);
			else
				reply_success(wc, 1);
			break;
		default:
			reply_error(wc, "wire", "unknown command tag %u", xcmd->tag);
			break;
	}
	kpw_task_end();
}

/* ----------------------------------------------------------------
 * connection handling
 * ----------------------------------------------------------------
 */
static void
client_free(wire_client *wc)
{
	if (wc->fd >= 0)
		close(wc->fd);
	free(wc->rbuf);
	scan_prog_free(&wc->prog);
	agg_state_free(&wc->agg);
	join_inner_free(wc->eng, &wc->join);
	free(wc->session);
	free(wc);
}

static void
handle_ping(wire_server *ws, wire_client *wc)
{
	gpuserv_sock_pong pong;

	memset(&pong, 0, sizeof(pong));
	pong.magic = GPUSERV_SOCK_MAGIC;
	pong.pid = (int32_t) getpid();
	pong.generation = __atomic_load_n(&ws->ctl->generation, __ATOMIC_SEQ_CST);
	pong.device = ws->device;
	pong.tier = ws->ctl->devcaps[ws->device].tier;
	(void) write_all(wc->fd, &pong, sizeof(pong));
	wc->closing = 1;
}

/* consume the complete frames in the receive buffer */
static void
client_process(wire_server *ws, wire_client *wc)
{
	for (;;)
	{
		const XpuCommand *xcmd = (const XpuCommand *) wc->rbuf;
		uint32_t	magic;
		uint64_t	length;

		if (wc->closing || wc->rlen < sizeof(uint32_t))
			return;
		memcpy(&magic, wc->rbuf, sizeof(magic));
		if (magic == GPUSERV_SOCK_MAGIC)
		{
			if (wc->rlen >= sizeof(gpuserv_sock_req))
				handle_ping(ws, wc);
			return;
		}
		if (wc->rlen < offsetof(XpuCommand, u))
			return;
		if (xcmd->magic != XpuCommandMagicNumber ||
			xcmd->length < offsetof(XpuCommand, u) ||
			xcmd->length > WIRE_MAX_FRAME)
		{
			wlog(ws, 1, "invalid command frame (magic %08x, length %llu), closing the connection",
				 xcmd->magic, (unsigned long long) xcmd->length);
			wc->closing = 1;
			return;
		}
		length = xcmd->length;
		if (wc->rlen < length)
		{
			if (wc->rcap < length)
			{
				unsigned char *nb = realloc(wc->rbuf, length);

				if (!nb)
				{
					wc->closing = 1;
					return;
				}
				wc->rbuf = nb;
				wc->rcap = length;
			}
			return;
		}
		handle_command(ws, wc, (const XpuCommand *) wc->rbuf);
		memmove(wc->rbuf, wc->rbuf + length, wc->rlen - length);
		wc->rlen -= length;
	}
}

static void
client_read(wire_server *ws, wire_client *wc)
{
	ssize_t		n;

	if (wc->rcap - wc->rlen < 65536)
	{
		size_t	ncap = (wc->rcap ? wc->rcap * 2 : 262144);
		unsigned char *nb = realloc(wc->rbuf, ncap);

		if (!nb)
		{
			wc->closing = 1;
			return;
		}
		wc->rbuf = nb;
		wc->rcap = ncap;
	}
	n = read(wc->fd, wc->rbuf + wc->rlen, wc->rcap - wc->rlen);
	if (n > 0)
	{
		wc->rlen += n;
		client_process(ws, wc);
	}
	else if (n == 0 || (errno != EINTR && errno != EAGAIN))
		wc->closing = 1;
}

void
wire_server_init(wire_server *ws, gpuserv_control *ctl, gpu_vk_ctx *vk, uint32_t device)
{
	const char *trace = getenv("PGSTROM_GPUSERV_TRACE");

	memset(ws, 0, sizeof(*ws));
	ws->ctl = ctl;
	ws->vk = vk;
	ws->device = device;
	ws->trace = (trace && trace[0] && strcmp(trace, "0") != 0);
	scan_engine_init(&ws->scan, vk);
	ws->scan.trace = ws->trace;
	gcache_init(vk);
}

void
wire_server_accept(wire_server *ws, int listen_fd)
{
	wire_client *wc;
	int			fd = accept(listen_fd, NULL, NULL);

	if (fd < 0)
		return;
	if (ws->nclients >= WIRE_MAX_CLIENTS || !(wc = calloc(1, sizeof(*wc))))
	{
		close(fd);
		return;
	}
	(void) fcntl(fd, F_SETFD, FD_CLOEXEC);
	wc->fd = fd;
	wc->eng = &ws->scan;
	ws->clients[ws->nclients++] = wc;
}

int
wire_server_pollfds(wire_server *ws, struct pollfd *pfds, int max)
{
	int		i, n = 0;

	for (i = 0; i < ws->nclients && n < max; i++)
	{
		pfds[n].fd = ws->clients[i]->fd;
		pfds[n].events = POLLIN;
		pfds[n].revents = 0;
		n++;
	}
	return n;
}

void
wire_server_service(wire_server *ws, const struct pollfd *pfds, int n)
{
	int		i, j;

	for (i = 0; i < n; i++)
	{
		wire_client *wc = NULL;

		if (pfds[i].revents == 0)
			continue;
		for (j = 0; j < ws->nclients; j++)
			if (ws->clients[j]->fd == pfds[i].fd)
				wc = ws->clients[j];
		if (!wc)
			continue;
		if (pfds[i].revents & (POLLIN | POLLHUP | POLLERR))
			client_read(ws, wc);
	}
	/* reap closed connections */
	for (i = 0; i < ws->nclients; )
	{
		if (ws->clients[i]->closing)
		{
			client_free(ws->clients[i]);
			ws->clients[i] = ws->clients[--ws->nclients];
		}
		else
			i++;
	}
}

void
wire_server_close_all(wire_server *ws)
{
	while (ws->nclients > 0)
		client_free(ws->clients[--ws->nclients]);
}

void
wire_server_fini(wire_server *ws)
{
	wire_server_close_all(ws);
	gcache_fini();
	scan_engine_fini(&ws->scan);
}
