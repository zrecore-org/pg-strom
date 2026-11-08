/*
 * scan_caps.h
 *
 * The part of the scan VM's capabilities both the planner and the service need.
 */
#ifndef GPUSERV_SCAN_CAPS_H
#define GPUSERV_SCAN_CAPS_H
#include <stdbool.h>
#include <stdint.h>

enum {
#define SCANVM_OP(n, v)	SOP_##n = v,
#include "kernels_vm/scan_vm_ops.inc"
#undef SCANVM_OP
	SOP_END
};

/*
 * The optional instruction groups of the interpreter (kernels_vm/scan_common.slang compiles each out with -DVM_NO_<group>).
 * scan_prog.feat is the set of groups a program uses, and it picks one of three kernel tiers (every VM kernel family is built
 * three times, see the Makefile): "lite" for a program that uses none of them (a fraction of the size, the first pipeline
 * creation takes a second where Metal needs minutes for the full join kernel), "nogeom" (-DVM_NO_GEOM) for numeric, jsonb and
 * float8 without geometry (the geometry soft-float code is about 70% of the full kernel), and "full" for geometry.
 */
#define SCAN_FEAT_NUM	0x1			/* numeric */
#define SCAN_FEAT_JSON	0x2			/* jsonb */
#define SCAN_FEAT_F8	0x4			/* float8 (software binary64) */
#define SCAN_FEAT_GEOM	0x8			/* geometry */
#define SCAN_FEAT_ALL	0xf

/* kernel tiers; the index into the engine's pipeline arrays is (register/aggregate variant) + (variants per tier) * tier */
#define SCAN_TIER_FULL		0
#define SCAN_TIER_LITE		1
#define SCAN_TIER_NOGEOM	2
#define SCAN_NTIERS			3

/* PGSTROM_GPUSERV_TIERS=full: every program runs on the full kernels (what the service did before the tiers; for measurements) */
extern int scan_tier_full_only;

static inline int
scan_feat_tier(uint32_t feat)
{
	if (scan_tier_full_only)
		return SCAN_TIER_FULL;
	if (feat == 0)
		return SCAN_TIER_LITE;
	return (feat & SCAN_FEAT_GEOM) ? SCAN_TIER_FULL : SCAN_TIER_NOGEOM;
}

typedef struct
{
	int		cls;			/* 0: unsupported, else SOP_CMPI / SOP_CMPF / SOP_ARITH / ..., or
							 * -SOP_ARITH for unary minus */
	int		kind;			/* comparison or arithmetic kind */
	int		width;			/* result width in bits (ARITH), target width (CAST) */
} scan_func_class;

scan_func_class scan_vm_classify_func(uint32_t opcode);
const char *scan_vm_opcode_name(uint32_t opcode);
/* the optional instruction group of an opcode (SCAN_FEAT_* of scan_prog.h, 0 for the always compiled ones); test/gpuserv feat_check.c */
uint32_t scan_vm_op_feat(uint32_t op);
/* A6: see scan_caps.c. Returns SOP_LIKE_*, or -1 if the VM cannot match the pattern */
int		scan_vm_like_shape(const unsigned char *pat, int plen, unsigned char *lit, int *p_litlen);
/* true if the VM implements this function opcode */
bool	scan_vm_func_supported(uint32_t opcode);
/* an aggregate action the kernels implement without GROUP BY only (float8) */
bool	scan_vm_agg_action_nogroup(int action);

/*
 * A6 numeric: a register is { unscaled int64, scale }. These helpers are shared by the planner (a numeric
 * constant is pushed down only if it fits), the program builder, the CPU copy of the VM and the tuple former.
 */
#define SCAN_NUMERIC_MAXLEN		32		/* bytes of a formed numeric (1-byte header, short or long form) */
int		scan_vm_num_up(int64_t *v, uint32_t d);
int		scan_vm_num_cmp(int64_t va, uint32_t sa, int64_t vb, uint32_t sb);
int		scan_vm_numeric_payload_decode(const unsigned char *p, uint32_t plen, int64_t *v, uint32_t *sc);
int		scan_vm_numeric_datum_decode(const unsigned char *d, int64_t *v, uint32_t *sc);
uint32_t scan_vm_numeric_encode(int64_t v, uint32_t sc, unsigned char *buf);
/* numeric.c's division, remainder, rounding ...: return 1 with the result, or 0 if the backend must evaluate the row */
int		scan_vm_num_div(int kind, int64_t a, uint32_t sa, int64_t b, uint32_t sb, int64_t *r, uint32_t *rsc);
int		scan_vm_num_round(int kind, int64_t v, uint32_t sc, int64_t n, int64_t *r, uint32_t *rsc);
int		scan_vm_num_unary(int kind, int64_t v, uint32_t sc, int64_t *r, uint32_t *rsc);

/*
 * A8 jsonb: navigation over the binary form of a JsonbContainer (utils/jsonb.h), shared by the planner (the shape of a
 * containment pattern), the CPU copy of the VM and the unit test. All offsets are byte offsets into 'b'. A function
 * that finds the data malformed returns -1 (the row is then evaluated by the backend).
 */
#define SCAN_JB_STR		0		/* kinds: the JEntry type; a column or a nested container is SCAN_JB_CONT */
#define SCAN_JB_NUM		1
#define SCAN_JB_FALSE	2
#define SCAN_JB_TRUE	3
#define SCAN_JB_NULL	4
#define SCAN_JB_CONT	5

typedef struct
{
	uint32_t	count;			/* pairs of an object, elements of an array */
	uint32_t	nent;			/* JEntry words: 2 * count for an object */
	int			isobj, isarr, isscalar;
} scan_jb_cont;

int		scan_jb_open(const unsigned char *b, uint32_t cb, uint32_t clen, scan_jb_cont *c);
int		scan_jb_child(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
					  uint32_t i, uint32_t *kind, uint32_t *voff, uint32_t *vlen);
/* 1: found (*idx = index of the pair), 0: absent, -1: malformed */
int		scan_jb_find_key(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
						 const unsigned char *key, uint32_t kl, uint32_t *idx);
/* ? : a key of an object or a string element of an array. 1 / 0 / -1 */
int		scan_jb_exists(const unsigned char *b, uint32_t cb, uint32_t clen, const scan_jb_cont *c,
					   const unsigned char *key, uint32_t kl);
/* the numeric varlena at [off, off+len) as { v, sc } (1) or not representable (0); -1 malformed */
int		scan_jb_numeric(const unsigned char *b, uint32_t off, uint32_t len, int64_t *v, uint32_t *sc);
/*
 * Compile the constant jsonb of "col @> const" (the payload of the varlena: a JsonbContainer) into the pattern
 * the VM matches (JSONCONT): returns its length in out (at most outcap), or -1 if the VM cannot evaluate this
 * constant exactly (a nested array of containers, a raw scalar, a number that does not fit ...).
 */
int		scan_vm_jsonb_pattern(const unsigned char *jb, uint32_t len, unsigned char *out, uint32_t outcap);
#define SCAN_JB_PATTERN_MAX		2048
/* pattern terms (JSONCONT): see scan_caps.c */
#define SCAN_JBT_EQ		1
#define SCAN_JBT_ISOBJ	2
#define SCAN_JBT_ISARR	3
#define SCAN_JBT_ELEM	4

/*
 * Aggregation (GpuPreAgg): which partial-aggregate actions (KAGG_ACTION__*, xpu_common.h)
 * the service implements. The planner offers only these.
 */
bool	scan_vm_agg_action_supported(int action);

/*
 * Translation limits of the service. scan_prog.c fails a program that exceeds them, and a failed program
 * fails the query (there is no CPU fallback for a refused program), so the planner must not offer what
 * does not fit: codegen.c refuses (conservatively) every expression / plan whose estimate is above them.
 *
 *  SCAN_MAX_REGS     registers of the VM (VM_MAX_REGS of scan_common.slang). Register i < nslots is the
 *                    kernel variable of slot i; temporaries of expressions start at nslots.
 *  SCAN_AGG_MAX      output columns of a GpuPreAgg (agg.slang AGG_MAX): keys + partial aggregate columns.
 *  SCAN_VM_TEMP_BUDGET   registers one expression may need above the slots (planner's estimate, see
 *                    pgstrom_vulkan_expr_limits in codegen.c)
 *  SCAN_VM_PERM_RESERVE  registers kept for values that must survive the next expression (GROUP BY keys,
 *                    aggregate arguments, computed columns that cannot use their own slot)
 *  SCAN_VM_SLOT_BUDGET   kernel variables (slots) a plan may use: SCAN_MAX_REGS - temps - reserve.
 *  SCAN_VM_EXPR_NODES    nodes of one expression (the program buffer of a join is limited to 16384 words)
 */
#define SCAN_MAX_REGS			32		/* must equal VM_MAX_REGS of scan_common.slang */
#define SCAN_AGG_MAX			12		/* must equal AGG_MAX of agg.slang */
#define SCAN_VM_TEMP_BUDGET		6
#define SCAN_VM_PERM_RESERVE	3
#define SCAN_VM_SLOT_BUDGET		(SCAN_MAX_REGS - SCAN_VM_TEMP_BUDGET - SCAN_VM_PERM_RESERVE)
#define SCAN_VM_EXPR_NODES		120
#define SCAN_VM_BOOL_ARGS		60		/* operands of one AND/OR (scan_prog.c: 64) */

/* GROUP BY: up to this many key columns on the session-wide table of a heap scan (gtab.slang's GA_NK), of these types;
 * SCAN_GROUP_KEYS_SHARED for the shared-memory kernels (gagg.slang's GA_NK), Arrow/GPU-Cache sources and joins */
#define SCAN_GROUP_KEYS		4
#define SCAN_GROUP_KEYS_SHARED	2
#define SCAN_JOIN_MAXDEPTH	4		/* inner relations of one GpuJoin; must equal JOIN_MAXD of join.slang */
bool	scan_vm_groupkey_type_supported(int type_code);

/*
 * GPU-Sort (A7): which sort keys of a GpuPreAgg result the service sorts on. 'kind' is KSORT_KEY_KIND__*,
 * 'action' the KAGG_ACTION__* of the aggregate the key comes from (0 for a GROUP BY key), 'type_code' the
 * TypeOpCode of a GROUP BY key. At most this many keys; GPU-Sort needs the group count to fit a
 * device buffer (gpusort.c) or it sorts on the CPU of the service.
 */
#define SCAN_SORT_KEYS		8
bool	scan_vm_gpusort_key_supported(int kind, int action, int type_code);

#endif	/* GPUSERV_SCAN_CAPS_H */
