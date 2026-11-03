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

typedef struct
{
	int		cls;			/* 0: unsupported, else SOP_CMPI / SOP_CMPF / SOP_ARITH / ..., or
							 * -SOP_ARITH for unary minus */
	int		kind;			/* comparison or arithmetic kind */
	int		width;			/* result width in bits (ARITH), target width (CAST) */
} scan_func_class;

scan_func_class scan_vm_classify_func(uint32_t opcode);
const char *scan_vm_opcode_name(uint32_t opcode);
/* A6: see scan_caps.c. Returns SOP_LIKE_*, or -1 if the VM cannot match the pattern */
int		scan_vm_like_shape(const unsigned char *pat, int plen, unsigned char *lit, int *p_litlen);
/* true if the VM implements this function opcode */
bool	scan_vm_func_supported(uint32_t opcode);

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

/* GROUP BY: up to this many key columns (gagg.slang's GA_NK), of these types */
#define SCAN_GROUP_KEYS		2
#define SCAN_JOIN_MAXDEPTH	4		/* inner relations of one GpuJoin; must equal JOIN_MAXD of join.slang */
bool	scan_vm_groupkey_type_supported(int type_code);

#endif	/* GPUSERV_SCAN_CAPS_H */
