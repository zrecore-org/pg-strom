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

/* GROUP BY: up to this many key columns (gagg.slang's GA_NK), of these types */
#define SCAN_GROUP_KEYS		2
#define SCAN_JOIN_MAXDEPTH	4		/* inner relations of one GpuJoin; must equal JOIN_MAXD of join.slang */
bool	scan_vm_groupkey_type_supported(int type_code);

#endif	/* GPUSERV_SCAN_CAPS_H */
