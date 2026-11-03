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
/* true if the VM implements this function opcode */
bool	scan_vm_func_supported(uint32_t opcode);

/*
 * Aggregation (GpuPreAgg): which partial-aggregate actions (KAGG_ACTION__*, xpu_common.h)
 * the service implements. The planner offers only these.
 */
bool	scan_vm_agg_action_supported(int action);

#endif	/* GPUSERV_SCAN_CAPS_H */
