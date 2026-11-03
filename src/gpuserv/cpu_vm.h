/*
 * cpu_vm.h
 *
 * The scan VM on the CPU: a port of eval_row() and vm_run() of scan_common.slang. Used by the
 * service for the rows the GPU could not handle (GROUP BY overflow), and as a reference.
 */
#ifndef GPUSERV_CPU_VM_H
#define GPUSERV_CPU_VM_H
#include <stdint.h>
#include "scan_prog.h"

typedef struct
{
	int64_t		v;
	uint32_t	isnull;
} vm_reg;

/*
 * Evaluate one heap tuple (a pointer to its HeapTupleHeaderData) with the program: deform the
 * columns it needs, run the qual part and, if the tuple qualifies, the rest. Returns 1 if it
 * qualifies (regs hold the values), 0 if not, -1 on an evaluation error (*err = the error code
 * the kernel would report: FuncOpCode | reason << 16).
 */
int		cpu_eval_row(const scan_prog *prog, const unsigned char *htup,
					 vm_reg regs[SCAN_MAX_REGS], uint32_t *err);

#endif	/* GPUSERV_CPU_VM_H */
