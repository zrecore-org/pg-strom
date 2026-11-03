/*
 * vm_selftest.h - self-test of the opcode-switch VM on a real device
 */
#ifndef VM_SELFTEST_H
#define VM_SELFTEST_H
#include <stddef.h>
#include "gpu_vk.h"

/*
 * Assemble expression programs on the host, run them on the device and compare with a
 * CPU reference (exact matches, three-valued NULL logic, error reporting). Returns 0 if
 * everything agrees, -1 (message in err) otherwise.
 */
int gpuserv_vm_selftest(gpu_vk_ctx *ctx, char *err, size_t errlen);

#endif	/* VM_SELFTEST_H */
