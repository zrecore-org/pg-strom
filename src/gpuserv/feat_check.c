/*
 * feat_check.c
 *
 * Offline check of the kernel tier selection (no GPU, no kernels): every instruction of kernels_vm/scan_vm_ops.inc must be
 * classified by scan_vm_op_feat() with the optional instruction group scan_common.slang compiles it under. A geometry
 * instruction whose group were missing would let a program run on the "nogeom" or "lite" kernels, where the interpreter
 * answers "malformed program" for it. Opcode numbers 130 up are the geometry group; a new one that is neither classified
 * nor NULLMRG (the one always compiled in) fails here.
 *
 *   make -C src/gpuserv feat_check && src/gpuserv/feat_check
 */
#include <stdio.h>
#include <string.h>
#include "xpu_common.h"
#include "scan_caps.h"

static int	nfail;
static int	nopcodes;

static void
expect(const char *name, uint32_t op, uint32_t want)
{
	uint32_t	got = scan_vm_op_feat(op);

	nopcodes++;
	if (got != want)
	{
		fprintf(stderr, "feat_check: %s (%u): feat 0x%x, expected 0x%x\n", name, op, got, want);
		nfail++;
	}
}

static int
is_one_of(const char *name, const char *const *list)
{
	for (; *list; list++)
		if (strcmp(name, *list) == 0)
			return 1;
	return 0;
}

static const char *const num_ops[] = {"NUMCONST", "NUMCMP", "NUMARITH", "I2N", "N2I", "NUMDIV", "NUMRND", NULL};
static const char *const f8_ops[] = {"CMPD", "DARITH", "DUNARY", "I2D", "D2I", "F2D", "D2F", NULL};

static void
check_one(const char *name, uint32_t op)
{
	if (strncmp(name, "GEOM", 4) == 0 && op < 200)
		expect(name, op, SCAN_FEAT_GEOM);
	else if (strcmp(name, "NULLMRG") == 0)
		expect(name, op, 0);
	else if (strncmp(name, "JSON", 4) == 0 && op >= 100 && op < 120)
		expect(name, op, SCAN_FEAT_JSON);
	else if (is_one_of(name, num_ops))
		expect(name, op, SCAN_FEAT_NUM);
	else if (is_one_of(name, f8_ops))
		expect(name, op, SCAN_FEAT_F8);
	else if (op >= 130 && op < 200)
	{
		fprintf(stderr, "feat_check: %s (%u) is in the geometry opcode range but is not a GEOM* opcode\n", name, op);
		nfail++;
	}
	else if (op >= 1 && op < 80 && strchr(name, '_') == NULL)
		expect(name, op, 0);	/* the always compiled instructions */
}

int
main(void)
{
#define SCANVM_OP(n, v)	check_one(#n, (v));
#include "kernels_vm/scan_vm_ops.inc"
#undef SCANVM_OP

	/* the tiers */
	if (scan_feat_tier(0) != SCAN_TIER_LITE ||
		scan_feat_tier(SCAN_FEAT_NUM | SCAN_FEAT_JSON | SCAN_FEAT_F8) != SCAN_TIER_NOGEOM ||
		scan_feat_tier(SCAN_FEAT_F8) != SCAN_TIER_NOGEOM ||
		scan_feat_tier(SCAN_FEAT_GEOM) != SCAN_TIER_FULL ||
		scan_feat_tier(SCAN_FEAT_ALL) != SCAN_TIER_FULL ||
		scan_feat_tier(SCAN_FEAT_GEOM | SCAN_FEAT_NUM) != SCAN_TIER_FULL)
	{
		fprintf(stderr, "feat_check: scan_feat_tier() is wrong\n");
		nfail++;
	}
	if (nopcodes < 40)
	{
		fprintf(stderr, "feat_check: only %d opcodes were checked\n", nopcodes);
		nfail++;
	}
	if (nfail)
		return 1;
	printf("feat_check: %d opcodes classified, tiers ok\n", nopcodes);
	return 0;
}
