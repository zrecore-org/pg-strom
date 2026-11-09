/*
 * The backend module's copy of PostGIS 3.7's ptarray_signed_area in software binary64 (gpuserv/kernels_vm/area_ring.inc: the text the GPU
 * kernel and the service's CPU VM compile too). codegen.c calls it to check, before ST_Area is pushed down, that the installed PostGIS
 * returns the bits this code computes. No PostgreSQL headers: the file is plain C.
 */
#define F64SOFT_C 1
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "gpuserv/kernels_vm/f64_soft.inc"
#define AREA_FN static inline
#define AR_F2 struct ArF2s
#define AR_SRC_T const unsigned char *
#define AR_LD(src, off) svc_area_ld64((src) + (off))
static inline uint64_t
svc_area_ld64(const unsigned char *p)
{
	uint64_t	v;

	memcpy(&v, p, sizeof(v));
	return v;
}
#include "gpuserv/kernels_vm/area_ring.inc"

/* the signed area of `npoints` points (X, Y at `stride` bytes apart); returns false when the kernel would hand the ring to the backend */
bool
pgstrom_svc_area_ring(const void *pts, uint32_t stride, uint32_t npoints, double *signed_area)
{
	uint64_t	r = area_ring_signed((const unsigned char *) pts, 0, stride, npoints);

	if (r == AR_NONE)
		return false;
	memcpy(signed_area, &r, sizeof(r));
	return true;
}
