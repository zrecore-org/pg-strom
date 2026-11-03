/*
 * nocuda/cuda_stub.c
 *
 * Implementation of the stand-in CUDA driver API (see cuda.h): a driver that initialises
 * and sees no device. Anything that would need a device fails with
 * CUDA_ERROR_NOT_SUPPORTED, but the host code never gets that far because
 * cuDeviceGetCount() reports zero GPUs.
 */
#include "cuda.h"
#include "cudaProfiler.h"
#include <string.h>

#define NOT_SUPPORTED	return CUDA_ERROR_NOT_SUPPORTED

CUresult cuInit(unsigned int flags) { (void) flags; return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int *version) { *version = CUDA_VERSION; return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int *count) { *count = 0; return CUDA_SUCCESS; }

CUresult
cuGetErrorName(CUresult error, const char **pStr)
{
	switch (error)
	{
		case CUDA_SUCCESS:					*pStr = "CUDA_SUCCESS"; break;
		case CUDA_ERROR_INVALID_VALUE:		*pStr = "CUDA_ERROR_INVALID_VALUE"; break;
		case CUDA_ERROR_OUT_OF_MEMORY:		*pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
		case CUDA_ERROR_NOT_INITIALIZED:	*pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
		case CUDA_ERROR_NO_DEVICE:			*pStr = "CUDA_ERROR_NO_DEVICE"; break;
		case CUDA_ERROR_NOT_SUPPORTED:		*pStr = "CUDA_ERROR_NOT_SUPPORTED"; break;
		default:							return CUDA_ERROR_INVALID_VALUE;
	}
	return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) { (void) device; (void) ordinal; return CUDA_ERROR_INVALID_VALUE; }
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) { (void) name; (void) len; (void) dev; NOT_SUPPORTED; }
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) { memset(uuid, 0, sizeof(*uuid)); (void) dev; NOT_SUPPORTED; }
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) { *bytes = 0; (void) dev; NOT_SUPPORTED; }
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) { *pi = 0; (void) attrib; (void) dev; NOT_SUPPORTED; }
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) { *pctx = NULL; (void) dev; NOT_SUPPORTED; }
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { (void) dev; (void) flags; NOT_SUPPORTED; }
CUresult cuCtxSetCurrent(CUcontext ctx) { (void) ctx; NOT_SUPPORTED; }
CUresult cuCtxGetCurrent(CUcontext *pctx) { *pctx = NULL; return CUDA_SUCCESS; }
CUresult cuCtxSetLimit(CUlimit limit, size_t value) { (void) limit; (void) value; NOT_SUPPORTED; }
CUresult cuModuleLoad(CUmodule *module, const char *fname) { *module = NULL; (void) fname; NOT_SUPPORTED; }
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) { *hfunc = NULL; (void) hmod; (void) name; NOT_SUPPORTED; }
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) { *dptr = 0; if (bytes) *bytes = 0; (void) hmod; (void) name; NOT_SUPPORTED; }
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) { *pi = 0; (void) attrib; (void) hfunc; NOT_SUPPORTED; }
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) { (void) hfunc; (void) attrib; (void) value; NOT_SUPPORTED; }
CUresult
cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
								 CUoccupancyB2DSize blockSizeToDynamicSMemSize,
								 size_t dynamicSMemSize, int blockSizeLimit)
{
	*minGridSize = 0; *blockSize = 0;
	(void) func; (void) blockSizeToDynamicSMemSize; (void) dynamicSMemSize; (void) blockSizeLimit;
	NOT_SUPPORTED;
}
CUresult
cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
			   unsigned int bx, unsigned int by, unsigned int bz,
			   unsigned int shmem, CUstream s, void **params, void **extra)
{
	(void) f; (void) gx; (void) gy; (void) gz; (void) bx; (void) by; (void) bz;
	(void) shmem; (void) s; (void) params; (void) extra;
	NOT_SUPPORTED;
}
CUresult cuStreamSynchronize(CUstream hStream) { (void) hStream; NOT_SUPPORTED; }
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) { *dptr = 0; (void) bytesize; NOT_SUPPORTED; }
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) { *dptr = 0; (void) bytesize; (void) flags; NOT_SUPPORTED; }
CUresult cuMemAllocHost(void **pp, size_t bytesize) { *pp = NULL; (void) bytesize; NOT_SUPPORTED; }
CUresult cuMemFree(CUdeviceptr dptr) { (void) dptr; NOT_SUPPORTED; }
CUresult cuMemFreeHost(void *p) { (void) p; NOT_SUPPORTED; }
CUresult cuMemcpyHtoD(CUdeviceptr d, const void *s, size_t n) { (void) d; (void) s; (void) n; NOT_SUPPORTED; }
CUresult cuMemcpyDtoH(void *d, CUdeviceptr s, size_t n) { (void) d; (void) s; (void) n; NOT_SUPPORTED; }
CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) { (void) d; (void) s; (void) n; NOT_SUPPORTED; }
CUresult cuMemcpyPeer(CUdeviceptr d, CUcontext dc, CUdeviceptr s, CUcontext sc, size_t n) { (void) d; (void) dc; (void) s; (void) sc; (void) n; NOT_SUPPORTED; }
CUresult cuMemPrefetchAsync(CUdeviceptr p, size_t n, CUmemLocation l, unsigned int f, CUstream s) { (void) p; (void) n; (void) l; (void) f; (void) s; NOT_SUPPORTED; }
CUresult cuMemAdvise(CUdeviceptr p, size_t n, CUmem_advise a, CUmemLocation l) { (void) p; (void) n; (void) a; (void) l; NOT_SUPPORTED; }
CUresult cuMemRangeGetAttribute(void *d, size_t ds, CUmem_range_attribute a, CUdeviceptr p, size_t n) { (void) d; (void) ds; (void) a; (void) p; (void) n; NOT_SUPPORTED; }
CUresult cuProfilerStart(void) { NOT_SUPPORTED; }
CUresult cuProfilerStop(void) { NOT_SUPPORTED; }
