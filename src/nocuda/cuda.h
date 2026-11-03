/*
 * nocuda/cuda.h
 *
 * Stand-in for the CUDA driver API header, used when the machine has no CUDA toolkit
 * (macOS, or Linux without NVIDIA). Together with cuda_stub.c it lets the host side of
 * PG-Strom build and load as a "no GPU" module: cuInit() succeeds, cuDeviceGetCount()
 * reports zero devices, and every other call fails with CUDA_ERROR_NOT_SUPPORTED.
 *
 * Scaffolding of the apple-silicon branch (milestone A0). It is deleted with the CUDA
 * code in milestone A2. Only what the host code uses is declared; values follow the real
 * header (CUDA 13) where the code depends on them.
 */
#ifndef PGSTROM_NOCUDA_CUDA_H
#define PGSTROM_NOCUDA_CUDA_H
#define __cuda_cuda_h__		1	/* the include guard of the real cuda.h; heterodb_extra.h tests it */
#include <stddef.h>
#include <stdint.h>

#define CUDA_VERSION		13000	/* selects the CUDA 13 prototypes in pg_compat.h */

typedef int					CUdevice;
typedef unsigned long long	CUdeviceptr;
typedef struct CUctx_st	   *CUcontext;
typedef struct CUmod_st	   *CUmodule;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct { char bytes[16]; } CUuuid;

typedef enum
{
	CUDA_SUCCESS				= 0,
	CUDA_ERROR_INVALID_VALUE	= 1,
	CUDA_ERROR_OUT_OF_MEMORY	= 2,
	CUDA_ERROR_NOT_INITIALIZED	= 3,
	CUDA_ERROR_NO_DEVICE		= 100,
	CUDA_ERROR_NOT_SUPPORTED	= 801,
	CUDA_ERROR_UNKNOWN			= 999
} CUresult;

#define CU_DEVICE_CPU				((CUdevice)-1)
#define CU_STREAM_PER_THREAD		((CUstream)0x2)
#define CU_CTX_SCHED_BLOCKING_SYNC	0x04
#define CU_MEM_ATTACH_GLOBAL		0x1

typedef enum { CU_COMPUTEMODE_DEFAULT = 0, CU_COMPUTEMODE_PROHIBITED = 2,
			   CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3 } CUcomputemode;
typedef enum { CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
			   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8 } CUfunction_attribute;
typedef enum { CU_LIMIT_STACK_SIZE = 0 } CUlimit;
typedef enum { CU_MEM_ADVISE_SET_READ_MOSTLY = 1 } CUmem_advise;
typedef enum { CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1 } CUmem_range_attribute;
typedef enum { CU_MEM_LOCATION_TYPE_DEVICE = 1,
			   CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 5 } CUmemLocationType;
typedef struct { CUmemLocationType type; int id; } CUmemLocation;

/* device attributes (values as in the real header; the list the host code needs) */
typedef enum
{
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
	CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
	CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134
} CUdevice_attribute;

typedef size_t (*CUoccupancyB2DSize)(int blockSize);

#ifdef __cplusplus
extern "C" {
#endif

/* driver API. cuStrError() is not CUDA's: PG-Strom defines it (gpu_service.c). */
CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int *version);
CUresult cuGetErrorName(CUresult error, const char **pStr);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxSetLimit(CUlimit limit, size_t value);
CUresult cuModuleLoad(CUmodule *module, const char *fname);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value);
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
										  CUoccupancyB2DSize blockSizeToDynamicSMemSize,
										  size_t dynamicSMemSize, int blockSizeLimit);
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
						unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
						unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
						void **kernelParams, void **extra);
CUresult cuStreamSynchronize(CUstream hStream);
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
CUresult cuMemAllocHost(void **pp, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemFreeHost(void *p);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
					  CUcontext srcContext, size_t ByteCount);
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUmemLocation location,
							unsigned int flags, CUstream hStream);
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
					 CUmemLocation location);
CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute,
								CUdeviceptr devPtr, size_t count);

#ifdef __cplusplus
}
#endif

#endif	/* PGSTROM_NOCUDA_CUDA_H */
