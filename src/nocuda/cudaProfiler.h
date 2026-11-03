/* nocuda/cudaProfiler.h - stand-in for <cudaProfiler.h> (see cuda.h) */
#ifndef PGSTROM_NOCUDA_CUDAPROFILER_H
#define PGSTROM_NOCUDA_CUDAPROFILER_H
#include "cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuProfilerStart(void);
CUresult cuProfilerStop(void);
#ifdef __cplusplus
}
#endif
#endif
