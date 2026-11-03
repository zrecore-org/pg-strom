/*
 * nocuda/gpu_devattrs.h
 *
 * Replaces the header that src/Makefile generates from the real cuda.h. Same format:
 * one DEV_ATTR(LABEL, description) per device attribute the host code reads, each
 * matching a CU_DEVICE_ATTRIBUTE_<LABEL> in nocuda/cuda.h.
 */
DEV_ATTR(MAX_THREADS_PER_BLOCK, "Maximum number of threads per block")
DEV_ATTR(MAX_BLOCK_DIM_X, "Maximum block dimension X")
DEV_ATTR(MAX_BLOCK_DIM_Y, "Maximum block dimension Y")
DEV_ATTR(MAX_BLOCK_DIM_Z, "Maximum block dimension Z")
DEV_ATTR(MAX_GRID_DIM_X, "Maximum grid dimension X")
DEV_ATTR(MAX_GRID_DIM_Y, "Maximum grid dimension Y")
DEV_ATTR(MAX_GRID_DIM_Z, "Maximum grid dimension Z")
DEV_ATTR(MAX_SHARED_MEMORY_PER_BLOCK, "Maximum shared memory available per block in bytes")
DEV_ATTR(TOTAL_CONSTANT_MEMORY, "Memory available on device for constant variables in a kernel in bytes")
DEV_ATTR(WARP_SIZE, "Warp size in threads")
DEV_ATTR(MAX_PITCH, "Maximum pitch in bytes allowed by memory copies")
DEV_ATTR(MAX_REGISTERS_PER_BLOCK, "Maximum number of 32-bit registers available per block")
DEV_ATTR(CLOCK_RATE, "Typical clock frequency in kilohertz")
DEV_ATTR(MULTIPROCESSOR_COUNT, "Number of multiprocessors on device")
DEV_ATTR(COMPUTE_MODE, "Compute mode (See CUcomputemode for details)")
DEV_ATTR(PCI_BUS_ID, "PCI bus ID of the device")
DEV_ATTR(PCI_DEVICE_ID, "PCI device ID of the device")
DEV_ATTR(MEMORY_CLOCK_RATE, "Peak memory clock frequency in kilohertz")
DEV_ATTR(GLOBAL_MEMORY_BUS_WIDTH, "Global memory bus width in bits")
DEV_ATTR(L2_CACHE_SIZE, "Size of L2 cache in bytes")
DEV_ATTR(MAX_THREADS_PER_MULTIPROCESSOR, "Maximum resident threads per multiprocessor")
DEV_ATTR(PCI_DOMAIN_ID, "PCI domain ID of the device")
DEV_ATTR(COMPUTE_CAPABILITY_MAJOR, "Major compute capability version number")
DEV_ATTR(COMPUTE_CAPABILITY_MINOR, "Minor compute capability version number")
DEV_ATTR(MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "Maximum shared memory available per multiprocessor in bytes")
DEV_ATTR(MAX_REGISTERS_PER_MULTIPROCESSOR, "Maximum number of 32-bit registers available per multiprocessor")
DEV_ATTR(MANAGED_MEMORY, "Device can allocate managed memory on this system")
DEV_ATTR(CONCURRENT_MANAGED_ACCESS, "Device can coherently access managed memory concurrently with the CPU")
DEV_ATTR(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "Maximum optin shared memory per block")
DEV_ATTR(GPU_DIRECT_RDMA_SUPPORTED, "Device supports GPUDirect RDMA APIs")
DEV_ATTR(HOST_NUMA_ID, "NUMA ID of the host node closest to the device, or -1 when the system does not support NUMA")
