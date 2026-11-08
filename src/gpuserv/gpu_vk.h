/*
 * gpu_vk.h
 *
 * Thin Vulkan compute wrapper for the PG-Strom GPU service. Library code never calls
 * exit()/abort(): every function returns 0 on success and a negative value on failure,
 * with a message in ctx->err (or the caller's buffer for the enumerator).
 */
#ifndef GPU_VK_H
#define GPU_VK_H
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define GPU_VK_ERRLEN			512
#define GPU_VK_MAX_DEVICES		16
#define GPU_VK_DEFAULT_TIMEOUT_MS	10000
/* the first dispatch of a pipeline can include a lazy shader compile (Metal, lavapipe): minutes for the big kernels */
#define GPU_VK_FIRST_DISPATCH_TIMEOUT_MS	600000

/* result of enumeration (temporary instance, nothing opened) */
typedef struct gpu_vk_devinfo
{
	int			index;
	char		name[256];
	char		driver_name[VK_MAX_DRIVER_NAME_SIZE];
	char		driver_info[VK_MAX_DRIVER_INFO_SIZE];
	char		uuid[33];			/* 32 hex digits, "" if unknown */
	uint32_t	vendor_id;
	uint32_t	device_id;
	uint32_t	device_type;		/* VkPhysicalDeviceType */
	uint32_t	api_version;
	uint32_t	driver_version;
	uint64_t	max_storage_buffer_range;
	uint32_t	max_shared_memory;
	uint32_t	max_wg_invocations;
	uint32_t	max_push_constants;
} gpu_vk_devinfo;

/* features the device reports (and that gpu_vk_open() enabled) */
typedef struct gpu_vk_feat
{
	int		int64, int16, int8, float64, float16;
	int		storage8;				/* storageBuffer8BitAccess && uniformAndStorageBuffer8BitAccess */
	int		scalar_block_layout;
	int		buffer_int64_atomics, shared_int64_atomics;
	int		bda;
	int		memory_model, memory_model_device_scope;
	int		subgroup_extended_types;
	int		subgroup_size_control;
	int		atomic_float32_add;		/* VK_EXT_shader_atomic_float shaderBufferFloat32AtomicAdd */
	/* extensions */
	int		ext_host_import;		/* VK_EXT_external_memory_host */
	int		ext_portability;		/* VK_KHR_portability_subset enabled (MoltenVK) */
	/* subgroup properties (compute stage) */
	uint32_t	subgroup_size;
	uint32_t	subgroup_size_min, subgroup_size_max;
	uint32_t	subgroup_ops;		/* VkSubgroupFeatureFlags */
	int		subgroup_compute;		/* compute stage supports subgroup ops */
} gpu_vk_feat;

typedef struct gpu_vk_ctx
{
	VkInstance			instance;
	VkPhysicalDevice	phys;
	VkDevice			dev;
	VkQueue				queue;
	uint32_t			queue_family;
	VkCommandPool		cmd_pool;
	VkCommandBuffer		cmd;
	VkFence				fence;
	VkPipelineCache		pcache;					/* persisted compiled pipelines, or VK_NULL_HANDLE */
	size_t				pcache_saved;			/* size of the data last written to pcache_path */
	char				pcache_path[768];
	VkPipelineCache		pcache_sp;				/* the same for the specialized kernels (their own file) */
	size_t				pcache_sp_saved;
	char				pcache_sp_path[768];

	gpu_vk_devinfo		info;
	gpu_vk_feat			feat;
	VkPhysicalDeviceMemoryProperties mem;
	VkPhysicalDeviceLimits limits;
	uint32_t			host_import_alignment;	/* 0 if VK_EXT_external_memory_host is not enabled */
	uint64_t			max_buffer_size;		/* 0 if not reported */
	int					unified_memtype;		/* DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT, or -1 */
	int					hostvis_memtype;		/* HOST_VISIBLE|HOST_COHERENT, or -1 */
	uint64_t			device_memory;			/* largest DEVICE_LOCAL heap */

	int					lost;					/* timeout or VK_ERROR_DEVICE_LOST seen */
	uint32_t			timeout_ms;
	uint32_t			first_timeout_ms;		/* limit for the first dispatch of a pipeline */
	char				err[GPU_VK_ERRLEN];
	char				ext_enabled[1024];		/* space separated list, for logs */

	/* function pointers resolved at open time (NULL if unavailable) */
	PFN_vkGetBufferDeviceAddress		fn_bda;
	PFN_vkGetMemoryHostPointerPropertiesEXT	fn_host_ptr_props;
} gpu_vk_ctx;

typedef struct gpu_vk_buffer
{
	VkBuffer		buf;
	VkDeviceMemory	mem;
	void		   *map;			/* persistently mapped (or the imported pointer) */
	size_t			size;
	uint64_t		addr;			/* device address if created with want_bda */
	int				unified;		/* memory type is DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT */
	int				imported;
} gpu_vk_buffer;

typedef struct gpu_vk_pipeline
{
	VkShaderModule			module;
	VkDescriptorSetLayout	dsl;
	VkPipelineLayout		layout;
	VkPipeline				pipeline;
	VkDescriptorPool		dpool;
	VkDescriptorSet			dset;
	uint32_t				nbind;
	uint32_t				pc_size;
	int						dispatched;			/* a dispatch of it has finished (the compile is paid) */
} gpu_vk_pipeline;

/* time limit for the next dispatch of 'p' */
static inline uint32_t
gpu_vk_wait_ms(const gpu_vk_ctx *ctx, const gpu_vk_pipeline *p)
{
	return p->dispatched ? ctx->timeout_ms : ctx->first_timeout_ms;
}

/*
 * Enumerate physical devices using a temporary instance. Returns the number of devices
 * stored in 'out' (0 if there is no Vulkan driver/device) or -1 on a hard failure
 * (message in err).
 */
int  gpu_vk_enumerate(gpu_vk_devinfo *out, int max, char *err, size_t errlen);

/* Open device 'index' (as numbered by gpu_vk_enumerate). 0 on success. */
int  gpu_vk_open(gpu_vk_ctx *ctx, int index);
void gpu_vk_close(gpu_vk_ctx *ctx);
/* Human line describing enabled features/extensions, written into buf. */
void gpu_vk_describe(const gpu_vk_ctx *ctx, char *buf, size_t buflen);

/* Host-visible+coherent buffer (storage + transfer usage). */
int  gpu_vk_buffer_create(gpu_vk_ctx *ctx, gpu_vk_buffer *b, size_t size, int want_bda);
/* Import existing host memory; ptr and size must be multiples of host_import_alignment. */
int  gpu_vk_buffer_import_host(gpu_vk_ctx *ctx, gpu_vk_buffer *b, void *ptr, size_t size);
void gpu_vk_buffer_destroy(gpu_vk_ctx *ctx, gpu_vk_buffer *b);

/*
 * Compute pipeline from SPIR-V (entry point 'main'), nbind storage buffers at set 0
 * binding 0..nbind-1 and a push-constant range of pc_size bytes (0 = none).
 * Returns <0 (no crash) if the driver's shader compiler rejects the module.
 */
int  gpu_vk_pipeline_create(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, const void *spirv,
							size_t spirv_bytes, uint32_t nbind, uint32_t pc_size);
/*
 * The same with the message going to the caller's buffer instead of ctx->err: usable from a second
 * thread (pipeline creation does not need external synchronization; ctx->err would race).
 */
int  gpu_vk_pipeline_create_ex(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, const void *spirv,
							   size_t spirv_bytes, uint32_t nbind, uint32_t pc_size,
							   char *err, size_t errlen);
/* a specialized kernel: persisted in the specialized kernels' pipeline cache file, apart from the interpreter's */
int  gpu_vk_pipeline_create_spec(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, const void *spirv,
								 size_t spirv_bytes, uint32_t nbind, uint32_t pc_size,
								 char *err, size_t errlen);
void gpu_vk_pipeline_destroy(gpu_vk_ctx *ctx, gpu_vk_pipeline *p);

/*
 * Bind bufs[0..nbind-1], push 'pc', dispatch gx*gy*gz workgroups and wait for the fence
 * (ctx->timeout_ms). On timeout / device lost the context is marked 'lost'. The elapsed
 * wall time (submit to completion) is stored in *elapsed_ms if non-NULL.
 */
int  gpu_vk_dispatch(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, gpu_vk_buffer *const *bufs,
					 const void *pc, uint32_t gx, uint32_t gy, uint32_t gz,
					 double *elapsed_ms);

/*
 * A sequence of dispatches submitted at once with a compute-to-compute memory barrier after each (steps
 * that share a pipeline share its buffers). One fence wait for the whole sequence (gpu_vk_seq.c).
 */
typedef struct gpu_vk_step
{
	gpu_vk_pipeline *p;
	gpu_vk_buffer *const *bufs;
	const void	   *pc;
	uint32_t		gx, gy, gz;
} gpu_vk_step;
int  gpu_vk_dispatch_seq(gpu_vk_ctx *ctx, const gpu_vk_step *steps, uint32_t nsteps, double *elapsed_ms);

#endif	/* GPU_VK_H */
