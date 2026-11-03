/*
 * gpu_vk.c
 *
 * Thin Vulkan compute wrapper, see gpu_vk.h. No exit()/abort(): errors are returned.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_vk.h"

/* ------------------------------------------------------------------------ */
/* helpers                                                                   */
/* ------------------------------------------------------------------------ */
static const char *
vkr_str(VkResult r)
{
	switch (r)
	{
		case VK_SUCCESS:						return "VK_SUCCESS";
		case VK_NOT_READY:						return "VK_NOT_READY";
		case VK_TIMEOUT:						return "VK_TIMEOUT";
		case VK_ERROR_OUT_OF_HOST_MEMORY:		return "VK_ERROR_OUT_OF_HOST_MEMORY";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:		return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		case VK_ERROR_INITIALIZATION_FAILED:	return "VK_ERROR_INITIALIZATION_FAILED";
		case VK_ERROR_DEVICE_LOST:				return "VK_ERROR_DEVICE_LOST";
		case VK_ERROR_MEMORY_MAP_FAILED:		return "VK_ERROR_MEMORY_MAP_FAILED";
		case VK_ERROR_LAYER_NOT_PRESENT:		return "VK_ERROR_LAYER_NOT_PRESENT";
		case VK_ERROR_EXTENSION_NOT_PRESENT:	return "VK_ERROR_EXTENSION_NOT_PRESENT";
		case VK_ERROR_FEATURE_NOT_PRESENT:		return "VK_ERROR_FEATURE_NOT_PRESENT";
		case VK_ERROR_INCOMPATIBLE_DRIVER:		return "VK_ERROR_INCOMPATIBLE_DRIVER";
		case VK_ERROR_TOO_MANY_OBJECTS:			return "VK_ERROR_TOO_MANY_OBJECTS";
		case VK_ERROR_FORMAT_NOT_SUPPORTED:		return "VK_ERROR_FORMAT_NOT_SUPPORTED";
		case VK_ERROR_INVALID_SHADER_NV:		return "VK_ERROR_INVALID_SHADER_NV";
		case VK_ERROR_UNKNOWN:					return "VK_ERROR_UNKNOWN";
		case VK_ERROR_INVALID_EXTERNAL_HANDLE:	return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
		default:								return "VkResult(other)";
	}
}

static void
set_err(char *err, size_t errlen, const char *fmt, ...)
{
	va_list		ap;

	if (!err || errlen == 0)
		return;
	va_start(ap, fmt);
	vsnprintf(err, errlen, fmt, ap);
	va_end(ap);
}

#define CTX_ERR(ctx, ...)	set_err((ctx)->err, sizeof((ctx)->err), __VA_ARGS__)

static void
copy_str(char *dst, size_t dstsz, const char *src)
{
	snprintf(dst, dstsz, "%s", src ? src : "");
}

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int
has_ext(const VkExtensionProperties *list, uint32_t n, const char *name)
{
	for (uint32_t i = 0; i < n; i++)
		if (strcmp(list[i].extensionName, name) == 0)
			return 1;
	return 0;
}

static uint32_t
min_u32(uint32_t a, uint32_t b)
{
	return a < b ? a : b;
}

/* ------------------------------------------------------------------------ */
/* instance                                                                  */
/* ------------------------------------------------------------------------ */
typedef struct
{
	VkInstance	instance;
	uint32_t	api;			/* api version requested for the instance */
	int			portability;	/* VK_KHR_portability_enumeration enabled */
} inst_t;

static VkResult
create_instance(inst_t *in, char *err, size_t errlen)
{
	PFN_vkEnumerateInstanceVersion fn_ver;
	uint32_t	loader_api = VK_API_VERSION_1_0;
	uint32_t	nexts = 0;
	VkExtensionProperties *exts = NULL;
	const char *enabled[2];
	uint32_t	nenabled = 0;
	VkApplicationInfo app;
	VkInstanceCreateInfo ici;
	VkResult	rc;

	memset(in, 0, sizeof(*in));
	fn_ver = (PFN_vkEnumerateInstanceVersion)
		vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
	if (fn_ver)
		fn_ver(&loader_api);
	if (loader_api < VK_API_VERSION_1_1)
	{
		set_err(err, errlen, "Vulkan loader reports API %u.%u, need >= 1.1",
				VK_API_VERSION_MAJOR(loader_api), VK_API_VERSION_MINOR(loader_api));
		return VK_ERROR_INCOMPATIBLE_DRIVER;
	}
	in->api = min_u32(loader_api, VK_API_VERSION_1_3);

	if (vkEnumerateInstanceExtensionProperties(NULL, &nexts, NULL) == VK_SUCCESS && nexts > 0)
	{
		exts = calloc(nexts, sizeof(*exts));
		if (exts && vkEnumerateInstanceExtensionProperties(NULL, &nexts, exts) != VK_SUCCESS)
			nexts = 0;
	}
	memset(&ici, 0, sizeof(ici));
	if (exts && has_ext(exts, nexts, "VK_KHR_portability_enumeration"))
	{
		enabled[nenabled++] = "VK_KHR_portability_enumeration";
		ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
		in->portability = 1;
	}
	free(exts);

	memset(&app, 0, sizeof(app));
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "pgstrom-gpuserv";
	app.applicationVersion = 1;
	app.pEngineName = "pg-strom";
	app.apiVersion = in->api;
	ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ici.pApplicationInfo = &app;
	ici.enabledExtensionCount = nenabled;
	ici.ppEnabledExtensionNames = enabled;
	rc = vkCreateInstance(&ici, NULL, &in->instance);
	if (rc != VK_SUCCESS)
	{
		set_err(err, errlen, "vkCreateInstance failed: %s", vkr_str(rc));
		in->instance = VK_NULL_HANDLE;
	}
	return rc;
}

static VkPhysicalDevice *
list_physical(VkInstance inst, uint32_t *pn, char *err, size_t errlen)
{
	VkPhysicalDevice *pd;
	VkResult	rc;
	uint32_t	n = 0;

	*pn = 0;
	rc = vkEnumeratePhysicalDevices(inst, &n, NULL);
	if (rc != VK_SUCCESS)
	{
		set_err(err, errlen, "vkEnumeratePhysicalDevices failed: %s", vkr_str(rc));
		return NULL;
	}
	if (n == 0)
		return NULL;
	pd = calloc(n, sizeof(*pd));
	if (!pd)
	{
		set_err(err, errlen, "out of memory");
		return NULL;
	}
	rc = vkEnumeratePhysicalDevices(inst, &n, pd);
	if (rc != VK_SUCCESS && rc != VK_INCOMPLETE)
	{
		set_err(err, errlen, "vkEnumeratePhysicalDevices failed: %s", vkr_str(rc));
		free(pd);
		return NULL;
	}
	*pn = n;
	return pd;
}

static VkExtensionProperties *
list_device_ext(VkPhysicalDevice pd, uint32_t *pn)
{
	VkExtensionProperties *exts = NULL;
	uint32_t	n = 0;

	*pn = 0;
	if (vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL) != VK_SUCCESS || n == 0)
		return NULL;
	exts = calloc(n, sizeof(*exts));
	if (!exts)
		return NULL;
	if (vkEnumerateDeviceExtensionProperties(pd, NULL, &n, exts) != VK_SUCCESS)
	{
		free(exts);
		return NULL;
	}
	*pn = n;
	return exts;
}

static void
fill_devinfo(gpu_vk_devinfo *di, int index, VkPhysicalDevice pd, uint32_t inst_api)
{
	VkPhysicalDeviceProperties props;
	VkPhysicalDeviceProperties2 p2;
	VkPhysicalDeviceIDProperties idp;
	VkPhysicalDeviceDriverProperties drv;

	memset(di, 0, sizeof(*di));
	di->index = index;
	vkGetPhysicalDeviceProperties(pd, &props);
	copy_str(di->name, sizeof(di->name), props.deviceName);
	di->vendor_id = props.vendorID;
	di->device_id = props.deviceID;
	di->device_type = props.deviceType;
	di->api_version = props.apiVersion;
	di->driver_version = props.driverVersion;
	di->max_storage_buffer_range = props.limits.maxStorageBufferRange;
	di->max_shared_memory = props.limits.maxComputeSharedMemorySize;
	di->max_wg_invocations = props.limits.maxComputeWorkGroupInvocations;
	di->max_push_constants = props.limits.maxPushConstantsSize;

	if (inst_api >= VK_API_VERSION_1_1 && props.apiVersion >= VK_API_VERSION_1_1)
	{
		int		have_drv = (inst_api >= VK_API_VERSION_1_2 &&
							props.apiVersion >= VK_API_VERSION_1_2);

		memset(&idp, 0, sizeof(idp));
		idp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
		memset(&drv, 0, sizeof(drv));
		drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
		memset(&p2, 0, sizeof(p2));
		p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		p2.pNext = &idp;
		if (have_drv)
			idp.pNext = &drv;
		vkGetPhysicalDeviceProperties2(pd, &p2);
		for (int i = 0; i < VK_UUID_SIZE; i++)
			snprintf(di->uuid + i * 2, 3, "%02x", idp.deviceUUID[i]);
		if (have_drv)
		{
			copy_str(di->driver_name, sizeof(di->driver_name), drv.driverName);
			copy_str(di->driver_info, sizeof(di->driver_info), drv.driverInfo);
		}
	}
}

int
gpu_vk_enumerate(gpu_vk_devinfo *out, int max, char *err, size_t errlen)
{
	inst_t		in;
	VkPhysicalDevice *pd;
	uint32_t	n = 0;
	VkResult	rc;

	if (err && errlen)
		err[0] = '\0';
	rc = create_instance(&in, err, errlen);
	if (rc == VK_ERROR_INCOMPATIBLE_DRIVER)
	{
		/* no usable Vulkan driver at all: that is "zero devices", not a hard failure */
		return 0;
	}
	if (rc != VK_SUCCESS)
		return -1;
	pd = list_physical(in.instance, &n, err, errlen);
	if (!pd && err && err[0])
	{
		vkDestroyInstance(in.instance, NULL);
		return -1;
	}
	if (n > (uint32_t) max)
		n = max;
	for (uint32_t i = 0; i < n; i++)
		fill_devinfo(&out[i], (int) i, pd[i], in.api);
	free(pd);
	vkDestroyInstance(in.instance, NULL);
	return (int) n;
}

/* ------------------------------------------------------------------------ */
/* open / close                                                              */
/* ------------------------------------------------------------------------ */
static void
append_ext(gpu_vk_ctx *ctx, const char **names, uint32_t *pn, const char *name)
{
	size_t	len = strlen(ctx->ext_enabled);

	names[(*pn)++] = name;
	snprintf(ctx->ext_enabled + len, sizeof(ctx->ext_enabled) - len, "%s%s",
			 len ? " " : "", name);
}

int
gpu_vk_open(gpu_vk_ctx *ctx, int index)
{
	inst_t		in;
	VkPhysicalDevice *pds;
	uint32_t	npd = 0, nexts = 0, api, i;
	VkExtensionProperties *exts = NULL;
	VkPhysicalDeviceProperties props;
	VkResult	rc;
	int			have12, have13;
	int			ext_afloat, ext_hostimp, ext_extmem, ext_sgsc, ext_portab;

	/* feature query structs */
	VkPhysicalDeviceFeatures2 f2;
	VkPhysicalDeviceVulkan12Features f12;
	VkPhysicalDeviceVulkan13Features f13;
	VkPhysicalDeviceSubgroupSizeControlFeatures sgsc;
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afl;
	/* enable structs */
	VkPhysicalDeviceFeatures2 e2;
	VkPhysicalDeviceVulkan12Features e12;
	VkPhysicalDeviceVulkan13Features e13;
	VkPhysicalDeviceSubgroupSizeControlFeatures esgsc;
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT eafl;
	/* property structs */
	VkPhysicalDeviceProperties2 p2;
	VkPhysicalDeviceSubgroupProperties sgp;
	VkPhysicalDeviceSubgroupSizeControlProperties sgscp;
	VkPhysicalDeviceExternalMemoryHostPropertiesEXT ehp;
	VkPhysicalDeviceMaintenance4Properties m4p;
	void	  **tail;

	const char *dev_exts[8];
	uint32_t	ndev_exts = 0;
	float		prio = 1.0f;
	VkDeviceQueueCreateInfo qci;
	VkDeviceCreateInfo dci;
	uint32_t	nqf = 0;
	VkQueueFamilyProperties *qfp;
	int			qf = -1;
	gpu_vk_feat *ft;

	memset(ctx, 0, sizeof(*ctx));
	ctx->timeout_ms = GPU_VK_DEFAULT_TIMEOUT_MS;
	ctx->unified_memtype = ctx->hostvis_memtype = -1;
	ft = &ctx->feat;

	rc = create_instance(&in, ctx->err, sizeof(ctx->err));
	if (rc != VK_SUCCESS)
		return -1;
	ctx->instance = in.instance;
	pds = list_physical(ctx->instance, &npd, ctx->err, sizeof(ctx->err));
	if (index < 0 || (uint32_t) index >= npd)
	{
		if (!ctx->err[0])
			CTX_ERR(ctx, "device index %d out of range (%u devices)", index, npd);
		free(pds);
		gpu_vk_close(ctx);
		return -1;
	}
	ctx->phys = pds[index];
	free(pds);
	fill_devinfo(&ctx->info, index, ctx->phys, in.api);
	vkGetPhysicalDeviceProperties(ctx->phys, &props);
	ctx->limits = props.limits;
	api = min_u32(props.apiVersion, in.api);
	have12 = (api >= VK_API_VERSION_1_2);
	have13 = (api >= VK_API_VERSION_1_3);
	vkGetPhysicalDeviceMemoryProperties(ctx->phys, &ctx->mem);

	exts = list_device_ext(ctx->phys, &nexts);
	ext_afloat  = exts && has_ext(exts, nexts, "VK_EXT_shader_atomic_float");
	ext_hostimp = exts && has_ext(exts, nexts, "VK_EXT_external_memory_host");
	ext_extmem  = exts && has_ext(exts, nexts, "VK_KHR_external_memory");
	ext_sgsc    = exts && has_ext(exts, nexts, "VK_EXT_subgroup_size_control");
	ext_portab  = exts && has_ext(exts, nexts, "VK_KHR_portability_subset");
	free(exts);

	/* ---- features ---- */
	memset(&f2, 0, sizeof(f2));
	memset(&f12, 0, sizeof(f12));
	memset(&f13, 0, sizeof(f13));
	memset(&sgsc, 0, sizeof(sgsc));
	memset(&afl, 0, sizeof(afl));
	f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	sgsc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
	afl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	tail = &f2.pNext;
	if (have12) { *tail = &f12; tail = &f12.pNext; }
	if (have13) { *tail = &f13; tail = &f13.pNext; }
	else if (ext_sgsc) { *tail = &sgsc; tail = &sgsc.pNext; }
	if (ext_afloat) { *tail = &afl; tail = &afl.pNext; }
	vkGetPhysicalDeviceFeatures2(ctx->phys, &f2);

	ft->int64 = f2.features.shaderInt64 ? 1 : 0;
	ft->int16 = f2.features.shaderInt16 ? 1 : 0;
	ft->float64 = f2.features.shaderFloat64 ? 1 : 0;
	if (have12)
	{
		ft->int8 = f12.shaderInt8;
		ft->float16 = f12.shaderFloat16;
		ft->storage8 = f12.storageBuffer8BitAccess && f12.uniformAndStorageBuffer8BitAccess;
		ft->scalar_block_layout = f12.scalarBlockLayout;
		ft->buffer_int64_atomics = f12.shaderBufferInt64Atomics;
		ft->shared_int64_atomics = f12.shaderSharedInt64Atomics;
		ft->bda = f12.bufferDeviceAddress;
		ft->memory_model = f12.vulkanMemoryModel;
		ft->memory_model_device_scope = f12.vulkanMemoryModelDeviceScope;
		ft->subgroup_extended_types = f12.shaderSubgroupExtendedTypes;
	}
	ft->subgroup_size_control = have13 ? f13.subgroupSizeControl
				: (ext_sgsc ? sgsc.subgroupSizeControl : 0);
	ft->atomic_float32_add = ext_afloat && afl.shaderBufferFloat32AtomicAdd;

	/* ---- properties ---- */
	memset(&p2, 0, sizeof(p2));
	memset(&sgp, 0, sizeof(sgp));
	memset(&sgscp, 0, sizeof(sgscp));
	memset(&ehp, 0, sizeof(ehp));
	memset(&m4p, 0, sizeof(m4p));
	p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	sgp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	sgscp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
	ehp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
	m4p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES;
	tail = &p2.pNext;
	*tail = &sgp; tail = &sgp.pNext;
	if (have13 || ext_sgsc) { *tail = &sgscp; tail = &sgscp.pNext; }
	if (ext_hostimp) { *tail = &ehp; tail = &ehp.pNext; }
	if (have13) { *tail = &m4p; tail = &m4p.pNext; }
	vkGetPhysicalDeviceProperties2(ctx->phys, &p2);

	ft->subgroup_size = sgp.subgroupSize;
	ft->subgroup_ops = sgp.supportedOperations;
	ft->subgroup_compute = (sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? 1 : 0;
	ft->subgroup_size_min = ft->subgroup_size_max = sgp.subgroupSize;
	if ((have13 || ext_sgsc) && sgscp.minSubgroupSize && sgscp.maxSubgroupSize)
	{
		ft->subgroup_size_min = sgscp.minSubgroupSize;
		ft->subgroup_size_max = sgscp.maxSubgroupSize;
	}
	if (have13)
		ctx->max_buffer_size = m4p.maxBufferSize;

	/* ---- memory types ---- */
	for (i = 0; i < ctx->mem.memoryTypeCount; i++)
	{
		VkMemoryPropertyFlags fl = ctx->mem.memoryTypes[i].propertyFlags;
		VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
								   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		if ((fl & hv) != hv)
			continue;
		if (ctx->hostvis_memtype < 0)
			ctx->hostvis_memtype = (int) i;
		if ((fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && ctx->unified_memtype < 0)
			ctx->unified_memtype = (int) i;
	}
	for (i = 0; i < ctx->mem.memoryHeapCount; i++)
		if ((ctx->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
			ctx->mem.memoryHeaps[i].size > ctx->device_memory)
			ctx->device_memory = ctx->mem.memoryHeaps[i].size;

	/* ---- queue ---- */
	vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, NULL);
	qfp = calloc(nqf ? nqf : 1, sizeof(*qfp));
	if (!qfp)
	{
		CTX_ERR(ctx, "out of memory");
		gpu_vk_close(ctx);
		return -1;
	}
	vkGetPhysicalDeviceQueueFamilyProperties(ctx->phys, &nqf, qfp);
	for (i = 0; i < nqf; i++)
		if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			qf = (int) i;
			/* prefer a pure compute family if there is one */
			if (!(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
				break;
		}
	free(qfp);
	if (qf < 0)
	{
		CTX_ERR(ctx, "no compute queue family");
		gpu_vk_close(ctx);
		return -1;
	}
	ctx->queue_family = (uint32_t) qf;

	/* ---- enable exactly what the device reports ---- */
	memset(&e2, 0, sizeof(e2));
	memset(&e12, 0, sizeof(e12));
	memset(&e13, 0, sizeof(e13));
	memset(&esgsc, 0, sizeof(esgsc));
	memset(&eafl, 0, sizeof(eafl));
	e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	e13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	esgsc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
	eafl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	e2.features.shaderInt64 = f2.features.shaderInt64;
	e2.features.shaderInt16 = f2.features.shaderInt16;
	e2.features.shaderFloat64 = f2.features.shaderFloat64;
	tail = &e2.pNext;
	if (have12)
	{
		e12.shaderInt8 = f12.shaderInt8;
		e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
		e12.uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess;
		e12.scalarBlockLayout = f12.scalarBlockLayout;
		e12.shaderFloat16 = f12.shaderFloat16;
		e12.shaderBufferInt64Atomics = f12.shaderBufferInt64Atomics;
		e12.shaderSharedInt64Atomics = f12.shaderSharedInt64Atomics;
		e12.bufferDeviceAddress = f12.bufferDeviceAddress;
		e12.vulkanMemoryModel = f12.vulkanMemoryModel;
		e12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
		e12.shaderSubgroupExtendedTypes = f12.shaderSubgroupExtendedTypes;
		*tail = &e12; tail = &e12.pNext;
	}
	if (have13)
	{
		e13.subgroupSizeControl = f13.subgroupSizeControl;
		*tail = &e13; tail = &e13.pNext;
	}
	else if (ext_sgsc)
	{
		esgsc.subgroupSizeControl = sgsc.subgroupSizeControl;
		*tail = &esgsc; tail = &esgsc.pNext;
	}
	if (ext_afloat)
	{
		eafl.shaderBufferFloat32AtomicAdd = afl.shaderBufferFloat32AtomicAdd;
		*tail = &eafl; tail = &eafl.pNext;
	}

	if (ext_hostimp)
	{
		if (ext_extmem)
			append_ext(ctx, dev_exts, &ndev_exts, "VK_KHR_external_memory");
		append_ext(ctx, dev_exts, &ndev_exts, "VK_EXT_external_memory_host");
		ft->ext_host_import = 1;
		ctx->host_import_alignment = (uint32_t) ehp.minImportedHostPointerAlignment;
	}
	if (ext_afloat)
		append_ext(ctx, dev_exts, &ndev_exts, "VK_EXT_shader_atomic_float");
	if (!have13 && ext_sgsc)
		append_ext(ctx, dev_exts, &ndev_exts, "VK_EXT_subgroup_size_control");
	if (ext_portab)
	{
		append_ext(ctx, dev_exts, &ndev_exts, "VK_KHR_portability_subset");
		ft->ext_portability = 1;
	}

	memset(&qci, 0, sizeof(qci));
	qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qci.queueFamilyIndex = ctx->queue_family;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	memset(&dci, 0, sizeof(dci));
	dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.pNext = &e2;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = ndev_exts;
	dci.ppEnabledExtensionNames = dev_exts;
	rc = vkCreateDevice(ctx->phys, &dci, NULL, &ctx->dev);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateDevice failed: %s", vkr_str(rc));
		ctx->dev = VK_NULL_HANDLE;
		gpu_vk_close(ctx);
		return -1;
	}
	vkGetDeviceQueue(ctx->dev, ctx->queue_family, 0, &ctx->queue);

	{
		VkCommandPoolCreateInfo cpi;
		VkCommandBufferAllocateInfo cai;
		VkFenceCreateInfo fci;

		memset(&cpi, 0, sizeof(cpi));
		cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpi.queueFamilyIndex = ctx->queue_family;
		rc = vkCreateCommandPool(ctx->dev, &cpi, NULL, &ctx->cmd_pool);
		if (rc != VK_SUCCESS)
		{
			CTX_ERR(ctx, "vkCreateCommandPool failed: %s", vkr_str(rc));
			gpu_vk_close(ctx);
			return -1;
		}
		memset(&cai, 0, sizeof(cai));
		cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cai.commandPool = ctx->cmd_pool;
		cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cai.commandBufferCount = 1;
		rc = vkAllocateCommandBuffers(ctx->dev, &cai, &ctx->cmd);
		if (rc != VK_SUCCESS)
		{
			CTX_ERR(ctx, "vkAllocateCommandBuffers failed: %s", vkr_str(rc));
			gpu_vk_close(ctx);
			return -1;
		}
		memset(&fci, 0, sizeof(fci));
		fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		rc = vkCreateFence(ctx->dev, &fci, NULL, &ctx->fence);
		if (rc != VK_SUCCESS)
		{
			CTX_ERR(ctx, "vkCreateFence failed: %s", vkr_str(rc));
			gpu_vk_close(ctx);
			return -1;
		}
	}

	ctx->fn_bda = (PFN_vkGetBufferDeviceAddress)
		vkGetDeviceProcAddr(ctx->dev, "vkGetBufferDeviceAddress");
	if (!ctx->fn_bda)
		ctx->fn_bda = (PFN_vkGetBufferDeviceAddress)
			vkGetDeviceProcAddr(ctx->dev, "vkGetBufferDeviceAddressKHR");
	ctx->fn_host_ptr_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)
		vkGetDeviceProcAddr(ctx->dev, "vkGetMemoryHostPointerPropertiesEXT");
	return 0;
}

void
gpu_vk_close(gpu_vk_ctx *ctx)
{
	if (ctx->dev)
	{
		if (!ctx->lost)
			vkDeviceWaitIdle(ctx->dev);
		if (ctx->fence)
			vkDestroyFence(ctx->dev, ctx->fence, NULL);
		if (ctx->cmd_pool)
			vkDestroyCommandPool(ctx->dev, ctx->cmd_pool, NULL);
		vkDestroyDevice(ctx->dev, NULL);
	}
	if (ctx->instance)
		vkDestroyInstance(ctx->instance, NULL);
	ctx->fence = VK_NULL_HANDLE;
	ctx->cmd_pool = VK_NULL_HANDLE;
	ctx->dev = VK_NULL_HANDLE;
	ctx->instance = VK_NULL_HANDLE;
}

void
gpu_vk_describe(const gpu_vk_ctx *ctx, char *buf, size_t buflen)
{
	const gpu_vk_feat *f = &ctx->feat;

	snprintf(buf, buflen,
			 "features: int64=%d int16=%d int8=%d fp64=%d fp16=%d storage8=%d scalar=%d "
			 "i64atomics=%d/%d bda=%d memmodel=%d/%d sgext=%d sgsizectl=%d f32atomic=%d | "
			 "subgroup size=%u [%u..%u] ops=0x%x compute=%d | exts: %s | "
			 "unified_memtype=%d hostimport_align=%u",
			 f->int64, f->int16, f->int8, f->float64, f->float16, f->storage8,
			 f->scalar_block_layout, f->buffer_int64_atomics, f->shared_int64_atomics,
			 f->bda, f->memory_model, f->memory_model_device_scope,
			 f->subgroup_extended_types, f->subgroup_size_control, f->atomic_float32_add,
			 f->subgroup_size, f->subgroup_size_min, f->subgroup_size_max,
			 f->subgroup_ops, f->subgroup_compute, ctx->ext_enabled,
			 ctx->unified_memtype, ctx->host_import_alignment);
}

/* ------------------------------------------------------------------------ */
/* buffers                                                                   */
/* ------------------------------------------------------------------------ */
/* pick a memory type from 'bits': prefer unified, then host-visible+coherent */
static int
pick_memtype(const gpu_vk_ctx *ctx, uint32_t bits)
{
	if (ctx->unified_memtype >= 0 && (bits & (1u << ctx->unified_memtype)))
		return ctx->unified_memtype;
	if (ctx->hostvis_memtype >= 0 && (bits & (1u << ctx->hostvis_memtype)))
		return ctx->hostvis_memtype;
	for (uint32_t i = 0; i < ctx->mem.memoryTypeCount; i++)
	{
		VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
								   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		if ((bits & (1u << i)) &&
			(ctx->mem.memoryTypes[i].propertyFlags & hv) == hv)
			return (int) i;
	}
	return -1;
}

int
gpu_vk_buffer_create(gpu_vk_ctx *ctx, gpu_vk_buffer *b, size_t size, int want_bda)
{
	VkBufferCreateInfo bci;
	VkMemoryRequirements req;
	VkMemoryAllocateInfo mai;
	VkMemoryAllocateFlagsInfo mafi;
	VkResult	rc;
	int			mt;

	memset(b, 0, sizeof(*b));
	if (size == 0)
	{
		CTX_ERR(ctx, "buffer_create: zero size");
		return -1;
	}
	if (want_bda && (!ctx->feat.bda || !ctx->fn_bda))
	{
		CTX_ERR(ctx, "buffer_create: bufferDeviceAddress is not enabled");
		return -1;
	}
	memset(&bci, 0, sizeof(bci));
	bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.size = size;
	bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
				VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	if (want_bda)
		bci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	rc = vkCreateBuffer(ctx->dev, &bci, NULL, &b->buf);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateBuffer(%zu) failed: %s", size, vkr_str(rc));
		b->buf = VK_NULL_HANDLE;
		return -1;
	}
	vkGetBufferMemoryRequirements(ctx->dev, b->buf, &req);
	mt = pick_memtype(ctx, req.memoryTypeBits);
	if (mt < 0)
	{
		CTX_ERR(ctx, "no host-visible coherent memory type for buffer");
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	memset(&mai, 0, sizeof(mai));
	mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.allocationSize = req.size;
	mai.memoryTypeIndex = (uint32_t) mt;
	if (want_bda)
	{
		memset(&mafi, 0, sizeof(mafi));
		mafi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		mafi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		mai.pNext = &mafi;
	}
	rc = vkAllocateMemory(ctx->dev, &mai, NULL, &b->mem);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkAllocateMemory(%zu) failed: %s", (size_t) req.size, vkr_str(rc));
		b->mem = VK_NULL_HANDLE;
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	rc = vkBindBufferMemory(ctx->dev, b->buf, b->mem, 0);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkBindBufferMemory failed: %s", vkr_str(rc));
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	rc = vkMapMemory(ctx->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkMapMemory failed: %s", vkr_str(rc));
		b->map = NULL;
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	b->size = size;
	b->unified = (mt == ctx->unified_memtype);
	if (want_bda)
	{
		VkBufferDeviceAddressInfo bai;

		memset(&bai, 0, sizeof(bai));
		bai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		bai.buffer = b->buf;
		b->addr = (uint64_t) ctx->fn_bda(ctx->dev, &bai);
	}
	return 0;
}

int
gpu_vk_buffer_import_host(gpu_vk_ctx *ctx, gpu_vk_buffer *b, void *ptr, size_t size)
{
	VkMemoryHostPointerPropertiesEXT hpp;
	VkExternalMemoryBufferCreateInfo ebci;
	VkBufferCreateInfo bci;
	VkMemoryRequirements req;
	VkImportMemoryHostPointerInfoEXT imp;
	VkMemoryAllocateInfo mai;
	VkResult	rc;
	uint32_t	bits;
	int			mt;

	memset(b, 0, sizeof(*b));
	if (!ctx->feat.ext_host_import || !ctx->fn_host_ptr_props || !ctx->host_import_alignment)
	{
		CTX_ERR(ctx, "VK_EXT_external_memory_host is not enabled");
		return -1;
	}
	if (!ptr || size == 0 ||
		((uintptr_t) ptr % ctx->host_import_alignment) != 0 ||
		(size % ctx->host_import_alignment) != 0)
	{
		CTX_ERR(ctx, "host import: ptr/size must be multiples of %u", ctx->host_import_alignment);
		return -1;
	}
	memset(&hpp, 0, sizeof(hpp));
	hpp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	rc = ctx->fn_host_ptr_props(ctx->dev,
								VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
								ptr, &hpp);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkGetMemoryHostPointerPropertiesEXT failed: %s", vkr_str(rc));
		return -1;
	}
	memset(&ebci, 0, sizeof(ebci));
	ebci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	ebci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	memset(&bci, 0, sizeof(bci));
	bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bci.pNext = &ebci;
	bci.size = size;
	bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
				VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	rc = vkCreateBuffer(ctx->dev, &bci, NULL, &b->buf);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateBuffer(import) failed: %s", vkr_str(rc));
		b->buf = VK_NULL_HANDLE;
		return -1;
	}
	vkGetBufferMemoryRequirements(ctx->dev, b->buf, &req);
	bits = req.memoryTypeBits & hpp.memoryTypeBits;
	if (!bits)
	{
		CTX_ERR(ctx, "host import: no compatible memory type (buffer 0x%x, pointer 0x%x)",
				req.memoryTypeBits, hpp.memoryTypeBits);
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	mt = pick_memtype(ctx, bits);
	if (mt < 0)
	{
		for (uint32_t i = 0; i < ctx->mem.memoryTypeCount; i++)
			if (bits & (1u << i)) { mt = (int) i; break; }
	}
	memset(&imp, 0, sizeof(imp));
	imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
	imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	imp.pHostPointer = ptr;
	memset(&mai, 0, sizeof(mai));
	mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	mai.pNext = &imp;
	mai.allocationSize = (req.size + ctx->host_import_alignment - 1) /
						 ctx->host_import_alignment * ctx->host_import_alignment;
	mai.memoryTypeIndex = (uint32_t) mt;
	rc = vkAllocateMemory(ctx->dev, &mai, NULL, &b->mem);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkAllocateMemory(host import) failed: %s", vkr_str(rc));
		b->mem = VK_NULL_HANDLE;
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	rc = vkBindBufferMemory(ctx->dev, b->buf, b->mem, 0);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkBindBufferMemory(import) failed: %s", vkr_str(rc));
		gpu_vk_buffer_destroy(ctx, b);
		return -1;
	}
	b->map = ptr;
	b->size = size;
	b->imported = 1;
	return 0;
}

void
gpu_vk_buffer_destroy(gpu_vk_ctx *ctx, gpu_vk_buffer *b)
{
	if (b->mem && b->map && !b->imported)
		vkUnmapMemory(ctx->dev, b->mem);
	if (b->buf)
		vkDestroyBuffer(ctx->dev, b->buf, NULL);
	if (b->mem)
		vkFreeMemory(ctx->dev, b->mem, NULL);
	memset(b, 0, sizeof(*b));
}

/* ------------------------------------------------------------------------ */
/* pipeline / dispatch                                                       */
/* ------------------------------------------------------------------------ */
int
gpu_vk_pipeline_create(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, const void *spirv,
					   size_t spirv_bytes, uint32_t nbind, uint32_t pc_size)
{
	VkShaderModuleCreateInfo smi;
	VkDescriptorSetLayoutBinding *binds = NULL;
	VkDescriptorSetLayoutCreateInfo dsli;
	VkPushConstantRange pcr;
	VkPipelineLayoutCreateInfo pli;
	VkComputePipelineCreateInfo cpi;
	VkResult	rc;

	memset(p, 0, sizeof(*p));
	p->nbind = nbind;
	p->pc_size = pc_size;
	if (ctx->lost)
	{
		CTX_ERR(ctx, "device is lost");
		return -1;
	}
	if (spirv_bytes == 0 || (spirv_bytes % 4) != 0 || ((uintptr_t) spirv % 4) != 0)
	{
		CTX_ERR(ctx, "invalid SPIR-V blob (size %zu)", spirv_bytes);
		return -1;
	}
	if (pc_size > ctx->limits.maxPushConstantsSize)
	{
		CTX_ERR(ctx, "push constant range %u exceeds the device limit %u",
				pc_size, ctx->limits.maxPushConstantsSize);
		return -1;
	}
	memset(&smi, 0, sizeof(smi));
	smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smi.codeSize = spirv_bytes;
	smi.pCode = (const uint32_t *) spirv;
	rc = vkCreateShaderModule(ctx->dev, &smi, NULL, &p->module);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateShaderModule failed: %s", vkr_str(rc));
		p->module = VK_NULL_HANDLE;
		return -1;
	}
	if (nbind > 0)
	{
		binds = calloc(nbind, sizeof(*binds));
		if (!binds)
		{
			CTX_ERR(ctx, "out of memory");
			gpu_vk_pipeline_destroy(ctx, p);
			return -1;
		}
		for (uint32_t i = 0; i < nbind; i++)
		{
			binds[i].binding = i;
			binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			binds[i].descriptorCount = 1;
			binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}
	}
	memset(&dsli, 0, sizeof(dsli));
	dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dsli.bindingCount = nbind;
	dsli.pBindings = binds;
	rc = vkCreateDescriptorSetLayout(ctx->dev, &dsli, NULL, &p->dsl);
	free(binds);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateDescriptorSetLayout failed: %s", vkr_str(rc));
		p->dsl = VK_NULL_HANDLE;
		gpu_vk_pipeline_destroy(ctx, p);
		return -1;
	}
	memset(&pli, 0, sizeof(pli));
	pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pli.setLayoutCount = 1;
	pli.pSetLayouts = &p->dsl;
	if (pc_size > 0)
	{
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = pc_size;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges = &pcr;
	}
	rc = vkCreatePipelineLayout(ctx->dev, &pli, NULL, &p->layout);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreatePipelineLayout failed: %s", vkr_str(rc));
		p->layout = VK_NULL_HANDLE;
		gpu_vk_pipeline_destroy(ctx, p);
		return -1;
	}
	memset(&cpi, 0, sizeof(cpi));
	cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.module = p->module;
	cpi.stage.pName = "main";
	cpi.layout = p->layout;
	rc = vkCreateComputePipelines(ctx->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &p->pipeline);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkCreateComputePipelines failed (shader compiler rejected the "
				"module?): %s", vkr_str(rc));
		p->pipeline = VK_NULL_HANDLE;
		if (rc == VK_ERROR_DEVICE_LOST)
			ctx->lost = 1;
		gpu_vk_pipeline_destroy(ctx, p);
		return -1;
	}
	if (nbind > 0)
	{
		VkDescriptorPoolSize ps;
		VkDescriptorPoolCreateInfo dpi;
		VkDescriptorSetAllocateInfo dai;

		ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		ps.descriptorCount = nbind;
		memset(&dpi, 0, sizeof(dpi));
		dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		dpi.maxSets = 1;
		dpi.poolSizeCount = 1;
		dpi.pPoolSizes = &ps;
		rc = vkCreateDescriptorPool(ctx->dev, &dpi, NULL, &p->dpool);
		if (rc != VK_SUCCESS)
		{
			CTX_ERR(ctx, "vkCreateDescriptorPool failed: %s", vkr_str(rc));
			p->dpool = VK_NULL_HANDLE;
			gpu_vk_pipeline_destroy(ctx, p);
			return -1;
		}
		memset(&dai, 0, sizeof(dai));
		dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dai.descriptorPool = p->dpool;
		dai.descriptorSetCount = 1;
		dai.pSetLayouts = &p->dsl;
		rc = vkAllocateDescriptorSets(ctx->dev, &dai, &p->dset);
		if (rc != VK_SUCCESS)
		{
			CTX_ERR(ctx, "vkAllocateDescriptorSets failed: %s", vkr_str(rc));
			p->dset = VK_NULL_HANDLE;
			gpu_vk_pipeline_destroy(ctx, p);
			return -1;
		}
	}
	return 0;
}

void
gpu_vk_pipeline_destroy(gpu_vk_ctx *ctx, gpu_vk_pipeline *p)
{
	if (p->dpool)
		vkDestroyDescriptorPool(ctx->dev, p->dpool, NULL);	/* frees the set */
	if (p->pipeline)
		vkDestroyPipeline(ctx->dev, p->pipeline, NULL);
	if (p->layout)
		vkDestroyPipelineLayout(ctx->dev, p->layout, NULL);
	if (p->dsl)
		vkDestroyDescriptorSetLayout(ctx->dev, p->dsl, NULL);
	if (p->module)
		vkDestroyShaderModule(ctx->dev, p->module, NULL);
	memset(p, 0, sizeof(*p));
}

int
gpu_vk_dispatch(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, gpu_vk_buffer *const *bufs,
				const void *pc, uint32_t gx, uint32_t gy, uint32_t gz, double *elapsed_ms)
{
	VkCommandBufferBeginInfo cbi;
	VkMemoryBarrier mb;
	VkSubmitInfo si;
	VkResult	rc;
	double		t0;

	if (elapsed_ms)
		*elapsed_ms = 0.0;
	if (ctx->lost)
	{
		CTX_ERR(ctx, "device is lost");
		return -1;
	}
	if (p->nbind > 0)
	{
		VkDescriptorBufferInfo *dbi = calloc(p->nbind, sizeof(*dbi));
		VkWriteDescriptorSet *wds = calloc(p->nbind, sizeof(*wds));

		if (!dbi || !wds)
		{
			free(dbi);
			free(wds);
			CTX_ERR(ctx, "out of memory");
			return -1;
		}
		for (uint32_t i = 0; i < p->nbind; i++)
		{
			if (!bufs || !bufs[i] || !bufs[i]->buf)
			{
				free(dbi);
				free(wds);
				CTX_ERR(ctx, "dispatch: buffer %u is missing", i);
				return -1;
			}
			dbi[i].buffer = bufs[i]->buf;
			dbi[i].offset = 0;
			dbi[i].range = VK_WHOLE_SIZE;
			wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			wds[i].dstSet = p->dset;
			wds[i].dstBinding = i;
			wds[i].descriptorCount = 1;
			wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			wds[i].pBufferInfo = &dbi[i];
		}
		vkUpdateDescriptorSets(ctx->dev, p->nbind, wds, 0, NULL);
		free(dbi);
		free(wds);
	}

	rc = vkResetCommandBuffer(ctx->cmd, 0);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkResetCommandBuffer failed: %s", vkr_str(rc));
		return -1;
	}
	memset(&cbi, 0, sizeof(cbi));
	cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	rc = vkBeginCommandBuffer(ctx->cmd, &cbi);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkBeginCommandBuffer failed: %s", vkr_str(rc));
		return -1;
	}
	vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
	if (p->nbind > 0)
		vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout,
								0, 1, &p->dset, 0, NULL);
	if (p->pc_size > 0 && pc)
		vkCmdPushConstants(ctx->cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
						   p->pc_size, pc);
	vkCmdDispatch(ctx->cmd, gx, gy, gz);
	memset(&mb, 0, sizeof(mb));
	mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
	rc = vkEndCommandBuffer(ctx->cmd);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkEndCommandBuffer failed: %s", vkr_str(rc));
		return -1;
	}

	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &ctx->cmd;
	t0 = now_ms();
	rc = vkQueueSubmit(ctx->queue, 1, &si, ctx->fence);
	if (rc != VK_SUCCESS)
	{
		if (rc == VK_ERROR_DEVICE_LOST)
			ctx->lost = 1;
		CTX_ERR(ctx, "vkQueueSubmit failed: %s", vkr_str(rc));
		return -1;
	}
	rc = vkWaitForFences(ctx->dev, 1, &ctx->fence, VK_TRUE,
						 (uint64_t) ctx->timeout_ms * 1000000ULL);
	if (elapsed_ms)
		*elapsed_ms = now_ms() - t0;
	if (rc == VK_TIMEOUT)
	{
		ctx->lost = 1;
		CTX_ERR(ctx, "kernel did not finish within %u ms (device marked lost)",
				ctx->timeout_ms);
		return -1;
	}
	if (rc != VK_SUCCESS)
	{
		if (rc == VK_ERROR_DEVICE_LOST)
			ctx->lost = 1;
		CTX_ERR(ctx, "vkWaitForFences failed: %s", vkr_str(rc));
		return -1;
	}
	rc = vkResetFences(ctx->dev, 1, &ctx->fence);
	if (rc != VK_SUCCESS)
	{
		CTX_ERR(ctx, "vkResetFences failed: %s", vkr_str(rc));
		return -1;
	}
	return 0;
}
