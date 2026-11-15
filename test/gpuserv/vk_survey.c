/*
 * vk_survey.c: create compute pipelines from SPIR-V files in a fresh process and say whether the Vulkan driver accepts them.
 *
 *   vk_survey [-l] [-d N] <nbind> <push-const bytes> <file.spv> [<nbind> <pc> <file.spv> ...]
 *
 * One line per file: size, "ok in X s" or "FAIL: <VkResult>". -l lists the devices and their features and stops; -d picks a device
 * (default 0). The device is created with the same features gpu_vk_open() enables (everything the device reports), the pipeline
 * layout is the one of gpu_vk.c (nbind storage buffers in set 0, one compute push-constant range). No dispatch, no memory.
 *
 * Portable on purpose (the Vulkan loader is loaded at run time, only the Vulkan headers are needed) so that it also builds for
 * Windows (MinGW, MSVC or clang) and checks the compiler of the NVIDIA driver there, where the service does not run:
 *   cc -O2 -o vk_survey vk_survey.c                       (Linux, macOS)
 *   x86_64-w64-mingw32-gcc -O2 -o vk_survey.exe vk_survey.c
 * This is NOT a reproduction of A17 (a pipeline that fails inside the long-running service but not in a fresh process).
 */
#define VK_NO_PROTOTYPES
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
static double now_s(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double) c.QuadPart / (double) f.QuadPart; }
#else
#include <dlfcn.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
#endif

static PFN_vkGetInstanceProcAddr pGIPA;
#define GI(i, n) PFN_##n n = (PFN_##n) pGIPA(i, #n)

static int
load_loader(void)
{
#ifdef _WIN32
	HMODULE h = LoadLibraryA("vulkan-1.dll");
	if (h)
		pGIPA = (PFN_vkGetInstanceProcAddr) GetProcAddress(h, "vkGetInstanceProcAddr");
#else
	static const char *names[] = {"libvulkan.so.1", "libvulkan.1.dylib", "libvulkan.dylib", "libvulkan.so",
								  "/opt/homebrew/lib/libvulkan.1.dylib", "/usr/local/lib/libvulkan.1.dylib"};
	const char *env = getenv("VKS_LOADER");		/* a loader path of your own */
	void *h = env ? dlopen(env, RTLD_NOW) : NULL;
	for (int i = 0; i < 6 && !h; i++)
		h = dlopen(names[i], RTLD_NOW);
	if (h)
		pGIPA = (PFN_vkGetInstanceProcAddr) dlsym(h, "vkGetInstanceProcAddr");
#endif
	return pGIPA ? 0 : -1;
}

static unsigned char *
slurp(const char *path, size_t *len)
{
	FILE *fp = fopen(path, "rb");
	unsigned char *b;
	long n;

	if (!fp)
		return NULL;
	fseek(fp, 0, SEEK_END);
	n = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	b = malloc(n > 0 ? (size_t) n : 4);
	if (b && n > 0 && fread(b, 1, (size_t) n, fp) != (size_t) n)
	{
		free(b);
		b = NULL;
	}
	fclose(fp);
	*len = b ? (size_t) n : 0;
	return b;
}

int
main(int argc, char **argv)
{
	int			a = 1, list = 0, dev_index = 0, nfail = 0;
	VkInstance	inst;
	VkApplicationInfo ai;
	VkInstanceCreateInfo ici;
	uint32_t	n = 0, nexts = 0, nqf = 0, qf = 0;
	VkPhysicalDevice pds[16], phys;
	VkPhysicalDeviceProperties props;
	VkPhysicalDeviceFeatures2 f2, e2;
	VkPhysicalDeviceVulkan12Features f12, e12;
	VkPhysicalDeviceVulkan13Features f13, e13;
	VkDevice	dev;
	VkResult	rc;
	VkExtensionProperties *exts;
	VkQueueFamilyProperties qfp[16];
	VkDeviceQueueCreateInfo qci;
	VkDeviceCreateInfo dci;
	float		prio = 1.0f;
	const char *dev_exts[4];
	uint32_t	ndev_exts = 0;
	int			have_sgsc = 0, have_afl = 0, have_port = 0;
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT afl, eafl;
	void	  **tail;

	while (a < argc && argv[a][0] == '-')
	{
		if (!strcmp(argv[a], "-l"))
			list = 1;
		else if (!strcmp(argv[a], "-d") && a + 1 < argc)
			dev_index = atoi(argv[++a]);
		else
			break;
		a++;
	}
	if (!list && argc - a < 3)
	{
		fprintf(stderr, "usage: %s [-l] [-d N] <nbind> <pc bytes> <file.spv> ...\n", argv[0]);
		return 2;
	}
	if (load_loader() != 0)
	{
		fprintf(stderr, "no Vulkan loader (vulkan-1.dll / libvulkan.so.1)\n");
		return 2;
	}
	{
		GI(NULL, vkCreateInstance);
		GI(NULL, vkEnumerateInstanceExtensionProperties);
		const char *iexts[1] = {"VK_KHR_portability_enumeration"};
		uint32_t	nie = 0;
		VkExtensionProperties *ie;
		memset(&ai, 0, sizeof(ai));
		ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		ai.pApplicationName = "vk_survey";
		ai.apiVersion = VK_API_VERSION_1_3;
		memset(&ici, 0, sizeof(ici));
		ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ici.pApplicationInfo = &ai;
		/* MoltenVK is a portability implementation: the loader hides it unless the instance asks for it */
		vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
		ie = calloc(nie ? nie : 1, sizeof(*ie));
		vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
		for (uint32_t i = 0; i < nie; i++)
			if (!strcmp(ie[i].extensionName, iexts[0]))
			{
				ici.flags |= 0x00000001;		/* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
				ici.enabledExtensionCount = 1;
				ici.ppEnabledExtensionNames = iexts;
			}
		free(ie);
		if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
		{
			/* 1.3 is refused by an old loader: retry 1.1 */
			ai.apiVersion = VK_API_VERSION_1_1;
			if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS)
			{
				fprintf(stderr, "vkCreateInstance failed\n");
				return 2;
			}
		}
	}
	GI(inst, vkEnumeratePhysicalDevices);
	GI(inst, vkGetPhysicalDeviceProperties);
	GI(inst, vkGetPhysicalDeviceFeatures2);
	GI(inst, vkGetPhysicalDeviceQueueFamilyProperties);
	GI(inst, vkEnumerateDeviceExtensionProperties);
	GI(inst, vkCreateDevice);
	GI(inst, vkGetDeviceProcAddr);
	n = 16;
	vkEnumeratePhysicalDevices(inst, &n, pds);
	if (n == 0)
	{
		fprintf(stderr, "no Vulkan device\n");
		return 2;
	}
	for (uint32_t i = 0; i < n; i++)
	{
		vkGetPhysicalDeviceProperties(pds[i], &props);
		printf("device %u: %s, API %u.%u.%u, driver 0x%x, vendor 0x%x, type %d\n", i, props.deviceName,
			   VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
			   props.driverVersion, props.vendorID, (int) props.deviceType);
	}
	if (dev_index < 0 || (uint32_t) dev_index >= n)
	{
		fprintf(stderr, "no device %d\n", dev_index);
		return 2;
	}
	phys = pds[dev_index];
	vkGetPhysicalDeviceProperties(phys, &props);

	exts = NULL;
	vkEnumerateDeviceExtensionProperties(phys, NULL, &nexts, NULL);
	exts = calloc(nexts ? nexts : 1, sizeof(*exts));
	vkEnumerateDeviceExtensionProperties(phys, NULL, &nexts, exts);
	for (uint32_t i = 0; i < nexts; i++)
	{
		if (!strcmp(exts[i].extensionName, "VK_EXT_subgroup_size_control"))
			have_sgsc = 1;
		if (!strcmp(exts[i].extensionName, "VK_EXT_shader_atomic_float"))
			have_afl = 1;
		if (!strcmp(exts[i].extensionName, "VK_KHR_portability_subset"))
			have_port = 1;
	}
	free(exts);

	/* query what the device reports and enable exactly that (gpu_vk_open() does the same) */
	memset(&f2, 0, sizeof(f2)); memset(&f12, 0, sizeof(f12)); memset(&f13, 0, sizeof(f13)); memset(&afl, 0, sizeof(afl));
	f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	afl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	tail = &f2.pNext;
	if (props.apiVersion >= VK_API_VERSION_1_2)
	{
		*tail = &f12; tail = &f12.pNext;
	}
	if (props.apiVersion >= VK_API_VERSION_1_3)
	{
		*tail = &f13; tail = &f13.pNext;
	}
	if (have_afl)
	{
		*tail = &afl; tail = &afl.pNext;
	}
	vkGetPhysicalDeviceFeatures2(phys, &f2);
	if (list || getenv("VKS_VERBOSE"))
		printf("features: int64 %d int16 %d float64 %d | int8 %d storage8 %d scalarBlock %d float16 %d bufInt64Atomics %d "
			   "sharedInt64Atomics %d BDA %d memModel %d | subgroupSizeControl %d | atomicFloat %d\n",
			   (int) f2.features.shaderInt64, (int) f2.features.shaderInt16, (int) f2.features.shaderFloat64,
			   (int) f12.shaderInt8, (int) f12.storageBuffer8BitAccess, (int) f12.scalarBlockLayout, (int) f12.shaderFloat16,
			   (int) f12.shaderBufferInt64Atomics, (int) f12.shaderSharedInt64Atomics, (int) f12.bufferDeviceAddress,
			   (int) f12.vulkanMemoryModel, (int) f13.subgroupSizeControl, (int) afl.shaderBufferFloat32AtomicAdd);
	if (list)
		return 0;

	memset(&e2, 0, sizeof(e2)); memset(&e12, 0, sizeof(e12)); memset(&e13, 0, sizeof(e13)); memset(&eafl, 0, sizeof(eafl));
	e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	e13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	eafl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	e2.features.shaderInt64 = f2.features.shaderInt64;
	e2.features.shaderInt16 = f2.features.shaderInt16;
	e2.features.shaderFloat64 = f2.features.shaderFloat64;
	tail = &e2.pNext;
	if (props.apiVersion >= VK_API_VERSION_1_2)
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
	if (props.apiVersion >= VK_API_VERSION_1_3)
	{
		e13.subgroupSizeControl = f13.subgroupSizeControl;
		*tail = &e13; tail = &e13.pNext;
	}
	if (have_afl)
	{
		eafl.shaderBufferFloat32AtomicAdd = afl.shaderBufferFloat32AtomicAdd;
		*tail = &eafl; tail = &eafl.pNext;
		dev_exts[ndev_exts++] = "VK_EXT_shader_atomic_float";
	}
	if (have_port)
		dev_exts[ndev_exts++] = "VK_KHR_portability_subset";
	(void) have_sgsc;

	nqf = 16;
	vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfp);
	for (uint32_t i = 0; i < nqf; i++)
		if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			qf = i;
			if (!(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
				break;
		}
	memset(&qci, 0, sizeof(qci));
	qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qci.queueFamilyIndex = qf;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	memset(&dci, 0, sizeof(dci));
	dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.pNext = &e2;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = ndev_exts;
	dci.ppEnabledExtensionNames = dev_exts;
	rc = vkCreateDevice(phys, &dci, NULL, &dev);
	if (rc != VK_SUCCESS)
	{
		fprintf(stderr, "vkCreateDevice failed: %d\n", (int) rc);
		return 2;
	}
	{
		PFN_vkCreateShaderModule vkCreateShaderModule = (PFN_vkCreateShaderModule) vkGetDeviceProcAddr(dev, "vkCreateShaderModule");
		PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout) vkGetDeviceProcAddr(dev, "vkCreateDescriptorSetLayout");
		PFN_vkCreatePipelineLayout vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout) vkGetDeviceProcAddr(dev, "vkCreatePipelineLayout");
		PFN_vkCreateComputePipelines vkCreateComputePipelines = (PFN_vkCreateComputePipelines) vkGetDeviceProcAddr(dev, "vkCreateComputePipelines");
		PFN_vkDestroyPipeline vkDestroyPipeline = (PFN_vkDestroyPipeline) vkGetDeviceProcAddr(dev, "vkDestroyPipeline");
		PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout) vkGetDeviceProcAddr(dev, "vkDestroyPipelineLayout");
		PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout) vkGetDeviceProcAddr(dev, "vkDestroyDescriptorSetLayout");
		PFN_vkDestroyShaderModule vkDestroyShaderModule = (PFN_vkDestroyShaderModule) vkGetDeviceProcAddr(dev, "vkDestroyShaderModule");

		for (; a + 2 < argc; a += 3)
		{
			uint32_t	nbind = (uint32_t) atoi(argv[a]), pc = (uint32_t) atoi(argv[a + 1]);
			const char *path = argv[a + 2];
			size_t		len;
			unsigned char *buf = slurp(path, &len);
			uint32_t   *code;
			VkShaderModuleCreateInfo smi;
			VkDescriptorSetLayoutBinding *binds;
			VkDescriptorSetLayoutCreateInfo dsli;
			VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
			VkPushConstantRange pcr;
			VkPipelineLayoutCreateInfo pli;
			VkPipelineLayout layout = VK_NULL_HANDLE;
			VkShaderModule mod = VK_NULL_HANDLE;
			VkComputePipelineCreateInfo cpi;
			VkPipeline	pipe = VK_NULL_HANDLE;
			double		t0;

			if (!buf || len == 0 || (len % 4) != 0)
			{
				printf("%s: cannot read (or not SPIR-V)\n", path);
				nfail++;
				free(buf);
				continue;
			}
			code = malloc(len);		/* malloc'd memory is aligned for uint32_t */
			memcpy(code, buf, len);
			free(buf);
			memset(&smi, 0, sizeof(smi));
			smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			smi.codeSize = len;
			smi.pCode = code;
			rc = vkCreateShaderModule(dev, &smi, NULL, &mod);
			if (rc != VK_SUCCESS)
			{
				printf("%s: %zu bytes of SPIR-V: FAIL vkCreateShaderModule rc=%d\n", path, len, (int) rc);
				nfail++;
				free(code);
				continue;
			}
			binds = calloc(nbind ? nbind : 1, sizeof(*binds));
			for (uint32_t i = 0; i < nbind; i++)
			{
				binds[i].binding = i;
				binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				binds[i].descriptorCount = 1;
				binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			}
			memset(&dsli, 0, sizeof(dsli));
			dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			dsli.bindingCount = nbind;
			dsli.pBindings = binds;
			vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl);
			free(binds);
			memset(&pli, 0, sizeof(pli));
			pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pli.setLayoutCount = 1;
			pli.pSetLayouts = &dsl;
			if (pc > 0)
			{
				pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pcr.offset = 0;
				pcr.size = pc;
				pli.pushConstantRangeCount = 1;
				pli.pPushConstantRanges = &pcr;
			}
			vkCreatePipelineLayout(dev, &pli, NULL, &layout);
			memset(&cpi, 0, sizeof(cpi));
			cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			cpi.stage.module = mod;
			cpi.stage.pName = "main";
			cpi.layout = layout;
			t0 = now_s();
			rc = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
			if (rc == VK_SUCCESS)
				printf("%s: %zu bytes of SPIR-V: ok in %.1f s\n", path, len, now_s() - t0);
			else
			{
				printf("%s: %zu bytes of SPIR-V: FAIL vkCreateComputePipelines rc=%d after %.3f s\n", path, len, (int) rc, now_s() - t0);
				nfail++;
			}
			fflush(stdout);
			if (pipe)
				vkDestroyPipeline(dev, pipe, NULL);
			vkDestroyPipelineLayout(dev, layout, NULL);
			vkDestroyDescriptorSetLayout(dev, dsl, NULL);
			vkDestroyShaderModule(dev, mod, NULL);
			free(code);
		}
	}
	return nfail ? 1 : 0;
}
