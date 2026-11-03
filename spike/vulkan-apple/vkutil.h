/*
 * vkutil.h - minimal Vulkan compute boilerplate for the spike harnesses.
 * Header-only, single translation unit. Unified-memory buffers only.
 */
#ifndef VKUTIL_H
#define VKUTIL_H
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VK_CHECK(expr)														\
	do { VkResult __rc = (expr);											\
		 if (__rc != VK_SUCCESS) {											\
			 fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #expr, __rc); \
			 exit(2); } } while (0)

typedef struct
{
	VkBuffer		buf;
	VkDeviceMemory	mem;
	void		   *map;
	VkDeviceSize	size;
} VkBuf;

static VkInstance				vk_inst;
static VkPhysicalDevice			vk_pdev;
static VkDevice					vk_dev;
static VkQueue					vk_queue;
static uint32_t					vk_qfamily;
static VkCommandPool			vk_cpool;
static VkPhysicalDeviceMemoryProperties vk_memprops;
static VkDescriptorSetLayout	vk_dsl;
static VkPipelineLayout			vk_playout;
static VkDescriptorSet			vk_dset;

/* VK_EXT_external_memory_host: wrap existing host memory as a buffer (zero copy) */
static int			vk_host_import_ok;			/* extension present and enabled */
static VkDeviceSize	vk_host_align;				/* minImportedHostPointerAlignment */
static PFN_vkGetMemoryHostPointerPropertiesEXT vk_pfn_host_props;

static double vk_now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* instance + device with the features the PG-Strom kernels need */
static void vk_init(void)
{
	const char *iexts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
	VkApplicationInfo app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "pgstrom-vk-spike",
		.apiVersion = VK_API_VERSION_1_2,
	};
	VkInstanceCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
		.pApplicationInfo = &app,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = iexts,
	};
	uint32_t n = 1;
	VkPhysicalDeviceProperties props;
	VkPhysicalDeviceVulkan12Features f12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
	};
	VkPhysicalDeviceFeatures2 f2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
	};
	float prio = 1.0f;
	const char *dexts[4] = {"VK_KHR_portability_subset"};
	uint32_t ndexts = 1;
	VkPhysicalDeviceVulkan12Features e12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.shaderInt8 = VK_TRUE,
		.storageBuffer8BitAccess = VK_TRUE,
		.uniformAndStorageBuffer8BitAccess = VK_TRUE,
		.scalarBlockLayout = VK_TRUE,
	};
	VkPhysicalDeviceFeatures2 e2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12,
	};
	VkQueueFamilyProperties qf[8];
	uint32_t qn = 8, i;

	VK_CHECK(vkCreateInstance(&ici, NULL, &vk_inst));
	if (vkEnumeratePhysicalDevices(vk_inst, &n, &vk_pdev) < 0 || n == 0)
	{
		fprintf(stderr, "no Vulkan device\n");
		exit(2);
	}
	vkGetPhysicalDeviceProperties(vk_pdev, &props);
	vkGetPhysicalDeviceMemoryProperties(vk_pdev, &vk_memprops);
	vkGetPhysicalDeviceFeatures2(vk_pdev, &f2);
	if (!f2.features.shaderInt64 || !f12.shaderInt8 ||
		!f12.storageBuffer8BitAccess || !f12.uniformAndStorageBuffer8BitAccess)
	{
		fprintf(stderr, "required device features missing\n");
		exit(2);
	}
	printf("device: %s\n", props.deviceName);

	{
		uint32_t ne = 0, k;
		VkExtensionProperties *ep;
		int has_host = 0, has_ext_mem = 0;

		vkEnumerateDeviceExtensionProperties(vk_pdev, NULL, &ne, NULL);
		ep = malloc(sizeof(*ep) * ne);
		vkEnumerateDeviceExtensionProperties(vk_pdev, NULL, &ne, ep);
		for (k = 0; k < ne; k++)
		{
			has_host |= !strcmp(ep[k].extensionName, "VK_EXT_external_memory_host");
			has_ext_mem |= !strcmp(ep[k].extensionName, "VK_KHR_external_memory");
		}
		free(ep);
		if (has_host && has_ext_mem)
		{
			VkPhysicalDeviceExternalMemoryHostPropertiesEXT hprops = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
			};
			VkPhysicalDeviceProperties2 p2 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hprops,
			};
			vkGetPhysicalDeviceProperties2(vk_pdev, &p2);
			vk_host_align = hprops.minImportedHostPointerAlignment;
			dexts[ndexts++] = "VK_KHR_external_memory";
			dexts[ndexts++] = "VK_EXT_external_memory_host";
			vk_host_import_ok = 1;
		}
	}
	vkGetPhysicalDeviceQueueFamilyProperties(vk_pdev, &qn, qf);
	for (i = 0; i < qn; i++)
		if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			break;
	vk_qfamily = i;
	{
		VkDeviceQueueCreateInfo qci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = vk_qfamily, .queueCount = 1, .pQueuePriorities = &prio,
		};
		VkDeviceCreateInfo dci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &e2,
			.queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
			.enabledExtensionCount = ndexts, .ppEnabledExtensionNames = dexts,
		};
		VkCommandPoolCreateInfo pci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = vk_qfamily,
		};
		e2.features.shaderInt64 = VK_TRUE;
		VK_CHECK(vkCreateDevice(vk_pdev, &dci, NULL, &vk_dev));
		vkGetDeviceQueue(vk_dev, vk_qfamily, 0, &vk_queue);
		if (vk_host_import_ok)
			vk_pfn_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)
				vkGetDeviceProcAddr(vk_dev, "vkGetMemoryHostPointerPropertiesEXT");
		VK_CHECK(vkCreateCommandPool(vk_dev, &pci, NULL, &vk_cpool));
	}
}

/* device-local + host-visible + host-coherent: unified memory, no staging */
static VkBuf vk_buf(VkDeviceSize size)
{
	VkBuf b = {0};
	VkBufferCreateInfo bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkMemoryRequirements mr;
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t i;

	VK_CHECK(vkCreateBuffer(vk_dev, &bi, NULL, &b.buf));
	vkGetBufferMemoryRequirements(vk_dev, b.buf, &mr);
	for (i = 0; i < vk_memprops.memoryTypeCount; i++)
		if ((mr.memoryTypeBits & (1u << i)) &&
			(vk_memprops.memoryTypes[i].propertyFlags & want) == want)
			break;
	if (i == vk_memprops.memoryTypeCount)
	{
		fprintf(stderr, "no unified memory type\n");
		exit(2);
	}
	ai.allocationSize = mr.size;
	ai.memoryTypeIndex = i;
	VK_CHECK(vkAllocateMemory(vk_dev, &ai, NULL, &b.mem));
	VK_CHECK(vkBindBufferMemory(vk_dev, b.buf, b.mem, 0));
	VK_CHECK(vkMapMemory(vk_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
	b.size = size;
	return b;
}

/*
 * Wrap [ptr, ptr+size) as a storage buffer without copying. ptr and size must be
 * multiples of vk_host_align. Returns VK_SUCCESS, or the failing result with *why set.
 */
static VkResult vk_buf_import(void *ptr, VkDeviceSize size, VkBuf *out, const char **why)
{
	VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
	VkExternalMemoryBufferCreateInfo ebi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	};
	VkBufferCreateInfo bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ebi, .size = size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkMemoryRequirements mr;
	VkResult rc;
	uint32_t bits, i;

	memset(out, 0, sizeof(*out));
	if (!vk_host_import_ok) { *why = "extension not available"; return VK_ERROR_EXTENSION_NOT_PRESENT; }
	if (((uintptr_t)ptr % vk_host_align) || (size % vk_host_align))
	{
		*why = "pointer or size not aligned to minImportedHostPointerAlignment";
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	rc = vk_pfn_host_props(vk_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp);
	if (rc != VK_SUCCESS) { *why = "vkGetMemoryHostPointerPropertiesEXT failed"; return rc; }
	if ((rc = vkCreateBuffer(vk_dev, &bi, NULL, &out->buf)) != VK_SUCCESS) { *why = "vkCreateBuffer"; return rc; }
	vkGetBufferMemoryRequirements(vk_dev, out->buf, &mr);
	bits = hp.memoryTypeBits & mr.memoryTypeBits;
	if (!bits) { *why = "no memory type shared by the pointer and the buffer"; return VK_ERROR_FEATURE_NOT_PRESENT; }
	for (i = 0; !(bits & (1u << i)); i++)
		;
	VkImportMemoryHostPointerInfoEXT imp = {
		.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = ptr,
	};
	VkMemoryAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
		.allocationSize = size, .memoryTypeIndex = i,
	};
	if ((rc = vkAllocateMemory(vk_dev, &ai, NULL, &out->mem)) != VK_SUCCESS) { *why = "vkAllocateMemory (import)"; return rc; }
	if ((rc = vkBindBufferMemory(vk_dev, out->buf, out->mem, 0)) != VK_SUCCESS) { *why = "vkBindBufferMemory"; return rc; }
	out->map = ptr;
	out->size = size;
	printf("      imported: memory type %u, flags 0x%x\n", i, vk_memprops.memoryTypes[i].propertyFlags);
	return VK_SUCCESS;
}

/* one descriptor set of N buffers (binding i = bufs[i]; storage, or uniform if
 * bit i of ubo_mask is set) + push constants */
static void vk_bind(const VkBuf *bufs, uint32_t nbufs, uint32_t pcsize, uint32_t ubo_mask)
{
	VkDescriptorSetLayoutBinding bnd[8];
	VkDescriptorBufferInfo bi[8];
	VkWriteDescriptorSet w[8];
	VkDescriptorPool dpool;
	uint32_t i;

	for (i = 0; i < nbufs; i++)
	{
		bnd[i] = (VkDescriptorSetLayoutBinding){
			.binding = i, .descriptorType = ((ubo_mask >> i) & 1) ?
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
		bi[i] = (VkDescriptorBufferInfo){bufs[i].buf, 0, VK_WHOLE_SIZE};
	}
	VkDescriptorSetLayoutCreateInfo li = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = nbufs, .pBindings = bnd,
	};
	VK_CHECK(vkCreateDescriptorSetLayout(vk_dev, &li, NULL, &vk_dsl));
	VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize};
	VkPipelineLayoutCreateInfo pli = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1, .pSetLayouts = &vk_dsl,
		.pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
	};
	VK_CHECK(vkCreatePipelineLayout(vk_dev, &pli, NULL, &vk_playout));
	VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nbufs},
								  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nbufs}};
	VkDescriptorPoolCreateInfo dpi = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps,
	};
	VK_CHECK(vkCreateDescriptorPool(vk_dev, &dpi, NULL, &dpool));
	VkDescriptorSetAllocateInfo dai = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &vk_dsl,
	};
	VK_CHECK(vkAllocateDescriptorSets(vk_dev, &dai, &vk_dset));
	for (i = 0; i < nbufs; i++)
		w[i] = (VkWriteDescriptorSet){
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_dset,
			.dstBinding = i, .descriptorCount = 1,
			.descriptorType = ((ubo_mask >> i) & 1) ?
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &bi[i]};
	vkUpdateDescriptorSets(vk_dev, nbufs, w, 0, NULL);
}

/* returns VK_NULL_HANDLE if the Metal compiler rejects the shader */
static VkPipeline vk_pipeline(const char *spv)
{
	FILE *f = fopen(spv, "rb");
	long len;
	uint32_t *code;
	VkShaderModule mod;
	VkPipeline pl;

	if (!f) { perror(spv); exit(2); }
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	code = malloc(len);
	if (fread(code, 1, len, f) != (size_t)len) exit(2);
	fclose(f);
	VkShaderModuleCreateInfo mci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = len, .pCode = code,
	};
	VK_CHECK(vkCreateShaderModule(vk_dev, &mci, NULL, &mod));
	free(code);
	VkComputePipelineCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main",
		},
		.layout = vk_playout,
	};
	VkResult rc = vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &ci, NULL, &pl);
	vkDestroyShaderModule(vk_dev, mod, NULL);
	return rc == VK_SUCCESS ? pl : VK_NULL_HANDLE;
}

/* submit one dispatch and wait on a fence (the cuLaunchKernel+sync equivalent) */
static double vk_dispatch(VkPipeline pl, const void *pc, uint32_t pcsize, uint32_t groups)
{
	VkCommandBuffer cb;
	VkCommandBufferAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBufferBeginInfo bi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	VkFence fence;
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1, .pCommandBuffers = &cb,
	};
	double t0 = vk_now_ms();

	VK_CHECK(vkAllocateCommandBuffers(vk_dev, &ai, &cb));
	VK_CHECK(vkBeginCommandBuffer(cb, &bi));
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, vk_playout, 0, 1, &vk_dset, 0, NULL);
	vkCmdPushConstants(cb, vk_playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
	vkCmdDispatch(cb, groups, 1, 1);
	VK_CHECK(vkEndCommandBuffer(cb));
	VK_CHECK(vkCreateFence(vk_dev, &fci, NULL, &fence));
	VK_CHECK(vkQueueSubmit(vk_queue, 1, &si, fence));
	VK_CHECK(vkWaitForFences(vk_dev, 1, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyFence(vk_dev, fence, NULL);
	vkFreeCommandBuffers(vk_dev, vk_cpool, 1, &cb);
	return vk_now_ms() - t0;
}
#endif /* VKUTIL_H */
