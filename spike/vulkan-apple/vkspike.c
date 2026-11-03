/*
 * vkspike.c - Apple/MoltenVK capability spike for the PG-Strom Vulkan backend
 *
 * Throwaway code. Verifies on real hardware what .claude/hw-acceleration.md
 * assumes about the device: heap-row deform with unaligned byte access,
 * int64 arithmetic, subgroup ballot compaction, 32-bit atomics, 64-bit sum
 * built from 32-bit atomics, buffer device address, and zero-copy memory.
 *
 * Build/run: see Makefile in this directory.
 */
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NROWS		(16u * 1024 * 1024)
#define ROW_STRIDE	13u		/* int32 @0, int64 @4, uint8 @12: all unaligned */
#define THRESH		(1LL << 30)

#define CHECK(expr)															\
	do { VkResult __rc = (expr);											\
		 if (__rc != VK_SUCCESS) {											\
			 fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #expr, __rc); \
			 exit(2); } } while (0)

static VkInstance		inst;
static VkPhysicalDevice	pdev;
static VkDevice			dev;
static VkQueue			queue;
static uint32_t			qfamily;
static VkCommandPool	cpool;
static VkPhysicalDeviceMemoryProperties memprops;

typedef struct
{
	VkBuffer		buf;
	VkDeviceMemory	mem;
	void		   *map;
	VkDeviceSize	size;
	VkDeviceAddress	addr;
} Buf;

static double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static Buf make_buf(VkDeviceSize size, int want_bda)
{
	Buf b = {0};
	VkBufferCreateInfo bi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
				 (want_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	VkMemoryRequirements mr;
	VkMemoryAllocateFlagsInfo fi = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
		.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
	};
	VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	/* device-local AND host-visible AND coherent = unified memory, zero copy */
	const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t i;

	CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
	vkGetBufferMemoryRequirements(dev, b.buf, &mr);
	for (i = 0; i < memprops.memoryTypeCount; i++)
		if ((mr.memoryTypeBits & (1u << i)) &&
			(memprops.memoryTypes[i].propertyFlags & want) == want)
			break;
	if (i == memprops.memoryTypeCount)
	{
		fprintf(stderr, "no DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT memory type\n");
		exit(2);
	}
	ai.allocationSize = mr.size;
	ai.memoryTypeIndex = i;
	if (want_bda)
		ai.pNext = &fi;
	CHECK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
	CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
	CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
	b.size = size;
	if (want_bda)
	{
		VkBufferDeviceAddressInfo di = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
			.buffer = b.buf,
		};
		b.addr = vkGetBufferDeviceAddress(dev, &di);
	}
	return b;
}

static VkShaderModule load_shader(const char *path)
{
	FILE *f = fopen(path, "rb");
	long len;
	uint32_t *code;
	VkShaderModule mod;
	VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};

	if (!f) { perror(path); exit(2); }
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	code = malloc(len);
	if (fread(code, 1, len, f) != (size_t)len) exit(2);
	fclose(f);
	ci.codeSize = len;
	ci.pCode = code;
	CHECK(vkCreateShaderModule(dev, &ci, NULL, &mod));
	free(code);
	return mod;
}

static VkDescriptorSetLayout	dsl;
static VkPipelineLayout			playout;
static VkDescriptorPool			dpool;
static VkDescriptorSet			dset;

static VkPipeline make_pipeline(const char *spv)
{
	VkShaderModule mod = load_shader(spv);
	VkPipeline pl;
	VkComputePipelineCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = mod,
			.pName = "main",
		},
		.layout = playout,
	};
	VkResult rc = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &pl);
	vkDestroyShaderModule(dev, mod, NULL);
	return rc == VK_SUCCESS ? pl : VK_NULL_HANDLE;
}

/* submit one dispatch, wait on a fence (the CUDA launch+sync equivalent) */
static double dispatch(VkPipeline pl, const void *pc, uint32_t pcsize, uint32_t groups)
{
	VkCommandBuffer cb;
	VkCommandBufferAllocateInfo ai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = cpool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
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
		.commandBufferCount = 1,
		.pCommandBuffers = &cb,
	};
	double t0 = now_ms();

	CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
	CHECK(vkBeginCommandBuffer(cb, &bi));
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, NULL);
	vkCmdPushConstants(cb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
	vkCmdDispatch(cb, groups, 1, 1);
	CHECK(vkEndCommandBuffer(cb));
	CHECK(vkCreateFence(dev, &fci, NULL, &fence));
	CHECK(vkQueueSubmit(queue, 1, &si, fence));
	CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
	vkDestroyFence(dev, fence, NULL);
	vkFreeCommandBuffers(dev, cpool, 1, &cb);
	return now_ms() - t0;
}

static int cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return x < y ? -1 : x > y;
}

static uint64_t xs = 88172645463325252ULL;
static uint64_t rnd(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs; }

int main(int argc, char **argv)
{
	const char *shaderdir = argc > 1 ? argv[1] : "shaders";
	const char *scan_spv = argc > 2 ? argv[2] : "scan.spv";
	char path[512];
	int failures = 0;

	/* ---- instance (portability enumeration is mandatory for MoltenVK) ---- */
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

	CHECK(vkCreateInstance(&ici, NULL, &inst));
	/* take the first device; VK_INCOMPLETE just means there were more */
	if (vkEnumeratePhysicalDevices(inst, &n, &pdev) < 0 || n == 0)
	{
		fprintf(stderr, "no Vulkan device\n");
		return 2;
	}
	vkGetPhysicalDeviceProperties(pdev, &props);
	vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
	printf("device: %s (api %u.%u.%u)\n", props.deviceName,
		   VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
		   VK_VERSION_PATCH(props.apiVersion));

	/* ---- features actually enabled ---- */
	VkPhysicalDeviceVulkan12Features f12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
	};
	VkPhysicalDeviceFeatures2 f2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12,
	};
	vkGetPhysicalDeviceFeatures2(pdev, &f2);
	printf("features: int64=%d float64=%d int8=%d 8bitStorage=%d bda=%d "
		   "scalarLayout=%d subgroupExtTypes=%d bufInt64Atomics=%d\n",
		   f2.features.shaderInt64, f2.features.shaderFloat64, f12.shaderInt8,
		   f12.storageBuffer8BitAccess, f12.bufferDeviceAddress,
		   f12.scalarBlockLayout, f12.shaderSubgroupExtendedTypes,
		   f12.shaderBufferInt64Atomics);
	if (!f2.features.shaderInt64 || !f12.shaderInt8 || !f12.storageBuffer8BitAccess ||
		!f12.scalarBlockLayout || !f12.bufferDeviceAddress)
	{
		fprintf(stderr, "required features missing\n");
		return 2;
	}

	/* ---- queue family with compute ---- */
	{
		VkQueueFamilyProperties qf[8];
		uint32_t qn = 8, i;
		vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qn, qf);
		for (i = 0; i < qn; i++)
			if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				break;
		qfamily = i;
	}

	/* ---- device ---- */
	{
		float prio = 1.0f;
		const char *dexts[] = {"VK_KHR_portability_subset"};
		VkPhysicalDeviceVulkan12Features e12 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.shaderInt8 = VK_TRUE,
			.storageBuffer8BitAccess = VK_TRUE,
			.uniformAndStorageBuffer8BitAccess = f12.uniformAndStorageBuffer8BitAccess,
			.scalarBlockLayout = VK_TRUE,
			.bufferDeviceAddress = VK_TRUE,
			.shaderSubgroupExtendedTypes = f12.shaderSubgroupExtendedTypes,
		};
		VkPhysicalDeviceFeatures2 e2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12,
		};
		VkDeviceQueueCreateInfo qci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = qfamily, .queueCount = 1, .pQueuePriorities = &prio,
		};
		VkDeviceCreateInfo dci = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &e2,
			.queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
			.enabledExtensionCount = 1, .ppEnabledExtensionNames = dexts,
		};
		e2.features.shaderInt64 = VK_TRUE;
		CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
		vkGetDeviceQueue(dev, qfamily, 0, &queue);
	}
	{
		VkCommandPoolCreateInfo pci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = qfamily,
		};
		CHECK(vkCreateCommandPool(dev, &pci, NULL, &cpool));
	}

	/* ---- buffers: unified memory, persistently mapped, no staging ---- */
	size_t rows_bytes = (size_t)NROWS * ROW_STRIDE + 16;
	Buf rows = make_buf(rows_bytes, 0);
	Buf outb = make_buf(sizeof(uint32_t) * (NROWS + 4), 0);

	/* descriptor set: binding 0 = rows, binding 1 = out/acc */
	{
		VkDescriptorSetLayoutBinding bnd[2] = {
			{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
			{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
		};
		VkDescriptorSetLayoutCreateInfo li = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 2, .pBindings = bnd,
		};
		VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
		VkPipelineLayoutCreateInfo pli = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1, .pSetLayouts = &dsl,
			.pushConstantRangeCount = 1, .pPushConstantRanges = &pcr,
		};
		VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
		VkDescriptorPoolCreateInfo dpi = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps,
		};
		VkDescriptorSetAllocateInfo dai = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorSetCount = 1, .pSetLayouts = &dsl,
		};
		VkDescriptorBufferInfo bi0 = {rows.buf, 0, VK_WHOLE_SIZE};
		VkDescriptorBufferInfo bi1 = {outb.buf, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet w[2] = {
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0,
			 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 .pBufferInfo = &bi0},
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1,
			 .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			 .pBufferInfo = &bi1},
		};
		CHECK(vkCreateDescriptorSetLayout(dev, &li, NULL, &dsl));
		pli.pSetLayouts = &dsl;
		CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &playout));
		CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
		dai.descriptorPool = dpool;
		dai.pSetLayouts = &dsl;
		CHECK(vkAllocateDescriptorSets(dev, &dai, &dset));
		w[0].dstSet = dset;
		w[1].dstSet = dset;
		vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
	}

	/* ---- generate rows + CPU reference ---- */
	uint8_t *p = rows.map;
	uint64_t ref_count = 0;
	int64_t ref_sum = 0;
	double t0 = now_ms();
	for (uint32_t i = 0; i < NROWS; i++)
	{
		int32_t a = (int32_t)(rnd() & 0x7fffffff);
		int64_t b = (int64_t)(rnd() % (1ULL << 41)) - (1LL << 40);
		uint8_t flag = (rnd() % 10) != 0;
		uint8_t *r = p + (size_t)i * ROW_STRIDE;
		memcpy(r, &a, 4);
		memcpy(r + 4, &b, 8);
		r[12] = flag;
		ref_sum += b;							/* sum64 ignores flag */
		if (a % 7 == 0 && (b + (int64_t)a * 3 > THRESH) && flag)
			ref_count++;
	}
	printf("generated %u rows (%.0f MB) in %.0f ms; reference matches=%llu\n",
		   NROWS, rows_bytes / 1e6, now_ms() - t0, (unsigned long long)ref_count);

	/* CPU baseline timing (single thread, same predicate) */
	{
		volatile uint64_t cnt = 0;
		double c0 = now_ms();
		for (uint32_t i = 0; i < NROWS; i++)
		{
			int32_t a; int64_t b; const uint8_t *r = p + (size_t)i * ROW_STRIDE;
			memcpy(&a, r, 4); memcpy(&b, r + 4, 8);
			if (a % 7 == 0 && (b + (int64_t)a * 3 > THRESH) && r[12])
				cnt++;
		}
		printf("cpu single-thread scan: %.1f ms\n", now_ms() - c0);
	}

	uint32_t groups = (NROWS + 255) / 256;
	struct { uint32_t nrows, stride; int64_t thresh; } pc = {NROWS, ROW_STRIDE, THRESH};

	/* ---- test 1: scan-style kernel ---- */
	{
		VkPipeline pl = make_pipeline((snprintf(path, sizeof path, "%s/%s", shaderdir, scan_spv), path));
		uint32_t *ob = outb.map;
		double best = 1e9;
		int ok = 1;

		for (int it = 0; it < 5; it++)
		{
			ob[0] = 0;
			double ms = dispatch(pl, &pc, sizeof pc, groups);
			if (ms < best) best = ms;
			if (ob[0] != ref_count) ok = 0;
		}
		if (ok)
		{
			/* check index set is exactly the CPU's */
			uint32_t *idx = ob + 1;
			qsort(idx, ob[0], 4, cmp_u32);
			uint64_t k = 0;
			for (uint32_t i = 0; i < NROWS && ok; i++)
			{
				int32_t a; int64_t b; const uint8_t *r = p + (size_t)i * ROW_STRIDE;
				memcpy(&a, r, 4); memcpy(&b, r + 4, 8);
				if (a % 7 == 0 && (b + (int64_t)a * 3 > THRESH) && r[12])
				{
					if (k >= ob[0] || idx[k] != i) ok = 0;
					k++;
				}
			}
		}
		printf("[%s] scan[%s] (unaligned deform + int64 qual + ballot compaction): "
			   "gpu count=%u best=%.1f ms (%.0f Mrows/s)\n",
			   ok ? "PASS" : "FAIL", scan_spv, ob[0], best, NROWS / best / 1e3);
		failures += !ok;
	}

	/* ---- test 2: 64-bit sum from 32-bit atomics ---- */
	{
		VkPipeline pl = make_pipeline((snprintf(path, sizeof path, "%s/sum64.spv", shaderdir), path));
		uint32_t *acc = outb.map;
		double best = 1e9;
		int ok = 1;

		for (int it = 0; it < 3; it++)
		{
			acc[0] = acc[1] = acc[2] = 0;
			double ms = dispatch(pl, &pc, sizeof pc, groups);
			if (ms < best) best = ms;
			int64_t got = (int64_t)(((uint64_t)acc[1] << 32) | acc[0]);
			if (got != ref_sum) ok = 0;
			if (it == 0)
				printf("    sum64: gpu=%lld cpu=%lld\n", (long long)got, (long long)ref_sum);
		}
		printf("[%s] sum64 via 16-bit-limb subgroupAdd + lo/hi 32-bit atomics w/ carry: best=%.1f ms\n",
			   ok ? "PASS" : "FAIL", best);
		failures += !ok;
	}

	/* ---- probe: int64 subgroupAdd (feature flag says yes; does Metal agree?) ---- */
	{
		VkPipeline pl = make_pipeline((snprintf(path, sizeof path, "%s/probe_i64_subgroup.spv", shaderdir), path));
		printf("[INFO] int64 subgroupAdd (shaderSubgroupExtendedTypes=%d): %s\n",
			   f12.shaderSubgroupExtendedTypes,
			   pl ? "compiles" : "REJECTED by Metal shader compiler");
	}

	/* ---- test 3: buffer device address ---- */
	{
		VkPipeline pl = make_pipeline((snprintf(path, sizeof path, "%s/bda.spv", shaderdir), path));
		Buf tgt = make_buf(sizeof(uint32_t) * 100000, 1);
		struct { uint64_t addr; uint32_t n; } bpc = {tgt.addr, 100000};
		uint32_t *t = tgt.map;
		int ok = 1;

		memset(t, 0, tgt.size);
		dispatch(pl, &bpc, sizeof bpc, (100000 + 255) / 256);
		for (uint32_t i = 0; i < 100000; i++)
			if (t[i] != i * 3u + 1u) { ok = 0; break; }
		printf("[%s] buffer_device_address write through raw pointer (addr=0x%llx)\n",
			   ok ? "PASS" : "FAIL", (unsigned long long)tgt.addr);
		failures += !ok;
	}

	printf("%s\n", failures ? "SPIKE: FAILURES" : "SPIKE: ALL PASSED");
	return failures ? 1 : 0;
}
