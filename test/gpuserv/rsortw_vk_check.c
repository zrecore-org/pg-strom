/*
 * rsortw_vk_check.c: dispatches the multi-word radix sort kernels of GPU-Sort (src/gpuserv/kernels_vm/rsortk.slang) on any Vulkan
 * device and compares the result with a CPU stable sort. Standalone and portable like vk_survey.c (the Vulkan loader is loaded at
 * run time, only the Vulkan headers are needed), so it also checks the NVIDIA driver on Windows, where the service does not run:
 *
 *   cc -O2 -o rsortw_vk_check rsortw_vk_check.c                          (Linux, macOS)
 *   x86_64-w64-mingw32-gcc -O2 -o rsortw_vk_check.exe rsortw_vk_check.c  (Windows)
 *   rsortw_vk_check [-d N] [-b] <dir with rsortk_gather.spv rsortk_count.spv rsortk_scan.spv rsortk_scatter.spv>
 *
 * Cases: key words nw = 1, 2, 8, rows n around the block size (4096) up to 300000, keys full range / few distinct values /
 * already sorted / all equal. -b adds a timing table (wall time of the whole sort, 1M rows). No device feature is enabled (the kernels
 * need none). One line per case; exit status 0: all passed. Use 64-bit literals with ull, not ul (Windows long is 32 bits).
 */
#define VK_NO_PROTOTYPES
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1000.0 * (double) c.QuadPart / (double) f.QuadPart; }
#else
#include <dlfcn.h>
#include <time.h>
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }
#endif

static PFN_vkGetInstanceProcAddr pGIPA;
static PFN_vkGetDeviceProcAddr pGDPA;
#define GI(i, n) PFN_##n n = (PFN_##n) pGIPA(i, #n)
#define GD(d, n) PFN_##n n = (PFN_##n) pGDPA(d, #n)
#define VK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #call, (int) r_, __LINE__); exit(2); } } while (0)

typedef struct { uint32_t n, B, nw, word, shift, pad; } pc_t;

static uint64_t rng = 88172645463325252ULL;
static uint64_t
rnd(void)
{
	rng ^= rng << 13;
	rng ^= rng >> 7;
	rng ^= rng << 17;
	return rng;
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

static const uint32_t *g_keys;
static uint32_t g_nw;
static int
rowcmp(const void *a, const void *b)
{
	uint32_t	x = *(const uint32_t *) a, y = *(const uint32_t *) b;

	for (uint32_t w = 0; w < g_nw; w++)
	{
		uint32_t	u = g_keys[(size_t) x * g_nw + w], v = g_keys[(size_t) y * g_nw + w];

		if (u != v)
			return u < v ? -1 : 1;
	}
	return x < y ? -1 : (x > y ? 1 : 0);
}

/* Vulkan state */
static VkDevice dev;
static VkQueue queue;
static VkCommandBuffer cb;
static VkFence fence;
static VkPipeline pipes[4];
static VkPipelineLayout layout;
static VkDescriptorSet dsets[2];
static void *mkeys, *mp[2];		/* mapped staging buffers: keys, pairs */
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
static PFN_vkEndCommandBuffer vkEndCommandBuffer;
static PFN_vkResetCommandBuffer vkResetCommandBuffer;
static PFN_vkCmdBindPipeline vkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
static PFN_vkCmdPushConstants vkCmdPushConstants;
static PFN_vkCmdDispatch vkCmdDispatch;
static PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
static PFN_vkQueueSubmit vkQueueSubmit;
static PFN_vkWaitForFences vkWaitForFences;
static PFN_vkResetFences vkResetFences;

static VkBuffer bufs_all[6];
static PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
static double g_up_ms, g_down_ms;		/* host <-> device copies of the last sort (staging buffers) */

static void
submit_wait(double *ms)
{
	VkSubmitInfo si;
	double		t0;

	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cb;
	VK(vkResetFences(dev, 1, &fence));
	t0 = now_ms();
	VK(vkQueueSubmit(queue, 1, &si, fence));
	VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
	*ms = now_ms() - t0;
}

/*
 * Sorts keys (n rows of nw words): pairs (key word, row index) in two ping-pong buffers, per key word one gather, then the digit
 * passes. The compute buffers are device local (on a discrete GPU, host visible memory would be read over the PCIe link on
 * every pass); the keys go up and the order comes back through staging buffers. Returns the number of passes; *ms is the time of
 * the compute submission, g_up_ms / g_down_ms the copies.
 */
static int
gpu_sort_k(const uint32_t *keys, uint32_t n, uint32_t nw, uint32_t *perm, double *ms)
{
	uint32_t	B = (n + 4095) / 4096, npass = 0;
	uint32_t	diff[64] = {0};
	VkCommandBufferBeginInfo bi;
	VkMemoryBarrier mb, tb;
	VkBufferCopy cr;
	uint32_t   *stag_p = (uint32_t *) mp[0];
	double		t;

	for (uint32_t r = 1; r < n; r++)
		for (uint32_t w = 0; w < nw; w++)
			diff[w] |= keys[(size_t) r * nw + w] ^ keys[w];
	memcpy(mkeys, keys, (size_t) n * nw * 4);
	for (uint32_t i = 0; i < n; i++)
	{
		stag_p[2 * i] = 0;
		stag_p[2 * i + 1] = i;
	}
	memset(&bi, 0, sizeof(bi));
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	memset(&mb, 0, sizeof(mb));
	mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	tb = mb;
	tb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	tb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	/* upload */
	vkResetCommandBuffer(cb, 0);
	VK(vkBeginCommandBuffer(cb, &bi));
	cr.srcOffset = cr.dstOffset = 0;
	cr.size = (VkDeviceSize) n * nw * 4;
	vkCmdCopyBuffer(cb, bufs_all[4], bufs_all[0], 1, &cr);
	cr.size = (VkDeviceSize) n * 8;
	vkCmdCopyBuffer(cb, bufs_all[5], bufs_all[2], 1, &cr);
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &tb, 0, NULL, 0, NULL);
	VK(vkEndCommandBuffer(cb));
	submit_wait(&g_up_ms);
	/* compute */
	vkResetCommandBuffer(cb, 0);
	VK(vkBeginCommandBuffer(cb, &bi));
	for (uint32_t w = nw; w-- > 0;)
	{
		int			gathered = 0;

		for (uint32_t d = 0; d < 8; d++)
		{
			if (((diff[w] >> (4 * d)) & 15) == 0)
				continue;
			if (!gathered)
			{
				pc_t pc = {n, B, nw, w, 0, 0};

				vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[0]);
				vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dsets[npass & 1], 0, NULL);
				vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
				vkCmdDispatch(cb, (n + 255) / 256, 1, 1);
				vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
				gathered = 1;
			}
			for (int s = 0; s < 3; s++)
			{
				pc_t pc = {n, B, nw, w, 4 * d, 0};

				vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[1 + s]);
				vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dsets[npass & 1], 0, NULL);
				vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
				vkCmdDispatch(cb, s == 1 ? 1 : B, 1, 1);
				vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
			}
			npass++;
		}
	}
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &tb, 0, NULL, 0, NULL);
	VK(vkEndCommandBuffer(cb));
	submit_wait(&t);
	*ms = t;
	/* download the pairs (the second buffer after an odd number of passes) */
	vkResetCommandBuffer(cb, 0);
	VK(vkBeginCommandBuffer(cb, &bi));
	cr.size = (VkDeviceSize) n * 8;
	vkCmdCopyBuffer(cb, bufs_all[2 + (npass & 1)], bufs_all[5], 1, &cr);
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &tb, 0, NULL, 0, NULL);
	VK(vkEndCommandBuffer(cb));
	submit_wait(&g_down_ms);
	for (uint32_t i = 0; i < n; i++)
		perm[i] = stag_p[2 * i + 1];
	return (int) npass;
}

int
main(int argc, char **argv)
{
	int			a = 1, dev_index = 0, bench = 0, fails = 0;
	const char *dir;
	VkInstance	inst;
	VkApplicationInfo ai;
	VkInstanceCreateInfo ici;
	uint32_t	n = 16, nqf = 16, qf = 0, nie = 0, nde = 0;
	VkPhysicalDevice pds[16], phys;
	VkPhysicalDeviceProperties props;
	VkPhysicalDeviceMemoryProperties mprops;
	VkQueueFamilyProperties qfp[16];
	VkDeviceQueueCreateInfo qci;
	VkDeviceCreateInfo dci;
	VkExtensionProperties *ie, *de;
	const char *iexts[1] = {"VK_KHR_portability_enumeration"}, *dexts[1] = {"VK_KHR_portability_subset"};
	float		prio = 1.0f;
	VkBuffer	buf[6];
	VkDeviceMemory mem[6];
	size_t		maxn = 1000000, bytes[6];
	static const char *const names_k[4] = {"rsortk_gather.spv", "rsortk_count.spv", "rsortk_scan.spv", "rsortk_scatter.spv"};
	const char *const *names;
	int			npipe;

	setvbuf(stdout, NULL, _IOLBF, 0);
	while (a < argc && argv[a][0] == '-')
	{
		if (!strcmp(argv[a], "-d") && a + 1 < argc)
			dev_index = atoi(argv[++a]);
		else if (!strcmp(argv[a], "-b"))
			bench = 1;
		else
			break;
		a++;
	}
	if (a >= argc)
	{
		fprintf(stderr, "usage: %s [-d N] [-b] <dir with rsortk_gather/count/scan/scatter.spv>\n", argv[0]);
		return 2;
	}
	dir = argv[a];
	names = names_k;
	npipe = 4;
	{
#ifdef _WIN32
		HMODULE h = LoadLibraryA("vulkan-1.dll");
		if (h)
			pGIPA = (PFN_vkGetInstanceProcAddr) GetProcAddress(h, "vkGetInstanceProcAddr");
#else
		static const char *ln[] = {"libvulkan.so.1", "libvulkan.1.dylib", "libvulkan.dylib", "/opt/homebrew/lib/libvulkan.1.dylib", "/usr/local/lib/libvulkan.1.dylib"};
		void *h = NULL;

		for (int i = 0; i < 5 && !h; i++)
			h = dlopen(ln[i], RTLD_NOW);
		if (h)
			pGIPA = (PFN_vkGetInstanceProcAddr) dlsym(h, "vkGetInstanceProcAddr");
#endif
		if (!pGIPA)
		{
			fprintf(stderr, "no Vulkan loader\n");
			return 2;
		}
	}
	{
		GI(NULL, vkCreateInstance);
		GI(NULL, vkEnumerateInstanceExtensionProperties);

		memset(&ai, 0, sizeof(ai));
		ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		ai.pApplicationName = "rsortw_vk_check";
		ai.apiVersion = VK_API_VERSION_1_1;
		memset(&ici, 0, sizeof(ici));
		ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		ici.pApplicationInfo = &ai;
		vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
		ie = calloc(nie ? nie : 1, sizeof(*ie));
		vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
		for (uint32_t i = 0; i < nie; i++)
			if (!strcmp(ie[i].extensionName, iexts[0]))
			{
				ici.flags |= 0x00000001;
				ici.enabledExtensionCount = 1;
				ici.ppEnabledExtensionNames = iexts;
			}
		VK(vkCreateInstance(&ici, NULL, &inst));
	}
	{
		GI(inst, vkEnumeratePhysicalDevices);
		GI(inst, vkGetPhysicalDeviceProperties);
		GI(inst, vkGetPhysicalDeviceMemoryProperties);
		GI(inst, vkGetPhysicalDeviceQueueFamilyProperties);
		GI(inst, vkEnumerateDeviceExtensionProperties);
		GI(inst, vkCreateDevice);
		GI(inst, vkGetDeviceProcAddr);

		pGDPA = vkGetDeviceProcAddr;
		vkEnumeratePhysicalDevices(inst, &n, pds);
		if (n == 0 || dev_index < 0 || (uint32_t) dev_index >= n)
		{
			fprintf(stderr, "no Vulkan device %d\n", dev_index);
			return 2;
		}
		phys = pds[dev_index];
		vkGetPhysicalDeviceProperties(phys, &props);
		printf("device %d: %s, API %u.%u.%u, driver 0x%x, vendor 0x%x\n", dev_index, props.deviceName,
			   VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
			   props.driverVersion, props.vendorID);
		vkGetPhysicalDeviceMemoryProperties(phys, &mprops);
		vkEnumerateDeviceExtensionProperties(phys, NULL, &nde, NULL);
		de = calloc(nde ? nde : 1, sizeof(*de));
		vkEnumerateDeviceExtensionProperties(phys, NULL, &nde, de);
		vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qfp);
		for (uint32_t i = 0; i < nqf; i++)
			if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				qf = i;
				break;
			}
		memset(&qci, 0, sizeof(qci));
		qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci.queueFamilyIndex = qf;
		qci.queueCount = 1;
		qci.pQueuePriorities = &prio;
		memset(&dci, 0, sizeof(dci));
		dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		for (uint32_t i = 0; i < nde; i++)
			if (!strcmp(de[i].extensionName, dexts[0]))
			{
				dci.enabledExtensionCount = 1;
				dci.ppEnabledExtensionNames = dexts;
			}
		VK(vkCreateDevice(phys, &dci, NULL, &dev));
	}
	{
		GD(dev, vkGetDeviceQueue);
		GD(dev, vkCreateBuffer);
		GD(dev, vkGetBufferMemoryRequirements);
		GD(dev, vkAllocateMemory);
		GD(dev, vkBindBufferMemory);
		GD(dev, vkMapMemory);
		GD(dev, vkCreateShaderModule);
		GD(dev, vkCreateDescriptorSetLayout);
		GD(dev, vkCreatePipelineLayout);
		GD(dev, vkCreateComputePipelines);
		GD(dev, vkCreateDescriptorPool);
		GD(dev, vkAllocateDescriptorSets);
		GD(dev, vkUpdateDescriptorSets);
		GD(dev, vkCreateCommandPool);
		GD(dev, vkAllocateCommandBuffers);
		GD(dev, vkCreateFence);
		VkDescriptorSetLayoutBinding binds[4];
		VkDescriptorSetLayoutCreateInfo dsli;
		VkDescriptorSetLayout dsl;
		VkPushConstantRange pcr;
		VkPipelineLayoutCreateInfo pli;
		VkDescriptorPoolSize psz;
		VkDescriptorPoolCreateInfo dpi;
		VkDescriptorSetAllocateInfo dai;
		VkDescriptorSetLayout dsls[2];
		VkCommandPoolCreateInfo cpi;
		VkCommandPool cpool;
		VkCommandBufferAllocateInfo cbi;
		VkFenceCreateInfo fci;

		vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer) pGDPA(dev, "vkBeginCommandBuffer");
		vkEndCommandBuffer = (PFN_vkEndCommandBuffer) pGDPA(dev, "vkEndCommandBuffer");
		vkResetCommandBuffer = (PFN_vkResetCommandBuffer) pGDPA(dev, "vkResetCommandBuffer");
		vkCmdBindPipeline = (PFN_vkCmdBindPipeline) pGDPA(dev, "vkCmdBindPipeline");
		vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets) pGDPA(dev, "vkCmdBindDescriptorSets");
		vkCmdPushConstants = (PFN_vkCmdPushConstants) pGDPA(dev, "vkCmdPushConstants");
		vkCmdDispatch = (PFN_vkCmdDispatch) pGDPA(dev, "vkCmdDispatch");
		vkCmdPipelineBarrier = (PFN_vkCmdPipelineBarrier) pGDPA(dev, "vkCmdPipelineBarrier");
		vkQueueSubmit = (PFN_vkQueueSubmit) pGDPA(dev, "vkQueueSubmit");
		vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer) pGDPA(dev, "vkCmdCopyBuffer");
		vkWaitForFences = (PFN_vkWaitForFences) pGDPA(dev, "vkWaitForFences");
		vkResetFences = (PFN_vkResetFences) pGDPA(dev, "vkResetFences");
		vkGetDeviceQueue(dev, qf, 0, &queue);

		/* buffers: keys (maxn rows x 8 words), histogram, two index arrays; host visible and coherent */
		bytes[0] = maxn * 8 * 4;
		bytes[1] = 16 * ((maxn + 4095) / 4096) * 4;
		bytes[2] = bytes[3] = maxn * 8;
		bytes[4] = bytes[0];		/* staging: keys */
		bytes[5] = maxn * 8;		/* staging: pairs */
		for (int i = 0; i < 6; i++)
		{
			VkBufferCreateInfo bci;
			VkMemoryRequirements req;
			VkMemoryAllocateInfo mai;
			uint32_t	mt = 0;
			int			found = 0;
			void	   *map;

			memset(&bci, 0, sizeof(bci));
			bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bci.size = bytes[i];
			bci.usage = (i < 4 ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0) | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			VK(vkCreateBuffer(dev, &bci, NULL, &buf[i]));
			vkGetBufferMemoryRequirements(dev, buf[i], &req);
			/* the compute buffers: device local memory (not host visible when there is such a type); the staging buffers: host visible */
			const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

			if (i < 4)
				for (uint32_t t = 0; t < mprops.memoryTypeCount && !found; t++)
					if ((req.memoryTypeBits & (1u << t)) &&
						(mprops.memoryTypes[t].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
					{
						mt = t;
						found = 1;
					}
			for (uint32_t t = 0; t < mprops.memoryTypeCount && !found; t++)
				if ((req.memoryTypeBits & (1u << t)) && (mprops.memoryTypes[t].propertyFlags & hv) == hv)
				{
					mt = t;
					found = 1;
				}
			if (!found)
			{
				fprintf(stderr, "no suitable memory type\n");
				return 2;
			}
			printf("buffer %d: memory type %u (%s)\n", i, mt, (mprops.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "host visible" : "device local only");
			memset(&mai, 0, sizeof(mai));
			mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			mai.allocationSize = req.size;
			mai.memoryTypeIndex = mt;
			VK(vkAllocateMemory(dev, &mai, NULL, &mem[i]));
			VK(vkBindBufferMemory(dev, buf[i], mem[i], 0));
			bufs_all[i] = buf[i];
			if (i >= 4)
			{
				VK(vkMapMemory(dev, mem[i], 0, VK_WHOLE_SIZE, 0, &map));
				if (i == 4) mkeys = map; else mp[0] = map;
			}
		}
		for (int i = 0; i < 4; i++)
		{
			binds[i].binding = i;
			binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			binds[i].descriptorCount = 1;
			binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			binds[i].pImmutableSamplers = NULL;
		}
		memset(&dsli, 0, sizeof(dsli));
		dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dsli.bindingCount = 4;
		dsli.pBindings = binds;
		VK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));
		pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pcr.offset = 0;
		pcr.size = sizeof(pc_t);
		memset(&pli, 0, sizeof(pli));
		pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pli.setLayoutCount = 1;
		pli.pSetLayouts = &dsl;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges = &pcr;
		VK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));
		for (int s = 0; s < npipe; s++)
		{
			char		path[1024];
			size_t		len;
			unsigned char *spv;
			uint32_t   *code;
			VkShaderModuleCreateInfo smi;
			VkShaderModule mod;
			VkComputePipelineCreateInfo cp;
			double		t0;

			snprintf(path, sizeof(path), "%s/%s", dir, names[s]);
			spv = slurp(path, &len);
			if (!spv || len == 0 || len % 4)
			{
				fprintf(stderr, "cannot read %s\n", path);
				return 2;
			}
			code = malloc(len);
			memcpy(code, spv, len);
			free(spv);
			memset(&smi, 0, sizeof(smi));
			smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			smi.codeSize = len;
			smi.pCode = code;
			VK(vkCreateShaderModule(dev, &smi, NULL, &mod));
			memset(&cp, 0, sizeof(cp));
			cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			cp.stage.module = mod;
			cp.stage.pName = "main";
			cp.layout = layout;
			t0 = now_ms();
			VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, NULL, &pipes[s]));
			printf("pipeline %s: %zu bytes, ok in %.1f ms\n", names[s], len, now_ms() - t0);
		}
		psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		psz.descriptorCount = 8;
		memset(&dpi, 0, sizeof(dpi));
		dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		dpi.maxSets = 2;
		dpi.poolSizeCount = 1;
		dpi.pPoolSizes = &psz;
		{
			VkDescriptorPool dpool;

			VK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
			dsls[0] = dsls[1] = dsl;
			memset(&dai, 0, sizeof(dai));
			dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			dai.descriptorPool = dpool;
			dai.descriptorSetCount = 2;
			dai.pSetLayouts = dsls;
			VK(vkAllocateDescriptorSets(dev, &dai, dsets));
		}
		for (int par = 0; par < 2; par++)
		{
			VkDescriptorBufferInfo bi[4];
			VkWriteDescriptorSet wr[4];
			VkBuffer	use[4] = {buf[0], buf[1], buf[2 + par], buf[3 - par]};

			for (int i = 0; i < 4; i++)
			{
				bi[i].buffer = use[i];
				bi[i].offset = 0;
				bi[i].range = VK_WHOLE_SIZE;
				memset(&wr[i], 0, sizeof(wr[i]));
				wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				wr[i].dstSet = dsets[par];
				wr[i].dstBinding = i;
				wr[i].descriptorCount = 1;
				wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				wr[i].pBufferInfo = &bi[i];
			}
			vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);
		}
		memset(&cpi, 0, sizeof(cpi));
		cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpi.queueFamilyIndex = qf;
		VK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
		memset(&cbi, 0, sizeof(cbi));
		cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbi.commandPool = cpool;
		cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbi.commandBufferCount = 1;
		VK(vkAllocateCommandBuffers(dev, &cbi, &cb));
		memset(&fci, 0, sizeof(fci));
		fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		VK(vkCreateFence(dev, &fci, NULL, &fence));
	}

	{
		static const uint32_t nws[] = {1, 2, 8};
		static const uint32_t sizes[] = {1, 2, 255, 4095, 4096, 4097, 65536, 300000};
		static const char *const shapes[] = {"full range", "few distinct", "sorted", "all equal", "narrow"};

		for (size_t wi = 0; wi < 3; wi++)
			for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++)
				for (int sh = 0; sh < 5; sh++)
				{
					uint32_t	nw = nws[wi], rows = sizes[si];
					uint32_t   *keys = malloc((size_t) rows * nw * 4), *perm = malloc(rows * 4), *ref = malloc(rows * 4);
					double		ms = 0;
					int			npass, bad = 0;

					for (uint32_t r = 0; r < rows; r++)
						for (uint32_t w = 0; w < nw; w++)
						{
							uint32_t	v;

							switch (sh)
							{
								case 0: v = (uint32_t) rnd(); break;
								case 1: v = (uint32_t) (rnd() % 5) * 0x01010101u; break;
								case 2: v = w == nw - 1 ? r * 3 : 7; break;
								case 3: v = 42; break;
								default: v = (uint32_t) (rnd() % 1000); break;
							}
							keys[(size_t) r * nw + w] = v;
						}
					g_keys = keys;
					g_nw = nw;
					for (uint32_t r = 0; r < rows; r++)
						ref[r] = r;
					qsort(ref, rows, 4, rowcmp);
					npass = gpu_sort_k(keys, rows, nw, perm, &ms);
					for (uint32_t r = 0; r < rows && bad < 3; r++)
						if (perm[r] != ref[r])
						{
							printf("  FAIL nw=%u n=%u %s: position %u got row %u want %u\n", nw, rows, shapes[sh], r, perm[r], ref[r]);
							bad++;
						}
					fails += bad ? 1 : 0;
					printf("  %s nw=%u n=%-7u %-12s %2d passes %8.2f ms\n", bad ? "FAIL" : "ok  ", nw, rows, shapes[sh], npass, ms);
					free(keys);
					free(perm);
					free(ref);
				}
		if (bench)
		{
			printf("%9s %4s %10s %10s %10s %10s\n", "rows", "nw", "compute ms", "up ms", "down ms", "qsort ms");
			for (size_t wi = 0; wi < 3; wi++)
			{
				uint32_t	nw = nws[wi], rows = 1000000;
				uint32_t   *keys = malloc((size_t) rows * nw * 4), *perm = malloc(rows * 4), *ref = malloc(rows * 4);
				double		ms = 0, t0, t1;

				for (size_t i = 0; i < (size_t) rows * nw; i++)
					keys[i] = (uint32_t) (rnd() % 100000);
				g_keys = keys;
				g_nw = nw;
				for (uint32_t r = 0; r < rows; r++)
					ref[r] = r;
				t0 = now_ms();
				qsort(ref, rows, 4, rowcmp);
				t1 = now_ms();
				gpu_sort_k(keys, rows, nw, perm, &ms);
				printf("%9u %4u %10.2f %10.2f %10.2f %10.2f%s\n", rows, nw, ms, g_up_ms, g_down_ms, t1 - t0, memcmp(perm, ref, rows * 4) ? "  MISMATCH" : "");
				free(keys);
				free(perm);
				free(ref);
			}
		}
	}
	printf("rsortw_vk_check: %s\n", fails ? "FAILED" : "all passed");
	return fails ? 1 : 0;
}
