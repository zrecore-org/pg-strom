/* How much Metal/Vulkan initialisation can a parent do before fork() while staying single-threaded
 * (PostgreSQL 17 refuses to run a multithreaded postmaster), and does it let the child compile? */
#include "vkutil.h"
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	const char *level = argv[1];		/* none | instance | device | buffers | compile */
	const char *spv = argv[2];
	printf("parent init level '%s': threaded before = %d\n", level, pthread_is_threaded_np());
	if (strcmp(level, "none") != 0)
	{
		if (!strcmp(level, "instance"))
		{
			const char *iexts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
			VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
			VkInstanceCreateInfo ici = {
				.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
				.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR, .pApplicationInfo = &app,
				.enabledExtensionCount = 1, .ppEnabledExtensionNames = iexts};
			VkInstance inst;
			vkCreateInstance(&ici, NULL, &inst);
		}
		else
		{
			vk_init();					/* instance + physical device + logical device + queue */
			if (strcmp(level, "device") != 0)
			{
				VkBuf a = vk_buf(4096), b = vk_buf(4096), c = vk_buf(4096);
				VkBuf bufs[3] = {a, b, c};
				vk_bind(bufs, 3, 8, 0);
				if (!strcmp(level, "compile"))
					vk_pipeline("/tmp/hl_80.spv");	/* some shader, compiled in the parent */
			}
		}
	}
	printf("  after init: threaded = %d\n", pthread_is_threaded_np());
	fflush(stdout);
	pid_t pid = fork();
	if (pid == 0)
	{
		vk_init();
		VkBuf a = vk_buf(4096), b = vk_buf(4096), c = vk_buf(4096);
		VkBuf bufs[3] = {a, b, c};
		vk_bind(bufs, 3, 8, 0);
		VkPipeline pl = vk_pipeline(spv);
		printf("  child compile of a fresh shader: %s\n", pl ? "OK" : "REJECTED");
		fflush(stdout);
		_exit(pl ? 0 : 1);
	}
	int st;
	waitpid(pid, &st, 0);
	return WEXITSTATUS(st);
}
