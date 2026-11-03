/* Can Metal's compiler service be reached from a fork()ed child (no exec)? PostgreSQL backends are. */
#include "vkutil.h"
#include <sys/wait.h>
#include <unistd.h>

static int try_compile(const char *spv)
{
	VkBuf a, b, c;
	vk_init();
	a = vk_buf(4096); b = vk_buf(4096); c = vk_buf(4096);
	VkBuf bufs[3] = {a, b, c};
	vk_bind(bufs, 3, 8, 0);
	double t0 = vk_now_ms();
	VkPipeline pl = vk_pipeline(spv);
	printf("  [pid %d] pipeline %s in %.0f ms\n", getpid(), pl ? "CREATED" : "REJECTED", vk_now_ms() - t0);
	return pl != VK_NULL_HANDLE;
}

int main(int argc, char **argv)
{
	/* mode "plain": compile in this process. mode "fork": the parent NEVER touches Vulkan;
	 * the child, forked without exec (as PostgreSQL backends are), does everything. */
	if (argc > 2 && !strcmp(argv[2], "fork"))
	{
		fflush(stdout);
		pid_t pid = fork();
		if (pid == 0)
		{
			int ok = try_compile(argv[1]);
			fflush(stdout);
			_exit(ok ? 0 : 1);
		}
		int status;
		waitpid(pid, &status, 0);
		printf("child exit status %d\n", WEXITSTATUS(status));
		return WEXITSTATUS(status) != 0;
	}
	return !try_compile(argv[1]);
}
