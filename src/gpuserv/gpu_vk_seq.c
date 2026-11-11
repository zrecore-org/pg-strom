/*
 * gpu_vk_seq.c
 *
 * A sequence of compute dispatches in one command buffer (gpu_vk.h: gpu_vk_dispatch_seq). Used by
 * GPU-Sort, whose bitonic network is hundreds of dependent passes: submitting them one by one would
 * cost a fence wait each. Kept apart from gpu_vk.c; it only uses the context's public fields.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_vk.h"

static void
seq_err(gpu_vk_ctx *ctx, const char *fmt, ...)
{
	va_list		ap;

	va_start(ap, fmt);
	vsnprintf(ctx->err, sizeof(ctx->err), fmt, ap);
	va_end(ap);
}

static double
seq_now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* write the descriptor set of pipeline p with the buffers of the step that first uses it */
static int
seq_write_descriptors(gpu_vk_ctx *ctx, gpu_vk_pipeline *p, gpu_vk_buffer *const *bufs)
{
	VkDescriptorBufferInfo *dbi = calloc(p->nbind, sizeof(*dbi));
	VkWriteDescriptorSet *wds = calloc(p->nbind, sizeof(*wds));

	if (!dbi || !wds)
	{
		free(dbi);
		free(wds);
		seq_err(ctx, "out of memory");
		return -1;
	}
	for (uint32_t i = 0; i < p->nbind; i++)
	{
		if (!bufs || !bufs[i] || !bufs[i]->buf)
		{
			free(dbi);
			free(wds);
			seq_err(ctx, "dispatch: buffer %u is missing", i);
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
	return 0;
}

int
gpu_vk_dispatch_seq(gpu_vk_ctx *ctx, const gpu_vk_step *steps, uint32_t nsteps, double *elapsed_ms)
{
	VkCommandBufferBeginInfo cbi;
	VkMemoryBarrier mb;
	VkSubmitInfo si;
	VkResult	rc;
	double		t0;
	uint32_t	wait_ms = 0;
	gpu_vk_pipeline *cur = NULL;

	if (elapsed_ms)
		*elapsed_ms = 0.0;
	if (ctx->lost)
	{
		seq_err(ctx, "device is lost");
		return -1;
	}
	if (nsteps == 0)
		return 0;
	for (uint32_t s = 0; s < nsteps; s++)
	{
		gpu_vk_pipeline *p = steps[s].p;
		int			seen = 0;

		if (gpu_vk_wait_ms(ctx, p) > wait_ms)
			wait_ms = gpu_vk_wait_ms(ctx, p);	/* a step that was never dispatched pays for its compile */

		for (uint32_t u = 0; u < s; u++)
			if (steps[u].p == p)
				seen = 1;
		if (!seen && p->nbind > 0 && seq_write_descriptors(ctx, p, steps[s].bufs) != 0)
			return -1;
	}
	if (vkResetCommandBuffer(ctx->cmd, 0) != VK_SUCCESS)
	{
		seq_err(ctx, "vkResetCommandBuffer failed");
		return -1;
	}
	memset(&cbi, 0, sizeof(cbi));
	cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (vkBeginCommandBuffer(ctx->cmd, &cbi) != VK_SUCCESS)
	{
		seq_err(ctx, "vkBeginCommandBuffer failed");
		return -1;
	}
	for (uint32_t s = 0; s < nsteps; s++)
	{
		gpu_vk_pipeline *p = steps[s].p;

		if (p != cur)
		{
			vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
			if (p->nbind > 0)
				vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout,
										0, 1, &p->dset, 0, NULL);
			cur = p;
		}
		if (p->pc_size > 0 && steps[s].pc)
			vkCmdPushConstants(ctx->cmd, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
							   p->pc_size, steps[s].pc);
		vkCmdDispatch(ctx->cmd, steps[s].gx, steps[s].gy ? steps[s].gy : 1, steps[s].gz ? steps[s].gz : 1);
		memset(&mb, 0, sizeof(mb));
		mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		if (s + 1 < nsteps)
		{
			mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
		}
		else
		{
			mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			vkCmdPipelineBarrier(ctx->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
								 VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
		}
	}
	if (vkEndCommandBuffer(ctx->cmd) != VK_SUCCESS)
	{
		seq_err(ctx, "vkEndCommandBuffer failed");
		return -1;
	}
	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &ctx->cmd;
	t0 = seq_now_ms();
	pthread_mutex_lock(&ctx->queue_lock);
	rc = vkQueueSubmit(ctx->queue, 1, &si, ctx->fence);
	pthread_mutex_unlock(&ctx->queue_lock);
	if (rc != VK_SUCCESS)
	{
		if (rc == VK_ERROR_DEVICE_LOST)
			ctx->lost = 1;
		seq_err(ctx, "vkQueueSubmit failed (%d)", (int) rc);
		return -1;
	}
	rc = vkWaitForFences(ctx->dev, 1, &ctx->fence, VK_TRUE, (uint64_t) wait_ms * 1000000ULL);
	if (elapsed_ms)
		*elapsed_ms = seq_now_ms() - t0;
	if (rc == VK_TIMEOUT)
	{
		ctx->lost = 1;
		seq_err(ctx, "kernel did not finish within %u ms (device marked lost)", wait_ms);
		return -1;
	}
	if (rc != VK_SUCCESS)
	{
		if (rc == VK_ERROR_DEVICE_LOST)
			ctx->lost = 1;
		seq_err(ctx, "vkWaitForFences failed (%d)", (int) rc);
		return -1;
	}
	if (vkResetFences(ctx->dev, 1, &ctx->fence) != VK_SUCCESS)
	{
		seq_err(ctx, "vkResetFences failed");
		return -1;
	}
	for (uint32_t s = 0; s < nsteps; s++)
		steps[s].p->dispatched = 1;
	return 0;
}
