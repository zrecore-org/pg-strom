// probe: Vulkan memory model (built with -capability vk_mem_model => OpCapability VulkanMemoryModel
// and VulkanMemoryModelDeviceScope); a plain 32-bit atomic add must be exact.
struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0,0)]] RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= pc.n)
		return;
	uint o;
	InterlockedAdd(out[0], 1u, o);
	InterlockedAdd(out[1 + (tid.x % 4)], tid.x, o);
}
