# GPU runners for CI

The lavapipe jobs of `.github/workflows/vulkan-check.yml` run on a CPU implementation of Vulkan. Real GPU coverage comes from
three places:

| Where | How | State |
|---|---|---|
| Linux, NVIDIA | job `linux-gpu`, on a GitHub-hosted GPU larger runner (Tesla T4, Ubuntu) or a self-hosted Linux runner with a GPU | off until the repository variable `GPU_RUNNER_LABEL` is set |
| Apple silicon | job `macos-gpu-test`, on a self-hosted Mac (`[self-hosted, macOS, ARM64]`), manual dispatch | needs a registered runner |
| Apple silicon, hosted | job `macos-xlarge-probe` (`macos-latest-xlarge`), manual dispatch | experiment: GitHub's docs say Metal Performance Shaders are not supported in the virtualized runner |
| Windows | none | the GPU service is not ported to Windows (Unix sockets, `shm_open`, `exec`) |

## GitHub-hosted NVIDIA runner (Linux)

GPU larger runners are generally available (2024-07-08) for Linux and Windows: NVIDIA Tesla T4. They are a *larger runner*, which
GitHub offers to organizations and enterprises on the Team or Enterprise Cloud plan, not to repositories owned by a personal
account. `zrecore/pg-strom` is a fork under a personal account, so:

1. Put the repository in an organization on the Team plan (fork `heterodb/pg-strom` into the organization, or create the
   repository there and push the branch).
2. As an organization owner: Settings > Actions > Runners > New runner > New GitHub-hosted runner. Pick the Linux (Ubuntu) image,
   the GPU size, a name (for example `gpu-t4`) and a runner group that includes the repository.
3. In the repository: Settings > Secrets and variables > Actions > Variables > New repository variable
   `GPU_RUNNER_LABEL` = the runner's name. The next run (or a manual dispatch) starts the `linux-gpu` job.
4. Larger runners are billed per minute while a job runs; check GitHub's "Actions runner pricing" page for the GPU rate. The job
   has a 120-minute limit.

Open questions only a first run answers: whether the GPU image's NVIDIA driver ships the Vulkan ICD (the job prints `nvidia-smi`,
`vulkaninfo --summary` and the ICD files first), and how long the full set takes on a T4.

## Self-hosted alternatives

Any Linux machine with an NVIDIA GPU and its Vulkan driver (or an AMD/Intel GPU with Mesa) can register as a runner with a label of
your choice and be used the same way (`GPU_RUNNER_LABEL`). For Apple silicon use a Mac mini or a bare-metal cloud Mac: Metal works
there because it is not a virtual machine.

## Without a GPU machine

- **Vulkan validation layer** (`sh test/gpuserv/vk_validation.sh <dev script>`, for example `gist_dev.sh`, `arrow_dev.sh`,
  `postgis_dev.sh`): runs a dev test with the Khronos validation layer on the service (`brew install vulkan-validationlayers` on
  macOS) and fails if the layer logs anything. It finds spec violations that one driver tolerates and another does not; the first
  run found specialized Arrow scan kernels whose push-constant block (20 bytes) was larger than their pipeline layout (16 bytes).
  The sizes of the tests are for the caller to reduce (validation is slow).
- **Rented GPU** (`sh test/cloud/gpu_check.sh` on an Ubuntu machine with a GPU, for example AWS g4dn (NVIDIA T4) or g4ad (AMD)):
  installs the dependencies, builds, runs the service test and the dev scripts with `ALLOW_CPU=off`, and prints one summary. Not
  run yet.
- The service does not branch on the capability tier (it only reports it), so forcing a tier on lavapipe would test nothing; the
  subgroup sizes the CI covers are 4 (lavapipe) and 32 (Apple).
