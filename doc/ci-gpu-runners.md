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
account. `zrecore-org/pg-strom` is now under an organization (moved from a personal account on 2026-11-10), so:

1. Make sure the organization is on the Team plan (GPU larger runners need it).
2. As an organization owner: Settings > Actions > Runners > New runner > New GitHub-hosted runner. Pick the Linux (Ubuntu) image,
   the GPU size, a name (for example `gpu-t4`) and a runner group that includes the repository.
3. In the repository: Settings > Secrets and variables > Actions > Variables > New repository variable
   `GPU_RUNNER_LABEL` = the runner's name. The next run (or a manual dispatch) starts the `linux-gpu` job.
4. Larger runners are billed per minute while a job runs; check GitHub's "Actions runner pricing" page for the GPU rate. The job
   has a 120-minute limit.

What the first runs answered: the GPU image's NVIDIA driver ships the Vulkan ICD (the job prints `nvidia-smi`,
`vulkaninfo --summary` and the ICD files first); a cold run takes about 110 minutes, and a run with the saved NVIDIA pipeline
cache about 21-29 minutes. The runner pool has one machine, and it is shared with another repository: jobs queue first in, first
out, so push only when no T4 job is queued or running there. The job also logs the host memory bandwidth of each memory type
(`pgstrom-gpuserv --probe-memory`).

A manual run with the `bench_gpu` input ticked adds an informational step that runs `test/gpuserv/perf_bench.sh` on the T4
(`CPU_WORKERS` is the runner's core count, so the PostgreSQL column is cpuN, not cpu8) and writes the timings to the run summary.
It adds the benchmark minutes to the billed job, so tick it only when a measurement is wanted.

## Self-hosted alternatives

Any Linux machine with an NVIDIA GPU and its Vulkan driver (or an AMD/Intel GPU with Mesa) can register as a runner with a label of
your choice and be used the same way (`GPU_RUNNER_LABEL`). For Apple silicon use a Mac mini or a bare-metal cloud Mac: Metal works
there because it is not a virtual machine.

## AWS spot runners (RunsOn), shared by several repositories

The hosted T4 runner costs about $0.052 per minute (a CI run of 17-20 minutes is about $0.90-1.00). A g4dn.xlarge (one T4, 4 vCPU)
on AWS spot is roughly 10 times cheaper (about $0.21-0.27 per hour in the US East regions, about 20% more in us-west-1; third-party
price tables, check the EC2 spot price history for the live figure). The plan is [RunsOn](https://runs-on.com/guides/install): a
CloudFormation stack in your own AWS account plus a GitHub App on the organization; a workflow picks a runner with a `runs-on` label.
GitHub also charges $0.002 per minute for self-hosted runners (from March 2026).

**Account layout (decided 2026-11-14).** One AWS Organization: the management account holds billing only (MFA on the root user, no
workloads); a member account `zrecore-ci` holds the CI runners, and later a separate account for deployed GPU applications. People
sign in through IAM Identity Center (organization instance, single Region, us-east-1; permission sets assigned per account), not
with root or IAM access keys. "Centralize root access" is enabled, so member accounts have no root credentials of their own. The
repositories that share it (this one and `../mongo`) run one GPU job each at a time, so the quota is 8 vCPUs of spot and 8 of on-demand.

**Setup steps, in order**
1. Create the AWS account on the Paid plan (the Free plan does not allow GPU instances as far as third-party sources say); Basic
   support is enough. Turn on MFA for the root user.
2. Create the organization, enable "Centralize root access" (leave the delegated administrator empty), create the member account.
   Billing and budgets for a member account are managed from the management account (a member account cannot open Budgets until
   IAM access to billing is switched on, which only its root user can do).
3. Enable IAM Identity Center (organization instance), create a user, a permission set (`AdministratorAccess` for the person) and
   assign it to the member account. Sign in at the AWS access portal.
4. In the member account, request the quotas **All G and VT Spot Instance Requests** = 8 and **Running On-Demand G and VT
   instances** = 8 (Service Quotas, Amazon EC2, per Region). On a new account both requests become support cases; wait for approval.
5. Install the RunsOn stack in the member account and register its GitHub App on the organization (selected repositories only).
6. The workflow is already prepared: a manual run with the `gpu_runner_aws` input ticked runs the `linux-gpu` job on a RunsOn label
   (`runs-on=<run id>/family=g4dn.xlarge/image=ubuntu24-gpu-x64/spot=true`, label keys from RunsOn's [labels](https://runs-on.com/docs/runners/labels/)
   and [GPU runners](https://runs-on.com/runners/gpu/) pages); every other run keeps the hosted T4 (`GPU_RUNNER_LABEL`). The job's
   "NVIDIA Vulkan driver" step checks the driver and the Vulkan ICD (and tries `libnvidia-gl-<driver major>` if the ICD is missing).
   Not confirmed until the first AWS run: that the RunsOn GPU image ships the NVIDIA Vulkan ICD, and that `family=g4dn.xlarge` is
   accepted exactly like that. The hosted T4 stays the default and the fallback.
7. Run the `bench_gpu` dispatch and compare with the hosted T4: driver and `vulkaninfo`, the join-kernel defect (A17), cost per run.

**Cost control.** A monthly cost budget ("CI Budget", $30, alerts at 80% actual and 100% forecast) lives in the management account.
Enable Cost Explorer there for per-account views. Spot instances can be reclaimed during a job; RunsOn can fall back to on-demand,
which costs more. Check the Cost Explorer after the first runs.

**Security.** Runner instances execute workflow code inside the account: require approval for workflows from outside contributors,
give the instances the smallest IAM role, and install the GitHub App on selected repositories only. The CLI for agents uses an
Identity Center permission set with minimal rights (quotas, read-only EC2, budgets), never root or long-lived keys.

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

## Why the lavapipe jobs skip the GPU-Cache joins

Measured on 2026-11-04 with `test/cloud/orbstack_lavapipe.sh` (an Ubuntu 24.04 arm64 container on the Mac, Mesa 25.2.8, LLVM 20.1.2, the
same stack as the CI runner) and the CI logs of both Linux jobs: the first GPU-Cache join (the Arrow-source join kernel, run by the
cache scan) keeps lavapipe's LLVM compiler busy for more than 5 minutes (the service's 300-second watchdog for CPU devices then marks
the device lost; on arm64 the service died with signal 11 afterwards) and its resident memory reached 7 GB before the kernel killed the
service (signal 9, in an 8 GB VM). The llvmpipe worker threads were idle: it is the compile, not a stuck kernel. The same join runs in
seconds on the Apple GPU. So the lavapipe CI step sets `GCJOINS=off` (`gpucache_tests.inc` then skips the three join queries per table)
and the Apple Silicon run covers them. Not known yet: whether a real NVIDIA or AMD driver compiles this kernel cheaply (it should: they
do not lower dynamic register-array indexing through LLVM), which a first run on a real GPU will show.

Open follow-ups: the client of a query hangs when the service is killed in the middle of its task (the backend should get an error), and
the service crashes with signal 11 on the device-lost exit path.

## The T4 job is opt-in (2026-11-16)

The hosted T4 has a monthly budget, so `vulkan-check.yml` no longer starts the NVIDIA job on a push or a pull request. Start it by hand
(Actions > vulkan-check > Run workflow, `run_gpu_t4` = true, or `gh workflow run vulkan-check.yml -f run_gpu_t4=true --ref vulkan-gpu`),
after joining the shared T4 queue (`t4_ci_turn.sh` in the mongo checkout) and after telling the user the running T4 total. The lavapipe and
macOS-build jobs still run on every push. A skipped job counts as success for branch protection.
