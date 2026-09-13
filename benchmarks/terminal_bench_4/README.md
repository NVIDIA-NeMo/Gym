# Terminal-Bench 4.0 / OpenCode

This benchmark runs the 66 official TB4 task packages through `harbor_agent_general`.
Harbor owns agent setup, task MCP servers, artifact collection, separate verifier
environments, grading, and ATIF trajectories. Task instructions and tests are not
rewritten or copied into Gym. The dataset and published task images are content-pinned.
The Compose metadata file also pins the two upstream sidecar tags to resolved digests.

The default is one attempt per task, no OpenCode step limit, and the task's official
eight-hour agent timeout. Verifier timeouts and CPU, memory, disk, and GPU requirements come
from each task package. The manifest records 52 CPU single-container, 11 Compose,
and three H100 tasks.

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` in the environment. Sandboxes must
be able to reach the Gym model server; `++use_absolute_ip=true` advertises the host's
address when Gym runs on a compute node.

For separate CPU and GPU deployments, set `OPENSANDBOX_DOMAIN_CPU`,
`OPENSANDBOX_API_KEY_CPU`, `OPENSANDBOX_DOMAIN_GPU`, and `OPENSANDBOX_API_KEY_GPU`,
then add `++tb4_split_sandbox_endpoints=true`. EFB enables this mode for TB4; use
the environment file containing those four variables (for example `.env_combined`).
The benchmark remains one evaluation. Each environment selects the GPU endpoint
when its effective GPU count is positive, otherwise the CPU endpoint. Separate
verifiers select independently; all TB4 Compose services use the CPU endpoint.
The selected pool is recorded in sandbox metadata. API keys are read into private
provider configuration at runtime, without changing shared configuration or
serializing keys into Harbor job files. Missing scoped credentials fail explicitly.
This simple routing is for TB4's CPU-only Compose tasks; it is not per-service
routing for a Compose application with mixed CPU and GPU services.

Resource requests explicitly match each task or Compose service's limits through
`sandbox_provider_options.resource_requests: limits`. CPU values use cores and
memory values use `Mi`/`Gi` in both API fields, avoiding deployment-specific request
defaults. Services with no declared resources retain the SDK's one-CPU, 2-GiB
defaults. Commands use `bash -c`, matching Harbor 0.23's Docker backend and keeping
interactive-shell startup warnings out of machine-parsed setup output.

Sandbox creation retries transient failures, including client, server, and readiness
timeouts, five times after the initial attempt. Backoff is randomized exponentially
from 0–5 seconds initially up to 0–60 seconds. Agent commands and whole trials are
not replayed by this policy. Failed requests without a returned sandbox handle are
not explicitly deleted by the client.

Create HTTP requests are capped at 120 seconds and individual create attempts at
150 seconds, leaving room for retries before Harbor's task setup deadline. The
whole Compose collection has a 1200-second limit, also subject to that task deadline.
Read-only command-status polling uses 20-second attempts and five randomized retries;
command submissions are not retried. Official verifier deadlines remain unchanged.

For `nextjs-performance`, `sandbox_env_by_task` sets `CIRCLE_NODE_TOTAL=3` in the
agent and separate verifier environments. [Next.js 15.4.10](https://github.com/vercel/next.js/blob/v15.4.10/packages/next/src/server/config-shared.ts#L1340)
uses this value minus one for build workers. Without this runtime override, it sees
all 192 host CPUs on the tested deployment and starts 191 workers despite the task's
two-CPU quota, causing the four-GiB verifier container to be OOM-killed. The override
uses two build workers and preserves the official task files, resource limits, and
grading deadline. Other tasks are unaffected; explicit `sandbox_env` values can
override these task defaults. Record this runtime adaptation when comparing scores.

The configured GPU deployment must serve H100s. `sandbox_request_gpu_type: false`
omits the deployment's unsupported `gpu_type` filter while preserving the task's GPU
count and all other resources. Set this option to true when using a deployment that
supports GPU type selection. Task packages retain their original H100 requirement.

Sandboxes use an eight-hour lifetime, renewed every 30 minutes while Gym owns them.
This keeps setup and artifact collection from consuming the official agent budget
on deployments that cap individual lifetimes at eight hours. The sandbox service
must support expiration renewal. Renewal stops before cleanup and does not change
agent or verifier timeouts; failures are reported as infrastructure errors.

```sh
gym eval prepare --benchmark terminal_bench_4/opencode
gym eval run --benchmark terminal_bench_4/opencode \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4/rollouts.jsonl --concurrency 8 \
  ++use_absolute_ip=true ++tb4_jobs_dir=results/tb4/harbor
```

For a CPU smoke run:

```sh
gym eval prepare --benchmark terminal_bench_4/opencode \
  '++prepare_script_args.task_names=[formal-crypto,ks-solver-cpp,interleaved-vigenere]'
# Add ++tb4_max_steps=3 to the eval run command.
```

Preparation also accepts `++prepare_script_args.category=cpu` (or `compose`, `gpu`).
These filters only select tasks. Always prepare again without filters for a full run.
An eight-hour task budget requires a scheduler allocation long enough for setup,
agent execution, and verification; a four-hour smoke allocation is not an official
score run. Official leaderboard submissions use five attempts per task, while this
integration defaults to one for coverage testing.

Compose services run through `AsyncSandboxCompose`, including dependency health
checks, sidecar artifact collection, and TCP forwarding for the two tasks that use
`network_mode: service:...`. The adapter checks `SYS_PTRACE` and shared-memory
requirements against the sandbox deployment and fails explicitly if they cannot
be honored. TCP forwarding provides the declared localhost ports; it does not
create a shared Linux network namespace.

`compose-images.json` records verified Linux/amd64 OCI startup metadata for all
38 Compose images. Normalization applies Harbor's standard prebuilt main command,
image entrypoints, working directories, users, and exposed ports before creating
sandboxes. Original task files remain unchanged. Resolved Compose files are saved
beside the Harbor trial artifacts.

## Provenance

- [Release announcement](https://www.tbench.ai/news/terminal-bench-4-0)
- Dataset: `terminal-bench/terminal-bench@4.0.0`
- Dataset digest: `sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732`
- [Source release](https://github.com/harbor-framework/terminal-bench/tree/452bf305c6daa62fc59061d22133a7cbc7c1572e)
- Runtime: Harbor 0.23.0; OpenCode 1.17.11.
