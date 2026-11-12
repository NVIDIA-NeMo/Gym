# Terminal-Bench 4.0

[Official Terminal-Bench 4.0 benchmark](https://www.tbench.ai/news/terminal-bench-4-0) (66 tasks).

## Profiles

- `terminal_bench_4/miniswe`: mini-SWE **2.4.6** `DefaultAgent`, with upstream
  `mini.yaml` prompts, native bash tool calls through Gym's Responses model
  adapter, and task-local MCP CLI, in [`mini_swe_agent`](../../responses_api_agents/mini_swe_agent/README.md)
  agent sessions. The TB4 resources server lends each task sandbox, and
  `single_agent_turn_legacy` drives the episode and grades it after the agent closes.
- `terminal_bench_4/miniswe_legacy_agent` (deprecated): the same mini-SWE version
  through the deprecated `miniswe_sandboxed_agent` over the `legacy_agent` `/run`
  relay. The two adapters are configured differently, so do not compare their
  scores as the same measurement.

### Session profile

Rows come from `prepare_sessions.py` (`data/benchmark_sessions.jsonl`): each embeds the
task's `instruction.md` as one user message and its `[agent].timeout_sec` as
`metadata.agent_timeout_sec`, read from the pinned package. mini-SWE runs for the
smaller of that budget and `++tb4_agent_max_timeout_sec` (default 28800 seconds),
as on the `/run` path. Before lending the sandbox, the resources server writes the
task's MCP servers and skills directory to `/tmp/.nemo-gym/task.json`, the file the
Harbor environment server writes, so MCP and skills tasks keep their tools.
mini-SWE runs as the image's default user; no TB4 task sets `[agent].user`.

With `++tb4_split_sandbox_endpoints=true`, each task sandbox is lent through its
endpoint pool's provider block (`terminal_bench_4_cpu_sandbox` or
`terminal_bench_4_gpu_sandbox` in `resources.yaml`), so the agent reconnects to the
deployment the task runs on. With one deployment, it is lent through
`terminal_bench_4_sandbox`.

mini-SWE's trajectory is returned in the agent close bundle. This profile writes no
`<tb4_jobs_dir>/agent` artifacts and has no `tb4_sandbox_model_base_url`: mini-SWE
calls the Gym model server from inside the sandbox at the server's own address, so a
loopback address is refused at seed (hence `++use_absolute_ip=true`). Episodes are bounded by
`++tb4_episode_timeout_sec` (default 54000 seconds: setup, the 8-hour agent
budget, and the slowest verifier).

## Artificial Analysis comparison

The version, prompts, 500-step limit, and 30-second command timeout follow
[Artificial Analysis's Terminal-Bench 4.0 methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
(checked September 16, 2026). Prompts and command environment defaults are loaded
from the pinned package's `mini.yaml`, not copied into this repository.

The following choices remain intentional differences:

- **Repeats:** keep `num_repeats: 1` in this profile. Clients must explicitly set
  `++num_repeats=3` to reproduce AA's repetition protocol and average pass@1 over
  all three attempts, rather than reporting pass@3.
- **Observations:** retain full command output, without the upstream template's
  first/last 5,000-character truncation. It is unclear whether AA intended that
  truncation or was unaware of the upstream behavior. No history compaction or
  summarization is applied.

The adapter also keeps `cost_limit=0` and unattended `DefaultAgent` execution.
Task MCP schemas and CLI invocation instructions are added to the prompt; MCP
calls run through native bash. These adaptations mean this profile is not an
exact reproduction of AA's evaluation.

The benchmark defaults can be overridden with `++tb4_max_steps=...` and
`++tb4_step_timeout_sec=...`. An explicit `++tb4_max_steps=0` disables the step cap.
The generic Python adapter retains its 600-second command timeout default; the
benchmark supplies 30 seconds. Task-specific overall deadlines remain separate.

## Failure scoring

A verifier that exhausts its own time budget scores zero and stays in the
pass-rate denominator (`failure_kind: verifier_error`, `mask_sample: false`).
It retains `evaluation_completed: false` because no official reward was produced.
Provider/socket timeouts and sandbox startup failures remain infrastructure
failures; an earlier agent infrastructure failure also keeps that classification.

mini-SWE's `LimitsExceeded` means its step or cost budget was reached. This
profile permits 500 steps and disables the cost budget. The official verifier
still grades the artifacts after this exit. It does not indicate a filesystem
kill. Explicit provider reports that the sandbox exceeded its memory or local
ephemeral-storage allocation score zero, with `sandbox_resource_limit` recording
the resource. See the [resource server's failure contract](../../resources_servers/terminal_bench_4/README.md#contract).

## Sandbox request timeouts

OpenSandbox SDK requests use a five-minute timeout for both freshly created
sandboxes and reconnected agent handles (`create.request_timeout_s` and
`connection.request_timeout_s`). The created handle retains its creation
connection settings, so both values must be set. Status polls keep their separate
20-second cap. These transport budgets do not extend task, command, or verifier
execution limits; an exhausted transport timeout still reports an infrastructure
failure, and an uncertain command submission is not replayed.

## Dataset and deployment

The dataset is `terminal-bench/terminal-bench@4.0.0`, pinned to
`sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732`.
`manifest.json` retains all 52 CPU, 11 CPU Compose, and 3 H100 tasks and their
individual package digests. Preparation writes identities only; the resources
server validates the dataset and task digests before allocation.

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for one deployment. For the
established split deployment, set `OPENSANDBOX_DOMAIN_CPU`,
`OPENSANDBOX_API_KEY_CPU`, `OPENSANDBOX_DOMAIN_GPU`, and
`OPENSANDBOX_API_KEY_GPU`, then use `++tb4_split_sandbox_endpoints=true`.
The resources server resolves credentials and returns the sandbox descriptor and
connection configuration to the agent server, which reconnects for execution.
Keep resolved configs and internal seed responses private.

Agent and verifier select the endpoint independently from their official GPU
requirements. The selected GPU deployment must supply H100s; its unsupported
`gpu_type` filter is disabled explicitly. Each task retains its CPU, memory,
storage, GPU count, build budget, agent budget, and verifier deadline.

New agent, verifier, and Compose sandboxes request process-level OOM handling
with `nemo.nvidia.com/oom-kill: process`. A memory-limit breach can kill a task
process while leaving the sandbox available for subsequent commands. The agent
itself can still be selected as the OOM victim. Set this metadata value to `pod`
to restore whole-sandbox OOM handling.

Compose uses the existing digest-verified `compose-images.json`. It preserves
startup commands, users, dependency health checks, shared-memory requirements,
and sidecar artifact collection. Declared-port TCP forwarding for shared-network
mode is not full Linux namespace sharing. Required capabilities and privileged
setup must be supplied by the deployment. The explicit `nextjs-performance`
overlay `CIRCLE_NODE_TOTAL=3` matches its two-CPU allocation; disclose this runtime
adaptation in comparisons. No official task package or grader is edited.

The OpenSandbox adapter maps separate-verifier `no-network` policies to deny-all
egress. The deployment must enforce that policy for hostname and direct-IP
traffic. Dynamic allowlists and offline Compose are not supported by this adapter.

## Run

The commands are the same for both profiles; replace `terminal_bench_4/miniswe` with
`terminal_bench_4/miniswe_legacy_agent` for the `/run` relay.

```sh
gym eval prepare --benchmark terminal_bench_4/miniswe
gym eval run --benchmark terminal_bench_4/miniswe \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4/rollouts.jsonl --concurrency 8 \
  ++use_absolute_ip=true ++tb4_split_sandbox_endpoints=true
```

In the session profile, mini-SWE's runtime is built by the agent server and uploaded
to the lent sandbox, which needs glibc 2.28+ or musl and about 450 MB in `/tmp`.
In the relay profile, mini-SWE's loop runs in the resources process, using its existing
sandbox and calling the Gym model server; MCP tasks there also need Python venv/pip
for its pinned task-local `mcp==1.29.0` client.

Select tasks during preparation:

```sh
gym eval prepare --benchmark terminal_bench_4/miniswe \
  '++prepare_script_args.task_names=[formal-crypto,interleaved-vigenere,ks-solver-cpp]'
```

Preparation also accepts `++prepare_script_args.category=cpu`, `compose`, or `gpu`.
Prepare again without filters for all tasks. The profile defaults to one attempt;
AA reproduction requires explicitly setting `++num_repeats=3` (the official
Terminal-Bench leaderboard uses a separate five-attempt protocol). Scheduler allocations must cover
setup, the full official agent budget, and verification.

## Sandbox startup

Each OpenSandbox creation or reconnect attempt has a five-minute client budget
(`environment.sandbox_provider.opensandbox.create.timeout_s` and
`connect_attempt_timeout_s`). The task package's overall environment build
budget still bounds provisioning, including retries and Compose services.
The 1,200-second readiness setting is an upper bound within those outer budgets.

## Infra validation

For capped smoke runs, add `++tb4_max_steps=3` and
`++tb4_agent_max_timeout_sec=900`. The cap can only shorten the task's official
agent budget. Default benchmark runs allow 500 steps and 30 seconds per command,
with no override of the task's overall agent timeout. Installation
uses the separate 360-second harness-setup budget. Provider renewal keeps
resources alive without extending agent execution.

Validate CPU, then Compose, then GPU. A grade of zero can be a healthy smoke
outcome; absent setup, model execution, grading, or required artifacts is not a
successful model run.
Infrastructure failures carry `infrastructure_error` and `_ng_failure_class` and
must be excluded from model-negative aggregates.

The standalone smoke runner exercises the relay profile (`miniswe_legacy_agent.yaml`).
It starts real Gym HTTP agent, resources, and model servers on loopback. The agent runs mini-SWE and calls the Gym model server;
resources owns task provisioning, grading, and cleanup.
It requires the existing sandbox endpoint credentials and `OPENAI_API_KEY`.

```sh
PYTHONPATH=. python benchmarks/terminal_bench_4/smoke.py \
  --harness miniswe --category cpu --env-file /path/to/private.env \
  --output results/tb4-smoke/cpu
```

Repeat for the Compose and GPU categories. `health.json`
requires both model-output evidence and an official grade. Inspect trajectories,
verifier output, and resource cleanup before promoting coverage. Capped runs are
not benchmark scores; missing submissions can exit grading before deeper tests.
