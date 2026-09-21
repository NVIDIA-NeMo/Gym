# Terminal-Bench 4.0

[Official Terminal-Bench 4.0 benchmark](https://www.tbench.ai/news/terminal-bench-4-0) (66 tasks).

## Profiles

- `terminal_bench_4/miniswe`: mini-SWE **2.4.6** `DefaultAgent`, with upstream
  `mini.yaml` prompts, native bash tool calls through Gym's Responses model
  adapter, and task-local MCP CLI.

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
- **Verifier timeouts:** preserve the current behavior until we understand how
  often these occur on real workloads. Timeouts remain incomplete evaluations
  classified as infrastructure failures and excluded from default aggregates,
  whereas AA describes counting them as failures.

The adapter also keeps `cost_limit=0` and unattended `DefaultAgent` execution.
Task MCP schemas and CLI invocation instructions are added to the prompt; MCP
calls run through native bash. These adaptations mean this profile is not an
exact reproduction of AA's evaluation.

The benchmark defaults can be overridden with `++tb4_max_steps=...` and
`++tb4_step_timeout_sec=...`. An explicit `++tb4_max_steps=0` disables the step cap.
The generic Python adapter retains its 600-second command timeout default; the
benchmark supplies 30 seconds. Task-specific overall deadlines remain separate.

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

```sh
gym eval prepare --benchmark terminal_bench_4/miniswe
gym eval run --benchmark terminal_bench_4/miniswe \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4/rollouts.jsonl --concurrency 8 \
  ++use_absolute_ip=true ++tb4_split_sandbox_endpoints=true
```

mini-SWE's loop runs in the resources process, using its existing sandbox and
calling the Gym model server. The agent endpoint forwards the collector's run
request and returns the result; it does not manage task environments.
MCP tasks also need Python venv/pip
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

The standalone smoke runner starts real Gym HTTP agent, resources, and model
servers on loopback. The resources process runs the complete episode and calls
the Gym model server through the mini-SWE harness.
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
