# Terminal-Bench 4.0

TB4 uses separate Gym agent and resources servers. The resources server resolves
66 official task packages from the pinned manifest, provisions their environments,
and runs the official verifier. The selected agent owns model calls and harness
execution on the supplied main sandbox.

## Profiles

- `terminal_bench_4/opencode`: OpenCode **1.17.11** through `opencode_sandboxed_agent`.
- `terminal_bench_4/miniswe`: mini-SWE **2.1.0** `DefaultAgent`, with a generic
  text-action prompt, Gym Responses model adapter, and task-local MCP CLI.

Both profiles load `resources.yaml`. Harness settings live in the profile;
provisioning, CPU/GPU endpoint aliases, task budgets, and cleanup live in the
resources configuration. The existing general Harbor and SWE-bench integrations
remain available with their existing dependencies and defaults.

The resources runtime uses Gym's native TB4 lifecycle, with no Harbor package
required. Resources own preparation, deadlines, main/sidecar artifact collection,
separate verification, and cleanup. See the [handoff contract](../../resources_servers/terminal_bench_4/README.md)
and [native lifecycle notes](native-lifecycle.md).

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
Credentials resolve in server configuration; handoffs carry only a provider
alias, sandbox ID, and working directory. Keep resolved configs private.

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
gym eval prepare --benchmark terminal_bench_4/opencode
gym eval run --benchmark terminal_bench_4/opencode \
  --model-type vllm_model --model-url http://MODEL_HOST:8000/v1 \
  --model MODEL_NAME --output results/tb4/rollouts.jsonl --concurrency 8 \
  ++use_absolute_ip=true ++tb4_split_sandbox_endpoints=true
```

Use `terminal_bench_4/miniswe` to select mini-SWE. The model-server URL must be
reachable from the OpenCode sandbox; mini-SWE's loop runs in the agent worker.
The installer supports apt, apk, and dnf task images and needs their normal
package permissions. MCP tasks in the mini-SWE profile also need Python venv/pip
for its pinned task-local `mcp==1.29.0` client.

For capped smoke runs, add `++tb4_max_steps=3` and
`++tb4_agent_max_timeout_sec=900`. The cap can only shorten the task's official
agent budget. Default runs have no step cap or timeout override. Installation
uses the separate 360-second harness-setup budget. Provider renewal keeps
resources alive without extending agent execution.

Select tasks during preparation:

```sh
gym eval prepare --benchmark terminal_bench_4/opencode \
  '++prepare_script_args.task_names=[formal-crypto,interleaved-vigenere,ks-solver-cpp]'
```

Preparation also accepts `++prepare_script_args.category=cpu`, `compose`, or `gpu`.
Prepare again without filters for all tasks. Each profile defaults to one attempt;
official leaderboard submissions use five. Scheduler allocations must cover
setup, the full official agent budget, and verification.

## Validation

Validate CPU, then Compose, then GPU. Use the existing task-health records as
baseline evidence. A grade of zero can be a healthy smoke outcome; absent setup,
model execution, grading, or required artifacts is not a successful model run.
Infrastructure failures carry `infrastructure_error` and `_ng_failure_class` and
must be excluded from model-negative aggregates.

The standalone smoke runner starts real Gym HTTP agent, resources, and model
servers on loopback. mini-SWE calls the Gym model server; remote OpenCode uses the
public OpenAI endpoint directly in this mode. This validates the split lifecycle,
but does not certify remote Gym model routing or semantic-turn observability.
It requires the existing sandbox endpoint credentials and `OPENAI_API_KEY`.

```sh
PYTHONPATH=. python benchmarks/terminal_bench_4/smoke.py \
  --harness opencode --category cpu --env-file /path/to/private.env \
  --baseline-health benchmarks/terminal_bench_4/health-baseline.json \
  --output results/tb4-smoke/cpu
```

Repeat with `--harness miniswe`, then the Compose and GPU categories. `health.json`
requires both model-output evidence and an official grade. Inspect trajectories,
verifier output, and resource cleanup before promoting coverage. Capped runs are
not benchmark scores; missing submissions can exit grading before deeper tests.

See [native lifecycle notes](native-lifecycle.md) for the current implementation and
validation record. The [earlier migration notes](migration.md) describe the historical
Harbor-backed reference.
