# terminal_bench_2_1

TODO: Describe this benchmark and replace the sample data.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

## Harness and benchmark contract

Environment Server is the default entry point for rollout execution. For session-based
integrations, compose independent Resources, harness, and Environment Server configs.
The single-agent Environment Server binds `resources_server` and `agent_server` and
coordinates setup, execution, verification, and cleanup. The harness does not need a
benchmark-specific runner or another combined preset. For tasks with a Resources-owned
sandbox, the agent borrows that sandbox.

| Responsibility | Session-based task-sandbox execution |
| --- | --- |
| Dataset | The benchmark prepares flat JSONL with `responses_create_params` and its task fields. |
| Collection | `single_agent_turn_legacy` adapts flat request/result rows to the shared `single_agent_turn` execution. |
| Task state | Resources receives `ResourcesSeedSessionRequest`, owns the sandbox, and returns `SandboxAccess` with an absolute workdir. |
| Harness | Receives `AgentSeedSessionRequest`, borrows the sandbox, and runs its own tools/model loop. |
| Verification | EnvironmentServer closes the agent before calling Resources `/verify`. |
| Cleanup | EnvironmentServer calls Resources `/close_session`; Resources destroys the sandbox and retains failed cleanup for retry. |

Here `legacy` describes only the collector's flat-row input/output contract.
It does not select the agent's old direct `/run` path. Moving to materialized taskset
input and `single_agent_turn` is a separate input-format migration; changing
only the EnvironmentServer config is insufficient.

TB's `_ResourcesSessionState` is private Resources bookkeeping for request identity
and seed/verdict replay. Its state fields reflect TB's verification needs; the public
session protocol defines the Resources/Agent boundary. Harness process execution uses
Gym's shared `SandboxSession` and `supervisor_client` in `nemo_gym.agent_utils`;
Resources request/verdict bookkeeping remains benchmark-owned. These are separate responsibilities.

Choose either the [Pi](#pi-in-a-task-sandbox) or [Codex](#codex-in-a-task-sandbox)
recipe below. Each recipe provides its own `run.yaml` and `model-provider.yaml`.

## Pi in a task sandbox

Compose the independent benchmark, Pi, and single-agent Environment Server configs.
Pi borrows the task sandbox; Resources owns it. The Environment Server closes Pi before
verification, then closes Resources, including when agent setup or execution fails.
Existing OpenCode and Terminus2 recipes keep their legacy seed/verify paths.

Save as `run.yaml` in the Gym checkout:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/pi_agent/configs/pi_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: pi_agent
      resources_tool_transports: []

pi_agent:
  responses_api_agents:
    pi_agent:
      thinking: high
      max_output_tokens: 32768
      timeout: 600

terminal_bench_2_1_resources_server:
  resources_servers:
    terminal_bench_2_1:
      evaluation_timeout: 300
```

Save the independent model/provider settings as `model-provider.yaml`.
The task container must be able to reach the Gym Model Server.

```yaml
config_paths:
  - nemo_gym/sandbox/providers/docker/configs/docker.yaml
  - responses_api_models/openai_model/configs/openai_model.yaml
```

These are single-task smoke limits, not full-benchmark settings. Put `policy_base_url`,
`policy_api_key`, and `policy_model_name` in your private `env.yaml`. The task container
must reach the Gym model proxy; use `++use_absolute_ip=true` for Docker on a reachable host.
Both servers use one worker because sessions are process-local.

Prepare the pinned benchmark checkout and select one task:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
mkdir -p results
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["task_name"] == "terminal-bench/regex-log"]
assert len(rows) == 1
assert (Path(rows[0]["task_folder"]) / "tests/test.sh").is_file()
Path("results/pi-tb21-input.jsonl").write_text(json.dumps(rows[0]) + "\n")
PY
gym env start --config run.yaml --config model-provider.yaml ++use_absolute_ip=true
```

In a second terminal, in the same checkout and Python environment:

```bash
gym eval run --no-serve --config run.yaml --config model-provider.yaml ++use_absolute_ip=true \
  --agent pi_agent --input results/pi-tb21-input.jsonl \
  --output results/pi-tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Keep task tests/solutions on the host; do not mount the benchmark checkout into the task
sandbox. Only verifier assets are uploaded, after Pi closes. Resources session setup fails early if
the host task's `tests/test.sh` is missing. Do not enable golden-patch mode for model runs.
Pi installs its private Node/Pi runtime and, if absent, Python 3 and curl using apt/apk as root;
non-root images must contain bootstrap dependencies. Existing task runtimes are not replaced.
See [Pi requirements and lifecycle](../../responses_api_agents/pi_agent/README.md).

Inspect the rollout's `response.metadata.harness_execution` (`sandbox`), nonempty output,
`ng_agent_observations`, `evaluation_completed`, reward, and verifier `test_output`.
Also inspect failure sidecars and confirm sandbox teardown. An incomplete evaluation with
reward 0 is an infrastructure failure, not a valid model score. A successful collector exit
alone does not establish a passing smoke; one task does not establish a benchmark baseline.

After a successful harness exit, the shared supervisor preserves task processes in
the Resources-owned sandbox so verification can inspect running services. Resources
stops the sandbox after verification. Timeout, cancellation, and failed harness exits
still stop descendants. Services must use task-owned files, not the agent's temporary
session directory, which agent close removes. A file-producing smoke alone does not
validate service-dependent tasks.

## Codex in a task sandbox

Compose the independent benchmark, Codex, and single-agent Environment Server configs.
Codex borrows the task sandbox; Resources owns it. The Environment Server closes Codex before
verification, then closes Resources, including when agent setup or execution fails.
Existing OpenCode and Terminus2 recipes keep their legacy seed/verify paths.

Save as `run.yaml` in the Gym checkout:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/codex_agent/configs/codex_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: codex_agent
      resources_tool_transports: []

codex_agent:
  responses_api_agents:
    codex_agent:
      model_context_window: 32768
      model_auto_compact_token_limit: 28672
      timeout: 600

terminal_bench_2_1_resources_server:
  resources_servers:
    terminal_bench_2_1:
      evaluation_timeout: 300
```

Save the independent model/provider settings as `model-provider.yaml`.
The task container must be able to reach the Gym Model Server.

```yaml
config_paths:
  - nemo_gym/sandbox/providers/docker/configs/docker.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml

policy_model:
  responses_api_models:
    vllm_model:
      uses_reasoning_parser: false
      uses_interleaved_reasoning: false
      sampling_overrides:
        temperature: 0.0
        max_tokens: 8192
```

This example uses a non-reasoning Chat Completions endpoint; the Gym vLLM Model
Server adapts it to Codex's streaming Responses API. Configure the parser flags for
your model, or use the OpenAI Model Server with a provider that serves the Responses API directly.
These are single-task smoke limits, not full-benchmark settings. Set the context window
to the served model limit. Keep sampling and per-call output limits on the Gym Model
Server; Codex task-sandbox sessions reject request-level sampling/output overrides. Put `policy_base_url`,
`policy_api_key`, and `policy_model_name` in your private `env.yaml`. The task container
must reach the Gym model proxy; use `++use_absolute_ip=true` for Docker on a reachable host.
Both servers use one worker because sessions are process-local.

Prepare the pinned benchmark checkout and select one task:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
mkdir -p results
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["task_name"] == "terminal-bench/regex-log"]
assert len(rows) == 1
assert (Path(rows[0]["task_folder"]) / "tests/test.sh").is_file()
Path("results/codex-tb21-input.jsonl").write_text(json.dumps(rows[0]) + "\n")
PY
gym env start --config run.yaml --config model-provider.yaml ++use_absolute_ip=true
```

In a second terminal, in the same checkout and Python environment:

```bash
gym eval run --no-serve --config run.yaml --config model-provider.yaml ++use_absolute_ip=true \
  --agent codex_agent --input results/codex-tb21-input.jsonl \
  --output results/codex-tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Keep task tests/solutions on the host; do not mount the benchmark checkout into the task
sandbox. Only verifier assets are uploaded, after Codex closes. Resources session setup fails early if
the host task's `tests/test.sh` is missing. Do not enable golden-patch mode for model runs.
Codex installs its pinned CLI and private Node runtime and, if absent, Python 3 and
bootstrap packages using apt/apk as root;
non-root images must contain bootstrap dependencies. Existing task runtimes are not replaced.
See [Codex requirements and lifecycle](../../responses_api_agents/codex_agent/README.md).

Inspect the rollout's `response.metadata.harness_execution` (`sandbox`), nonempty output,
`ng_agent_observations`, `evaluation_completed`, reward, and verifier `test_output`.
Also inspect failure sidecars and confirm sandbox teardown. An incomplete evaluation with
reward 0 is an infrastructure failure, not a valid model score. A successful collector exit
alone does not establish a passing smoke; one task does not establish a benchmark baseline.

After a successful harness exit, the shared supervisor preserves task processes in
the Resources-owned sandbox so verification can inspect running services. Resources
stops the sandbox after verification. Timeout, cancellation, and failed harness exits
still stop descendants. Services must use task-owned files, not the agent's temporary
session directory, which agent close removes. A file-producing smoke alone does not
validate service-dependent tasks.
