# terminal_bench_2_1

TODO: Describe this benchmark and replace the sample data.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

## Harness and benchmark contract

Use the same composition as SWE-bench Pro: import an independent Resources config,
an independent harness config, and the single-agent EnvironmentServer config.
The EnvironmentServer binds `resources_server` and `agent_server`; the harness does
not need a benchmark-specific runner or another combined preset.

| Responsibility | SWE-bench Pro and Terminal-Bench 2.1 |
| --- | --- |
| Dataset | The benchmark prepares flat JSONL with `responses_create_params` and its task fields. |
| Collection | `single_agent_turn_legacy` adapts flat request/result rows to the shared `single_agent_turn` execution. |
| Task state | Resources receives `ResourcesSeedSessionRequest`, owns the sandbox, and returns `SandboxAccess` with an absolute workdir. |
| Harness | Receives `AgentSeedSessionRequest`, borrows the sandbox, and runs its own tools/model loop. |
| Verification | EnvironmentServer closes the agent before calling Resources `/verify`. |
| Cleanup | EnvironmentServer calls Resources `/close_session`; Resources destroys the sandbox and retains failed cleanup for retry. |

Here `legacy` describes only the collector's flat-row input/output contract.
It does not select the agent's old direct `/run` path. Moving to materialized taskset
input and `single_agent_turn` is a separate migration for both benchmarks; changing
only the EnvironmentServer config is insufficient.

TB's `_ResourcesSessionState` is private Resources bookkeeping for request identity
and seed/verdict replay. Its state fields reflect TB's verification needs; the public
session protocol and sandbox ownership match SWE-Pro. Harness process execution uses
Gym's shared `SandboxSession` and `supervisor_client` in `nemo_gym.agent_utils`,
as in Hermes #3961; Resources
request/verdict bookkeeping remains benchmark-owned. These are separate responsibilities.

## OpenClaw in a task sandbox

Compose the independent benchmark, harness, and single-agent EnvironmentServer
configs, as in SWE-Pro. OpenClaw borrows the task sandbox; Resources owns it.
The EnvironmentServer closes the agent before verification and closes Resources
afterward, including after failed setup or execution. Existing legacy recipes
retain their seed/verify path.

Save as `run.yaml` in the Gym checkout:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

environment_routing_mode: legacy
environment_server_name: single_agent_turn_legacy

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: terminal_bench_2_1_resources_server
      agent_server:
        name: openclaw_agent
      resources_tool_transports: []

openclaw_agent:
  responses_api_agents:
    openclaw_agent:
      context_window: 32768
      model_timeout_seconds: 600
      timeout: 600

terminal_bench_2_1_resources_server:
  resources_servers:
    terminal_bench_2_1:
      evaluation_timeout: 300
```

Save the independent model/provider settings as `model-provider.yaml`. The task
container must be able to reach the Gym Model Server. This example uses a
local vLLM endpoint through Gym's vLLM adapter, which enforces the sampling
overrides below.

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

These are single-task smoke limits, not full-benchmark settings. Set the context
window to the actual model capacity and leave room for the prompt. OpenClaw rejects request/config `max_output_tokens` in native sessions; the
model-server override below supplies the per-call limit.

Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in your private
`env.yaml`. For Docker on a reachable host, pass `++use_absolute_ip=true` so the
task container can reach Gym. Both Resources and the agent use one worker because
their sessions are process-local. This TB pairing is in the allow-list; no
unsupported-pairing override is needed.

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
Path("results/openclaw-tb21-input.jsonl").write_text(json.dumps(rows[0]) + "\n")
PY
gym env start --config run.yaml --config model-provider.yaml ++use_absolute_ip=true
```

In a second terminal, in the same checkout and Python environment:

```bash
gym eval run --no-serve --config run.yaml --config model-provider.yaml ++use_absolute_ip=true \
  --agent openclaw_agent --input results/openclaw-tb21-input.jsonl \
  --output results/openclaw-tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Keep task tests/solutions on the host; do not mount the benchmark checkout into
the task sandbox. Verifier assets are uploaded only after OpenClaw closes.
Resources rejects a missing local `tests/test.sh`; do not enable golden-patch
mode for model runs. The harness installs missing Python/Bash with apt/apk when
running as root and preserves existing task runtimes. Non-root images must
provide prerequisites. See [OpenClaw runtime and lifecycle requirements](../../responses_api_agents/openclaw_agent/README.md).

Inspect `response.metadata.harness_execution` (`sandbox`), actual tool calls and
results, `ng_agent_observations`, `evaluation_completed`, reward and verifier
`test_output`. Check failure sidecars and sandbox teardown. Reward 0 with incomplete
evaluation is an infrastructure failure. A completed verifier can return reward 0
for a wrong answer; one task does not establish a benchmark baseline.

The shared supervisor stops agent-started task processes before verification.
Tasks requiring those services to stay running are not reliably supported; a
file-producing smoke does not validate service lifetime.
