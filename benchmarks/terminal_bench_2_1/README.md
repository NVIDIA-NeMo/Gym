# Terminal-Bench 2.1

Terminal-Bench 2.1 evaluates agents on tasks in isolated containers. The benchmark
prepares task data and owns sandbox setup, verification, and cleanup. Pi, Codex,
OpenCode, and OpenClaw can run their model/tool loops in the task sandbox through
Gym's Environment Server.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

## Prepare data

Run from the Gym checkout with the benchmark's preparation dependencies installed:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
```

Keep task tests and solutions on the host; do not mount the benchmark checkout
into task containers. Resources uploads verifier assets after the agent closes
and rejects tasks whose local `tests/test.sh` is missing. Do not enable
golden-patch mode for model runs.

## Configure a run

Compose the Resources, harness, and Environment Server configs. Save this Codex
example as `run.yaml`:

```yaml
config_paths:
  - resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml
  - responses_api_agents/codex_agent/configs/codex_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

environment_routing_mode: legacy
environment_server_name: single_agent_turn_legacy

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

The prepared flat JSONL uses `single_agent_turn_legacy` to adapt its input/output
format to the shared session lifecycle. Environment Server seeds Resources and
the agent, runs the agent, closes it before verification, then closes Resources.
Resources owns the sandbox throughout. Keep one worker for both Resources and
the agent because their sessions are process-local.

### Choose a harness

To use another harness, replace its config import, the entire `codex_agent`
settings block, `agent_server.name`, and the `--agent` argument below. The settings
in the last column belong under `<name>.responses_api_agents.<name>`.
`--agent` selects an already configured harness; it does not perform this rebinding.

| Harness requirements | Config import | Name | Example settings |
| --- | --- | --- | --- |
| [Pi](../../responses_api_agents/pi_agent/README.md) | `responses_api_agents/pi_agent/configs/pi_agent.yaml` | `pi_agent` | `thinking: "off"`, `max_output_tokens: 8192`, `timeout: 600` |
| [Codex](../../responses_api_agents/codex_agent/README.md) | `responses_api_agents/codex_agent/configs/codex_agent.yaml` | `codex_agent` | `model_context_window: 32768`, `model_auto_compact_token_limit: 28672`, `timeout: 600` |
| [OpenCode](../../responses_api_agents/opencode_agent/README.md) | `responses_api_agents/opencode_agent/configs/opencode_agent.yaml` | `opencode_agent` | `context_window: 32768`, `max_output_tokens: 8192`, `timeout: 600` |
| [OpenClaw](../../responses_api_agents/openclaw_agent/README.md) | `responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml` | `openclaw_agent` | `context_window: 32768`, `model_timeout_seconds: 600`, `timeout: 600` |

All four pairings are allowed by this benchmark. These are single-task smoke
settings; adjust context windows and budgets for your model and evaluation.
See each harness's README for runtime prerequisites and model requirements.

### Model and sandbox provider

Save this example for a non-reasoning Chat Completions endpoint as `model-provider.yaml`:

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

Gym's vLLM Model Server also adapts Chat Completions to Codex's streaming Responses
API. Adjust parser settings for your model, or select the OpenAI Model Server for
a provider that serves the required API directly. Keep sampling and output caps
on the model server; the Pi/OpenCode agent caps above also limit individual calls.
Codex/OpenClaw sandbox sessions reject request-level `max_output_tokens`.

Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in your private
`env.yaml`. The task container must reach the Gym Model Server; for Docker on a
reachable host, use `++use_absolute_ip=true` as below.

## Run a smoke task

Select one prepared task:

```bash
mkdir -p results
python - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["task_name"] == "terminal-bench/regex-log"]
assert len(rows) == 1
assert (Path(rows[0]["task_folder"]) / "tests/test.sh").is_file()
Path("results/tb21-input.jsonl").write_text(json.dumps(rows[0]) + "\n")
PY

gym env start --config run.yaml --config model-provider.yaml ++use_absolute_ip=true
```

In a second terminal, use the same checkout, Python environment, and configs:

```bash
gym eval run --no-serve --config run.yaml --config model-provider.yaml ++use_absolute_ip=true \
  --agent codex_agent --input results/tb21-input.jsonl \
  --output results/tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Inspect `response.metadata.harness_execution` (`sandbox`), actual tool calls and
results, `ng_agent_observations`, `evaluation_completed`, reward, and verifier
`test_output`. Check failure sidecars and sandbox teardown. An incomplete
evaluation with reward 0 is an infrastructure failure; a completed evaluation
can score 0 for an incorrect solution. A collector exit alone does not establish
a successful smoke, and one task does not establish a benchmark baseline.

## Tasks that start services

Task processes in the Resources-owned sandbox remain available to the verifier.
The supervisor bounds harness termination; Resources stops all remaining processes
by closing the sandbox after verification or episode failure. Unknown harness
termination blocks verification and requires sandbox teardown. Services must use
task-owned files, since agent close removes its temporary session directory.

Services must also outlive the harness's shell tool. `nohup ... &` alone is not
portable across tools: a shell deadline or process-group cleanup can stop the
service. Detach its process group and standard streams, for example with
`subprocess.Popen(start_new_session=True)`, stdin from `DEVNULL`, and stdout/stderr
in task-owned log files. Check that the service is reachable before finishing.
A file-producing smoke does not validate this service-dependent behavior.
