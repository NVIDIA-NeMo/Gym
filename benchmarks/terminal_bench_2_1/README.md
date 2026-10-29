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
  - environment_servers/single_agent_turn/configs/single_agent_turn.yaml

environment_routing_mode: taskset
environment_server_routes:
  terminal_bench_2_1: single_agent_turn

single_agent_turn:
  environment_servers:
    single_agent_turn:
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

The smoke input below wraps the prepared row in `task_id` and `task_input`.
Its `terminal_bench_2_1` taskset routes directly to `single_agent_turn`.
Environment Server seeds Resources and the agent, runs the agent, closes it before
verification, then closes Resources. Resources owns the sandbox throughout.
Keep one worker for both Resources and the agent because their sessions are
process-local.

### Choose a harness

To use another harness, replace its config import, the entire `codex_agent`
settings block, and `agent_server.name`. The settings in the last column belong
under `<name>.responses_api_agents.<name>`. The Environment Server's binding selects
the harness for this taskset.

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

from nemo_gym.data.task_materialization import materialize_task

rows = [json.loads(line) for line in Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["task_name"] == "terminal-bench/regex-log"]
assert len(rows) == 1
assert (Path(rows[0]["task_folder"]) / "tests/test.sh").is_file()
task = materialize_task(rows[0], taskset="terminal_bench_2_1")
Path("results/tb21-input.jsonl").write_text(json.dumps(task) + "\n")
PY

gym env start --config run.yaml --config model-provider.yaml ++use_absolute_ip=true
```

In a second terminal, use the same checkout, Python environment, and configs:

```bash
gym eval run --no-serve --config run.yaml --config model-provider.yaml ++use_absolute_ip=true \
  --input results/tb21-input.jsonl \
  --output results/tb21-rollouts.jsonl --limit 1 --num-repeats 1 --concurrency 1
```

Inspect `response.metadata.harness_execution` (`sandbox`), actual tool calls and
results, `ng_agent_observations`, `evaluation_completed`, reward, and verifier
`test_output`. Check failure sidecars and sandbox teardown. An incomplete
evaluation with reward 0 is an infrastructure failure; a completed evaluation
can score 0 for an incorrect solution. A collector exit alone does not establish
a successful smoke, and one task does not establish a benchmark baseline.

## Tasks that start services

The supervisor stops live processes in the harness group before capture or verification.
On timeout or close, it stops the harness, then kills within the cleanup deadline the group,
the harness's descendants in other process groups (such as in-flight tools in separate
sessions) and processes that inherited the harness's `NEMO_GYM_SUPERVISOR` environment
marker. Processes that left the harness before the stop remain for Resources to stop after verification or episode
failure. This includes unfinished background jobs such as `make &` left by an
earlier tool call, as well as services; either can continue writing during verification.

To survive this forced-stop cleanup, a service must be outside the harness group and no
longer descend from the harness when the stop begins. In the usual tool-launch path, let the
process that launched it exit before the harness deadline or stop. A service still parented
by an in-flight tool descends from the harness at the stop and is killed with the in-flight
tools.

Natural CLI exit has no such cutoff: unfinished tools in separate sessions cannot be
distinguished from services and may still change task files. PID reuse and a tool
that starts a new session after the stop is recorded also limit this rule.
A confirmed receipt is not a guarantee that the sandbox is idle or a security boundary.
Unconfirmed termination blocks verification; a final failure receipt still lets agent
close release its connection for owner teardown.

Tool shells retain the task HOME and XDG directories, so home-directory deliverables
remain available to verification. Harness configuration and transcripts stay in a private
session directory that agent close removes; services must not depend on those files.

Services must also outlive the harness's shell tool. `nohup ... &` alone is not
portable across tools: a shell deadline or process-group cleanup can stop the
service. Detach its process group and standard streams, for example with
`subprocess.Popen(start_new_session=True)`, stdin from `DEVNULL`, and stdout/stderr
in task-owned log files. Check that the service is reachable before finishing.
A file-producing smoke does not validate this service-dependent behavior.

Resources retains completed close receipts for `session_close_retry_window_seconds`
(default 300 seconds); retries do not extend the window. Pending setup or failed stops
retain their handles. Close fences an in-progress setup before waiting for its lock,
so setup finishing after a close timeout attempts its own bounded teardown. If that
attempt fails after the final close callback is cancelled, there is no durable automatic
retry. The server retains the fence and handle for an explicit close retry or shutdown;
provider TTL can stop the abandoned sandbox but does not purge the in-memory record.
Shutdown attempts session stops concurrently, but process termination can interrupt them.
