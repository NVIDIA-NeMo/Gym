# Terminal-Bench 2.1

Terminal-Bench 2.1 evaluates agents on 89 terminal tasks, each in its own container.

| Preset | Harness | Runs on |
| --- | --- | --- |
| `terminal_bench_2_1/terminus_2` | Harbor's Terminus 2 (`terminus_2_agent` agent sessions) | Harbor environment server |
| `terminal_bench_2_1/opencode` | OpenCode (`opencode_agent`) | Harbor environment server |
| `terminal_bench_2_1/openclaw` | OpenClaw 2026.6.11 (`openclaw_agent`) | Harbor environment server |
| `terminal_bench_2_1/nooa` | NOOA | `single_agent_turn` on the `terminal_bench_2_1` resources server |
| `terminal_bench_2_1/inkling_small` | Terminus 2, pinned rows with a terminal-interaction note | the resources server's `legacy_agent` relay |
| `terminal_bench_2_1/terminus_2_legacy_agent`, `terminal_bench_2_1/opencode_legacy_agent` | the previous `terminus_2` and `opencode` recipes (deprecated) | the resources server's `legacy_agent` relay |

## Run

`terminal_bench_2_1/terminus_2`, `terminal_bench_2_1/opencode` and `terminal_bench_2_1/openclaw` run the published task folders of
[`environments/harbor/terminal_bench_2_1`](../../environments/harbor/terminal_bench_2_1/README.md), pinned by content
hash, through the [`harbor` environment server](../../environment_servers/harbor/README.md). The server starts each
task's prebuilt image as the sandbox, lends it to the harness's agent session, and then grades the episode with the
task's own `tests/test.sh`; resources and time budgets come from each task's `task.toml`. Each preset takes eight
rollouts per task. `terminal_bench_2_1/openclaw` keeps OpenClaw 2026.6.11: the `exec` tool of the adapter's default
2026.11.5 needs glibc 2.33, and the `qemu-alpine-ssh` and `qemu-startup` images ship glibc 2.31, so session setup
would refuse those two tasks.

```bash
gym dataset pull terminal_bench_2_1
gym eval run --benchmark terminal_bench_2_1/terminus_2 --sandbox opensandbox --model-type vllm_model \
  --only-tasks 'fix-git,regex-log' --num-repeats 1
```

- `--only-tasks` and `--exclude-tasks` select task folders; drop them, and `--num-repeats`, for the full run.
- `--agent-type <harness>` runs another session-capable harness (for example `opencode_agent`, `pi_agent`,
  `codex_agent` or `mini_swe_agent/mini_swe_agent_sandboxed`) on the same tasks; it uses that harness's shipped
  settings. For OpenClaw, use `terminal_bench_2_1/openclaw`: `--agent-type openclaw_agent` runs the shipped 2026.11.5,
  which refuses the two glibc 2.31 tasks.
- `gym dataset validate terminal_bench_2_1 --sandbox opensandbox` runs each task's reference solution.
- These presets need no `gym eval prepare`, and refuse `--no-serve` and `gym eval plan`. Results land where
  `gym eval run` writes them for a Harbor target; the
  [Terminal-Bench 2.1 Harbor tutorial](https://docs.nvidia.com/nemo/gym/main/tutorials/evaluation-tutorials/terminal-bench-2-1-harbor)
  describes the files, the deployment overlay that reproduces the resources server's fixed sandbox size, and the two
  tasks whose published images fail their own tests (`qemu-startup`, `qemu-alpine-ssh`).

### Scoring

Each harness gets the task's agent budget (`[agent].timeout_sec`, capped by its own 10,800-second limit). When the
budget runs out, the harness stops and the task's `tests/test.sh` grades the sandbox as the agent left it. Once
`test.sh` has run, the row counts: a missing reward file scores 0. Infrastructure endings are retried or masked,
never scored 0: a model-server or transport failure the harness reports, a sandbox or provider failure, a lost
sandbox, and an episode that outlives its task's total budget. See the
[`harbor` environment server](../../environment_servers/harbor/README.md#what-it-does).

### Tasks the resources server repairs

The Harbor path runs every task as published; the resources server repairs eleven of them at run time (see
"Differences" below). Their reference solutions on the Harbor path
(`gym dataset validate terminal_bench_2_1 --sandbox opensandbox --only-tasks ...`):

| Task | Repair on the resources-server path | Reference solution on the Harbor path |
| --- | --- | --- |
| `qemu-startup`, `qemu-alpine-ssh` | Debian Bullseye security snapshot | 0: `tests/test.sh` installs `curl` from the Bullseye security mirror, which answers 404 |
| `compile-compcert`, `extract-moves-from-video` | `ca-certificates` in `tests/test.sh` | 1 |
| `pytorch-model-recovery`, `torch-tensor-parallelism`, `torch-pipeline-parallelism`, `mteb-retrieve` | PyTorch CPU wheel index in `tests/test.sh` | 1; the verifier downloads the CUDA build of `torch` and took 52 to 102 seconds of its 900 or 1,800 |
| `mcmc-sampling-stan`, `build-cython-ext` | `solution/solve.sh` (golden-patch mode only) | 0: the published reference solution fails its tests; model rollouts are unaffected |
| `build-pov-ray` | `solution/solve.sh` (golden-patch mode only) | 1 |

`qemu-startup` and `qemu-alpine-ssh` score 0 on every rollout, as they do in Harbor itself: 2 of 89 tasks. When you
compare with a source that repairs them, exclude them on both sides (`--exclude-tasks qemu-startup,qemu-alpine-ssh`)
and say so. The `ca-certificates` repairs matter only for an agent that leaves `curl` without its CA bundle, which a
reference solution does not show; such a rollout can score 0 on the Harbor path.

### Differences from the resources-server path

The `*_legacy_agent` recipes and the sections below run the same tasks through the `terminal_bench_2_1` resources
server. Scores on the two paths are not interchangeable:

- **Tasks.** The Harbor entry is Harbor's package-store copy; the resources server prepares rows from a clone of
  `harbor-framework/terminal-bench-2-1` at `7131e437`. The two differ in one file,
  `sanitize-git-repo/tests/test_outputs.py`.
- **Repairs.** The resources server edits eight tasks' `tests/test.sh` (or their image's package sources) and, in
  golden-patch mode, three reference solutions at run time; the Harbor path runs them as published. The table above
  lists them and what each does on the Harbor path.
- **Sandbox size and budgets.** The resources server gives every task 4 CPUs, 16 GiB and 30 GiB of disk and the
  harness's own run limit (10,800 seconds for both harnesses); the Harbor path uses each task's declared resources
  and agent and verifier timeouts (600 to 12,000 seconds of agent time, median 900). On a slow or shared endpoint
  more episodes stop at the task's budget on the Harbor path, which lowers its score against the resources server's.
  `rstan-to-pystan` declares 8 GiB, and its sandbox can be OOM-killed there (a masked or failed sample, not a 0). The
  [Harbor tutorial](https://docs.nvidia.com/nemo/gym/main/tutorials/evaluation-tutorials/terminal-bench-2-1-harbor)
  shows the overlay that gives the Harbor path the resources server's sandbox size.
- **OpenCode harness.** `opencode` runs `opencode_agent` in place of `opencode_sandboxed_agent`; the two adapters ship
  different defaults, so treat their scores as separate measurements. Both deny `webfetch`, `websearch` and `skill`
  and keep OpenCode subagents (`task`) enabled. Both bound the whole run at 10,800 seconds; `opencode_sandboxed_agent`
  also caps each model call at one hour, and `opencode_agent`'s context window is metadata, not an output cap.
  `opencode_sandboxed_agent` captures token IDs and requests `max_tokens` equal to its 262,144-token
  window on every call, so an endpoint that serves exactly that window refuses every first call
  (`context_length_exceeded`) and every rollout scores 0 without a recorded failure. Drop the per-call caps, as
  [`policy_model_override.yaml`](../nemotron_3.5_super/policy_model_override.yaml) does, when you run
  `opencode_legacy_agent` against such an endpoint.
- **Terminus 2 harness.** `terminus_2` runs `terminus_2_agent` (Harbor's Terminus 2 at the Harbor commit its
  `pyproject.toml` pins) in place of `terminus_2_sandboxed_agent` (Harbor 0.22.0); treat their scores as separate
  measurements. `terminus_2_agent` samples at temperature 0.7 and caps each model call at 2,400 seconds;
  `terminus_2_sandboxed_agent` leaves the temperature to the model server and caps each call at 600 seconds. Both
  summarize proactively at 8,000 tokens and bound the whole run at 10,800 seconds; on Harbor the run limit is the
  task's agent budget.

### From the resources-server commands

`terminal_bench_2_1/terminus_2` and `terminal_bench_2_1/opencode` keep their names but no longer read the resources
server's keys, and `++` keys merge without an error, so map them:

- `gym eval prepare --benchmark ...`: `gym dataset pull terminal_bench_2_1` fetches the task folders.
- Row selection in the prepared `benchmark.jsonl` or `--input`: `--only-tasks` and `--exclude-tasks`.
- The dataset's `num_repeats`: the preset's eight, or `--num-repeats`.
- `terminal_bench_2_1_resources_server...sandbox_config.resources` and `.provider_options.resource_requests`:
  `sandbox_resources_override` and `sandbox_provider_options` under
  `harbor_environment_server.environment_servers.harbor` (the tutorial's overlay).
- `...sandbox_config.metadata` (deployment labels such as those in
  [`sandbox_utils.yaml`](../nemotron_3.5_super/sandbox_utils.yaml)) and `ready_timeout_s` (1,200 seconds there):
  `sandbox_metadata` and `sandbox_ready_timeout_s` (default 900) in the same block.
- `++use_absolute_ip=true`: not needed; the harness in the sandbox calls Gym's model server at the address the run's
  servers listen on, so set `default_host` to one the sandboxes can reach.
- `gym env start` with `gym eval run --no-serve`, and `gym eval plan`: not available for a Harbor-backed preset.
- `gym eval submit` and launchers built on it: not yet. A submit benchmark passes only `+key=value` overrides to
  `gym eval prepare` and `gym eval run`, so it cannot carry `--sandbox`; keep those configs on the `_legacy_agent`
  names until submit can run a Harbor-backed preset.

The old commands still run under `terminal_bench_2_1/terminus_2_legacy_agent` and `opencode_legacy_agent`.

## The resources-server path (legacy)

The `terminal_bench_2_1` resources server is being retired in favor of the Harbor path above. It still backs
`nooa`, `inkling_small` and the `*_legacy_agent` recipes. On it, the benchmark prepares task data and owns sandbox
setup, verification, and cleanup. Pi, Codex, OpenCode, OpenClaw, Copilot, and CrewAI can run their model/tool loops
in the task sandbox through Gym's Environment Server.

- Integration profile: `custom-gym-verifier`
- Scorer: `terminal_bench_2_1`

### Prepare data

Run from the Gym checkout with the benchmark's preparation dependencies installed:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 \
  benchmarks/terminal_bench_2_1/terminal-bench-2-1
git -C benchmarks/terminal_bench_2_1/terminal-bench-2-1 \
  checkout --detach 7131e4375048a0e408a8fb404b5f499d726b695b
python benchmarks/terminal_bench_2_1/prepare.py
```

`prepare()` also takes `task_revision`, a full commit of the task repository. It checks that commit out, cloning
first if the checkout is missing, so two runs read the same tasks. Through the CLI, pass it as
`++prepare_script_args.task_revision=<commit>` at the top level; in a YAML config it is the top-level
`prepare_script_args: {task_revision: <commit>}`. Without it the checkout is left as it is, and a fresh
clone reads the repository's default branch.

Keep task tests and solutions on the host; do not mount the benchmark checkout
into task containers. Resources uploads verifier assets after the agent closes
and rejects tasks whose local `tests/test.sh` is missing. Do not enable
golden-patch mode for model runs.

### Configure a run

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
Both servers accept several workers, because session routing sends each
session's requests to the worker that holds it.

#### Choose a harness

To use another harness, replace its config import, the entire `codex_agent`
settings block, and `agent_server.name`. The settings in the last column belong
under `<name>.responses_api_agents.<name>`. The Environment Server's binding selects
the harness for this taskset.

| Harness requirements | Config import | Name | Example settings |
| --- | --- | --- | --- |
| [Pi](../../responses_api_agents/pi_agent/README.md) | `responses_api_agents/pi_agent/configs/pi_agent.yaml` | `pi_agent` | `thinking: "off"`, `max_output_tokens: 8192`, `timeout: 600` |
| [Codex](../../responses_api_agents/codex_agent/README.md) | `responses_api_agents/codex_agent/configs/codex_agent.yaml` | `codex_agent` | `model_context_window: 32768`, `model_auto_compact_token_limit: 28672`, `timeout: 600` |
| [OpenCode](../../responses_api_agents/opencode_agent/README.md) | `responses_api_agents/opencode_agent/configs/opencode_agent.yaml` | `opencode_agent` | `context_window: 32768`, `max_output_tokens: 8192`, `timeout: 600` |
| [OpenClaw](../../responses_api_agents/openclaw_agent/README.md) | `responses_api_agents/openclaw_agent/configs/openclaw_agent.yaml` | `openclaw_agent` | `openclaw_version: 2026.6.11` (2026.11.5 refuses the two glibc 2.31 task images), `context_window: 32768`, `model_timeout_seconds: 600`, `timeout: 600` |
| [Copilot](../../responses_api_agents/copilot_agent/README.md) | `responses_api_agents/copilot_agent/configs/copilot_agent.yaml` | `copilot_agent` | `context_window: 32768`, `max_output_tokens: 8192`, `timeout: 600` |
| [CrewAI](../../responses_api_agents/crewai_agent/README.md) (development pairing) | `responses_api_agents/crewai_agent/configs/crewai_agent.yaml` | `crewai_agent` | `context_window_size: 131072`, `shell_timeout_seconds: 300`, `timeout: 900` |

Pi, Codex, OpenCode, and OpenClaw are allowed by this benchmark. Copilot and CrewAI remain
unsupported development pairings pending full benchmark qualification; add
`--allow-unsupported-pairing` to both `gym env start` and `gym eval run` when selecting either.
These are single-task smoke settings; adjust context windows and budgets for your model and evaluation.
See each harness's README for runtime prerequisites and model requirements.

#### Model and sandbox provider

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
on the model server; the Pi/OpenCode/Copilot agent caps above also limit individual calls.
Codex/OpenClaw/Copilot sandbox sessions reject request-level `max_output_tokens`.

Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in your private
`env.yaml`. The task container must reach the Gym Model Server; for Docker on a
reachable host, use `++use_absolute_ip=true` as below.

### Run a smoke task

Select one prepared task:

```bash
mkdir -p results
python - <<'PY'
import json
from pathlib import Path

from nemo_gym.data import materialize_task

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

### Tasks that start services

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

The resources server fences each closed session for `session_close_retry_window_seconds`
(default 300 seconds); retries do not extend the window. Pending setup or failed stops
retain their handles. Close fences an in-progress setup before waiting for its lock,
so setup finishing after a close timeout attempts its own bounded teardown. If that
attempt fails after the final close callback is cancelled, there is no durable automatic
retry. The server retains the fence and handle for an explicit close retry or shutdown;
provider TTL can stop the abandoned sandbox but does not purge the in-memory record.
Shutdown attempts session stops concurrently, but process termination can interrupt them.
