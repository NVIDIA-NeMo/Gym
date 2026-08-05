# Legal Agent Bench

This resource server runs the public Harvey
[Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs/tree/f46ef86e4788545622db25dcffa3aebb7a139929)
through NeMo Gym. The benchmark default is a direct Gym-native implementation
of LAB's model/tool loop. Compatibility configurations run Harbor or Gym's
Hermes, Claude Code, and Codex harnesses through the same LAB-owned sandbox
runner. The integration is pinned to upstream commit
`f46ef86e4788545622db25dcffa3aebb7a139929`: 1,749 tasks and the public
`docx`, `pptx`, and `xlsx` skills.

NeMo Gym schedules rollouts, the selected agent works with each task's
documents inside a sandbox, and the task-local verifier scores every rubric
criterion with an OpenAI-compatible judge model. `--benchmark legal_agent_bench`
selects the native loop; Harbor is available explicitly as
`legal_agent_bench/config_harbor`.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker with a running daemon for local execution and portable runtime
  provisioning. ECS Fargate execution is also supported with an immutable
  registry image; see the benchmark README.
- An OpenAI-compatible policy endpoint and judge endpoint
- At least 10 GB of free working space for preparation and the first Docker build

Not required:

- Separate Harbor, Hermes, Claude Code, or Codex installations
- Anthropic or OpenAI vendor subscriptions or CLI logins for the Claude Code
  and Codex harnesses

Gym provisions the pinned harness dependencies automatically. Every harness
uses the configured policy model endpoint. Access to the configured policy and
judge endpoints is still required and may itself be metered or paid.

The pinned source download is about 579 MiB; allow a few GiB of free working
space during preparation. The first task also builds a
document-tooling Docker image and can take several minutes. Later tasks reuse
Docker layers.

From a fresh clone, create the repository environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
docker info >/dev/null
```

Keep this environment activated when running LAB commands. Invoke `gym`
directly rather than `uv run gym` for workflows that start servers: Ray starts
components from their own working directories, which can conflict with uv's
project discovery.

Create a `env.yaml` file in the root `Gym/` directory and add your endpoint settings (note: `env.yaml` is .gitignored):

```yaml
policy_base_url: https://your-policy-endpoint.example/v1
policy_api_key: your-policy-key
policy_model_name: your-policy-model

judge_base_url: https://your-judge-endpoint.example/v1
judge_api_key: your-judge-key
judge_model_name: your-judge-model
```

The judge credentials are injected only into the regenerated, gitignored
runtime task tree. They are never written into the cache. The configurable
runner does not mount rubric files or pass judge credentials during the agent
phase; both are staged only after the agent exits.

## Prepare explicitly (recommended)

From the repository root:

```bash
python resources_servers/legal_agent_bench/prepare.py
```

The command downloads the pinned LAB source archive from GitHub with retries and visible
progress, verifies SHA-256
`e45cbdf3236b22866e034bcc62fb23bf00ef2f2e49db7a0cd8a4b07dbae9212c`,
rejects unsafe archive entries, generates deterministic runtime tasks, and
builds each cache in staging before replacing the previous valid cache. A
handled preparation failure leaves the previous cache in place.

Useful options:

```bash
python resources_servers/legal_agent_bench/prepare.py --asset tasks
python resources_servers/legal_agent_bench/prepare.py --asset skills
python resources_servers/legal_agent_bench/prepare.py --force
python resources_servers/legal_agent_bench/prepare.py \
  --tasks-dir /custom/task-cache \
  --skills-dir /custom/skills-cache
```

## Collate datasets

The five example tasks are included in the repo and can be collated without downloading the full LAB archive:

```bash
gym dataset collate \
  --resources-server legal_agent_bench \
  --model-type vllm_model \
  --output-dir results/legal_agent_bench_prepare \
  --mode example_validation
```

Preparation using `prepare.py` generates the full 1,749-row task index inside the task cache. Prepare the assets before collating the full validation dataset:

```bash
python resources_servers/legal_agent_bench/prepare.py

gym dataset collate \
  --resources-server legal_agent_bench \
  --model-type vllm_model \
  --output-dir results/legal_agent_bench_prepare \
  --mode train_preparation
```

If you skip explicit preparation, `gym env start` prepares the missing task index, task mirrors, and skill cache during startup. Full validation collation still requires the generated index.

## Test the environment

Run the resource-server tests:

```bash
gym env test --resources-server legal_agent_bench
```

## Run and smoke test

For the shortest copy-paste path that prepares and tests the native loop,
Harbor, Hermes, Claude Code, and Codex as benchmark variants, use the
[benchmark README](../../benchmarks/legal_agent_bench/README.md#test-the-various-harnesses).
The commands below describe the lower-level resource-server and standalone
agent workflow.

The resource-server discovery config predates the benchmark wrapper and starts
the Harbor compatibility agent:

```bash
gym env start \
  --resources-server legal_agent_bench \
  --model-type vllm_model
```

On a clean cache, startup visibly downloads and prepares the pinned assets.
Every startup regenerates the runtime task tree so old judge credentials cannot
be reused.

In a second activated terminal, collect one rollout:

```bash
gym eval run --no-serve \
  --agent legal_agent_bench_harbor_agent \
  --input resources_servers/legal_agent_bench/data/example.jsonl \
  --output results/legal_agent_bench_smoke_rollout.jsonl \
  --concurrency 1 \
  --limit 1
```

The generated and example JSONL files do not contain `agent_ref`. Direct runs
must therefore specify the desired agent. The available standalone
configurations and agent names are:

| Config path | Direct-run agent |
| --- | --- |
| `resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml` | `legal_agent_bench_harbor_agent` |
| `responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_native.yaml` | `legal_agent_bench_native_agent` |
| `responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_hermes.yaml` | `legal_agent_bench_hermes_agent` |
| `responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_claude_code.yaml` | `legal_agent_bench_claude_code_agent` |
| `responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_codex.yaml` | `legal_agent_bench_codex_agent` |

The configurable files include
`resources_servers/legal_agent_bench/configs/resources_only.yaml`, which starts
the LAB resource server without also launching the compatibility Harbor agent.

For example, start and run the Gym-native loop with:

```bash
gym env start \
  --config responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_native.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --config responses_api_agents/legal_agent_bench_agent/configs/legal_agent_bench_native.yaml \
  --agent legal_agent_bench_native_agent \
  --input resources_servers/legal_agent_bench/data/example.jsonl \
  --output results/legal_agent_bench_native_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Use the corresponding config and agent name for Hermes, Claude Code, or Codex.
All four non-Harbor choices use the configured Gym policy endpoint. That
endpoint must support the selected harness's protocol. Their runtime
dependencies are provisioned into a portable cache on the first rollout; CLI
dependencies are pinned where applicable.

The default `full_task` reward is LAB's official all-criteria score: a task
earns `1.0` only when every criterion passes. For diagnostic partial credit,
start with:

```bash
gym env start \
  --resources-server legal_agent_bench \
  --model-type vllm_model \
  +legal_agent_bench.resources_servers.legal_agent_bench.reward_mode=criteria_pass_rate
```

The verifier evaluates up to six criteria concurrently by default.
Adjust the resource setting when the judge endpoint has a lower
concurrency limit:

```bash
gym env start \
  --resources-server legal_agent_bench \
  --model-type vllm_model \
  +legal_agent_bench.resources_servers.legal_agent_bench.judge_parallelism=2
```

This setting is forwarded to each task verifier as
`LAB_JUDGE_PARALLELISM`.

The task container requires network access because the policy agent and
verifier call configured endpoints.

## Caches and outputs

The default paths are:

- Generated tasks: `data/cache/harbor_tasks/legal_agent_bench`
- Public skills: `data/cache/harness/skills`
- Credential-bearing runtime tasks: `data/runtime/harbor_tasks/legal_agent_bench`
- Harbor jobs: `results/legal_agent_bench/harbor_jobs`
- Configurable agent runtimes: `responses_api_agents/legal_agent_bench_agent/.deps`
- Gym-native rollout artifacts: `results/legal_agent_bench/native_jobs`
- Hermes rollout artifacts: `results/legal_agent_bench/hermes_jobs`
- Claude Code rollout artifacts: `results/legal_agent_bench/claude_code_jobs`
- Codex rollout artifacts: `results/legal_agent_bench/codex_jobs`
- Rollout output: the path passed to `gym eval run`

The runtime tree hardlinks immutable documents from the cache when the
filesystem permits it, avoiding a second copy of the large document corpus.
Set `auto_prepare_assets: false` to require a prepopulated valid cache and avoid
network access at startup.

Each successful Harbor trial contains `result.json`, `verifier/reward.json`,
`verifier/scores.json`, `verifier/transcript.jsonl`, `agent/trajectory.json`,
and `agent/artifacts/lab-run/transcript.jsonl`. The agent config artifact
should list exactly `docx`, `pptx`, and `xlsx`.

Each configurable trial contains the inner Gym trajectory, agent stdout and
stderr, LAB `config.json` and `metrics.json`, completed deliverables under
`agent/artifacts/lab-run/output`, downloaded verifier artifacts, and a compact
top-level `run_summary.json`. The rollout JSONL row exposes direct paths to the
summary, trajectory, stdout, stderr, output directory, and verifier report.
Trials are grouped as
`<harness>_jobs/<model>/<YYYYMMDD-HHMMSS_hash>/<task_name>_<run_id>` so runs
are browsable and safe under concurrent execution. `<model>` is the configured
`policy_model_name`, normalized into one safe path segment. The task name is
normalized, and the run ID is an eight-character unique suffix. Starting or
validating an agent server does not create an empty session directory; the
directory is created by its first rollout.
The Docker image is content-addressed from the task environment and reused
across harnesses. Agent and verifier phases use separate containers: the agent
sees only its selected Gym package and public task inputs, while the verifier
starts after that container is destroyed and receives the completed LAB run
read-only.

To inspect a native or configurable smoke run:

```bash
ARTIFACT_DIR=$(jq -r '.artifact_dir' results/legal_agent_bench_native_smoke.jsonl)
jq . "$ARTIFACT_DIR/run_summary.json"
jq . "$ARTIFACT_DIR/agent/trajectory.json"
open "$ARTIFACT_DIR/verifier/report.html"  # macOS; use xdg-open on Linux
```

In `run_summary.json`, a reliable scored rollout has `mask_sample: false`, all
failure flags false, `judge_error_count: 0`, and `verifier_error: 0`. This
includes incomplete max-turn or context-limit outcomes: their partial output is
still judged and saved in the main rollout JSONL.
`output_files` lists the deliverables and `agent/trajectory.json` is the inner
harness trace. A zero `full_task` reward can still be a valid rollout when one
or more criteria fail; use `criteria_pass_rate` to see partial success.

## Troubleshooting

- A missing judge setting produces a verifier error in the trial artifacts.
  Confirm the endpoint permits the exact `judge_model_name`.
- Treat a nonzero `judge_error_count` or `verifier_error` as a judge or
  infrastructure failure, not an ordinary model failure, even though Harbor
  receives a numeric zero reward so it can preserve a complete trial result.
- Configurable rollouts additionally expose `agent_failed`,
  `model_connection_failed`, `agent_timed_out`, `verifier_failed`,
  `verifier_timed_out`, `sandbox_failed`, `task_failed`,
  `configuration_failed`, and `mask_sample`. Before the harness starts, the
  runner checks the policy endpoint from inside Docker. On Docker Desktop,
  derived loopback model URLs are automatically routed through
  `host.docker.internal`; `sandbox_model_base_url` is an explicit override.
  Connectivity, harness, sandbox, and verifier failures are masked, skip
  judging when applicable, and carry `_ng_failure_class`, which routes them to
  the failure sidecar for bounded retry. Task-loading and harness-configuration
  failures additionally carry `_ng_failure_terminal: true`, so they are not
  retried. `mask_sample` is a training hint, not the routing signal. A zero
  reward without an infrastructure/judge flag is an ordinary model/task result;
  a flagged result should be excluded from model-quality comparisons.
- Harbor agent, adapter, connection, and timeout failures preserve any partial
  trajectory but skip judging. Their result rows report `mask_sample`,
  `agent_failed`, `model_connection_failed`, `agent_timed_out`, and
  `failure_reason`, carry `_ng_failure_class`, and have their reward forced to
  zero. Harbor context-limit and max-turn stops are instead judged as valid
  incomplete outcomes.
- Hermes normally probes `/v1/models` and `/models` for optional pricing and
  context metadata. The LAB runner disables those lookups because Gym supplies
  the model explicitly and its internal policy proxy does not implement model
  discovery. This does not suppress access logs for actual chat-completion
  requests.
- If Docker appears idle on the first rollout, inspect `docker ps` and the
  `gym env start` terminal; Harbor is normally building the task image.
- Do not copy or publish `data/runtime/`: it can contain local judge credentials.
- Results are revision-specific and should not be compared directly with runs
  that use a different task snapshot or skill set.

## Licensing and source modifications

Harvey LAB is MIT-licensed. Its license and the details of the minimal modified
runtime are in `vendor/harvey_labs/`. Task documents and public skills are
downloaded from the pinned public repository and are never tracked here.
