# Legal Agent Bench Benchmark

This benchmark registers the
[Legal Agent Bench resource server](../../resources_servers/legal_agent_bench/README.md)
with Gym's benchmark catalog. Every variant evaluates the same 1,749 public
Harvey LAB tasks with the same skills and verifier.

This benchmark uses the Harbor integration by default. Three additional variants use the LAB-owned
[configurable runner](../../responses_api_agents/legal_agent_bench_agent/README.md)
with Gym's built-in agent harnesses:

| Benchmark | Agent |
| --- | --- |
| `legal_agent_bench` | Harbor (default) |
| `legal_agent_bench/config_hermes` | Hermes |
| `legal_agent_bench/config_claude_code` | Claude Code |
| `legal_agent_bench/config_codex` | Codex |

Benchmark preparation reuses the resource server's pinned task and skill
caches and copies a small, gitignored, deterministic index. Source, example,
and benchmark JSONL rows are agent-neutral. Gym stamps the variant's configured
agent during dataset collation.

## Requirements

- Python 3.12 and the repository environment installed with `uv`
- Docker with a running daemon
- Authorized OpenAI-compatible policy and judge endpoints in the root
  `env.yaml`
- At least 10 GB of free working space

Not required:

- Separate Harbor, Hermes, Claude Code, or Codex installations
- Anthropic or OpenAI vendor subscriptions or CLI logins for the Claude Code
  and Codex harnesses

Gym provisions the pinned harness dependencies automatically. Every harness
uses the configured policy model endpoint. Access to your configured policy and
judge endpoints is still required and may itself be metered or paid.

See the [resource-server README](../../resources_servers/legal_agent_bench/README.md)
for endpoint configuration, source and license details, cache locations, and
troubleshooting. The initial source download is several hundred MiB, and the
first rollout builds a document-tooling Docker image that can take several
minutes.

## Set up

From the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
docker info >/dev/null
```

Configure the policy and judge endpoints in the gitignored root `env.yaml` as
shown in the
[resource-server README](../../resources_servers/legal_agent_bench/README.md#requirements).
Keep the virtual environment activated for every command below.

Use the `gym` executable from the activated environment directly. Do not prefix
these server-starting commands with `uv run`: Ray starts components from their
own working directories, which can conflict with uv's project discovery.

## Prepare and validate

From the repository root, run:

```bash
gym eval prepare --benchmark legal_agent_bench
```

Preparation is shared by all four variants. It validates or prepares the task
and skill caches, then writes the deterministic benchmark index to
`benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl`.
Repeated preparation reuses valid caches and does not download a second copy of
LAB.

Validate that all four configurations resolve before spending time on a
rollout:

```bash
gym env validate --model-type vllm_model --benchmark legal_agent_bench
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_hermes
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_claude_code
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_codex
```

## Test the various harnesses

Run these one at a time for each harness you want to test. Each command starts the required Gym services, runs
the first benchmark task, writes one JSONL result row, and stops the services.
Alongside the path passed to `--output`, Gym writes
`<output_stem>_materialized_inputs.jsonl` and
`<output_stem>_aggregate_metrics.json`.

Harbor:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench \
  --split benchmark \
  --output results/legal_agent_bench_harbor_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Hermes:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_hermes \
  --split benchmark \
  --output results/legal_agent_bench_hermes_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Claude Code:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_claude_code \
  --split benchmark \
  --output results/legal_agent_bench_claude_code_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Codex:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_codex \
  --split benchmark \
  --output results/legal_agent_bench_codex_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Hermes uses chat completions. Claude Code and Codex use their respective CLI
adapters against the configured OpenAI-compatible policy endpoint. The first
run of each configurable harness provisions its pinned portable runtime, so it
can be much slower than a cached run. The LAB runner disables Hermes's optional
pricing and context-metadata lookups because Gym already supplies the model and
its internal policy proxy does not expose `/models`; real model-call access
logging is enabled.

## Check a configurable result

For Hermes, for example, inspect the result row and the persisted artifact bundle with:

```bash
jq '{
  reward,
  criteria_pass_rate,
  mask_sample,
  agent_failed,
  model_connection_failed,
  verifier_failed,
  judge_error_count,
  verifier_error,
  artifact_dir
}' results/legal_agent_bench_hermes_smoke.jsonl

ARTIFACT_DIR=$(jq -r '.artifact_dir' results/legal_agent_bench_hermes_smoke.jsonl)
jq . "$ARTIFACT_DIR/run_summary.json"
jq . "$ARTIFACT_DIR/agent/trajectory.json"
open "$ARTIFACT_DIR/verifier/report.html"  # macOS; use xdg-open on Linux
```

Substitute the Claude Code or Codex output filename to inspect those runs.
A reliable completed rollout has `mask_sample: false`, no failure flags,
`judge_error_count: 0`, and `verifier_error: 0`. Check `output_files` in
`run_summary.json` for the deliverables. A `reward` of `0.0` can still be a
valid run: the default `full_task` reward requires every rubric criterion to
pass, while `criteria_pass_rate` shows partial success.

Harbor keeps its detailed trial files under
`results/legal_agent_bench/harbor_jobs`; configurable variants use the
corresponding `hermes_jobs`, `claude_code_jobs`, or `codex_jobs` directory.
Their model directory comes from `policy_model_name`, with path separators and
other unsafe characters normalized for the filesystem. A session directory is
created only when its first rollout starts. Each configurable rollout is stored
under
`<harness>_jobs/<model>/<YYYYMMDD-HHMMSS_hash>/<task_name>_<run_id>/`, where
the task name is normalized and the run ID is an eight-character unique suffix.
Harbor's result row uses the established Harbor response format rather than the
configurable runner's top-level flags.

## Run a larger evaluation

Remove `--limit 1` from the desired smoke command. Choose a new output filename
and increase `--concurrency` only after confirming that Docker, the policy
endpoint, and the judge endpoint can sustain it.

## Manage servers separately

To manage the servers separately, start the desired variant first:

```bash
gym env start \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_hermes
```

Then run against them with `--no-serve` from a second activated terminal:

```bash
gym eval run --no-serve \
  --benchmark legal_agent_bench/config_hermes \
  --agent legal_agent_bench_benchmark_hermes_agent \
  --input benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl \
  --output results/legal_agent_bench_hermes_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

## Scoring

The default `full_task` reward is LAB's official all-criteria score: a task
receives `1.0` only when every criterion passes. To use diagnostic criterion
pass rate instead, add this override to `gym env start` or the one-shot
`gym eval run` command:

```bash
+legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench.reward_mode=criteria_pass_rate
```

This changes only the reported reward; it does not change the tasks, agent, or
judge criteria.

For configurable variants, the rollout also reports `agent_failed`,
`model_connection_failed`, `agent_timed_out`, `verifier_failed`,
`verifier_timed_out`, `sandbox_failed`, `task_failed`,
`configuration_failed`, `judge_error_count`, and `verifier_error`.
`task_failed` identifies an unsafe, unknown, incomplete, or malformed task;
`configuration_failed` identifies an invalid harness selection or missing
required pin. The runner checks policy-model connectivity from inside the
Docker sandbox before starting the harness. A connectivity or harness failure
is masked and is not sent to the judge. A normal model/task result can still
receive zero reward without those flags. Infrastructure, configuration,
task-loading, or judge failures set `mask_sample`; do not treat those zeroes as
model-quality results.

Configurable-runner artifacts are grouped by harness, model, dated session, and
task. The default roots are `results/legal_agent_bench/hermes_jobs`,
`results/legal_agent_bench/claude_code_jobs`, and
`results/legal_agent_bench/codex_jobs`. Harbor continues to use
`results/legal_agent_bench/harbor_jobs`.

Each configurable output row includes `artifact_dir`, `run_summary_path`,
`agent_trace_path`, `agent_stdout_path`, `agent_stderr_path`,
`verifier_report_path`, and `output_dir`. The inspection commands above use
these paths.

The agent-server log also prints the artifact directory when the rollout is
created and again when it completes or fails.

## Test

Run the benchmark and resource-server tests with:

```bash
uv run pytest -q \
  benchmarks/legal_agent_bench/tests \
  resources_servers/legal_agent_bench/tests \
  responses_api_agents/legal_agent_bench_agent/tests
```

Generated indexes, collation metrics, Harbor jobs, source documents, and skills
must not be committed.
