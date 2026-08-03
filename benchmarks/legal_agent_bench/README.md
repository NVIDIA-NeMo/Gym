# Legal Agent Bench Benchmark

This benchmark registers the
[Legal Agent Bench resource server](../../resources_servers/legal_agent_bench/README.md)
with Gym's benchmark catalog. Every variant evaluates the same 1,749 public
Harvey LAB tasks with the same skills and verifier.

The unqualified benchmark uses a LAB-owned, Gym-native implementation of the
upstream LAB model/tool loop. Four explicit variants run the same tasks through
Harbor or one of Gym's built-in agent harnesses. All non-Harbor choices use the
[configurable runner](../../responses_api_agents/legal_agent_bench_agent/README.md).

| Benchmark | Agent |
| --- | --- |
| `legal_agent_bench` | LAB Gym-native loop (default) |
| `legal_agent_bench/config_harbor` | Harbor compatibility integration |
| `legal_agent_bench/config_hermes` | Hermes |
| `legal_agent_bench/config_claude_code` | Claude Code |
| `legal_agent_bench/config_codex` | Codex |

Benchmark preparation reuses the resource server's pinned task and skill
caches and copies a small, gitignored, deterministic index. Source, example,
and benchmark JSONL rows are agent-neutral. Gym stamps the variant's configured
agent during dataset collation.

## Requirements

- Python 3.12 and the repository environment installed with `uv`
- Docker with a running daemon for the default local backend and portable
  harness provisioning
- The `sandbox` dependency extra when using ECS Fargate
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

For a remote sandbox provider, also install its dependencies with:

```bash
uv sync --extra dev --extra sandbox
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

Preparation is shared by all five variants. It validates or prepares the task
and skill caches, then writes the deterministic benchmark index to
`benchmarks/legal_agent_bench/data/legal_agent_bench_benchmark.jsonl`.
Repeated preparation reuses valid caches and does not download a second copy of
LAB.

Validate that all five configurations resolve before spending time on a
rollout:

```bash
gym env validate --model-type vllm_model --benchmark legal_agent_bench
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_harbor
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_hermes
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_claude_code
gym env validate --model-type vllm_model --benchmark legal_agent_bench/config_codex
```

## Test the various harnesses

Run these one at a time for each harness you want to test. Each command starts
the required Gym services, runs the first benchmark task, writes one JSONL
result row, and stops the services.
Alongside the path passed to `--output`, Gym writes
`<output_stem>_materialized_inputs.jsonl` and
`<output_stem>_aggregate_metrics.json`.

Gym-native LAB loop:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench \
  --split benchmark \
  --output results/legal_agent_bench_native_smoke.jsonl \
  --concurrency 1 \
  --limit 1
```

Harbor compatibility integration:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_harbor \
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

The default native loop sends LAB's canonical function tools through Gym's
Responses API and executes them directly inside the task sandbox. Hermes uses
chat completions. Claude Code and Codex use their respective CLI adapters
against the configured OpenAI-compatible policy endpoint. The first run of
each non-Harbor harness provisions its portable runtime, so it can be much
slower than a cached run. The LAB runner disables Hermes's optional pricing and
context-metadata lookups because Gym already supplies the model and its
internal policy proxy does not expose `/models`; real model-call access logging
is enabled.

## Check a native or configurable result

For the default native loop, inspect the result row and persisted artifact
bundle with:

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
}' results/legal_agent_bench_native_smoke.jsonl

ARTIFACT_DIR=$(jq -r '.artifact_dir' results/legal_agent_bench_native_smoke.jsonl)
jq . "$ARTIFACT_DIR/run_summary.json"
jq . "$ARTIFACT_DIR/agent/trajectory.json"
open "$ARTIFACT_DIR/verifier/report.html"  # macOS; use xdg-open on Linux
```

Substitute the Hermes, Claude Code, or Codex output filename to inspect those
runs.
A reliable completed rollout has `mask_sample: false`, no failure flags,
`judge_error_count: 0`, and `verifier_error: 0`. Check `output_files` in
`run_summary.json` for the deliverables. A `reward` of `0.0` can still be a
valid run: the default `full_task` reward requires every rubric criterion to
pass, while `criteria_pass_rate` shows partial success.

Harbor keeps its detailed trial files under
`results/legal_agent_bench/harbor_jobs`; the native and configurable variants
use the corresponding `native_jobs`, `hermes_jobs`, `claude_code_jobs`, or
`codex_jobs` directory.
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
and increase client and agent-server concurrency together only after confirming
that the sandbox backend, policy endpoint, and judge endpoint can sustain it.
For example, a two-way Hermes run uses both settings:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench/config_hermes \
  --split benchmark \
  --output results/legal_agent_bench_hermes.jsonl \
  --concurrency 2 \
  +legal_agent_bench_benchmark_hermes_agent.responses_api_agents.legal_agent_bench_agent.concurrency=2
```

Use `legal_agent_bench_benchmark_native_agent` for the default native loop, or
the corresponding `legal_agent_bench_benchmark_claude_code_agent` or
`legal_agent_bench_benchmark_codex_agent` prefix for those variants. The
agent-server default stays at `1`; changing only `--concurrency` leaves the
server-side semaphore serial.

## Use ECS Fargate

Gym's native `ecs_fargate` provider can run the native, Hermes, Claude Code,
or Codex LAB variants without changing their agent loops. The reference ECS
infrastructure is discovered from `/<ssm_project>/ecs-sandbox/config` in SSM;
`ssm_project` defaults to `harbor`. Configure an AWS profile locally and export
the profile and region before starting Gym:

```bash
export AWS_PROFILE=gym-ecs
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION="$AWS_REGION"
```

The orchestrator must be able to reach the ECS task SSH sidecar on TCP port
`52222`. The SSM configuration must provide the cluster, subnets, security
groups, task roles, ECR repository, S3 staging bucket, and SSH-sidecar key
ARNs. Install the `sandbox` dependency extra as shown above.

Build the shared LAB environment for `linux/amd64`, push it to ECR, and use its
immutable digest. The prepared LAB tasks currently share this environment:

```bash
LAB_ENV_DIR=resources_servers/legal_agent_bench/data/runtime/harbor_tasks/legal_agent_bench/antitrust-competition__analyze-antitrust-hsr-strategy/environment
ECR_REPOSITORY=<account>.dkr.ecr.<region>.amazonaws.com/<repository>
ECR_REGISTRY=${ECR_REPOSITORY%%/*}
ECR_REPOSITORY_NAME=${ECR_REPOSITORY#*/}
LAB_IMAGE_TAG=legal-agent-bench-smoke

aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker buildx build \
  --platform linux/amd64 \
  --tag "$ECR_REPOSITORY:$LAB_IMAGE_TAG" \
  --push \
  "$LAB_ENV_DIR"
LAB_IMAGE_DIGEST=$(aws ecr describe-images \
  --region "$AWS_REGION" \
  --repository-name "$ECR_REPOSITORY_NAME" \
  --image-ids "imageTag=$LAB_IMAGE_TAG" \
  --query 'imageDetails[0].imageDigest' \
  --output text)
export LAB_ECS_IMAGE="$ECR_REPOSITORY@$LAB_IMAGE_DIGEST"
docker pull --platform linux/amd64 "$LAB_ECS_IMAGE"
```

Run one native smoke through ECS:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench \
  --config nemo_gym/sandbox/providers/ecs_fargate/configs/ecs_fargate.yaml \
  --split benchmark \
  --output results/legal_agent_bench_native_ecs_smoke.jsonl \
  --concurrency 1 \
  --limit 1 \
  +legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.sandbox_provider=sandbox \
  "+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.sandbox_image=${LAB_ECS_IMAGE}" \
  +legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.runtime_docker_platform=linux/amd64
```

ECS tunnels the rollout-scoped Gym policy-model URL into the agent task, so
model credentials remain in Gym and `sandbox_model_base_url` is unnecessary.
An explicit `sandbox_model_base_url` still bypasses the tunnel when the endpoint
is already reachable from the task. The verifier contacts the configured judge
endpoint directly, so that endpoint must be reachable from the Fargate VPC.
Each completed rollout creates one agent task followed by one verifier task;
both are stopped during cleanup. Artifacts use the same local paths documented
above.

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
judge criteria. The command above uses the native default's resource-server
prefix. For an explicit variant, replace
`legal_agent_bench_benchmark_resources_server` with the corresponding
`legal_agent_bench_benchmark_harbor_resources_server`,
`legal_agent_bench_benchmark_hermes_resources_server`,
`legal_agent_bench_benchmark_claude_code_resources_server`, or
`legal_agent_bench_benchmark_codex_resources_server` prefix.

For the native and configurable variants, the rollout also reports `agent_failed`,
`model_connection_failed`, `agent_timed_out`, `verifier_failed`,
`verifier_timed_out`, `sandbox_failed`, `task_failed`,
`configuration_failed`, `judge_error_count`, and `verifier_error`.
`task_failed` identifies an unsafe, unknown, incomplete, or malformed task;
`configuration_failed` identifies an invalid harness selection or missing
required pin. The runner checks policy-model connectivity from inside the
selected sandbox before starting the harness. A connectivity or harness failure
is masked and is not sent to the judge. A normal model/task result can still
receive zero reward without those flags. Infrastructure, configuration,
task-loading, or judge failures set `mask_sample`; do not treat those zeroes as
model-quality results.

Configurable-runner artifacts are grouped by harness, model, dated session, and
task. The default roots are `results/legal_agent_bench/native_jobs`,
`results/legal_agent_bench/hermes_jobs`,
`results/legal_agent_bench/claude_code_jobs`, and
`results/legal_agent_bench/codex_jobs`. Harbor uses
`results/legal_agent_bench/harbor_jobs`.

Each configurable output row includes `artifact_dir`, `run_summary_path`,
`agent_trace_path`, `agent_stdout_path`, `agent_stderr_path`,
`verifier_report_path`, and `output_dir`. The inspection commands above use
these paths.

The agent-server log prints the artifact directory after the agent sandbox has
stopped and its downloaded files pass validation, then again when verification
completes or the rollout fails.

## Test

Run the benchmark and resource-server tests with:

```bash
uv run pytest -q \
  benchmarks/legal_agent_bench/tests \
  resources_servers/legal_agent_bench/tests \
  responses_api_agents/legal_agent_bench_agent/tests \
  responses_api_agents/legal_agent_bench_native_agent/tests
```

Generated indexes, collation metrics, Harbor jobs, source documents, and skills
must not be committed.
