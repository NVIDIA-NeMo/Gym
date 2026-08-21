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

- Python 3.13.14 and the repository environment installed with `uv`
- One supported Gym sandbox provider: Docker, ECS Fargate, Enroot, Apptainer,
  OpenSandbox, Daytona, or OpenShell
- The provider's local CLI, service, credentials, and Gym dependency extra, as
  documented under `nemo_gym/sandbox/providers/<provider>/`
- Authorized OpenAI-compatible policy and judge endpoints in the root
  `env.yaml`
- At least 10 GB of free working space

Not required:

- Separate Harbor, Hermes, Claude Code, or Codex installations
- Anthropic or OpenAI vendor subscriptions or CLI logins for the Claude Code
  and Codex harnesses
- Docker when a non-Docker provider is selected and a compatible LAB image is
  already available

Gym provisions the pinned harness dependencies automatically. Every harness
uses the configured policy model endpoint. Access to your configured policy and
judge endpoints is still required and may itself be metered or paid.

See the [resource-server README](../../resources_servers/legal_agent_bench/README.md)
for endpoint configuration, source and license details, cache locations, and
troubleshooting. The initial source download is several hundred MiB, and the
first rollout provisions the selected harness inside a temporary sandbox. The
default Docker backend also builds the document-tooling image on first use.

## Set up

From the repository root:

```bash
uv venv --python 3.13.14
source .venv/bin/activate
uv sync --extra dev
```

For Docker, verify the default local backend:

```bash
docker info >/dev/null
```

For SDK-backed providers, install the sandbox dependencies as well:

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
Harbor retains its established response format and additionally reports
`mask_sample`, `agent_failed`, `model_connection_failed`, `agent_timed_out`, and
`failure_reason` for agent-phase failures. It preserves any partial trajectory,
but skips judging and forces reward to zero when one of those failures occurs.

## Choose output, context, and timeout limits

LAB does not prescribe one model-independent output-token limit. Its
[upstream adapters](https://github.com/harveyai/harvey-labs/tree/main/harness/adapters)
choose provider- and model-specific per-call limits, generally using the
model's large output capacity. For a locally hosted model, `64,000` is a
sensible starting point when the endpoint and hardware support it:

```bash
++responses_create_params.max_output_tokens=64000
```

Append that override to the native or configurable-harness command. Set the
limit as high as the model, server, total context window, and available KV
cache can sustain. A larger output reservation can reduce the input space
available within a fixed context window, so it is not always safe to use the
largest numerical value accepted by the API. If a higher limit exceeds local
memory capacity, reduce rollout and agent-server concurrency together before
lowering the limit.

An unreasonably low output limit can stop a long tool call or reasoning turn
before the agent finishes its deliverables. A context window that is too small
can have the same effect later in a multi-turn task. LAB scores those incomplete
outcomes, so either constraint can skew model-quality results downward. Record
both limits with published results and compare models using settings that do
not prematurely truncate otherwise supported work.

### Timeouts and turn limits

LAB does not define one canonical wall-clock timeout for a task. Its current
upstream [runner CLI](https://github.com/harveyai/harvey-labs/blob/main/harness/run.py)
exposes a turn budget and a per-command shell timeout, but the
[agent loop](https://github.com/harveyai/harvey-labs/blob/main/harness/agent_loop.py)
has no overall deadline. Upstream
[model adapters](https://github.com/harveyai/harvey-labs/tree/main/harness/adapters)
also do not share a model-request timeout, and the
[evaluator CLI](https://github.com/harveyai/harvey-labs/blob/main/evaluation/run_eval.py)
does not expose a uniform judge timeout. Provider SDK defaults therefore
differ.

Gym adds layered operational deadlines so a stalled endpoint or sandbox cannot
hold a rollout worker indefinitely. The checked-in defaults are the recommended
starting values for full evaluations:

| Layer | Default | Guidance |
| --- | ---: | --- |
| Complete agent phase | 10,800 seconds (3 hours) | Use for all harnesses. Claude Code and Codex also receive this as their inner harness timeout. |
| One policy-model request | 1,800 seconds (30 minutes) | Used by the native and Harbor loops. This is intentionally generous for slow local reasoning models. |
| Native tool preflight | 120 seconds | Allows document tooling to initialize on a cold or CPU-throttled sandbox. This does not increase the timeout for normal tool calls. |
| Shell command | 60 seconds | Used by the native and Harbor loops. Hermes uses a 180-second terminal timeout. A shell timeout is returned to the agent as a tool error. |
| Sandbox staging/collection | 900 seconds (15 minutes) | Covers extraction and collection of portable runtimes and task artifacts. Remote providers should give file-transfer API requests at least the same budget. Reduce concurrency if large parallel transfers saturate the provider. |
| Complete verifier phase | 3,600 seconds (1 hour) | Covers output staging, all criterion calls, and artifact collection. |
| One judge request | 90 seconds, with one retry | Increase only when the judge endpoint is healthy but consistently needs longer than 90 seconds. |

Start with these values. For a slow locally hosted policy model, reduce both
rollout and agent-server concurrency before changing timeouts. If successful
model generations genuinely take longer than 30 minutes, raise the policy-call
timeout and the enclosing agent-phase timeout together. Keep the outer agent
deadline comfortably above a single model call, and keep the verifier deadline
above the judge request timeout plus retries. Raising a timeout does not reserve
GPU memory; increasing output or context limits can.

The default native and Harbor configurations use a 60-turn budget, while
Hermes uses 90 turns and the CLI harnesses use their own configured stopping
behavior. A turn limit is a model-behavior constraint, not a timeout. Current
upstream LAB `main` uses 200 turns, but this Gym integration is pinned to an
earlier LAB revision and does not silently adopt later harness changes. Record
the turn budget as well as output, context, and timeout settings with published
results.

The common defaults are configured in
`responses_api_agents/legal_agent_bench_agent/configs/` and
`resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml`. For the
default native benchmark, the relevant override paths are:

```text
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.agent_timeout_seconds=<seconds>
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.sandbox_staging_timeout_seconds=<seconds>
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.verifier_timeout_seconds=<seconds>
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.agent_kwargs.model_timeout_seconds=<seconds>
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.agent_kwargs.preflight_timeout_seconds=<seconds>
+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.agent_kwargs.shell_timeout=<seconds>
+legal_agent_bench_benchmark_resources_server.resources_servers.legal_agent_bench.judge_request_timeout_seconds=<seconds>
```

Substitute the selected variant's agent and resource-server prefixes when
running Harbor, Hermes, Claude Code, or Codex. A whole-agent, policy-connection,
sandbox, or verifier timeout is an operational failure and is routed through
Gym's failure handling; it is not a completed zero-reward model outcome.

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

## Choose a sandbox provider

The native, Hermes, Claude Code, and Codex variants use Gym's shared sandbox
API. Their runtime builder, agent phase, and isolated verifier phase all use the
same selected provider. The Harbor compatibility variant uses Harbor's own
container orchestration instead and is Docker-only.

| Provider | LAB image | Policy-proxy routing | Provider setup |
| --- | --- | --- | --- |
| Docker (default) | Automatically built locally, or `sandbox_image` | Derived loopback URLs are translated for Docker Desktop or Linux bridge networking | Running Docker daemon |
| ECS Fargate | OCI registry image; prefer an immutable digest | Automatic Gym SSH reverse tunnel | AWS/SSM/ECR/S3 infrastructure and TCP access to the task SSH sidecar |
| Enroot | Registry/Docker URI or local `.sqsh` | Shares the orchestrator host network | `enroot` CLI |
| Apptainer | Registry/Docker URI or local `.sif` | Shares the orchestrator host network | `apptainer` CLI |
| OpenSandbox | Provider-accessible OCI image | Set a reachable proxy URL when Gym's proxy is host-local | OpenSandbox service and credentials |
| Daytona | Provider-accessible OCI image or provider-supported snapshot configuration | Set a reachable proxy URL when Gym's proxy is host-local | Daytona service and credentials |
| OpenShell | Provider-accessible OCI image | Set a reachable proxy URL when Gym's proxy is host-local | OpenShell gateway and credentials when required |

For the checked-in ECS Fargate configuration, export `AWS_PROFILE`,
`AWS_REGION`, and `AWS_DEFAULT_REGION`. Gym discovers the reference
infrastructure from `/<ssm_project>/ecs-sandbox/config` in SSM;
`ssm_project` defaults to `harbor`. That configuration must identify the ECS
cluster, subnets, security groups, task roles, ECR repository, S3 staging
bucket, and SSH-sidecar key material. The orchestrator must be able to reach
the task SSH sidecar on TCP port `52222`. See each provider YAML and README for
its current environment variables and service-specific options.

Non-Docker providers do not invoke the Docker CLI. Supply a provider-compatible
image through `NEMO_GYM_LAB_SANDBOX_IMAGE` or the `sandbox_image` override. The
image must contain LAB's document tooling plus `bash`, `curl`, and `tar`, and
must permit writes under `/sandbox`. The first rollout starts a short-lived runtime
builder sandbox with outbound package-download access. The resulting portable,
content-addressed runtime is reused by later rollouts.

### Build and publish the LAB image

After preparation, every task contains the same generated LAB image context.
The following example selects one of those contexts and uses Docker Buildx to
build and publish an OCI image to a registry that the sandbox provider can
access:

```bash
gym eval prepare --benchmark legal_agent_bench

LAB_IMAGE_CONTEXT="$(find \
  resources_servers/legal_agent_bench/data/cache/harbor_tasks/legal_agent_bench \
  -mindepth 2 -maxdepth 2 -type d -name environment -print -quit)"
export LAB_IMAGE="docker.io/<account>/legal-agent-bench:<tag>"

docker login docker.io
docker buildx build \
  --platform linux/amd64 \
  --tag "$LAB_IMAGE" \
  --push \
  "$LAB_IMAGE_CONTEXT"
docker buildx imagetools inspect "$LAB_IMAGE"
```

Replace `linux/amd64` when the target sandbox uses another architecture. For
reproducible evaluations, copy the published digest from the inspection output
and configure the immutable reference rather than its mutable tag:

```bash
export NEMO_GYM_LAB_SANDBOX_IMAGE="docker.io/<account>/legal-agent-bench@sha256:<digest>"
```

Docker Buildx is only an example image-building workflow. The image may instead
be produced and published by any preferred OCI-compatible builder, including a
CI system; Docker does not need to be installed on the machine that launches a
non-Docker LAB sandbox. That machine needs only the selected provider and an
image reference it can use. Enroot and Apptainer may also use local `.sqsh` and
`.sif` images, respectively, without publishing them to a registry. For a
private registry, configure image-pull credentials through the selected
provider rather than placing credentials in the LAB image reference.

OpenShell's Docker compute driver additionally requires `iproute2`, a
restricted `sandbox` user and group, and a work directory writable by that
identity. Images generated by LAB preparation include these requirements.

For OpenSandbox, LAB treats each task's declared CPU, memory, disk, and GPU as
its limits. The checked-in agent configs set `opensandbox_request_fraction:
0.25`, which requests 25% of the CPU and memory limits while retaining the full
disk and GPU requests. This request/limit split permits deliberate
oversubscription when many LAB sandboxes run concurrently. It matches the
[Mini SWE Agent 2 example](https://github.com/NVIDIA-NeMo/Gym/blob/main/responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml),
which requests 0.5 of a 2 CPU limit and 2 GiB of an 8 GiB memory limit. It
applies to the runtime builder, agent, and verifier. Raise the fraction toward
`1.0` if your cluster needs firmer CPU or memory reservations, or set it to
`null` to disable the split. For example, add
`+legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.opensandbox_request_fraction=0.5`
to the native `gym eval run` command below.

The standard provider YAMLs define a top-level sandbox named `sandbox`. Select
one by adding its config and pointing the LAB agent at that name. This native
example works for Docker, ECS Fargate, Enroot, Apptainer, OpenSandbox, Daytona,
or OpenShell by replacing `<provider>` and `<image>`:

```bash
gym eval run \
  --model-type vllm_model \
  --benchmark legal_agent_bench \
  --config nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml \
  --split benchmark \
  --output results/legal_agent_bench_native_<provider>_smoke.jsonl \
  --concurrency 1 \
  --limit 1 \
  +legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.sandbox_provider=sandbox \
  +legal_agent_bench_benchmark_native_agent.responses_api_agents.legal_agent_bench_agent.sandbox_image=<image>
```

For Hermes, Claude Code, or Codex, replace the benchmark and the native agent
prefix with the corresponding variant prefix. `sandbox_image` is passed through
unchanged, so use the native reference expected by the selected provider rather
than converting it in LAB.

ECS Fargate automatically tunnels the rollout-scoped Gym policy proxy. Enroot
and Apptainer can reach a host-local proxy through their shared host network.
OpenSandbox, Daytona, and OpenShell cannot reach the orchestrator's loopback
address. Prefer exposing the credential-free Gym policy proxy on a route
reachable from the sandbox, then set `NEMO_GYM_SANDBOX_MODEL_BASE_URL` or the
`sandbox_model_base_url` override. Do not put credentials in that URL. If only
a directly authenticated model endpoint is reachable, set
`NEMO_GYM_LAB_SANDBOX_MODEL_API_KEY_ENV` to the name of a launcher environment
variable containing a narrowly scoped, short-lived model key. LAB copies that
key only into the agent sandbox as `LAB_POLICY_API_KEY`; it is not serialized
into the runner configuration or supplied to the runtime builder or verifier.
The evaluated agent can read its own environment, so this fallback accepts key
exposure to untrusted agent code. Use a dedicated key with the minimum required
permissions and rotate it after the run. Every provider's verifier sandbox must
also be able to reach the configured judge endpoint.

OpenShell should use three separate policies through
`runtime_builder_provider_options`, `agent_sandbox_provider_options`, and
`verifier_sandbox_provider_options`. Give the builder only dependency-registry
egress, the agent only policy-proxy egress, and the verifier only judge egress.
The filesystem policy must allow the image runtime and
`/opt/legal-agent-bench` read-only and `/sandbox` writable. OpenShell injects a
policy-enforcing HTTP(S) proxy; the LAB runner opts Gym's inner HTTP client into
that proxy only for this provider. In the builder policy, configure the
`registry.npmjs.org` endpoint with `protocol: rest`, `access: read-only`,
`enforcement: enforce`, and `allow_encoded_slash: true`; npm uses encoded
slashes when resolving scoped packages such as the Claude Code dependency.
See OpenShell's
[policy schema](https://docs.nvidia.com/openshell/latest/reference/policy-schema#endpoint-object).

Benchmark variants inherit their agent configuration, so supply the complete
phase-option maps through the decoded environment settings rather than adding
nested map keys on the command line:

```bash
export NEMO_GYM_LAB_RUNTIME_BUILDER_PROVIDER_OPTIONS='{policy: /path/to/builder-policy.yaml}'
export NEMO_GYM_LAB_AGENT_SANDBOX_PROVIDER_OPTIONS='{policy: /path/to/agent-policy.yaml}'
export NEMO_GYM_LAB_VERIFIER_SANDBOX_PROVIDER_OPTIONS='{policy: /path/to/verifier-policy.yaml}'
```

The values are OmegaConf mappings and default to `{}`, so they have no effect
on providers that do not need phase-specific options.

Before spending model tokens, exercise create, upload, execute, download, and
cleanup through the same public API. For a checked-in provider YAML:

```bash
python responses_api_agents/legal_agent_bench_agent/scripts/smoke_provider.py \
  --config nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml \
  --image <image>
```

You can also use `--provider <name>` instead of `--config ...` to smoke a
provider's constructor defaults. A passing lifecycle smoke does not test the
policy or judge endpoints; follow it with the one-task `gym eval run` above.

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
The Harbor variant reports the agent-phase subset described above and applies
the same masking and no-judge behavior to those failures.

Configurable-runner artifacts are grouped by harness, model, dated session, and
task. The default roots are `results/legal_agent_bench/native_jobs`,
`results/legal_agent_bench/hermes_jobs`,
`results/legal_agent_bench/claude_code_jobs`, and
`results/legal_agent_bench/codex_jobs`. Harbor uses
`results/legal_agent_bench/harbor_jobs`.
Set `NEMO_GYM_LAB_RESULTS_DIR` before `gym eval run` to redirect the native and
configurable artifact root, for example to VM-native storage when Gym runs in
a Linux VM over a macOS-shared checkout. The rollout JSONL paths remain the
ones supplied with `--output`.

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
