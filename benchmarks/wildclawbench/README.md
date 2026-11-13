# WildClawBench with OpenClaw

Evaluate the original 60-task suite through **Eval Factory → NeMo Gym →
OpenClaw → the original task grader**. Gym owns preparation, episode dispatch,
policy-call correlation, trace conversion and aggregation. Factory supplies
launch composition. No OpenCode adapter or training changes are included.

This is an evaluation-only baseline using the original runner, not a template for new Gym agents.
Follow-up: [migrate to Gym agent sessions](https://github.com/bpritam14/WildClawBench-gym/issues/1).

This is a development integration. Offline checks are not a fresh benchmark
baseline, a leaderboard reproduction or live sandbox qualification.

## Provision the original benchmark once

Use Python satisfying Gym's `pyproject.toml` on the driver. The separate native
worker needs Python 3.11+ with `python-dotenv`, `pyyaml` and `tokenizers>=0.20,<0.23`.
The driver also needs Git, Bash and either Docker or Enroot already installed.
The integration does not install a container runtime on a shared host.

The Factory Slurm launch path depends on upstream [host-driver PR #4878](https://github.com/NVIDIA-NeMo/Gym/pull/4878)
and [benchmark-directory PR #5932](https://github.com/NVIDIA-NeMo/Gym/pull/5932).
Their shared orchestration changes are deliberately excluded here; compose those
prerequisites before using Factory on Slurm. Standalone Gym execution below does
not need them. Both prerequisites were still open during this review update.

From a Gym checkout:

```bash
export WILDCLAW_CACHE_ROOT="$PWD/.runtime/wildclawbench-cache"
export XDG_CACHE_HOME="$WILDCLAW_CACHE_ROOT/xdg"
export XDG_CONFIG_HOME="$WILDCLAW_CACHE_ROOT/config"
export XDG_DATA_HOME="$WILDCLAW_CACHE_ROOT/data"
export PIP_CACHE_DIR="$WILDCLAW_CACHE_ROOT/pip"
export UV_CACHE_DIR="$WILDCLAW_CACHE_ROOT/uv"
export UV_PYTHON_INSTALL_DIR="$WILDCLAW_CACHE_ROOT/python"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
export WILDCLAW_ROOT=/shared/benchmarks/wildclawbench
python -m benchmarks.wildclawbench.setup --source "$WILDCLAW_ROOT"
python3 -m venv "$WILDCLAW_ROOT/.venv"
export WILDCLAW_PYTHON="$WILDCLAW_ROOT/.venv/bin/python"
"$WILDCLAW_PYTHON" -m pip install python-dotenv pyyaml 'tokenizers>=0.20,<0.23'
```

`source-pin.json` selects the public MIT-licensed
[Gym execution snapshot](https://github.com/bpritam14/WildClawBench-gym/tree/878f4129f117f84dda6072b7fe2e1b4af3ffa0aa)
at `878f4129f117f84dda6072b7fe2e1b4af3ffa0aa` and hashes all 204 native
source, task, skill and instruction-prefix files used by this protocol.
Public source provisioning requires no NVIDIA network or credentials.
The snapshot carries gateway isolation/readiness, bounded process cleanup,
compaction and LF-only transcript fixes. The upstream checkout is not an
interchangeable dependency at this pin. No benchmark implementation is
vendored into Gym; native execution and graders remain in that source.

Download and prepare the official `internlm/WildClawBench` Hugging Face task
`workspace/` and `Images/wildclawbench-ubuntu_v1.3.tar` using that source's
README and `script/prepare.sh`. Preparation includes football/lecture/launch
videos, SAM3 weights and safety-task repositories; it can require authenticated
video download access. Source cloning alone does not supply these assets.

For Docker, load the image as `wildclawbench-ubuntu:v1.3` and set
`WILDCLAW_SANDBOX_RUNTIME=docker`. For Enroot, convert the official image with
a site-approved Docker-to-Enroot conversion. The required result is
`$WILDCLAW_ROOT/.cache/wildclawbench-ubuntu_v1.3.sqsh`; set
`WILDCLAW_SANDBOX_RUNTIME=enroot`. Review asset-preparation scripts before
running them on a shared cluster. Their historical defaults are not current
cluster-policy recommendations.

Provision and inspect your own workspace and image before any GPU allocation.

## Credentials and storage

Load private credentials into the compute job's environment:

- `BRAVE_API_KEY`: original search tool.
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `JUDGE_MODEL`: original external
  graders and task-authorized auxiliary API calls. The variable names do not
  require using OpenRouter as the actual provider.
- Policy credentials belong to Gym's separately selected model configuration.

The judge endpoint must accept the original graders' Chat Completions requests,
including `max_tokens=300` and JSON-object mode when the JSON instruction appears
only in the system message. A plain "reply OK" probe does not test this contract.
On NVIDIA's inference API, the tested `azure/openai/gpt-5.4` route rejected that
request shape; `openai/openai/gpt-5.4` accepted it unchanged. Select the compatible
route with `JUDGE_MODEL`; the generic default is the upstream `openai/gpt-5.4`
identifier. The effective identifier is recorded with every result. Do not modify
grader prompts or suppress judge-failure masks to work around a provider error.

The evaluated model calls Gym's rollout-specific `/v1` endpoint with an inert
local key; provider credentials stay with the Gym model server. The judge is
separate and must stay fixed across compared models. Auxiliary calls explicitly
permitted by original tasks are part of the native protocol, not necessarily
calls to the evaluated policy. Inspect the task's `Env` section when choosing
credentials. Do not commit `.env`, cookies, captures or populated artifacts.

Native worker caches and Enroot state live beneath each episode's
`storage/`, with home mounting disabled. Container `/root` is part of the
isolated workspace-backed rootfs; it does not bind the host's `/root`.
The driver itself must also receive workspace-backed XDG, uv, pip and Python
cache settings. Never point storage at a protected host home directory.
Artifact directories are private (`0700`) and contain original tool output.

## Prepare and run a smoke

Run from the Gym checkout with its environment activated. This example uses a
separately managed OpenAI-compatible Chat Completions policy endpoint, including
native tool calling and image support for multimodal tasks:

```bash
export WILDCLAW_OUTPUT_ROOT=/shared/evals/wildclaw-smoke/episodes
export WILDCLAW_DATASET_PATH=/shared/evals/wildclaw-smoke/input.jsonl
export WILDCLAW_TASKS=06_Safety_Alignment_task_4_authority
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1
gym eval prepare --config benchmarks/wildclawbench/config.yaml
gym eval run --config benchmarks/wildclawbench/config.yaml \
  --agent wildclaw_agent --split benchmark \
  --model-type vllm_model --model POLICY_MODEL --model-url POLICY_URL \
  --output /shared/evals/wildclaw-smoke/rollouts.jsonl \
  --concurrency 1 --num-repeats 1 \
  +policy_model.responses_api_models.vllm_model.entrypoint=wildclawbench.py \
  +policy_model.responses_api_models.vllm_model.propagate_context_overflow_errors=true \
  +observability_enabled=true \
  +model_call_capture_dir=/shared/evals/wildclaw-smoke/model-calls \
  +require_complete=true
```

Configure the policy key via the model config's runtime environment reference;
avoid putting secrets on a command line. The endpoint must be reachable from
the task sandbox. Docker uses its normal bridge network. Expose only the model proxy with
`+use_absolute_ip=true '+absolute_ip_server_types=[responses_api_models]'`;
keep the head and agent servers on loopback. Enroot shares host networking.

The working context follows the fork's tested convention: serving context
131072, completion cap 8192, working context 98304, compaction reserve/floor
32768 and recent history 4096. Sampling comes from the selected policy profile;
declare it explicitly when comparing models. These settings are distinct from
historical benchmark runs with different compaction or serving budgets.

Original timeouts, prompts, skills, warmups and rubrics are preserved. Agent
process errors, malformed/missing transcripts, gateway fallback, missing
reasoning replay signatures and explicit judge/transport errors are unscored.
Recovered API errors remain audit evidence. A clean generation cap or the task's
own deadline can produce an incomplete answer that the original grader scores;
a missing required deliverable is a measured zero, not a retry trigger.

## Full suite, shards and artifacts

Set `WILDCLAW_TASKS=all` for the complete suite, or supply comma-separated exact
task IDs. `WILDCLAW_SHARD_COUNT=N` and `WILDCLAW_SHARD_INDEX=0..N-1` partition
sorted selected tasks without overlap. Give each worker a distinct output root
and prepared dataset. Warmup services have fixed ports, so run only one native
worker on each physical node, not multiple independent drivers on one node.
Preparation hashes selected workspace assets, including large media; allow CPU
time for that preflight before reserving GPUs. The worker checks those hashes
again before execution. Do not modify shared assets during a run.

Each episode retains `binding.json`, `worker.json`, `result.json`, `chat.jsonl`,
`trace.json`, `usage.json`, `score.json`, agent/gateway/worker logs and task
outputs. Rollout rows expose original reasoning, tool calls and tool results
through Gym's OpenClaw conversion and link to these full native artifacts.
Conversion reports unavailable lineage/compaction evidence as observation gaps;
it does not claim to reconstruct discarded history or provide training tokens.
Gym model-call sidecars supply rollout-correlated policy requests/responses.

`wildclawbench/partial_mean_score` is diagnostic. `wildclawbench/overall_score`
appears only for all 60 unique, valid tasks with unchanged native rewards.
Masked or missing tasks never become policy zeros in this metric. Merge
disjoint shards on a host that can read their original episode receipts:

```bash
python -m benchmarks.wildclawbench.merge \
  --rollouts /shared/evals/shard-0/rollouts.jsonl \
  --rollouts /shared/evals/shard-1/rollouts.jsonl \
  --output /shared/evals/merged-metrics.json
```

Supply all shards. Duplicate task attempts, source drift, reward mismatches or
incomplete coverage fail the merge. Never average shard means. Infrastructure
retries are an explicit operator decision in a new run; choose one valid
attempt per task before merging. HTTP retries of a completed rollout reuse its
saved result. An interrupted rollout ID is refused rather than reexecuted.

Factory's binding uses standard Gym prepare/run and pins this adapter. Slurm
`driver.container: null` selects only the trusted host controller; a non-null
image preserves the usual containerized driver. Host drivers reject container
mount directives, and Kubernetes requires a container image.

Before a local-model run, use the site's real per-GPU utilization guard, release
the allocation after 25 idle minutes (including startup), fail closed on missing
telemetry and require every node's acknowledgement before model startup. Model
readiness has a 20-minute budget; limit pending/warming replicas to two at
submission. This benchmark adapter does not submit or monitor serving GPUs.

## Execution and scoring limitations

Enroot is an opt-in trusted-host compatibility mode: it shares the host PID and
network namespaces. Use a dedicated trusted allocation; it is not a Docker
security boundary. A policy process can reach loopback services and host PIDs.
Use environment references for credentials rather than literal keys in service
registries. Container copies retain file contents, symlinks and POSIX executable modes without
requiring host ACL/ownership preservation on the Enroot filesystem.
Process markers permit best-effort cleanup of detached descendants;
processes that erase their environment can evade this mechanism. Copy-path
checks reject known rootfs escapes, but shared-host concurrent filesystem mutation
is not a supported security boundary.

The original grader runs in the policy-used writable container, so policy
modification of grader dependencies is an upstream score-trust limitation.
Seven graders expose machine-readable judge fallback methods; these are masked.
Eleven other upstream graders only log certain judge failures, without a
machine-readable flag. Those failures cannot all be detected by this adapter;
inspect private grader logs before reporting a baseline. The pinned task text
predates upstream November 4/5 fixes: scores are version-specific and cannot be
claimed as a reproduction of a newer leaderboard without a matched revision.

The protocol records the judge model and immutable sandbox image identity;
merging different values is rejected. Native hidden model/thinking/image-model
environment overrides are disabled. The policy provider explicitly declares
reasoning and text/image input, so OpenClaw routes vision through the evaluated
policy rather than an ambient auxiliary image model. Select sampling in the policy profile and
retain the resolved Gym configuration with results. Required search and task
`Env` credentials are checked before policy execution. A missing key masks the
sample, with a namespaced failure reason, rather than producing a scored zero.

Agent-session follow-up scope: gateway mode, image layout/OpenClaw version,
configuration and compaction passthrough, image/search/skills tools, and transcript
compatibility with all nine safety graders. The current PR does not implement
this migration. Live four-task plus full-suite leaderboard validation remain required before
claiming integration qualification.
