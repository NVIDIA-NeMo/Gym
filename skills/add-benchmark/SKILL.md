---
name: add-benchmark
license: Apache-2.0
description: >
  Add or scaffold a new benchmark, evaluation, or training environment in
  NeMo-Gym, through reward profiling. Not for debugging runs or editing docs.
metadata:
  author: NVIDIA <nemo-gym@nvidia.com>
  tags:
    - benchmark
    - training-environment
    - resources-server
    - reward-profiling
    - evaluation
---

# Add Benchmark to NeMo-Gym

## Purpose

Integrate a new benchmark, evaluation, or training environment into NeMo-Gym
end to end: scaffold a resources/agent server, convert source data to Gym JSONL,
implement reward verification, wire the YAML config, and validate with reward
profiling (baselining).

## When not to use this skill

- Debugging or fixing an existing benchmark, rollout, or reward-profiling run.
  Use the `nemo-gym-debugging` skill instead.
- Editing or adding documentation pages. Use the `nemo-gym-docs` skill instead.
- General code questions unrelated to adding a benchmark.

## Prerequisites

- NeMo Gym installed with the `ng_*` CLI available (`ng_init_resources_server`,
  `ng_prepare_data`, `ng_collect_rollouts`, `ng_reward_profile`, `ng_test`).
- A Linux host; executed code and servers run on Linux.
- A model server for smoke testing and baselining (an OpenAI-compatible endpoint
  or a local vLLM model server).
- For `train`/`validation` datasets: GitLab dataset-registry access and MLflow
  credentials in `env.yaml`.
- For external benchmarks: the upstream library and its dependencies, plus the
  original repo to reproduce published numbers.

## Determine Integration Type

Before starting, determine which type of benchmark you're adding:

**Native benchmark** — verification logic implemented directly in a Gym resources server:
- Resources server implements `verify()` with reward logic
- Agent server orchestrates model calls (use `simple_agent` for single-turn, or custom agent for multi-turn)
- Example: `code_gen`, `instruction_following`, `math_with_judge`

**External benchmark** — wrapping a 3rd-party library that has its own orchestration:
- Integrate at the agent server level (not resources server)
- Agent's `/run` endpoint wraps the external library
- Pre-process from Gym schema to library input, post-process back to `BaseVerifyResponse`
- Reproduce publicly reported numbers with the original repo first, then reproduce again after Gym integration
- Add the dependency in `requirements.txt`

## Instructions

### Step 1: Scaffold the server

Run `ng_init_resources_server` to generate the directory structure:

```bash
ng_init_resources_server +entrypoint=resources_servers/my_benchmark
```

This creates:
```
resources_servers/my_benchmark/
├── app.py              # Server template
├── configs/my_benchmark.yaml
├── data/.gitignore
├── tests/test_app.py
├── requirements.txt
└── README.md
```

For external benchmarks, create the agent server manually under `responses_api_agents/my_agent/` with the same structure.

### Step 2: Prepare data

Convert your source dataset to Gym JSONL format. Each line must have `responses_create_params.input` (OpenAI message format). Task-specific verification data goes in `verifier_metadata`.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "System prompt"},
      {"role": "user", "content": "Problem statement"}
    ]
  },
  "verifier_metadata": {
    "test_cases": [{"input": "...", "expected_output": "..."}],
    "task_id": "unique_id"
  }
}
```

**Data conversion**: Write conversion scripts in the **source repo** (e.g. your dataset repository), not in NeMo-Gym. Prompt files also belong in the source repo. Exception: when there is no external source repo. See `references/patterns.md` § "Data Conversion Script Pattern".

**`example.jsonl`**: Generate 5 entries for smoke testing. This file is committed directly to git in `data/example.jsonl`.

**`train`/`validation` datasets**: Upload to the GitLab dataset registry. These must not be committed to git.

```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=my_benchmark \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl
```

Requires MLflow credentials in `env.yaml`:
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

Security note: `env.yaml` holds credentials, so it should not be committed.
Confirm it is covered by `.gitignore` before staging. Where possible, source the
token from an environment variable or a secrets manager rather than a plaintext
file, and avoid passing tokens as command-line arguments, since these can be
recorded in shell history and process listings.

**`data/.gitignore`**: The scaffold generates default patterns (`*train.jsonl`, `*validation.jsonl`, etc.). If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`). If data was previously tracked, run `git rm --cached <file>`.

**Validate** your data with `ng_prepare_data` (example validation for PR
submission, and train/validation download from GitLab). See
[Verification with `ng_prepare_data`](references/patterns.md#verification-with-ng_prepare_data)
in patterns.md for the exact commands.

### Step 3: Implement verify()

Edit `app.py`. The `verify()` method receives model output + `verifier_metadata`, returns reward.

For code execution benchmarks, see `references/patterns.md` § "Subprocess Execution with Ray" and "Resources Server Pattern".

Security note: code-execution benchmarks run untrusted, model-generated code,
and dataset payloads may be adversarial. NeMo-Gym does not currently sandbox
executed code; the `tempfile` working directory used by the subprocess pattern
isolates files but is not a security boundary. Until per-execution sandboxing is
available, run these benchmarks only in an isolated, disposable environment that
you control, such as a dedicated CI container or VM, and not on hosts with access
to sensitive credentials or networks. Apply the timeout and concurrency limits
described below. See the security note in `references/patterns.md` § "Subprocess
Execution with Ray".

Critical rules:
- Return `reward` as 0.0 or 1.0 (binary)
- Handle empty/missing model output gracefully — return 0.0, don't crash
- Must handle 4k-65k concurrent requests without crashing
- Use `asyncio.Semaphore` for subprocess concurrency control
- For Ray remote tasks: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` in async context.
- Decode subprocess output with `errors="replace"`
- Strip `<think>`/`<thinking>` blocks before parsing model output (thinking models emit these)
- Tests should `pytest.mark.skipif` when external tools aren't installed
- If the benchmark auto-installs its tool (see Step 3b), add a `pytest_configure` hook in `conftest.py` to run the install before test collection — `skipif` evaluates at import time, before fixtures run

### Step 3b: Auto-install external tools (if applicable)

If the benchmark requires an external tool (compiler, runtime, etc.), auto-install it on server startup so users don't need manual setup. See `references/patterns.md` § "External Tool Auto-Install Pattern".

Key points:
- Create `setup_<tool>.py` with `ensure_<tool>()` — checks PATH, forks on `sys.platform` (brew on macOS, build from source on Linux)
- Call it in `model_post_init()` before semaphore init
- Build scripts should be idempotent and install into a local gitignored prefix
- Add a `pytest_configure` hook in `tests/conftest.py` that calls `ensure_<tool>()` before collection

### Step 4: Wire YAML config

Edit `configs/my_benchmark.yaml`. Define the resources server instance and agent pairing(s). See `references/patterns.md` § "YAML Config Pattern".

Key points:
- `verified: false` is auto-added by pre-commit hook (set to `true` after baselining)
- `license` is required for `train` and `validation` datasets
- Agent references resources server and model server by instance name

For multi-turn benchmarks, either use `proof_refinement_agent` or create a custom agent. See `references/patterns.md` § "Agent Patterns".

For `train`/`validation` datasets, add `gitlab_identifier` alongside `jsonl_fpath`:
```yaml
datasets:
- name: my_dataset
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: my_dataset.jsonl
  license: MIT
- name: example
  type: example
  jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

Both fields must coexist: `jsonl_fpath` is the local download destination, `gitlab_identifier` tells the system where to fetch from. `example` datasets don't need `gitlab_identifier` — they're committed to git directly.

### Step 5: Test

```bash
# Run server tests (creates isolated .venv, slow on first run)
ng_test +entrypoint=resources_servers/my_benchmark

# Run core library tests to check nothing broke
pytest tests/unit_tests/ -x
```

Test coverage must be >= 95%. Write tests for: verify pass, verify fail (wrong output), verify fail (no code extracted), verify fail (compilation error if applicable), verify timeout.

### Step 6: Smoke test end-to-end

```bash
# Start servers
ng_run "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Collect rollouts. This is the canonical rollout command used throughout:
# for a quick smoke test point at example.jsonl with +num_repeats=1; for
# baselining (Step 7) point at the full dataset and raise +num_repeats.
ng_collect_rollouts +agent_name=my_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/my_benchmark/data/example.jsonl \
  +output_jsonl_fpath=results/example_rollouts.jsonl \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Inspect results
```

### Step 7: Baseline (reward profiling)

Run against multiple models to validate correctness. Recommended suite:
- Your policy model of interest
- At least one open-source instruct model (e.g. Qwen 3 30B A3B Instruct)
- At least one open-source thinking model (e.g. Qwen 3 30B A3B Thinking)
- At least one closed-source model (e.g. GPT-5 Nano or GPT-5)

Re-run the Step 6 `ng_collect_rollouts` command against the full dataset
(`my_dataset.jsonl`, `+output_jsonl_fpath=results/rollouts.jsonl`) with
`+num_repeats=5` for variance estimation, then profile the rewards:

```bash
# Compute per-task pass rates
ng_reward_profile +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl \
  +rollouts_jsonl_fpath=results/rollouts.jsonl \
  +output_jsonl_fpath=results/profiled.jsonl \
  +pass_threshold=1.0

# Aggregate metrics (pass@1 = avg_reward, pass@k from max_reward)
python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl
```

Increase `num_repeats` until variance < 1% across runs on the same model.

Closed-source models should score at or above open-source models. If not, investigate for bugs. Inspect actual failure cases in the rollout JSONL, not just aggregate numbers.

For external benchmarks: reproduce the original repo's published numbers first. Then reproduce after Gym integration. Scores should match.

### Step 8: Pre-commit and PR

```bash
pre-commit run --all-files
```

First run may fail as hooks auto-modify files (`verified: false` flag, README table). Stage changes and run again.

Set `verified: true` in YAML config after successful baselining. Include W&B links and screenshots of results in the PR description.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```

## Examples

**Native code-gen benchmark (e.g. HumanEval+):** classify as native, run
`ng_init_resources_server`, convert problems to Gym JSONL with `test_cases` in
`verifier_metadata`, implement `verify()` that extracts code and runs it in a
sandbox returning binary reward, wire `simple_agent` in YAML, then smoke test
and baseline.

**External library wrapper (e.g. SWE-bench):** classify as external, create an
agent server under `responses_api_agents/`, pre-process the Gym schema into the
library's input and post-process its result into `BaseVerifyResponse`, add the
dependency to `requirements.txt`, and reproduce the published numbers with the
original repo both before and after integration.

**Training environment (e.g. grade-school math):** scaffold a native resources
server, upload `train`/`validation` splits to the GitLab dataset registry (only
`example.jsonl` is committed to git), add `gitlab_identifier` alongside
`jsonl_fpath` in the YAML, baseline across instruct/thinking/closed-source
models, then run GRPO training with NeMo RL.

## Constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/openai_utils.py`), not LiteLLM/Anthropic/other
- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through `nemo_gym.server_utils.request()` (aiohttp). httpx has O(n^2) connection pooling that hangs at high concurrency. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter — see `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern and `docs/infrastructure/engineering-notes/aiohttp-vs-httpx.md` for the rationale.
- Pass configuration through Gym config (YAML), not environment variables
- Code must run on Linux
- `/run` endpoint must be async
- Errors from tool execution or bad model output must return error responses, not crash
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)

## Limitations

- No built-in sandboxing for executed code; isolation is the operator's
  responsibility (see the security notes above).
- Reward is binary only; this workflow does not cover graded or partial credit.
- Supported on Linux only.
- `train` and `validation` data must come from the GitLab dataset registry, not
  git; only `example.jsonl` is committed.
- Meaningful pass-rate variance requires multiple rollouts per task.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| Verifier returns 422 or 500 | Row schema does not match the resources server request model | Compare the failing JSONL row against the server's request model before changing infra |
| `ng_test` skips all tests | External tool not installed at collection time | Add a `pytest_configure` hook in `conftest.py` that runs the auto-install before collection |
| Empty or partial reward-profiling output | Verifier exceptions or a partial rollout cache | Inspect the first real verifier exception; for deeper diagnosis use the `nemo-gym-debugging` skill |
| Dataset accidentally tracked by git | Filename does not match the `data/.gitignore` patterns | Add a matching pattern and run `git rm --cached <file>` |
| Closed-source model scores below open-source | Usually a verifier or code-extraction bug | Inspect failure cases in the rollout JSONL rather than aggregate numbers |

## Reference

For detailed code patterns, schemas, and examples: see [references/patterns.md](references/patterns.md).
