---
name: add-benchmark
description: >
  Guide for adding a new benchmark or training environment to NeMo-Gym.
  Use when the user asks to add, create, or integrate a benchmark, evaluation,
  training environment, or resources server into NeMo-Gym. Also use when wrapping
  an existing 3rd-party benchmark library. Covers the full workflow: data preparation,
  resources server implementation, agent wiring, YAML config, testing, and reward
  profiling (baselining). Triggered by: "add benchmark", "new resources server",
  "integrate benchmark", "wrap benchmark", "add training environment", "add eval".
---

# Add Benchmark to NeMo-Gym

The Gym page [Add a benchmark](../../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx) is the current command reference. When it and this skill disagree, the page wins. This skill adds the order of work and the checks a benchmark must pass before it merges (Step 7).

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

## Workflow

### Step 1: Scaffold the server

Run `gym env init` to generate the directory structure:

```bash
gym env init --resources-server my_benchmark
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

**`train`/`validation` datasets**: Upload to the GitLab dataset registry — these must NOT be committed to git.

```bash
gym dataset upload --storage gitlab \
    --name my_benchmark \
    --revision 0.0.1 \
    --input resources_servers/my_benchmark/data/my_dataset.jsonl
```

Requires MLflow credentials in `env.yaml` (or passed via CLI):
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

**`data/.gitignore`**: The scaffold generates default patterns (`*train.jsonl`, `*validation.jsonl`, etc.). If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`). If data was previously tracked, run `git rm --cached <file>`.

**Validate** your data:
```bash
# Validate example data (for PR submission)
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir /tmp/prepare \
    --mode example_validation

# Download and prepare train/validation from GitLab
gym dataset collate --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --output-dir data/my_benchmark \
    --mode train_preparation \
    --download \
    +data_source=gitlab
```

### Step 3: Implement verify()

Edit `app.py`. The `verify()` method receives model output + `verifier_metadata`, returns reward.

For code execution benchmarks, see `references/patterns.md` § "Subprocess Execution with Ray" and "Resources Server Pattern".

Critical rules:
- Return `reward` as 0.0 or 1.0 (binary)
- Handle empty/missing model output gracefully — return 0.0, don't crash
- For an environment failure (sandbox down, judge unreachable), do not score 0.0. Return `mask_sample`, `failure_kind` (as `<server>:<kind>`) and `failure_reason` so the row leaves the aggregate. Add one test case that masks a row
- Must handle 4k-65k concurrent requests without crashing
- Use `asyncio.Semaphore` for subprocess concurrency control
- For Ray remote tasks: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` in async context.
- Decode subprocess output with `errors="replace"`
- Strip `<think>`/`<thinking>` blocks before parsing model output (thinking models emit these)
- Tests should `pytest.mark.skipif` when external tools aren't installed
- If the benchmark auto-installs its tool (see Step 3b), add a `pytest_configure` hook in `conftest.py` to run the install before test collection — `skipif` evaluates at import time, before fixtures run

If the server keeps state per session (a sandbox handle, a connection, a session dict):
- Set `num_workers: 1` and enforce it in `model_post_init` (raise `ValueError` otherwise). The framework checks this for agent servers, not for resources servers
- Override `close_resources_session()`; do not add your own `/close_session` route. It must succeed for a session that is already closed or never seeded, must not let a later seed recreate the session, and must raise if the release fails
- Hold a per-session `asyncio.Lock` around the session dict, so a close racing a seed cannot leak a sandbox
- If seed fails after allocating a resource, free it before re-raising
- Export `VERIFIER_FIXTURE` (`gym env test` needs it). Set `REVERIFY_MODE = ReverifyMode.STATELESS` on the config only when `verify()` uses just the request payload

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
gym env test --resources-server my_benchmark

# Run core library tests to check nothing broke
pytest tests/unit_tests/ -x
```

Test coverage must be >= 95%. Write tests for: verify pass, verify fail (wrong output), verify fail (no code extracted), verify fail (compilation error if applicable), verify timeout.

### Step 6: Smoke test end-to-end

```bash
# Start servers
gym env start \
    --config resources_servers/my_benchmark/configs/my_benchmark.yaml \
    --model-type openai_model

# Quick test with example data
gym eval run --no-serve \
  --agent my_benchmark_simple_agent \
  --input resources_servers/my_benchmark/data/example.jsonl \
  --output results/example_rollouts.jsonl \
  --num-repeats 1 \
  --max-output-tokens 16384 \
  --temperature 1.0

# Inspect results
```

### Step 7: Exit checks

A benchmark merges only when it passes these checks. Put the evidence in the PR.

| Check | What it asks | How to check |
|---|---|---|
| Health | Very few rollouts are unhealthy. | `gym eval run` writes `quality_summary.json` into the run folder. `gym eval health-check <run_dir>` reruns the checks. Read the counts: it does not fail the run for you. `unobserved` means Gym has no model-call record to judge by. It is not unhealthy. |
| Evidence | Every rollout passes the `gym-p0/v1` evidence gate. Harness work only. | `python scripts/inspect_harness_conformance.py --bundle <run_dir>/rollouts.jsonl --output <dir>`. Exit 0 pass, 1 evidence failure, 2 checker error. |
| Score | The score matches the reference score, inside the tolerance, on one current frontier model. | `gym eval compare` |
| Config | The resolved Gym config is saved with the result. | `resolved-config.yaml` is in the run folder. |

Run on one benchmark and harness pair. Read the rollouts, not the exit code: an exit of 0, an empty failures file and a rising rollout count prove nothing. List every health check you ignore (`--health-check-ignore`) in the PR, with the reason.

Reward profiling for training environments is a separate skill: `nemo-gym-reward-profiling`.

For an external benchmark: reproduce the original repo's published numbers first, then again after the Gym integration. The scores should match.

### Step 8: Pre-commit and PR

```bash
pre-commit run --all-files
```

First run may fail as hooks auto-modify files (`verified: false` flag, README table). Stage changes and run again.

Set `verified: true` in YAML config after the exit checks pass. Link the run folder and its `quality_summary.json` in the PR description.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```


## Adding a harness instead

Use the [`add-harness`](../add-harness/SKILL.md) skill. It covers the session contract, pairing configuration, harness conformance, real pairing runs, and the harness exit checks. The health and evidence checks above apply to harness work too, on live rollouts: the conformance probes use scripted replies and no real sandbox. An Evaluation-level adapter fails the evidence check by design; `add-harness` says what to record instead.

## Constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/openai_utils.py`), not LiteLLM/Anthropic/other
- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through `nemo_gym.client.http_client.request()` (aiohttp). httpx has O(n^2) connection pooling that hangs at high concurrency. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter — see `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern and `fern/versions/latest/pages/infrastructure/engineering-notes/aiohttp-vs-httpx.mdx` for the rationale.
- Pass configuration through Gym config (YAML), not environment variables
- Code must run on Linux
- `/run` endpoint must be async
- Errors from tool execution or bad model output must return error responses, not crash
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)

## Reference

For detailed code patterns, schemas, and examples: see [references/patterns.md](references/patterns.md).
