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

## Determine Integration Type

Before starting, determine which type of benchmark you're adding:

**Native overlay** — reuse an existing resources server. Do this first. Add `benchmarks/<name>/config.yaml` with `config_paths` and `_inherit_from` (copy `benchmarks/gsm8k`). Do **not** run `gym env init --resources-server` unless no scorer exists. Human docs: `fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx`.

**Native new scorer** — verification logic implemented directly in a Gym resources server:
- Resources server implements `verify()` with reward logic
- Agent server orchestrates model calls (use `simple_agent` for single-turn, or custom agent for multi-turn)
- Example: `code_gen`, `instruction_following`, `math_with_judge`
- CI requires `data/example.jsonl` (5), `data/example_metrics.json`, `data/example_rollouts.jsonl` (5); local `gym env test --resources-server` skips data checks unless `+should_validate_data=true`

**External benchmark** — wrapping a 3rd-party library that has its own orchestration:
- Integrate at the agent server level (not resources server)
- Agent's `/run` endpoint wraps the external library
- Pre-process from Gym schema to library input, post-process back to `BaseVerifyResponse`
- Reproduce publicly reported numbers with the original repo first, then reproduce again after Gym integration
- Add the dependency in `requirements.txt`

## Workflow

If this is an overlay on `math_with_judge`, `mcqa`, `code_gen`, or similar, skip scaffolding a new server. Copy the closest overlay and stop after prepare + CLI wiring:

| Task | Copy |
| --- | --- |
| Math / short answer | `benchmarks/gsm8k` |
| Multiple choice | `benchmarks/gpqa` |
| Unit-test code | `benchmarks/livecodebench/v5_2408_2502` |

Human walkthrough: `fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx`. Only do Step 1 when no existing scorer fits.

When the overlay sets `prompt_config`, JSONL rows are raw fields (`question`, `expected_answer`, …). Do **not** bake `responses_create_params.input` into those rows. `prepare()` must return a `Path` equal to `jsonl_fpath`. Keep generated JSONL gitignored.

Search scorers with `gym search resources-servers "…"` (bare `gym search` defaults to environments). Do not use `gym env init --benchmark` for in-tree overlays.

Smoke test (output is required):

```bash
gym env validate --benchmark my_bench
gym eval prepare --benchmark my_bench
gym eval run --benchmark my_bench \
  --model-type openai_model \
  --split benchmark \
  --output results/my_bench_rollouts.jsonl \
  --limit 2
```

`--no-serve` also needs `--agent`, `--input`, `--prompt-config`, and `--output`. Overlay-only PRs skip `gym env test --resources-server`.

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

Convert your source dataset to Gym JSONL format.

**Overlay with `prompt_config`:** write raw fields the prompt and verifier need. Do not include `responses_create_params.input`. Example (`math_with_judge`):

```json
{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "expected_answer": 72}
```

MCQA: see `benchmarks/gpqa/prepare.py` (`question`, `problem`, `options`, `expected_answer`). `prepare()` must return a `Path` equal to `jsonl_fpath`. Keep generated JSONL gitignored.

**New resources server example rows / no prompt_config:** each line has `responses_create_params.input` (OpenAI message format). Task-specific verification data goes in `verifier_metadata` or top-level scorer fields:

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

**Data conversion:** Overlay `prepare.py` lives next to `benchmarks/<name>/config.yaml`. For a new resources server, conversion scripts often live in the source dataset repo; exception when there is no external source repo. See `references/patterns.md` § "Data Conversion Script Pattern".

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

### Step 7: Baseline (reward profiling)

Run against multiple models to validate correctness. Recommended suite:
- Your policy model of interest
- At least one open-source instruct model (e.g. Qwen 3 30B A3B Instruct)
- At least one open-source thinking model (e.g. Qwen 3 30B A3B Thinking)
- At least one closed-source model (e.g. GPT-5 Nano or GPT-5)

```bash
# Collect rollouts
gym eval run --no-serve \
  --agent my_benchmark_simple_agent \
  --input resources_servers/my_benchmark/data/my_dataset.jsonl \
  --output results/rollouts.jsonl \
  --num-repeats 5 \
  --max-output-tokens 16384 \
  --temperature 1.0

# Compute per-task pass rates
gym eval profile --inputs resources_servers/my_benchmark/data/my_dataset.jsonl \
  --rollouts results/rollouts.jsonl

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

Leave `verified: false` on the first PR. Maintainers flip it after reward profiling. Include W&B links and screenshots in the PR if you already ran profiling.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```

## Constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/openai_utils.py`), not LiteLLM/Anthropic/other
- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through `nemo_gym.server_utils.request()` (aiohttp). httpx has O(n^2) connection pooling that hangs at high concurrency. When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter — see `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the pattern and `fern/versions/latest/pages/infrastructure/engineering-notes/aiohttp-vs-httpx.mdx` for the rationale.
- Pass configuration through Gym config (YAML), not environment variables
- Code must run on Linux
- `/run` endpoint must be async
- Errors from tool execution or bad model output must return error responses, not crash
- All commits require DCO sign-off (`-s`). Cryptographic signing (`-S`) is optional.

## Reference

For detailed code patterns, schemas, and examples: see [references/patterns.md](references/patterns.md).
