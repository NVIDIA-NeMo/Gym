# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NeMo-Gym is NVIDIA's library for building RL training environments for LLMs (RLVR). It uses a microservice architecture with three composable FastAPI server types that communicate over async HTTP.

## Common Commands

```bash
# Setup
uv venv && uv sync --extra dev --group docs
pre-commit install

# Run servers
ng_run "+config_paths=[resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Run tests for a specific server (creates .venv per server, installs deps, runs pytest)
# First run is slow. Use skip_venv_if_present config or place a .venv to skip venv creation.
ng_test +entrypoint=resources_servers/example_single_tool_call

# Run all server tests
ng_test_all

# Run core library unit tests
pytest tests/unit_tests/ -x

# Run a single test file
pytest tests/unit_tests/test_openai_utils.py -x

# Lint and format
ruff check --fix .
ruff format .

# Pre-commit (runs ruff, formatting, custom hooks)
pre-commit run --all-files

# Collect rollouts
ng_collect_rollouts +agent_name=<agent> +input_jsonl_fpath=<data.jsonl> +output_jsonl_fpath=<output.jsonl> +num_repeats=5 "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Profile results (compute per-task pass rates)
ng_profile +input_jsonl_fpath=<data.jsonl> +rollouts_jsonl_fpath=<rollouts.jsonl> +output_jsonl_fpath=<profiled.jsonl> +pass_threshold=1.0

# Check server health
ng_status

# Dev test (runs pytest directly in server dir, no venv isolation)
ng_dev_test +entrypoint=resources_servers/example_single_tool_call

# Dump merged config
ng_dump_config "+config_paths=[...]"

# Dataset management (HF)
ng_upload_dataset_to_hf +dataset_name=<name> +version=<ver> +input_jsonl_fpath=<path> +hf_repo_id=<repo>
ng_download_dataset_from_hf +dataset_name=<name> +version=<ver> +output_jsonl_fpath=<path> +hf_repo_id=<repo>
```

## Architecture

Three server types, all FastAPI apps communicating via aiohttp:

- **Resource Servers** (`resources_servers/`): Implement `verify()` — task verification and reward computation. Return reward 0.0 or 1.0.
- **Response API Models** (`responses_api_models/`): Implement `chat_completions()` and `responses()` — LLM inference. Four variants: openai, azure_openai, vllm, local_vllm.
- **Response API Agents** (`responses_api_agents/`): Implement `responses()` and `run()` — orchestrate model-tool call loops. `simple_agent` is the default single-turn agent; others include `proof_refinement_agent` (multi-turn correction), `verifiers_agent`, `swe_agents`, etc.

A **HeadServer** coordinates all server lifecycles, config, and Ray cluster init.

### Base Class Hierarchy

```
BaseServer (Pydantic model with config + server_client)
└── SimpleServer (FastAPI app setup, middleware stack)
    ├── SimpleResourcesServer  →  implement verify()
    ├── SimpleResponsesAPIModel  →  implement chat_completions(), responses()
    └── SimpleResponsesAPIAgent  →  implement responses(), run()
```

### Data Flow

JSONL input → agent `/run` → model `/v1/responses` → (tool calls if any) → resource server `/verify` → reward → JSONL output

### Inter-Server Communication

`ServerClient` wraps aiohttp with retry logic (3 tries, exponential backoff). Session cookies propagate through the call stack for stateful environments. The global aiohttp client is a singleton with connection pooling.

## Configuration

Hydra + OmegaConf for hierarchical YAML composition. CLI overrides use `+key=value` syntax.

Each server instance is a top-level key in YAML that maps to a server type + config:
```yaml
my_server_instance:
  resources_servers:        # server type directory
    my_server:              # server subdirectory name
      entrypoint: app.py
      domain: coding
      # ... server-specific config fields
```

Agent configs reference their resource and model servers:
```yaml
my_agent_instance:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: my_dataset
        type: train
        jsonl_fpath: path/to/data.jsonl
```

Model endpoint config goes in `env.yaml` at project root:
```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: your-key
policy_model_name: your-model
```

## JSONL Data Schema

Each line in input JSONL:
```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  },
  "verifier_metadata": { ... }
}
```

`responses_create_params.input` follows OpenAI message format. `verifier_metadata` is passed through to the resource server's `verify()` for task-specific validation data (test cases, expected answers, etc.).

Output JSONL (from `ng_collect_rollouts`) contains the full verify response per rollout, including at minimum:
```json
{
  "reward": 1.0,
  "response": { "output_text": "..." },
  "task_index": 0
}
```
Additional fields depend on the resource server's `VerifyResponse` class.

## Dataset Management

### Dataset types and where they live

- **`example`** datasets (5 entries for smoke testing) are committed directly to git in `data/example.jsonl`.
- **`train`** and **`validation`** datasets are hosted in the GitLab dataset registry. They must NOT be committed to git.

### GitLab dataset registry

Upload a JSONL dataset:
```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=my_benchmark \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl
```

Requires MLflow credentials in `env.yaml` (or passed via CLI):
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

The tracking URI format is `https://<gitlab-host>/api/v4/projects/<PROJECT_ID>/ml/mlflow`.

### YAML config: gitlab_identifier + jsonl_fpath

Both fields coexist. `jsonl_fpath` is the local download destination; `gitlab_identifier` tells the system where to fetch from:
```yaml
- name: my_dataset
  type: validation
  jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: my_dataset.jsonl
  license: MIT
```

### data/.gitignore

Every resource server has `data/.gitignore` (generated by `ng_init_resources_server`):
```
*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
```

If your filename doesn't match these patterns (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`). If data was previously tracked, run `git rm --cached <file>`.

### ng_prepare_data

Validate example data (for PR submission):
```bash
ng_prepare_data "+config_paths=[...]" +output_dirpath=/tmp/prepare +mode=example_validation
```

Download and prepare train/validation from GitLab:
```bash
ng_prepare_data "+config_paths=[...]" +output_dirpath=data/my_benchmark \
    +mode=train_preparation +should_download=true +data_source=gitlab
```

## Adding a New Benchmark (Resource Server + Agent)

For wrapping an existing 3rd-party benchmark library, integrate at the agent server level: wrap the library in `/run`, pre-process from Gym schema to library input, post-process back to `BaseVerifyResponse`. Reproduce publicly reported numbers with the original repo first, then reproduce again after Gym integration.

For native benchmarks, follow these steps:

### 1. Create the resource server

Copy an existing server as template:
- `example_single_tool_call` — simplest example
- `code_gen` — subprocess execution with Ray (good for compilation/execution benchmarks)

Required structure:
```
resources_servers/my_server/
├── app.py              # Server class extending SimpleResourcesServer
├── configs/my_server.yaml
├── data/example.jsonl  # 5 examples for quick testing
├── tests/__init__.py
├── tests/test_app.py
├── requirements.txt    # just: -e nemo-gym[dev] @ ../../
└── README.md
```

The `verify()` method receives the model output and `verifier_metadata`, returns a response with `reward` field. The `verifier_metadata` dict is opaque to the framework — define whatever fields your benchmark needs (test cases, expected answers, task IDs, etc.) and pass them through the JSONL data.

### 2. Create or reuse an agent

- `simple_agent` — single-turn, works for most benchmarks. Just pair it with your resource server in the YAML config.
- `proof_refinement_agent` — multi-turn correction loop (model gets error feedback and retries). Copy this if your benchmark benefits from iterative refinement.

Agent structure:
```
responses_api_agents/my_agent/
├── app.py              # Server class extending SimpleResponsesAPIAgent
├── configs/my_agent.yaml
├── tests/__init__.py
├── tests/test_app.py
└── requirements.txt
```

For multi-turn agents, propagate cookies from the incoming request through all downstream calls: `cookies=request.cookies`. Also propagate token IDs (`prompt_token_ids`, `generation_token_ids`, `generation_log_probs`) from model responses when constructing the next turn's input — these are needed for RL training.

### 3. Wire up the YAML config

A single YAML file in `configs/` typically defines both the resource server and its agent pairings. The agent references the resource server and model server by name.

### 4. Prepare data

Input JSONL has one problem per line. System prompt goes in the `input` messages. Task-specific verification data goes in `verifier_metadata`.

If converting from another format, write the conversion script in the source repo (e.g. your dataset source repo) — conversion scripts and prompt files do not belong in the NeMo-Gym repo. Upload only the converted JSONL to the GitLab registry.

Generate `data/example.jsonl` with 5 entries (committed to git). Upload `train`/`validation` datasets with `ng_upload_dataset_to_gitlab`. Add `gitlab_identifier` to the YAML config. See "Dataset Management" above for the full workflow.

Validate your data:
```bash
ng_prepare_data "+config_paths=[...]" +output_dirpath=/tmp/prepare +mode=example_validation
ng_prepare_data "+config_paths=[...]" +output_dirpath=data/my_benchmark +mode=train_preparation +should_download=true +data_source=gitlab
```

### 5. Baseline (reward profiling)

Run against multiple models to validate correctness:

```bash
# Start servers
ng_run "+config_paths=[resources_servers/my_server/configs/my_server.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts (start with example.jsonl for quick smoke test)
ng_collect_rollouts +agent_name=my_agent +input_jsonl_fpath=<data.jsonl> +output_jsonl_fpath=results/rollouts.jsonl +num_repeats=5 "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Compute per-task pass rates
ng_profile +input_jsonl_fpath=<data.jsonl> +rollouts_jsonl_fpath=results/rollouts.jsonl +output_jsonl_fpath=results/profiled.jsonl +pass_threshold=1.0

# Aggregate metrics (pass@1 = avg_reward, pass@k from max_reward)
python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl
```

Run on both instruct and thinking models. Thinking models emit `<think>`/`<thinking>` blocks in their output — your code extraction logic must strip these before parsing.

Use `openai_model` for endpoints supporting `/v1/responses`, `vllm_model` for `/v1/chat/completions`.

### 6. Important constraints

- Use NeMo Gym's OpenAI client (`nemo_gym/openai_utils.py`), not LiteLLM/Anthropic/other clients. It's `openai<=2.6.1` for schema compatibility.
- Pass all configuration through Gym config (YAML), not environment variables. This includes model URLs, API keys, etc.
- Environments must handle errors gracefully — tool failures and bad model outputs should return meaningful error responses, not crash the server. Must handle 4k-65k concurrent requests without crashing.
- The `/run` endpoint must be async. Use `asyncio.Semaphore` for concurrency control if shelling out to external processes.
- Tests should skip gracefully if external tools aren't installed (e.g. `pytest.mark.skipif(shutil.which("tool") is None, ...)`).
- If a benchmark auto-installs its tool dependency (see "External Tool Auto-Install" below), add a `pytest_configure` hook in `conftest.py` to run the install before test collection — `skipif` markers evaluate at import time, before fixtures run.
- Executables must run on Linux.
- Increase num_repeats until variance is < 1% across runs on the same model.

## Code Style

- Line length: 119
- Python 3.12+, async-first
- Ruff for linting and formatting (double quotes, isort)
- Test coverage must be >= 95%
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)

## Pre-commit Hooks

Notable custom hooks that auto-modify files:
- `add-verified-flag`: Adds `verified: false` to new resource server YAML configs (`verified: true` means the benchmark has been baselined and reviewed; new servers start as `false`)
- `update-readme-table`: Updates the resource server table in root README.md
- `ruff-format`: Auto-formats code

First run may fail as hooks modify files. Stage the changes and commit again.

To avoid committing unrelated auto-fixes from other servers, scope pre-commit to your files:
```bash
pre-commit run --files resources_servers/my_benchmark/**/*
```
If hooks modify files in other directories, discard those changes:
```bash
git checkout -- resources_servers/other_server/
```

## External Tool Auto-Install

When a benchmark requires an external tool (compiler, runtime, etc.), auto-install it on server startup so users don't need manual setup:

1. Create a `setup_<tool>.py` module with an `ensure_<tool>()` function that:
   - Checks `shutil.which("tool")` — returns early if already on PATH
   - Forks on `sys.platform`: macOS (brew), Linux (build from source via bash script)
   - Updates `os.environ["PATH"]` and `os.environ["LD_LIBRARY_PATH"]` for the current process
   - Verifies the tool runs successfully after install
2. Call `ensure_<tool>()` in the server's `model_post_init()` (runs once at startup)
3. For tests: add a `pytest_configure` hook in `conftest.py` that calls `ensure_<tool>()` before collection, so `skipif(shutil.which("tool") is None)` markers see the installed tool
4. Build-from-source scripts should be idempotent (skip if artifacts exist) and install into a local prefix (e.g. `.<tool_name>/` in the server dir, gitignored)

## Cluster / HPC Gotchas

- **Ray socket path length**: On systems with long working directory paths (e.g. Lustre mounts), Ray's AF_UNIX socket paths can exceed the 107-byte Linux limit. Fix: `RAY_TMPDIR=/tmp` before running tests or `ray.init()`.
- **`ng_test` venv isolation**: `ng_test` creates isolated venvs per resource server. `os.environ` changes in Python don't propagate — set env vars externally (e.g. `RAY_TMPDIR=/tmp ng_test ...`).

## Async Patterns

- Use `asyncio.Semaphore` to bound concurrent subprocess/external calls
- For Ray remote tasks in async code: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` directly in async context.
- Decode all subprocess output with `errors="replace"` to handle non-UTF8
- Guard optional nested fields: `(body.field or {}).get("key", default)`
