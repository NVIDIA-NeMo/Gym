# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

NeMo Gym is a library for evaluating and improving models and agents using environments. It provides infrastructure to develop environments, scalably run evaluation and training, and a collection of popular benchmarks and training environments. All components are composable and modular — bring your own agent, model, or environment and integrate with Gym where you need it.

An environment is the complete system an agent interacts with to complete a task. It consists of a dataset (tasks to solve), an agent harness (how the model interacts with the world), a verifier (task completion scoring), and state (per-task execution context).

## Architecture

Environments decompose into four concepts:

| Concept | NeMo Gym Component |
|---------|-------------------|
| Dataset | JSONL: one row per task |
| Agent Harness | FastAPI Agent Server (`responses_api_agents/`) |
| Verifier + State | FastAPI Resources Server (`resources_servers/`) |
| Model | FastAPI Model Server (`responses_api_models/`) or your own |

Base class hierarchy:
```
BaseServer (Pydantic model with config + server_client)
└── SimpleServer (FastAPI app setup, middleware stack)
    ├── SimpleResourcesServer  →  implement verify()
    ├── SimpleResponsesAPIModel  →  implement chat_completions(), responses()
    └── SimpleResponsesAPIAgent  →  implement responses(), run()
```

For full architecture and concepts (environments, training approaches, verification), see `fern/versions/latest/pages/about/`.

## Creating Environments

The typical workflow is to create your own environments tailored to your evaluation or training task. An environment consists of:

1. **Dataset** — JSONL with one task per row. NeMo Gym uses the OpenAI Responses API as its native format because it natively represents multi-turn, tool-calling agentic trajectories without custom serialization. Each row has `responses_create_params.input` (the input messages in Responses API format) and `verifier_metadata` (task-specific data passed to the verifier)
2. **Resources Server** — implements verification logic, environment-specific tools, and per-task state isolation
3. **Agent Harness** — reuse a built-in agent harness (e.g. OpenHands) or bring your own
4. **Model** — use any LLM endpoint via the Model Server (supports inference providers like OpenAI, and vLLM for local/open models), or manage inference in your own agent harness
5. **YAML config** — wires the resources server, agent, and model server together

For guidance on how to build environments, see `fern/versions/latest/pages/environment-tutorials/`. For evaluation, see `fern/versions/latest/pages/get-started/quickstart.mdx`. For training framework integrations, see `fern/versions/latest/pages/training-tutorials/`.

## Environment Design Recommendations

- **Use NeMo Gym's Model Server for inference** — standardizes different model providers behind a common format and manages token IDs needed for training.
- **Hydra YAML for configuration** — pass configuration through Gym's Hydra config system so it's composable and reproducible across runs.
- **Graceful error handling** — environments must handle tool failures and bad model outputs with meaningful error responses, not crash the server.
- **Async endpoints** — the `/run` endpoint must be async. Use `asyncio.Semaphore` for concurrency control if shelling out to external processes.
- **Test skip guards** — tests should skip gracefully if external tools aren't installed (e.g. `pytest.mark.skipif(shutil.which("tool") is None, ...)`).

## Communication & Async Patterns

Servers communicate via `ServerClient`, which wraps aiohttp with retry logic (3 tries, exponential backoff) and connection pooling via a singleton aiohttp client.

- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through NeMo Gym's global aiohttp client (`nemo_gym.server_utils.request()`). Do not use `httpx.AsyncClient` — httpx/httpcore has O(n^2) connection pooling that causes hangs at high concurrency (16k+ requests). When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter. See `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the adapter pattern.
- **Propagate session cookies** through all downstream calls (`cookies=request.cookies`) for stateful environments.
- Use `asyncio.Semaphore` to bound concurrent subprocess/external calls
- For Ray remote tasks in async code: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` directly in async context.
- Decode all subprocess output with `errors="replace"` to handle non-UTF8
- Guard optional nested fields: `(body.field or {}).get("key", default)`

## External Tool Auto-Install

When an environment requires an external tool (compiler, runtime, etc.), auto-install it on server startup so users don't need manual setup:

1. Create a `setup_<tool>.py` module with an `ensure_<tool>()` function that:
   - Checks `shutil.which("tool")` — returns early if already on PATH
   - Forks on `sys.platform`: macOS (brew), Linux (build from source via bash script)
   - Updates `os.environ["PATH"]` and `os.environ["LD_LIBRARY_PATH"]` for the current process
   - Verifies the tool runs successfully after install
2. Call `ensure_<tool>()` in the server's `model_post_init()` (runs once at startup)
3. For tests: add a `pytest_configure` hook in `conftest.py` that calls `ensure_<tool>()` before collection, so `skipif(shutil.which("tool") is None)` markers see the installed tool
4. Build-from-source scripts should be idempotent (skip if artifacts exist) and install into a local prefix (e.g. `.<tool_name>/` in the server dir, gitignored)

## Common Commands for Building & Testing Environments

```bash
# Setup
uv venv && uv sync --extra dev --group docs
pre-commit install

# Run servers
gym env start \
    --resources-server example_single_tool_call \
    --model-type vllm_model

# Run tests for a specific server (creates .venv per server, installs deps, runs pytest)
# First run is slow. Use skip_venv_if_present config or place a .venv to skip venv creation.
gym env test --resources-server example_single_tool_call

# Run all server tests
gym env test

# Run core library unit tests
pytest tests/unit_tests/ -x

# Run a single test file
pytest tests/unit_tests/test_openai_utils.py -x

# Lint and format
ruff check --fix .
ruff format .

# Pre-commit (runs ruff, formatting, custom hooks)
pre-commit run --all-files

# Check server health
gym env status

# Dev test (runs the core unit tests with coverage: pytest --cov)
gym dev test

# Dump merged config
gym env resolve --config ...
```

## Code Style

- Line length: 119
- Python 3.12+, async-first
- Ruff for linting and formatting (double quotes, isort)
- Test coverage must be >= 96%
- All commits require DCO sign-off (`-s`) and cryptographic signature (`-S`)

## Pre-commit Hooks

Notable custom hooks that auto-modify files:
- `add-verified-flag`: Adds `verified: false` to new resources server YAML configs (`verified: true` means the benchmark has been baselined and reviewed; new servers start as `false`)
- `update-readme-table`: Updates the resources server table in root README.md
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

## Cluster / HPC Gotchas

- **Ray socket path length**: On systems with long working directory paths (e.g. Lustre mounts), Ray's AF_UNIX socket paths can exceed the 107-byte Linux limit. Fix: `RAY_TMPDIR=/tmp` before running tests or `ray.init()`.
- **`gym env test` venv isolation**: `gym env test` creates isolated venvs per resources server. `os.environ` changes in Python don't propagate — set env vars externally (e.g. `RAY_TMPDIR=/tmp gym env test ...`).
