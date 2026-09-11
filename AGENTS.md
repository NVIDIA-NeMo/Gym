# AGENTS.md

Unified root instructions for AI coding assistants (Cursor, Claude, Codex, OpenCode, Copilot, and similar).

`CLAUDE.md` is a symlink to this file so Claude Code and other tools share one source of truth.

Humans: see [Development Setup → Use of AI and LLM Tools](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup#use-of-ai-and-llm-tools) (maintainer response policy) and [Agent Skills](https://docs.nvidia.com/nemo/gym/latest/contribute/agent-skills).

## Quality bar

- Prefer focused changes. Do not make unrelated "drive-by" edits. If a drive-by fix is worth keeping, open a separate issue or PR.
- Intentional synthetic scaling of environments is fine when scoped via an issue or focused PR; do not dump unreviewed bulk diffs.
- You (the human author) own every line submitted. Treat model output as untrusted until reviewed.
- For behavior-changing environment or agent work: run representative real smoke rollouts with a model and inspect
  agent and verifier behavior. Green unit tests alone are not enough. Metadata-only catalog or manifest changes and
  docs-only changes do not require model compute.
- Before opening a PR, run the local checks that mirror CI: tests (skip or N/A for docs-only), `pre-commit run --all-files`, and DCO sign-off (`git commit -s`). Cryptographic `-S` signing is optional and not required.
- AI-generated tests must assert real behavior; avoid vacuous pass-through tests.
- Prefer the vetted skills under `.agents/skills/` (see [Agent Skills](https://docs.nvidia.com/nemo/gym/latest/contribute/agent-skills)).
- Docs live under `fern/versions/latest/pages/`. Bleeding-edge nav is `fern/versions/main.yml`. See `fern/README.md` and the `nemo-gym-docs` skill.
- Do not introduce licenses incompatible with Apache-2.0. New source files need the standard NVIDIA SPDX header.

## Pull Requests

- For normal authored PRs, use a Conventional Commit-style title: `type(optional-scope): imperative summary`.
  Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, and `chore`; use `design` for a
  design-only proposal. The scope is optional but should name the affected component, such as `agent`, `eval`,
  `sandbox`, or a specific environment. Automated release and cherry-pick PRs may keep their generated title format.
- Do not copy Megatron Bridge's `[area]` title prefix into Gym. Use the repository's `area:*` label taxonomy instead.
- The PR body must explain what changed and why, link the relevant issue or state why one is unnecessary, list the
  exact validation performed, and include rollout evidence or an explicit `N/A` with justification. Call out user-facing
  compatibility, migration, or benchmark-result impact when applicable.
- Keep incomplete work as a draft. Mark it ready for review only after reviewing the final diff and recording the
  applicable local checks. Use the `nemo-gym-pr-checks-and-labels` skill for CI routing and label selection.

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
uv venv && uv sync --extra dev
pre-commit install

# Run servers
gym env start \
    --resources-server example_single_tool_call \
    --model-type vllm_model

# Run tests for a specific server (creates .venv per server, installs deps, runs pytest)
# First run is slow. Use skip_venv_if_present config or place a .venv to skip venv creation.
gym env test --resources-server example_single_tool_call

# Run all server tests (slow: one venv per server, so it needs an explicit opt-in)
gym env test --all

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

- `pyproject.toml` is authoritative for Python, Ruff, formatter, and coverage settings;
  `.pre-commit-config.yaml` is authoritative for the hooks that enforce them.
- Line length: 119
- Python 3.13.14+, async-first
- Ruff for linting and formatting (double quotes, isort)
- Add parameter and return annotations to new or changed public functions and methods, and explicit types to public
  dataclass and Pydantic fields. Avoid `Any` at public boundaries when a concrete model, protocol, `TypedDict`, or
  `object` plus narrowing expresses the contract. Match surrounding annotation style and do not perform unrelated
  `Optional`/`List` syntax migrations.
- Mypy is available as a development dependency but is not a repository-wide strict CI gate. Run it on a focused area
  when that area is already type-checkable; do not claim repo-wide `mypy --strict` compliance.
- Public APIs and objects included in generated reference docs need useful docstrings. Comments should explain intent,
  invariants, or tradeoffs rather than narrating the code.
- Use keyword-only arguments for new public parameters that are easy to swap or misread, especially repeated same-type
  values and boolean controls.
- Use module loggers for library and server diagnostics. Direct console output is acceptable in CLI/user-facing paths;
  do not impose a blanket ban on `print()` copied from another repository.
- Raise specific exceptions with actionable context. Broad exception handling is appropriate only at a deliberate
  request, task, or process boundary where the error is logged, translated, or preserved for the caller.
- CI reads the coverage threshold from `[tool.coverage.report].fail_under` via `scripts/ci/cov_fail_under.py`; do not
  hard-code a second threshold in contributor guidance.

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
