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

## Code provenance

- Avoid copying or adapting code from other repositories. Write an independent implementation whenever possible.
- When copying or adapting external code is necessary, you **must** record the source URL, exact revision (commit SHA),
  and applicable license in **both the PR description and the header of every file containing that code**. Identify
  the original source file(s) and the affected Gym files in the PR description.
- Retain all required copyright notices, license notices, and attribution. Follow the Apache-2.0 compatibility and
  vendoring requirements in [CONTRIBUTING.md](CONTRIBUTING.md#licensing-of-contributions).

## Pull Requests

- For normal authored PRs, use a Conventional Commit-style title: `type(optional-scope): imperative summary`.
  Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, and `chore`; use `design` for a
  design-only proposal. The scope is optional but should name the affected component, such as `agent`, `eval`,
  `sandbox`, or a specific environment. Automated release and cherry-pick PRs may keep their generated title format.
- Do not copy Megatron Bridge's `[area]` title prefix into Gym. Use the repository's `area:*` label taxonomy instead.
- The PR body must explain what changed and why, link the relevant issue or state why one is unnecessary, list the
  exact validation performed, and include rollout evidence or an explicit `N/A` with justification. Call out user-facing
  compatibility, migration, or benchmark-result impact when applicable.
- **CI does not start on its own for unverified commits or external contributors.** A qualified NVIDIA-NeMo member
  must comment `/ok to test <full 40-character commit SHA>` on the PR (get it with `git rev-parse HEAD`; a short SHA
  is not enough). After the head changes, the new full SHA needs a new comment. If CI checks are missing, check this
  first. That comment authorizes CI to run contributor code, so never post it yourself unless the user explicitly asks.
  Details: [CONTRIBUTING.md](CONTRIBUTING.md#continuous-integration-ci).
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

1. **Dataset** — JSONL with one task per row. NeMo Gym uses the OpenAI Responses API as its data format because it represents multi-turn, tool-calling agentic trajectories without custom serialization. Each row has `responses_create_params.input` (the input messages in Responses API format) and `verifier_metadata` (task-specific data passed to the verifier)
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

- **Use aiohttp, not httpx, for async HTTP.** All async HTTP calls must go through NeMo Gym's global aiohttp client (`nemo_gym.client.http_client.request()`). Do not use `httpx.AsyncClient` — httpx/httpcore has O(n^2) connection pooling that causes hangs at high concurrency (16k+ requests). When wrapping external libraries that use httpx internally, replace their HTTP transport with an aiohttp adapter. See `resources_servers/tavily_search/app.py` (`TavilySearchAIOHTTPClient`) for the adapter pattern.
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

For sandboxed harnesses, install the harness runtime inside the sandbox during agent-session setup, before activation;
do not install an unused copy on the agent-server host. An optional local execution path may install its host runtime
on first use instead of startup. Document that first-call setup, share concurrent installation attempts, surface
installation failures, and allow a later call to retry. Required host tools still follow the startup rule above.

## Sandbox

Servers and agent harnesses run tasks in remote sandboxes through `nemo_gym.sandbox`, which hides provider differences.

- **Keep the `nemo_gym.sandbox` public API minimal and lifecycle-only**: create/start, exec, upload/download, stop
  (and provider capabilities such as pause/resume or endpoints). Put utilities and helpers built on top of it outside
  `nemo_gym.sandbox`, for example in `nemo_gym/agent_utils/` or the server that uses them.
- **Resolve providers from config** with `resolve_provider_config(sandbox_provider, global_config_dict)` and merge
  `resolve_provider_metadata(...)` into `SandboxSpec.metadata`. Do not hard-code a provider or parse provider blocks by
  hand. Merge metadata as provider defaults, then `sandbox_config.metadata`, then per-task labels, and keep label
  values short (truncate IDs to 63 characters).
- **Deliver files through `SandboxSpec.files` or `start_with_setup()`**, not ad hoc writes after `start()`. Both stop
  the sandbox if file delivery or setup fails. Send text with `upload`, not by interpolating it into an `exec` shell
  command, and `shlex.quote` every value that does go into a command.
- **Stop every sandbox you start, on every path.** Wrap use in `try/finally` and call `stop()`. For sandboxes kept per
  session, key them by session ID, pop the entry only after `stop()` succeeds (so a later close can retry), and stop
  any remaining sandboxes in the server's lifespan shutdown. Set `ttl_s` so the provider reclaims leaked sandboxes.
- **Check `return_code` and `error_type` on every `exec` result.** A non-`None` `error_type` (for example a timeout)
  means the sandbox did not report a process exit code. Treat `stdout`/`stderr` as optional (`result.stdout or ""`).
- **Separate infrastructure failures from task failures.** When verification is inconclusive (sandbox start, exec,
  or provider error), retry on a fresh sandbox a bounded, configurable number of times
  (`inconclusive_verification_retries`), and report `evaluation_completed=False` instead of a silent reward of 0.

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

## Imports from nemo_gym

- Code outside `nemo_gym/` (servers, benchmarks, scripts, tests, and code samples in docs and skills) imports public
  names from the package's public API: `from nemo_gym.<package> import Name`. If such code needs a name the package
  does not export, add it to the package's `_PUBLIC_API` in its `__init__.py` rather than importing the submodule.
- Tests may import private names from submodules, and a `mock.patch` target names the submodule that looks the name
  up.
- Code inside `nemo_gym/` imports submodules directly (`from nemo_gym.<package>.<module> import Name`), never a
  package's public API, so the import-linter contract in `pyproject.toml` sees every dependency. A package imports
  only packages below it.

## Code Style

- `pyproject.toml` is authoritative for Python, Ruff, formatter, and coverage settings;
  `.pre-commit-config.yaml` is authoritative for the hooks that enforce them.
- Line length: 119
- Python 3.13.14+, async-first
- Ruff for linting and formatting (double quotes, isort)
- Prefer the smallest clear implementation that satisfies current requirements. Do not add abstractions, options,
  dependencies, or layers for hypothetical future needs.
- Simplicity means obvious code, not merely fewer lines. Preserve required behavior, compatibility, performance, and
  observability; extract helpers when they genuinely improve clarity or reuse.
- Add parameter and return annotations to new or changed public functions and methods, and explicit types to public
  dataclass and Pydantic fields. Avoid `Any` at public boundaries when a concrete model, protocol, `TypedDict`, or
  `object` plus narrowing expresses the contract. Match surrounding annotation style and do not perform unrelated
  `Optional`/`List` syntax migrations.
- Mypy is available as a development dependency but is not a repository-wide strict CI gate. Run it on a focused area
  when that area is already type-checkable; do not claim repo-wide `mypy --strict` compliance.
- Public APIs and objects included in generated reference docs need useful docstrings. Comments should explain intent,
  invariants, or tradeoffs rather than narrating the code.
- Name things by what they are. Avoid qualifiers that carry no information, such as "native", "legacy", "new", or
  "old"; name the specific mechanism, owner, or format instead (for example, "agent session" rather than "native
  session"). This applies to identifiers, comments, docs, error messages, and test names.
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
