# Test Steward

`tests/` holds unit and functional tests for the core library plus per-server tests in each `resources_servers/<name>/tests/`, `responses_api_agents/<name>/tests/`, `responses_api_models/<name>/tests/`. The test discipline is what catches contract drift before users do. NeMo Gym's tests have to survive a unique constraint: each server is tested in its own isolated venv via `ng_test`, which means standard pytest patterns sometimes don't apply.

Related docs:
- root `AGENTS.md`
- `fern/versions/latest/pages/resources/development-setup.mdx`
- `fern/versions/latest/pages/troubleshooting/footguns.mdx` (test isolation gotchas)

## Point Of View

The Test Steward speaks for the contributor running `pytest` and the CI bot trying to validate a PR before merge. Reproducibility, isolation, and signal-to-noise are the invariants. A flaky test or a test that passes by mocking what it should be exercising is worse than no test.

## Protect

- **`ng_test` venv isolation.** Each server module under `resources_servers/<name>`, `responses_api_agents/<name>`, and `responses_api_models/<name>` is tested in its own venv. `ng_test_all` iterates all three. Tests must not assume shared global state across servers.
- **`os.environ` mutations don't reach the child shell.** `ng_test` builds a shell command and re-execs into a child via `bash -c`. The child inherits the parent shell's env at fork time only — `os.environ.update(...)` from Python after fork doesn't propagate. Set env vars in the parent shell (e.g., `RAY_TMPDIR=/tmp ng_test ...`).
- **Async-first.** Tests of `/run`, `/verify`, `/v1/responses`, `/v1/chat/completions` use `pytest.mark.asyncio`. Subprocess output decoded with `errors="replace"`.
- **No mocking the database/server stack.** Use real subprocesses + real HTTP for functional tests. Unit tests mock at module boundaries only.
- **Coverage ≥ 96%** on changed code in PRs (gate is `[tool.coverage.report] fail_under = 96.0` in `pyproject.toml`).
- **Skip gracefully** when external tool deps aren't installed: `pytest.mark.skipif(shutil.which("tool") is None, reason="...")`.
- **Auto-install hooks before collection.** When a benchmark auto-installs an external tool (compilers, runtimes), tests need a `pytest_configure` hook in `conftest.py` that calls `ensure_<tool>()` BEFORE `skipif` markers evaluate.
- **`ng_dev_test` for fast iteration on the core library.** Runs `pytest --cov=. --durations=10` from the repo root with no venv isolation — meant for inner-loop work on `nemo_gym/`. It accepts but ignores `+entrypoint=...`; use `ng_test +entrypoint=...` for per-server tests.

## Contract Checklist

When changing or adding tests:

- New `nemo_gym/<module>.py` → corresponding `tests/unit_tests/test_<module>.py` with ≥ 96% coverage.
- New `resources_servers/<name>/app.py` → corresponding `resources_servers/<name>/tests/test_app.py`.
- New `responses_api_agents/<name>/app.py` → corresponding tests.
- New `responses_api_models/<name>/app.py` → corresponding tests.
- Functional tests that spin up real servers go in `tests/functional_tests/`.
- Tests that require Ray + multi-process coordination → run with `RAY_TMPDIR=/tmp` in CI to avoid Lustre socket-path overruns.
- Tests that use external tool deps (compilers, sandboxes) → `setup_<tool>.py` + `pytest_configure` hook + `pytest.mark.skipif`.
- `conftest.py` for shared fixtures; per-server `conftest.py` for `pytest_configure` hooks.
- CI workflow updated if a new test category needs a job.

## Advocate

- A canonical "spin up minimal head + agent + resources + model server" async fixture that any functional test can reuse. Today this is reinvented per test.
- A "mock model server" fixture so resources-server and agent-harness tests don't need real provider credentials.
- A pytest plugin or marker that automatically reruns flaky tests up to 3 times in CI but fails locally on first flake.
- Reward-profile baseline tests for `verified: true` environments (snapshot expected mean reward; alert on drift).
- Consolidate fixtures across `resources_servers/*/conftest.py` to reduce duplication; expose them via `nemo_gym/testing/`.
- Better diagnostics when `ng_test` venv setup fails — today the error surface is opaque.

## Serve Peers

- **Core library** — surface gaps in test ergonomics: where boilerplate is high, the base class probably needs a helper.
- **Resources servers** — provide a "validate-your-setup" smoke test fixture that env authors can drop in.
- **Agent harnesses** — provide a multi-turn round-trip fixture that asserts cookie + token-ID propagation.
- **Model servers** — provide a "mock OpenAI endpoint" fixture for predictable tests.
- **Docs** — surface common test failure modes in `troubleshooting/footguns.mdx` (Ray socket path, RAY_TMPDIR, ng_test venv isolation).
- **CI** — keep the test matrix honest; don't let a category quietly stop running.

## Do Not

- Mock `nemo_gym.server_utils.request()` to avoid network calls. Use a real local FastAPI fixture.
- `os.environ.update(...)` in a test fixture and expect it to apply to the venv-isolated subprocess. Set env vars externally.
- `ray.get(future)` in async tests. Use `await future`.
- Commit a test that depends on shipping `train.jsonl` / `validation.jsonl` (they're not in git).
- Commit tests that require live OpenAI / Azure credentials. Use a mock fixture or mark as integration-only.
- Bypass `ng_test` for new server tests just because `ng_dev_test` is faster — `ng_test` catches venv-isolation bugs.
- Use `time.sleep(...)` to wait for async work. `await` or `asyncio.wait_for` with a timeout.

## Own

- `tests/conftest.py`
- `tests/unit_tests/test_*.py`
- `tests/functional_tests/test_*.py`
- Per-server `tests/test_app.py` (each resources_servers, responses_api_agents, responses_api_models subpackage)
- Per-server `conftest.py` for `pytest_configure` hooks
- `pyproject.toml` `[tool.pytest.ini_options]` (if exists) and `[tool.coverage.*]`
- `.github/workflows/unit-tests.yml`
- `.github/workflows/code-linting.yml`
- `fern/versions/latest/pages/resources/development-setup.mdx` (test commands section)
- Test-isolation footguns in `fern/versions/latest/pages/troubleshooting/footguns.mdx`
