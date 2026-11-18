# Agent-harness review

Use this when a PR changes anything under `responses_api_agents/`, an agent's configs or presets, or
text that tells users which agent or config to run: log warnings, error messages, READMEs, Fern pages,
and skills.

Start with [agent-review/review.md](agent-review/review.md), the findings reviewers raise most often
on agent PRs, item by item. This file adds how to classify the change, the blast-radius map, and the
contract tables.

Two documents own the contract. Read them in the worktree before reviewing. Where they and this
checklist disagree, they win:

- `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`: the session contract (seed,
  activation, close), sandbox ownership, tools, runtime installation, output conversion, retries and
  close.
- `.agents/skills/add-harness/SKILL.md`: what a new or upgraded harness must pass before it merges.
  For those PRs, review against its Step 1 checklist and Step 7 exit checks, and report each row that
  has no evidence in the PR.

## Classify the change

Write these down before reviewing. Most agent findings come from a change that is right for one case
and wrong for another.

- **Path.** On the session path, an Environment Server (`single_agent_turn`,
  `single_agent_turn_legacy`, `harbor`, or a benchmark's own) seeds `/v1/agent_sessions`, activates
  `/v1/responses`, and closes `/v1/agent_sessions/close`. On the legacy path, a `legacy_agent` relay
  calls the agent's own `/run`. Some agents serve both and choose by config: `terminus_2_sandboxed_agent`
  and `miniswe_sandboxed_agent` take `/run` only when `resources_server` is set. A change to one path
  must leave the other alone unless the PR says so.
- **Sandbox ownership.** Borrowed: Resources returns `sandbox_access` at seed and destroys the sandbox.
  Agent-owned: the agent's `sandbox_provider` creates it and stops it on close and on failed setup.
  In-process: no sandbox.
- **Capability level** the PR claims: Evaluation, Observability, Environment tools and task sandboxes,
  or Training. It decides which add-harness exit checks apply.
- **Inheritance.** Find subclasses with `grep -rn "(<ClassName>)" responses_api_agents/`; each one
  inherits the change. Note subclasses that skip a parent's hooks: `PiSandboxedAgent.model_post_init`
  calls `SimpleResponsesAPIAgent.model_post_init` directly, so nothing `PiAgent.model_post_init` adds
  reaches it.

## Map the blast radius

For each changed agent, list every config that loads it and trace each one:

```
preset: benchmarks/, resources_servers/*/configs/, environments/, nemotron_recipes/,
        and the per-agent overrides in nemo_gym/data/tasks/harbor/prepare.py
  -> resources server, and the sandbox shape its seed returns
     (sandbox_access | sandbox_handle | sandbox_descriptor | none)
  -> allowed_agents in that resources server's config
  -> environment server (single_agent_turn, single_agent_turn_legacy, legacy_agent, harbor)
  -> the agent config the preset sets: num_workers, resources_server, tool_servers, sandbox_provider
```

```bash
grep -rln "<agent_name>" benchmarks resources_servers/*/configs environments nemotron_recipes \
  responses_api_agents/*/configs nemo_gym/data/tasks/harbor/prepare.py
grep -n "sandbox_access=\|sandbox_handle\|sandbox_descriptor" resources_servers/<server>/app.py
grep -n "allowed_agents" resources_servers/<server>/configs/*.yaml
```

Check every finding against every row of the map, not only the preset the PR tested.

## What to check

### Session contract

| Check | Finding when it fails |
|---|---|
| A seed the agent cannot serve (a sandbox it cannot use, a required grant it never consumes) raises `AgentSeedRejectedError`, a final 422. Optional grants it does not use are skipped. | The episode fails later, or runs without its tools. |
| Activation for an unknown or closed session returns 409. A retried activation replays the bound turn instead of running the harness again. | Duplicate harness runs and mixed trajectories. |
| Close returns an `AgentObservationBundle`, with gaps for evidence the harness cannot report, even when activation failed. | Lost observations, or a close that raises. |
| Session state is process-local, so the agent runs with `num_workers: 1`, or overrides `multiple_workers_unsupported_reason()` so that `run_webserver` refuses more. | A session's requests reach a worker that never saw it. |
| Outcomes follow "Keep execution health separate from reward" on the add-or-update page: a setup or provider failure is reported as a failure, not as a scored answer, and a reached limit keeps partial work. | Infrastructure noise scored as model failures, or model failures retried. |

### Sandbox, runtime, and tools

| Check | Finding when it fails |
|---|---|
| A borrowed sandbox is never destroyed by the agent. An agent-owned one is stopped on close and on failed setup. | Leaked or double-freed sandboxes. |
| A sandbox failure never falls back to running on the host. | Task code runs on the Gym host. |
| A sandboxed harness installs its runtime into the sandbox during session setup. Building or downloading it on the Agent Server and unpacking it in the sandbox is fine (`SandboxPythonRuntime`, Claude Code, Droid); an unused copy installed on the host is not (`AGENTS.md`, "External Tool Auto-Install"). | Host pollution, and runs that pass locally but fail in a clean sandbox. |
| Every model call, including summarization, compaction, and subagent calls, uses the rollout-prefixed Model Server URL. | Missing token IDs for training, and unattributed calls. |
| Async HTTP goes through `nemo_gym.client.http_client.request()` (aiohttp), session cookies propagate, subprocess and external calls are bounded by an `asyncio.Semaphore`, and subprocess output is decoded with `errors="replace"`. | Hangs at high concurrency, lost session state, or crashes on non-UTF-8 output. |
| Required MCP grants are connected and ready before the first model request. | The harness runs without the tools the task needs. |
| Controls the harness does not support are rejected or documented, not silently dropped. | A config that looks applied but is not. |

### Startup and process model

| Check | Finding when it fails |
|---|---|
| Code in `model_post_init` or at module import runs once per server build. With `num_workers: N`, the main process builds the server, then each uvicorn worker builds it again, up to twice (as `__mp_main__` and on the `module:app` import). Startup logs, installs, downloads, and file writes either check `is_nemo_gym_fastapi_worker()` or are idempotent and safe to run concurrently. | A one-time warning printed 2N+1 times, or N+1 racing installs. |
| A startup side effect that depends on config (for example "only when `resources_server` is set") matches how every preset in the map sets that field, including the overrides in `nemo_gym/data/tasks/harbor/prepare.py`. | A warning or refusal on a path that is meant to stay quiet. |

### User-facing guidance

Apply this to every new or changed warning, error message, README banner, Fern page, or skill line
that tells users to switch agent, config, or command. Check it against every preset in the map that
shows the message.

| Check | Finding when it fails |
|---|---|
| The named replacement is in the resources server's `allowed_agents`. Otherwise the pairing guard refuses it unless the user passes `--allow-unsupported-pairing` or sets `NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING=1`. | Users follow the advice and the run is refused. |
| The resources server returns the sandbox shape the replacement needs. `single_agent_turn` forwards only `sandbox_access`, so a server that returns only a `sandbox_handle` or `sandbox_descriptor` gives a session harness no task sandbox. | Users follow the advice and the agent gets no sandbox. |
| The replacement supports what the preset uses. For example, `tool_servers` and `network_access` on `*_sandboxed_agent` presets have no session-path equivalent. | Users lose tools or network policy without being told. |
| A config hint taken literally does not break the presets that show it. For example, `resources_server: null` makes `/run` return 422. | Users break a working preset by following the message. |
| Each quoted command runs as written. A `gym eval run <entry>` target must exist under `environments/harbor/` and needs `gym dataset pull <entry>` first. Each README anchor resolves. | Dead commands and links. |
| A subclass that inherits the message still reads it correctly. | Confusing warnings for subclass users (for example `vibench_agent` inheriting OpenCode's). |
| Sibling agents changed in the same PR say the same thing about the same preset. | One README says a preset has no replacement while the other names one. |

### Docs, catalog, and pairing

| Check | Finding when it fails |
|---|---|
| The adapter README states the harness version, capability level, supported pairings, and limits. The Fern tables in `integrate-existing-agents.mdx` and `add-or-update-a-harness.mdx`, and the paired benchmark's README harness table where one exists, match the adapter. | Users pick a pairing the adapter does not support. |
| An `allowed_agents` addition comes with a real pairing run, and the benchmark's owner reviews it. | Declared support with no evidence. |
| An adapter whose default config sets `resources_server: null`, or that has no `/run`, is registered in `scripts/add_legacy_agent_environment_servers.py` as add-harness Step 8 describes. | A stray `legacy_agent` relay beside the session Environment Server. |

### Tests

| Check | Finding when it fails |
|---|---|
| Each new test fails when the behavior it covers is reverted. Tests of startup side effects clear any `functools.cache` first, set worker env vars with `monkeypatch`, and filter `caplog` by the module logger's name. | Vacuous tests, or tests that pass because of order. |
| Unsupported seeds are covered with `nemo_gym.agents.testing.agent_session_conformance.check_agent_seed_rejected`. | The 422 contract is untested. |
| Tests that need an external tool skip when it is missing (`pytest.mark.skipif(shutil.which(...) is None, ...)`). | CI failures on hosts without the tool. |
| The PR ran `gym env test +entrypoint=responses_api_agents/<agent>` for each changed agent and each subclass, plus `tests/unit_tests/test_session_contract_repo.py` and every test in `tests/` that names a changed config (`grep -rn <config or agent name> tests/`). | Breakage in a suite the PR never ran. |
| Behavior-changing agent work shows real smoke rollouts with a model (`AGENTS.md`, "Quality bar"). A log-only or docs-only change may say `N/A` with a reason. | Green unit tests standing in for real runs. |

## Reporting

Use `[BUG]`, `[ASYNC]`, `[CONFIG]`, `[DOC]`, `[TEST]`, or `[GUIDELINE]`. For a finding about
user-facing guidance or a config condition, name the preset rows from the map where it goes wrong, so
the author can see which users are affected. Put the map itself in the leader's summary when it has
more than three rows.
