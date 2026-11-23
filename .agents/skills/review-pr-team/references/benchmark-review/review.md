# Benchmark PR review

Use this when a PR adds or changes a benchmark or environment: anything under `resources_servers/`,
`environment_servers/`, `benchmarks/`, `environments/` or `nemo_gym/data/tasks/harbor/`, a dataset or its
`prepare.py`, a benchmark's configs or pairings, or text that tells users how to run one.

Read this page first. Open a file under `details/` only when its item applies to the diff.

The code on main is the source of truth. The guides below describe the same contract, but parts of them are stale.
Where they disagree with the code, follow the code. The known disagreements are listed in
`details/known-conflicts.md`.

- `.agents/skills/add-benchmark/SKILL.md` and its `references/patterns.md`
- `fern/versions/latest/pages/environment-tutorials/`, `fern/versions/latest/pages/data/`,
  `fern/versions/latest/pages/contribute/environments/`
- `environment_servers/*/README.md`
- The repo's `AGENTS.md`: "Quality bar", "Environment Design Recommendations", "Communication & Async Patterns",
  "Sandbox"

## Before you review

**Classify the change.**

- Format: Gym (JSONL rows and a Resources Server) or Harbor (task directories run by the `harbor` Environment
  Server). Both are equal choices for a new dataset (decided 2026-11-11); review the PR against the format it
  chose.
- Path: typed taskset on `single_agent_turn`, flat rows on `single_agent_turn_legacy`, Harbor, or the `legacy_agent`
  relay.
- State: stateless; per-session state; lends a task sandbox; grades in a sandbox of its own.
- Kind: new benchmark, verifier or reward change, data change, config-only, or a shared change (base classes, an
  Environment Server, the Harbor loader).

**List who inherits it.** That means every config that loads the server, every harness in its `allowed_agents`,
every Environment Server that drives it, and every other caller of a shared helper the PR touches. Check each finding
against all of them, not only the pairing the PR tested. Recipe: `details/scope-and-docs.md`.

## Checklist

### 1. New work uses agent sessions, not the old `/run` path

- New benchmarks, pairings and configs run through an Environment Server that drives agent sessions:
  - `single_agent_turn` with a typed taskset for Gym-format rows (preferred);
  - `single_agent_turn_legacy` for Gym-format flat rows: the same session protocol, collected from flat rows
    (accepted for new work, decided 2026-11-11);
  - `harbor` for Harbor task directories;
  - or a benchmark's own Environment Server for another interaction pattern.
- On either `single_agent_turn` server, the bound agent and the Resources Server implement agent sessions. Nothing
  checks that the agent does (`AGENT_SESSION_AGENT_TYPES` is read only when an agent is swapped).
- Flag new work on:
  - a `legacy_agent` relay, including the one `gym env init` writes;
  - an agent without session support;
  - a deprecated agent: `opencode_sandboxed_agent`, `pi_sandboxed_agent`, `hermes_sandboxed_agent`,
    `miniswe_sandboxed_agent`, the `/run` path of `terminus_2_sandboxed_agent`, `harbor_agent`,
    `harbor_agent_general`.
- Exception (decided 2026-11-13): a benchmark that does not need to run with more than one harness may keep a
  `legacy_agent` relay. The usual case is an `external-agent-loop` benchmark whose own harness is the agent loop (for
  example #5743 `collab_intel`, where swapping the seat's harness would break parity with upstream). Ask only that
  the PR says it needs a single harness. Flag the relay when the benchmark should pair with several harnesses.
- Many shipped configs still use `legacy_agent`. Existing configs are not a precedent. Nothing in code warns on an
  explicit `legacy_agent`, and the scaffold still writes one, so the reviewer is the check.
- → `details/session-contract.md`

### 2. The Resources Server keeps the session contract

- **Seed:** accepts the typed `ResourcesSeedSessionRequest`, validates `task_data` with `seed_request_model`, and is
  idempotent for the same episode.
  - Refuses input no retry can fix with `ResourcesSeedRejectedError` (422).
  - Refuses a conflicting ID, a seed after verify and a seed after close with `ResourcesSessionConflictError` (409).
  - Never answers either with a plain exception (500).
- **Close:** releases by `resources_session_id` without a cookie, succeeds for unknown or closed IDs, raises when
  release fails, fences the ID. Stateful servers implement the three hooks instead of their own routes.
- **Verify:** a server whose score depends on session state answers a lost, closed or never-seeded session with
  `lost_session_verification()`, never a bare 0. A tool call for such a session raises `ResourcesSessionLostError`
  (404).
- **Test:** stateful servers call `check_resources_session_contract`.
- #5987 (open) adds `responses_create_params` to the seed. Only a server that reads the prompt at seed declares the
  field, and its PR says it depends on #5987 (decided 2026-11-11). Don't ask other servers to declare it.
- → `details/session-contract.md`

### 3. Sandbox lending: the Resources Server owns the task sandbox

- A task that needs a sandbox gets one from the Resources Server at seed, as `sandbox_access`, through
  `LentSandboxes`.
- `single_agent_turn` forwards only `sandbox_access`. The seed reply forbids extra fields, so a server that returns a
  `sandbox_handle` or `sandbox_descriptor` fails every episode at seed.
- The descriptor is `scope="operate"`, so on providers with leases the harness cannot destroy it.
- The server releases it at close and calls `release_all()` from its lifespan shutdown.
- `sandbox_config.ttl_s` is set above the episode deadline plus grading.
- A network policy is set with `network_access`.
- → `details/session-contract.md`

### 4. Pairing and tool grants are declared and proven

- `allowed_agents` lists the harnesses the benchmark supports. Each addition needs a real pairing run and the
  benchmark owner's review. Absent means any harness.
- Tools reach the harness through grants the Environment Server makes: `resources_tool_transports: [direct_http,
  mcp]` and `tool_servers`.
  - `mcp` needs `expose_tools_over_mcp: true`, and nothing checks that at startup.
  - Every grant is required. Check that each harness in `allowed_agents` consumes the grants the config makes.
- → `details/session-contract.md`

### 5. The reward measures the task, and only the task

- Every correct answer passes. The empty answer, a wrong answer and a gamed answer score 0.
- The agent cannot write the reward, the tests or the expected answer.
- The reward stays inside the manifest's `reward.range`.
- Judges and other helper models are named explicitly, never the policy.
- Scores are comparable with the published reference: code and data are pinned.
- Verifier and scorer correctness comes first in the shared rubric.

### 6. Gym-format rows are valid and typed

- Every row is a valid run request (`responses_create_params.input`).
- Task fields are declared in the server's `task_data.py` `TaskData` (no `extra="ignore"`), and every row carries a
  stable task ID.
- New rows carry no `agent_ref`.
- New datasets sit on the Resources Server. A `taskset` with an `environment_server_routes` entry to
  `single_agent_turn` is preferred; flat rows on `single_agent_turn_legacy` are accepted. Suggest the taskset; don't
  block on it.
- Images are base64 `data:` URIs of png, jpeg, gif or webp in `input_image`; URLs are refused.
- `gym dataset collate` does not catch a bad row; `gym env validate` does.
- → `details/gym-format.md`; for schema drift, also `../data-contracts.md`

### 7. Harbor-format tasks are well-formed and text-only

- Each task has `task.toml`, `instruction.md`, `tests/` (or a dataset `verifier.py`) and an image.
- `dataset.toml [gym]` has a license and `allowed_agents`. Per-harness settings live in overrides, not data.
- Rows and run configs are generated, never hand-edited.
- `gym dataset validate` passes. Tasks without a `solution/` are listed as `unvalidated` rather than failing it, so
  read that list.
- Transcripts must be text-only: an image anywhere in a harness's transcript masks every turn as
  `harbor:trajectory_export_failed` (#5871). That kind is namespaced, so recovery never retries it.
- `harbor_agent` and `harbor_agent_general` are deprecated (#4826).
- → `details/harbor-format.md`

### 8. Status codes say whether a retry can help

- **422:** bad or unsupported input (body validation, `ResourcesSeedRejectedError`, an unenforceable network policy).
  Terminal: the episode fails once.
- **409:** a session conflict (`ResourcesSessionConflictError`). Terminal.
- **404:** a tool call on a lost session (`ResourcesSessionLostError`).
- **5xx (and 408, 425, 429):** retryable. The whole episode is dispatched again on a new sandbox.
- What to flag:
  - a deterministic failure that raises a plain exception, a `ValidationError` raised inside a handler, or an
    unmapped `UnsupportedContentError`: each becomes a retried 500;
  - a transient failure answered with another 4xx: never retried;
  - sandbox errors caught and turned into a timeout, an empty result or a reward. They pass through.
- → `details/serving.md`

### 9. Know what gets retried, and how often

- `ServerClient` retries exceptions only, never an HTTP status, with a fixed 0.5 s backoff. Refused or dropped
  connections retry until the caller's bound; other exceptions get 3 tries on external calls.
- The Environment Server never reruns an episode in process (Harbor retries image pulls).
- The collector redispatches non-terminal failures on resume, up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3).
- A masked row is retried only when its `failure_kind` is registered (not namespaced) and either evaluation-side or
  marked `failure_outcome_action: retry`.
- Sandbox-graded benchmarks retry inconclusive verification on a fresh sandbox a bounded, configurable number of
  times (`inconclusive_verification_retries`; per server, not in core), then mask.
- Ask which layer retries each new failure, and for a test of it.
- → `details/serving.md`

### 10. Mask infrastructure failures; never score them as 0

- Only `mask_sample: true` removes a row from the score. `evaluation_completed: false` alone is a scored 0, and ten
  SWE servers on main end that way. In new code, flag it as a `RISK` (decided 2026-11-11): ask for
  `mask_sample: true` beside it, but don't block on it alone.
- Not the model's 0:
  - a sandbox that would not start or answer;
  - a lost session;
  - a judge or model-server outage;
  - a verifier that could not run;
  - a transcript the server could not read.
- Each of those is masked with a `failure_kind` from the shared vocabulary, ideally as a typed `NgFailureEvent`
  through `observe_failure` and the outcome policy.
- A real task failure still scores 0. Ask which side of the line each new ending is on.
- → `details/serving.md`

### 11. Async, bounded, aiohttp, cookies

- Endpoints are `async`.
- Subprocess, sandbox and judge calls are bounded by `asyncio.Semaphore`, and subprocess output is decoded with
  `errors="replace"`.
- HTTP goes through `nemo_gym.client.http_client.request()` or `ServerClient`, never `httpx.AsyncClient`. Nothing
  enforces this, and at least seven older files break it.
- Session cookies reach every downstream call.
- Startup work is idempotent across workers.
- → `details/serving.md`

### 12. Every sandbox the server starts is stopped on every path

- Grading sandboxes:
  - resolve the provider from config;
  - deliver files through `SandboxSpec.files` or `start_with_setup()`;
  - quote every value that goes into a command;
  - check `return_code` and `error_type` on every `exec`.
- Stop in `try/finally`, with the stop bounded by a timeout or shielded from cancellation. Key per-session sandboxes
  by session ID, pop only after `stop()` succeeds, and stop leftovers in the lifespan shutdown.
- Set `ttl_s`. It defaults to `None`, and nothing checks it.
- → `details/serving.md`

### 13. Configs are validated, not unit-tested

- A config-only PR shows `gym env validate --config <file>` (or `gym env validate <benchmark>` for a manifest) and a
  real pairing run.
- Do not ask for unit tests that resolve the YAML and assert its bindings.
- The main pairing config is named after the benchmark, not after the Environment Server path:
  `configs/<name>.yaml`, never `<name>_single_agent_turn.yaml` (decided 2026-11-13). A config that keeps the
  `legacy_agent` relay beside it is `<name>_legacy_agent.yaml`, as nine servers on main already do. The rename also
  updates every reference: the benchmark's `config.yaml`, tests, READMEs and the root README row (rerun the
  `update-readme-table` hook rather than editing the row).
- Every `resources_servers` block under `resources_servers/*/configs/`, override blocks included, says
  `verified: false`. Otherwise the `add-verified-flag` hook rewrites the file with `yaml.dump` and drops its SPDX
  header and comments.
- The author may flip `verified: true` in their own PR once the add-benchmark exit checks pass (decided
  2026-11-11). The PR links the run folder and `quality_summary.json` and shows the item 14 baselining numbers. A
  flip without them is a finding. A manifest's `experimental: false` still needs Gym team approval.
- → `details/configs-and-evidence.md`

### 14. Evidence before trust

- Behaviour-changing environment work shows real-model smoke rollouts. Someone inspected both the agent's and the
  verifier's behaviour. A scripted model is not enough.
- A new benchmark meets the add-benchmark baselining bar (decided 2026-11-11). It shows `gym eval run` +
  `gym eval profile` numbers from the recommended model suite, instruct and thinking. Open models score 30% or more,
  closed models score at or above them, and failure cases were read. The top open model is repeated until variance
  is under 1%. A miss comes with what the author investigated.
- A new training environment needs only the real-model smoke (decided 2026-11-11). Don't ask for a reward profile
  or a training run.
- An external benchmark reproduces its published numbers before and after integration.
- A verifier or reward change shows before/after numbers.
- CI checks only that the five example rollouts exist.
- → `details/configs-and-evidence.md`

### 15. Tests fail when the fix is reverted

- Each claimed behaviour and each new ending has a test that fails on revert.
- The verifier is tested on the known-good answer, the known-bad answer and each infrastructure failure: masked, not
  0.
- Stateful servers run `check_resources_session_contract`.
- Every wait is bounded.
- A missing external tool skips, or is auto-installed with `ensure_<tool>()`; it never fails silently.
- → `details/configs-and-evidence.md`

### 16. Claims match the code; shared changes reach every caller

- The README, docstrings, Fern pages and the PR description say what the code does: the path, the harnesses, the
  limits, the score impact.
- A change to a shared base class, an Environment Server, the Harbor loader or a shared grading helper names the
  other benchmarks it reaches, and states the score impact or says there is none.
- → `details/scope-and-docs.md`

### 17. Public text

- Keep cluster names, internal hosts, internal paths and ticket ids out of the PR title, body, commits and code
  comments. Guardwords CI also fails on the PR description.
- The title follows the Conventional Commit format, and commits are signed off.
- Adapted external code records its source, commit SHA and license.
- → `details/configs-and-evidence.md`

## Don't flag

- The choice of Gym or Harbor format for a new dataset. Both are equal, although `gym env init` prints that new
  datasets should use `gym dataset init`.
- Existing configs on `legacy_agent` that the PR does not touch.
- A new config on `single_agent_turn_legacy` with flat rows, when its agent implements sessions.
- A server that doesn't read the prompt at seed and doesn't declare `responses_create_params` on its seed model
  (#5987).
- A PR that follows the code side of a conflict in `details/known-conflicts.md`.
- A missing root README row for a config whose datasets sit on the Resources Server or Environment Server block: the
  `update-readme-table` hook skips those.
- Missing per-config unit tests.
- Style nits that ruff and pre-commit already enforce.

## How to report

- One finding per issue.
- For each: the item number, `file:line`, the concrete input or state that breaks, and what happens to the score or
  the run.
- Name the configs, harnesses or benchmarks affected.
- Say whether the rule is enforced by code on main, written in a guide, or only a review convention (items 1, 11 and
  12 are mostly convention).
- Example review comments, by item: `details/examples.md`.
