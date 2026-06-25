---
name: review-pr-team
description: >
  Agent-team-based parallel code review for NVIDIA-NeMo/Gym pull requests.
  Spawns specialized agents (core-library expert, component experts for
  resources servers / agents / models, bug finder, test agent, devil's
  advocate, comment reviewer, validation runner) that coordinate via the
  shared Task list and direct messaging. The leader orchestrates, collates
  ALL findings, and presents to the user for approval before posting.
  Triggered by: "team review PR", "review PR with agents", "parallel PR review".
argument-hint: "<pr-number>"
allowed-tools:
  - AskUserQuestion
  - Bash
  - Read
  - Glob
  - Grep
  - Agent
  - TeamCreate
  - TeamDelete
  - SendMessage
  - TaskCreate
  - TaskList
  - TaskGet
  - TaskUpdate
  - TaskStop
  - WebFetch
---

# Agent Team PR Review — NVIDIA-NeMo/Gym

Review a pull request using a coordinated team of specialized agents.

**Repo**: `NVIDIA-NeMo/Gym`

> For reviewing your *local working diff* (uncommitted changes), prefer the
> built-in `/code-review` skill. This skill is for **GitHub PRs** and uses a
> team of agents for broad, parallel coverage.

## Coordination model (this harness)

### Side-panel layout (preferred when available)

If `TeamCreate` appears in your available tools, use it — each teammate opens in its
own tmux pane, giving the user a side-panel view with the leader on the left and
agents on the right.

**Requirements** (already configured in this repo's `.claude/settings.json`):
```json
{ "enableExperimentalFeatures": { "agentTeams": true }, "teammateMode": "tmux" }
```

With `TeamCreate` available:

- **Spawn** each Wave 1 agent with `TeamCreate` (one call per agent). Each teammate
  appears in its own pane. Pass the full prompt as the `initialMessage`.
- **Communicate** with running teammates via `SendMessage` (by name or ID).
- **Tear down** teammates after collation with `TeamDelete` (see Phase 7).
- Agent findings arrive via `SendMessage` from the teammate back to the leader.

### Background-task fallback (when TeamCreate is unavailable)

If `TeamCreate` is NOT in your tool list, fall back to the `Agent` tool:

- **Spawn** each wave-1 agent with the `Agent` tool using `run_in_background: true`
  and a stable `description`/name. You are notified when each finishes; its final
  message is the agent's findings (returned to you, not shown to the user).
- **Continue / question** an already-spawned agent with `SendMessage` addressed to
  that agent's ID or name — its context is preserved.

### Common to both modes

- Use `TaskCreate` / `TaskUpdate` / `TaskList` to maintain a shared, visible task
  list so the user can follow progress. The leader (you) owns orchestration; agents
  do NOT poll — they report back on completion and you push follow-ups.
- Pick `subagent_type: "general-purpose"` for the analysis agents (they need Bash,
  Read, Grep, Glob, WebFetch, SendMessage). Use `Explore` only for read-only scouting.

**Sandbox**: Pass `dangerouslyDisableSandbox: true` on Bash calls that need the
network (`gh`, `git fetch`, `uv sync`) or that run in worktree/container
environments where bubblewrap fails to initialize
(`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`). When in doubt for
this skill's `gh`/`git`/`uv` calls, set it to avoid a wasted round-trip.

---

## Phase 0: Parse & Validate

Extract `$PRNUM` from `$ARGUMENTS`. A PR number is required.

```bash
gh pr view $PRNUM --repo NVIDIA-NeMo/Gym --json number
```

If invalid, ask the user for a valid PR number.

---

## Phase 1: Setup & Context

### 1.1 Checkout PR

```bash
git fetch origin pull/$PRNUM/head:pr-$PRNUM-team-review
git checkout pr-$PRNUM-team-review
```

(NeMo Gym has no git submodules — no `git submodule update` needed.)

### 1.2 Gather PR metadata (parallel)

Run these in parallel:

```bash
# PR metadata (include mergeable to detect conflicts)
gh pr view $PRNUM --repo NVIDIA-NeMo/Gym \
  --json title,body,author,baseRefName,headRefOid,labels,files,comments,reviews,reviewRequests,mergeable,mergeStateStatus

# Full diff
gh pr diff $PRNUM --repo NVIDIA-NeMo/Gym

# Inline review comments
gh api repos/NVIDIA-NeMo/Gym/pulls/$PRNUM/comments
```

Record: `$TITLE`, `$AUTHOR`, `$BASE_BRANCH`, `$HEAD_SHA`, changed files list, existing
comments, existing reviews.

**Merge conflict check**: If `mergeable` is `"CONFLICTING"` or `mergeStateStatus` is
`"DIRTY"`, include a prominent note asking the author to rebase onto `$BASE_BRANCH` and
resolve conflicts. Add this as the first item in the review body.

**Read the PR description (`body`) carefully.** It contains the author's intent,
motivation, and test plan. Parse it for linked issues (`Fixes #123`, `Closes #456`,
`Related: #789`). For each, fetch:

```bash
gh issue view <ISSUE_NUM> --repo NVIDIA-NeMo/Gym --json title,body,comments,labels
```

The PR description + linked issues + diff together form the full context. Pass this
context to every agent so they understand *why* the change is being made.

### 1.2a Reward-profiling / baselining evidence check

NeMo Gym benchmarks are baselined before they're trusted. New `verified: true` configs
and changes to verifier/reward logic should be backed by **reward-profiling evidence**.

- **New resources server / benchmark**: the author should report baseline reward numbers
  from a profiling run (e.g. `ng_collect_rollouts` + `ng_reward_profile`, the
  `*_aggregate_metrics.json` summary), demonstrating the verifier produces sensible
  rewards on known-good and known-bad trajectories.
- **Verifier / reward modification**: expect a before/after comparison or proof the
  reward distribution didn't silently shift.
- A pre-commit hook (`add-verified-flag`) sets new resources-server YAMLs to
  `verified: false`. If a PR flips a config to `verified: true`, it MUST cite the
  baselining evidence. Flag missing evidence as a `[BASELINE]` finding with a specific
  ask (which profiling numbers to provide).

Record `$BASELINE_EVIDENCE_FOUND` (yes/no/not-applicable) and pass to the
`resources-server-expert` and `gym-core-expert`.

### 1.2b Documentation check

NeMo Gym docs live in `fern/` (see the `nemo-gym-docs` skill) and the resources-server
table in root `README.md` is auto-maintained by the `update-readme-table` pre-commit hook.

1. New user-facing feature / benchmark → check `fern/versions/latest/pages/` for a
   matching page or tutorial. If a notable feature is undocumented, flag `[DOC]`.
2. New resources server → confirm the README table row exists (the hook adds it; if the
   PR hand-edited the table or the row is missing, flag it).

Record `$HAS_DOCS` (yes/no) and pass to agents.

### 1.3 Determine touched components

```bash
gh pr diff $PRNUM --repo NVIDIA-NeMo/Gym --name-only
```

Map top-level paths to which experts to spawn (conditional — only spawn an expert when
its area is touched):

| Path prefix | Spawn agent |
|-------------|-------------|
| `nemo_gym/` (core library, CLI, server_utils) | `gym-core-expert` |
| `resources_servers/` | `resources-server-expert` |
| `responses_api_agents/` | `agent-harness-expert` |
| `responses_api_models/` | `model-server-expert` |
| `tests/`, any `*/tests/` | `test-agent` (always spawn) |
| `fern/` | fold into `gym-core-expert` (docs review) |

`bug-finder`, `test-agent`, and `comment-reviewer` are always spawned. If only one
component area is touched, still spawn `gym-core-expert` as the generalist owner of the
cross-cutting guidelines.

### 1.4 Read root context (load coding guidelines)

- Read `CLAUDE.md` from the repo root — it is the canonical source of NeMo Gym
  conventions (async patterns, aiohttp-not-httpx, error handling, pre-commit hooks,
  code style, HPC gotchas).
- Glob `.claude/skills/*/SKILL.md` — read every match (e.g. `add-benchmark`,
  `nemo-gym-debugging`, `nemo-gym-docs`, `nemo-gym-reward-profiling`,
  `nemo-gym-pivot-datasets`). Do NOT hardcode skill names; skills may be added or renamed.
- Glob `.claude/review-memory/*.md` — if any exist, read them.

### 1.5 Detect local test capability

```bash
# GPU presence (some servers/tests need GPUs, e.g. vllm_model, functional tests)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
# Is there a populated venv to reuse for ng_dev_test?
ls -d .venv 2>/dev/null
```

Set `$GPU_TESTING_AVAILABLE` (true/false). Most NeMo Gym unit tests are CPU-only
(`pytest tests/unit_tests/`); functional GPU tests live in
`tests/functional_tests/L2_Functional_Tests_GPU.sh`. Agents note "not verified — no GPU
environment" when a check needs GPUs.

### 1.6 Create the shared task list

Use `TaskCreate` to register every task below so the user can watch progress. Update
status with `TaskUpdate` as waves complete.

---

## Phase 2: Create Tasks & Spawn Agents

### 2.1 Tasks

**Wave 1 (parallel, no blockers):**

| Task | Owner | Description |
|------|-------|-------------|
| `analyze-core` | `gym-core-expert` | Analyze `nemo_gym/` + cross-cutting guidelines (async, HTTP, config, docs). Report ALL findings. |
| `analyze-resources` | `resources-server-expert` | (conditional) Verifier/state/tool/YAML correctness, `verified` flag, baselining. |
| `analyze-agents` | `agent-harness-expert` | (conditional) Agent harness correctness, Responses API format, `/run` async. |
| `analyze-models` | `model-server-expert` | (conditional) Model server correctness, chat_completions/responses, token-id handling. |
| `review-existing-comments` | `comment-reviewer` | Review all PR comment threads; identify responses needed. |
| `review-and-suggest-tests` | `test-agent` | Review tests; suggest+verify new tests; run locally where possible. |
| `scan-for-bugs` | `bug-finder` | Independently scan the diff for bugs; write tests when uncertain. |

**Wave 2 (blocked by ALL Wave 1 tasks):**

| Task | Owner | Description |
|------|-------|-------------|
| `challenge-findings` | `devil-advocate` | Stress-test ALL Wave 1 findings. Up to 2 rounds per agent. Waits for the leader to push consolidated findings via `SendMessage`. |

**Wave 3 (leader):**

| Task | Owner | Description |
|------|-------|-------------|
| `collate-review` | leader | Merge findings, apply verdicts, deduplicate, present to user. |

**Wave 4 (conditional, after user approves posting):**

| Task | Owner | Description |
|------|-------|-------------|
| `run-validation` | `validation-runner` | (only if requested) Run lint + unit tests + targeted server tests locally; report results. |

Set `challenge-findings` blockedBy all Wave 1 task IDs; `collate-review` blockedBy
`challenge-findings`.

### 2.2 Spawn agents

**If `TeamCreate` is available** (side-panel mode): call `TeamCreate` once per agent,
passing the full agent prompt as `initialMessage` and a short unique `name`. Each
teammate opens in its own tmux pane. Spawn all Wave 1 agents before proceeding.

**If `TeamCreate` is unavailable** (fallback): spawn all Wave 1 agents in a single
message with multiple `Agent` calls (`run_in_background: true`,
`subagent_type: "general-purpose"`), each with a unique `description`/name.

---

## Agent Prompt Specifications

### Common preamble (include in EVERY agent prompt)

```
You are a member of the PR #$PRNUM review team for NVIDIA-NeMo/Gym (NeMo Gym).

PR: #$PRNUM "$TITLE" by $AUTHOR (base: $BASE_BRANCH, head: $HEAD_SHA)

PR Description:
$PR_BODY

Related Issues:
$RELATED_ISSUES (title, body, key comments for each linked issue)

Instructions:
- Read and understand the PR description and related issues FIRST — they explain the
  author's intent, motivation, and test plan. Review the code in that context.
- Claim your task with TaskUpdate(status="in_progress") and mark it completed when done.
- Use Read, Glob, Grep for local lookups (faster than gh CLI). The PR is checked out
  on branch pr-$PRNUM-team-review.
- Do NOT git checkout other commits — use `git show <sha>:<path>` for history lookups.
- Include GitHub permalinks in ALL findings:
  https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/<path>#L<line>
- External reference claims: when claiming "library X does Y" or citing upstream/OpenAI
  Responses API behavior, fetch the actual source (WebFetch on a raw GitHub URL at a
  pinned SHA, or curl) and link the exact line. Never claim from memory alone.
- Report ALL findings — never cap, truncate, or summarize. Every issue must be reported.
- When done, mark your task completed with TaskUpdate and send your findings to the
  leader via SendMessage. Your final message IS your findings (raw data, not prose for
  a human).
```

### `gym-core-expert`

```
You are the NeMo Gym core-library and conventions expert.

Scope: nemo_gym/ (BaseServer/SimpleServer hierarchy, server_utils, ServerClient, CLI,
Hydra config), root config files, and fern/ docs touched by this PR.

FIRST: load ALL guidelines:
1. Read CLAUDE.md at the repo root.
2. Glob `.claude/skills/*/SKILL.md` — read every match. Do NOT hardcode skill names.

Check the diff against NeMo Gym's cross-cutting rules (cite CLAUDE.md):
1. HTTP: async HTTP MUST go through nemo_gym.server_utils.request() (global aiohttp
   client). Flag any httpx.AsyncClient usage — httpx/httpcore has O(n^2) pooling that
   hangs at high concurrency. If wrapping a library that uses httpx internally, it must
   use an aiohttp adapter (see resources_servers/tavily_search/app.py).
2. Async: the /run endpoint must be async; bound concurrent subprocess/external calls
   with asyncio.Semaphore.
3. Sessions: stateful environments must propagate request.cookies through downstream calls.
4. Ray in async: `await future` (Ray futures are awaitable); never `ray.get()` in async.
5. Subprocess output decoded with errors="replace".
6. Guard optional nested fields: (body.field or {}).get("key", default).
7. External-tool auto-install pattern (setup_<tool>.py / ensure_<tool>() in
   model_post_init + conftest pytest_configure) when a new external tool is required.
8. Code style: line length 119, Python 3.12+, double quotes, isort; ruff clean.
9. Coverage: test coverage must stay >= 96%.
10. Pre-commit hooks that auto-modify files (add-verified-flag, update-readme-table,
    ruff-format) — confirm the PR is consistent with their output, not fighting them.

Categorize findings as [BUG], [ASYNC], [GUIDELINE], [CONFIG], [DOC], [DOCSTRING] with
file:line and permalink.

DOCSTRING REVIEW: for any new/modified public function/class/method missing a docstring,
draft it as a GitHub ```suggestion``` block; confer with the relevant component expert
via SendMessage to verify params/returns before finalizing. Tag [DOCSTRING].

DOCS: if a notable user-facing feature is undocumented in fern/, or the README resources
table row is missing/hand-edited, tag [DOC].

You also answer questions from other agents via SendMessage.
```

### `resources-server-expert` (conditional)

```
You are the resources-server (verifier + state + tools) expert.

Scope: resources_servers/ paths in the diff.

FIRST: Read CLAUDE.md and the `add-benchmark` and `nemo-gym-reward-profiling` SKILL.md
files (Glob `.claude/skills/*/SKILL.md`).

Tasks:
1. Verify the verify() implementation: correct reward semantics, graceful handling of
   tool failures and bad model outputs (no crashes — meaningful error responses), and
   per-task state isolation.
2. Verify YAML config wiring (Hydra): resources server + agent + model composed correctly.
3. `verified` flag: new server YAMLs should be `verified: false` (set by the
   add-verified-flag hook). If the PR sets `verified: true`, it MUST cite reward-profiling
   / baselining evidence — flag missing evidence as [BASELINE] with a specific ask.
4. Dataset rows: responses_create_params.input is valid Responses API format;
   verifier_metadata carries the task-specific data the verifier needs.
5. Async/subprocess/HTTP rules (see core expert) for any I/O the server does.

Report findings with file:line and permalink, categories [BUG], [CONFIG], [BASELINE],
[GUIDELINE], [TEST]. Answer rl/agent/test agents' questions via SendMessage.
```

### `agent-harness-expert` (conditional)

```
You are the agent-harness expert.

Scope: responses_api_agents/ paths in the diff.

FIRST: Read CLAUDE.md. Read the base class: tests/unit_tests/test_base_responses_api_agent.py
and the SimpleResponsesAPIAgent contract (implement responses(), run()).

Tasks:
1. Verify the /run endpoint is async and handles tool failures / bad model outputs
   gracefully (no crashes).
2. Verify Responses API trajectory handling (multi-turn, tool-calling) is correct.
3. Verify HTTP goes through the global aiohttp client and cookies propagate.
4. Check concurrency control (asyncio.Semaphore) for any subprocess/external calls.

Report findings with file:line and permalink; categories [BUG], [ASYNC], [GUIDELINE],
[TEST]. Answer other agents via SendMessage.
```

### `model-server-expert` (conditional)

```
You are the model-server expert.

Scope: responses_api_models/ paths in the diff.

FIRST: Read CLAUDE.md and tests/unit_tests/test_base_responses_api_model.py
(SimpleResponsesAPIModel contract: chat_completions(), responses()).

Tasks:
1. Verify provider adapters return the common format correctly and manage token IDs
   needed for training.
2. Verify the Anthropic/OpenAI/vLLM conversion paths (see test_anthropic_converter*.py)
   if touched — request/response field mapping, streaming, tool calls.
3. HTTP/async rules (global aiohttp client; no httpx.AsyncClient; await Ray futures).

Report findings with file:line and permalink; categories [BUG], [ASYNC], [GUIDELINE],
[TEST]. Answer other agents via SendMessage.
```

### `test-agent`

```
You are the test reviewer and test author.

Scope: tests/ and any test files in the PR.

FIRST:
1. Glob `.claude/skills/*/SKILL.md` — read relevant ones (testing guidance in CLAUDE.md).
2. Read CLAUDE.md "Common Commands for Building & Testing Environments".
3. Read tests/conftest.py to understand fixtures and marks.

TEST COMMANDS (NeMo Gym):
  Core unit tests:        pytest tests/unit_tests/ -x
  Single unit file:       pytest tests/unit_tests/test_<x>.py -x
  Server tests (no venv): ng_dev_test +entrypoint=resources_servers/<name>
  Server tests (venv):    ng_test +entrypoint=resources_servers/<name>   (slow first run)
  Functional GPU tests:   tests/functional_tests/L2_Functional_Tests_GPU.sh  (GPUs only)
  Lint:                   ruff check --fix . ; ruff format .
On long Lustre paths set RAY_TMPDIR=/tmp before ng_test / ray.init().

GPU WORK: if $GPU_TESTING_AVAILABLE=true you may run GPU-needing tests; otherwise run
only CPU-only checks (grep, read, collect-only, CPU unit tests) and note "not verified —
no GPU environment".

Tasks:
1. REVIEW existing tests: correctness, edge cases, assertions, cleanup, skip guards for
   missing external tools (pytest.mark.skipif(shutil.which("tool") is None, ...)).
2. COVERAGE: the repo requires coverage >= 96%. Before suggesting a new test, Grep
   tests/ to confirm the path isn't already covered; if covered, note the existing
   test file:line instead of suggesting a redundant test.
3. SUGGEST tests where coverage is genuinely missing; match weight to risk and follow
   nearby test conventions. Provide as ```suggestion``` blocks.
4. VERIFY: run every suggested test locally with the matching command above. Only
   include a suggestion if it PASSES. If it fails, fix and re-run.

Ask component experts for domain knowledge via SendMessage.
```

### `comment-reviewer`

```
You are the comment reviewer.

Tasks:
1. Read ALL existing PR comment threads (in the PR metadata).
2. For each, decide if action is needed (author replied and needs an answer; a coworker
   raised a point to affirm/challenge; or no action).
3. For threads needing a response, get technical context from the relevant expert via
   SendMessage before drafting.
4. Every response MUST include permalink references.

Report: list of (thread_comment_id, action, draft_reply_text, permalinks).
```

### `bug-finder`

```
You are the bug finder. Scan the entire diff for bugs beyond what other experts found.

FIRST: Read CLAUDE.md; Glob `.claude/skills/*/SKILL.md`.

Tasks:
1. Scan for: logic errors, None refs, race conditions, type errors, missing imports,
   incorrect API usage, resource leaks, off-by-one, and NeMo Gym-specific hazards
   (httpx.AsyncClient usage, blocking calls in async endpoints, ray.get() in async,
   undecoded subprocess output, unguarded nested-field access, missing Semaphore).
2. Read surrounding context for each changed file.
3. When uncertain, write a self-contained test to validate (run with `pytest -x` or
   `ng_dev_test` as appropriate; note GPU-not-available where relevant).
4. Delegate domain questions to expert agents via SendMessage.
```

### `devil-advocate`

```
You are the devil's advocate. Stress-test ALL findings from ALL other agents.

Wait for the leader to send you the consolidated Wave 1 findings via SendMessage. Do NOT
poll TaskList in a loop — stay idle until that message arrives.

For EACH finding:
1. Demand the source/reference/permalink. If missing, challenge it.
2. Independently verify by reading the actual code yourself.
3. Challenge whether it matters — real problem or noise?
4. UP TO 2 ROUNDS of challenge per agent (via SendMessage), then render a final verdict.

ALSO challenge whether the PR is even needed (trivial/whitespace/formatting-only with no
functional change).

Report ALL verdicts — each finding gets: CONFIRMED (with justification),
DISPUTED (with what's wrong), or DOWNGRADED (over-rated severity).
```

### `validation-runner` (spawned conditionally, after approval)

```
You are the local validation runner.

Tasks:
1. SAFETY CHECK: inspect the diff for malicious patterns (crypto miners, data
   exfiltration, credential theft, supply-chain attacks). If anything is suspicious,
   report to the leader via SendMessage and STOP.
2. If clean, run locally and report results to the leader:
   - pre-commit run --files <changed files>   (or: ruff check . ; ruff format --check .)
   - pytest tests/unit_tests/ -x
   - For each touched resources server: ng_dev_test +entrypoint=resources_servers/<name>
     (set RAY_TMPDIR=/tmp on long paths). Skip GPU-only servers if no GPU.
3. Report pass/fail with the relevant output excerpts. The leader stages results for
   user review before posting.
```

---

## Phase 3: Collation (leader)

### Wave 1 → Wave 2 handoff

When ALL Wave 1 tasks are completed (you receive each agent's completion with findings),
send a single consolidated `SendMessage` to `devil-advocate` containing the full set of
Wave 1 findings grouped by source agent. This push unblocks the otherwise-idle
devil-advocate.

### Brokering devil-advocate challenges

When devil-advocate challenges another agent, that agent may go idle. The leader brokers:
on a devil-advocate peer-DM summary, `SendMessage` a nudge to the target agent to check
its inbox and respond — preventing deadlocks.

### Tone guidelines

Review comments represent the team — constructive and helpful, especially for community
contributors:

- **Ask, don't accuse**: "It would help to include baseline reward numbers" not "The PR
  contains no evidence."
- **Suggest, don't demand**: "Consider adding..." / "This could be improved by..."
- **Don't single out the author**; frame references positively.
- **Acknowledge the work** before listing issues.
- **Be specific**: "Could you share the aggregate reward on the eval split before/after?"
  beats "Please add numbers."

After all Wave 2 agents complete:

1. Gather ALL findings. Categories: `[BUG]`, `[ASYNC]`, `[TEST]`, `[GUIDELINE]`,
   `[CONFIG]`, `[DOC]`, `[DOCSTRING]`, `[BASELINE]`.
2. Apply devil-advocate verdicts: remove DISPUTED, adjust DOWNGRADED.
3. Deduplicate: same file + same line range + same core issue = one finding.
4. Confidence threshold: discard anything scoring below 80.
5. **Show ALL surviving findings** — no caps, no "top N", no summarization.
6. For each finding, construct the comment:
   - Concise (2-3 sentences max), starting with a permalink to the code.
   - Evidence permalinks for any claim about external/library code (pin the SHA).
   - Include a ```suggestion``` block when the fix is known.
7. Incorporate test-agent's verified suggestions and comment-reviewer's thread drafts.

---

## Phase 4: Preview & Confirm

Display ALL findings using the card layout, grouped by severity:

```markdown
## Review: PR #$PRNUM — $TITLE
by @$AUTHOR | <count> files changed | <count> agents

### Critical (<count>)

> **BUG** `nemo_gym/server_utils.py:42` [confidence: 95]
> <concise description>
> [view on GitHub](https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/nemo_gym/server_utils.py#L42)
>
> ```suggestion
> corrected code here
> ```
>
> _bug-finder, confirmed by devil-advocate_

### Suggestions (<count>)

> **ASYNC** `resources_servers/foo/app.py:15` [confidence: 88]
> Uses httpx.AsyncClient — route through nemo_gym.server_utils.request() (CLAUDE.md).
> [view on GitHub](https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/resources_servers/foo/app.py#L15)
>
> _gym-core-expert_

> **BASELINE** `resources_servers/foo/configs/foo.yaml:1` [confidence: 85]
> Config sets verified: true without cited reward-profiling evidence.
> [view on GitHub](https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/resources_servers/foo/configs/foo.yaml#L1)
>
> _resources-server-expert_

> **DOCSTRING** `nemo_gym/x.py:30` [confidence: 90]
> `function_name` missing docstring.
> [view on GitHub](https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/nemo_gym/x.py#L30)
>
> ```suggestion
> def function_name(self, ...):
>     """Brief description.
>
>     Args:
>         param: Description.
>     """
> ```
>
> _gym-core-expert_

> **TEST** `tests/unit_tests/test_foo.py` [confidence: 88]
> Suggest test for <feature> (verified passing locally).
> [view on GitHub](https://github.com/NVIDIA-NeMo/Gym/blob/$HEAD_SHA/tests/unit_tests/test_foo.py)
>
> ```suggestion
> def test_feature():
>     ...
> ```
>
> _test-agent_

### Threads (<count>)

> **Reply** to @author on `file.py:10`
> <draft reply with permalinks>
>
> _comment-reviewer_

_or: No responses needed._

### Devil's Advocate
<N> confirmed | <N> disputed | <N> downgraded
PR necessity: **<verdict>** — <one-line justification>

---
_<N> findings filtered (scored <80)_
```

Then use `AskUserQuestion`:
- **(1) Stage all** — stage everything as a PENDING review for preview on GitHub
- **(2) Discuss individually** — iterate through each item before staging
- **(3) Cancel** — do nothing
- **(4) Stage + Run validation** — stage review AND run local lint/tests via
  `validation-runner`

If "Discuss individually", walk each finding: approve, edit text, or skip; then stage
approved items as PENDING.

---

## Phase 5: Post Review

**IMPORTANT**: Post everything as a single review. Never post separate standalone
comments via the issues API.

### Known GitHub limitation: PENDING review bodies get wiped on UI submit

When a PENDING review is created via the API with a `body`, the GitHub UI's "Submit
review" dialog defaults its text area to empty and **overwrites** the API-set body on
submit. Only inline `comments` survive. **Fix**: don't ask the user to submit from the
UI — submit via API (below).

### Step 1: Create PENDING review with body + inline comments

Omit `event` to create a PENDING review. Add `_Generated by Claude Code_` at the end of
the body.

**All actionable findings MUST be inline comments**, not body text. The body is for
general context (PR summary, merge-conflict note, agent count, non-actionable
observations). Tie "general" findings to the most relevant diff line:
- Bug in a file not in the diff → comment on the diff line that introduces the
  requirement; reference the external file in the body.
- Missing test coverage → comment on the most relevant test file in the diff, listing
  untested functions with permalinks.

Use Python `json.dump` to build the review JSON (avoids shell-escaping issues with
backticks/markdown):

```bash
cat <<'REVIEW_JSON' > "$TMPDIR/review.json"
{
  "commit_id": "$HEAD_SHA",
  "body": "<brief summary — merge conflict note, agent count, etc.>\n\n_Generated by Claude Code_",
  "comments": [
    {"path": "<file>", "line": <line>, "side": "RIGHT", "body": "[`<file>:<line>`](<permalink>)\n\n<comment with evidence permalinks>"}
  ]
}
REVIEW_JSON

gh api repos/NVIDIA-NeMo/Gym/pulls/$PRNUM/reviews --method POST --input "$TMPDIR/review.json"
```

Save the returned review `id` as `$REVIEW_ID` and `node_id` as `$REVIEW_NODE_ID`. Print
the pending review URL:

```
https://github.com/NVIDIA-NeMo/Gym/pull/$PRNUM#pullrequestreview-$REVIEW_ID
```

If the POST is blocked by permissions, print the exact command for the user to run
(prefixed with `!`), then parse the returned `id`.

### Step 1a: Add more comments to a PENDING review

Use the GraphQL `addPullRequestReviewThread` mutation (REST cannot add comments to an
existing pending review). Use the review **node_id** (e.g. `PRR_kw...`), NOT the integer
id. The mutation is `addPullRequestReviewThread` (not `addPullRequestReviewComment`).

```python
import json
gql = {
  "query": ("mutation($reviewId: ID!, $body: String!, $path: String!, $line: Int!) {"
            " addPullRequestReviewThread(input: {pullRequestReviewId: $reviewId,"
            " body: $body, path: $path, line: $line, side: RIGHT})"
            " { thread { id comments(first:1){ nodes { id url } } } } }"),
  "variables": {"reviewId": "$REVIEW_NODE_ID", "body": "<comment>", "path": "<file>", "line": 42},
}
json.dump(gql, open("$TMPDIR/add_comment.json", "w"))
```
```bash
gh api graphql --input "$TMPDIR/add_comment.json"
```

### Step 1.5: Preview and confirm

Show the user the pending-review URL and ask them to preview on GitHub. Use
`AskUserQuestion`: **Publish**, **Edit comments**, **Cancel**.

- **Edit comments**: ask which comment, update via
  `gh api repos/NVIDIA-NeMo/Gym/pulls/comments/$COMMENT_ID --method PATCH -f body="..."`,
  then re-ask.
- **Cancel**: delete the pending review
  (`gh api repos/NVIDIA-NeMo/Gym/pulls/$PRNUM/reviews/$REVIEW_ID --method DELETE`) and stop.

**CRITICAL**: Do NOT submit immediately after creating. The Phase 4 "Stage all" only
stages — it does NOT authorize publishing. Wait for explicit "Publish".

### Step 2: Submit after "Publish"

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/$PRNUM/reviews/$REVIEW_ID/events \
  --method POST -f event=COMMENT
```

### Step 3: Post thread replies (after the review is submitted)

Thread replies can't be posted while a PENDING review exists (422: one pending review per
PR). Post after Step 2:

```bash
gh api repos/NVIDIA-NeMo/Gym/pulls/$PRNUM/comments/$COMMENT_ID/replies \
  --method POST -f body="<reply text>"
```

Then tell the user: "Review published with <N> inline comments and a summary."

---

## Phase 6: Post-Review Actions

If the user selected "Stage + Run validation", spawn `validation-runner` and stage its
results for user review before posting as a PR comment.

---

## Phase 7: Teardown

**If `TeamCreate` was used**: call `TeamDelete` for each teammate by name/ID to close
their panes. Then clean up the local review branch if the user wants:
```bash
git checkout $BASE_BRANCH && git branch -D pr-$PRNUM-team-review
```

**If `Agent` fallback was used**: send shutdown to each agent individually:
```
SendMessage(to="<agent-name>", message={"type": "shutdown_request"})
```
After all confirm, stop any remaining background tasks with `TaskStop` and clean up:
```bash
git checkout $BASE_BRANCH && git branch -D pr-$PRNUM-team-review
```