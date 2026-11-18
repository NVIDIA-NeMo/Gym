# Agent PR review

Use this when a PR touches `responses_api_agents/`, `nemo_gym/agents/`, an agent's configs or presets, or text that tells users which agent to run.

Read this page first. Open a file under `details/` only when its item applies to the diff.

Where this page disagrees with them, two documents win:

- `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx` (the guide)
- `.agents/skills/add-harness/SKILL.md`

Known disagreements between sources are listed in `details/known-conflicts.md`.

For the benchmark side of a pairing (Resources Server, Environment Server, dataset, configs), use `../benchmark-review/review.md`.

## Before you review

**Classify the change.**

- Path: agent sessions, legacy `/run`, or both.
- Sandbox: borrowed, owned by the agent, or none.
- Capability level the PR claims.

**List who inherits it.** That means subclasses, every preset that loads the agent, and every other caller of a shared helper the PR touches. Check each finding against all of them, not only the preset the PR tested. Recipe: "Classify" and "Map the blast radius" in `.agents/skills/review-pr-team/references/agent-harness-review.md`.

## Checklist

### 1. No new legacy agent work

- New benchmarks, agents and configs use agent sessions (seed, activate, close), driven by an Environment Server. Today that is `single_agent_turn`, `single_agent_turn_legacy` (flat rows) or `harbor`; later it may be another, such as a multi-agent-turn server.
- Review an adapter against the session contract, not against one Environment Server's behaviour. Flag code that only works because of how `single_agent_turn` happens to call it.
- The `/run` path of `*_sandboxed_agent` adapters and the `legacy_agent` relay are deprecated. They may take bug fixes only.
- Flag a PR that adds a preset, feature or consumer on them.
- Flag a PR that changes legacy-only code without saying which path a claim covers.
- → `details/legacy.md`

### 2. Every ending is typed from structured evidence

- **Refusals:** a seed the agent cannot serve raises `AgentSeedRejectedError`, a final 422. Examples: an unsupported platform, version or row, or a required grant it never uses. A conflict is 409.
- **Model-side endings get a kind** through `classify_outcome` and `ng_agent_outcome`:
  - `model_request_error` with its real HTTP status;
  - `model_transport_error`;
  - a context limit reported as `incomplete`.
- **What to flag:**
  - parsing error-text substrings when a structured status exists;
  - inventing a status;
  - a bare 502 or 503;
  - a deterministic failure answered with a retryable 500.
- **Step 1 is accepted** (shared outcome ladder, #5870). New code must not contradict it.
- → `details/outcomes-and-retry.md`

### 3. Know what gets retried

- **The Environment Server reruns the whole episode** on a new sandbox for any 5xx, 408, 425, 429, transport error, or `unknown` kind.
  - So a deterministic failure answered with 500 is rerun every attempt.
  - A diagnostic-only failure must never fail the episode.
- **Harness-side model retries** (accepted decision 6):
  - retry only 408, 425, 429, 5xx and transport errors;
  - bound the total wait;
  - never retry another 4xx;
  - outlast a short 429 burst.
- Ask which layer retries, and for a test of it.
- → `details/outcomes-and-retry.md`

### 4. Mask infrastructure failures; never score them as 0

These are not the model's zero:

- a lost sandbox or session;
- a verifier infrastructure error;
- an unconfirmed harness termination;
- an unreachable model server;
- an exit with no result event.

They must be masked or typed:

- `mask_sample` or `evaluation_completed=False`;
- `lost_session_verification()`;
- a typed `NgFailureEvent` through the outcome policy.

They must never be cached as the session's verification. A real task failure still scores 0. Ask which side of the line each new ending is on.

→ `details/masking.md`

### 5. Conformance: run it, don't bend it

- A new or changed harness is registered for the probes and lists them in the PR, with the result.
- Never edit checks, scenarios or scripted replies to make a harness pass. Fix the checker in its own PR, and exercise the scenario instead of skipping it.
- Refusals are tested with `check_agent_seed_rejected`.
- Reviewers accept documented gaps at the Evaluation level. Missing conformance is a finding only when the PR claims a level that needs it.
- → `details/conformance.md`

### 6. Token capture and usage

- **The rollout-prefixed Model Server URL** carries every model call, including compaction, summary, title and subagent calls.
- **Every request carries `x-session-id`** equal to the `invocation_id` of an `AgentInvocation` the adapter returns. Without it, failed attempts have no owner and the P0 ownership checks fail.
- **Usage** is reported only when every call reported it. Otherwise it is unknown plus a gap: never a partial sum, never 0.
- If not every call can be routed, the adapter is evaluation-only.
- → `details/token-capture.md`

### 7. Keep the failure evidence

- A failed response keeps a bounded, redacted native reason.
- A cleanup error must not replace the original failure.
- Interrupted paths emit the same observation gaps.
- Errors are never flattened with `str(error)`; project safe fields.
- Close returns an observation bundle even when activation never ran.
- → `details/outcomes-and-retry.md`

### 8. Clean up on every path; stop the whole tree

- **Only the owner destroys a sandbox.**
  - A borrowed one: remove only the adapter's files.
  - An owned one: stop it on close and on failed setup, and set `ttl_s`.
- **Release everything on failure, retire and shutdown too:** temporary files, archives and per-session maps. Close always ends in a final state.
- **At a deadline or close, stop everything the harness started,** including `setsid` tool commands and processes that respawn. A "confirmed" cleanup receipt means nothing is left running.
- → `details/sandbox-and-runtime.md`

### 9. Don't depend on, or change, the task image

- The harness needs nothing beyond bash: no python3, uv or network to install.
  - A runtime built on the Agent Server and unpacked in the sandbox is fine.
  - Installing on the host and running there is not.
- The task's `HOME`, `BASH_ENV` and `uv.toml` must not change the install.
- The harness's private `HOME` or XDG dirs must not leak into the tool shells the verifier depends on.
- Probe the image correctly:
  - glibc before a musl loader;
  - non-root users and user namespaces;
  - older bash.
- Owned sandboxes offer `network_access`. Web fetch is off by default.
- → `details/sandbox-and-runtime.md`

### 10. Unsupported input is refused, not dropped

- Every config key and request field is either applied, refused with 422, or documented.
- Silently ignoring a user override is a finding. Examples: an unknown config key, a `SandboxAccess.user` or `env`, a metadata key.
- If the harness calls the model from inside the sandbox, the entrypoint class declares `calls_model_from_sandbox` as a literal. Gym reads it from the source file.
- → `details/sandbox-and-runtime.md`

### 11. Shared changes reach every caller

- When a PR changes a shared helper, version pin, aggregation or Environment Server path, trace the other callers: other benchmarks, sibling harnesses, reverification.
- The PR states the score impact, or says there is none.
- If the shared part is large, ask to split it into its own PR.
- → `details/docs-and-scope.md`

### 12. Claims match the code

- READMEs, docstrings, the guide, migration notes and the PR description say what the code does. That includes docs outside the diff that describe changed behaviour.
- Each claim names its path: session, `/run`, or reverification.
- This was the most frequent finding on these PRs.
- → `details/docs-and-scope.md`

### 13. Evidence before support

- Behaviour changes need a real-model smoke, not only unit tests or a scripted model. This is the repo's quality bar.
- Adding an agent to `allowed_agents`, or marking a pairing declared, needs full qualification runs and the benchmark owner's review.
- Version bumps and migrations need a before/after comparison.
- → `details/evidence-and-tests.md`

### 14. Tests fail when the fix is reverted

- Each claimed behaviour and each new ending has a test that fails on revert.
- Every wait in a test is bounded.
- A dependency missing in `conftest` fails loudly instead of skipping.
- Linux-only lifecycle tests and probes ran on Linux, not only on the Mac.
- → `details/evidence-and-tests.md`

### 15. Fail closed, bound sizes, stay inside the root

- Check the exit status of every install, probe and redirect step, and confirm a hook or plugin actually loaded. Never warn and continue.
- Cap logs, stdout and error text.
- Keep parsing linear; test with lines of 1 MB or more.
- Resolve paths strictly, so symlinks cannot escape.
- → `details/sandbox-and-runtime.md`

### 16. Don't add new inconsistencies

- The harness consistency plan (`details/consistency.md`) lists differences between adapters that look accidental, with steps to remove them. Don't let a PR add a new one. Examples:
  - its own retry loop;
  - its own label keys;
  - a new client-specific field fix in `nemo_gym/core/responses_api_types.py`;
  - a runner named `*.py` that a task's `pkill` can hit;
  - a time budget read from a new place.
- Only step 1 and decision 6 are accepted. Raise departures from the other steps as suggestions, not blockers.
- Use one term per concept, in Gym's vocabulary, and export no unused API.
- → `details/consistency.md`

### 17. Public text

- Keep internal ticket ids, hostnames, cluster names and internal URLs out of PR titles, bodies, commits and code comments. Guardwords CI scans the PR body.
- PR titles follow the Conventional Commit format, and commits are signed off.

## Don't flag

- Differences that come from the upstream runtime: model API, packaging, state-dir variables, a CLI's own quirks. They are listed in `details/consistency.md`.
- Conformance gaps that the PR documents at the Evaluation level.
- Runtime archives built on the Agent Server and unpacked in the sandbox.
- Style nits that ruff and pre-commit already enforce.

## How to report

- One finding per issue.
- For each: the item number, `file:line`, the concrete input or state that breaks, and what happens.
- Name the presets or callers affected.
- Say whether the rule is on main, accepted but pending a PR, or only proposed.
- Example review comments, by item: `details/examples.md`.
