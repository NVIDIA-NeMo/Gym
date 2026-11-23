# Conformance (item 5)

Line numbers are at main `099832fa7e`.

- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `S` = `.agents/skills/add-harness/SKILL.md`

## Probes

**Run:**

```bash
python scripts/run_harness_conformance.py --harness <key> --output <new-dir>
```

- `<key>` is the agent's directory name without `_agent`. #5367 changes the command to `python -m scripts.run_harness_conformance`.
- Session probes run on Linux only.
- The P0 gate requires every applicable P0 check to pass, including ownership.
- Probes apply only to a harness with a function-calling shell tool. Otherwise the PR says so.
- Pointers: G:822-834; S:97-113; `scripts/harness_conformance/episode.py:195-197`.

**Register a new harness:**

- an entry in `scripts/harness_conformance/registry.py` (`HARNESSES`);
- a runtime install in `ci.install_runtime` (`scripts/harness_conformance/ci.py:96`);
- an episode config.

See `scripts/harness_conformance/README.md:44-45` and S:99.

**Saved-rollout evidence:** `scripts/inspect_harness_conformance.py --require-sandbox` (G:836-845).

## Contract tests

- **Seed refusals:** every refusal is tested with `nemo_gym.agents.testing.agent_session_conformance.check_agent_seed_rejected`, which needs a 422 with a string detail. Note that it also accepts a plain `HTTPException(422)`, which the base class does not log.
- **Session hooks:** `tests/unit_tests/test_session_contract_repo.py` checks that an adapter overrides only the state hooks, and that a seed hook comes with a close hook.
- **Session registration:** a session adapter is listed in `AGENT_SESSION_AGENT_TYPES`. A test in `tests/unit_tests/config/test_global_config.py` enforces it.

## Evidence a PR shows

- The S Step 7 table (S:150-162).
- A full TB2.1 run with 8 repeats when TB2.1 support is claimed (G:864-874).
- At the Evaluation level the evidence check fails by design. The PR records the failure and the declared gaps.

## What reviewers said

**Fix the checker or probe; never bypass it.**

- The checker rejected `status=incomplete` tool calls, and the reviewer asked to extend the checker instead (#5082).
- Run the compaction scenario rather than its bypass (#5082).
- Checker fixes go in their own PR (S Step 4).

**Not a merge blocker at the Evaluation level.** Three reviews said documented conformance gaps do not block a harness PR (#5537, #5543, #5546).

## Pending or proposed

- #5870 (step 1, accepted): a first-request 4xx that is not retryable becomes `model_request_rejected`; the `model_error` scenario changes with it.
- #5873 adds seven outcome-ladder scenarios.
- Plan step 7 (proposed): a versioned contract `gym.agent-session/v1`, plus an offline suite for every adapter that blocks merges.
- The `retry_429` and `retry_500` scenarios record a harness that never retries as incomplete. Decision 6 allows a harness not to retry. See `known-conflicts.md`.
