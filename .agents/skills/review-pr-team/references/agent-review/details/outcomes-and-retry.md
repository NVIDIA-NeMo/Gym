# Outcomes, failure codes, retries and evidence (items 2, 3, 7)

Line numbers are at main `099832fa7e`.

- `B` = `nemo_gym/agents/base_responses_api_agent.py`
- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `P` = the harness consistency plan (summarized in `details/consistency.md`)

## HTTP codes from an adapter (main)

| Reply | Meaning | What the Environment Server does |
|---|---|---|
| 422 `AgentSeedRejectedError` | The agent cannot serve this seed: unusable sandbox, unused required grant, unenforceable network, unsupported platform, version or row (B:360-370, B:903-905; G:187, G:254, G:379). | Final. The episode fails once, with no new sandbox. |
| 409 `AgentSessionConflictError` | Conflicting seed, cookie, activation or closed session (B:373-382; G:181-184). | Final. |
| 500 | Any unexpected exception becomes `agent_route_failed` (B:395-426). | Retried. **The whole episode is seeded again on a new sandbox.** |
| bare 502/503 `HTTPException` | Passes through with **no failure kind** (B:413-414). | Retried, untyped. Flag new ones. |
| 200 + `failed` + `ng_agent_outcome` | `classify_outcome` → `report_outcome`; written only on failed responses (B:1336-1352). | The policy below decides. |

## Default outcome policy (`nemo_gym/agents/agent_outcome.py:77-147`)

| Kind or status | Action |
|---|---|
| `completed`, `incomplete`, or `failed` with no outcome | Verify (graded) |
| 400, 413, 422; `agent_invalid_output`, `agent_limit`, `provider_oom_killed` | Verify |
| 408, 425, 429, 5xx; `model_transport_error`; transient `provider_*` kinds; `unknown` | Retry |
| 401, 403, 404; `model_context_limit`; `agent_run_error`; anything else; malformed metadata | Fail (neither graded nor retried) |

A status rule wins over a kind rule. A context-limit stop is reported as `incomplete`, which is graded.

## Accepted step 1: the shared outcome ladder (pending #5870)

Accepted on 2026-11-09. Plan sections "Decisions" and "Step 1".

**How it works:** the adapter only parses its evidence into `RunEvidence`. `decide_outcome()` then picks the status and failure kind, and `finish_response()` always answers HTTP 200. A 5xx stays only for an adapter crash or a typed lifecycle failure.

**The decisions:**

1. Context limit → `incomplete`, graded.
2. Deadline before the first model reply → graded, unless there is recorded infrastructure evidence. This waits for `x-session-id` on every call. Until then, #5870's `retry_deadline_before_reply` keeps today's retry.
3. No usable result → `failed` + `agent_run_error`, RETRY. Harbor parity is the policy override `agent_run_error: verify`; no per-adapter flag.
4. A 4xx other than 408/425/429 before any reply → `model_request_rejected`, FAIL.
5. Refusal → graded as the answer. A nonzero exit or no result → `agent_run_error`.
6. Harness-side model retries only on 408/425/429/5xx and transport errors, with a bounded total wait.

**Review rule until #5870 merges:** new code must not contradict these. In particular:

- no new bare 502/503;
- no context limit reported as `failed`;
- no new retry loop that retries a 4xx.

## Typing endings: what reviewers flagged

- A classifier that matches error-text substrings when a structured status exists; a missing context-limit kind (#5537).
- Every failed result raised as `RuntimeError` instead of typing the model errors (#5538).
- A fallback that invents HTTP 400 from the SDK error type (#5543).
- Following `__context__` to guess a cause.

**Rules:**

- Text matching is allowed only against a pinned, tested CLI version, as Droid does on 1.1.0.
- A run deadline comes only from the supervisor receipt, never from message text (Fabric's rule; plan H7).

## Retries: what reviewers flagged

- A deadline with no events gave an untyped 500, which was retried forever (#5543).
- A diagnostic validation failure gave a 500 and reran the episode (#3961).
- Two harness retries with sub-second backoff ended the episode on a 429 burst (#5540).

**Layers:**

- `ServerClient` retries 3 times.
- The Model Server retries upstream.
- The collector's attempt cap is `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`, default 3 (`nemo_gym/rollouts/rollout_recovery.py:684-699`; G:724-726).
- Ask which layer retries the failure in question, and whether a test proves it.

Decision 6 is applied in drafts #5910 (NOOA, Hermes, Stirrup), #5911 (Claude Code), #5912 (Codex) and #5913 (OpenCode, Kilo). It is also applied in-branch on #5650, #5540, #5003 and #5543.

## Failure evidence (item 7)

- Keep a bounded, redacted native reason on failed responses (#5537).
- A cleanup exception must not mask the original connect failure (#5082).
- Harbor's `str(error)` dropped the 422 refusal detail (#4680).
- Close returns an `AgentObservationBundle` with gaps even when activation never ran (G § close).
