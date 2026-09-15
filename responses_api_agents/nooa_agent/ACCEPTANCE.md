# NOOA Gym Integration Acceptance

This document is the review contract for the NOOA agent integration in NeMo Gym. It maps the
trajectory acceptance criteria and rollout health checks to implementation evidence, test evidence,
and remaining work.

## Status vocabulary

- **Met**: implementation and focused tests provide the required evidence.
- **Partial**: the data model or implementation exists, but evidence or behavior is incomplete.
- **Missing**: the integration cannot currently provide the criterion.
- **Unobserved**: a health check cannot be computed because its required capability evidence is absent;
  this is distinct from an unhealthy finding.

## Acceptance criteria

| ID | Requirement | Current status | Implementation evidence | Remaining acceptance work |
|---|---|---:|---|---|
| C1 | Model-call identity, status, error, and termination metadata | Met | `TrajectoryModelCall`, `TrajectoryResponseMetadata`; model capture projection in `nemo_gym/rollout_collection.py`; NOOA termination gaps in `responses_api_agents/nooa_agent/app.py` | Keep failure-path regression coverage. |
| C2 | Prompt, completion, reasoning, total, and cached token counts when available | Met | `TrajectoryTokenStats`; `extract_token_stats()` / `build_model_call_record()`; canonical capture projection | Assert every available field through persisted JSONL. |
| C3 | Canonical `TrajectoryTurn` records preserve answer, reasoning, and tool-call activity | Partial | `TrajectoryTurn`; `GymTraceHooks.record_model_response()` preserves raw response output in `answer` | Populate the dedicated `reasoning_content` field and add a persisted reasoning + tool-call round-trip test. |
| C4 | Model-visible input and output history is reconstructable from persisted rollout JSONL | Met | `AgentInvocation.conversation`; captured model request/response payloads; canonical `ng_trajectory` | Add one acceptance fixture proving reconstruction after JSON round-trip. |
| C5 | Tool output, status, start/completion timestamps, and duration | Met | `TrajectoryToolCall`; `_GymToolDispatcher.invoke()`; NOOA code-execution hooks; collector output merge | Add failure/cancellation round-trip coverage. |
| C6 | Independent timing for parallel tool calls | Partial | Per-call IDs and timing fields exist | Prove independent records under concurrent requests; define serialized stateful-resource versus truly parallel stateless/sandbox semantics. |
| C7 | Resource-server-backed and custom or sandbox-backed benchmarks | Partial | Resource-server capability E2E; seeded sandbox attachment in `runner.py`; real external rollouts | Add a committed seeded-Docker E2E test on the final stack. |
| C8 | Turn identity, timestamp, question, resolution, and step metadata | Partial | Identity, timestamp, and step placeholder are emitted | Populate question and resolution; use producer step metadata or emit a documented gap instead of implying it. |
| C9 | Captured model requests and responses retain their payloads | Met | `TrajectoryModelCall.request` / `.response`; capture projection | Keep payload round-trip assertions. |
| C10 | Model calls have exact ownership references on an `AgentInvocation` | Met | `ModelCallRef`; `join_model_call_observations()` exact join and ambiguity/conflict gaps | Retain exact-join tests. |
| C11 | Model calls have exact ownership references on a `TrajectoryTurn` | Partial | Turns carry exact `(model_ref, response_id)` refs and collector binding supports them | Add turn-level canonicalization, ambiguity, unmatched, duplicate, and ownership-conflict tests. |

## Health checks

| ID | Health check | Capability basis | Current status | Remaining acceptance work |
|---|---|---|---:|---|
| H4 | `rollout_missing_agent_turns` | C3 | Met | Preserve focused behavior tests. |
| H5 | `agent_turn_hollow` | C3 | Met | Preserve reasoning/tool activity tests. |
| H6 | `trajectory_capture_mismatch` | C10 or C11 | Met | Add turn-specific contradictory ownership coverage. |
| H7 | `model_call_failed` | C1 and C10/C11 | Met | Preserve status/error-category coverage. |
| H8 | `model_call_missing_token_counts` | C2 and C10/C11 | Met | Preserve unobserved-versus-unhealthy behavior. |
| H9 | `model_call_zero_completion_tokens` | C2 and C10/C11 | Met | Preserve length-limited exemption behavior. |
| H10 | `model_call_runaway_generation` | C1, C9, and C10/C11 | Met | Keep Responses status/incomplete-reason variants covered. |
| H11 | `rollout_token_count_mismatch` | C2, C11, complete accounting, top-level totals | Met | Add one persisted acceptance-matrix fixture. |
| H13 | `task_no_successful_model_calls` | C11 and complete accounting for every repeat | Partial | Add focused success, incomplete-repeat/unobserved, duplicate-repeat, multi-repeat, and ignore-path tests. |

## Required implementation order

1. Complete canonical turn evidence (C3/C8).
2. Harden exact turn ownership and tests (C11).
3. Prove independent concurrent tool timing (C6).
4. Add the seeded Docker benchmark route E2E (C7).
5. Complete H13 edge-case coverage.
6. Add a persisted acceptance-matrix rollout fixture that exercises C1-C11 and H4-H11/H13.
7. Run final unit tests, scoped pre-commit, and a real rollout on the final rebased stack.

## Acceptance-matrix fixture

The fixture must include:

- multi-turn model-visible questions and answers;
- explicit reasoning content;
- function/tool calls and model-visible outputs;
- independent timing for concurrent sibling tool calls;
- successful and length-limited model calls;
- prompt, completion, reasoning, total, and cached token counts;
- exact invocation and turn ownership;
- top-level rollout token totals;
- JSON serialization and reload before health evaluation.

Every applicable health check must be either evaluated with the expected result or marked `unobserved`
with a documented missing capability. Missing evidence must never be silently interpreted as healthy.

## Verification baseline

The audit that created this document ran:

- NOOA observability, Gym tool, rollout observability, and collector suites: **240 passed**.
- Standalone rollout-health suite: **53 passed**.
- Combined controller acceptance suite: **87 passed**.

## PR completion checklist

- [ ] C3/C8 canonical turn fields implemented and persisted round-trip test added.
- [ ] C11 turn ownership ambiguity/conflict tests added.
- [ ] C6 concurrent timing behavior and semantics tested.
- [ ] C7 resource-server and seeded-sandbox E2E routes tested.
- [ ] H13 edge cases tested.
- [ ] Acceptance-matrix fixture passes all applicable health checks.
- [ ] Representative real rollout inspected on the final stack.
- [ ] Scoped/full pre-commit passes.
- [ ] All PR commits carry DCO `Signed-off-by` trailers.
