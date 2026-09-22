# Harness capability conformance

Check whether a Gym harness retains the evidence needed to inspect model calls,
token usage, invocation ownership, semantic turns, and tool execution. Harness
owners run this checker against their collected rollouts during onboarding and
after changes to their harness or its export path.

## Run the checker

Install Gym using the repository's development setup (`uv sync --extra dev`),
then inspect a completed evaluation:

```bash
gym eval conformance \
    --bundle results/my-harness/rollouts.jsonl \
    --profile gym-artifacts-p0/v1 \
    --output results/my-harness/capabilities
```

`--bundle` accepts any rollout JSONL file. It also accepts a directory containing
exactly one `rollouts.jsonl` or `evaluator_rollouts.jsonl`, at its root or under
`artifacts/`. Pass the file explicitly if the directory contains multiple candidates.
Files are streamed one rollout at a time. No model endpoint is needed for inspection.

The module interface exposes the same checks:

```bash
python -m nemo_gym.harness_capabilities inspect \
    --bundle results/my-harness \
    --output results/my-harness/capabilities
```

Collection normally retains full payloads in `ng_trajectory.model_calls`. If you
also have the original shared model capture directory, add
`--capture-dir /absolute/path/to/model-calls`. This restores missing payloads and
checks exact call identity, conflicting representations, missing/extra calls,
and incomplete capture markers. Preserve the original capture directory until
inspection finishes; use a separate capture directory for each evaluation.

Each inspection publishes a content-addressed directory containing:

| File | Contents |
| --- | --- |
| `capability_summary.json` | Requested gate, C0–C11 verdicts, counts, source/checker/schema hashes, qualification limits |
| `capability_results.jsonl` | Per-rollout verdicts and findings with assertion IDs and source field locations |
| `capability_report.md` | Capability matrix and passing-record counts |

Exit codes are **0** for a met gate, **1** for an unmet gate, and **2** for a
checker/input error. A gate requires every inspected record to pass each required
capability. An empty input is an error. Report files are published together only
after inspection completes and input hashes are rechecked. Repeating an identical
inspection reuses its report directory. Prompts and response bodies are omitted
from reports; source paths and field locations are retained.

## Onboard a harness

### 1. Enable model-call capture

Add these top-level settings to the shared configuration used by your agent,
model servers, and collector:

```yaml
observability_enabled: true
model_call_capture_dir: /absolute/shared/path/my-harness/model-calls
```

Route every model request through Gym's correlated model URL
`/ng-rollout/<rollout_id>/v1/...`. Preserve that URL when constructing an SDK
client, forwarding requests, spawning child agents, or retrying a call. Calls
sent directly to a provider do not reach Gym's capture middleware. See
[Model-call Capture](https://docs.nvidia.com/nemo/gym/main/model-server/model-call-capture)
for the custom-agent URL helpers and a runnable collection example.

Use Gym's capture records for request/response bodies, HTTP or transport outcomes,
timing, and provider usage. Preserve absent token values as absent; do not replace
them with zero. P0 requires prompt, completion, reasoning, total, and cached token
counts on successful calls. A provider that omits one cannot meet this strict gate.

### 2. Return invocation evidence

Return `ng_agent_observations` using `AgentObservationBundle` and `AgentInvocation`
from `nemo_gym.rollout_observability`. Give each root or child invocation a stable
ID, retain the parent relationship, and record its ordered conversation.

Associate every captured attempt with exactly one invocation. `ModelCallRef`
supports either `model_call_id` or the exact pair of `model_ref` and `response_id`.
All supplied reference fields must agree. For failures without response IDs,
propagate the invocation ID as `x-session-id` to the Gym model server; the collector
uses captured `client_session_id` to join those attempts to the invocation.
Model names, timestamps, and list order cannot establish ownership. Retries are
separate attempts and need separate references, even if their request bodies match.

For a successful response, an invocation can include:

```python
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ModelCallRef

observations = AgentObservationBundle(
    source="my_harness",
    records=[
        AgentInvocation(
            invocation_id=invocation_id,
            model_calls=[ModelCallRef(model_ref=model_ref, response_id=response.id)],
            conversation=conversation,
        )
    ],
)
```

Here `invocation_id`, `model_ref`, `response`, and `conversation` come from the
actual harness execution. Return the bundle alongside the normal `/run` response;
the collector joins captures and produces `ng_trajectory`.

### 3. Add turn and tool evidence

For semantic turns, supply explicit `TrajectoryTurn` entries in a producer
`ng_trajectory`, including task/rollout/invocation identity, decision number,
timestamp, input, answer, executed-step count, resolution state, and exact call
references. The collector preserves these entries. Keep `resolved=None` until an
authoritative result exists; do not propagate a final reward to intermediate turns.

For tools, put requests and results in the invocation conversation and measured
execution intervals in `ToolCallObservation`. Match both by invocation and
tool-call ID. Record start/end/duration, tool name, timing source, and terminal
status, including failures, timeouts, cancellation, and final submission. A model
request to execute a tool is insufficient evidence that execution occurred.

See [Rollout Evidence](https://docs.nvidia.com/nemo/gym/main/observability/rollout-evidence)
and `nemo_gym/rollout_observability.py` for the producer contracts.

### 4. Collect, inspect, and fix gaps

Run a small representative evaluation with your harness configuration and capture
enabled, using normal `gym env start` / `gym eval run` commands. Wait for collection
to finish, then run `gym eval conformance` on the output JSONL. Inspect failing
assertions in `capability_results.jsonl`, fix the producer, and collect a fresh run.
Editing the report or fabricating missing evidence cannot qualify the harness.

Keep regression coverage for successful calls, tool failures, HTTP errors, retries,
missing usage, rejected model output, and cancellation as supported by your harness.
Include concurrent/child invocation cases if supported. Check the actual emitted
records, including ownership of attempts that never produce an accepted answer.

Run `gym eval health-check RUN_DIR --rollouts-file PATH` separately to assess
rollout health. Conformance reports and `quality_summary.json` answer different
questions; conformance does not rewrite health findings or assert health readiness.
For onboarding review, retain the harness version, run configuration, native
rollouts/captures, conformance reports, and representative rollout/health evidence.
Attach reports to your normal CI artifacts if using the exit status as a gate.

## Profiles and supported scope

These profiles check **retained artifacts**. Their verdicts do not independently
certify retry/cancellation scenario coverage, capture completeness, or execution
behavior. No live probe runner is included. Reports state
`is_behavioral_qualification: false` and zero witnessed scenario coverage.

| Capability | Required in | Current check |
| --- | --- | --- |
| C0: records and identity | P0, P1, All | Readable native records, task/rollout identity, trajectory version, reader integrity |
| C1: calls and statuses | P0, P1, All | Unique call IDs, model/dialect, timing, terminal HTTP/transport evidence |
| C2: five token counts | P0, P1, All | Nonnegative counts, faithful absence/zero, retained usage agreement, subset/total semantics |
| C3: semantic turns | All | Explicit turn fields and resolvable invocation-owned call references |
| C4: model-visible history | P0, P1, All | Ordered retained requests/responses; unresolved media or opaque reasoning fails |
| C5: tool execution | P1, All | Executed tool IDs, request/result joins, terminal status and separate timing |
| C6: independent parallel timing | P1, All | Validator incomplete; always unfulfilled |
| C7: resource lifecycle | All | Validator incomplete; always unfulfilled |
| C8: verifier provenance | All | Validator incomplete; always unfulfilled |
| C9: call payloads | P0, P1, All | Retained request and response/error, explicit no-response transport failure |
| C10: invocation ownership | P0, P1, All | Exactly one explicit owner per captured attempt, consistent references and parent graph |
| C11: complete turn-call accounting | All | Validator incomplete; reference defects diagnosed, completeness never certified |

Profile names are `gym-artifacts-p0/v1`, `gym-artifacts-p1/v1`, and
`gym-artifacts-all/v1`. P1 and All currently always fail their gates because they
require incomplete validators. All C0–C11 verdicts are reported even when selecting
P0, so owners can inspect C3/C5 independently. `onboarding-p0/v1` is not an available
profile. The supported input is native Gym trajectory schema `1.0`; ATIF exports
and normalized third-party records need separate readers.

## Develop the checker

```bash
python -m pytest tests/unit_tests/harness_capabilities -q
```

Contract fixtures use synthetic inputs through Gym's mini-SWE/OpenCode producers;
they are not live harness qualifications. Mutation tests reject missing payloads,
fabricated token counts, broken ownership, duplicate attempts, and missing tool
execution. Integration tests exercise current Gym producer/collector code and the
native CLI. Schema predicates are versioned in `schemas.py`; profile requirements
are in `checker.py`. Keep regression tests and documentation aligned when changing
either contract.
