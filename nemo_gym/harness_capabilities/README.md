# Harness trajectory evidence

Check P0 evidence from a collected evaluation:

```bash
python scripts/inspect_harness_conformance.py \
    --bundle results/my-harness/rollouts.jsonl \
    --output results/my-harness/evidence
```

Use `matrix --harness NAME=PATH` to compare multiple harnesses. The checker uses
TE-1–TE-9 and the `gym-p0/v1` profile. Reports are `evidence_summary.json`,
`evidence_results.jsonl`, and `evidence_report.md`.

For example, this excerpt from `evidence_report.md` shows retained model-call
evidence passing while one tool record is missing its name:

```markdown
Profile: `gym-p0/v1`. Gate: **not_fulfilled**. Records: 1.

| Evidence | Artifact verdict | Passing / applicable records |
|---|---|---|
| TE-1: model_call_status | fulfilled | 1/1 |
| TE-5: tool_record | not_fulfilled | 0/1 |
```

One corresponding entry in that rollout's `findings` array in
`evidence_results.jsonl` identifies the missing field:

```json
{
  "evidence": "TE-5",
  "assertion": "identity.required",
  "location": "rollouts.jsonl:1/ng_trajectory/tool_calls/0/tool_name",
  "reason": "nonempty value is required"
}
```

These verdicts describe evidence conformance, not task success. A rollout with
zero reward can have fully conforming evidence; a successful task can have
incomplete evidence.

Known producer gap: `miniswe_sandboxed_agent` retains its final submit tool call
as `incomplete`, without output or error evidence. This fails TE-5 with
`tool.terminal` and `tool.outcome`, so the P0 gate is `not_fulfilled` for those
rollouts. mini-SWE must save a terminal result correlated with the submit call
before exiting, and Gym must retain it in the projected evidence. See the
[mini-SWE observability limits](../../responses_api_agents/miniswe_sandboxed_agent/README.md#native-behavior-and-observability).

TE-6 requires a finite numeric `reward` even when `mask_sample=true` excludes it
from scoring. Masked results and explicitly incomplete verification require
nonblank `failure_kind` and `failure_reason`; a masked `reward=0.0` is valid
evidence. The Gym profile uses the default unmasked behavior when `mask_sample`
is omitted and checks `evaluation_completed` when supplied. Both flags must be
booleans when present.

Evidence objects must validate Gym's shared models at these paths:

| JSON path | Model |
|---|---|
| `ng_model_call_capture.calls[]` | `ModelCallRecord` |
| `ng_trajectory.model_calls[]` | `TrajectoryModelCall` |
| `ng_trajectory.turns[]` | `TrajectoryTurn` |
| `ng_trajectory.invocations[]` | `AgentInvocation` |
| `ng_trajectory.tool_calls[]` | `TrajectoryToolCall` |
| `ng_agent_observations.records[]` | `AgentObservationRecord` (discriminated by `kind`) |

Validation is strict and reports the object's JSON path. Invalid objects are
record integrity failures; missing evidence is evaluated by the TE checks and
explicit applicability. Both trajectory and observation representations are
validated when present. Invocation and tool checks retain their supported
fallback between these paths.

TE requirements apply on top of model validity. For example, TE-3 requires a
non-null `question`, a non-null `answer` or `reasoning_content`, and an explicitly
present `resolved` value (including `null` for unknown). TE-5 requires a tool name
and terminal status; TE-1 requires model/server identity and a supported dialect;
TE-2 requires nonnegative counts consistent with retained provider usage. Reports
hash the path/model registry and shared model sources as well as the checker.

For live harness checks and regenerating the capability table, see the
[runner guide](../../scripts/harness_conformance/README.md).

See [Harness Conformance](../../fern/versions/latest/pages/observability/harness-conformance.mdx)
for contracts, applicability, matrix usage, producer onboarding, and limitations.
