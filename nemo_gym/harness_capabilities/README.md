# Harness trajectory evidence

Check P0 evidence from a collected evaluation:

```bash
python scripts/check_harness_conformance.py \
    --bundle results/my-harness/rollouts.jsonl \
    --output results/my-harness/evidence
```

Use `matrix --harness NAME=PATH` to compare multiple harnesses. The checker uses
TE-1–TE-9 and the `gym-p0/v1` profile. Reports are `evidence_summary.json`,
`evidence_results.jsonl`, and `evidence_report.md`.

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

Unit tests use hand-authored contracts in `tests/unit_tests/harness_capabilities/synthetic.py`;
they validate the checker independently of harness implementations. To measure
actual harnesses and regenerate the commit-pinned documentation table, use
`python -m scripts.harness_conformance.table --commit <full-sha> --output <new-directory>`.
See the [runner guide](../../scripts/harness_conformance/README.md) for prerequisites and test gates.

See [Harness Conformance](../../fern/versions/latest/pages/observability/harness-conformance.mdx)
for contracts, applicability, matrix usage, producer onboarding, and limitations.
