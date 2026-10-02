# NeMo UserSim Resources Server

The Resources Server validates prepared UserSim tasks and verifies the native
episode result. It does not sample personas, reconstruct a probe runtime, or
expose probe tool endpoints.

## Episode contract

Preparation asks the pinned UserSim package to materialize one complete row
for each registered probe. Every Gym task carries that row unchanged:

```json
{
  "resolved_row": {
    "persona": {"first_name": "Morgan"},
    "probe_type": "tool_calling",
    "conversation_language": "English",
    "trajectory_id": "native-trajectory-id",
    "usersim_config": {},
    "usersim_provenance": "{\"code_sha\":\"a5f676bf6dc5a73914c8a0860f97c10dd2c214ee\",\"bank_version\":{}}"
  },
  "responses_create_params": {}
}
```

New UserSim rows encode `usersim_provenance` as JSON text so an empty
`bank_version` remains Arrow- and Parquet-safe. Resources also accepts the
legacy mapping form when replaying previously stored rows.

At `/seed_session`, Resources:

1. validates required resolved fields and the pinned UserSim revision;
2. stores the unchanged row and a canonical SHA-256 digest in session state;
3. returns the unchanged row to the Environment Server.

At `/verify`, Resources rejects mismatched episode identity, task identity,
row content, or trajectory identity. `/close_session` removes the immutable
snapshot. The server exposes no `/runtime` or probe tool routes.

## Environment-owned execution

The Environment Server creates the episode's sole
`ProbeEpisodeRuntime.from_resolved_row(...)`. That runtime owns conversation
control, probe tools, mutable state, native call indices, and evidence.

For `tool_calling`, `safety_agentic`, and `financial_services`, the Assistant
Agent performs one model activation and returns function calls without
executing them. UserSim validates and executes accepted calls, appends tool
messages, and requests another Assistant activation when required. UserSim's
internal API-response synthesis routes through the Environment-owned
`tool_simulation_model`.

## Native verification

Resources combines the unchanged resolved row with UserSim's authoritative
episode result and runs `TrajectoryEvaluatorGenerator.agenerate()`. An
explicit probe-to-scorer map selects UserSim's registered scorer for tool use,
safety, financial services, sovereign AI, identity disclosure, and guarded
health-disclosure trajectories. The default health variants intentionally do
not apply the concealment scorer.

The scalar Gym reward is the mean of normalized helpfulness, accuracy, and
coherence when:

- the native conversation completed;
- both User and Assistant messages are present; and
- any applicable probe scorer passed.

The full native `assistant_eval`, normalized quality axes, and scorer output
remain in `verifier_data`. Failures attributed outside the Assistant policy,
including evaluator or scorer execution errors, are masked. Completed
Assistant-policy failures remain unmasked.

The Environment copies `reward`, `mask_sample`, `failure_kind`,
`failure_reason`, and `reward_components` from the verifier response onto the
top level of the persisted rollout for Gym aggregation. The nested
`verification` object retains the full UserSim-specific evidence.

## Run

Configure independent `user_model_*`, `policy_*`, and `support_model_*`
endpoint settings, then run:

```bash
gym eval prepare --config environments/usersim/config.yaml

gym eval run \
  --environment usersim \
  --split example \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

The native episode result preserves Environment-owned calls under
`invocations`. Filter `assistant` or `user` roles for participant-specific
training data. Judge, Summary, and `tool_simulation` calls retain separate
roles and cannot be mistaken for participant examples.
