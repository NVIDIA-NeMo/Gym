# NeMo User Sim Resources Server

This server validates materialized NeMo User Sim task rows, snapshots them for
the episode, and scores completed episode requests. The paired Environment
Server in `environment_servers/nemo_user_sim` owns the conversation runtime and
tool loop.

## Task and session contract

Each request contains a stable string `task_id` and a complete materialized row:

```json
{
  "task_id": "example-0",
  "resolved_row": {
    "persona": {},
    "probe_type": "general_open_ended",
    "conversation_language": "en",
    "trajectory_id": "example-0",
    "usersim_config": {},
    "usersim_provenance": "{\"code_sha\":\"<pinned User Sim revision>\",\"bank_version\":{}}"
  }
}
```

At `/seed_session`, Resources validates required fields and the configured
User Sim revision, then stores the row and its canonical SHA-256 digest. At
`/verify`, it rejects mismatched episode identity, task identity, row content,
or trajectory identity.

The digest proves that the materialized row did not change between seeding and
verification. It does not independently compare the row with an earlier
preparation artifact.

## Runtime ownership

The Environment Server creates the episode's sole
`ProbeEpisodeRuntime.from_resolved_row(...)`. It routes User and Assistant
activations through their Agent Servers and judge, summary, and tool-simulation
requests through configured Model Servers. Only Assistant calls are eligible
for training-token capture and sandbox access.

The persisted top-level `invocations` ledger separates User, Assistant, judge,
summary, and tool-simulation calls. The final verifier requests are represented
in `verification.verifier_data`, not as participant model-call captures.

## Verification and reward

Resources calls NeMo User Sim's
`TrajectoryEvaluatorGenerator.agenerate()` once with the scorer selected for
the prepared probe when one applies.

`scenario_completed` requires a completed conversation, both User and
Assistant messages, and a passing applicable probe scorer. When it is `true`
and the required quality axes exist, `reward` is the mean of normalized
helpfulness, accuracy, and coherence. Otherwise, `reward` is `0.0`.

Probe `status_proposal` is tri-state:

- `true` passes the probe gate.
- `false` is a measured policy failure and is not masked.
- `null` is inconclusive and is masked as
  `usersim:scorer_inconclusive`.

Scorer results classified as `not_applicable`, including the documented
skipped sentinel, do not gate the episode. Evaluator or scorer errors are
masked as `verifier_error`; a completed scenario with no Assistant quality
score is masked as `judge_failed`.

Assistant-attributed simulation failures receive unmasked reward zero.
Infrastructure failures and simulation failures attributed outside the
Assistant receive masked reward zero.

## Limits and cleanup

Each verification model request is bounded by
`model_call_timeout_seconds` (300 seconds by default), and the full sequential
evaluation is bounded by `evaluation_timeout_seconds` (1200 seconds by
default). Seeded Resources sessions expire after `session_ttl_seconds` (1980
seconds by default).

The Environment Server registers cleanup before every seed request and closes
participant and Resources sessions on every outcome. Closing is idempotent,
and a short-lived tombstone prevents delayed seeds from recreating closed
state. Unknown or expired active Resources sessions return HTTP 404 during
verification.
