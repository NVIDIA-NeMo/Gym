# NeMo User Sim Resources Server

This server validates materialized NeMo User Sim task rows, snapshots them for
the episode, and scores completed episode requests. The paired Environment
Server in `environment_servers/nemo_user_sim` owns the conversation runtime and
tool loop.

## Task and session contract

Each request contains a stable string `task_id` and a complete materialized
row. The following row is a schematic with nested values elided; see
`data/example.jsonl` for complete rows:

```json
{
  "task_id": "example-0",
  "resolved_row": {
    "persona": {"first_name": "...", "last_name": "..."},
    "probe_type": "general_open_ended",
    "conversation_language": "English",
    "trajectory_id": "example-0",
    "usersim_config": {"...": "..."},
    "usersim_provenance": "{\"code_sha\":\"<pinned NeMo User Sim revision>\",\"nemotron_personas_version\":\"0.0.2\",\"bank_version\":{}}"
  }
}
```

At `/seed_session`, the Resources Server validates required fields, the
configured NeMo User Sim revision, and the Nemotron-Personas version (`"0.0.2"`,
or `"synthetic"` for the committed example rows), and rejects guarded health
rows. It then stores the row and its canonical SHA-256 digest. At `/verify`, it
rejects mismatched episode identity, task identity, row content, or trajectory
identity.

The digest proves that the materialized row did not change between seeding and
verification. It does not independently compare the row with an earlier
preparation artifact.

## Runtime ownership

The Environment Server creates the episode's sole
`ProbeEpisodeRuntime.from_resolved_row(...)`. It routes User and Assistant
activations through their Agent Servers and judge, summary, and tool-simulation
requests through configured Model Servers. Embedding requests go directly to
`/embeddings` on the endpoint configured for `embedding_model`, using its
`base_url` and `api_key`. Only Assistant calls are eligible for training-token
capture and sandbox access.

The persisted top-level `invocations` ledger separates User, Assistant, judge,
summary, and tool-simulation calls. The final verifier requests are represented
in `verification.verifier_data`, not as participant model-call captures.

## Verification and reward

The Resources Server calls NeMo User Sim's
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

Assistant-attributed simulation failures, including Assistant contract
rejections and non-session 4xx responses, receive an unmasked reward of zero.
Simulation failures that NeMo User Sim attributes outside the Assistant receive
a masked reward of zero. Transport failures, participant timeouts, session 404 and 409
responses, and 408, 425, 429, and 5xx responses produce no reward; they are
written to `rollouts_failures.jsonl`. Guarded health rows are rejected at seed
time because the pinned NeMo User Sim revision cannot run their user-side tool
activation safely.

## Limits and cleanup

Each verification model request is bounded by
`model_call_timeout_seconds` (300 seconds by default), and the full sequential
evaluation is bounded by `evaluation_timeout_seconds` (1200 seconds by
default). Seeded Resources Server sessions expire after `session_ttl_seconds`
(1980 seconds by default).

The Environment Server registers cleanup before every seed request and closes
participant and Resources Server sessions on every outcome. Closing is
idempotent, and a short-lived tombstone prevents a delayed seed from recreating
closed state. A verify for an unknown or closed Resources Server session returns
a masked sample with `failure_kind` `session_lost`, and one whose seed outlived
`session_ttl_seconds` a masked sample with `session_expired`.
