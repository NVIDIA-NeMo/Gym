# NeMo UserSim runtime

This directory contains the Resources Server half of the UserSim runtime. The
paired Environment Server lives in `environment_servers/nemo_user_sim`.

Dataset preparation is intentionally separate. Each episode request must
already contain a materialized UserSim row:

```json
{
  "resolved_row": {
    "persona": {},
    "probe_type": "general_open_ended",
    "conversation_language": "en",
    "trajectory_id": "example-0",
    "usersim_config": {},
    "usersim_provenance": {
      "code_sha": "<pinned UserSim revision>"
    }
  },
  "role_request_params": {}
}
```

The Resources Server validates the row and its pinned UserSim revision, stores
an immutable digest for the session, and rejects verification if the
Environment Server returns a different row or task identity.

## Runtime ownership

The Environment Server owns `ProbeEpisodeRuntime` and the tool loop. It routes
User and Assistant activations through their Agent Servers, and routes judge,
summary, and tool-simulation requests through configured Model Servers.
Embedding requests use the endpoint and credentials selected by the
`embedding_model` reference. Only Assistant calls are eligible for
training-token capture and sandbox access.

The Resources Server owns verification. A successful `UserSimEpisodeResponse`
preserves the UserSim result and an ordered `UserSimInvocation` list. The list
separates User, Assistant, judge, summary, and tool-simulation calls so
downstream training can select a participant without relabeling another
participant's output.

## Reward and failure handling

Verification averages normalized helpfulness, accuracy, and coherence.
UserSim's 1–5 judge scores are divided by 5, so a fully scored episode has a
reward in `[0.2, 1.0]`. A failed scenario or failed probe gate receives `0.0`.

Probe `status_proposal` is tri-state:

- `True` passes the probe gate.
- `False` is a measured policy failure and is not masked.
- `None` is inconclusive and is masked as `usersim:scorer_inconclusive`.

UserSim scorer results classified as `not_applicable`, including its documented
"scorer skipped" sentinel, do not gate the episode. A completed scenario with
no assistant-quality score is masked as `judge_failed`. Other evaluator and
scorer errors are masked as `verifier_error`.

Assistant-caused invalid responses, including UserSim contract rejection, and
non-session 4xx responses are measured as unmasked reward zero. Participant
transport failures, timeouts, session-plumbing 404 and 409 responses, 408, 425,
429, and 5xx responses remain retryable infrastructure failures. Guarded health
rows are rejected at seed time because the pinned UserSim revision cannot run
their user-side tool activation safely.

## Limits and cleanup

Each verification model request is bounded by
`model_call_timeout_seconds` (300 seconds by default), and the whole sequential
evaluation is bounded by `evaluation_timeout_seconds` (1200 seconds by
default). Seeded Resources sessions expire after `session_ttl_seconds` (1980
seconds by default, matching the episode plus cleanup windows).

The Environment Server registers cleanup before every seed request and closes
participant and Resources sessions on every outcome. Closing is idempotent, and
a short-lived tombstone prevents a delayed seed from recreating closed state.
Unknown or expired active Resources sessions return HTTP 404 when verified.
