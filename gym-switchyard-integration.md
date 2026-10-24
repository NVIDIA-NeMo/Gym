# Experimental Switchyard capture for OpenHands

This adapter routes OpenHands policy calls through an external Switchyard proxy
and rebuilds a Gym training rollout from the proxy's exact token records. It is
opt-in and specific to the legacy Switchyard capture wire contract.

## Dependency and readiness

The required Switchyard implementation is
[PR #63](https://github.com/NVIDIA-NeMo/Switchyard/pull/63), merged as
`97424fc108e267a0885ce6d7f34806fd3b9ba3bd` on **`rl-token-capture-beta`**.
It did not merge into Switchyard `main`. Current native Switchyard deployments
must not be assumed to implement the session-retrieval endpoint below.

Gym now also has native token-ID capture. This external adapter must not be
combined with native capture on the same agent; startup rejects that combination.
Use the native path for current Gym model-server deployments. This adapter remains
an experimental option for deployments that deliberately retain the beta proxy.

A complete OpenHands → beta Switchyard → vLLM rollout, including patch grading,
is still required before this integration is ready for review. Unit and transport
fixtures do not establish that deployment-level compatibility.

## Configuration

The adapter uses the current OpenHands commit pinned by Gym's example configs;
it does not require changes to the OpenHands source. Set the agent config:

```yaml
agent_framework: openhands
switchyard_base_url: http://switchyard-host:4000
```

The URL must be reachable from both the Gym process and the agent container.
Only an HTTP host and explicit port are supported by the existing container
client. OpenCode and simultaneous native `token_id_capture` are rejected at
startup. With the setting absent, the existing Gym path is used. Golden-patch
verification makes no policy calls and does not create a capture session.

Each agent run gets a new UUID. Gym writes the Switchyard route ID (the Gym model
server name) into the OpenHands TOML model field, adds the UUID as
`completion_kwargs.proxy_x_session_id`, and rewrites the model-server host/port
inside the agent container's `NEMO_GYM_CONFIG_DICT`. Evaluation keeps the original
configuration. The beta proxy consumes and strips the correlation field before
forwarding the policy request.

## Beta proxy wire contract

The beta implementation uses the legacy Python route-bundle path:

```yaml
routes:
  policy_model:
    type: model
    token_capture_engine: vllm
    target:
      model: actual-model-served-by-vllm
      base_url: http://vllm-host:8000/v1
      api_key: dummy
      format: openai
```

```bash
switchyard --routing-profiles single-vllm.yaml \
  --enable-rl-logging --rl-log-dir /data/switchyard-traces/job-id -- serve
```

This is not the native `serve --config` path. After the agent exits, Gym reads
`GET /v1/sessions/<uuid>/completions` through its global aiohttp client.
The endpoint returns:

```json
{
  "schema_version": 1,
  "session_id": "run-uuid",
  "completions": []
}
```

A valid envelope has at least one record. Every record must have matching
`schema_version` and `session_id`, a unique `uuid`, nonempty `request_id` and
`model`, `is_valid: true`, chat `messages` ending with the generated assistant
turn, `tools`, and the exact `prompt_token_ids`, `generation_token_ids`, and
`generation_log_probs`. Token arrays must be nonempty integer lists; generation
IDs and finite log probabilities must have equal lengths.

Tools use the beta record shape `{id, description, inputSchema: {jsonSchema}}`.
The adapter maps them into Gym function tools. Model, tools, and tool choice must
remain consistent across the session.

## Reconstruction and failure behavior

Every later prompt must extend the complete prior message history with a
nonempty environment-message suffix. Uncaptured assistant turns, changed
history, reordered/duplicate records, and tool-configuration changes are rejected.
The later prompt tokens must also extend the previous exact prompt plus generated
tokens. The adapter does not repair diverging chat templates or compacted history.

The shared Responses converter places each generation's exact token triple on
its last generated output item. The first prompt becomes rollout input; later
environment messages and generations become rollout output. Successful retrieval
adds the session ID, record UUIDs, and model to response metadata.

A retrieval, schema, or history failure keeps the original diagnostic rollout,
patch, evaluation metrics, and reward, but marks the sample with `mask_sample`
and supplies `switchyard_trace_error`. Such a sample must not be used for training.

## Local checks

```bash
python -m pytest tests/unit_tests/test_switchyard_trace.py \
  tests/unit_tests/test_responses_converter.py \
  responses_api_agents/swe_agents/tests/test_app.py
pre-commit run --all-files
```

Tests cover trace continuity, exact token attachment, failure masking, transport
configuration, session isolation, golden-patch behavior, and capture-mode guards.
Run these against the reviewed checkout; an editable installation pointing at a
different Gym checkout can otherwise import the wrong agent namespace.
