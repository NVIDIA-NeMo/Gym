# Simple agent with context compaction

This Responses API agent follows the same model/tool rollout loop as
`simple_agent`, while maintaining a semantic history that can be materialized
through a configured context-compaction policy before each model call.

The policy, history, materialization, and guards now live in the shared
`nemo_gym.context_management` package. This agent contains the ordinary simple-agent
model/tool/verification loop plus the integration calls; existing `simple_agent`
and other agents are unchanged.

## Compatibility with simple_agent

The task-facing contract is preserved: seed the resource session, execute tool
calls sequentially, feed malformed arguments and definite tool HTTP errors back
to the model, stop on an assistant answer without tool calls or an incomplete
response, accumulate usage, and pass the full uncompacted conversation to the
ordinary verifier. Verification can be skipped with the same configured reward;
otherwise verifier annotations and metrics are retained. Cookies remain isolated
between model and resource calls until they are combined for verification.

This is not a drop-in replacement for every simple-agent configuration. Intentional
differences are mandatory training capture, bounded calls/steps (simple_agent's
default is unbounded), explicit failure outcomes, no automatic transport retries,
direct local episode execution, accumulated session cookies, and the final media
projection described below. Adapter-provided seed/image observations are additions
to the ordinary agent contract. Capture evidence describes selected model actions,
not any subsequent verifier annotations.

Observability limitation: this agent does **not** currently emit simple_agent's
optional `ng_trajectory` / `_ng_trajectory`, including per-tool timing/status and
per-turn resolution records, even when evaluation observability is enabled.
The ordinary verifier's `resolved` field is retained. Segment/action identities
in `context_compaction_result` are training metadata, not a replacement for that
trajectory schema. We deliberately do not copy `TrajectoryTurn.question` with
the complete prompt on every turn: that would restore quadratic prefix storage.
Adding trajectory support is a separate, optional integration that must use
references/deltas or explicitly omit full questions; no new logging schema or
shared-agent refactor is part of this migration.

## Requirements and behavior

- Pair `configs/simple_agent_with_compaction.yaml` with
  `responses_api_models/vllm_model/configs/vllm_model.yaml` (the `policy_model`
  server). There is no dedicated CC model server. Keep
  `return_token_id_information: false`: external capture uses worker-staged tokens,
  not the inline-token settings in `vllm_model_for_training.yaml`.
- Enable global `token_id_capture.enabled` and external staging. This dedicated
  agent opts into capture by default. It requires the framework's `_ng_rollout_id`
  owner identity (for example `dispatch_g0`); there is no legacy inline-token fallback.
  If the framework appends `_a<UUID hex>`, preserve that attempt suffix in all
  capture namespaces. NeMo-RL maps those scopes to stable training-owner IDs.
- Set `context_history.enabled: true` and choose a policy to compact. Setting it
  false selects identity policy through the same capture path, useful for comparison.
- Each intentional rewrite starts an ordinary capture namespace (`dispatch_g0_s0`,
  `dispatch_g0_s1`, ...). The shared client sends an explicit selected parent response
  ID for continuation, or null for a new root. Gym never handles training tokens.
- `/run` returns the ordinary semantic response and verifier reward, plus a top-level
  `context_compaction_result`: selected action identities per segment, terminal
  outcome, ordered media references, and each media asset once. Full request prefixes
  are not recorded per turn; training deltas remain in ordinary capture storage.
  After verification, raw image parts are removed from output and echoed input;
  their non-media semantic content remains, and the asset map owns the raw payloads.
- The loop calls its local episode method directly rather than making a self-HTTP
  request. All remaining mutating HTTP hops use `_retry=False`. A definite tool HTTP
  error remains a model-visible observation; an ambiguous transport/read failure
  aborts the rollout. Do not automatically redispatch that owner identity.
- Default bounds are 256 model calls, 256 segments, zero response resamples, and 256
  agent steps. Adjust `context_history.max_model_calls`, `max_segments`,
  `max_response_retries`, and agent `max_steps` explicitly. Rejected definite responses
  count toward the call limit. `max_steps` and `max_output_tokens` are normal stops.

## Adding CC to another sequential agent

No shared agent refactor is required. An agent can compose
`ContextManagedResponsesClient` with its existing `ServerClient`:

```python
client = ContextManagedResponsesClient(
    server_client=self.server_client,
    model_server=self.config.model_server,
    logical_rollout_id=logical_owner,
    config=self.config.context_history,
    initial_request=initial_request,
)
response = await client.create()
client.append_observation(tool_observations)
response = await client.create()
result = client.finish(response, outcome="completed")
```

Alternatively, `await client.create(full_request)` consumes only the new suffix of
the agent's complete **uncompacted, append-only** source history. It must include
previous selected outputs. Request settings outside `input` are locked throughout
the rollout. Do not combine this client with provider-side history or truncation.
`finish` checks the final selected response ID even when the returned response
accumulates output for verification.

Agent-specific image/tool decoding belongs in a small adapter. This agent exposes
`_tool_response_items` and `_seed_session_response_messages` for that purpose;
they feed the same shared state machine. Sequential agents with different control
flow can integrate the client directly. Concurrent branches require a separate design.

Rolling recency and K-action chunks reuse the original semantic policies. Pending
observations remain protected. Token guards call the read-only
`/context/{capture_id}/measure` route, preserving reserved generation budget, early
chunk close, and remeasurement. Image-only guards do not perform a token probe.
The continuation comparison includes the completed selected response, fixing the
old immediately-preceding-reasoning rewrite bug.

## Validation scope

Focused tests cover both client adapters, unchanged ordinary tool/verifier behavior,
image hooks, bounds/failures, and real model-server capture custody with scripted
worker token deltas. `tests/test_simple_agent_parity.py` runs both actual agent
loops against the same scripted external services, compares request/tool/verifier
payloads and returned results, and checks that an actual reasoning rewrite changes
the model view without changing verifier history. These are local CPU tests, not
a real-vLLM/GPU qualification or the known-good end-to-end parity gate. The
known-good implementation is untouched.

# Licensing information

Code: Apache 2.0

Data: N/A

# Dependencies

- nemo_gym: Apache 2.0
