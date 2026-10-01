# NeMo UserSim Resources Server

The Resources Server hosts one canonical UserSim tool session per Gym episode. The
prepared task contains an unchanged `resolved_row` produced by UserSim at
revision `b3381ae021baac2a6fb314b5f08a55845243017d`.

At `/seed_session`, Resources validates the row's UserSim revision and
trajectory identity, then constructs the episode only through:

```python
ProbeToolSession.from_resolved_row(resolved_row, models=support_models)
```

Gym does not rebuild persona behavior, themes, toolsets, configuration,
provenance, or trajectory IDs. UserSim remains authoritative for those values.
The seed response returns the same resolved row plus Assistant tool schemas.
The Environment separately owns `ConversationRuntime` and the outer lifecycle.

## Tool execution

The Assistant Agent retains the mechanical model → tool → model loop.
Resources exposes each probe-selected tool through the normal
`POST /{tool_name}` flow. The ordinary route body remains the tool arguments.
The direct-HTTP Agent transport carries hidden turn context, the raw Assistant
response, call ID, and round ID alongside those arguments.

`ProbeToolSession.execute_call()` lazily initializes native effect state,
validates the named route against the raw call, and assigns semantic turn and
call indices. Its result separates the opaque model-visible payload from a
typed receipt and cumulative `CompletedTurnEvidence`. The Agent returns that
hidden evidence with its complete turn trace. A cap may produce no payload or
receipt for a call beyond the executable prefix.

The shared Resources session cookie selects the episode allowlist, state, and
evidence. Identical UserSim call-ID retries are idempotent; conflicting reuse
is rejected by UserSim.

## Lifecycle and verification

The Environment drives its local `ConversationRuntime`, routing non-Assistant
activations normally. For an Assistant turn it invokes the Assistant Agent
once for the complete autonomous loop, accepts the evidence returned with the
Agent trace, and advances the runtime with one `CompletedAssistantTurn`.
Resources owns native tool effects and evidence, not outer conversation
control. It exposes no UserSim-specific `/runtime` endpoints.

The User Agent uses `user_model`, the Assistant Agent uses `policy_model`, and
tool-response synthesis, native scoring, Judge, and Summary use
`support_model`. At verification, Resources invokes UserSim's hosted
`TrajectoryEvaluatorRuntime` once. Its native `assistant_eval` envelope contains
both the applicable probe scorer and the universal quality axes; Gym derives
the scalar reward without redispatching those scorers. Successful episode
output retains the unchanged resolved row, native UserSim result, complete
evaluator evidence, and ordered participant invocations.

## Run

```bash
gym eval prepare --config environments/usersim/config.yaml

gym eval run \
  --environment usersim \
  --split example \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

Preparation uses UserSim's canonical sampler to generate one resolved row for
every registered probe. The Resources-owned dataset config is the single
source of Gym dataset routing.
