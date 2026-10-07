# Interactive agent environment

Compose a Resources Server with a continuation-capable Agent Server to run several
user turns in one native conversation and task sandbox. Import
`environment_servers/interactive_agent/configs/interactive_agent.yaml` and bind
`resources_server.name` and `agent_server.name` to independent server definitions.
Materialized tasks provide `task_input.task_data`; Resources returns the initial
candidate prompt after preparing the task. Submit episodes through `/run`.

The protocol is:

1. Resources `/seed_session` returns `InteractiveResourcesSeedResponse`, including
   sandbox access, initial Responses parameters, and continuation requirements.
2. Agent `/v1/agent_sessions` validates runtime prerequisites and the requested
   capabilities before creating candidate state. Existing one-turn agents reject
   continuation until their adapters explicitly support it.
3. Agent `/v1/agent_sessions/activate` receives one input delta and an activation
   ID starting at zero. It retains the conversation, sandbox, and runtime HOME.
4. Resources `/step` receives the completed activation and decides whether to
   continue, which input to append, and whether the continuation is synthetic.
5. Agent close cancels and awaits any active operation, confirms remote cleanup,
   and returns cumulative evidence. Verification receives that receipt and the
   full activation/step history, then Resources closes the task sandbox.

Wire types live in `nemo_gym.interactive_agent_types`. The environment imports no
benchmark or harness implementation. Resources owns prompts, simulator policy,
artifact capture, and scoring; the adapter owns native execution and evidence.
Resources orchestration does not require exposing simulator calls as candidate
tools. Candidate-native tools operate directly in the granted sandbox.

Resources may return a required `runtime_policy` with a versioned `format` and
JSON `settings`. The environment forwards it unchanged to the adapter, including
it in the immutable session seed. Adapters override
`_validate_agent_runtime_policy(policy)` to reject unsupported formats and settings
before runtime setup, then apply the policy during setup. The default rejects any
policy; orchestration does not translate native settings or silently discard them.

Adapters implement `_agent_continuation_capabilities()` and
`_activate_agent_session_state(state, body, request)`. The shared session layer
handles ordered IDs, full-input fingerprints, retry join/replay, and close fencing.
Declare observation capabilities, runtime prerequisites, and budget semantics;
validate image/runtime compatibility in the seed hook. The close hook must return
`cleanup_confirmed=True` only after remote process cleanup and sandbox disconnect
succeed. A failed hook leaves state available for another close attempt.

A disconnected activation or close HTTP waiter does not cancel shared work.
Conflicting inputs, skipped IDs, parallel distinct activations, and continuations
after an execution failure are rejected. Successful close receipts remain available
for `session_close_retry_window_seconds`; use unique session IDs and provider TTLs.
The ledger is process-local and does not provide recovery after a server crash.
Resources can use `OrderedOperationLedger` for the same semantics on `/step`;
Resources must bind its own episode identity and session cookie.

Ordered events distinguish visible text, tools, reasoning, and compaction. Timing,
harness steps, and provider calls have separate fields. Preserve raw native
artifacts and model-call correlation in the adapter; each benchmark chooses its
permitted view without discarding the underlying evidence. Simulator failures are
reported at stage `step`. Candidate runtime/provider failures must raise; declared
model or wall-time limits may return a partial checkpoint after confirmed cleanup.
The environment's `max_activations` is a safety fence: exceeding it fails the episode.
Failed dependency calls retain their HTTP status and a redacted response-body
preview of at most 8192 characters. The failure includes the agent close receipt
and observations collected during deferred cleanup, including failed-invocation
evidence that is absent from the completed activation history.
The collector saves the complete failure under `_ng_failure` in its failures
sidecar alongside the existing summary fields.

Validation covers scripted independent endpoints, multi-turn retry/disconnect/close
races, capability rejection, failure propagation, and cleanup before verification.
Real-model remote sandbox qualification belongs to each concrete composition.
Training and task services that must survive candidate close require separate
qualification and explicit service ownership/handoff support.

## Validation record

On 2026-10-07, an independent HTTP smoke connected this environment to an
in-process conversation fixture, a Resources fixture, and Gym's model adapter with
GPT-5.5. Two real model calls returned `alpha` and then `alpha verified`; the
fixture verifier returned reward `1.0`. The recorded lifecycle was
`resources_seed → agent_seed → activation_0 → step_0 → activation_1 → step_1 →
agent_close → verify → resources_close`, with zero remaining Resources sessions.
This checks real inference and HTTP orchestration; it does not qualify a native
harness, remote sandbox, tools, or benchmark scoring.
