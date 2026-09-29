# Pivot Row Contract

## Required Top-Level Fields

For the single-step tool-use verifier family, each pivot row should contain:

```json
{
  "responses_create_params": {},
  "expected_action": {},
  "agent_ref": {"type": "responses_api_agents", "name": "dataset_agent_name"}
}
```

`responses_create_params` and `expected_action` are required. `agent_ref` is required only when rows
route themselves rather than being routed by the config — see [agent_ref](#agent_ref).

Other fields are optional metadata. They can be useful for traceability, filtering, and debugging,
but they are not part of the `single_step_tool_use_with_argument_comparison` resource-server input
contract.

## responses_create_params

The target object should be a minimal non-null Responses API request body:

```json
{
  "input": [],
  "tools": [],
  "parallel_tool_calls": true,
  "tool_choice": "auto"
}
```

Rules:

- `input` is the model-call prefix before the pivot action.
- `tools` is the full tool list available at that state.
- `tool_choice` should be `"auto"`. Avoid `"required"` for this workflow because it can trigger
  structured decoding paths in inference engines such as vLLM.
- `parallel_tool_calls` must be `true` for any dataset that can carry `function_call_batch` labels.
  Asking the policy for a single call and then labelling the turn with several is a contradiction the
  verifier cannot resolve.
- `responses_create_params` forbids unknown keys, so a stray field inside it is a hard error. Unknown
  keys at the *row* top level are accepted, but they are dropped from rollout records — see
  [rollout-artifact-pitfalls.md](rollout-artifact-pitfalls.md).
- omit optional fields whose value would be null, especially `metadata`.
- include `model` only if the local agent/model path expects row-level model names.

## expected_action

Supported action types for `single_step_tool_use_with_argument_comparison`, one per model call:

- `function_call`: one tool call with `name` and JSON-string `arguments`.
- `function_call_batch`: several calls emitted in one response,
  `{"type": "function_call_batch", "calls": [<function_call>, ...]}`, at least one call.
- `message`: expected assistant text response.

For tool actions:

- every expected tool name should appear in `responses_create_params.tools`, including every tool
  named inside a batch.
- every `arguments` value should decode as JSON.
- the comparator matches expected against actual calls as an unordered multiset, so the order a batch
  is written in is not part of the label.

## One Model Call, One Row

A pivot covers one whole model call, and `expected_action` describes all of it.

- No calls: `message`.
- Exactly one call: `function_call`.
- Two or more calls in the same response: `function_call_batch`.

Never split a batch across rows and never drop it. A multi-call turn labelled with only its first
call is scored against that one call alone, so the row silently teaches the model to stop early.

Never emit a one-element `function_call_batch`. The response side normalizes a lone emitted call to
`function_call` and never wraps it, so a one-element batch describes a shape no response can have —
it means the converter took the batch path for a single call.

Tool calls take precedence over assistant text. A turn that narrates *and* calls is labelled with its
calls; keep the text as provenance rather than choosing `message`.

## agent_ref

`agent_ref` is optional. A dataset routed by its config omits it — the environment's own
`data/parallel_example.jsonl` has no `agent_ref` at all — and Gym fills it in at rollout time. Carry
it when rows route themselves, which is the usual case for a pivot dataset:

```json
{"type": "responses_api_agents", "name": "example_single_step_tool_use_with_argument_comparison_agent"}
```

The `agent_ref.name` should match the config agent block that consumes the dataset. The matching
resource-server block is usually the agent name with `_agent` replaced by `_resources_server`, but
always verify against the actual config.

## Optional Provenance

Provenance is recommended but not required by the verifier. Keep enough metadata to debug a pivot
without reopening the full source artifact when practical:

- task id or sample id
- source uuid, if present
- source trajectory/rollout id, if present
- source reward and clean-positive marker, if available
- pivot item index or assistant-action index, if meaningful
- depth or assistant depth, if meaningful
- original metadata that is useful and safe to carry
- `reference_output`: the model call this pivot was taken from, verbatim and in emission order —
  its reasoning, its assistant message if it narrated, and its tool calls. `expected_action` is the
  reduced label the verifier compares; `reference_output` is the whole response, so the narration a
  narrate-and-call label drops and the reasoning the prefix withholds are both still on the row.
  Carry the one call, not the whole trajectory — see
  [rollout-artifact-pitfalls.md](rollout-artifact-pitfalls.md).

## Compatibility Checks

A pivot row is not done until it passes:

- JSONL parse and fixed row-shape checks.
- expected-action schema validation, including no one-element `function_call_batch`.
- `parallel_tool_calls: true` whenever batch labels are present.
- every label self-scores 1.0 when replayed through the response-side normalizer under the
  comparator the dataset will actually be scored with.
- Responses request schema validation when Gym models are available.
- agent/config alignment checks.
