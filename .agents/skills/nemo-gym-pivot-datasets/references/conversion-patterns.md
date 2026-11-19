# Conversion Patterns

## Normalize By Meaning

Source artifacts differ in field names, nesting, and whether model outputs are stored as chat
messages, Responses API items, completion objects, or custom rollout records. Do not make those
containers the skill's contract. For each source, first identify:

- the state before the target assistant action
- the target assistant action
- the tools available at that state
- the local reward/verifier target
- the task and trajectory provenance

Once those are identified, build the Gym pivot row from those semantic pieces.

## Model Input

`responses_create_params.input` should represent exactly the context the policy should see before
producing the pivot action.

Common mappings:

- system/developer/user turns become Responses `message` input items.
- prior assistant text becomes completed assistant message output items.
- prior assistant reasoning can be preserved as completed reasoning items when the target agent/model
  path accepts it. Only reasoning from *earlier* turns: the pivot turn's own reasoning is part of the
  action being predicted, and replaying it leaks the answer. A replayed reasoning item needs an `id`.
  See [rollout-artifact-pitfalls.md](rollout-artifact-pitfalls.md).
- prior tool results become `function_call_output` items with the matching `call_id`.
- prior assistant tool calls become completed `function_call` items with `name`, `arguments`, `call_id`, and `status`.

The pivot action itself must not be included in the input prefix.

## Segmenting A Flattened Output List

A rollout artifact stores the trajectory as one flat item list, so "one model call" has to be
reconstructed before it can become a pivot. The portable rule: a tool-call batch ends only at a tool
result or a message, because the environment executes every call of one response before returning
any result. Reasoning never ends a model call. Cross-check the reconstructed count against whatever
per-trajectory model-call count the source records.

Get this wrong and a parallel batch is split across two rows, each labelled with part of what the
model did. Read [rollout-artifact-pitfalls.md](rollout-artifact-pitfalls.md) before writing the
converter, and check the result with `scripts/audit_pivot_turn_boundaries.py`.

## Tools

Preserve the tool schema visible at the pivot state. For function tools, prefer the normalized
shape accepted by the target agent/resource server:

```json
{
  "type": "function",
  "strict": true,
  "name": "tool_name",
  "description": "Tool description",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```

Do not drop tools just because a pivot's expected action uses only one of them. The policy should
see the same action space that existed in the original state.

## Expected Action

For `single_step_tool_use_with_argument_comparison`, use these canonical action shapes:

```json
{"type": "function_call", "name": "tool_name", "arguments": "{\"x\": 1}"}
```

```json
{"type": "function_call_batch", "calls": [{"type": "function_call", "name": "a", "arguments": "{}"},
                                          {"type": "function_call", "name": "b", "arguments": "{}"}]}
```

```json
{"type": "message", "content": "final assistant text"}
```

Keep `arguments` as a JSON string for function calls. Validate that each argument string decodes as
JSON before writing the row.

Pick the type from what the turn did, not from a preference:

- no calls, text only → `message`
- exactly one call → `function_call`
- two or more calls in one response → `function_call_batch`, never a one-element batch

Tool calls take precedence over assistant text: a turn that narrates *and* calls is labelled with its
calls, and the text belongs in provenance. This is not a style choice — it mirrors
`resources_servers/single_step_tool_use_with_argument_comparison/common/response_utils.py::extract_action`,
which is what the verifier runs on the policy's response. A label built any other way is compared
against a shape the response side cannot produce. In one source, roughly a fifth of all pivots came
from turns that narrated and called in the same response.

## Carry The Model Call As reference_output

`expected_action` is the reduced label the verifier compares. Carrying the model call verbatim
alongside it — reasoning, assistant message, tool calls, in emission order — keeps the narration a
narrate-and-call label drops and the reasoning the prefix withholds. Carry the one call, not the
whole trajectory.

## Provenance

Every converter should preserve enough optional provenance to debug a bad pivot without reopening
the full source artifact when practical:

- source task id and source row id
- source trajectory id or rollout id
- source reward and source success marker when available
- pivot message index and assistant-action index
- trajectory length and pivot depth
- original uuid or a generated stable uuid if the source lacks one
- original metadata, unless it contains fields the target schema rejects

Use a dataset-specific provenance object rather than flattening every source field into the top
level when carrying metadata would help later analysis.

## Skips And Metrics

Write skipped-row audit artifacts when rows are rejected. Track at least:

- no expected action
- malformed JSON arguments
- missing tool schema
- unsupported action type
- turns whose arguments do not decode as JSON
- turns whose expected tool is missing from the tool list

Write count-and-percent summaries for action type, tool name, depth, source reward group, and any
selection filters. This catches silent data-shape changes before training. For tool-call datasets
also report the parallel / single / chat split with a batch-size histogram, the number of turns that
narrated and called in the same response, and the number of trajectories whose segmentation
disagreed with the source's own model-call count.
