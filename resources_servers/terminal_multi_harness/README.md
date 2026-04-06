# Description
This is a harness-aware resources server for structured next-action verification across multiple agent harnesses.

It is intended for harnesses such as `codex` and `opencode`, where the next step is naturally represented as:

- a chat message
- a single tool call
- a batch of tool calls in one turn

Unlike `terminus_judge`, this server does not assume the model response is assistant text containing a serialized command JSON blob. Terminus-style command-sequence comparison should be added later as a separate harness adapter rather than mixed into this first structured-action path.

## Current scope

- immediate harnesses: `codex`, `opencode`
- current action types:
  - `message`
  - `function_call`
  - `function_call_batch`
- current comparison behavior:
  - structured tool-call argument comparison
  - optional ignored argument keys for volatile tool arguments
  - ordered or unordered batch comparison

## Notes

- For Codex collection, Aspen's raw teacher-model `backend_response` should be
  treated as the source of truth for the expected answer. The verifier's
  structured `expected_action` should be normalized from that payload.
- Aspen's synthesized `responses_api_response` is still useful as a compatibility
  mirror, but it should normalize to the same canonical action as the raw
  `backend_response`.
- For `opencode`, the top-level model action should be the reward target. If the model emits a `batch` tool call, the expected action should usually be a single `function_call` named `batch`, not the runtime fan-out of child tool executions.
- Terminus-2 is intentionally not handled here yet. The correct extension point is a future harness-specific normalizer and comparator.
