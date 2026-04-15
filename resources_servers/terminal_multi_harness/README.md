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
  - completion-message verification is structural: any non-empty assistant `message` matches
  - actual tool calls must validate against the declared tool schema
  - actual tool-call names must match expected tool names exactly
  - actual tool-call argument keys may not go beyond the expected answer
  - harness rulebooks may ignore selected argument values, but they do not waive the no-extra-keys rule
  - tool-value comparison is harness-specific and documented in `docs/rulebooks/`
  - multiple tool calls are aligned and compared pairwise using the harness rulebook

## Notes

- For Codex collection, Aspen's raw teacher-model `backend_response` should be
  treated as the source of truth for the expected answer. The verifier's
  structured `expected_action` should be normalized from that payload.
- Aspen's synthesized `responses_api_response` is still useful as a compatibility
  mirror, but it should normalize to the same canonical action as the raw
  `backend_response`.
- The verifier should receive the declared tool definitions from the current
  sample. If they are not passed explicitly, it falls back to
  `responses_create_params.tools`.
- For `opencode`, the reward target should come from the teacher response's sibling child tool calls. The Opencode rulebook is the normative source for how those tool calls are aligned and compared.
- Terminus-2 is intentionally not handled here yet. The correct extension point is a future harness-specific normalizer and comparator.

## Docs

- workflow:
  - `resources_servers/terminal_multi_harness/docs/development-workflow.md`
- extension architecture:
  - `resources_servers/terminal_multi_harness/docs/harness-extension-architecture.md`
- Codex rulebook:
  - `resources_servers/terminal_multi_harness/docs/rulebooks/codex-match-rules.md`
- Opencode rulebook:
  - `resources_servers/terminal_multi_harness/docs/rulebooks/opencode-match-rules.md`
