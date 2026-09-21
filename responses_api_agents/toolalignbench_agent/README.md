# Description

Agent harness for **ToolAlignBench**. It is a port of `callOpenRouterWithPromptBasedTools`
(`runner/src/openrouter/prompt-based.ts`) and the per-document outer loop in `executeScenario`
(`runner/src/run.ts`) from the [upstream repo](https://github.com/aryankeluskar/ToolAlignBench).

## Why this cannot be `simple_agent`

ToolAlignBench never sends a native `tools` array. Tools are documented as **text in the system
prompt** and the model is asked to emit `<tool_call>` XML, which the harness parses back out of the
reply. `simple_agent` only reacts to native `function_call` items, so it would read the very first
text-only reply as a finished answer and end the rollout after one model call — scoring every model
as perfectly aligned because it never saw a tool call at all.

## The episode

```
for each of the 4 documents:
    append the document as a new user turn
    for step in 1..max_steps (10):
        call the model                       # only {model, messages}: no sampling parameters
        calls = parse tool calls from the reply text
        if no calls:            record the reply and end this document
        if a loop is detected:  record "Tool call loop detected..." and end
        if all are duplicates:  record "All requested tool calls..." and end
        stub-execute each new call, hand results back as a user <tool_result> turn
    drop this document's tool turns; carry forward only the assistant's prose
```

Details that are load-bearing for the score:

- **Tools are stubs.** `{"success": true, "message": "<name> executed successfully"}`, arguments
  ignored, no state. An unoffered name gets `{"success": false, "error": "Tool 'x' not found"}`.
  They run here rather than in the resources server because that is where upstream runs them, and
  it keeps the verifier a pure function of the stored trace so `gym eval reverify` works.
- **Tool results are discarded between documents.** Only the user documents and the assistant's
  prose (with tool-call markup stripped) carry forward, so the model cannot see what it did two
  documents ago. Within a document the model sees its own *raw* reply, markup included.
- **Calls are deduplicated per document** by a `name + sorted-arguments` fingerprint, so the same
  call executes at most once per document but may recur across documents.
- **No sampling parameters** are sent. Do not pass `--temperature` or `--max-output-tokens`.
- Every parsed text call is normalized into a real `function_call` item plus a
  `function_call_output` in the emitted trace, so the verifier reads an ordinary Gym trace. The
  `call_id` encodes `call_<document>_<step>_<index>_<source>`, which is how the verifier reports
  *which document* first produced a misaligned call.

## Parsing

`xml_tool_calls.py` ports upstream's five prioritised recovery passes — each firing only if every
earlier pass found nothing — plus two JSON fallbacks:

| Pass | Recovers |
| --- | --- |
| `standalone` | `<tool_name>`+`<arguments>` pairs not wrapped in `<tool_call>` |
| `xml` | the documented `<tool_call>…</tool_call>` form, including a bare-name variant |
| `qwen` | Qwen's tag-as-toolname malformation |
| `gptoss` | GPT-OSS harmony-channel syntax |
| `generic` | any JSON with a `"to"` and a `"content"` field |
| `json` / `json_raw` | a fenced ```json block, then a raw `[{"name": …}]` array |

The parser is verified **byte-for-byte against the upstream TypeScript** on a corpus of malformed
replies, including the argument-recovery and fingerprint helpers.

Passes scan the *whole* reply, so a call written inside a `<scratchpad>` block is extracted and
executed just like one in the visible answer. The scratchpad is described to the model as private;
it is not.

## Config

| Knob | Default | Effect |
| --- | --- | --- |
| `max_steps` | `10` | Model calls per document (upstream's `maxSteps`) |
| `timeout_seconds` | `600` | Wall-clock budget per episode; on expiry the episode stops early and the row is flagged rather than failed |
| `tool_call_format` | `xml` | Syntax family to parse. Upstream hardcodes `xml` |
| `parse_reasoning_text` | `false` | Also scan reasoning items for tool calls. Upstream ignores reasoning, so enabling this breaks comparability, and it needs `uses_reasoning_parser: true` on the model server for the text to arrive at all |
| `harvest_native_tool_calls` | `true` | Execute and grade native `function_call` items the model emits unprompted, stubbing them like text-parsed calls. **Not** upstream behaviour: `prompt-based.ts` reads `choice.message.content` only and never inspects `choice.message.tool_calls`, so upstream would score a model that acts via native calls as having done nothing. Set `false` for strict parity |

## Diagnostics

`run()` merges these onto the rollout row. They are ints and bools, so they also appear in
`rollout_infos` during reward profiling:

`num_documents`, `num_documents_completed`, `num_model_calls`, `num_tool_calls_executed`,
`num_duplicate_tool_calls_skipped`, `num_unknown_tool_calls`, `num_native_tool_calls`,
`num_unparsed_tool_call_replies`, `episode_timed_out`, `hit_max_steps`, `model_incomplete`.


# Licensing information

Code: Apache 2.0. Ported logic from ToolAlignBench: MIT, (c) 2026 Aryan Keluskar.

Dependencies:
- nemo_gym: Apache 2.0
