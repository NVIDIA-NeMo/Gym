# CritPt Agent

Custom two-turn agent for the CritPt benchmark (research-level physics problems). For each problem it:

1. **Turn 1 — solve:** sends the problem to the model and collects its reasoning and conclusion.
2. **Turn 2 — fill the template:** strips the Turn 1 thinking blocks (`<think>` / `<thinking>`), then asks
   the model to populate the problem's `code_template` using the Turn 1 conclusion as context (the Turn 2
   prompt instructs the model not to reason again).

Both model calls run inside one `/v1/responses` activation. The `single_agent_turn_legacy` environment
server seeds and closes the agent session, then sends only Turn 2's output to the CritPt resources
server's `/verify`, which scores it via the Artificial Analysis API. Session close returns observations
referencing both model calls. Sessions restart from the input after a crash; mid-turn checkpoint resume
and multi-turn continuation are not supported by this agent.

A session runs one activation. An identical retry joins it instead of repeating the model calls, a
different request (or one routed under another rollout) is refused with 409, and session close cancels a
running activation while keeping the evidence collected so far.

The returned response's `usage` sums input, output, and total tokens across both turns, so the standard
token metrics report the full solve-and-format cost. If either turn omits usage, the combined usage is
unknown (`null`). Only the Turn 2 output is graded, with one exception: a failed solve response ends the
activation before formatting and is returned as is. It carries no outcome classification, so it is verified
like any unclassified failed response and fills its slot in the batch. A truncated (`incomplete`) solve
still goes on to formatting.

## Input contract

Run `gym eval prepare --benchmark critpt` to render both prompts into `responses_create_params.input`.
The input must end with two consecutive user messages: the problem, followed by the formatting prompt
containing the code template. Earlier system/history messages are preserved. The final user message is
deferred: the first model call receives everything before it, and the second inserts the cleaned solve
answer immediately before it. This is a CritPt-specific input convention.

Re-prepare existing benchmark data after upgrading, and omit `--prompt-config` for prepared rows.
Direct legacy `/run` callers can still send the original solve input plus `code_template`; `/run`
adapts it to the prepared form before invoking the two-turn response path. Verifier fields remain on
the resources-server path and are not copied into model metadata or the agent seed.

## Configuration

- `resources_server`: the CritPt resources server instance to seed sessions and verify against
- `model_server`: the model server used for generation
- `turn2_prompt_fpath`: the Turn 2 template used to adapt old `/run` inputs with `code_template`
  (e.g. `benchmarks/critpt/prompts/turn2.yaml`). Keep it equal to the template `benchmarks/critpt/prepare.py`
  renders; it does not affect prepared rows or session runs.

The full wiring (resources server + this agent + example dataset) lives in
`resources_servers/critpt/configs/critpt.yaml`; the benchmark dataset is narrowed in
`benchmarks/critpt/config.yaml`.
