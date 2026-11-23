# CritPt Agent

Custom two-turn agent for the CritPt benchmark (research-level physics problems). For each problem it:

1. **Turn 1 — solve:** sends the problem to the model and collects its reasoning and conclusion.
2. **Turn 2 — fill the template:** strips the Turn 1 thinking blocks (`<think>` / `<thinking>`), then asks
   the model to populate the problem's `code_template` using the Turn 1 conclusion as context (the Turn 2
   prompt instructs the model not to reason again).

The accumulated Turn 2 output is submitted to the CritPt resources server's `/verify`, which scores it via
the Artificial Analysis API.

The returned response's `usage` sums input, output, and total tokens across both turns, so the standard
token metrics report the full solve-and-format cost. If either turn omits usage, the combined usage is
unknown (`null`). Only the Turn 2 output is graded.

## Configuration

- `resources_server`: the CritPt resources server instance to seed sessions and verify against
- `model_server`: the model server used for generation
- `turn2_prompt_fpath`: the Turn 2 user-prompt template filled with the problem's `code_template`
  (e.g. `benchmarks/critpt/prompts/turn2.yaml`)

The full wiring (resources server + this agent + example dataset) lives in
`resources_servers/critpt/configs/critpt.yaml`; the benchmark dataset is narrowed in
`benchmarks/critpt/config.yaml`.
