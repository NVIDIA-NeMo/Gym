# Description

A single-tool-call environment that returns **multiple decoupled reward components**,
intended for multi-reward RL such as GDPO (https://arxiv.org/abs/2601.05242).

Each rollout asks the model to call `get_weather` for a city. The verifier scores the
response on three independent `{0, 1}` components:

- `correctness`  — a predicted call matches the expected name and arguments.
- `schema_valid` — the call's arguments parse as a JSON object containing every required
  parameter of the tool.
- `format`       — exactly one tool call was emitted, with no extra assistant text.

These are returned in the `reward_components` field of the verify response (in addition
to the summed scalar `reward`). NeMo-RL's NeMo Gym bridge exposes them as
`reward1, reward2, ...` (ordered by component name: `correctness`, `format`,
`schema_valid`) for the GDPO advantage estimator. A GRPO baseline reads the summed
`reward` and therefore cannot distinguish responses with the same total but different
composition — the advantage collapse GDPO is designed to fix.

The example data can be found in `tool_call_multireward/data/example.jsonl` and is
regenerated with `python resources_servers/tool_call_multireward/create_examples.py`.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
