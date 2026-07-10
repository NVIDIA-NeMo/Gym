# Description

A single-tool-call environment that returns **multiple reward components**
instead of a single scalar. This pattern is useful both for **evaluation** (profile each
objective independently) and for **multi-objective RL** such as GDPO
(https://arxiv.org/abs/2601.05242).

Each rollout asks the model to call `get_weather` for a city. The verifier scores the
rollout on three independent `{0, 1}` components:

- `correctness`  — a predicted call matches the expected name and arguments.
- `schema_valid` — the call's arguments parse as a JSON object containing every required
  parameter of the tool.
- `format`       — exactly one tool call was emitted, with no extra assistant text.

Each component is surfaced both as a top-level numeric field on the verify response and
inside the `reward_components` field, alongside the summed scalar `reward`.

- **Evaluation**: because the components are top-level numeric fields, NeMo Gym's
  aggregate-metrics step reports an independent pass rate for each one. This shows *how*
  an agent fails (wrong call vs. malformed arguments vs. extra prose) rather than
  collapsing everything into one score.
- **Multi-objective RL**: a GDPO-style algorithm reads the per-objective scores from
  `reward_components` and normalizes each component independently, rather than collapsing
  them into one number. A GRPO baseline instead reads the summed `reward` and therefore
  cannot distinguish rollouts with the same total but different composition — the advantage
  collapse GDPO is designed to fix. How `reward_components` reaches the trainer depends on
  the training framework's NeMo Gym integration.

The example data can be found in `example_tool_call_multireward/data/example.jsonl` and is
regenerated with `python resources_servers/example_tool_call_multireward/create_examples.py`.

## Tutorial

For a walkthrough of the multi-reward pattern — including how the components are surfaced
for both evaluation and multi-objective RL — see the [Multi-Reward Verification](https://docs.nvidia.com/nemo/gym/main/build-verifiers/multi-reward-verification) docs.

# Licensing information
Code: Apache 2.0
Data: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
