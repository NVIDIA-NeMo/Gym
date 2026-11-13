# EQ-Bench 4 Agent

Runs one EQ-Bench 4 roleplay conversation between the policy and a simulated persona, then sends it to the
[`eqbench4` resources server](../../resources_servers/eqbench4/README.md) for judging.

## Conversation

- The policy opens. Each policy turn uses that turn's system prompt from the row's `evaluated_system_prompts` and the
  dialogue so far; the first turn uses the row's `responses_create_params` as prepared (system prompt and opening
  message). Policy calls go through the agent's own `/v1/responses`, under the rollout's capture path.
- The persona model (`persona_model_server`) answers each policy turn. It sees the row's `persona_system_prompt`,
  the conversation with the roles swapped, its own latest reply in full and earlier replies as dialogue only, and
  `[Turn k of n]` before the policy's latest message.
- The returned response is the last policy reply. Its usage sums every model call of the conversation, the policy's
  and the persona's, because Gym's rollout usage covers the rollout's whole invocation tree.
- A persona reply must have a non-empty `<internal>` section and a non-empty `<dialogue>` without control tags.
  Otherwise the agent shows the reply back with `persona_correction_message` and asks again, up to
  `persona_max_attempts` replies; then the conversation ends as `persona_failed`. `<status>ended</status>` (or `end`,
  `done`, `finished`, `stop`, `stopped`) ends it as `persona_ended`.
- A policy reply with no visible text (after removing reasoning, including a reasoning block cut off before it
  closed) is asked again, up to `policy_max_attempts` replies (seven, as in EQ-Bench 4); then the conversation ends as
  `policy_failed` with a `policy_stop_reason` read from the model server's markings of the last reply:
  `context_limit` (the context-window overflow mark; not asked again, since the same prompt overflows again),
  `content_filter` (a refusal, or an incomplete reply cut by a content filter), `output_budget` (an incomplete reply
  cut at `max_output_tokens`), or `empty_reply`. The agent logs a warning with the scenario, turn and cause; the
  resources server leaves the conversation out of the score and reports how many stopped and why.
- Each side speaks at most `num_turns` times (eight in EQ-Bench 4), so a full conversation ends with the persona.

The agent forwards `/aggregate_metrics` to its resources server, which holds the published comparisons the rating needs.

## Trajectory record

With model-call capture on (`++observability_enabled=true ++model_call_capture_dir=<absolute path>`), the result
carries `ng_trajectory`, and a conversation that breaks off publishes the record so far with its failure. The record
has two agent invocations:

- **`root`, the policy.** Its status is `failed` when the policy stopped (with `stop_reason` set to the
  `policy_stop_reason`), `incomplete` when the persona broke the conversation off, and `completed` otherwise.
- **`persona`, the persona simulator.** Gym counts it as a second agent of the episode, as `nemo_user_sim` does its
  simulated user. Its status is `failed` after an unusable reply in every attempt, and `completed` otherwise.

Every model call is one turn of its speaker's invocation, retries of an empty or unusable reply included. A turn names
its call by model server and response id, and holds the call's input and output. The judge calls belong to the
resources server's verifier invocation. Turns carry no tool steps. `resolved` is null, with a `resolution_unavailable`
gap, because the grade is a fraction. The record passes the `gym-p0/v1` evidence gate with `--no-tools`:

```bash
python scripts/inspect_harness_conformance.py --bundle <rollouts.jsonl> --output <dir> --no-tools
```

## Configuration

| Field | Default | Meaning |
| --- | --- | --- |
| `resources_server`, `model_server` | | The EQ-Bench 4 resources server and the policy |
| `persona_model_server` | | The persona simulator |
| `persona_responses_create_params` | temperature 0.7, 10,000 output tokens | Persona sampling |
| `persona_max_attempts` | 3 | Persona replies per persona turn |
| `persona_correction_message` | | Shown after an unusable persona reply; `{problems}` lists what was wrong |
| `policy_max_attempts` | 7 | Policy replies per policy turn |

## Licensing

Code: Apache-2.0. The conversation protocol follows [EQ-bench/eqbench4](https://github.com/EQ-bench/eqbench4) at
revision `93dcb7f5d430433ad49d501e672a98bcff2c9678` (BSD-3-Clause with the Commons Clause condition), implemented
independently.

Dependencies
- nemo_gym: Apache 2.0
