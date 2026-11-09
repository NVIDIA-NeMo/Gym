# Proof Refinement Agent

Multi-turn self-correction agent for formal proof verification, paired with `math_formal_lean`.

Each turn generates a proof. A failed proof's `correction_prompt` becomes the next turn's only input, with the task's
`model`, `temperature`, `top_p` and `max_output_tokens`. The loop stops when a proof passes, when no correction prompt
comes back, or after `max_correction_turns` corrections. The result carries `total_turns` and, with
`include_all_attempts`, `all_attempts`: each attempt's input, response, generation, proof status, reward, error
feedback and correction prompt. The response's `usage` covers every turn's model call.

## Two ways to run it

- **Agent sessions** (`math_formal_lean_multi_turn_single_agent_turn.yaml`): `single_agent_turn_legacy` seeds the
  session with `math_formal_lean`'s direct HTTP grant and calls `/v1/responses` once. That activation runs the whole
  loop and checks each attempt but the last allowed one with `/check_proof`. The Environment Server then sends the
  final response to `/verify`, which grades the last attempt and reports `total_turns` and `all_attempts`.
  The seed is refused without a direct HTTP grant when `max_correction_turns > 0`. A crash restarts the episode
  from its input.
  A model call that answers with an HTTP error ends the loop with a failed response that names the status
  (`ng_agent_outcome`) and keeps the earlier attempts. With the Environment Server's default outcome policy, a 400,
  413 or 422 is graded (that attempt scores 0), a 408, 425, 429 or 5xx restarts the episode, and any other status
  fails it. An error whose failure report says the model may already have answered propagates instead, so the
  Environment Server does not give the model another chance. Closing the session returns the activation's model
  calls and `/check_proof` calls as observations.
  This path pairs only with `math_formal_lean`. Composed onto another environment (for example with
  `--agent-type proof_refinement_agent`), the agent runs in agent sessions too, and the episode fails with a 422:
  at the seed without a direct HTTP grant, or at the first check when the resources server has no `/check_proof`.
  To run it single-turn elsewhere, set `max_correction_turns: 0`.
- **Legacy `/run`** (`math_formal_lean_multi_turn.yaml`, behind a `legacy_agent` relay): the agent seeds the
  resources server and calls `/verify` for every attempt.

Both send the model the same inputs and give the same reward.

## Configuration

- `max_correction_turns`: Maximum number of correction attempts (0 = single-turn)
- `include_all_attempts`: Include all attempt records in output for training data
