# swe_if_agents — SWE-bench rollouts with injected instruction-following constraints

A thin wrapper around `responses_api_agents/swe_agents` (the OpenHands SWE agent) for instruction-following (IF) benchmarks and
RL data: every row is a SWE task plus the exact instruction surfaces the episode must run under, and the constraints to grade.
The task outcome reward is untouched; the IF grades are attached to the verify response as `if_constraints`.

## What a row carries (request metadata; values are strings, structured values JSON-encoded)

| key | meaning |
| --- | --- |
| `tool_name_overrides` | JSON object identifier → tool name, the full binding the episode exposes (e.g. `{"BASH_TOOL_NAME": "terminal", ...}`). Exported to nv-OpenHands as `TOOL_NAME_OVERRIDES`; takes precedence over `DIVERSIFY_TOOL_NAMES`. |
| `system_prompt_template_text` | the system prompt (Jinja template text) with the injected instruction; mounted as `system_prompt.j2` for this row |
| `user_prompt_template_text` | the user prompt template with the injected instruction; OpenHands renders it with the task; mounted as `user_prompt.j2` |
| `replay_observation_suffix` | `{"text": ..., "tool_call_id"?: ...}` — an instruction appended to one recorded tool output of a replayed prefix (mid-task injection); attached to that tool message as `observation_suffix`, appended by nv-OpenHands to the regenerated observation |
| `sdg_item` | `{"type", "constraints": [{"id", "verifier_parameter": {"template": "turn_output", "trigger", "obligation", "no_answer"}, "reference_instruction"}], ...}` — what to grade |

A row whose `input` contains a prior trajectory (function calls and outputs) is a continuation: the base agent replays the
prefix through OpenHands' replay mechanism and the model generates from there; `response.output` then holds only the
generated continuation and the grades cover those turns.

## What the wrapper does (all in `app.py`, about a hundred lines)

1. `_maybe_build_replay_messages`: after the base converts the replayed input to chat messages, tags the tool message named
   by `replay_observation_suffix`.
2. `_setup_params`: after the base has built the episode, applies the row's surfaces — validates and sets
   `resolved_tool_name_overrides` (a small hook in `swe_agents`: it is exported as `TOOL_NAME_OVERRIDES` in the agent
   command), writes the two template texts to files in the episode's persistent directory and points
   `resolved_system_prompt_template` / `resolved_user_prompt_template` at them — then rebuilds the agent command so the
   mounts and exports see them.
3. `run`: calls the base, grades `sdg_item.constraints` on the generated turns with `if_constraints/grader.py`, and returns
   `SWEIFVerifyResponse` = the base response plus `if_constraints`. `reward` stays the SWE-bench verdict.

## Grading (`if_constraints/`)

`verifier/` is the constraint verifier's only implementation (registries of matchers, triggers and templates; README and tests
beside it); the design recipe `agentic-if/recipes/if-constraint-design` imports it and holds its specification (`verifier/VERIFIER_SPEC.md`); `grader.py` segments the generated output
into assistant turns (a turn = one message and/or one tool-call group; the final turn is the last one without a tool call),
re-indexes a continuation from zero, and grades every constraint. Each record:

```
{id, trigger, match, no_answer, instruction, n_steps, n_pass, n_silent, step_avg, all_pass, graded_turns, continuation_only,
 steps: [{turn, reward, detail, items}]}  — `items` = the ids of the output items (message id, tool-call id) that form the graded turn, so a step can be found in `response.output` without counting turns
```

Every matcher declares how a silent in-scope turn (a bare tool call with no text) is treated (`no_answer`): a required shape
(`fail`: prefix, exact, fence, JSON, regex, script, minimum bound) fails it, and an episode with no final message fails its
final-message rules once; a no-answer-compliant rule (`ungradable`: ban, maximum bound, sentinel) is simply not graded on it.
`n_silent` counts those turns so a report can show the no-answer rate next to the score. A retired or unknown matcher yields a
not-applicable record with an `error` field; the row is never lost. Aggregation is done downstream by the recipe: CR and SCR
as means over traces — the headline with the matched no-op (`sdg/trace_metrics.py`) and the breakdowns by group and by
no-answer kind (`sdg/score_if.py`).

## Requirements

- `agent_framework: openhands` with an nv-OpenHands checkout that honours `TOOL_NAME_OVERRIDES` and `observation_suffix`:
  branch `swe-if-agents` of the nv-OpenHands fork = `sdevare-nv/nv-OpenHands` @ `7466868e2` plus five commits
  (`configs/swebench_opencode_if.yaml` pins it at the GitHub fork `jialeiwang/nv-OpenHands`, branch `swe-if-agents`, which builds on
  the upstream branch `codex-opencode-tool-parity` of `sdevare-nv/nv-OpenHands`). Two of the five commits are schema fixes, not features: since Gym PR #2456 the model server validates
  chat requests with `extra="forbid"`, and stock nv-OpenHands sends `aws_region_name: null` and `name` on tool messages, both
  outside the OpenAI chat schema; the fork omits them (in `LLM.completion` and in `nemo_gym_client.py`, which the OpenCode and
  Terminus agents use instead of `LLM.completion`).
- `empty_response_retries` (config, default 0; this config sets 2): exported to the agent as `OPENCODE_EMPTY_RESPONSE_RETRIES`.
  The OpenCode agent has no finish tool, so a tool-free reply ends the episode; a reply with neither content nor tool calls
  (a reasoning-only turn, roughly one turn in ten for nemotron-3.5-lightning) would end it by accident. With N > 0 the fork's
  agent re-issues the identical request up to N times (no message is added) and only a still-empty reply reaches the finish path.
- From `swe_agents` on this branch: the hook `resolved_agent_env` (extra environment variables exported to the agent process, used
  here for `TOOL_NAME_OVERRIDES` and `OPENCODE_EMPTY_RESPONSE_RETRIES`), and `_dump_tool_for_replay`, which dumps the
  response tools so they re-validate under the pinned openai 2.44.0 schema (`defer_loading` is a non-nullable bool there).

## Tests

```
python3 responses_api_agents/swe_if_agents/tests/test_hooks.py                    # pure helpers, no gym dependency
python3 responses_api_agents/swe_if_agents/tests/test_if_constraints_grading.py   # grader parity with the offline scorer on two rolled-out batches
```

## Config

`configs/swebench_opencode_if.yaml`: the OpenCodeAgent persona, no YAML-level system prompt (each row ships its own), the
fork pin, `if_grading: true`. Row files are produced by the recipe's `sdg/build_gym_rows.py`.
