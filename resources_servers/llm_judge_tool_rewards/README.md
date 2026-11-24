# LLM judging with tool rewards

This training component wraps `equivalence_llm_judge` and retains its task-data
schema, answer extraction, rescue and swap checks. The grader is harness-independent;
tool rewards consume Gym canonical execution observations. Shell and search tool names
and the rate-limit error pattern are configurable. The shipped configuration retains
OpenCode-compatible settings, and OpenCode is the currently validated tool-using pairing.
Other harnesses must supply matching tool settings and execution observations.
Correctness and tool rewards are recorded separately;
malformed, conflicting or incomplete judge replies are masked.

The shared defaults apply only penalties, only on correct answers: failed bash
calls cost 0.05, consecutive duplicate calls cost 0.02, and the third through sixth
bash attempts incur total usage penalties of 0.025, 0.05, 0.075 and 0.10. The usage
penalty stays capped at 0.10. Incorrect answers receive no tool adjustment.
Recognized failed WebSearch HTTP 429 calls incur no failure or duplicate penalty;
the retry after such a failure is also exempt from duplicate penalties.

For fractional grades, opt into proportional penalties under the resources-server settings:

```yaml
tool_rewards:
  penalty_mode: proportional
```

This applies failure, duplicate-call and shell-usage penalties at every positive
correctness score, even with `correct_only: true`. Their combined cost is multiplied
by correctness and capped at that score. For a nominal cost of 0.10, correctness
0.99 produces reward 0.891, while correctness 1.0 produces 0.90. Zero scores incur
no penalty; already-negative verifier scores are preserved without further penalties.
Positive bonuses keep their existing gates and amounts, including exploration bonuses
when `correct_only: false`. The default mode and this generic wrapper's preset remain unchanged.
This can also be used with the generic judge's existing 0.5 partial-credit rescue.

Compose `configs/llm_judge_tool_rewards.yaml` with a separate `training_judge` model
and an environment targeting `llm_judge_with_tools`. Enable
`forward_agent_evidence: true` on that environment and separately on reverification
when recomputing tool rewards. Default judge prompt paths resolve independently
of the component working directory.

The wrapper imports the shared grader and benchmark reward helper; neither the
generic grader nor the visual task schema depends on this component. Dataset rows
use the generic judge's `expected_answer` contract, with no visual metadata required.
See the [harness integration guide](../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx).

## Committed examples

`configs/example.yaml` binds five native-grader examples for strict data validation:

```bash
gym dataset collate --config resources_servers/llm_judge_tool_rewards/configs/example.yaml \
  --output-dir /tmp/llm_judge_tool_rewards-examples --mode example_validation
gym env test --resources-server llm_judge_tool_rewards +should_validate_data=true
```

The committed `data/example_rollouts.jsonl` contains real NVIDIA-hosted
GPT-4o-mini answers and separate GPT-4o-mini judge responses collected on
2026-11-12, scored by this wrapper with its default reward settings. These are
single-turn, text-only examples with no tool calls. Tool penalties, failed-judge
masking, swap passes, and original-response preservation are covered by the
wrapper tests. The example config declares the judge reference for collation;
configure `training_judge` with model connection settings for live judging.

The component and main config were renamed from `opencode_llm_judge_tool_rewards` to
`llm_judge_tool_rewards`. Update imports and config paths; the logical server instance
remains `llm_judge_with_tools`, so environment bindings do not change.
