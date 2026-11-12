# opencode_math_tool_rewards

Training adapter for `math_with_judge` with optional OpenCode tool penalties. It reuses
the original grader and its dataset schema; image inputs do not require visual
benchmark metadata.

Select `resources_servers: { opencode_math_tool_rewards: { entrypoint: app.py } }` and set
`forward_agent_evidence: true` on the paired `single_agent_turn` environment.
Use the stock OpenCode agent. Each sample can route to a preconfigured environment
through `task_source`.

See [training reward settings](../../benchmarks/opencode_multimodal_benchmarks/README.md#training-rewards-and-row-profiles)
for defaults and overrides. `tool_rewards: null` disables shaping.

## Committed examples

`configs/example.yaml` binds five native-grader examples for strict data validation:

```bash
gym dataset collate --config resources_servers/opencode_math_tool_rewards/configs/example.yaml \
  --output-dir /tmp/opencode_math_tool_rewards-examples --mode example_validation
gym env test --resources-server opencode_math_tool_rewards +should_validate_data=true
```

The committed `data/example_rollouts.jsonl` contains real NVIDIA-hosted
GPT-4o-mini answers collected on 2026-11-12 and scored by this adapter with its
default reward settings. These are single-turn, text-only verifier examples:
no tool calls were requested or invented, so they do not demonstrate tool-use
behavior. Tool penalties and original-response preservation are covered by the
adapter tests; the separate training smoke exercises tool-bearing trajectories.
