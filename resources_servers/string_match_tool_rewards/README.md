# string_match_tool_rewards

Training adapter for `string_match` with optional configurable tool penalties. It reuses
the original grader and its dataset schema; image inputs do not require visual
benchmark metadata.

Select `resources_servers: { string_match_tool_rewards: { entrypoint: app.py } }` and set
`forward_agent_evidence: true` on the paired `single_agent_turn` environment.
The shipped defaults are OpenCode-compatible. Other harnesses must provide canonical
execution observations and matching `shell_tool_name` and
`invalid_tool_call_names` settings under `tool_rewards`.
OpenCode is the currently validated tool-using pairing. Each sample can route to a preconfigured environment
through `task_source`.

See [training reward settings](../../benchmarks/opencode_multimodal_benchmarks/README.md#training-rewards-and-row-profiles)
for defaults and overrides. `tool_rewards: null` disables shaping.

## Committed examples

`configs/example.yaml` binds five native-grader examples for strict data validation:

```bash
gym dataset collate --config resources_servers/string_match_tool_rewards/configs/example.yaml \
  --output-dir /tmp/string_match_tool_rewards-examples --mode example_validation
gym env test --resources-server string_match_tool_rewards +should_validate_data=true
```

The committed `data/example_rollouts.jsonl` contains real NVIDIA-hosted
GPT-4o-mini answers collected on 2026-11-12 and scored by this adapter with its
default reward settings. These are single-turn, text-only verifier examples:
no tool calls were requested or invented, so they do not demonstrate tool-use
behavior. Tool penalties and original-response preservation are covered by the
adapter tests; the separate training smoke exercises tool-bearing trajectories.

Renamed from `opencode_string_match_tool_rewards`. Update the component key, import and config paths;
keep logical server-instance names to preserve environment bindings. Reward defaults,
native task schemas and saved reward field names are unchanged.
