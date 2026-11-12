# opencode_string_match_tool_rewards

Training adapter for `string_match` with optional OpenCode tool penalties. It reuses
the original grader and its dataset schema; image inputs do not require visual
benchmark metadata.

Select `resources_servers: { opencode_string_match_tool_rewards: { entrypoint: app.py } }` and set
`forward_agent_evidence: true` on the paired `single_agent_turn` environment.
Use the stock OpenCode agent. Each sample can route to a preconfigured environment
through `task_source`.

See [training reward settings](../../benchmarks/opencode_multimodal_benchmarks/README.md#training-rewards-and-row-profiles)
for defaults and overrides. `tool_rewards: null` disables shaping.
