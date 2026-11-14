# mcqa_tool_rewards

Training adapter for `mcqa` with optional configurable tool penalties. It reuses
the original grader and its dataset schema; image inputs do not require visual
benchmark metadata.

Select `resources_servers: { mcqa_tool_rewards: { entrypoint: app.py } }` and set
`forward_agent_evidence: true` on the paired `single_agent_turn` environment.
The shipped defaults are OpenCode-compatible. Other harnesses must provide canonical
execution observations and set `tool_rewards.shell_tool_name` to their shell tool name.
The failure filter `invalid_tool_call_names` defaults to that name unless explicitly set.
A nonzero shell exit must be reported as a failed tool observation. `shell_tool_name`
controls shell counting, duplicate normalization and success bonuses; it does not
infer exit status from tool output.
OpenCode and a local Pi CLI smoke have been exercised. The Pi smoke registered
its shell as `shell` and checked default failure filtering and call-count penalties;
it does not establish sandbox-session support or other harness pairings. Each sample can route to a preconfigured environment
through `task_source`.

See [training reward settings](../../benchmarks/opencode_multimodal_benchmarks/README.md#training-rewards-and-row-profiles)
for defaults and overrides. `tool_rewards: null` disables shaping.

The [saved-rollout comparison](../opencode_multimodal_benchmarks/tests/training_tool_rewards_replay.json)
records all 12 before/after rewards from the OpenCode replay. These are integration
checks, not benchmark scores. The [Pi smoke summary](../opencode_multimodal_benchmarks/tests/training_tool_rewards_pi_smoke.json)
records 10 live synthetic episodes, 16 shell calls and 21 grading checks using
GPT-4o-mini, including real nonzero exits and three-call usage penalties.

## Committed examples

`configs/example.yaml` binds five examples using the parent grader's schema for strict data validation:

```bash
gym dataset collate --config resources_servers/mcqa_tool_rewards/configs/example.yaml \
  --output-dir /tmp/mcqa_tool_rewards-examples --mode example_validation
gym env test --resources-server mcqa_tool_rewards +should_validate_data=true
```

The committed `data/example_rollouts.jsonl` contains real NVIDIA-hosted
GPT-4o-mini answers collected on 2026-11-12 and scored by this adapter with its
default reward settings. These are single-turn, text-only verifier examples:
no tool calls were requested or invented, so they do not demonstrate tool-use
behavior. Tool penalties and original-response preservation are covered by the
adapter tests; the separate training smoke exercises tool-bearing trajectories.

Renamed from `opencode_mcqa_tool_rewards`. Update the component key, import and config paths;
keep logical server-instance names to preserve environment bindings. Reward defaults,
the parent grader's task schemas and saved reward field names are unchanged.
