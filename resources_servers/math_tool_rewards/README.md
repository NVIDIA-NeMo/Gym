# math_tool_rewards

Training adapter for `math_with_judge` with optional configurable tool penalties. It reuses
the original grader and its dataset schema; image inputs do not require visual
benchmark metadata.

Select `resources_servers: { math_tool_rewards: { entrypoint: app.py } }` and set
`forward_agent_evidence: true` on the paired `single_agent_turn` environment.
The shipped defaults are OpenCode-compatible. Other harnesses must provide canonical
execution observations and set `tool_rewards.shell_tool_name` for one shell, or
`tool_rewards.shell_tool_names` for a nonempty list sharing one count and bonus schedule.
Do not set a singular name together with a non-null list. The failure filter
`invalid_tool_call_names` defaults to the configured shell names unless explicitly set.
A nonzero shell exit must be reported as a failed tool observation. The configured shell
names control counting, duplicate normalization and success bonuses; they do not
infer exit status from tool output.
OpenCode and a local Pi CLI smoke have been exercised. The Pi smoke registered
its shell as `shell` and checked default failure filtering and call-count penalties;
it does not establish sandbox-session support or other harness pairings. Each sample can route to a preconfigured environment
through `task_source`.

See [training reward settings](../../benchmarks/opencode_multimodal_benchmarks/README.md#training-rewards-and-row-profiles)
for defaults and overrides. `tool_rewards: null` disables shaping.

The historical [replay](../opencode_multimodal_benchmarks/evidence/training_tool_rewards_replay.json),
[Pi](../opencode_multimodal_benchmarks/evidence/training_tool_rewards_pi_smoke.json) and
[OpenCode](../opencode_multimodal_benchmarks/evidence/training_tool_rewards_opencode_smoke.json) reports
cover single-shell behavior; they are integration reports, not test fixtures or plural-setting validation.

## Configuration validation

The parent math grader requires a `policy_model` reference even with
`should_use_judge: false`. Compose the shipped config with your model configuration.
For a service-free wiring check, these placeholder values start no server:

```bash
gym env validate --config resources_servers/math_tool_rewards/configs/math_tool_rewards.yaml \
  --model-type vllm_model --strict \
  ++policy_model_name=validation-only ++policy_base_url=http://127.0.0.1:8000/v1 \
  ++policy_api_key=unused
```

## Committed examples

`configs/example.yaml` binds five examples using the parent grader's schema for strict data validation:

```bash
gym dataset collate --config resources_servers/math_tool_rewards/configs/example.yaml \
  --output-dir /tmp/math_tool_rewards-examples --mode example_validation
gym env test --resources-server math_tool_rewards +should_validate_data=true
```

The committed `data/example_rollouts.jsonl` contains real NVIDIA-hosted
GPT-4o-mini answers collected on 2026-11-12 and scored by this adapter with its
default reward settings. These are single-turn, text-only verifier examples:
no tool calls were requested or invented, so they do not demonstrate tool-use
behavior. Tool penalties and original-response preservation are covered by the
adapter tests; the separate training smoke exercises tool-bearing trajectories.

Renamed from `opencode_math_tool_rewards`. Update the component key, import and config paths;
keep logical server-instance names to preserve environment bindings. Reward defaults,
the parent grader's task schemas and saved reward field names are unchanged.
