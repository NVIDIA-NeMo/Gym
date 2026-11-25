# RoboLab

Run RoboLab manipulation episodes on Isaac Sim / Isaac Lab with EmberBridge's
OpenCode harness and sealed Nanoverse native grading. Supports both `SDK 1.5`
(RGB) and `open_model` (RGB-D with `core` or `assisted` perception bundles)
across hosted models. Use this experimental benchmark for evaluation.

## Runtime and inputs

The launcher supplies a prepared RoboLab native environment, EmberBridge
runtime, TTC code kernel, and pinned `robolab-condition` backend launcher. Plan
for one compatible CUDA GPU per environment server instance.

Set these variables on the Gym driver:

| Variable | Value |
| --- | --- |
| `ROBOLAB_CASE_MANIFEST` | Absolute path to the staged `robolab.efb-candidate.v1` JSON manifest |
| `ROBOLAB_BACKEND` | Absolute path to the executable `robolab-condition` adapter |
| `ROBOLAB_OUTPUT_ROOT` | Writable directory for attempt condition receipts and logs |
| `ROBOLAB_NATIVE_SOURCE_SHA256` | Expected 64-hex source boundary SHA-256 of the frozen runtime |
| `ROBOLAB_RUNTIME_LOCK` | Frozen runtime lock JSON consumed by `robolab-condition` |
| `NANOVERSE_TTC_MANIFEST`, `NANOVERSE_GUEST_PROGRAMS` | Installed TTC SDK runtime and guest programs |

The manifest binds `integration: robolab`, `seed: 0`, `execution: ttc`,
`oracle: none`, `contact: false`, `perception: false`, and observe-only tools.
Set `sdk.version` to `"1.5"` (`vision: "rgb"`) or `"open_model"`
(`vision: "rgbd"`, `bundle: "core"` or `"assisted"`).

## Evaluate

Configure `policy_model` using a Gym model-server config, then run from the
Gym checkout:

```bash
gym eval prepare --benchmark robolab
gym eval run --benchmark robolab --model-type openai_model \
  --split benchmark --limit 1 --concurrency 1 --output results/robolab-smoke.jsonl
```

When `use_policy_model_relay: true` is configured on `robolab_environment`, the
environment server exposes an internal aiohttp-backed loopback OpenCode
compatibility relay (`/_opencode_relay/{relay_id}/v1/responses`) on its own
FastAPI app that fills missing `input_image.detail` defaults and expands
buffered `response.output_item.done` events into item lifecycle deltas while
forwarding every request through Gym's configured `policy_model` server.

## Results

Gym reward is the audited native score in `[0.0, 1.0]` read from the sealed
`terminal-record.json`, `episode-result.json`, `condition-result.json`, and
`official/episode_results.jsonl` receipts after confirmed process-group cleanup.
Unscored runs report `native_score: null` and their `native_outcome` class.
The external-agent-loop integration uses `environment_servers/robolab` with
`responses_api_agents/robolab_agent` (`resources_server: null`). Run component
tests with:

```bash
pytest environment_servers/robolab/tests responses_api_agents/robolab_agent/tests
```
