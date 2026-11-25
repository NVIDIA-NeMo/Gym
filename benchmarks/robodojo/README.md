# RoboDojo

Run RoboDojo manipulation episodes on Newton or Isaac Sim with EmberBridge's
OpenCode harness and Nanoverse's native grader. Use this experimental benchmark for evaluation.
The comparison unit is the selected panel, simulator, SDK profile, model and
episode budget.

## Runtime and inputs

The launcher supplies a prepared installation of EmberBridge, Nanoverse and
RoboDojo. Plan for one compatible CUDA GPU per agent instance. The runtime
must provide the EmberBridge CLI, the Nanoverse process supervisor and the
RoboDojo horizon auditor. Install these components and build their simulator
dependencies before starting Gym. RoboDojo code and assets retain their
upstream licenses. [RoboDojo source and license](https://github.com/RoboDojo-Benchmark/RoboDojo)

Set these variables on the Gym driver.

| Variable | Value |
| --- | --- |
| `ROBODOJO_BRIDGE_PYTHON` | Interpreter containing `ember_bridge.cli` |
| `ROBODOJO_SIMULATION_PYTHON` | Interpreter containing the Nanoverse simulator and grading modules |
| `ROBODOJO_BRIDGE_CONFIG` | Absolute path to the operator's base EmberBridge episode JSON |
| `ROBODOJO_PANEL` | Frozen RoboDojo case manifest, including native source and layout checksums |
| `ROBODOJO_CASE_ID` | Optional case selector when preparing a single-slot smoke split |
| `ROBODOJO_VALIDITY` | Optional assignment manifest for pre-registered substitutions |
| `ROBODOJO_TASK_POLICY` | Optional EmberBridge task policy directory |
| `ROBODOJO_OUTPUT_ROOT` | Short writable directory for SDK sessions and their evidence. Defaults to `results/robodojo`. |
| `ROBODOJO_EVIDENCE_EXPORT_ROOT` | Optional job artifact directory for grading receipts and video |
| `ROBODOJO_POOL_CASES` | Optional comma-separated list of 2 to 5 distinct panel cases sharing one simulator pool |
| `ROBODOJO_POOL_LAUNCHER` | Path to the simulator pool launcher paired with `ROBODOJO_POOL_CASES` |
| `ROBODOJO_SHARED_PLANNER_TABLE` | Optional JSON map from `case_id` to an absolute shared motion-planner Unix socket path |
| `ROBODOJO_SHARED_STARTUP_GATE` | Optional directory used with `ROBODOJO_SHARED_STARTUP_GROUP` to synchronize multi-pool GPU startup and verify initial head RGB capture |
| `NANOVERSE_TTC_MANIFEST`, `NANOVERSE_GUEST_PROGRAMS` | Installed TTC SDK runtime and guest programs |

The base JSON selects `integration: robodojo` and the backend through
`simulator_backend`. Use `isaaclab3` for Newton or `isaacsim` for PhysX.
Provide the OpenCode harness command, the simulator and asset paths, and an
ungated TTC SDK profile. Headless Isaac Sim also needs the worker's NVIDIA
graphics libraries and Vulkan and EGL configuration. Validate the configuration
with the installed EmberBridge CLI. Gym binds the
task, evaluation seed, layout, control budget and model-server URL to each
episode. Its default `native_x2` profile lets the actor use up to twice
RoboDojo's standard control limit. The actor receives task instructions and
observations from EmberBridge. Private state, auxiliary policies, gated checks,
and the Decisions service are disabled for this configuration.

## Evaluate

Configure `policy_model` using a Gym model-server config, then run from the
Gym checkout.

```bash
gym eval prepare --benchmark robodojo
gym env validate robodojo
gym eval run --benchmark robodojo --model-type openai_model \
  --split benchmark --limit 1 --concurrency 1 --output results/robodojo-smoke.jsonl
```

Configure EmberBridge's OpenCode harness with its standard `@ai-sdk/openai`
Responses provider. The RoboDojo environment server hosts an internal loopback
compatibility relay on its own FastAPI server (backed by NeMo Gym's global
aiohttp client) that fills missing `input_image.detail` defaults and expands
buffered `response.output_item.done` events into item lifecycle deltas while
forwarding every request through Gym's configured `policy_model` server.

The prepared split contains one row per supplied panel slot. Drop `--limit 1`
to evaluate the complete supplied panel. A scheduler such as Eval Factory may
call the same prepare and run commands with a prebuilt runtime. Gym starts and
stops the actor process as part of each episode.

## Results

Gym reward is the native success flag at the selected control budget. The
rollout also contains the native-limit and extended `Score` and success flags,
slot and assigned case identifiers, a layout hash, the simulator identity,
and paths to the native evidence and video when available. Gym checks the
backend against the simulator's reset receipt before scoring. Compare
results using the same simulator. The aggregate includes both horizon means
and counts of scored and unscored episodes. Missing or unverified simulation
evidence is masked. The engine owns policy replies and simulator trajectories,
which are written alongside the Gym rollout.

The external-agent-loop integration uses `environment_servers/robodojo` with
`responses_api_agents/robodojo_agent` and supplies its own scoring checks. Run
the component tests with `pytest environment_servers/robodojo/tests
responses_api_agents/robodojo_agent/tests`. Static checks and subprocess
fixtures exercise the Gym contract. A representative GPU rollout is required to
qualify any runtime or panel combination.
