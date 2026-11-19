# RoboDojo

Run RoboDojo manipulation episodes on Newton with EmberBridge's OpenCode harness
and Nanoverse's native grader. Use this experimental benchmark for evaluation.
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
| `NANOVERSE_TTC_MANIFEST`, `NANOVERSE_GUEST_PROGRAMS` | Installed TTC SDK runtime and guest programs |

The base JSON selects `integration: robodojo`, `simulator_backend: isaaclab3`,
the OpenCode harness command, the simulator and asset paths, and an ungated
TTC SDK profile. Validate it with the installed EmberBridge CLI. Gym binds the
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
  --split benchmark --limit 1 --concurrency 1 --output results/robodojo-smoke.jsonl \
  ++policy_model.responses_api_models.openai_model.responses_stream_item_events=true
```

OpenCode reads item creation and content events in Responses streams. Set
`responses_stream_item_events` on the selected Gym model server when it
buffers the upstream response before emitting the stream.

The prepared split contains one row per supplied panel slot. Drop `--limit 1`
to evaluate the complete supplied panel. A scheduler such as Eval Factory may
call the same prepare and run commands with a prebuilt runtime. Gym starts and
stops the actor process as part of each episode.

## Results

Gym reward is the native success flag at the selected control budget. The
rollout also contains the native-limit and extended `Score` and success flags,
slot and assigned case identifiers, a layout hash, and paths to the native
evidence and video when available. The aggregate includes both horizon means
and counts of scored and unscored episodes. Missing or unverified simulation
evidence is masked. The engine owns policy replies and simulator trajectories,
which are written alongside the Gym rollout.

The external-agent-loop integration uses the existing episode relay and
supplies its own scoring checks. Run the component tests with
`pytest responses_api_agents/robodojo_agent/tests`. Static checks and
subprocess fixtures exercise the Gym contract. A representative GPU rollout
is required to qualify any runtime or panel combination.
