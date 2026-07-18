# OSWorld Responses API Agent

The OSWorld agent runs complete desktop-computer tasks through NeMo Gym. Each
`/run` request creates an OSWorld `DesktopEnv`, sends observations to the
configured model, parses and executes actions, invokes OSWorld's inline
evaluator, and returns the trajectory and reward in Gym's Responses API shape.

This directory owns the reusable runtime. Dataset preparation, benchmark
configuration, model-specific overlays, serving recipes, and the full user
guide live in the [OSWorld benchmark](../../benchmarks/osworld/README.md).

## Request and response contract

The rollout collector sends the complete upstream task under
`verifier_metadata.osworld_task`. The runtime passes that task to
`DesktopEnv.reset(task_config=...)` without translating its setup or evaluator
semantics. `responses_create_params` supplies per-rollout sampling overrides.

A completed response includes:

- Gym `reward`, using binary or raw OSWorld reward according to `reward_mode`;
- `mask_sample`, set for infrastructure failures, timeouts, and unfinished
  rollouts whose reward is not suitable for training;
- `verifier_metadata.osworld_score`, `osworld_steps`, completion/error state,
  termination reason, model identity, artifact directory, and proxy provenance;
- one assistant output item per executed model step.

OSWorld evaluates inside `env.evaluate()`, so this agent does not require a
separate OSWorld resources server.

## Runtime components

| File | Responsibility |
| --- | --- |
| `app.py` | Gym server, request validation, model transport, Ray dispatch, response and aggregate metrics |
| `client.py` | `DesktopEnv` lifecycle, cache staging, action execution, evaluation, logging, and artifacts |
| `runner_registry.py` | Runner names, upstream class paths, and default observation/action contracts |
| `adapter_agents.py` | Gym-owned model scaffolds, including `NemotronV3NanoOmniAgent` |
| `action_parser.py` | Gym pyautogui/control-action parsing and validation |
| `proxy.py` | Explicit proxy-task configuration validation and non-secret provenance |

The runtime uses a pinned, unmodified OSWorld dependency. Compatibility code
is opt-in or narrowly scoped in the Gym adapter rather than patched into the
OSWorld checkout.

## Supported runners

`runner_name` selects the model-facing scaffold:

| Runner | Ownership and contract |
| --- | --- |
| `gym_pyautogui` | Gym prompt and Python/pyautogui actions |
| `prompt_agent` and `prompt_agent_*` | Upstream OSWorld `PromptAgent` observation/action variants |
| `pointer_agent` | Upstream PointerAgent planner/executor/verifier loop |
| `m3_agent` | Upstream MiniMax M3 scaffold and protocol |
| `nemotron_v3_nano_omni_agent` | Gym-owned Nemotron 3 Nano Omni scaffold and parser |
| `qwen3_omni_agent` | Upstream Qwen3VL scaffold through Gym model transport |

The benchmark directory contains the model- and runner-specific YAML overlays.
Those examples do not change the generic runtime defaults in this directory.

## Configuration

The base configuration is
[`configs/osworld_agent.yaml`](configs/osworld_agent.yaml). Important fields
are grouped below.

Environment and execution:

- `provider_name`, `container_image`, `headless`, `screen_width`, and
  `screen_height` configure `DesktopEnv`.
- `concurrency` limits simultaneous `/run` requests.
- `max_steps`, `sleep_after_execution`, `step_timeout`, and `task_timeout`
  bound rollout work.
- `cache_dir` is OSWorld's mutable per-run cache; `setup_cache_dir` points to
  the read-only cache populated by benchmark preparation.

Runner and model behavior:

- `runner_name`, `action_space`, and `observation_type` select a registered
  runner contract.
- `env_class_path` and `agent_class_path` allow explicit compatible classes.
- `agent_kwargs` supplies runner-specific constructor options.
- `max_tokens`, `temperature`, and `top_p` provide server defaults; request
  values can override sampling parameters.

Evaluation and operations:

- `reward_mode` is `binary` or `raw`; aggregate metrics always report both
  binary success and raw OSWorld reward rates.
- `evaluator_disable_gpu` prevents evaluator helpers from reserving rollout
  GPU memory.
- `enable_proxy` and `proxy_config_file` apply only to tasks explicitly marked
  `proxy: true`.
- `asset_input_jsonl` lets server startup idempotently fill missing prepared
  assets before accepting work.

See the benchmark guide for complete field semantics, logging controls, model
recipes, VM requirements, and troubleshooting.

## Running the benchmark

The current Gym CLI commands are:

```bash
cd benchmarks/osworld
python3 prepare.py

# Terminal 1: start configured servers.
gym env start

# Terminal 2: collect against those running servers.
gym eval run --no-serve
```

To let Gym start and stop the servers around collection, use `gym eval run`
instead of the two server/collector commands.

The legacy aliases remain compatibility shims but are deprecated:

| Deprecated alias | Current command |
| --- | --- |
| `ng_run` | `gym env start` |
| `ng_collect_rollouts` | `gym eval run --no-serve` |
| `ng_e2e_collect_rollouts` | `gym eval run` |

For data selection, host setup, advanced launchers, model-specific examples,
and expected outputs, use the [benchmark README](../../benchmarks/osworld/README.md).

## Licensing

- Gym adapter code: Apache 2.0.
- OSWorld code and task data retain their upstream licenses. See the benchmark
  README and pinned dependency metadata for details.
