# OSWorld Responses API Agent

The OSWorld agent runs complete desktop-computer tasks through NeMo Gym. The
agent and benchmark stay in the Gym process; only the OSWorld desktop
container/VM lifecycle moves into Gym Docker Sandbox. Each `/run` request
creates an OSWorld `DesktopEnv`, sends observations to the configured model,
parses and executes actions, invokes OSWorld's inline evaluator, and returns
the trajectory and reward in Gym's Responses API shape.

This directory owns the reusable runtime. Dataset preparation, benchmark
configuration, model-specific overlays, serving recipes, and the full user
guide live in the [OSWorld benchmark](../../benchmarks/osworld/README.md).

The OSWorld agent and its vLLM transport use a managed Python interpreter at
the repository's declared Python floor. This matters when Gym is loaded from
a newer checkout inside an older accepted container: the parent CLI may keep
running there, while each isolated server venv is resolved with the compatible
managed interpreter. For offline runs, pre-seed uv's Python install directory
and export `UV_PYTHON_INSTALL_DIR` before `gym env prefetch`.
If the parent Ray cluster is still on a different 3.13 patch, the operator must
either upgrade the whole cluster or explicitly set
`RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor`; Ray otherwise requires an
exact patch match. The relaxed mode is only valid within one Python minor and
must be covered by a real server-registration smoke, not assumed from a
successful resolver.

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
- a schema-v2 `trajectory_contract` and one semantic `(state, action, reward,
  next_state, done)` transition per environment step;
- `trajectory_model_calls`, preserving each materialized prompt, sampled
  action, reward/done linkage, parser outcome, and any available token/logprob
  evidence. Screenshot bytes live once in `media_assets`; prompts reference
  them by ordered `media_id`.

### Semantic trajectory and exact model-call evidence

Trajectory collection is automatic; it is not a training mode. Every runner,
including closed model APIs that do not expose tokens, returns the semantic
contract. `trajectory_contract.capabilities` says which stronger evidence is
available.

For endpoints that return `prompt_token_ids`, `generation_token_ids`, and
`generation_log_probs`, Gym additionally emits `context_compaction_contract`
exact authority. Each materialized model call is independent, so successive
prompts may rewrite any earlier token or media position. Parser retries are
also separate model calls and are not collapsed into the environment step.
NeMo-RL can therefore reconstruct prefix-contiguous physical traces while one
logical rollout retains one reward and advantage.

Training manifests should supply a model-independent caller-owned identity:

```json
{
  "trajectory_identity": {
    "schema_version": 1,
    "group_id": "chrome-task-001",
    "task_id": "task-001",
    "rollout_index": 0,
    "attempt_index": 0
  }
}
```

The trace-aware NeMo-RL launcher derives and stamps `rollout_id` inside that
object and binds a runtime generation contract before training dispatch.
Standalone benchmarks derive a stable identity automatically and still emit
the same semantic contract. A trainer must fail closed unless the identity is
caller-owned, exact evidence is complete, and runtime admission proves the
tokenizer/template/processor contract. The `context_compaction_contract` wire
name is retained for compatibility with the existing NeMo-RL physical-trace
reconstructor; it is evidence capability, not a Gym training switch.

OSWorld continues to evaluate inside `env.evaluate()`. The environment backend
is selectable between OSWorld's provider directly and Gym Sandbox. In the
Sandbox path, OSWorld still owns `DesktopEnv`, setup, controllers, actions,
and evaluators; Gym Sandbox owns only the VM container lifecycle, dynamic
service endpoints, status, and cleanup.

## Runtime components

| File | Responsibility |
| --- | --- |
| `app.py` | Gym server, request validation, model transport, Ray dispatch, response and aggregate metrics |
| `client.py` | `DesktopEnv` lifecycle, cache staging, action execution, evaluation, logging, and artifacts |
| `runner_registry.py` | Runner names, upstream class paths, and default observation/action contracts |
| `adapter_agents.py` | Gym-owned model scaffolds, including `NemotronV3NanoOmniAgent` |
| `trajectory.py` | Model-independent semantic trajectory identity, transitions, and evidence capabilities |
| `exact_trace.py` | Optional exact model-call/token/media evidence for trace-aware trainers |
| `action_parser.py` | Gym pyautogui/control-action parsing and validation |
| `proxy.py` | Explicit proxy-task configuration validation and non-secret provenance |
| `runtime_dependencies.py` | Version/import readiness check and explicit-install remediation for excluded packages |
| `sandbox_desktop_env.py` | Scoped `DesktopEnv` compatibility wiring for the Gym Sandbox backend |
| `sandbox_provider.py` | OSWorld provider contract backed by Gym Sandbox lifecycle and endpoints |

### OSWorld source dependency

This agent intentionally installs the immutable
[`JeffPengCoder/OSWorld`](https://github.com/JeffPengCoder/OSWorld) fork at
commit `4858905d1ddfecc1cee979742d0f113a6d19728e`, as declared in
[`requirements.txt`](requirements.txt). That revision starts from upstream
OSWorld `83e85344` and includes the `nv-gym` provider overlay, proxy-runtime
repair, VLC gateway-auth fallback, and the privileged-setup fix used by the
canonical restricted-home fixture. It also removes the canonical VS Code
theme fixture's undeclared `jq` dependency, feeds the normal guest password to
the Chrome-history ownership repair, scopes desktop-database updates to the
user directory, and lets the missing-Charles infeasible task reach the agent
instead of executing the impossible operation during setup. It is therefore a
runtime dependency of this branch, not an interchangeable link to upstream
OSWorld `main`.

The dependency is consumed as a commit-addressed source archive so uv does not
initialize optional OSWorld submodules. Gym does not mutate the installed
checkout at runtime: additional compatibility behavior remains opt-in or
narrowly scoped in this adapter. Update the fork URL or commit only together
with resolver tests, the restricted-home fixture, and a real OSWorld rollout.

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

### Nemotron response contract

The adapter-owned Nemotron parser requires an explicit `## Code` section, so
it never executes an unrelated code block from prose. `## Thought`,
`## Action`, and `## Code` values may begin on the heading line or the next
line, and Code may be fenced or unfenced. Thought and Action are descriptive
metadata; an explicit, syntactically valid Code section remains executable even
when an Action description is absent. Python is syntax-checked before OSWorld
executes it, and terminal actions require an explicit `success` or `failure`
status.

The current model response must contain Code, but maintained conversation
history deliberately retains only Thought and Action. Thinking-mode assistant
history preserves the model's `<think>...</think>` wrapper. Omitting previously
executed Code matches the validated Nano Omni prompt contract and avoids sending
the same executable payload twice; this behavior is part of the standard
`NemotronV3NanoOmniAgent`, not a run-local subclass or import overlay.

When adding or upgrading a model, capture representative lossless responses
and add focused parser regressions for heading placement, fenced and unfenced
Code, literal newline escaping, reasoning/content separation, tool calls, and
terminal status syntax. Supported formats should remain explicit rather than
recovering executable code from arbitrary prose.

### PromptAgent variants

The registered upstream PromptAgent variants are:

- `prompt_agent_screenshot_pyautogui`
- `prompt_agent_computer_13`
- `prompt_agent_a11y_tree_pyautogui`
- `prompt_agent_a11y_tree_computer_13`
- `prompt_agent_screenshot_a11y_tree_pyautogui`
- `prompt_agent_screenshot_a11y_tree_computer_13`
- `prompt_agent_som_pyautogui`

Runners that need accessibility data enable it when constructing `DesktopEnv`.
Reasoning wrapped in `<think>` or `<thinking>` is removed before actions are
executed.

## Configuration

The base configuration is
[`configs/osworld_agent.yaml`](configs/osworld_agent.yaml). Important fields
are grouped below.

Environment and execution:

- `provider_name`, `container_image`, `headless`, `screen_width`, and
  `screen_height` configure `DesktopEnv`.
- `sandbox_provider` selects a named Gym Sandbox provider configuration;
  `sandbox_spec` supplies the provider-neutral image/resources/entrypoint, and
  `sandbox_vm_path` selects the read-only OSWorld qcow2 base.
  `sandbox_provider_overrides` applies an OSWorld-only recursive delta to the
  selected provider after named configuration resolution. For example, the
  default OpenSandbox delta bounds VM admission retries without shortening the
  shared provider budget used by other Gym workloads.
- `sandbox_require_kvm`, `sandbox_ready_timeout_s`, and
  `sandbox_ready_poll_s` control the OSWorld Sandbox startup gate.
- `concurrency` limits simultaneous `/run` requests.
- `max_steps`, `sleep_after_execution`, `step_timeout`, and `task_timeout`
  bound rollout work. `task_timeout` is the end-to-end Ray attempt deadline,
  covering sandbox creation, environment setup, agent steps, and evaluation;
  it is also checked cooperatively between child steps and applied to Pointer
  model requests. `task_cancel_grace_s` bounds sandbox cleanup before the
  parent force-cancels a worker that remains stuck.
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
  `proxy: true`. `allow_direct_proxy_tasks` preserves the benchmark's direct
  fallback by default on local and Gym Sandbox backends; strict training or
  deployment profiles can set it to `false` to mask those tasks instead.
- `asset_input_jsonl` lets server startup idempotently fill missing prepared
  assets before accepting work.

See the benchmark guide for complete field semantics, logging controls, model
recipes, VM requirements, and troubleshooting.

## Running the benchmark

The current Gym CLI commands are:

```bash
cd benchmarks/osworld
python3 prepare.py \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2

# Explicitly opt in to packages excluded from Gym's shipped environments.
# prepare.py prints these commands with the exact configured venv path.
gym env prefetch
bash ../../responses_api_agents/osworld_agent/install_optional_runtime_deps.sh \
  ../../responses_api_agents/osworld_agent/.venv

# Terminal 1: start configured servers.
gym env start

# Terminal 2: collect against those running servers.
gym eval run --no-serve
```

The installer targets only the managed OSWorld agent venv. It does not modify
the system Python, Gym's root venv, the model server, or the OSWorld VM. The
installer reads the same `uv-torch-backend.txt` marker as `gym env prefetch`,
so `torch` and `torchvision` come from the same CPU/CUDA wheel family. A plain
PyPI `torchvision` install is not equivalent: it can appear version-compatible
with an existing CPU `torch` while failing to load native operators such as
`torchvision::nms`. The managed environment excludes OSWorld's Azure, Aliyun,
and Volcengine VM
provisioning SDKs: this adapter supports direct Docker plus Gym Docker and
OpenSandbox lifecycle, and none of those paths imports the excluded providers.
The pinned OSWorld task/setup/evaluator code remains installed unchanged. The
public `benchmarks/osworld/tools/start_control.sh` wrapper checks that the
required package versions are importable and fails with the exact setup
commands when this explicit step has been omitted. The agent entrypoint repeats
that non-mutating check so a direct `gym env start` also fails early and
actionably; neither path installs packages automatically.

Choose a model-specific agent composition during preparation. For example:

```bash
python3 prepare.py \
  --profile pointer \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2 \
  --policy-base-url https://ANTHROPIC_COMPATIBLE_HOST/v1 \
  --policy-model-name SERVED_OPUS_4_7_MODEL
```

The Docker backend mounts that base as read-only `/System.qcow2`. Reset means
destroying the Sandbox container and recreating it from the base, matching
OSWorld's Docker-provider behavior. Live RAM/device-state snapshots are not
implemented; callers that require them must select a virtualization provider
with an explicit live-snapshot API.

For data selection, host setup, advanced launchers, model-specific examples,
and expected outputs, use the [benchmark README](../../benchmarks/osworld/README.md).

## Licensing

- Gym adapter code: Apache 2.0.
- OSWorld code and task data retain their upstream licenses. See the benchmark
  README and pinned dependency metadata for details.
