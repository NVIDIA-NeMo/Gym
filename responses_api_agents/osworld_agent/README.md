# OSWorld Agent

This NeMo Gym responses API agent runs the
[OSWorld](https://github.com/xlang-ai/OSWorld) desktop benchmark. One `/run`
request creates an OSWorld VM, drives a complete observation/action rollout,
runs the task evaluator, and returns the score as a NeMo Gym response.

The adapter supports Gym-built prompts as well as agents supplied by upstream
OSWorld. It
uses an unmodified, pinned OSWorld dependency and the upstream local Docker
provider.

### Agent ownership and terminology

The module `adapter_agents.py` contains the model-specific scaffold owned by
this Gym adapter: `NemotronV3NanoOmniAgent`. It implements prompt construction,
bounded image history with text compaction, response parsing, coordinate
projection, and retry behavior around Gym's model transport. Upstream OSWorld
does not currently define a Nemotron 3 Nano Omni scaffold, so the adapter
provides this class while continuing to use OSWorld's unmodified environment
and evaluator. Upstream-owned agents such as `PromptAgent`, `M3Agent`,
`PointerAgent`, and `Qwen3VLAgent` remain identified explicitly below.

## Requirements

- Linux x86_64 with Docker 20+ and access to the local Docker daemon.
- 16 GB or more host RAM per concurrent rollout.
- About 30 GB free disk for the Docker image and `Ubuntu.qcow2` cache.
- A reachable vision-language model endpoint. Text-only models cannot act on
  screenshot observations.
- `/dev/kvm` is strongly recommended. OSWorld can fall back to software QEMU,
  but VM startup and interaction are substantially slower.

The optional host helper installs Docker, `uv`, and the video-recording tools:

```bash
bash responses_api_agents/osworld_agent/scripts/bringup_local_host.sh
```

For a non-mutating audit of an existing host, run:

```bash
bash responses_api_agents/osworld_agent/scripts/check_host_prerequisites.sh
```

## Quickstart

From the Gym repository root:

```bash
uv sync --extra dev

cat > env.yaml <<'EOF'
policy_base_url: https://your-vlm-endpoint/v1
policy_api_key: <your-key>
policy_model_name: <your-vlm-model>
EOF
chmod 600 env.yaml

ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &

ng_collect_rollouts \
  +agent_name=osworld_simple_agent \
  +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
  +output_jsonl_fpath=results/osworld_example.jsonl \
  +num_repeats=1
```

The first run downloads the OSWorld container and VM image. When running more
than one environment concurrently, pre-stage and verify the VM image to avoid
concurrent workers racing on the same download:

```bash
bash responses_api_agents/osworld_agent/scripts/prepare_osworld_vm.sh
```

The downloader resumes interrupted transfers and verifies the extracted VM by
size and SHA-256. Override `VM_DIR`, `VM_URL`, `VM_SHA256`, or
`VM_SIZE_BYTES` only when intentionally selecting another upstream image.

## Runners

Set `runner_name` in the agent config or pass an override to `ng_run`.

| Runner | Prompt and action contract |
| --- | --- |
| `gym_pyautogui` | Gym prompt; Python/pyautogui code plus `WAIT`, `DONE`, and `FAIL` |
| `prompt_agent` | OSWorld `PromptAgent` defaults: screenshot + accessibility tree, `computer_13` actions |
| `prompt_agent_*` | Explicit PromptAgent observation/action combinations |
| `pointer_agent` | OSWorld PointerAgent planner/executor/verifier loop |
| `m3_agent` | OSWorld MiniMax M3 prompt, parser, and Anthropic-compatible transport |
| `nemotron_v3_nano_omni_agent` | Nemotron 3 Nano Omni prompt, bounded image history, parser, and coordinate projection |
| `qwen3_omni_agent` | OSWorld Qwen3VL scaffold with Gym model transport |

The adapter-owned Nemotron parser requires an explicit `## Code` section so it
never executes an unrelated code block from prose. Within that boundary it
accepts common equivalent model formats: `## Thought`, `## Action`, and
`## Code` values may start on the heading line or the following line; Code may
be fenced or unfenced. Thought and Action are descriptive metadata, so an
explicit, syntactically valid Code section is not discarded merely because an
Action description is absent. Python actions are still syntax-checked before
OSWorld executes them, and terminal actions still require an explicit
`success` or `failure` status.

When adding or upgrading a model, do not assume it serializes this protocol in
the same way as an existing checkpoint. Capture representative lossless raw
responses, including failures, then add focused parser regression cases before
accepting a new layout. In particular, check heading placement, fenced versus
unfenced Code, literal newline escaping, reasoning/content separation, tool
calls, and terminal status syntax. Extend only explicit, unambiguous formats;
do not make the parser recover executable code from arbitrary prose. Full
model-I/O logging for this investigation is described below.

The available PromptAgent variants are:

- `prompt_agent_screenshot_pyautogui`
- `prompt_agent_computer_13`
- `prompt_agent_a11y_tree_pyautogui`
- `prompt_agent_a11y_tree_computer_13`
- `prompt_agent_screenshot_a11y_tree_pyautogui`
- `prompt_agent_screenshot_a11y_tree_computer_13`
- `prompt_agent_som_pyautogui`

Runners that require accessibility data enable it when constructing
`DesktopEnv`. Reasoning wrapped in `<think>` or `<thinking>` is removed before
actions are executed.

### Native PromptAgent smoke

```bash
bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

Use `RUNNER_NAME`, `LIMIT`, `START_NG_RUN`, or `DRY_RUN` to change the smoke
run without editing the script.

### MiniMax M3

The M3 overlay uses OSWorld's `mm_agents.m3.M3Agent`, screenshot observations,
relative coordinates, a 100-step limit, and an 8192-token response limit.

```bash
LIMIT=4 NUM_ENVS=1 \
  bash responses_api_agents/osworld_agent/scripts/run_m3_multienv.sh
```

### PointerAgent

Use `configs/osworld_agent_pointer.yaml` to run OSWorld's PointerAgent. The
configured policy endpoint must support its Anthropic-compatible requests.
If `PARALLEL_API_KEY` is set, PointerAgent can also use its optional web tools;
without a key those tools are disabled automatically.

### Nemotron 3 Nano Omni with vLLM

Start the checkpoint on a model host with enough GPU memory. The launcher
defaults to tensor parallel size 1; set `TENSOR_PARALLEL_SIZE` to use multiple
GPUs.

```bash
bash responses_api_agents/osworld_agent/scripts/launch_omni_mini_vllm.sh
```

The launcher pins the checkpoint revision and requires vLLM 0.20.0 by default.
After the endpoint is reachable from the rollout host:

```bash
OMNI_MINI_VLLM_BASE_URL=http://model-host:8000/v1 \
  bash responses_api_agents/osworld_agent/scripts/run_omni_mini_vllm.sh
```

The runner script performs a real image request before starting Gym. It uses
the committed five-task example by default. Override `INPUT_JSONL`, `LIMIT`,
`NUM_ENVS`, `RUN_DIR`, or `RESUME_FROM_CACHE` for larger runs.

The default overlay sends the current screenshot plus at most two historical
screenshots and compacts older interactions into text. For an endpoint limited
to one image per request, use the same class and set
`agent_kwargs.max_image_history_length: 1`; no alternate agent class is
required. Qwen3-Omni uses a different tool-call protocol and therefore has its
own `configs/osworld_agent_qwen3_omni.yaml` overlay.

## Multi-environment runs

`run_multienv_osworld_agent.sh` starts the configured services, waits for
readiness, and invokes `ng_collect_rollouts`. `NUM_ENVS` controls parallel
`DesktopEnv` instances; `LIMIT` controls total input rows.

```bash
RUNNER_NAME=prompt_agent \
POLICY_MODEL_NAME=<your-model> \
NUM_ENVS=4 \
LIMIT=8 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

For resumable benchmark runs, keep the input ordering, output path, repeat
count, and sampling settings stable:

```bash
INPUT_JSONL=responses_api_agents/osworld_agent/data/test_all.jsonl \
LIMIT=null NUM_ENVS=4 RESUME_FROM_CACHE=1 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

For Internal/Gym parity runs, fail before launch if the materialized task
instruction, setup order, or evaluator differs. The JSON report is retained in
the run directory:

```bash
TASK_PARITY_REFERENCE_INPUT=/path/to/canonical.jsonl \
TASK_PARITY_IDS_FILE=/path/to/task_ids.txt \
INPUT_JSONL=/path/to/gym.jsonl \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

## Configuration

`configs/osworld_agent.yaml` defines the base `osworld_simple_agent`. Model
overlays should be listed after the base config so their values win.

| Field | Default | Meaning |
| --- | --- | --- |
| `provider_name` | `docker` | Upstream OSWorld VM provider |
| `headless` | `true` | Run without forwarding a desktop window |
| `screen_width`, `screen_height` | `1920`, `1080` | VM resolution |
| `require_a11y_tree` | `false` | Include AT-SPI accessibility data |
| `enable_proxy` | `false` | Global opt-in; only a task with `proxy: true` uses it |
| `proxy_config_file` | `null` | Local upstream-proxy JSON path; `PROXY_CONFIG_FILE` overrides it |
| `max_steps` | `15` | Maximum observation/action iterations |
| `max_trajectory_length` | `3` | Previous observations retained in Gym-built prompts |
| `sleep_after_execution` | `10.0` | Wait after each action before observing again |
| `concurrency` | `4` | Maximum simultaneous `/run` calls |
| `max_tokens` | `1500` | Default model output limit |
| `temperature`, `top_p` | `1.0`, `null` | Default sampling values |
| `runner_name` | `gym_pyautogui` | Runner contract selected from the registry |
| `task_timeout` | `1800` | Whole-rollout wall-clock limit in seconds |
| `docker_port_lock_timeout` | `300.0` | Maximum wait for concurrent Docker VM port allocation |
| `evaluator_disable_gpu` | `true` | Run EasyOCR evaluation on CPU |
| `reward_mode` | `binary` | `binary` or raw OSWorld partial scores |

Per-task `responses_create_params` override YAML sampling defaults. Explicit
CLI overrides have the highest priority.

The Nano Omni overlay uses the tested three-image window, a 4096-token response
limit, a five-second post-action wait, and parser retries. Optional parser-error
feedback, repeated-action guidance, and a pre-DONE checklist are available in
`agent_kwargs` without requiring another agent class.

Its `temperature=0.6` and `top_p=0.95` values follow NVIDIA's thinking-mode
recommendation in the
[checkpoint model card](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/24e67ea000b7c2837fc8f9488aa2008524fac8ba/README.md#model-parameters).
The 4096-token output limit is the tested OSWorld agent setting rather than the
model card's longer general-purpose reasoning budget; these values are
model-specific and do not change defaults for other runners.

The response sets `mask_sample=true` when a timeout, evaluator error, or
unfinished max-step rollout makes the reward unreliable.

### Proxy-required tasks

Proxy policy belongs to the Gym `feature/osworld` adapter; VM setup belongs to
the pinned OSWorld `nv-gym` runtime. The branch names are independent Git refs
in different repositories. Gym connects them only through the immutable
OSWorld commit in `pyproject.toml`. That OSWorld commit merges upstream main
`83e85344` and retains the `nv-gym` integration overlay.

#### OSWorld version selection

| Consumer workflow | Required OSWorld version |
| --- | --- |
| Gym `feature/osworld` | No manual checkout. The agent package installs the exact SHA from `pyproject.toml`. |
| Direct OSWorld, plain Docker/VMware, no proxy-required tasks | Upstream xlang OSWorld main is sufficient; this adapter's pre-fix baseline was `83e8534451ba8b3ab6477448ef3f0a8e563f05be`. |
| Direct OSWorld with `provider_name=remote_docker` | `JeffPengCoder/OSWorld` `nv-gym`, pinned to `95dc83aad1670b8ef399d5e37e696619417dafd8` or a documented successor. |
| Direct OSWorld with proxy-required tasks | The same `nv-gym` pinned SHA; set `PROXY_CONFIG_FILE` and construct `DesktopEnv(enable_proxy=True)`. |
| Direct OSWorld with both features | The same `nv-gym` pinned SHA provides both independent capabilities. |

For a direct integration of the tested version:

```bash
git clone https://github.com/JeffPengCoder/OSWorld.git
cd OSWorld
git checkout 95dc83aad1670b8ef399d5e37e696619417dafd8
```

Use an immutable SHA in a lockfile or deployment manifest. The `nv-gym`
branch is the integration line that follows upstream main, but its tip can
move as new upstream changes are merged.

The adapter defaults to proxy disabled. Set both variables only for a run that
is allowed to use a proxy:

```bash
OSWORLD_ENABLE_PROXY=1 \
PROXY_CONFIG_FILE=/run/secrets/osworld-proxy.json \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

The JSON file is local runtime configuration, not a repository asset. It is a
non-empty list of HTTP upstreams. Each entry requires `host` and `port`; omit
both `username` and `password` for an unauthenticated proxy, or provide both
for an authenticated proxy. No cluster-specific hostname is hard-coded.

| Task `proxy` | Global switch | Config | Result |
| --- | --- | --- | --- |
| `false` | `0` | absent | Normal direct-network task |
| `false` | `1` | valid | Normal task; no VM proxy is installed for this task |
| `true` | `0` | any | Masked infrastructure result: `proxy_required_but_disabled` |
| `true` | `1` | valid | OSWorld installs/uses the VM-local proxy |
| `true` | `1` | absent or invalid | Masked infrastructure result: `proxy_configuration_error` |

An exception while the proxy-marked task is resetting is reported as
`proxy_setup_error`, also with `mask_sample=true`. These cases are never
silently counted as formal reward-zero samples. Successful result metadata
records whether proxy was required, enabled, and configured, but never stores
credentials.

The multi-environment launcher accepts exactly `OSWORLD_ENABLE_PROXY=0` or
`1`, validates the config before starting Gym, and records the switch, config
path, SHA-256, and entry count in `run.env`. OSWorld loads the file lazily only
for `proxy: true` tasks, uses the upstream for APT with bounded retry, and
launches Chrome through the loopback-only `127.0.0.1:18888` tinyproxy endpoint.

Task setup commands may optionally declare `expected_returncodes` and
`on_nonzero: score_zero`. These fields express evaluator/setup semantics rather
than model behavior: an allowed return code continues normally, while
`score_zero` records a valid evaluator score of zero instead of masking the
rollout as a harness failure. Tasks that omit both fields continue through the
pinned upstream setup implementation unchanged.

The Docker port-allocation lock wait is configurable because concurrent VM
startup can legitimately take longer than the pinned upstream default. Raising
the wait changes only infrastructure failure handling; it does not change
observations, actions, prompts, or evaluator scoring.

### Pre-staged task files

Large task inputs can be prepared outside each rollout. Set one of
`OSWORLD_SETUP_CACHE_DIR`, `OW_SETUP_CACHE_DIR`,
`SPREADSHEETBENCH_SETUP_CACHE_DIR`, or `PPTC_SETUP_CACHE_DIR`; matching files
are linked into the per-task cache before `env.reset()`.

The pre-staged cache may be empty. Missing task files follow OSWorld's normal
download path and are cached on demand; prewarming affects speed and resilience
to transient download failures, not task semantics.

### Binary and raw OSWorld metrics

OSWorld aggregate metrics report both `osworld/binary_success_rate` and
`osworld/raw_reward_rate`. Binary success counts only evaluator scores of 1.0;
raw reward preserves fractional evaluator credit. Both rates use the same
completed-rollout denominator and are reported regardless of `reward_mode`,
which continues to control the training reward returned by each rollout.

## Logs and artifacts

`run_multienv_osworld_agent.sh` enables `${RUN_DIR}/task-artifacts` by default.
Every rollout receives a collision-safe `${domain}/${task_id}` directory with:

- `worker.log` and `runtime.log`;
- `traj.jsonl` with task/run identity, observation hashes, model/action steps,
  agent terminal status, evaluator stage, and compact result-file metadata;
- `step_000.png`, `step_001.png`, and subsequent VM observations;
- `vm-exec.jsonl` with controller commands and VM responses;
- `task.json`, `run.json`, `result.json`, and `manifest.json`.

The directory is returned as `verifier_metadata.osworld_artifact_dir`. Set
`TASK_ARTIFACTS=0` to disable it, or set `OSWORLD_TASK_ARTIFACT_ROOT` when
starting `ng_run` directly.

### Full model I/O

For a focused diagnostic run, opt in to exact agent and transport payloads:

```bash
FULL_MODEL_IO=1 RUN_DIR=results/omni-diagnostic \
  bash responses_api_agents/osworld_agent/scripts/run_omni_mini_vllm.sh
```

This adds `model-io-agent.jsonl` and `model-io-transport.jsonl`. The agent log
includes direct Anthropic Messages calls made by Pointer as well as Gym-routed
calls; credential fields are redacted while model-body content is retained.
Parser events also state whether retry feedback, the pre-DONE checklist, or
repeated-action recovery was injected. Requests may
contain embedded screenshots and prompt content, so these logs can be large
and sensitive. They are disabled by default.

The paths can be set independently with `OSWORLD_MODEL_IO_LOG`,
`OSWORLD_TRANSPORT_IO_LOG`, and `OSWORLD_VM_EXEC_LOG`.

## Video recording

Set `OSWORLD_RECORD_VIDEO_DIR` before `ng_run`, or use `RECORD_VIDEO=all` or
`RECORD_VIDEO=sample` with the multi-environment runner. Recording is
best-effort and does not fail the rollout if the VM cannot produce an mp4.

Schema-v2 events carry `run_id`, `adapter`, `task_id`, `domain`,
`task_attempt`, logical `step`, and `parse_attempt` in addition to the event,
call, timestamp, and process identifiers. The agent passes that identity to
the transport logger in HTTP headers; it is not inserted into the model JSON
body. Embedded image data remains in the full request; a separate image index
records encoded/decoded sizes and SHA-256 values for integrity checks. These
files can be large and can contain screenshots or prompt content, so keep the
option disabled for normal runs and apply the same access controls as the
source task data.

## Datasets

The repository includes:

- `data/example.jsonl`: five representative smoke tasks;
- `data/example_rollouts.jsonl`: five sample rollout responses;
- `data/test_small.jsonl`: the 39-task OSWorld smoke subset.

Generate other upstream manifests with `paper_to_gym_jsonl.py`:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /path/to/OSWorld
python responses_api_agents/osworld_agent/scripts/paper_to_gym_jsonl.py \
  --osworld-root /path/to/OSWorld \
  --manifest test_all \
  --output responses_api_agents/osworld_agent/data/test_all.jsonl
```

The converter supports `test_all`, `test_small`, `test_infeasible`, and
`test_nogdrive`. Generated full datasets are intentionally not committed.

## Troubleshooting

### `uv` is missing inside `ng_run`

`ng_run` starts component servers in non-interactive shells. If `uv` is only
on an interactive-shell path, expose it on the system path:

```bash
sudo ln -sf "$(command -v uv)" /usr/local/bin/uv
sudo ln -sf "$(command -v uvx)" /usr/local/bin/uvx
```

### The first screenshot times out

The VM image may still be downloading or the guest may still be booting. Check
the task `worker.log`, pre-stage `Ubuntu.qcow2`, and increase `task_timeout`
for slow software-emulated hosts.

### aarch64 hosts

The default `happysixd/osworld-docker` image is x86_64-only. Use an x86_64
rollout host or provide a compatible OSWorld image and provider configuration.

## Licensing

- Adapter code: Apache-2.0.
- OSWorld tasks and dependency: see the upstream OSWorld repository.
