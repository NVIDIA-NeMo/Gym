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
than one environment concurrently, pre-stage the VM image to avoid concurrent
workers racing on the same download:

```bash
mkdir -p docker_vm_data
cd docker_vm_data
curl -fL --retry 3 -O \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip
rm Ubuntu.qcow2.zip
cd ..
```

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
| `max_steps` | `15` | Maximum observation/action iterations |
| `max_trajectory_length` | `3` | Previous observations retained in Gym-built prompts |
| `sleep_after_execution` | `10.0` | Wait after each action before observing again |
| `concurrency` | `4` | Maximum simultaneous `/run` calls |
| `max_tokens` | `1500` | Default model output limit |
| `temperature`, `top_p` | `1.0`, `null` | Default sampling values |
| `runner_name` | `gym_pyautogui` | Runner contract selected from the registry |
| `task_timeout` | `1800` | Whole-rollout wall-clock limit in seconds |
| `evaluator_disable_gpu` | `true` | Run EasyOCR evaluation on CPU |
| `reward_mode` | `binary` | `binary` or raw OSWorld partial scores |

Per-task `responses_create_params` override YAML sampling defaults. Explicit
CLI overrides have the highest priority.

The Nano Omni overlay uses the tested three-image window, a 4096-token response
limit, a five-second post-action wait, and parser retries. Optional parser-error
feedback, repeated-action guidance, and a pre-DONE checklist are available in
`agent_kwargs` without requiring another agent class.

The response sets `mask_sample=true` when a timeout, evaluator error, or
unfinished max-step rollout makes the reward unreliable.

### Pre-staged task files

Large task inputs can be prepared outside each rollout. Set one of
`OSWORLD_SETUP_CACHE_DIR`, `OW_SETUP_CACHE_DIR`,
`SPREADSHEETBENCH_SETUP_CACHE_DIR`, or `PPTC_SETUP_CACHE_DIR`; matching files
are linked into the per-task cache before `env.reset()`.

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

This adds `model-io-agent.jsonl` and `model-io-transport.jsonl`. Parser events
also state whether retry feedback, the pre-DONE checklist, or repeated-action
recovery was injected. Requests may
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
