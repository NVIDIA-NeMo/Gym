# OSWorld Benchmark

This package prepares and runs the
[OSWorld](https://github.com/xlang-ai/OSWorld) desktop benchmark through NeMo
Gym. It owns the task data, benchmark composition, runner/model overlays,
asset preparation, launch recipes, and operational documentation.

The reusable Responses API runtime lives in
[`responses_api_agents/osworld_agent`](../../responses_api_agents/osworld_agent/README.md).
That README is the source of truth for request/response semantics, supported
runners, agent ownership, parser contracts, and runtime configuration. The
runtime uses an unmodified, pinned OSWorld dependency. In the current deployment
path, Gym Docker Sandbox owns the VM container lifecycle while OSWorld keeps
its setup, action, and evaluator behavior intact.

## Requirements

- Linux x86_64 with Docker 20+ and access to the local Docker daemon.
- 16 GB or more host RAM per concurrent rollout.
- About 30 GB free disk for the Docker image and `Ubuntu.qcow2` cache.
- A reachable vision-language model endpoint. Text-only models cannot act on
  screenshot observations.
- Read/write access to `/dev/kvm` for the Gym Docker Sandbox path.

## Quickstart

From the Gym repository root, enter the benchmark directory and prepare the
five-task default run. The Sandbox path requires an explicit, verified qcow2.
Endpoint values can also be supplied through `POLICY_BASE_URL`,
`POLICY_API_KEY`, and `POLICY_MODEL_NAME`:

```bash
cd benchmarks/osworld
export POLICY_BASE_URL="https://your-vlm-endpoint/v1"
export POLICY_API_KEY="your-key"  # pragma: allowlist secret
export POLICY_MODEL_NAME="your-vlm-model"
python3 prepare.py \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2

gym env start
```

With `gym env start` still active, use a second terminal in the same directory:

```bash
gym eval run --no-serve
```

For supervisor-managed deployments, the equivalent public wrappers are
`tools/start_control.sh` and `tools/run_eval.sh`. They require
`OSWORLD_RUN_ID` and write logs beneath an optional run-root argument.

`prepare.py` validates the committed input and qcow2, records the disk identity,
prefetches setup and evaluator files, and writes a private, gitignored
`env.yaml` containing the config, agent, input, output, cache, and rollout
settings. Hugging Face assets use the official client cache and `HF_TOKEN` when
configured. It keeps an existing env file unless `--force-env` is supplied.
Python component dependencies are installed by `gym env start` from the agent
and model server project files.

Asset preparation is idempotent: `gym env start` checks the same selected JSONL and
shared cache at server startup, then each rollout links only its task's
read-only files into the OSWorld cache. Use `--skip-assets` only to retain
OSWorld's upstream runtime-download behavior. A normal run connects directly;
`OSWORLD_ASSET_PROXY_URL` is an optional fallback used only after an official
Hugging Face download fails.

Prepare and verify the VM image before the first run:

```bash
bash benchmarks/osworld/tools/prepare_osworld_vm.sh
```

The downloader resumes interrupted transfers and verifies the extracted VM by
size and SHA-256. Pass the resulting qcow2 to `prepare.py --vm-path`. Override
`VM_DIR`, `VM_URL`, `VM_SHA256`, or `VM_SIZE_BYTES` only when intentionally
selecting another upstream image.

## Model profiles

Set `runner_name` in the agent config or pass an override to `gym env start`.
See the [agent runtime README](../../responses_api_agents/osworld_agent/README.md#supported-runners)
for the complete runner registry and prompt/action contracts. Prefer selecting
the maintained model/agent composition with `prepare.py --profile`.

### MiniMax M3

The M3 overlay uses OSWorld's `mm_agents.m3.M3Agent`, screenshot observations,
relative coordinates, a 100-step limit, and an 8192-token response limit. Use
the same canonical preparation flow with the M3 profile:

```bash
cd benchmarks/osworld
python3 prepare.py \
  --profile m3 \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2 \
  --policy-base-url http://MODEL_HOST:8000/v1 \
  --policy-model-name SERVED_M3_MODEL
```

This appends `benchmarks/osworld/configs/osworld_sandbox.yaml`, which pins the
OSWorld image digest, requests KVM, publishes all four OSWorld service ports on
dynamic loopback ports, and mounts the qcow2 read-only. Preparation fails before
starting Gym if the VM file is missing; the adapter checks `/dev/kvm` again at
Sandbox startup.

### PointerAgent

Use `benchmarks/osworld/configs/osworld_agent_pointer.yaml` to run OSWorld's PointerAgent. The
configured policy endpoint must support its Anthropic-compatible requests.
If `PARALLEL_API_KEY` is set, PointerAgent can also use its optional web tools;
without a key those tools are disabled automatically.

### Nemotron 3 Nano Omni with vLLM

Start the checkpoint through the deployment layer on a model host with enough
GPU memory. After its OpenAI-compatible endpoint is reachable from the Gym
host, prepare the benchmark with the Nano Omni profile:

```bash
cd benchmarks/osworld
python3 prepare.py \
  --profile nano_omni \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2 \
  --policy-base-url http://MODEL_HOST:8000/v1 \
  --policy-model-name SERVED_NANO_OMNI_MODEL
```

Then use `gym env start` and `gym eval run --no-serve` exactly as shown in the
quickstart.

The default overlay sends the current screenshot plus at most two historical
screenshots and compacts older interactions into text. For an endpoint limited
to one image per request, use the same class and set
`agent_kwargs.max_image_history_length: 1`; no alternate agent class is
required. Qwen3-Omni uses a different tool-call protocol and therefore has its
own `benchmarks/osworld/configs/osworld_agent_qwen3_omni.yaml` overlay.

Nano Omni history contains Thought and Action only; previously executed Code is
not repeated. This prompt contract is implemented directly by the standard
`NemotronV3NanoOmniAgent`, so deployments must not stage a Python subclass or
extend `PYTHONPATH` with a reproduction overlay.

## Multi-environment runs

Set concurrency and data selection during preparation, then use the same two
Gym commands. `--concurrency` controls simultaneous `DesktopEnv` instances;
the selected JSONL controls the task set.

```bash
cd benchmarks/osworld
python3 prepare.py \
  --profile nano_omni \
  --execution-backend gym_sandbox \
  --vm-path /absolute/path/to/Ubuntu.qcow2 \
  --input data/test_all.jsonl \
  --output /absolute/path/to/results/rollouts.jsonl \
  --concurrency 4 \
  --force-env

gym env start
# In a second terminal:
gym eval run --no-serve
```

## Configuration

`benchmarks/osworld/config.yaml` is the single default benchmark config. It
chains the base `responses_api_agents/osworld_agent/configs/osworld_agent.yaml`
runtime with the generic OpenAI-compatible model transport. Advanced model and
runner overlays remain under `benchmarks/osworld/configs/` and should be listed
after the base config so their values win.

The [agent configuration reference](../../responses_api_agents/osworld_agent/README.md#configuration)
documents the shared environment, runner, timeout, cache, proxy, evaluation,
and sampling fields. Per-task `responses_create_params` override YAML sampling
defaults, and explicit CLI overrides have the highest priority.

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
| Direct OSWorld with `provider_name=remote_docker` | `JeffPengCoder/OSWorld` `nv-gym`, pinned to `31b76bf1c4d4e589238b314caa91470afc52651e` or a documented successor. |
| Direct OSWorld with proxy-required tasks | The same `nv-gym` pinned SHA; set `PROXY_CONFIG_FILE` and construct `DesktopEnv(enable_proxy=True)`. |
| Direct OSWorld with both features | The same `nv-gym` pinned SHA provides both independent capabilities. |

For a direct integration of the tested version:

```bash
git clone https://github.com/JeffPengCoder/OSWorld.git
cd OSWorld
git checkout 31b76bf1c4d4e589238b314caa91470afc52651e
```

Use an immutable SHA in a lockfile or deployment manifest. The `nv-gym`
branch is the integration line that follows upstream main, but its tip can
move as new upstream changes are merged.

This pinned revision also prevents OSWorld's Chrome setup DEBUG logging from
serializing the complete worker environment into task artifacts. Model and
proxy credentials must remain runtime secrets and are never useful setup
diagnostics.

It also guards dynamic `tinyproxy` installation against PackageKit holding
APT locks after VM boot. PackageKit is stopped and runtime-masked only during
the install, then restored to its prior state; no custom VM image is required.

The adapter defaults to proxy disabled. Set both variables only for a run that
is allowed to use a proxy:

```bash
export OSWORLD_ENABLE_PROXY=1
export PROXY_CONFIG_FILE=/run/secrets/osworld-proxy.json
gym env start
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

The OSWorld agent accepts explicit boolean values for `OSWORLD_ENABLE_PROXY`
and validates the config before starting work. OSWorld loads the file lazily
only for `proxy: true` tasks, uses the upstream for APT with bounded retry, and
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

The default `prepare.py` flow writes task setup inputs, evaluator cloud files,
and evaluator postconfig downloads to `benchmarks/osworld/.cache/setup` and
records that absolute path in `env.yaml`. Existing externally prepared caches
remain supported by setting one of `OSWORLD_SETUP_CACHE_DIR`, `OW_SETUP_CACHE_DIR`,
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

Set `OSWORLD_TASK_ARTIFACT_ROOT` before `gym env start` to enable per-task
artifacts. Every rollout then receives a collision-safe
`${domain}/${task_id}` directory with:

- `worker.log` and `runtime.log`;
- `traj.jsonl` with task/run identity, observation hashes, model/action steps,
  agent terminal status, evaluator stage, and compact result-file metadata;
- `step_000.png`, `step_001.png`, and subsequent VM observations;
- `vm-exec.jsonl` with controller commands and VM responses;
- `task.json`, `run.json`, `result.json`, and `manifest.json`.

The directory is returned as `verifier_metadata.osworld_artifact_dir`. Leave
`OSWORLD_TASK_ARTIFACT_ROOT` unset to disable these files.

### Full model I/O

For a focused diagnostic run, opt in to exact agent and transport payloads
before starting the canonical Gym control process:

```bash
export OSWORLD_MODEL_IO_LOG="$PWD/results/omni-diagnostic/model-io-agent.jsonl"
export NEMO_GYM_VLLM_TRANSPORT_LOG="$PWD/results/omni-diagnostic/model-io-transport.jsonl"
gym env start
```

This adds `model-io-agent.jsonl` and `model-io-transport.jsonl`. The agent log
includes direct Anthropic Messages calls made by Pointer as well as Gym-routed
calls; credential fields are redacted while model-body content is retained.
Parser events also state whether retry feedback, the pre-DONE checklist, or
repeated-action recovery was injected. Requests may
contain embedded screenshots and prompt content, so these logs can be large
and sensitive. They are disabled by default.

The paths can be set independently with `OSWORLD_MODEL_IO_LOG`,
`NEMO_GYM_VLLM_TRANSPORT_LOG`, and `OSWORLD_VM_EXEC_LOG`.

## Video recording

Set `OSWORLD_RECORD_VIDEO_DIR` before `gym env start`. To select only specific
tasks, also set `OSWORLD_RECORD_VIDEO_TASK_IDS_FILE`. Recording is best-effort
and does not fail the rollout if the VM cannot produce an mp4.

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

Additional inputs may be supplied with `prepare.py --input`; each JSONL row
must follow the same `verifier_metadata.osworld_task` contract as the committed
examples. Generated full datasets are intentionally not committed.

## Troubleshooting

### `uv` is missing inside `gym env start`

`gym env start` starts component servers in non-interactive shells. If `uv` is only
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
