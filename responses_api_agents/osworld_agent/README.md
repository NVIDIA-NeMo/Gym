# OSWorld Agent

A NeMo Gym responses API agent that wraps the [OSWorld](https://github.com/xlang-ai/OSWorld)
desktop-environment harness so language models can be evaluated on the
full OSWorld benchmark (~369 real desktop-computer tasks across Chrome,
LibreOffice, GIMP, VS Code, VLC, Thunderbird, multi-app workflows, and
OS-level tasks).

## Table of Contents

- [Overview](#overview)
- [Deployment Modes](#deployment-modes)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [First-Run Cache Acquisition](#first-run-cache-acquisition)
- [Usage](#usage)
- [Configuration](#configuration)
- [Recording rollout videos](#recording-rollout-videos)
- [Troubleshooting](#troubleshooting)
- [Reward Profiling](#reward-profiling)
- [Dataset Information](#dataset-information)
- [Licensing](#licensing)

## Overview

OSWorld ships with a complete agent harness: a VM provider (Docker /
Apptainer / VMware / VirtualBox / AWS / Azure), a multi-step
observation→action loop, and a per-task evaluator. This agent wraps
that harness at the `/run` boundary — one request runs one full rollout
and returns the evaluator's score as a NeMo Gym `BaseVerifyResponse`.

The shape mirrors `mini_swe_agent`: there is no paired
`resources_servers/osworld/` because the evaluator runs inline inside
`DesktopEnv.evaluate()`. Each rollout is dispatched to a Ray worker so
many tasks can run concurrently against the same model server.

By default this wrapper keeps the existing Gym-built prompt path:
`gym_pyautogui` asks the model for ```python```/pyautogui code blocks,
plus the sentinel tokens `WAIT`, `DONE`, `FAIL`. For closer parity with
OSWorld's own harness, set `runner_name=prompt_agent` or one of the
explicit `prompt_agent_*` runners. That path instantiates upstream
`mm_agents.agent.PromptAgent`, lets it build prompts and parse actions,
and only routes its model call to the configured Gym policy endpoint.
The unqualified `prompt_agent` mirrors OSWorld's default
`screenshot_a11y_tree` observation mode with `computer_13` actions.
`<think>` / `<thinking>` blocks are stripped before execution so
reasoning models work out of the box.

### What happens under the hood per rollout

```
ng_collect_rollouts
   │
   ▼  POST /run  {instruction + osworld_task spec}
┌──────────────────┐
│ OSWorldAgent     │  (asyncio.Semaphore(concurrency))
└────────┬─────────┘
         │  ray.remote SPREAD across worker nodes
         ▼
┌──────────────────────────────────────────────────────────┐
│ run_osworld_task (sync, one Ray task per rollout)        │
│                                                          │
│   1. DesktopEnv(provider_name=…)                         │
│      - Apptainer: `apptainer instance start desktop.sif` │
│      - Docker:    `docker run happysixd/osworld-docker`  │
│                                                          │
│   2. env.reset(task_config)                              │
│   3. Loop k ≤ max_steps:                                 │
│        observe screenshot/a11y ─► policy model           │
│        ──► pyautogui code or computer_13 action ─► step  │
│        parse for DONE / FAIL / WAIT                      │
│   4. env.evaluate()  → float in [0, 1]                   │
│   5. env.close() (instance / container teardown)         │
└──────────────────────────────────────────────────────────┘
         │
         ▼
{reward, mask_sample, response, verifier_metadata{osworld_score, …}}
```

## Deployment Modes

OSWorld can run in three configurations. Pick based on your hardware and
scale needs. **For trying it out, use Mode A (local docker)** — `ng_run` +
`ng_collect_rollouts` work out of the box with no SSH, no remote host, no
helper scripts.

### Mode A — Single machine, local docker (recommended for first run)

Controller (`ng_run` + agent) and the docker VM host are the same box. The
local Docker daemon is used directly via its unix socket; no SSH, no remote
setup.

```
┌─────────────────────────────────────────────────────────────────┐
│  Your machine                                                    │
│                                                                  │
│  ng_run + agent  ──── docker run ───►  Docker daemon             │
│       │                                  (/var/run/docker.sock)  │
│       │                                       │                  │
│       │                                       ▼                  │
│       │                       ┌──────────────────────────────┐   │
│       └─── localhost:5000 ───►│ container: qemu + Ubuntu VM  │   │
│            localhost:9222     │   pyautogui_server :5000     │   │
│            localhost:8006     │   chromium debug :9222       │   │
│            localhost:5900     │   noVNC :8006                │   │
│                               │   VLC :8080                  │   │
│                               │   raw RFB :5900              │   │
│                               └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Prereqs**: Docker, ~15 GB disk, `/dev/kvm` (optional, ~10× faster), a
VLM API key.

**One-time setup**:

```bash
# Recommended one-shot host prep — installs docker + uv + ffmpeg/xvfb/
# tigervnc (for video recording) + symlinks uv into /usr/local/bin so
# ng_run's `bash -c` subshells can find it. Idempotent.
bash responses_api_agents/osworld_agent/scripts/bringup_local_host.sh
```

Or the bare-minimum manual equivalent:

```bash
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER  # re-login required for group to take effect
```

**In `configs/osworld_agent.yaml`**:

```yaml
provider_name: docker
```

This is the simplest path. OSWorld's `docker` provider automatically:

- Downloads `Ubuntu.qcow2` from HuggingFace on first run (~12 GB, cached after)
- Pulls `happysixd/osworld-docker:latest` (~80 MB)
- Manages container lifecycle per rollout (start/stop)

Jump to [Quickstart](#quickstart) for the actual commands.

### Mode B — Remote machine (controller + docker host on different machines)

Controller node SSHes to a remote host that runs the docker container.
Useful when:

- Controller nodes can't run docker locally (security policy, no
  `/dev/kvm`, nested-virt limits)
- You have many cheap CPU controllers + a few specialized KVM-capable hosts
- A lease/reservation model is used for VM hosts (HPC clusters, cloud
  bare-metal)

**Required — install the forked OSWorld that ships the `remote_docker`
provider**. This provider is not in upstream OSWorld yet; we maintain it on
a branch of a fork until the PR is merged upstream:

- Fork: <https://github.com/JeffPengCoder/OSWorld>
- Branch: [`nv-gym`](https://github.com/JeffPengCoder/OSWorld/tree/nv-gym)

In `responses_api_agents/osworld_agent/pyproject.toml`:

```
osworld @ git+https://github.com/JeffPengCoder/OSWorld.git@nv-gym
```

Once the upstream PR lands you can switch back to plain
`xlang-ai/OSWorld`. Mode A (local docker) and any other built-in
upstream provider do **not** need this fork.

**Architecture**:

```
┌────────────────────────┐                  ┌──────────────────────────────┐
│  Controller node       │                  │  Remote host (e.g. clusterX) │
│  (ng_run + agent)      │                  │  (docker + qemu)             │
│                        │                  │                              │
│   1. ssh ──────────────┼─── outbound ─────┼─►  sshd:22                   │
│   (single ControlMaster session)          │      ↓ (spawn docker run)    │
│                        │                  │  ┌─────────────────────────┐ │
│   2. ssh -L tunnels ───┼──── same socket ─┼─▶│ docker container        │ │
│      (controller opens │                  │  │  qemu + Ubuntu VM       │ │
│       local ports; ssh │                  │  │  pyautogui_server :5000 │ │
│       multiplexes them │                  │  │  chromium debug :9222   │ │
│       through the      │                  │  │  noVNC :8006            │ │
│       single TCP)      │                  │  │  raw RFB :5900          │ │
│   localhost:5000       │                  │  │                         │ │
│   localhost:9222   ◄───┼──────────────────┼──┤  (all traffic via the   │ │
│   localhost:8006       │   (responses     │  │   same ssh socket)      │ │
│   localhost:5900       │    flow back     │  └─────────────────────────┘ │
│                        │    through the   │                              │
│                        │    same socket,  │   3. mp4 written to /tmp     │
│                        │    multiplexed)  │      on remote (recording   │
│   3. scp ◄─────────────┼─── same socket ──┼──   enabled), fetched on     │
│                        │                  │      task end                │
└────────────────────────┘                  └──────────────────────────────┘
       SSH session = ONE outbound TCP        Remote sshd accepts; never
       (firewall: controller → remote:22)    initiates back to controller
```

Key properties:

- **All traffic on ONE outbound TCP socket**. Firewall only needs to allow
  `controller → remote:22`. No other ports.
- **Remote sshd accepts; never initiates back** to the controller. No
  reverse callbacks, webhooks, or push messages — controller polls.
- `pyautogui_server` / `noVNC` / raw RFB / chromium-debug all flow through
  the SAME ssh `ControlMaster` session, multiplexed by the SSH protocol.
- mp4 (if `OSWORLD_RECORD_VIDEO_DIR` is set) is recorded on the remote
  host's `/tmp/` and pulled back via `scp` at task end (still in the same
  ssh socket).

**One-time setup on the remote host**:

```bash
# Idempotent helper (recommended) — does ssh-key push if needed, installs
# docker + ffmpeg + xvfb + tigervnc-viewer, stages qcow2, docker-pulls the
# image, and runs a pre-flight verify, all in one shot.
bash responses_api_agents/osworld_agent/scripts/bringup_remote_host.sh \
    user@remote-host /path/to/Ubuntu.qcow2
```

Or, if you'd rather run the steps manually:

```bash
ssh user@remote-host "
  sudo apt-get install -y docker.io ffmpeg xvfb tigervnc-viewer
  sudo usermod -aG docker \$USER
  sudo systemctl enable --now docker
  docker pull happysixd/osworld-docker:latest
  mkdir -p ~/osworld-assets
"
# Stage the Ubuntu.qcow2 once (multi-GB):
scp Ubuntu.qcow2 user@remote-host:~/osworld-assets/
```

**On the controller**:

```bash
export OSWORLD_REMOTE_HOST=user@remote-host
# export OSWORLD_REMOTE_SSH_KEY=~/.ssh/id_ed25519  # optional; defaults to first existing id_*
```

**In `configs/osworld_agent.yaml`** (or via Hydra override on the CLI):

```yaml
provider_name: remote_docker
```

See the upstream `RemoteDockerProvider` README in
`desktop_env/providers/remote_docker/README.md` for the full env-var
reference, troubleshooting, and per-rollout port-allocation details.

**Why SSH and not plain HTTP / docker-over-TCP?** Technically SSH is not
required — alternatives include exposing the docker daemon over TLS
(`tcp://host:2376`), or pre-deploying a long-lived container behind an
HTTP reverse proxy. In practice SSH is simpler: every Linux host ships
with `sshd`; one port (22) handles command exec + port forwarding + file
transfer; no PKI / certificate management; no nginx/HAProxy config.
docker-over-TCP needs TLS PKI to be safe, and the HTTP-reverse-proxy
approach loses per-task container isolation.

## Prerequisites

### Compute environment

- Linux x86_64 (the upstream `happysixd/osworld-docker` image is
  x86_64; aarch64 needs a separately-built image — see
  [Troubleshooting](#aarch64-clusters)).
- One of:
  - **Apptainer 1.3+** (recommended on HPC/SLURM clusters), OR
  - **Docker 20+** with daemon access (recommended on standalone Linux).
- Per-rollout compute: 1 CPU node, 16+ GB RAM (configurable via
  `mem_limit_mb`). Hardware virtualization (`/dev/kvm`) is optional but
  gives ~10× rollout speed-up vs software qemu.
- Disk: ~30 GB free for the cache (one-time, persistent).
- Network: outbound HTTPS to `docker.io` (for the OSWorld VM image) and
  `huggingface.co` (for the Ubuntu desktop qcow2) on first run.

### Software

- NeMo Gym installed at the repo root (`uv sync --extra dev`).
- A vision-language policy model endpoint reachable from the worker
  node. OSWorld is a screenshot-driven benchmark — text-only models
  will score 0 on every task.

## Quickstart

```bash
# 1) Repo setup (one time)
git clone https://github.com/NVIDIA-NeMo/Gym.git && cd Gym
uv venv && uv sync --extra dev

# 2) Provide the policy model endpoint
cat > env.yaml <<EOF
policy_base_url: https://your-vlm-endpoint/v1
policy_api_key: <your-key>
policy_model_name: <your-vlm-model>      # must be a VLM (vision-language)
EOF

# 3) Smoke test on the 5-row example dataset
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &

ng_collect_rollouts \
  +agent_name=osworld_simple_agent \
  +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
  +output_jsonl_fpath=results/osworld_example.jsonl \
  +num_repeats=1

# 4) Inspect rollout trajectories — each line of the output JSONL carries
#    reward + verifier_metadata.osworld_steps (per-step action/observation)
jq '{task: .verifier_metadata.task_id, reward}' results/osworld_example.jsonl
```

The first rollout takes longer than subsequent ones — see [First-Run
Cache Acquisition](#first-run-cache-acquisition) for what happens and
how to pre-stage.

### Native OSWorld PromptAgent smoke

The default quickstart uses `gym_pyautogui`, which keeps Gym in charge of
prompt construction. To smoke-test OSWorld's upstream
`mm_agents.agent.PromptAgent` path, add the native overlay when starting
`ng_run`:

```bash
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_agents/osworld_agent/configs/osworld_agent_native_prompt_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &

ng_collect_rollouts \
  +agent_name=osworld_simple_agent \
  +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
  +output_jsonl_fpath=results/osworld_native_prompt_agent_example.jsonl \
  +limit=1 \
  +num_repeats=1 \
  +num_samples_in_parallel=1
```

The helper script wraps the same flow and prints a compact summary:

```bash
bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

Set `RUNNER_NAME=prompt_agent_computer_13` or another explicit
`prompt_agent_*` runner to exercise a specific observation/action
combination.

## First-Run Cache Acquisition

The first time a rollout runs on a given worker node, the agent populates
two caches (on the worker's filesystem, persistent across runs):

| Cache | Default Location | Size | Source | First-Run Time |
| ----- | ---------------- | ---- | ------ | -------------- |
| OSWorld VM image (qemu wrapper SIF / Docker image) | `cache/osworld-images/happysixd_osworld-docker_latest.sif` (Apptainer) or Docker daemon's image store | ~80 MB | `docker.io/happysixd/osworld-docker:latest` | ~30 s |
| Ubuntu desktop qcow2 | `docker_vm_data/Ubuntu.qcow2` (Docker provider) / `cache/osworld-images/Ubuntu.qcow2` (Apptainer) | ~12 GB compressed, ~23 GB extracted | `huggingface.co/datasets/xlangai/ubuntu_osworld` | ~1–15 min depending on bandwidth |

Subsequent rollouts on the same node skip both downloads and start in
seconds.

### Pre-staging the cache — required when `concurrency > 1`

> ⚠️ **Required for any run with concurrency > 1** (osworld_agent.yaml
> default is `concurrency: 4`). OSWorld's docker provider does not
> serialize first-run downloads: with 4 parallel rollouts each calling
> `DesktopEnv(...)` simultaneously, all four write the same
> `docker_vm_data/Ubuntu.qcow2.zip` at once. The resulting torn file
> (~49 GB observed vs 12 GB normal) trips
> `zipfile.BadZipFile: File is not a zip file` on every rollout,
> producing 5/5 reward=0 with no clear root-cause hint. Single-threaded
> prestage before the first `ng_collect_rollouts` eliminates the race
> for good (the cache is then reused, no concurrent writers).

Lower concurrency (`concurrency: 1`) avoids the race without prestage,
but is much slower. Prestage is the default-correct path.

#### Docker provider recipe (Mode A)

```bash
# From the gym-osworld project root, before ng_run + ng_collect_rollouts:
mkdir -p docker_vm_data && cd docker_vm_data
curl -fL --retry 3 -O \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip && rm Ubuntu.qcow2.zip
docker pull happysixd/osworld-docker:latest
# Verify: should be ~23 GB
ls -lh Ubuntu.qcow2
```

That's it — the docker provider auto-discovers `docker_vm_data/Ubuntu.qcow2`.

#### Apptainer provider recipe (Mode B / SLURM-style clusters)

```bash
# Pre-pull the Apptainer SIF
apptainer pull \
  $CACHE_DIR/osworld-images/happysixd_osworld-docker_latest.sif \
  docker://happysixd/osworld-docker:latest

# Pre-download the Ubuntu desktop qcow2
mkdir -p $CACHE_DIR/osworld-images && cd $_
curl -fL --retry 3 -O \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip && rm Ubuntu.qcow2.zip
```

Point the agent at the shared cache via env vars:

```bash
export OSWORLD_APPTAINER_SIF_CACHE=$CACHE_DIR/osworld-images
export OSWORLD_APPTAINER_VMS_DIR=$CACHE_DIR/osworld-images
```

## Usage

```bash
# (one-time) Download datasets from GitLab
ng_download_dataset_from_gitlab \
    +dataset_name=osworld +version=0.0.1 \
    +artifact_fpath=validation.jsonl \
    +output_fpath=responses_api_agents/osworld_agent/data/validation.jsonl

# Start servers (policy model + agent)
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &

# Smoke test on the example JSONL (5 tasks)
ng_collect_rollouts +agent_name=osworld_simple_agent \
    +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
    +output_jsonl_fpath=results/osworld_example.jsonl \
    +num_repeats=1

# Inspect rollout trajectories (per-step actions + final reward) — read the
# output JSONL directly; reward is top-level, steps in verifier_metadata
jq '{task: .verifier_metadata.task_id, reward}' results/osworld_example.jsonl

# Reward profile (per-task pass rates across multiple repeats)
ng_collect_rollouts +agent_name=osworld_simple_agent \
    +input_jsonl_fpath=responses_api_agents/osworld_agent/data/validation.jsonl \
    +output_jsonl_fpath=results/osworld_validation.jsonl \
    +num_repeats=5 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

ng_reward_profile \
    +input_jsonl_fpath=responses_api_agents/osworld_agent/data/validation.jsonl \
    +rollouts_jsonl_fpath=results/osworld_validation.jsonl \
    +output_jsonl_fpath=results/osworld_profiled.jsonl \
    +pass_threshold=1.0
```

### Choosing the VM provider

See [Deployment Modes](#deployment-modes) for the comparison of
`docker` (single-machine, recommended for first run) and `remote_docker`
(controller and VM host on different machines). Provider and runner
settings are agent-server configuration, so set them when launching
`ng_run`:

```bash
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" \
    'osworld_simple_agent.responses_api_agents.osworld_agent.provider_name=docker'
```

Both providers expose the same set of ports inside the VM (5000 for
pyautogui HTTP, 8006 for noVNC, 9222 for Chromium DevTools, 5900 for
raw RFB, 8080 for VLC) and share the same Ubuntu qcow2 — only the
container runtime + how the controller reaches it differs.

OSWorld also ships providers for VMware, VirtualBox, AWS, Azure, GCP,
Aliyun, Volcengine, and Apptainer; see upstream
[`xlang-ai/OSWorld`](https://github.com/xlang-ai/OSWorld) for details.

## Configuration

`configs/osworld_agent.yaml` defines an `osworld_simple_agent` instance
that wires the agent to a model server and the datasets. Key fields
(map 1:1 to `OSWorldAgentConfig` in `app.py`):

| Field                    | Default | Meaning                                                                          |
| ------------------------ | ------- | -------------------------------------------------------------------------------- |
| `provider_name`          | docker  | OSWorld VM provider (`docker` / `remote_docker` / `apptainer` / `vmware` / …) — see [Deployment Modes](#deployment-modes) |
| `container_image`        | docker://happysixd/osworld-docker:latest | VM container image (consumed by the docker / apptainer provider) |
| `headless`               | true    | Run VM without a window manager forwarding                                       |
| `screen_width`/`_height` | 1920x1080 | VM resolution                                                                  |
| `require_a11y_tree`      | false   | Pass the AT-SPI accessibility tree alongside the screenshot                      |
| `client_password`        | password| Guest sudo password (substituted into the system prompt)                         |
| `max_steps`              | 15      | Max observation→action iterations per rollout                                    |
| `max_trajectory_length`  | 3       | How many previous screenshots to keep in the prompt                              |
| `sleep_after_execution`  | 0.5     | Seconds to wait after each `pyautogui` block (matches upstream)                  |
| `cache_dir`              | cache   | Per-rollout cache (snapshots, downloaded blobs)                                  |
| `concurrency`            | 4       | `asyncio.Semaphore` bound on concurrent `/run` calls                             |
| `max_tokens`             | 1500    | Model max-tokens override (request value wins if present)                        |
| `temperature` / `top_p`  | 1.0 / 0.9 | Same — request value wins                                                      |
| `runner_name`            | gym_pyautogui | Runner contract. `gym_pyautogui` preserves the Gym-built prompt path; `prompt_agent` uses OSWorld's native `PromptAgent` defaults; explicit `prompt_agent_*` names select concrete observation/action combinations |
| `action_space`           | null    | Optional override for compatible runners (`pyautogui` / `computer_13`)           |
| `observation_type`       | null    | Optional override for compatible runners (`screenshot` / `a11y_tree` / `screenshot_a11y_tree` / `som`) |
| `env_class_path`         | null    | Optional Python import path for a custom OSWorld environment class                |
| `agent_class_path`       | null    | Optional Python import path for a custom native OSWorld agent class               |
| `agent_kwargs`           | `{}`    | Extra kwargs merged into the native agent constructor                             |
| `mem_limit_mb`           | 16384   | VM cgroup memory cap in MB (~16 GB; passed to provider)                          |
| `step_timeout`           | 60      | Per-action subprocess timeout (advisory; provider-dependent)                     |
| `task_timeout`           | 1800    | Whole-rollout wall-clock cap in seconds; trips `mask_sample=True` if exceeded    |

The response includes a `mask_sample` boolean (NeMo-RL convention) — true if
the rollout timed out, exhausted `max_steps` without the model emitting
DONE/FAIL, or the evaluator threw. RL trainers drop the gradient on those
samples so unreliable zero rewards don't pollute the policy update.

### Observation modes

OSWorld evaluates agents under four observation modes (cf. the original
paper, Table 5). For the native `PromptAgent` path, choose a runner name:

| Mode | Native runner examples |
| ---- | ---------------------- |
| Screenshot only | `prompt_agent_screenshot_pyautogui`, `prompt_agent_computer_13` |
| Accessibility tree only | `prompt_agent_a11y_tree_pyautogui`, `prompt_agent_a11y_tree_computer_13` |
| Screenshot + accessibility tree | `prompt_agent`, `prompt_agent_screenshot_a11y_tree_pyautogui`, `prompt_agent_screenshot_a11y_tree_computer_13` |
| Set-of-Mark (SoM) | `prompt_agent_som_pyautogui` |

Runners that need accessibility data automatically construct
`DesktopEnv(require_a11y_tree=True)`. The default `gym_pyautogui` path
also supports a lightweight a11y-only diagnostic mode via
`require_a11y_tree=true` plus `OSWORLD_OMIT_SCREENSHOT_IN_OBS=1`; that
mode blanks the screenshot from the Gym-built prompt but does not use
OSWorld's native `PromptAgent` prompt templates.

### Override priority

Per-rollout values can be set three ways, in increasing priority:
1. YAML defaults (above).
2. `responses_create_params` on the JSONL row (per-task `max_tokens` /
   `temperature` / `top_p`).
3. CLI overrides (`+responses_create_params={…}` on
   `ng_collect_rollouts`).

## Recording rollout videos

Set `OSWORLD_RECORD_VIDEO_DIR` before launching `ng_run` to capture an mp4
per rollout. The agent calls OSWorld's `controller.start_recording()` after
`env.reset()` and `controller.end_recording(path)` in the finally block,
which downloads the mp4 from the VM to the host filesystem.

```bash
export OSWORLD_RECORD_VIDEO_DIR=/tmp/osworld-videos
mkdir -p "$OSWORLD_RECORD_VIDEO_DIR"
ng_run "+config_paths=[...]" &
ng_collect_rollouts ...
ls $OSWORLD_RECORD_VIDEO_DIR/   # one mp4 per task, named {task_id}.mp4
```

- Works for **every provider** that exposes the controller recording API
  (verified on `docker`; same call path used in OSWorld's own
  `lib_run_single.py`, so `apptainer` / `vmware` / `virtualbox` also work).
- File size is typically 1-10 MB per task at OSWorld's 1920x1080 default
  (verified: 5 example tasks → 1.5-5.2 MB each, 16 MB total).
- Recording is best-effort — if `start_recording()` or `end_recording()`
  raises (e.g. ffmpeg-in-VM missing, network blip during mp4 download),
  the error is logged and the rollout still completes normally.
- Mode B `remote_docker` users get the same env var honored, plus an
  additional VNC-stream recording path on the bare-metal host (see
  the existing description in the Mode B architecture diagram).

## Troubleshooting

### `ng_run` reports `bash: line 1: uv: command not found`

`ng_run` launches each FastAPI server in a `bash -c "uv run ..."` subshell.
Non-interactive bash doesn't source `~/.bashrc`, so the uv installer's
PATH edit (`$HOME/.local/bin`) isn't visible. Quick fix:

```bash
sudo ln -sf "$(which uv)"  /usr/local/bin/uv
sudo ln -sf "$(which uvx)" /usr/local/bin/uvx
```

`/usr/local/bin` is on the default non-interactive `PATH` on every standard
Linux distro, so once symlinked uv is reachable from any subshell. The
host-prep helper at [`scripts/bringup_local_host.sh`](scripts/bringup_local_host.sh)
does this automatically.

### Rollouts time out at the first `/screenshot` poll

Most common cause: the VM image is still downloading on first run. Tail
the worker log and look for `docker pull` / `qemu-img` progress.
If the network is slow, set a generous `task_timeout` on the first
benchmark run (e.g. `task_timeout=3600`) or [pre-stage the
cache](#pre-staging-the-cache-optional-recommended-for-shared-clusters).

### `Selection 'E266' has no effect because preview is not enabled`

Harmless `ruff` warning, not an error.

### "Update environment list in README" pre-commit hook reports Failed

Expected on the first commit after adding/modifying this benchmark.
The hook auto-syncs the resources-server table in the root `README.md`
and reports `Failed` when it has to write. Stage the README change and
commit a second time; the hook will report `Passed`.

### aarch64 clusters

`happysixd/osworld-docker` is x86_64-only. On aarch64 hosts you can:
- Use `apptainer pull --arch amd64` and run qemu in emulation mode (very
  slow — software CPU only, no KVM).
- Build a custom aarch64 SIF that wraps `qemu-system-aarch64` + the
  upstream Ubuntu-arm qcow2
  (`https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu-arm.zip`).
- Submit to an x86_64 partition if the cluster is heterogeneous.

## Reward Profiling

We validated the reward pipeline by **strictly reproducing one cell of the
official [OSWorld verified leaderboard](https://os-world.github.io/)** —
matching the model, observation mode, `max_steps`, and task set so the
number is directly comparable:

| Model | Subset | Obs / max_steps | Our Pass@1 | Official | Δ |
| ----- | ------ | --------------- | ---------- | -------- | - |
| `o3` (via NVIDIA Inference Hub) | `test_all` (369) | Screenshot only / 15 | **9.21%** (34/369) | 9.10% (32.85/361) | +0.11 pp |

The +0.11 pp gap is within run-to-run variance. (Task count differs
slightly: we run the full 369 tasks, while the board's 361 reflects a few
retired tasks — a known < 2 pp effect.) Per-domain, `o3` is strong on
`os` (13/24 = 54.2%) and shut out on `libreoffice_impress` (0/47).

A 5-task smoke on `data/example.jsonl` with `claude-opus-4-7` (Screenshot
only) returns 4/5 reward=1.0 — the saved outputs are in
`data/example_rollouts.jsonl` (chrome / gimp / calc / writer pass,
impress fails), demonstrating the reward signal both awards and withholds
credit.

_Further model × obs-mode × max_steps cells are in progress (a
`claude-sonnet-4-6` / max=100 run targeting the board's 72.1% is next)._

## Dataset Information

OSWorld tasks live at three difficulty / coverage levels:

| Subset                | Tasks | Use                                                              |
| --------------------- | ----- | ---------------------------------------------------------------- |
| `test_small.json`     | 39    | Quick smoke subset (multi-app tasks dominate; ≤4 per application)|
| `test_all.json`       | ~369  | Full benchmark across all apps                                   |
| `test_infeasible.json`| —     | Infeasible-task subset (model is expected to return `FAIL`)      |
| `test_nogdrive.json`  | —     | All tasks except those that require Google Drive credentials     |

Source: <https://github.com/xlang-ai/OSWorld/tree/main/evaluation_examples>
(license: Apache-2.0).

The manifests above ship from upstream as `{domain: [task_id, ...]}`
dicts. Convert any of them into the gym-consumable JSONL the agent
expects with
[`scripts/paper_to_gym_jsonl.py`](scripts/paper_to_gym_jsonl.py):

```bash
python responses_api_agents/osworld_agent/scripts/paper_to_gym_jsonl.py \
  --osworld-root /path/to/OSWorld \
  --manifest test_all \
  --output responses_api_agents/osworld_agent/data/test_all.jsonl
```

Committed under `data/`:
- `example.jsonl` — 5 representative tasks for smoke testing
- `example_rollouts.jsonl` — saved rollouts for those 5 tasks
  (`claude-opus-4-7`, Screenshot only), 4/5 reward=1.0
- `test_small.jsonl` — the 39-task smoke subset, ready to run

`test_all.jsonl` (369 tasks) is **not** committed — regenerate it with the
command above. Train / validation JSONLs live in the GitLab dataset
registry (see `gitlab_identifier` blocks in the YAML).

JSONL row shape:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "<OSWorld task instruction>"}]
  },
  "verifier_metadata": {
    "task_id": "<uuid>",
    "domain": "chrome|gimp|libreoffice_calc|libreoffice_impress|libreoffice_writer|multi_apps|os|thunderbird|vlc|vs_code",
    "osworld_task": { /* full OSWorld task JSON: snapshot, config, evaluator, etc. */ }
  }
}
```

`osworld_task` is the literal task spec OSWorld's `DesktopEnv.reset(task_config=…)`
expects.

## Licensing

- Wrapper code: Apache-2.0 (this repository).
- OSWorld upstream (`xlang-ai/OSWorld`): Apache-2.0. The system prompt
  in `prompts.py` and the action-parsing helpers in `action_parser.py`
  are adapted from OSWorld and retain that license — see file headers.
- Task data (`evaluation_examples/`): Apache-2.0 per upstream.
- Ubuntu desktop qcow2 (HF `xlangai/ubuntu_osworld`): see HF dataset
  card for license.
- `happysixd/osworld-docker` image: see upstream OSWorld README.
