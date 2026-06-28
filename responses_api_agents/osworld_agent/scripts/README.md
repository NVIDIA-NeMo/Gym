# osworld_agent / scripts

Host-prep helpers plus a native `PromptAgent` smoke runner.

| Script | When to use it | What it does |
|---|---|---|
| [`bringup_local_host.sh`](bringup_local_host.sh) | **Mode A** — `ng_run` and the docker VM host are the **same** machine | apt-installs docker / git / curl / unzip / ffmpeg / xvfb / tigervnc-viewer · enables the docker daemon · adds `$USER` to the `docker` group · installs `uv` · symlinks `uv` into `/usr/local/bin` (so `ng_run`'s non-interactive `bash -c "uv run …"` subshells can find it) · pre-flight verify (docker daemon up, `/dev/kvm` present, uv visible to `bash -c`) |
| [`bringup_remote_host.sh`](bringup_remote_host.sh) | **Mode B** — controller runs locally, but the docker VM host is a **separate** machine reached via SSH | passwordless-SSH check · same apt installs over SSH · stages `Ubuntu.qcow2` via rsync · `docker pull happysixd/osworld-docker` on the remote · pre-flight verify (kvm + image + ports free on the remote) |
| [`run_native_prompt_agent_smoke.sh`](run_native_prompt_agent_smoke.sh) | Quick functional smoke for OSWorld's native `mm_agents.agent.PromptAgent` path | starts `ng_run` with `osworld_agent_native_prompt_agent.yaml`, collects one rollout from `data/example.jsonl`, and prints a compact reward/step/error summary |
| [`run_multienv_osworld_agent.sh`](run_multienv_osworld_agent.sh) | OSWorld-style multi-environment runner through Gym | starts `ng_run`, waits for the agent/model servers, then runs `ng_collect_rollouts` with `NUM_ENVS` mapped to `+num_samples_in_parallel`; optionally records per-task VM mp4s |
| [`run_m3_multienv.sh`](run_m3_multienv.sh) | MiniMax M3 through the official OSWorld M3Agent | selects the M3 Gym runner, InferenceHub model id, and leaderboard-oriented M3 config before delegating to `run_multienv_osworld_agent.sh` |

The host-prep scripts stop short of cloning the repo, running `uv sync`, or
prestaging `Ubuntu.qcow2`. Keep private configuration and VM images outside
Git, then expose them to the checkout through `env.yaml` and
`docker_vm_data/Ubuntu.qcow2` before starting the Gym servers.

## Mode A flow (using `bringup_local_host.sh`)

```bash
# 1. On the box that will host ng_run + docker:
bash bringup_local_host.sh

# 2. Get the agent code:
git clone https://github.com/<your-fork-or-NeMo-Gym>.git
cd <Gym-checkout>
uv venv && uv sync --extra dev

# 3. Write env.yaml (REQUIRED — ng_run resolves OmegaConf refs from this).
#    The policy MUST be a VLM. Text-only models score 0 on every OSWorld task.
cat > env.yaml <<YAML
policy_base_url: https://your-vlm-endpoint/v1
policy_api_key: <your-key>
policy_model_name: <your-vlm-model>     # must be VLM
YAML
chmod 600 env.yaml

# 4. MANDATORY when concurrency > 1: prestage Ubuntu.qcow2 (12 GB) so
#    parallel rollouts don't race to download it.
mkdir -p docker_vm_data && cd docker_vm_data
curl -fL --retry 3 -O \
  https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip
unzip Ubuntu.qcow2.zip && rm Ubuntu.qcow2.zip
cd ..

# 5. (optional) record per-rollout mp4
# export OSWORLD_RECORD_VIDEO_DIR=/tmp/smoke/videos && mkdir -p $OSWORLD_RECORD_VIDEO_DIR

# 6. Run:
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &
ng_collect_rollouts \
  +agent_name=osworld_simple_agent \
  +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
  +output_jsonl_fpath=results/osworld_example.jsonl \
  +num_repeats=1
```

## Native PromptAgent smoke

After the normal repo setup, `env.yaml`, and cache prestage are done:

```bash
bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

Useful variants:

```bash
# Use another explicit native runner.
RUNNER_NAME=prompt_agent_computer_13 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

# Reuse an already-running ng_run. It must have been started with the same
# native PromptAgent config and runner_name.
START_NG_RUN=0 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

# Run the full 5-row example dataset instead of the default 1-row smoke.
LIMIT=null \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

# Print the ng_run and ng_collect_rollouts commands without starting servers.
DRY_RUN=1 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

## Multi-Env OSWorld Agent

Use `run_multienv_osworld_agent.sh` when you want the OSWorld-style
`run_multienv_*.py` behavior through Gym. `NUM_ENVS` is the number of parallel
DesktopEnv instances. `LIMIT` is the total number of rows to collect from the
input JSONL, not the concurrency.

```bash
RUNNER_NAME=prompt_agent \
POLICY_MODEL_NAME=nvidia/minimaxai/minimax-m3 \
NUM_ENVS=4 \
LIMIT=4 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

For the PointerAgent leaderboard-anchor path, use the same public
`pointer_agent` runner. The Gym adapter routes the Anthropic-compatible model
endpoint and handles missing optional Parallel web tools at runtime.

```bash
RUNNER_NAME=pointer_agent \
POLICY_MODEL_NAME=azure/anthropic/claude-opus-4-7 \
NUM_ENVS=4 \
LIMIT=4 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

For a two-wave cleanup probe, use an input with at least 8 rows and set:

```bash
NUM_ENVS=4 LIMIT=8 MAX_STEPS=3 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

That runs at most four envs at once, then launches the next wave after earlier
envs finish and tear down. If the default `data/example.jsonl` only contains
five rows, `LIMIT=8` still collects only those five rows. To force two waves
from a four-row slice, repeat the limited slice:

```bash
NUM_ENVS=4 LIMIT=4 NUM_REPEATS=2 MAX_STEPS=3 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

## Mode B flow (using `bringup_remote_host.sh`)

```bash
# 1. From your controller machine, prep the remote docker host:
bash bringup_remote_host.sh user@remote-host /local/path/to/Ubuntu.qcow2

# 2. On the controller, point the agent at the remote:
export OSWORLD_REMOTE_HOST=user@remote-host
# export OSWORLD_REMOTE_SSH_KEY=~/.ssh/id_ed25519   # optional

# 3. In configs/osworld_agent.yaml, set:
#    provider_name: remote_docker

# 4. Run ng_run + ng_collect_rollouts as usual.
```

See the main [`README.md`](../README.md) "Deployment Modes" section for
the architectural difference between Mode A and Mode B, including which
ports flow where and why SSH was picked over docker-over-TCP.
