# osworld_agent / scripts

Host-prep helpers plus a native `PromptAgent` smoke runner.

| Script | When to use it | What it does |
|---|---|---|
| [`bringup_local_host.sh`](bringup_local_host.sh) | **Mode A** — `ng_run` and the docker VM host are the **same** machine | apt-installs docker / git / curl / unzip / ffmpeg / xvfb / tigervnc-viewer · enables the docker daemon · adds `$USER` to the `docker` group · installs `uv` · symlinks `uv` into `/usr/local/bin` (so `ng_run`'s non-interactive `bash -c "uv run …"` subshells can find it) · pre-flight verify (docker daemon up, `/dev/kvm` present, uv visible to `bash -c`) |
| [`bringup_remote_host.sh`](bringup_remote_host.sh) | **Mode B** — controller runs locally, but the docker VM host is a **separate** machine reached via SSH | passwordless-SSH check · same apt installs over SSH · stages `Ubuntu.qcow2` via rsync · `docker pull happysixd/osworld-docker` on the remote · pre-flight verify (kvm + image + ports free on the remote) |
| [`setup_colossus_runtime.sh`](setup_colossus_runtime.sh) | Colossus git checkout setup for Mode A | creates `~/osworld-run` runtime dirs · symlinks repo `env.yaml` and `docker_vm_data/Ubuntu.qcow2` to private assets · writes `.colossus-runtime.env` · verifies git / uv / docker / KVM / env / qcow2 without committing secrets or large files |
| [`run_native_prompt_agent_smoke.sh`](run_native_prompt_agent_smoke.sh) | Quick functional smoke for OSWorld's native `mm_agents.agent.PromptAgent` path | starts `ng_run` with `osworld_agent_native_prompt_agent.yaml`, collects one rollout from `data/example.jsonl`, and prints a compact reward/step/error summary |

The host-prep scripts stop short of cloning the repo, running `uv sync`, or
prestaging `Ubuntu.qcow2`. For Colossus runs, clone this repo first, keep
`env.yaml` under `~/osworld-run/private/`, keep `Ubuntu.qcow2` under
`~/osworld-run/osworld-vm-data/`, then run `setup_colossus_runtime.sh` from the
git checkout. The setup script runs `uv sync --extra dev` by default and adds
the checkout's `.venv/bin` to `.colossus-runtime.env`, so the normal Gym CLI
commands are available without wrapping them in `uv run`.

## Colossus git-checkout flow

```bash
git clone -b feature/osworld2 git@github.com:JeffPengCoder/Gym.git \
  ~/osworld-run/gym-osworld-git
cd ~/osworld-run/gym-osworld-git

# env.yaml is copied from desktop out-of-band and is not committed.
# Ubuntu.qcow2 is also kept outside git.
bash responses_api_agents/osworld_agent/scripts/setup_colossus_runtime.sh

DRY_RUN=1 bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

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
