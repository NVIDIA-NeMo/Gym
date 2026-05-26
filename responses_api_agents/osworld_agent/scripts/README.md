# osworld_agent / scripts

Two one-shot host-prep helpers. Both are idempotent (re-runs are safe).
They cover the parts of OSWorld setup that aren't fixable in the repo
itself: OS-level package install, uv toolchain, and Mode-B-specific
SSH + qcow2 staging.

| Script | When to use it | What it does |
|---|---|---|
| [`bringup_local_host.sh`](bringup_local_host.sh) | **Mode A** — `ng_run` and the docker VM host are the **same** machine | apt-installs docker / git / curl / unzip / ffmpeg / xvfb / tigervnc-viewer · enables the docker daemon · adds `$USER` to the `docker` group · installs `uv` · symlinks `uv` into `/usr/local/bin` (so `ng_run`'s non-interactive `bash -c "uv run …"` subshells can find it) · pre-flight verify (docker daemon up, `/dev/kvm` present, uv visible to `bash -c`) |
| [`bringup_remote_host.sh`](bringup_remote_host.sh) | **Mode B** — controller runs locally, but the docker VM host is a **separate** machine reached via SSH | passwordless-SSH check · same apt installs over SSH · stages `Ubuntu.qcow2` via rsync · `docker pull happysixd/osworld-docker` on the remote · pre-flight verify (kvm + image + ports free on the remote) |

Both scripts stop short of cloning the repo, running `uv sync`, or
prestaging `Ubuntu.qcow2` to `docker_vm_data/` — those steps are
documented in the agent's main [`README.md`](../README.md) (sections
"Quickstart" and "First-Run Cache Acquisition") and are intentionally
left to the user so they can pick the right repo location and cache
strategy for their environment.

## Mode A flow (using `bringup_local_host.sh`)

```bash
# 1. On the box that will host ng_run + docker:
bash bringup_local_host.sh

# 2. Get the agent code:
git clone https://github.com/<your-fork-or-NeMo-Gym>.git
cd <Gym-checkout>
uv venv && uv sync --extra dev

# 3. Fill in env.yaml at project root (see main README Quickstart for fields)

# 4. MANDATORY when concurrency > 1: prestage Ubuntu.qcow2 (12 GB) so
#    parallel rollouts don't race to download it. See main README
#    "First-Run Cache Acquisition" for the curl + unzip recipe.

# 5. Run:
ng_run "+config_paths=[\
responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml]" &
ng_collect_rollouts \
  +agent_name=osworld_simple_agent \
  +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \
  +output_jsonl_fpath=results/osworld_example.jsonl \
  +num_repeats=1
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
