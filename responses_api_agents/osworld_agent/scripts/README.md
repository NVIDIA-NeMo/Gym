# osworld_agent / scripts

Host-prep helpers plus a native `PromptAgent` smoke runner.

| Script | When to use it | What it does |
|---|---|---|
| [`bringup_local_host.sh`](bringup_local_host.sh) | Colossus/local Docker — `ng_run` and the Docker VM host are the **same** machine | apt-installs docker / git / curl / unzip / ffmpeg / xvfb / tigervnc-viewer · enables the docker daemon · adds `$USER` to the `docker` group · installs `uv` · symlinks `uv` into `/usr/local/bin` (so `ng_run`'s non-interactive `bash -c "uv run …"` subshells can find it) · pre-flight verify (docker daemon up, `/dev/kvm` present, uv visible to `bash -c`) |
| [`run_native_prompt_agent_smoke.sh`](run_native_prompt_agent_smoke.sh) | Quick functional smoke for OSWorld's native `mm_agents.agent.PromptAgent` path | starts `ng_run` with `osworld_agent_native_prompt_agent.yaml`, collects one rollout from `data/example.jsonl`, and prints a compact reward/step/error summary |
| [`run_multienv_osworld_agent.sh`](run_multienv_osworld_agent.sh) | OSWorld-style multi-environment runner through Gym | starts `ng_run`, waits for the agent/model servers, then runs `ng_collect_rollouts` with `NUM_ENVS` mapped to `+num_samples_in_parallel`; optionally records per-task VM mp4s |
| [`run_m3_multienv.sh`](run_m3_multienv.sh) | MiniMax M3 through the official OSWorld M3Agent | selects the M3 Gym runner, InferenceHub model id, and leaderboard-oriented M3 config before delegating to `run_multienv_osworld_agent.sh` |
| [`launch_omni_mini_vllm.sh`](launch_omni_mini_vllm.sh) | B200/H200/H100-80GB model node | starts the official Omni Mini BF16 checkpoint with vLLM 0.20.0 and the Nemotron reasoning parser |
| [`probe_omni_mini_vllm.py`](probe_omni_mini_vllm.py) | Before a rollout | checks `/models` and a real one-image Chat Completions request |
| [`run_omni_mini_local_vllm.sh`](run_omni_mini_local_vllm.sh) | 100-step Omni Mini parity run | points Gym's external-vLLM adapter at the endpoint, probes it, then runs clean OSWorld with the internal-compatible prompt/history/sleep settings |
| [`inspect_prenyx_b200.sh`](inspect_prenyx_b200.sh) | Prenyx login node | reports B200 partitions, Slurm associations, and current jobs without changing cluster state |
| [`start_omni_mini_vllm_prenyx.sh`](start_omni_mini_vllm_prenyx.sh) | After a Prenyx allocation reaches RUNNING | attaches an Enroot vLLM step to the allocation, exposes one B200, and prints the compute-node endpoint |
| [`watch_omni_mini_vllm_prenyx.sh`](watch_omni_mini_vllm_prenyx.sh) | While a Prenyx job/assets are pending | waits for the allocation and downloads, starts vLLM, then runs model-list and one-image probes |

The host-prep scripts stop short of cloning the repo, running `uv sync`, or
prestaging `Ubuntu.qcow2`. Keep private configuration and VM images outside
Git, then expose them to the checkout through `env.yaml` and
`docker_vm_data/Ubuntu.qcow2` before starting the Gym servers.

## Omni Mini on an external B200 vLLM

Keep inference separate from Gym/Ray so an OSWorld restart does not unload the
model. On the allocated B200 node (vLLM must be exactly 0.20.0):

```bash
bash responses_api_agents/osworld_agent/scripts/launch_omni_mini_vllm.sh
```

On the Colossus Docker/KVM host, use an address reachable from that host:

```bash
OMNI_MINI_VLLM_BASE_URL=http://B200_NODE:8000/v1 \
  bash responses_api_agents/osworld_agent/scripts/run_omni_mini_local_vllm.sh
```

The reference-aligned runner defaults to 361 no-GDrive inputs, four VMs, 100
steps, five seconds after each action, internal prompt, full text history,
current plus up to two historical screenshots, raw reward,
and resume enabled. Set `LIMIT=1 NUM_ENVS=1 RECORD_VIDEO=1` for the first real
smoke. Do not expose port 8000 broadly; use cluster routing or an SSH tunnel
when the Prenyx compute network is not directly reachable from Colossus.

Prenyx represents B200 as a node feature rather than a GPU GRES, so request a
whole exclusive node with `--constraint=b200`. Once its allocation is running
and the model/image caches are ready:

```bash
bash responses_api_agents/osworld_agent/scripts/start_omni_mini_vllm_prenyx.sh JOB_ID
```

This uses the complete eight-GPU B200 node with tensor parallel size 8, matching
`../internal-osworld-adapter-nano-omni/computelab/launch_vllm_b300_tp8.sbatch`. The public BF16
checkpoint could fit on one B200, but TP1 would no longer be that comparison.

## Colossus/local Docker flow (using `bringup_local_host.sh`)

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

For long runs, enable rollout-cache resume so restarting the same command and
output path skips rows already present in `rollouts.jsonl`:

```bash
INPUT_JSONL=responses_api_agents/osworld_agent/data/test_all.jsonl \
LIMIT=null NUM_ENVS=4 RESUME_FROM_CACHE=1 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

Keep `RUN_DIR`, `OUTPUT_JSONL`, input ordering, `NUM_REPEATS`, and the video
sampling seed unchanged when resuming. The runner also writes per-task
`worker.log`, `runtime.log`, `traj.jsonl`, screenshots, VM execution traces,
and result metadata under `${RUN_DIR}/task-artifacts`. Set `TASK_ARTIFACTS=0`
to disable this, or `TASK_ARTIFACT_ROOT=/shared/path` to move it.

For adapter-parity debugging, add `FULL_MODEL_IO=1`. This writes the complete
agent-facing request/response stream to `model-io-agent.jsonl`, the final VLLM
transport payload/response stream to `model-io-transport.jsonl`, and a
run-level VM command/response stream to `vm-exec.jsonl`. Full requests retain
embedded screenshots, so the files are intentionally opt-in and may be large.
The runner resolves `RUN_DIR` and all three model-I/O paths to absolute paths
before starting Gym, so agent and policy services cannot split the files
across their component-specific working directories.

## Legacy remote-host helper

`bringup_remote_host.sh` remains in the branch for historical deployments,
but it is not part of the clean-upstream configuration: upstream OSWorld
`main` has no `remote_docker` provider. The supported Colossus path runs the
built-in `docker` provider on the same worker as Gym.
