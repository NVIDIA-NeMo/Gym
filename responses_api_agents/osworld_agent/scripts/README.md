# OSWorld helper scripts

These scripts cover host setup, dataset conversion, smoke tests, concurrent
rollouts, and Nemotron 3 Nano Omni serving.

| Script | Purpose |
| --- | --- |
| [`bringup_local_host.sh`](bringup_local_host.sh) | Install Docker, `uv`, and optional video tools on an x86_64 rollout host |
| [`paper_to_gym_jsonl.py`](paper_to_gym_jsonl.py) | Convert upstream OSWorld manifests to Gym JSONL |
| [`run_native_prompt_agent_smoke.sh`](run_native_prompt_agent_smoke.sh) | Run a small OSWorld `PromptAgent` smoke test |
| [`run_multienv_osworld_agent.sh`](run_multienv_osworld_agent.sh) | Run multiple OSWorld environments through Gym |
| [`run_m3_multienv.sh`](run_m3_multienv.sh) | Select the MiniMax M3 runner and delegate to the multi-environment script |
| [`launch_omni_mini_vllm.sh`](launch_omni_mini_vllm.sh) | Serve Nemotron 3 Nano Omni with vLLM 0.20.0 |
| [`probe_omni_mini_vllm.py`](probe_omni_mini_vllm.py) | Check model discovery and a real one-image request |
| [`run_omni_mini_vllm.sh`](run_omni_mini_vllm.sh) | Probe an Omni endpoint, start Gym, and collect rollouts |

## Local Docker setup

Run the setup helper on the Linux x86_64 host that will run Gym and the
OSWorld Docker VMs:

```bash
bash responses_api_agents/osworld_agent/scripts/bringup_local_host.sh
```

The script does not clone Gym, create `env.yaml`, or download the VM image.
Follow the main [OSWorld agent README](../README.md) for those steps.

## Native PromptAgent smoke

```bash
bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

Common overrides:

```bash
RUNNER_NAME=prompt_agent_computer_13 LIMIT=1 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

START_NG_RUN=0 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

DRY_RUN=1 \
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
```

## Multi-environment rollouts

`NUM_ENVS` is the concurrency. `LIMIT` is the number of input rows, not the
number of environments.

```bash
RUNNER_NAME=prompt_agent \
POLICY_MODEL_NAME=<your-model> \
NUM_ENVS=4 \
LIMIT=8 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

For resumable runs:

```bash
INPUT_JSONL=responses_api_agents/osworld_agent/data/test_all.jsonl \
RUN_DIR=results/<failed-run> \
LIMIT=null NUM_ENVS=4 RESUME_FROM_CACHE=1 \
bash responses_api_agents/osworld_agent/scripts/run_multienv_osworld_agent.sh
```

Keep the input order, output path, repeat count, and video-sampling seed stable
when resuming. A normal invocation atomically creates `RUN_DIR` and refuses to
reuse any existing path. `RESUME_FROM_CACHE=1` is the only exception: the
launcher verifies that Gym's output and materialized-input JSONLs exist, every
prior attempt is terminal, and no prior attempt succeeded. Resume control
records are written to `${RUN_DIR}/resume-attempts/<attempt-id>` so earlier
records remain unchanged.

Every attempt writes `started_at.txt`, `launcher.pid`, `run.env`, and
`resolved-command.log`, then writes `finished_at.txt` and `exit_code.txt` on
normal completion, preflight failure, or signal termination. For a fresh run
these files live directly in `RUN_DIR`; a resume uses its attempt directory.

The script writes per-task logs, trajectories, screenshots, VM execution logs,
and result metadata under the current attempt's `task-artifacts` directory.
Disable them with `TASK_ARTIFACTS=0`, or move them with
`TASK_ARTIFACT_ROOT=/path/to/artifacts`.

For adapter-parity debugging, add `FULL_MODEL_IO=1`. This writes the complete
agent-facing request/response stream to `model-io-agent.jsonl`, the final VLLM
transport payload/response stream to `model-io-transport.jsonl`, and a
run-level VM command/response stream to `vm-exec.jsonl`. Full requests retain
embedded screenshots, so the files are intentionally opt-in and may be large.
The agent stream also captures Pointer's direct Anthropic Messages calls;
transport credential fields are redacted without removing model-body content.
Each schema-v2 event includes the run/task/domain/attempt/step identity; the
transport service receives it through headers rather than model-body fields.
The runner resolves `RUN_DIR` and all three model-I/O paths to absolute paths
before starting Gym, so agent and policy services cannot split the files
across their component-specific working directories.

Use `FULL_MODEL_IO=1` only for targeted diagnostics. It records full agent and
vLLM payloads, including embedded screenshots, in the current attempt directory.

## MiniMax M3

```bash
LIMIT=4 NUM_ENVS=1 \
  bash responses_api_agents/osworld_agent/scripts/run_m3_multienv.sh
```

The wrapper selects the M3 runner and its model overlay, then invokes the
multi-environment script.

## Nemotron 3 Nano Omni

On the model host:

```bash
bash responses_api_agents/osworld_agent/scripts/launch_omni_mini_vllm.sh
```

The launcher defaults to tensor parallel size 1. For a multi-GPU host:

```bash
TENSOR_PARALLEL_SIZE=8 \
  bash responses_api_agents/osworld_agent/scripts/launch_omni_mini_vllm.sh
```

On the rollout host, pass an endpoint reachable from the OSWorld worker:

```bash
OMNI_MINI_VLLM_BASE_URL=http://model-host:8000/v1 \
  bash responses_api_agents/osworld_agent/scripts/run_omni_mini_vllm.sh
```

The rollout wrapper probes the endpoint first and defaults to the committed
five-task example with one VM. Larger runs can override `INPUT_JSONL`,
`LIMIT`, `NUM_ENVS`, `RUN_DIR`, `RESUME_FROM_CACHE`, and `RECORD_VIDEO`.

Do not expose an unauthenticated vLLM port to an untrusted network. Use normal
network access controls or a secure tunnel when the model and rollout hosts
are separated.
