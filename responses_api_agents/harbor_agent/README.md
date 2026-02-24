# Harbor Agent for NeMo Gym

This agent integrates [Harbor](https://github.com/laude-institute/harbor) into NeMo Gym.
It runs Harbor agents (e.g., `terminus-2`) in Harbor-managed environments and returns NeMo Gym-compatible outputs.

## Quick Start

### 1) Prerequisites

- Install Apptainer/Singularity (required when running Harbor tasks on HPC clusters with the Singularity environment).

```bash
apt-get update && apt-get install -y git wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
apt-get install -y ./apptainer_1.4.2_amd64.deb
apptainer --version
```

- Prepare Apptainer/Singularity images. In each Harbor task's `task.toml`
  (`[environment]` section), set `docker_image` using one of these modes:
  - Pre-built `.sif` mode: `docker_image` points to a local `.sif` file path.
  - Docker reference mode: `docker_image = "repo/image:tag"`, and the
    environment converts that image to `.sif` in the cache directory.
    For examples of downloading and converting to `.sif`, see:
    https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/dataset/swe-bench/dump_images.py.

- (Optional) Set private registry credentials for Singularity pulls.
  This is only needed when `docker_image` is a registry reference and the
  environment must pull/convert it to `.sif` (for example:
  `docker_image: private-registry/my-image:tag`).

```bash
export APPTAINER_DOCKER_USERNAME=<registry-username>
export APPTAINER_DOCKER_PASSWORD=<registry-password-or-token>
```

### 2) Configure the vLLM model server

If using the harbor agent for RL training, the companion vLLM model server config must enable token ID information and disable thinking history truncation.
Use `configs/vllm_model_for_training.yaml`:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      chat_template_kwargs:
        enable_thinking: true
        truncate_history_thinking: false
      return_token_id_information: true
      uses_reasoning_parser: true
```

### 3) Configure Harbor agent

Modify `configs/harbor_agent.yaml`. The default config uses the custom `Terminus2NemoGym`
agent and `SingularityEnvironment` with the kwargs needed for RL training.

### 5) Start NeMo Gym servers

```bash
config_paths="responses_api_agents/harbor_agent/configs/harbor_agent.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### 6) Test Harbor agent

```bash
python responses_api_agents/harbor_agent/client.py
```

### 7) Collect rollouts

```bash
ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=responses_api_agents/harbor_agent/data/example_input.jsonl \
  +output_jsonl_fpath=responses_api_agents/harbor_agent/data/example_output.jsonl
```

### 8) View trajectories

```bash
ng_viewer +jsonl_fpath=responses_api_agents/harbor_agent/data/example_output.jsonl
```

## Required patches to Gym

### Pass `chat_template_kwargs` to the tokenize endpoint

**`Gym/responses_api_models/vllm_model/app.py`** — the `/tokenize` endpoint must
receive `chat_template_kwargs` (e.g., `truncate_history_thinking: false`) to match
the tokenization used during chat completion. Without this, the tokenize call uses
the template's default `truncate_history_thinking=True`, which strips reasoning from
historical messages and breaks token contiguity in multi-turn training.

Change the tokenize body construction from:

```python
for key in ("model", "messages", "tools"):
    if key in body_dict:
        tokenize_body_dict[key] = body_dict[key]
```

To:

```python
for key in ("model", "messages", "tools", "chat_template_kwargs"):
    if key in body_dict:
        tokenize_body_dict[key] = body_dict[key]
```

## Custom agents

Harbor ships several agents, but for NeMo Gym RL we had to adapt the integration
layer so agent outputs, trajectories, and token metadata are compatible with
NeMo Gym/NeMo RL expectations (especially multi-turn token accounting and rollout
details). In this repo, use the Terminus integration as the reference pattern for
those adaptations.

If you want to plug in a different Harbor agent, follow the Terminus wrapper flow
as an example: keep Harbor's core agent behavior, then add a thin compatibility
layer that normalizes message/output schema and preserves the metadata required by
training.

## Custom environments

The default Harbor environments are not sufficient for HPC training, so this repo 
includes a custom Singularity environment implementation.
It is designed around task-local setup, staged task files, and predictable runtime
paths used by Harbor jobs.
For Singularity installation, image preparation (`docker_image` as `.sif` path vs
registry reference), and optional registry auth, see `1) Prerequisites` above.

Any additional task files needed by the environment should be placed under
`environment/files/`. This directory is bind-mounted into the container staging
area and copied into the runtime filesystem during bootstrap, so scripts/assets are
available before agent execution.

For task setup, this environment supports an optional `environment/files/setup.sh`
script. When present, it is run during environment initialization and is the right
place for per-task dependency/setup steps.

Common `harbor_environment_kwargs` for this environment:
- `singularity_image_cache_dir`: cache directory for converted `.sif` images.
- `singularity_force_pull`: force re-pull/re-convert the image instead of using cache.
- `singularity_no_mount`: override/suppress selected Singularity default mounts.
- `workdir`: override container working directory.

Singularity does not enforce cgroups-based memory limits on most HPC clusters (no
systemd init). The environment runs a userspace memory watchdog that monitors PSS
and kills the container at 95% of the task's configured `memory_mb`.

## NeMo RL Training

### Recommended settings

These are the recommended settings for the NeMo RL training config:

```yaml
env:
  nemo_gym:
    use_absolute_ip: true               # Required for multi-node Ray clusters
    harbor_agent:
      responses_api_agents:
        harbor_agent:
          # Match concurrency to total rollouts per step for maximum throughput.
          concurrency: ${mul:${grpo.num_prompts_per_step}, ${grpo.num_generations_per_prompt}}

          # Limit on how long a single rollout can run (including all turns).
          # You can also set a per-task timeout in task.toml via [agent].timeout_sec.
          # If harbor_agent_timeout is set here, it is passed as Harbor's
          # agent.override_timeout_sec and overrides task.toml.
          harbor_agent_timeout: 900

          harbor_agent_kwargs:
            max_turns: 20                # Max turns per rollout. Configure this for your dataset.
            interleaved_thinking: true
            enable_summarize: false
            collect_rollout_details: true
            trajectory_config:
              raw_content: true
            model_info:
              max_input_tokens: ${policy.max_total_sequence_length}
              max_output_tokens: ${policy.max_total_sequence_length}
```

Additional policy settings required for multi-node training:

```yaml
policy:
  generation:
    vllm_kwargs:
      enable_chunked_prefill: false     # Disable chunked prefill for stability
```

### Finding failed rollouts

Harbor writes each rollout to a subdirectory under `harbor_jobs_dir`. A practical
way to debug is to inspect trajectories by run timestamp: start from the relevant
timestamped job directory, then drill into per-rollout subdirectories and compare
`trajectory.json`, verifier outputs, and exception files across nearby runs.

### Known failure cases during RL training

When the Harbor agent fails during rollout collection, the sample returns `reward=0.0`
and an empty `output` list (no output items with `generation_token_ids`). 

Common symptom: `IndexError: list index out of range` at `rollouts.py:1185`. This
usually means at least one rollout returned an empty `input_message_log`, and a
single failed rollout then crashes the entire training step. To identify which
rollout failed, scan the harbor job directories.

A recommended mitigation is to tolerate empty/failed rollouts by marking them as
degenerate, keeping training alive, and excluding those samples from gradient
contribution while tracking their rate in metrics.

**Failure scenarios that produce empty output:**

- **Context length exceeded on the first turn**: the model cannot generate any tokens,
  so there are no `generation_token_ids` to collect. `Terminus2NemoGym.run()` catches
  `ContextLengthExceededError` and returns gracefully, but if no turns completed, the
  output is empty.
- **Singularity environment setup failure**: `upload_file` or `upload_dir` fails during
  container initialization (e.g., tmux_session uploads `get-asciinema-timestamp.sh` to
  `/tmp`). The trial raises `RuntimeError` before the agent runs any turns.
- **Unhandled exception in `run_harbor_job`**: `app.py` catches all exceptions, sets
  `output_items=[]` and `reward=0.0`.

**Scenarios that preserve partial trajectories (do NOT produce empty output):**

- **Agent timeout**: Harbor handles `AgentTimeoutError` internally in `trial.py`.
  Terminus-2's `finally` block writes `trajectory.json` with all completed steps before
  the coroutine is cancelled, and the trial proceeds to verification. The partial
  trajectory flows through `app.py` normally — completed turns have `generation_token_ids`
  and are usable for training.
- **Context length exceeded on a later turn** (listed above): same behavior — completed
  turns are preserved.

### On-policy corrections for multi-turn training

In multi-turn RL training, turn `N+1` is built from the full conversation history
up to turn `N`. If that history is reconstructed from text, token alignment can
silently drift and break on-policy training assumptions.

Nemo-RL applies on-policy token corrections to preserve prompt/continuation
contiguity across turns. Details:
https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html

For Harbor related questions, check out the official Harbor docs: https://harborframework.com/docs.
