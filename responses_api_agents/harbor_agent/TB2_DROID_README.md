# TB2 Evaluation with Droid via NeMo Gym

Run [Terminal Bench 2](https://www.tbench.ai/leaderboard/terminal-bench/2.0) (89 tasks)
using [Factory Droid](https://www.factory.ai/droid) as the agent, orchestrated through
NeMo Gym's `ng_run` + `ng_collect_rollouts`.

## How it works

Droid is a `BaseInstalledAgent` — a binary installed inside containers that
calls the model endpoint directly. The wrapper (`DroidNemoGym`) dynamically generates
Droid's `~/.factory/settings.json` inside each container from NeMo Gym config.

```
ng_collect_rollouts
  → agent server (app.py /run)
    → Harbor Job (Docker or Singularity container)
      → DroidNemoGym.setup(): installs nvm + Node + droid
      → DroidNemoGym.create_run_agent_commands():
          command-0: writes settings.json (model endpoint, API key)
          command-1: droid exec --skip-permissions-unsafe --output-format stream-json
      → Harbor verifier: runs task's bundled pytest tests
      → result.json with reward (1.0 or 0.0)
    ← app.py reads result.json, returns reward
  ← output JSONL with reward per task
```

## Droid vs Terminus-2

| | Terminus-2 | Droid |
|---|---|---|
| Agent type | `BaseLLMAgent` (LLM injected) | `BaseInstalledAgent` (binary) |
| Use case | RL training + evaluation | Evaluation only |
| Trajectory | ATIF with token IDs/logprobs | stream-json trace (tool calls, no token IDs) |
| Environment | Singularity or Docker | Singularity or Docker |
| Model calls | Via NeMo Gym model server | Direct to endpoint |

## NeMo Gym server usage

Only the **agent server** is used. The model server and resource server are not:

- **Model server**: Must be included in `config_paths` (required by `app.py` at
  startup), but Droid calls the model endpoint directly via `api_base` in
  `harbor_agent_kwargs`. The model server starts but receives no requests.

- **Resource server**: Not needed. Verification is handled by Harbor's built-in
  verifier — each TB2 task bundles its own pytest tests inside the container.

## Model endpoint requirements

The endpoint must support OpenAI-compatible chat completions with **structured tool
calls** (not text-based). For self-hosted models:

```bash
vllm serve <model> \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Remote endpoints (e.g., NVIDIA Inference API) work directly.

## Running TB2 evaluation (Docker)

Use `configs/harbor_agent_tb2_droid.yaml` for Docker environments.

1. Configure the model endpoint in `env.yaml` at the project root:

```yaml
policy_base_url: https://inference-api.nvidia.com/v1  # or http://localhost:8234/v1 for local vLLM
policy_api_key: <API_KEY>
policy_model_name: <MODEL_NAME>
```

2. Start NeMo Gym servers:

```bash
ng_run \
  '+config_paths=[responses_api_agents/harbor_agent/configs/harbor_agent_tb2_droid.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+default_host=0.0.0.0'
```

Alternatively, pass the values as CLI overrides instead of `env.yaml`:
```bash
ng_run ... '+policy_base_url=<URL>' '+policy_api_key=<KEY>' '+policy_model_name=<MODEL>'
```

2. In a separate terminal, collect rollouts:

```bash
ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=responses_api_agents/harbor_agent/data/tb2_input.jsonl \
  +output_jsonl_fpath=<output.jsonl> \
  +num_samples_in_parallel=2
```

## Running TB2 evaluation (Singularity/Apptainer)

Use `configs/harbor_agent_tb2_droid_singularity.yaml` for HPC clusters without
Docker. This uses the custom `SingularityEnvironment` from the Harbor integration.

### Prerequisites

1. **Install Apptainer**:

```bash
apt-get update && apt-get install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.2/apptainer_1.4.2_amd64.deb
apt-get install -y ./apptainer_1.4.2_amd64.deb
apptainer --version
```

2. **Prepare TB2 task images**: Clone the TB2 task repo and write `setup.sh` files
   that install the in-container server dependencies (uvicorn + fastapi):

```bash
# Clone TB2 tasks
git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /raid/<user>/tb2-tasks

# Write setup.sh to all 89 tasks (installs uvicorn/fastapi inside the container)
python responses_api_agents/harbor_agent/custom_envs/singularity/scripts/write_min_setup_sh.py \
  --task-root /raid/<user>/tb2-tasks --force
```

The Singularity environment automatically converts Docker images to `.sif` format
on first run and caches them in `singularity_image_cache_dir`. Set the cache and
Apptainer temp directory to a fast filesystem:

```bash
export APPTAINER_CACHEDIR=/raid/<user>/apptainer_cache
```

3. **Update the config** to point to your local task directory and cache:

In `configs/harbor_agent_tb2_droid_singularity.yaml`:
```yaml
harbor_datasets:
  terminal_bench:
    local_dataset_path: "/raid/<user>/tb2-tasks"

harbor_environment_kwargs:
  singularity_image_cache_dir: "/raid/<user>/tb2_sif_cache"
```

### Run

Configure `env.yaml` as described in the Docker section above, then:

```bash
export APPTAINER_CACHEDIR=/raid/<user>/apptainer_cache

ng_run \
  '+config_paths=[responses_api_agents/harbor_agent/configs/harbor_agent_tb2_droid_singularity.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+default_host=0.0.0.0'
```

Then collect rollouts as usual:

```bash
ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=responses_api_agents/harbor_agent/data/tb2_input.jsonl \
  +output_jsonl_fpath=<output.jsonl> \
  +num_samples_in_parallel=2
```

The first run will be slow as each task's Docker image is converted to `.sif`.
Subsequent runs use the cache.

## Input JSONL format

Each line uses `instance_id` in the form `terminal_bench::<task_name>`:

```json
{"instance_id": "terminal_bench::fix-git", "responses_create_params": {"input": []}, "agent_ref": {"name": "harbor_agent"}}
```

A pre-built input file with all 89 TB2 tasks is at `data/tb2_input.jsonl`.

Use `+num_samples_in_parallel` to control concurrency. 2-4 is recommended.

## Output

Each line in the output JSONL contains:
- **`reward`**: 1.0 (pass) or 0.0 (fail) from the TB2 verifier
- **`output`**: Empty list — Droid does not produce an ATIF trajectory
- **`instance_id`**: The task identifier

Harbor job artifacts (agent traces, verifier output) are written under
`harbor_jobs_dir` (default `jobs/`), organized by `date/dataset/model/`.

## Key design decisions

- **Dynamic settings.json**: Generated per-container from NeMo Gym config.
  No host-side Droid configuration needed. Uses base64 encoding to avoid
  triggering the Singularity environment's command safety filter.

- **Droid custom model ID**: Droid matches `--model custom:<id>` against the
  `id` field in settings.json. The `model` field is sent as the API model
  parameter. If `droid_custom_model_id` is not set, both default to `model_name`.

- **stream-json output**: Captures the full agent trace (tool calls, results) in
  `command-1/stdout.txt`. Droid's default text output produces nothing in
  non-interactive mode.

- **`--skip-permissions-unsafe`**: Bypasses Droid's permission system. Safe because
  each task runs in an isolated, disposable container.

- **`trial_dir.resolve()`**: Trial paths returned as absolute paths to handle the
  cwd difference between Ray workers and the agent server process.

## Running on HPC clusters (SLURM)

> **Note**: This section is guidance based on the Apptainer setup above.
> It has not been validated end-to-end on a specific HPC cluster.

The eval node needs Apptainer and network access (to Docker Hub for image
pulls and to the model endpoint). No GPUs required — the model can be a
remote endpoint (e.g., NVIDIA Inference API) or a self-hosted vLLM server
on a separate GPU node.

### One-time setup (before submitting jobs)

1. Ensure Apptainer is available. On most HPC clusters:
```bash
module load apptainer
apptainer --version
```
If not available as a module, install it following the
[Apptainer docs](https://apptainer.org/docs/admin/latest/installation.html)
or ask your cluster admin.

2. Clone the NeMo Gym repo and create the venv on the shared filesystem:
```bash
git clone -b feat/droid-tb2-cleanup ssh://git@gitlab-master.nvidia.com:12051/yayu/nemo-gym.git /lustre/<user>/nemo-gym
cd /lustre/<user>/nemo-gym
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install "harbor @ git+https://github.com/laude-institute/harbor.git@9dddd797b57ab8a0f9d6352a20fce73abbb29573"
```

3. Prepare TB2 tasks (see the
[Singularity/Apptainer section](#running-tb2-evaluation-singularityapptainer)
above for full steps):
```bash
git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git /lustre/<user>/tb2-tasks
python responses_api_agents/harbor_agent/custom_envs/singularity/scripts/write_min_setup_sh.py \
  --task-root /lustre/<user>/tb2-tasks --force
```

4. Update paths in `configs/harbor_agent_tb2_droid_singularity.yaml` and
configure the model endpoint in `env.yaml`.

### Example batch script

```bash
#!/bin/bash
#SBATCH --job-name=tb2-droid-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

export APPTAINER_CACHEDIR=/lustre/<user>/apptainer_cache

cd /path/to/nemo-gym
source .venv/bin/activate

# Start NeMo Gym servers in the background
ng_run \
  '+config_paths=[responses_api_agents/harbor_agent/configs/harbor_agent_tb2_droid_singularity.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+default_host=0.0.0.0' &

# Wait for servers to be ready. First run creates venvs and installs
# dependencies (~1-2 min). Check with `ng_status` in another terminal.
sleep 120

# Run evaluation
ng_collect_rollouts +agent_name=harbor_agent \
  +input_jsonl_fpath=responses_api_agents/harbor_agent/data/tb2_input.jsonl \
  +output_jsonl_fpath=results/tb2_eval_output.jsonl \
  +num_samples_in_parallel=2
```

Ensure `singularity_image_cache_dir` and `harbor_jobs_dir` point to paths
on the shared filesystem so artifacts persist across jobs. On the first run,
image conversion may take 2-5 minutes per task — subsequent runs use the cache.

## Known issues

### Droid reasoning trace not visible

Droid's `stream-json` output shows `tool_call`/`tool_result` events but no
assistant `message` events with the model's reasoning text. The model may still
be reasoning, but Droid does not surface it in the trace.

### Non-determinism

Expect ~3-5 tasks to flip between pass and fail across runs.
