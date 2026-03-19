# GDPVal Agent

Multi-turn NeMo-Gym agent for [openai/gdpval](https://huggingface.co/datasets/openai/gdpval) tasks.

This agent:
- runs a tool-using model loop through `/v1/responses`
- executes tools in the `bash_sandbox` resources server
- supports reference-file download/upload for each task
- persists task outputs under per-task directories
- verifies results through the resources server `/verify` endpoint

The implementation is in `responses_api_agents/gdpval_agent/app.py`, with a helper CLI in
`responses_api_agents/gdpval_agent/client.py`.

## Tooling exposed to the model

The agent allows the model to call:
- `run_command`: execute bash in the sandbox
- `web_search`: search the web
- `web_fetch`: fetch webpage content
- `finish`: mark the task complete and optionally save output files

`session_id` and output directory handling are injected by the agent; the model does not need
to provide them.

## Configuration

Primary config file:
- `responses_api_agents/gdpval_agent/configs/gdpval_agent.yaml`

Common runtime knobs:
- `max_steps`: maximum agent turns before stop
- `context_window_tokens`: context budget used for summarization checks
- `context_summarization_cutoff`: summarize history after this fraction of context is used
- `remaining_step_warning_threshold`: inject warning messages when near step limit

This agent is typically run with:
- `resources_servers/bash_sandbox/configs/bash_sandbox.yaml` (resources + agent wiring)
- `responses_api_models/local_vllm_model/configs/...` (model server auto-launched by `ng_run`)

## Quick start

Use the Docker image for this agent. The expected runtime environment is the container defined in
`responses_api_agents/gdpval_agent/Dockerfile`.

### 1) (Optional) set `hf_token` in `env.yaml`

For gated Hugging Face models, set your token so `local_vllm_model` can download weights:

```yaml
hf_token: "hf_..."
```

### 2) Build the GDPVal Docker image

```bash
docker build -t gdpval-agent -f responses_api_agents/gdpval_agent/Dockerfile .
```

### 3) Run a shell in the container (repo mounted at `/workspace`)

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace gdpval-agent bash
```

### 4) Inside the container, install Gym deps

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

### 5) Start NeMo-Gym with auto-launched local vLLM

```bash
ng_run "+config_paths=[resources_servers/bash_sandbox/configs/bash_sandbox.yaml,responses_api_models/local_vllm_model/configs/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.yaml]" \
  ++bash_sandbox_agent.responses_api_agents.gdpval_agent.model_server.name=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

`local_vllm_model` starts `vllm serve` internally, so you do not need to run a separate manual
`vllm serve ...` command.

The provided Nano config defaults to `tensor_parallel_size=8` (8 GPUs). Override parallelism if
you are running on different hardware.

### Alternative: run your own standalone vLLM server

If you prefer to manage vLLM yourself (outside `local_vllm_model`), start a standalone
OpenAI-compatible endpoint in a vllm container:

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0
```

Then run NeMo-Gym with `vllm_model` and pass your endpoint via policy settings:

```bash
ng_run "+config_paths=[resources_servers/bash_sandbox/configs/bash_sandbox.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  ++policy_base_url=http://[STANDALONE_VLLM_SERVER_HOST_IP]:8000/v1 \
  ++policy_api_key=EMPTY \
  ++policy_model_name=NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

If NeMo-Gym is not running in Docker, replace `http://[STANDALONE_VLLM_SERVER_HOST_IP]:8000/v1` with your
actual host or IP (for example `http://localhost:8000/v1`).

## Prepare GDPVal JSONL input

Use the client helper to convert Hugging Face dataset rows into NeMo-Gym run requests:

```bash
python responses_api_agents/gdpval_agent/client.py prepare \
  --split train \
  --output-jsonl responses_api_agents/gdpval_agent/data/train.jsonl \
  --output-dir gdpval_output \
  --validate
```

Optional filters:
- `--limit N`
- `--task-ids id1,id2,...`

## Run rollouts

After servers are up and JSONL is prepared:

```bash
ng_collect_rollouts \
  +agent_name=bash_sandbox_agent \
  +input_jsonl_fpath=responses_api_agents/gdpval_agent/data/train.jsonl \
  +output_jsonl_fpath=responses_api_agents/gdpval_agent/data/train_rollouts.jsonl \
  +limit=5
```

Compute aggregate reward profile:

```bash
ng_reward_profile \
  +input_jsonl_fpath=responses_api_agents/gdpval_agent/data/train.jsonl \
  +rollouts_jsonl_fpath=responses_api_agents/gdpval_agent/data/train_rollouts.jsonl \
  +output_jsonl_fpath=responses_api_agents/gdpval_agent/data/train_profiled.jsonl \
  +pass_threshold=1.0
```

## Direct run mode (dev/debug)

You can post tasks directly to `/run` with:

```bash
python responses_api_agents/gdpval_agent/client.py run --split train --limit 3
```

This mode is useful for debugging individual tasks without the full rollout pipeline.

## Resume, caching, and timeout behavior

For each task, the agent writes sentinels inside its task directory:
- `finish_params.json` indicates the agent reached `finish`
- `reward.json` caches verified results
- `timeout.json` indicates a prior timeout
- `history.json` stores model/tool history for recovery

Behavior:
- if `reward.json` exists, `/run` returns cached result and skips work
- if only `finish_params.json` exists, `/run` skips model loop and re-runs verify
- if timeout occurred, the task is retried from scratch

## Docker image notes

`responses_api_agents/gdpval_agent/Dockerfile` defines a Ubuntu-based environment with
Python 3.12, LibreOffice, OCR/PDF/media/geospatial tooling, Graphviz, Java, and build tools.

## Tests

Run the agent test module:

```bash
pytest responses_api_agents/gdpval_agent/tests/test_app.py -x
```

## Licensing information

Code: Apache 2.0  
Data: GDPVal dataset terms apply (see dataset source).

Dependencies:
- `nemo_gym`: Apache 2.0
