# BixBench-Hypothesis

[BixBench-Hypothesis](https://www.futurehouse.org/research/bixbench) evaluates AI agents on real-world bioinformatics analyses that determine whether provided datasets support a scientific hypothesis.

This benchmark is a reimplementation as a set of [Harbor tasks](https://www.harborframework.com/docs/tasks) to be evaluated via Gym within Singularity containers.

## Usage

Build the task environment image:
```shell
docker build --file benchmarks/bixbench_hypothesis/Dockerfile --tag benchmarks/bixbench_hypothesis:latest --platform linux/amd64,linux/arm64 benchmarks/bixbench_hypothesis
```

Prepare the benchmark dataset using the task environment image:
```shell
gym eval prepare --benchmark bixbench_hypothesis \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/bixbench_hypothesis/data" \
  ++prepare_script_args.docker_image=benchmarks/bixbench_hypothesis:latest
```

> [!tip]
> Use `++prepare_script_args.<name>=<value>` to set or override preparation script arguments (see the adapter's [`prepare` function](./harbor/adapter.py)).
> Add `++prepare_script_args.overwrite=true` to overwrite existing generated tasks.

> [!warning]
> Preparation downloads the task dataset and capsule data from HuggingFace, which can be large.
> `++prepare_script_args.limit=1` prepares only one task and downloads only its capsule, but still loads the full task dataset.

Singularity must be installed; set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` to existing host directories,
and export `NVINF_API_KEY` for the NVIDIA Inference example below.

Run the benchmark evaluation:
```shell
RUBRIC_MODEL=openai/nvidia/zai-org/glm-5.2 \
RUBRIC_MODEL_API_BASE=https://inference-api.nvidia.com/v1 \
RUBRIC_MODEL_API_KEY="${NVINF_API_KEY}" \
gym eval run --resume --concurrency 8 \
  --benchmark bixbench_hypothesis \
  --split benchmark \
  --model nvinf/nvidia/nvidia/nemotron-3-ultra \
  --output "${PWD}/benchmarks/bixbench_hypothesis/logs/rollouts.jsonl" \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/bixbench_hypothesis/data"
```

### Configuration Variables

| Variable | Type | Description |
| --- | --- | --- |
| `prepare_script_args.output_dir` | Hydra value | Absolute directory path for generated Harbor tasks and `gym.jsonl`. |
| `output_jsonl_fpath` | Hydra value | Absolute rollout output path. Set by `-o/--output` on `gym eval run`. |
| `policy_model_name` | Hydra value | Model name passed to the OpenCode agent. Must use OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json). Automatically set when using `-m/--model` flag in the script above. |
| `SINGULARITY_CACHEDIR` | Environment variable | Singularity image cache directory; `/harbor` is appended. |
| `SINGULARITY_TMPDIR` | Environment variable | Singularity temp directory; Must already exist on host. |
| `NVINF_API_KEY` | Environment variable | NVIDIA Inference API key for OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json) |
| `VLLM_API_BASE` | Environment variable | vLLM API base for OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json) |
| `RUBRIC_MODEL` | Environment variable | Rubric model used to score the agent's response. This must use [LiteLLM](https://docs.litellm.ai/) provider prefixes. |
| `RUBRIC_MODEL_API_BASE` | Environment variable | API base URL for the rubric model. |
| `RUBRIC_MODEL_API_KEY` | Environment variable | API key for the rubric model. Optional for self-hosted models. |
| `RUBRIC_ATTEMPTS` | Environment variable | Rubric model calls tried before the agent is told judging failed, 2 by default. |
| `RUBRIC_TIMEOUT` | Environment variable | Seconds per rubric model call, 120 by default. Attempts times timeout must stay under the MCP timeout in [opencode.json](./harbor/task-template/environment/opencode.json). |
| `RUBRIC_MAX_TOKENS` | Environment variable | Token limit of the rubric model's answer, reasoning included, 16384 by default. |

## Dataset Reference

### Source

See [EdisonScientific/bixbench_hypothesis](https://huggingface.co/datasets/EdisonScientific/bixbench_hypothesis) for the source dataset.

### Harbor

The adapter generates the Harbor benchmark with one directory per task and a `gym.jsonl` index.

Each generated task starts from [`task-template/`](./harbor/task-template); `task.jsonl`, `instruction.md`, and `bbh.md` are
rendered with task-specific values, while the matching source capsule is copied to `environment/data/`.

```text
data/
├── gym.jsonl
└── bbh-task__<task id>/
    ├── task.toml
    ├── environment/
    │   ├── opencode.json
    │   ├── task.jsonl
    │   ├── data/
    │   │   └── <copied capsule contents>/
    │   └── .opencode/agents/bbh.md
    └── steps/rollout/
        ├── instruction.md
        ├── workdir/setup.sh
        └── tests/test.sh
```

- `gym.jsonl`: Harbor evaluation inputs for all generated tasks for use with [NeMo Gym](https://github.com/NVIDIA-NeMo/gym). Each row follows the `HarborRunRequest` schema.
  ```json
  {"task_name":"bbh-task__<task id>","responses_create_params":{"input":[]}}
  ```

- `task.toml`: [Harbor task configuration metadata](https://www.harborframework.com/docs/tasks#configuration--metadata), including the Docker image, and timeouts.
- `environment/task.jsonl`: A file with a single record following the `BBHTask` schema, consumed by the [BixBench-Hypothesis MCP server](./pkgs/bbh-mcp/src/bbh_mcp/server.py). This file is deleted after launch to avoid leakage of private task information.
- `environment/opencode.json`: [OpenCode environment configuration](https://opencode.ai/docs/config/#schema).
- `environment/.opencode/agents/bbh.md`: System prompt and operating instructions for an [OpenCode agent via markdown](https://opencode.ai/docs/agents/#markdown).
- `steps/rollout/instruction.md`: Task-specific [prompt](https://www.harborframework.com/docs/tasks#instruction) presented to the agent.
- `steps/rollout/workdir/setup.sh`: Starts the [BixBench-Hypothesis MCP server](./pkgs/bbh-mcp/src/bbh_mcp/server.py).
- `steps/rollout/tests/test.sh`: Collects the MCP server's reward artifact for Harbor.
