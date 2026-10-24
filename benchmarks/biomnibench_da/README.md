# BiomniBench-DA

[BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) is the data-analysis instantiation of [BiomniBench](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v1), 
a process-level evaluation framework for LLM agents on real-world biomedical research tasks.

This benchmark contains the utilities required to prepare the benchmark to be evaluated
as [Harbor tasks](https://www.harborframework.com/docs/tasks) via Gym within Singularity containers.

## Usage

Build the task environment image:
```shell
docker build --file benchmarks/biomnibench_da/Dockerfile --tag benchmarks/biomnibench_da:latest --platform linux/amd64,linux/arm64 benchmarks/biomnibench_da
```

Prepare the benchmark dataset using the task environment image:
```shell
gym eval prepare --benchmark biomnibench_da \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/biomnibench_da/data" \
  ++prepare_script_args.docker_image=benchmarks/biomnibench_da:latest
```

> [!tip]
> Use `++prepare_script_args.<name>=<value>` to set or override preparation script arguments (see the adapter's [`prepare` function](./harbor/adapter.py#L146)).
> Add `++prepare_script_args.overwrite=true` to overwrite existing generated tasks.

> [!warning]
> Preparation downloads the full dataset, which is large. `++prepare_script_args.limit=1` can prepare only one task
> for debugging, but does not limit the download; the full dataset is always downloaded.

Singularity must be installed; set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` to existing host directories,
and export `NVINF_API_KEY` for the NVIDIA Inference example below.

Run the benchmark evaluation:
```shell
JUDGE_MODEL=openai/nvidia/zai-org/glm-5.2 \
JUDGE_MODEL_API_BASE=https://inference-api.nvidia.com/v1 \
JUDGE_MODEL_API_KEY="${NVINF_API_KEY}" \
gym eval run --resume --concurrency 8 \
  --benchmark biomnibench_da \
  --split benchmark \
  --model nvinf/nvidia/nvidia/nemotron-3-ultra \
  --output "${PWD}/benchmarks/biomnibench_da/logs/rollouts.jsonl" \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/biomnibench_da/data"
```

## Configuration

| Variable | Type | Description |
| --- | --- | --- |
| `prepare_script_args.output_dir` | Hydra value | Absolute directory path for generated Harbor tasks and `gym.jsonl`. |
| `output_jsonl_fpath` | Hydra value | Absolute rollout output path. Set by `-o/--output` on `gym eval run`. |
| `policy_model_name` | Hydra value | Model name passed to the OpenCode agent. Must use OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json). Automatically set when using `-m/--model` flag in the script above. |
| `SINGULARITY_CACHEDIR` | Environment variable | Singularity image cache directory; `/harbor` is appended. |
| `SINGULARITY_TMPDIR` | Environment variable | Singularity temp directory; Must already exist on host. |
| `NVINF_API_KEY` | Environment variable | NVIDIA Inference API key for OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json) |
| `VLLM_API_BASE` | Environment variable | vLLM API base for OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json) |
| `JUDGE_MODEL` | Environment variable | Judge model used to score the agent's response. This must use [LiteLLM](https://docs.litellm.ai/) provider prefixes. |
| `JUDGE_MODEL_API_BASE` | Environment variable | API base URL for the judge model. |
| `JUDGE_MODEL_API_KEY` | Environment variable | API key for the judge model. Optional for self-hosted models. |

## Dataset Reference

### Source

The adapter relies on the HuggingFace dataset.
See [phylobio/BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) for source structure.

### Harbor

The adapter generates the Harbor benchmark at
`benchmarks/biomnibench_da/data` by default. It creates one Harbor task for each `da-*` directory
in the source dataset and a `gym.jsonl` index for use with NeMo Gym:

```text
data/
├── gym.jsonl
└── bob-task__<task id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   ├── opencode.json
    │   └── data/
    │       └── <source task data>
    └── tests/
        ├── llm_judge.py
        ├── rubric.txt
        └── test.sh
```

- `gym.jsonl`: Harbor evaluation inputs for all generated tasks.
- `task.toml`: Source task configuration with the adapter-provided Docker image and judge environment variables. The original `task.toml` is modified with some environment and verifier configurations.
- `instruction.md`: Task-specific instruction copied from the source dataset.
- `environment/data/`: Public data files copied from the source task.
- `environment/opencode.json`: OpenCode configuration used by the rollout agent.
- `tests/llm_judge.py`: The original file is modified to use a general `litellm` for general model providers.
- `tests/rubric.txt`: Used unchanged from the source task.
- `tests/test.sh`: Minor modifications from the source task to avoid re-installing required dependencies.
