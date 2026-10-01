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
uv run gym eval prepare --benchmark biomnibench_da \
  ++prepare_script_args.docker_image=benchmarks/biomnibench_da:latest \
  ++prepare_script_args.output_dir=benchmarks/biomnibench_da/data
```

> [!tip]
> Use `++prepare_script_args.<name>=<value>` to set or override preparation script arguments (see the adapter's [`prepare` function](./harbor/adapter.py#L144)).
> Add `++prepare_script_args.overwrite=true` to overwrite existing generated tasks.

> [!warning]
> This preparation is a large download. Use `++prepare_script_args.limit=5` to prepare only five tasks for debugging.

Start the environment servers:
```shell
TODO
```

Run the evaluation:
```shell
TODO
```

## Configuration

| Variable | Type | Description |
| --- | --- | --- |
| `harbor_dataset_path` | Hydra value | Path to the generated Harbor dataset. |
| `output_jsonl_fpath` | Hydra value | Rollout output path; used as the default jobs directory. Automatically set when using the `-o/--output` flag in the script above. |
| `policy_model_name` | Hydra value | Model name passed to the OpenCode agent. Must use OpenCode provider in [opencode.json](./harbor/task-template/environment/opencode.json). Automatically set when using `-m/--model` flag in the script above. |
| `SINGULARITY_CACHEDIR` | Environment variable | Singularity image cache directory; `/harbor` is appended. |
| `SINGULARITY_TMPDIR` | Environment variable | Singularity temp directory; Must already exist on host. |
| `SINGULARITY_FORCE_PULL` | Environment variable | Whether Singularity should force-pull images. Defaults to `false`. |
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
`~/store/data/harbor/bob`. It creates one Harbor task for each `da-*` directory
in the source dataset and a `gym.jsonl` index for use with NeMo Gym:

```text
bob/
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
