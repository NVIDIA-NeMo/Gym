# BioMysteryBench

[BioMysteryBench](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench) evaluates agents on real-world bioinformatics tasks with objective ground-truth answers.

This benchmark contains the utilities required to prepare the benchmark to be evaluated as [Harbor tasks](https://www.harborframework.com/docs/tasks) via Gym within Singularity containers.

## Usage

Build the task environment image:
```shell
docker build --file benchmarks/biomysterybench/Dockerfile --tag benchmarks/biomysterybench:latest --platform linux/amd64,linux/arm64 benchmarks/biomysterybench
```

Prepare the benchmark dataset using the task environment image:
```shell
gym eval prepare --benchmark biomysterybench \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/biomysterybench/data" \
  ++prepare_script_args.docker_image=benchmarks/biomysterybench:latest
```

> [!tip]
> Use `++prepare_script_args.<name>=<value>` to set or override preparation arguments (see the adapter's [`prepare` function](./harbor/adapter.py#L155)).
> Add `++prepare_script_args.overwrite=true` to overwrite existing generated tasks.

> [!warning]
> Preparation downloads a large dataset and requires access to [Anthropic/BioMysteryBench-full](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full). Authenticate with `uvx hf auth login` if required. Add `++prepare_script_args.limit=1` to convert one task for debugging; this does not limit the source download.

Singularity must be installed; set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` to existing host directories,
and export `NVINF_API_KEY` for the NVIDIA Inference example below. Make the task image available to Singularity on the evaluation host.

Run the benchmark evaluation:
```shell
JUDGE_MODEL=openai/nvidia/zai-org/glm-5.2 \
JUDGE_MODEL_API_BASE=https://inference-api.nvidia.com/v1 \
JUDGE_MODEL_API_KEY="${NVINF_API_KEY}" \
gym eval run --resume --concurrency 8 \
  --benchmark biomysterybench \
  --split benchmark \
  --model nvinf/nvidia/nvidia/nemotron-3-ultra \
  --output "${PWD}/benchmarks/biomysterybench/logs/rollouts.jsonl" \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/biomysterybench/data"
```

### Configuration Variables

| Variable | Type | Description |
| --- | --- | --- |
| `prepare_script_args.output_dir` | Hydra value | Absolute directory path for generated Harbor tasks and `gym.jsonl`. |
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
| `JUDGE_ATTEMPTS` | Environment variable | Judge calls tried before the trial is left without a reward, `3` by default. |
| `JUDGE_TIMEOUT` | Environment variable | Seconds per judge call, `300` by default. |
| `JUDGE_MAX_TOKENS` | Environment variable | Token limit of the judge's answer, reasoning included, `16384` by default. |

## Dataset Reference

### Source

The adapter relies on the HuggingFace dataset.
See [Anthropic/BioMysteryBench-full](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full) for source structure.

### Harbor

The adapter writes one Harbor directory containing all converted tasks. There
are no dataset splits. Each generated task starts from
[`task-template/`](./harbor/task-template); `task.toml`, the instruction, and the
judge prompt are rendered with task-specific values. The matching source
archive is extracted into `environment/data/`.

All tasks use `gitlab-master.nvidia.com/nemotron-life-science/benchmarks/bmb:latest`. This image is built automatically by CI from the [`Dockerfile`](./Dockerfile).

```text
harbor-bmb/
├── gym.jsonl
└── bmb-task__<task id>/
    ├── task.toml
    ├── environment/
    │   ├── opencode.json
    │   └── data/
    │       └── <extracted task data files>
    ├── instruction.md
    └── tests/
        ├── grade.py
        ├── prompt.txt
        └── test.sh
```

- `gym.jsonl`: Harbor evaluation inputs for all tasks. Each row follows the
  `HarborRunRequest` schema from [pkgs/Gym/responses_api_agents/harbor_agent/app.py](../../pkgs/Gym/responses_api_agents/harbor_agent/app.py):
  ```json
  {"task_name":"bmb-task__<task id>","responses_create_params":{"input":[]}}
  ```
- `task.toml`: [Harbor task configuration metadata](https://www.harborframework.com/docs/tasks#configuration--metadata), including the Docker image and timeouts.
- `environment/opencode.json`: [OpenCode environment configuration](https://opencode.ai/docs/config/#schema).
- `environment/data/`: The extracted data files made available to the agent.
- `instruction.md`: The task-specific question presented to the rollout agent.
- `tests/prompt.txt`: The task-specific judging prompt, including the private answer rubric.
- `tests/grade.py`: LLM-based verifier that grades the final trajectory response and writes a numeric reward.
- `tests/test.sh`: Verifier entrypoint; it runs [tests/grade.py](./harbor/task-template/tests/grade.py) with `litellm`.
