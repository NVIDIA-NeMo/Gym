# GeneBench-Pro

[GeneBench-Pro public case studies](https://huggingface.co/datasets/openai/genebench-pro-public-package)
contains 10 genomics and statistical-genetics analysis problems with staged input files,
public reference answers, and a deterministic Python grader.

This benchmark scaffolds evaluation as [Harbor tasks](https://www.harborframework.com/docs/tasks)
via Gym using OpenCode within Singularity containers.

## Usage

Run these commands from the repository root.

Build the task environment image:

```shell
docker build --file benchmarks/genebench_pro/Dockerfile \
  --tag benchmarks/genebench_pro:latest \
  --platform linux/amd64,linux/arm64 benchmarks/genebench_pro
```

Prepare the benchmark dataset using the task environment image:

```shell
gym eval prepare --benchmark genebench_pro \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/genebench_pro/data" \
  ++prepare_script_args.docker_image=benchmarks/genebench_pro:latest
```

The adapter supports `++prepare_script_args.limit=1` for a single-task smoke test
and `++prepare_script_args.overwrite=true` to replace existing generated tasks.

Singularity must be installed. Set `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR`
to existing host directories, and export `NVINF_API_KEY` for the NVIDIA Inference example below.

Run the benchmark evaluation:

```shell
gym eval run --resume --concurrency 8 \
  --benchmark genebench_pro \
  --split benchmark \
  --model nvinf/nvidia/nvidia/nemotron-3-ultra \
  --output "${PWD}/benchmarks/genebench_pro/logs/rollouts.jsonl" \
  ++prepare_script_args.output_dir="${PWD}/benchmarks/genebench_pro/data"
```

No judge model or judge API credentials are required. The verifier maps the
reference grader's authoritative `passed` decision to a binary reward: 1.0 for pass, 0.0 for fail.
Its `score` output is diagnostic only and must not be used as the benchmark reward.

The agent must save its final JSON object to `/app/eval_answer.json`. Verification preserves
the submission and `grading_result.json` under `/logs/verifier/` and writes `reward.json`
for Harbor. The reference grader is invoked through its CLI; missing submissions, malformed JSON,
or grader errors fail verification rather than being treated as incorrect answers.

## Configuration

| Variable | Type | Description |
| --- | --- | --- |
| `prepare_script_args.output_dir` | Hydra value | Absolute directory path for generated tasks and `gym.jsonl`. |
| `prepare_script_args.docker_image` | Hydra value | Shared task environment image. |
| `output_jsonl_fpath` | Hydra value | Absolute rollout output path, set by `-o/--output`. |
| `policy_model_name` | Hydra value | OpenCode model name, automatically set by `-m/--model`. |
| `SINGULARITY_CACHEDIR` | Environment variable | Singularity image cache directory; `/harbor` is appended. |
| `SINGULARITY_TMPDIR` | Environment variable | Singularity temporary directory; must already exist on the host. |
| `NVINF_API_KEY` | Environment variable | NVIDIA Inference API key for the OpenCode provider. |
| `VLLM_API_BASE` | Environment variable | API base URL for the self-hosted vLLM provider. |

## Dataset Reference

### Source

The source is [openai/genebench-pro-public-package](https://huggingface.co/datasets/openai/genebench-pro-public-package),
released under the MIT License. Each problem includes `eval_config.json`, staged `data_files/`,
and a public scientific report. The package also includes `reference_grader.py`, grading definitions,
and file checksums.

This is a public case-study and reproducibility release, not a hidden-answer leaderboard.
Reference answers and reports must remain outside the agent workspace. Their public availability
still permits memorization or retrieval if unrestricted internet access is enabled.

Although the released prompts mention reasoning quality, the public reference grader evaluates
answer fields rather than the scientific quality of the submitted reasoning.

### Harbor

The adapter generates the Harbor dataset at `benchmarks/genebench_pro/data` by default,
with one task per source problem and a `gym.jsonl` index for NeMo Gym:

```text
data/
├── gym.jsonl
└── genebench-pro__<eval_id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   ├── opencode.json
    │   └── data_files/
    │       └── <listed source input files>
    └── tests/
        ├── eval_config.json
        ├── reference_grader.py
        ├── LICENSE
        └── test.sh
```

- `gym.jsonl`: Task-name index used by the Gym Harbor agent.
- `task.toml`: Rendered task template with the task name, shared Docker image, and agent/verifier timeouts.
- `instruction.md`: Source prompt with input-location and submission-file instructions appended by the adapter.
- `environment/opencode.json`: OpenCode provider configuration used by the rollout agent.
- `environment/data_files/`: Only the inputs listed in the source config, preserving their relative paths.
  These are available to the agent under `/app/data_files/`.
- `tests/eval_config.json`: Unmodified source config, including ground truth and grading rules, for verification only.
- `tests/reference_grader.py` and `tests/LICENSE`: Unmodified upstream grader and its license.
- `tests/test.sh`: Runs the grader CLI and converts its `passed` decision to Harbor's binary reward.

Public reports are not copied into generated tasks. Reference grading assets are kept outside
`environment/`; the agent writes its submission to `/app/eval_answer.json` during execution.
