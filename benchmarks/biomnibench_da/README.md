# BiomniBench-DA

[BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) is the data-analysis instantiation of [BiomniBench](https://www.biorxiv.org/content/10.64898/2026.05.12.724604v1),
a process-level evaluation framework for LLM agents on real-world biomedical research tasks.

This benchmark contains the utilities required to prepare the benchmark to be evaluated
as [Harbor tasks](https://www.harborframework.com/docs/tasks) for NeMo Gym's `harbor` environment server and `opencode_agent` harness.

## Usage

Run these commands from the repository root. Build and push the task environment image to a registry accessible to your OpenSandbox deployment:

```shell
export BOB_DOCKER_IMAGE="<registry>/biomnibench-da:<tag>"
docker build --file benchmarks/biomnibench_da/Dockerfile --tag "${BOB_DOCKER_IMAGE}" --platform linux/amd64 benchmarks/biomnibench_da
docker push "${BOB_DOCKER_IMAGE}"
```

Prepare the task folders by calling the adapter directly:

```shell
uv run --project benchmarks/biomnibench_da \
  python -m benchmarks.biomnibench_da.harbor.adapter \
  --output-dir "${PWD}/benchmarks/biomnibench_da/data" \
  --docker-image "${BOB_DOCKER_IMAGE}"
```

> [!tip]
> Use `--data-dir <local-source-directory-or-huggingface-repo>` to select the source dataset and `--overwrite` to rebuild completed task folders. Incomplete tasks are rebuilt automatically. Run the adapter with `--help` for all options.

> [!tip]
> Add `--skip-data` when task data is supplied separately by the sandbox provider. This skips downloading and copying the entire `environment/` directory; task instructions and verifier files are still prepared. A sentence under `## Data Files` points to `/mnt/s3-data/nemotron-life-science/harbor-data/biomnibench_da/<task id>/environment/data`. Switching between included and skipped data rebuilds the affected task folders automatically.

> [!tip]
> Task data is downloaded on demand during preparation: each selected task's folder is fetched only when it needs conversion, excluding `environment/` with `--skip-data`. Completed output folders with the requested data mode are reused without downloading their data again. Required data must be available before evaluation begins.

> [!warning]
> Task data is large. Add `--limit 1` to convert and download at most one task; this does not remove tasks generated earlier. Hugging Face downloads require access to [phylobio/BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA); authenticate with `uvx hf auth login` if required.

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for your deployment. The sandbox must be able to reach NeMo Gym's model server on the evaluation host. NeMo Gym installs the OpenCode runtime in the sandbox and supplies its model configuration.

Configure the verifier's judge:

```shell
export JUDGE_MODEL=openai/nvidia/zai-org/glm-5.2
export JUDGE_MODEL_API_BASE=https://inference-api.nvidia.com/v1
export JUDGE_MODEL_API_KEY="${NVINF_API_KEY}"
```

Run the benchmark evaluation. NeMo Gym prepares its runtime inputs and configuration from the generated task folders automatically:

```shell
gym eval run "${PWD}/benchmarks/biomnibench_da/data" \
  --agent opencode_agent --sandbox opensandbox \
  --model-type openai_model \
  --model nvidia/nvidia/nemotron-3-ultra \
  --model-url https://inference-api.nvidia.com/v1 \
  --model-api-key "${NVINF_API_KEY}" \
  --resume --concurrency 8 --num-repeats 4 \
  --output "${PWD}/benchmarks/biomnibench_da/logs/rollouts.jsonl"
```

For a manual smoke run, replace the repeat and concurrency settings with `--limit 1 --num-repeats 1 --concurrency 1` and choose a separate output file. Inspect the agent trajectory, `/app/trace.md`, `/app/answer.txt`, verifier logs, reward, and health report before increasing the task count. The policy model uses the endpoint's model ID without the OpenCode `nvinf/` provider prefix.

## Configuration

| Variable | Type | Description |
| --- | --- | --- |
| `--output-dir` | Adapter argument | Directory for generated Harbor task folders. Pass it as NeMo Gym's positional task target. |
| `--docker-image` | Adapter argument | Task image accessible to the sandbox deployment. |
| `--skip-data` | Adapter argument | Omit the entire `environment/` directory from downloads and generated tasks; supply data separately at the path added to the instruction. |
| `--output` | Gym argument | Rollout output path. Reuse the path and material settings when resuming. |
| `--model`, `--model-url`, `--model-api-key` | Gym arguments | Policy endpoint's model ID, base URL, and API key. |
| `--num-repeats` | Gym argument | Rollouts per task; the full-run example explicitly preserves four repeats. |
| `OPENSANDBOX_DOMAIN` | Environment variable | OpenSandbox service endpoint. |
| `OPENSANDBOX_API_KEY` | Environment variable | OpenSandbox service API key. |
| `JUDGE_MODEL` | Environment variable | Judge model used to score the agent's response. This must use [LiteLLM](https://docs.litellm.ai/) provider prefixes. |
| `JUDGE_MODEL_API_BASE` | Environment variable | API base URL for the judge model. |
| `JUDGE_MODEL_API_KEY` | Environment variable | API key for the judge model. Optional for self-hosted models. |

## Dataset Reference

### Source

The adapter relies on the HuggingFace dataset.
See [phylobio/BiomniBench-DA](https://huggingface.co/datasets/phylobio/BiomniBench-DA) for source structure.

### Harbor

The adapter creates one Harbor task for each selected `da-*` directory in the source dataset under `--output-dir`:

```text
data/
└── bob-task__<task id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   └── data/
    │       └── <source task data>
    └── tests/
        ├── llm_judge.py
        ├── rubric.txt
        └── test.sh
```

- `task.toml`: Source task configuration with the shared Docker image, `/app` working directory, and judge environment variables. Source resource limits and agent/verifier timeouts are preserved.
- `instruction.md`: Task-specific instruction copied from the source dataset.
- `environment/data/`: Public data files copied from the source task; omitted with `--skip-data`.
- `tests/llm_judge.py`: The original file is modified to use a general `litellm` for general model providers.
- `tests/rubric.txt`: Used unchanged from the source task.
- `tests/test.sh`: Minor modifications from the source task to avoid re-installing required dependencies.

The verifier grades the required trace and answer against the source rubric and writes `reward.json` with a normalized `score` in `[0, 1]`. NeMo Gym's Harbor verifier reads this single-key reward without a custom reward-key setting. The judge and scoring logic are preserved.

Regenerate task folders from the source before switching an existing run to this workflow. Folders created by the old adapter have no completion marker and are rebuilt on first preparation. Use a new rollout output path when changing the harness or generated task definitions.
