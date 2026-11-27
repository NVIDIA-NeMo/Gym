# GeneBench-Pro

[GeneBench-Pro public case studies](https://huggingface.co/datasets/openai/genebench-pro-public-package)
contains 10 genomics and statistical-genetics analysis problems with staged input files,
public reference answers, and a deterministic Python grader.

This benchmark scaffolds evaluation as [Harbor tasks](https://www.harborframework.com/docs/tasks)
for NeMo Gym's `harbor` environment server and `opencode_agent` harness.

## Usage

Run these commands from the repository root.

Build and push the task environment image to a registry accessible to your OpenSandbox deployment:

```shell
export GENEBENCH_DOCKER_IMAGE="<registry>/genebench-pro:<tag>"
docker build --file benchmarks/genebench_pro/Dockerfile \
  --tag "${GENEBENCH_DOCKER_IMAGE}" \
  --platform linux/amd64 benchmarks/genebench_pro
docker push "${GENEBENCH_DOCKER_IMAGE}"
```

Prepare the task folders by calling the adapter directly:

```shell
uv run --project benchmarks/genebench_pro \
  python -m benchmarks.genebench_pro.harbor.adapter \
  --output-dir "${PWD}/benchmarks/genebench_pro/data" \
  --docker-image "${GENEBENCH_DOCKER_IMAGE}"
```

Use `--data-dir <local-source-directory-or-huggingface-repo>` to select the source package,
`--limit 1` to prepare and download at most one task, and `--overwrite` to rebuild completed tasks.
Incomplete tasks are rebuilt automatically. Completed tasks with the requested data mode are reused
without downloading their inputs again. Remote preparation lists problem configs and fetches only each
selected config, the shared grader and license, and the selected task's listed input files.

Add `--skip-data` when inputs are supplied separately by the sandbox provider. This skips downloading
and copying task input files and omits the entire `environment/` directory. Pass `--data-path-prefix /var/data`
to point instructions to `/var/data/genebench-pro__<eval_id>/environment/data_files/`. Without a prefix,
instructions point to `/app/data_files/`.
Switching data modes rebuilds affected folders. Required inputs must be available before evaluation;
`--limit` does not remove previously generated tasks.

Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for your deployment. The sandbox must be able to
reach Gym's model server on the evaluation host. Gym installs OpenCode inside the sandbox and supplies
its model configuration.

Run the benchmark evaluation:

```shell
gym eval run "${PWD}/benchmarks/genebench_pro/data" \
  --agent opencode_agent --sandbox opensandbox \
  --model-type openai_model \
  --model nvidia/nvidia/nemotron-3-ultra \
  --model-url https://inference-api.nvidia.com/v1 \
  --model-api-key "${NVINF_API_KEY}" \
  --resume --concurrency 8 --num-repeats 4 \
  --output "${PWD}/benchmarks/genebench_pro/logs/rollouts.jsonl"
```

For a manual smoke run, replace the repeat and concurrency settings with
`--limit 1 --num-repeats 1 --concurrency 1` and use a separate output file. Inspect the agent trajectory,
`/app/eval_answer.json`, verifier logs, reward, and health report before increasing the task count.
The policy model ID uses the endpoint's model ID without the OpenCode `nvinf/` provider prefix.

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
| `--output-dir` | Adapter argument | Generated Harbor task directory, passed as Gym's positional task target. |
| `--docker-image` | Adapter argument | Shared task image accessible to the sandbox deployment. |
| `--skip-data` | Adapter argument | Omit input files and `environment/`; supply data at the path in the instruction. |
| `--output` | Gym argument | Rollout output path. Reuse the path and material settings when resuming. |
| `--model`, `--model-url`, `--model-api-key` | Gym arguments | Policy endpoint's model ID, base URL, and API key. |
| `--num-repeats` | Gym argument | Rollouts per task; the example preserves four repeats. |
| `OPENSANDBOX_DOMAIN` | Environment variable | OpenSandbox service endpoint. |
| `OPENSANDBOX_API_KEY` | Environment variable | OpenSandbox service API key. |

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

The adapter generates one Harbor task per selected source problem under `--output-dir`:

```text
data/
└── genebench-pro__<eval_id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   └── data_files/
    │       └── <listed source input files>
    └── tests/
        ├── eval_config.json
        ├── reference_grader.py
        ├── LICENSE
        └── test.sh
```

- `task.toml`: Rendered task template with the task name, shared Docker image, 2 CPUs, 16384 MiB of memory
  (matching BiomniBench-DA), and agent/verifier timeouts.
- `instruction.md`: Source prompt with input-location and submission-file instructions appended by the adapter.
- `environment/data_files/`: Only the inputs listed in the source config, preserving their relative paths.
  These are available under `/app/data_files/`; omitted with `--skip-data`.
- `tests/eval_config.json`: Unmodified source config, including ground truth and grading rules, for verification only.
- `tests/reference_grader.py` and `tests/LICENSE`: Unmodified upstream grader and its license.
- `tests/test.sh`: Runs the grader CLI and converts its `passed` decision to Harbor's binary reward.

Public reports are not copied into generated tasks. Reference grading assets are kept outside
`environment/`; the agent writes its submission to `/app/eval_answer.json` during execution.

Regenerate task folders before switching an existing run to this workflow. Folders from the previous
adapter have no `.done` marker and are rebuilt on preparation. Use a new rollout output path when
changing the harness or generated task definitions. Agent and verifier timeouts remain 3600 and 600
seconds respectively in the task template; grading logic and binary rewards are unchanged.
