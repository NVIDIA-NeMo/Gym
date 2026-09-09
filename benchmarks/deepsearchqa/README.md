# DeepSearchQA

[DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) evaluates
multi-step web research questions with single-answer and set-answer targets.
This benchmark runs a Gym agent harness in OpenSandbox with Exa MCP search and
uses the [`deepsearchqa`](../../resources_servers/deepsearchqa/README.md)
resources server to verify the final answer. Claude Code is the only harness
tested so far.

## Prepare the data

The preparation script downloads the full DeepSearchQA CSV from Hugging Face
and writes approximately 900 tasks to `data/deepsearchqa_benchmark.jsonl`.

```bash
ng_prepare_benchmark \
  "+config_paths=[benchmarks/deepsearchqa/config.yaml]"
```

Five prepared tasks and their example rollouts are checked in under `data/`.

## Run

Set the OpenSandbox image and API credentials before starting the environment:

```bash
export DEEPSEARCHQA_IMAGE=<sandbox-image>
export EXA_API_KEY=<exa-api-key>
export NVIDIA_API_KEY=<nvidia-api-key>
```

The benchmark config defaults to `nvidia/qwen/qwen3.8-27b` for Claude Code.
Start it with the NVIDIA-hosted model server and OpenSandbox provider configs,
then collect rollouts from `data/deepsearchqa_benchmark.jsonl`. Use `--limit 1`
for a quick end-to-end check before running the full benchmark.

## Verification

The verifier compares the submitted answer with the reference answer using the
official DeepSearchQA [`judge_prompt.txt`](../../resources_servers/deepsearchqa/judge_prompt.txt).
It supports both `Single Answer` and `Set Answer` examples and rejects missing
required answers or excessive extra answers.
