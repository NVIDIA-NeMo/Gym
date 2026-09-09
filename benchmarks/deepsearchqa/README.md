# DeepSearchQA

[DeepSearchQA](https://huggingface.co/datasets/google/deepsearchqa) evaluates
multi-step web research questions with single-answer and set-answer targets.
This benchmark runs a Gym agent harness in a sandbox with Exa MCP search and
uses the [`deepsearchqa`](../../resources_servers/deepsearchqa/README.md)
resources server to verify the final answer. 

## Prepare the data

The preparation script downloads the full DeepSearchQA CSV from Hugging Face
and writes approximately 900 tasks to `data/deepsearchqa_benchmark.jsonl`.

```bash
gym eval prepare --benchmark deepsearchqa
```

Five prepared tasks and their example rollouts are checked in under the
[`deepsearchqa` resources server](../../resources_servers/deepsearchqa/data/).

## Run

The benchmark config defaults to `nvidia/qwen/qwen3.8-27b` in Claude Code.
Use `--limit 1` for a quick end-to-end check before running the full benchmark.

## Verification

The verifier compares the submitted answer with the reference answer using the
official DeepSearchQA [`judge_prompt.txt`](../../resources_servers/deepsearchqa/judge_prompt.txt).
It supports both `Single Answer` and `Set Answer` examples and rejects missing
required answers or excessive extra answers.
