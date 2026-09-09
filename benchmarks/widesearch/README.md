# WideSearch

[WideSearch](https://github.com/ByteDance-Seed/WideSearch) evaluates broad web
research: collecting many facts and returning them in a structured Markdown
table. The public dataset contains 200 tasks in English and Chinese.

This integration runs a Gym harness in OpenSandbox with Exa MCP search. The
[`widesearch`](../../resources_servers/widesearch/README.md) resources server
parses and grades the resulting table. Claude Code is the only harness tested
so far.

## Prepare the data

Preparation downloads the task metadata and per-task gold CSV files from
[Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/WideSearch), then
writes `data/widesearch_benchmark.jsonl`.

```bash
ng_prepare_benchmark \
  "+config_paths=[benchmarks/widesearch/config.yaml]"
```

## Run

```bash
export WIDESEARCH_IMAGE=<sandbox-image>
export EXA_API_KEY=<exa-api-key>
export NVIDIA_API_KEY=<nvidia-api-key>
```

The Claude Code config defaults to `nvidia/qwen/qwen3.8-27b`. Start the
NVIDIA-hosted model server and OpenSandbox provider with
`benchmarks/widesearch/config.yaml`, then collect from
`benchmarks/widesearch/data/widesearch_benchmark.jsonl`. Use `--limit 1` for
an end-to-end check before running all 200 tasks.

## Verification

The verifier follows the upstream evaluation specification for required and
unique columns, preprocessing, entity alignment, exact, URL, numeric, date,
and LLM-judged comparisons. It reports row-level and item-level
precision/recall/F1. Reward is the official strict score: `1` only when both
the complete table and every item match.
