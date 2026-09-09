# WideSearch

[WideSearch](https://github.com/ByteDance-Seed/WideSearch) evaluates broad web
research: collecting many facts and returning them in a structured Markdown
table. The public dataset contains 200 tasks in English and Chinese.

This integration runs a Gym harness in a sandbox with Exa MCP search. The
[`widesearch`](../../resources_servers/widesearch/README.md) resources server
parses and grades the resulting table.

## Prepare the data

Preparation downloads the task metadata and per-task gold CSV files from
[Hugging Face](https://huggingface.co/datasets/ByteDance-Seed/WideSearch), then
writes `data/widesearch_benchmark.jsonl`.

```bash
gym eval prepare --benchmark widesearch
```

Five prepared tasks and their diagnostic rollouts are checked in under the
[`widesearch` resources server](../../resources_servers/widesearch/data/).

## Run

The Claude Code config defaults to `nvidia/qwen/qwen3.8-27b`. Use `--limit 1` for
an end-to-end check before running all 200 tasks.

## Verification

The verifier follows the upstream evaluation specification for required and
unique columns, preprocessing, entity alignment, exact, URL, numeric, date,
and LLM-judged comparisons. It reports row-level and item-level
precision/recall/F1. Reward is the official strict score: `1` only when both
the complete table and every item match.
