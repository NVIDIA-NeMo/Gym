# swe_atlas_qna

Gym benchmark entry for
[SWE-Atlas — Codebase QnA](https://labs.scale.com/leaderboard/sweatlas-qna), a
benchmark of deep codebase-comprehension questions for coding agents.

## What it tests

124 open-ended questions across 10 real repositories (C, Go, Python, TypeScript)
and five categories (architecture & system design, root-cause analysis, code
onboarding, security, API/library usage). The agent explores the repository and
writes an evidence-backed answer, which is graded against per-task expert
**rubrics** by an LLM judge. See
[`resources_servers/swe_atlas_qna`](../../resources_servers/swe_atlas_qna/README.md)
for the rubric-judging protocol and reward.

## Data

Runtime preparation only — the benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `gym eval prepare`) to populate
`data/swe_atlas_qna_benchmark.jsonl`. The SWE-Atlas data is not hosted as a flat
download, so point the script at a local SWE-Atlas checkout via the
`SWE_ATLAS_DIR` environment variable (or `+prepare_script_args.swe_atlas_dir=...`);
without one it falls back to a shallow `git clone` of the public repo.

Each row carries the codebase `question` at the top level (rendered into the
model input by `prompt_config`) and a `verifier_metadata` block with the
`rubrics`, `problem_statement`, and task identity (`repository`, `base_commit`,
`docker_image`, ...).

## Example usage

```bash
# Prepare benchmark data (point at a local SWE-Atlas checkout)
SWE_ATLAS_DIR=/path/to/SWE-Atlas gym eval prepare --benchmark swe_atlas_qna

# Configure the judge endpoint (any OpenAI-compatible endpoint)
export SWE_ATLAS_QNA_JUDGE_BASE_URL=... SWE_ATLAS_QNA_JUDGE_API_KEY=... SWE_ATLAS_QNA_JUDGE_MODEL=...

# Run servers + collect rollouts
gym env start --model-type vllm_model --benchmark swe_atlas_qna

gym eval run --no-serve \
    --agent swe_atlas_qna_benchmark_simple_agent \
    --input benchmarks/swe_atlas_qna/data/swe_atlas_qna_benchmark.jsonl \
    --output results/swe_atlas_qna_rollouts.jsonl \
    --num-repeats 3
```

> **Note:** this entry is paired with `simple_agent` (single-turn, no repository
> access) as a placeholder so the benchmark is preparable and listable today.
> The real SWE-Atlas QnA eval runs through the mini-swe-agent QnA harness, which
> explores the repository in an Apptainer sandbox before answering; the config's
> agent will be swapped once that harness lands.

## Metrics

The headline is the **rubric pass rate** — `reward = 1.0` iff every scored
`must have` rubric passes (the upstream SWE-Atlas strict pass) — alongside a soft
`agg_score` (fraction of rubrics passed) and pass@k. See the
[resources server README](../../resources_servers/swe_atlas_qna/README.md#metrics).
