# FrontierFinance — NeMo Gym benchmark

[FrontierFinance](https://huggingface.co/datasets/samaya-ai/FrontierFinance) has 220
expert-written investor-research queries across six use cases (financial data and
modeling, sector and macro, earnings and events, company research, coverage and
catalyst monitoring, screening and discovery), with 11,543 rubrics in total. Each
query is answered "as of" its own date.

The agent runs on the Finance Agent v2 tools with every dated tool call capped at the
query's date, and answers are graded by the official FrontierFinance judge panel. See
the [resources server README](../../resources_servers/frontierfinance/README.md) for
how the date cap and grading work.

## Layout

| Path | Purpose |
|---|---|
| `config.yaml` | Benchmark overlay on `resources_servers/frontierfinance/configs/frontierfinance.yaml`. |
| `prepare.py` | Downloads the dataset at a pinned revision and converts it to Gym JSONL. |
| `data/frontier_finance_220q.jsonl` | The prepared dataset. Not committed. |

## Prepare

The dataset is gated. Accept its terms on Hugging Face, then:

```bash
source .venv/bin/activate
export HF_TOKEN=...
gym eval prepare --benchmark frontier_finance
```

To convert a copy you already have, place it at
`benchmarks/frontier_finance/data/frontier_finance_public.jsonl`, or run
`python benchmarks/frontier_finance/prepare.py --input <file> --output <file>`.

## Run

Set the tool keys and judge endpoints in `env.yaml` (see the resources server
README), then:

```bash
gym eval run --benchmark frontier_finance \
  -c responses_api_models/openai_model/configs/openai_model.yaml \
  --output results/frontier_finance.jsonl --num-repeats 3 --concurrency 32
```

Add `--limit 2 --num-repeats 1 --concurrency 2` for a smoke run.

The official headline numbers are
`frontier_finance/macro_avg_qualification_rate_on_all_queries` (equal to
`mean/reward`) and its `micro` and `must_have` variants. Samples where every judge
failed are masked and left out of all scores, as in the official grader.

## Comparability

Grading uses the official prompt, parser, vote and metrics, and the recommended
judge panel. Scores are comparable to published FrontierFinance results only when
the same judges are used. The agent runs the Finance Agent v2 tools with the paper's
limits of 200 tool calls and 300 seconds of tool calling per question, stated in the
system prompt as in the paper. Remaining differences: the tools are NeMo Gym's Finance Agent v2
implementation rather than the paper's, and the date cap does not cover live web
pages (see the resources server README).

## Licensing

The dataset is CC-BY-4.0 (Zhang et al., 2026, "FrontierFinance: A Challenging
Benchmark for Measuring Frontier Intelligence of Finance Agents") and may not be
reshared outside a gated repository. The code here is Apache-2.0.
