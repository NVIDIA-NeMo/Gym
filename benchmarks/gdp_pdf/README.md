# GDP.pdf Benchmark

Benchmark wrapper around the `gdp_pdf` resources server. See
[`resources_servers/gdp_pdf/README.md`](../../resources_servers/gdp_pdf/README.md) for the full
description of the dataset, grading, and configuration.

100 expert-written prompts over real professional PDFs across ten domains, graded per atomic rubric
criterion (1,275 criteria total) by an LLM judge. The default scalar reward is the mean criterion
pass rate, while the strict **all-pass rate** is also serialized and reported. All-pass is the share
of attempts where every criterion passed and is reported over **5 attempts per task**, matching the
published protocol.

> **Evaluation use only.** The upstream dataset has no train split by design; training on it
> contaminates the benchmark.

## Prepare

The single-turn dataset declares `prepare_project: resources_servers/gdp_pdf`. Gym syncs
that component's committed lock with `uv sync --locked` and runs preparation in
an isolated environment under `uv_venv_dir`. It does not modify the running Gym
interpreter or the agent's environment. The lock pins `liteparse` 2.5.0, the
version Artificial Analysis uses.

Then prepare the dataset:

```bash
gym eval prepare --benchmark gdp_pdf
```

Downloads the dataset table and ~467 MB of source PDFs into
`resources_servers/gdp_pdf/data/media/`, renders each document once (LiteParse OCR for text, page
screenshots at 150 DPI) into `resources_servers/gdp_pdf/data/documents/`, then writes
`data/gdp_pdf_benchmark.jsonl`. Already-rendered documents are skipped on re-run, unless they were
rendered by another LiteParse version: each manifest records the version, and prepare warns when the
installed one is not 2.5.0.

## Run

Set `judge_base_url` / `judge_api_key` / `judge_model_name` in `env.yaml`, then run the benchmark
with a vision-capable policy model.

## Scoring like Artificial Analysis

The defaults follow [Artificial Analysis's GDP.pdf method](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf):
LiteParse 2.5.0 with English OCR, page images at 150 DPI before the task and the complete extracted
text, five attempts per task, and GPT-5.6 Luna at medium reasoning effort judging one criterion per
call. Two things are set per run:

- **Sampling.** Artificial Analysis samples reasoning models at temperature 0.6 and non-reasoning
  models at 0, unless the model's developer recommends another, and allows the model's maximum
  output length.
- **The denominator.** Artificial Analysis divides by all 500 attempts, scores errors and missing
  attempts as zero, and accepts a grading only when every criterion has a verdict. Gym's aggregate
  leaves failed rows out instead: a row whose judge call failed goes to
  `<rollouts_stem>_failures.jsonl`. Re-judge those rows with `gym eval reverify --judge-failed-only`,
  then score the run over every task and attempt:

```bash
python benchmarks/gdp_pdf/report_aa.py \
  --expected benchmarks/gdp_pdf/data/gdp_pdf_benchmark.jsonl \
  --rollouts <run_dir>/rollouts.jsonl --output <run_dir>/gdp_pdf_aa_report.json
```

The report reads the run through its rollout store, so a re-judged or re-verified result replaces the
one it settles. Rollouts without a run manifest are read with the failures file next to them. It
counts every missing or failed attempt as zero, and gives all-pass, Mean Pass (each attempt's
criterion pass rate, averaged over tasks and attempts), Mean Pass by domain and the outcome of every
attempt. `judging_complete` stays false while any attempt is missing, failed or lacks a verdict for
one of its criteria; empty answers and inputs that never fit are final zeros.

## Comparability

Absolute scores are not directly comparable to Surge's published numbers. Surge's harness hands the
raw PDF to native provider document APIs (Claude/Gemini/OpenAI); that has no equivalent for
open-weight models served via vLLM, so we follow Artificial Analysis's approach instead --
LiteParse-extracted text plus rendered page images, with a reactive DPI backoff and page
compositing under tight image-count limits (see
`responses_api_agents/gdp_pdf_agent/README.md`). The judge model also differs, and (unlike Surge's
own scorer) our judge is shown the task prompt alongside each criterion. Artificial Analysis makes
a similar caveat about their own reimplementation not being interchangeable with Surge's.

## OpenCode

`benchmarks/gdp_pdf/opencode.yaml` prepares the same questions and raw PDFs without
OCR or rendering; it does not need the single-turn preparation environment.
OpenCode can inspect pages and run Python in a sandbox. Follow the
[OpenCode setup and evaluation recipe](../../resources_servers/gdp_pdf/README.md#gdppdf-with-opencode).
These scores use a different protocol from the single-turn scores above.
