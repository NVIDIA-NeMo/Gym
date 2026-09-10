# GDP.pdf Benchmark

Benchmark wrapper around the `gdp_pdf` resources server. See
[`resources_servers/gdp_pdf/README.md`](../../resources_servers/gdp_pdf/README.md) for the full
description of the dataset, grading, and configuration.

100 expert-written prompts over real professional PDFs across ten domains, graded per atomic rubric
criterion (1,275 criteria total) by an LLM judge. The headline metric is the **all-pass rate** —
the share of attempts where every criterion passed — reported over **5 attempts per task**, matching
the published protocol.

> **Evaluation use only.** The upstream dataset has no train split by design; training on it
> contaminates the benchmark.

## Prepare

```bash
gym eval prepare --benchmark gdp_pdf
```

Downloads the dataset table and ~467 MB of source PDFs into
`resources_servers/gdp_pdf/data/media/`, renders each document once (LiteParse OCR for text, page
screenshots at 150 DPI) into `resources_servers/gdp_pdf/data/documents/`, then writes
`data/gdp_pdf_benchmark.jsonl`. Already-rendered documents are skipped on re-run.

## Run

Set `judge_base_url` / `judge_api_key` / `judge_model_name` in `env.yaml`, then run the benchmark
with a vision-capable policy model.

## Comparability

Absolute scores are not directly comparable to Surge's published numbers. Surge's harness hands the
raw PDF to native provider document APIs (Claude/Gemini/OpenAI); that has no equivalent for
open-weight models served via vLLM, so we follow Artificial Analysis's approach instead --
LiteParse-extracted text plus rendered page images, with a reactive DPI backoff and page
compositing under tight image-count limits (see
`responses_api_agents/gdp_pdf_agent/README.md`). The judge model also differs, and (unlike Surge's
own scorer) our judge is shown the task prompt alongside each criterion. Artificial Analysis makes
a similar caveat about their own reimplementation not being interchangeable with Surge's.
