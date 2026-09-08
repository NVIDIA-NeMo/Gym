# GDP.pdf

NeMo Gym implementation of [Surge AI GDP.pdf](https://huggingface.co/datasets/surgeai/GDP.pdf), following [Artificial Analysis intelligence-benchmarking methodology v4.3](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf).

GDP.pdf contains 100 test-only professional tasks over 4,592 PDF pages in ten domains, with 1,275 independently judged rubric criteria. The source is pinned to Hugging Face revision `73e94c87235e0477f8a65996086acd3f47c98d2e`.

## Protocol

- LiteParse extracts every page with OCR enabled where required.
- Every policy receives the complete extracted text of every page.
- Vision policies receive ordered page images at 150 DPI by default.
- Endpoint-limit fallbacks are explicit agent settings: reduce to no lower than 72 DPI, compose 2 then 4 labeled pages per image, then limit image coverage to leading pages while retaining all extracted text.
- The policy gets one turn with no tools or browsing.
- GPT-5.6 Luna with medium reasoning judges every rubric criterion independently and never receives the PDF or contestant identity.
- All-pass is the headline metric; Mean Pass is the task-macro criterion pass rate.
- This implementation reports `pass@1[avg-of-k]`, not best-of-k `pass@k`.

## Prepare

```bash
uv run --frozen --with liteparse==2.14.4 gym eval prepare --benchmark gdp_pdf
```

The first preparation downloads the pinned public snapshot and parses/renders 4,592 pages. Source PDFs, page images, manifests, and generated JSONL all stay under `benchmarks/gdp_pdf/data/` and are gitignored. Repeated preparation reuses cache manifests only after validating their source hash, parser version, DPI, page count, and page-image presence.

## Evaluate

The benchmark defaults to AA's five rollouts per task:

```bash
uv run gym eval run \
  --benchmark gdp_pdf \
  --model-type openai_model \
  --agent gdp_pdf_benchmark_agent \
  --split benchmark \
  --output results/gdp_pdf.jsonl
```

For a one-rollout smoke test or routine RL iteration, override only:

```bash
--num-repeats 1
```

Set `judge_base_url`, `judge_api_key`, and, if necessary, `judge_model_name` for the GPT-5.6 Luna endpoint. Policy model configuration uses the normal Gym `--model-type` and model arguments.

For provider limits, override the `gdp_pdf_agent` instance fields (`image_dpi`, `pages_per_image`, `max_images`, or `image_format`). Any non-default delivery profile must be retained with the resolved run config because it affects comparability.

## RL use

The same agent and verifier can score compatible, independently created PDF tasks for RL. Scalar `reward` is Mean Pass and `reward_components` includes both Mean Pass and All-pass.
