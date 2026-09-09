# GDP.pdf

NeMo Gym implementation of [Surge AI GDP.pdf](https://huggingface.co/datasets/surgeai/GDP.pdf), following [Artificial Analysis intelligence-benchmarking methodology v4.3](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf).

GDP.pdf contains 100 test-only professional tasks over 4,592 PDF pages in ten domains, with 1,275 independently judged rubric criteria. The source is pinned to Hugging Face revision `73e94c87235e0477f8a65996086acd3f47c98d2e`.

## Protocol

- LiteParse extracts every page with OCR enabled where required.
- Every policy receives the complete extracted text of every page.
- Vision policies receive ordered page images at 150 DPI by default.
- Context/payload rejections automatically reduce image DPI, down to 72. Endpoint image-count caps select 2- or 4-page labeled composites and, only if necessary, leading-page image coverage; all extracted text is retained.
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

Set `--max-output-tokens` and sampling parameters explicitly for the evaluated model. AA uses 16,384 output tokens for non-reasoning models (subject to model limits), and the model creator's maximum output allowance for reasoning models. Temperature is 0 for non-reasoning and 0.6 for reasoning unless the model creator recommends otherwise. The output allowance must fit alongside the full input in the configured context window; it is preserved across DPI retries.

For a local vLLM endpoint, use `--model-type inference_provider`, set `policy_base_url`, `policy_api_key`, and `policy_model_name`, and enable `++policy_model.responses_api_models.inference_provider.uses_reasoning_parser=true`. This existing Gym adapter preserves input-limit errors. The `vllm_model` adapter used for token-ID training converts some context errors into empty responses; automatic overflow fitting is not supported through that path. An incomplete response without token usage fails visibly rather than triggering a potentially invalid second generation.

The benchmark fixes the denominator to 100 tasks and the requested `--num-repeats` (five by default), including missing attempts as zero. `rollouts/scored` and `rollouts/expected` expose completion separately. For partial runs, override `expected_task_count` and `expected_domain_task_counts` on `gdp_pdf_benchmark_resources_server.resources_servers.gdp_pdf` to the selected subset; otherwise the score is a lower bound against the full benchmark. Preserve and inspect Gym's failures sidecar before reporting any score.

Delivery starts at 150 DPI for every task and adapts to explicit endpoint rejections. Optionally set the agent's `max_images` to a known endpoint image-count cap. Results retain the selected delivery profile and rejection history. See [agent delivery details](../../responses_api_agents/gdp_pdf_agent/README.md) for the retry schedule and terminal-failure behavior.

## RL use

The same agent and verifier can score compatible, independently created PDF tasks for RL. Scalar `reward` is Mean Pass and `reward_components` includes both Mean Pass and All-pass.
