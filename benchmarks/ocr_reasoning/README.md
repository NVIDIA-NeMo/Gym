# OCR-Reasoning (via VLMEvalKitMcore)

Judge-scored P00 Omni benchmark over the shared `vlm_eval_kit` server (FEP-1143 wave 2).
mcore dataset: `OCR_Reasoning`. Scoring is two-stage per sample (mcore `OcrR_auxeval`,
`vlmeval/dataset/utils/ocr_reasoning.py:97-123`):

1. Impartial-judge rating of the model's reasoning vs the reference `reasoning` chain,
   parsed via `[[n]]` -> `reason_score = n/10` (always runs);
2. `post_check` prefetch (:68-94); on miss the judge extracts the answer, then
   `post_check` decides the hit.

The reward is the hit (accuracy); the reasoning score is reported separately via
`OCR_Reasoning_RP/<task>` metric keys and a per-sample `reason_score` response field —
mirroring `OcrR_acc`'s dual `acc`/`_RP` reporting (:126-169).

- **Source:** the mcore fork (dual-source pin in `config.yaml`:
  `vlmevalkit_url=matthieul/VLMEvalKitMcore`, `vlmevalkit_commit=c0dfe394…`) — the
  authoritative source for the P00 Omni benchmarks; same isolated per-benchmark pattern
  as OCRBench / MathVista / CharXiv
- **Resources server:** `resources_servers/vlm_eval_kit` (`_score_OCR_Reasoning`)
- **Judge:** `us/azure/openai/gpt-4o-mini` via the inference-api proxy (reference judge
  role); key injected at launch, never committed:
  `++ocr_reasoning_benchmark_resources_server.resources_servers.vlm_eval_kit.judge_api_key=$INFERENCE_API_KEY`
- **Data:** built by `prepare.py` via mcore `build_prompt` (question + language-dependent
  step-by-step instruction embedding the `format` column); carries `reasoning`,
  `question_type`, `answer_type`, `choices`, `answer_option` for the reference scorer;
  `category` = `task`
- **Target:** Nano 3 Omni paper (arXiv 2604.24954) OCR-Reasoning = 54.14 (reasoning-on)
- **num_repeats:** 1

## Run

```bash
ng_prepare_data "+config_paths=[benchmarks/ocr_reasoning/config.yaml]" +output_dirpath=data/ocr_reasoning
ng_run "+config_paths=[benchmarks/ocr_reasoning/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
  "++ocr_reasoning_benchmark_resources_server.resources_servers.vlm_eval_kit.judge_api_key=$INFERENCE_API_KEY"
ng_collect_rollouts +agent_name=ocr_reasoning_benchmark_simple_agent \
    +input_jsonl_fpath=benchmarks/ocr_reasoning/data/ocr_reasoning_benchmark.jsonl \
    +output_jsonl_fpath=results/ocr_reasoning_rollouts.jsonl +num_repeats=1
```

Judge behavior notes: EVERY sample costs at least one judge call (the reasoning rating);
answer extraction adds calls only on prefetch miss. Concurrency bounded by
`judge_max_concurrency`. A judge that never emits a `[[n]]` rating crashes the reference
`OcrR_auxeval` — the scorer catches it and scores 0 (verify never crashes). Without a
configured judge the scorer degrades to prefetch-only exact matching with
`reason_score = 0` — NOT reference-comparable.
