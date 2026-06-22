# GDPVal benchmark

[GDPVal](https://huggingface.co/datasets/openai/gdpval) — 220 professional
knowledge-work tasks scored by an LLM judge against per-task rubrics. This
benchmark wires the Stirrup-based agent (`responses_api_agents/stirrup_agent`)
to the GDPVal resources server (`resources_servers/gdpval`).

## Prepare data

Downloads `openai/gdpval` from HuggingFace and writes
`data/gdpval_benchmark.jsonl`:

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/gdpval/config.yaml]"
```

## Run rubric mode (default)

Each deliverable is scored 0–1 against the task rubric.

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/gdpval/config.yaml"
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/gdpval_rubric.jsonl \
    ++split=benchmark \
    ++policy_base_url=<vllm_base_url> \
    ++policy_api_key=<vllm_api_key> \
    ++policy_model_name=<served_model_name>
```

Required environment variables for the judge:

- `JUDGE_API_KEY` — sk- key for the judge inference API (nvapi- keys 401 on
  multimodal payloads)
- `JUDGE_BASE_URL` — defaults to NVIDIA's internal inference API
- `JUDGE_MODEL_NAME` — defaults to `gcp/google/gemini-3.1-pro-preview`
- `HF_TOKEN` — for downloading reference files (avoids HF anonymous rate limits)

## Run comparison mode (pairwise ELO vs. a reference model)

Each deliverable is judged against a reference model's deliverable for the
same `task_id`; aggregate metrics include ELO relative to a configurable
anchor (default 1000).

```bash
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/gdpval_compare.jsonl \
    ++split=benchmark \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.reference_deliverables_dir=/path/to/reference/output
```

The reference directory must be laid out as
`<reference_deliverables_dir>/task_<task_id>/` with `finish_params.json` and
the deliverable files (the same layout the Stirrup agent persists).

## Aggregate metrics

After `ng_e2e_collect_rollouts` returns, the resources server's
`/aggregate_metrics` endpoint emits headline scores in
`results/<output>_metrics.json`:

- Rubric mode: `mean/reward` (pass@1 equivalent)
- Comparison mode: `comparison/wins`, `comparison/losses`, `comparison/ties`,
  `comparison/win_rate`, `comparison/eval_elo`, `comparison/normalized_elo`
