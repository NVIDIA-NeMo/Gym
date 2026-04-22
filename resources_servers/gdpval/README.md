# GDPVal resources server

Scores deliverables produced by the Stirrup agent on the GDPVal benchmark.

Two modes via `reward_mode` config:

- `rubric` (default) — LLM judge scores each deliverable against a per-task
  rubric, reward in `[0.0, 1.0]`.
- `comparison` — pairwise judge compares eval deliverable vs. a reference
  rollout (`reference_deliverables_dir` must be set), reward in
  `{0.0, 0.5, 1.0}`. `aggregate_metrics` reduces to an ELO rating.

Canonical entry point is the benchmark at `benchmarks/gdpval/`:

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/gdpval/config.yaml]"
ng_e2e_collect_rollouts \
  "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,benchmarks/gdpval/config.yaml]" \
  ++split=benchmark
```

See `benchmarks/gdpval/README.md` for the full run recipe.
