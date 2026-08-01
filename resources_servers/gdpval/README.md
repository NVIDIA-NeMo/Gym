# GDPVal resources server

Scores deliverables produced by the Stirrup agent on the GDPVal benchmark.

Two modes via `reward_mode` config:

- `rubric` (default) — LLM judge scores each deliverable against a per-task
  rubric, reward in `[0.0, 1.0]`.
- `comparison` — pairwise judge compares eval deliverable vs. one or more
  reference rollouts (`reference_deliverables_dir`, or `reference_models` for
  multi-reference), reward in `{0.0, 0.5, 1.0}`. `aggregate_metrics` reduces to
  an ELO rating.

Comparison mode also supports **multi-stage adaptive ELO** — a sequence of
stages that judge sampled tasks against an adaptively-chosen reference subset,
enabled with `++multistage.enabled=true`. It is implemented in
`multistage_orchestrator.py` (pure logic in `multistage_elo.py`) and runs through
the standard `gym eval run` pipeline. See the "Run multi-stage adaptive ELO"
section of `benchmarks/gdpval/README.md`.

Canonical entry point is the benchmark at `benchmarks/gdpval/`:

```bash
gym eval prepare --benchmark gdpval
gym eval run \
  --model-type vllm_model \
  --benchmark gdpval \
  --split benchmark
```

See `benchmarks/gdpval/README.md` for the full run recipe.
