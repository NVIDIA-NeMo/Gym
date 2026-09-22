# NMRArena benchmark

Benchmark registration for the `nmrarena` environment: organic structure
elucidation from 1H/13C NMR peak lists over upstream's 105-molecule set, pinned to
`odanchem/NMRArena` commit `8b4ca8a8953185c00f0c4d7fa3c16c23aa616326`.

The environment, its scoring semantics, the deliberate departures from upstream
and the model-free validation are documented in
[`resources_servers/nmrarena/README.md`](../../resources_servers/nmrarena/README.md).
This directory only adds what makes it a *benchmark* rather than an environment.

## What this adds

**`num_repeats: 3`.** Upstream swept the benchmark three times under identical
settings and reports each model as the mean ± 1 SD over the sweeps. The
environment's own config defaults to a single draw, which reports a different
quantity from the published one. With three repeats a run produces the three
per-sweep values that `resources_servers/nmrarena/scripts/summarize_runs.py`
turns into the published summary and a Welch comparison.

Nothing else changes: prompt, decoding (`temperature` 1.0, `max_output_tokens`
24576, no `top_p`, no seed) and scoring come from the environment.

## Preparing the data

Run from the repository root with the server's environment active. The split is
not committed — `benchmarks/.gitignore` excludes it — so prepare it before running:

```bash
python benchmarks/nmrarena/prepare.py
```

→ `Wrote 105 rows` to `data/nmrarena_benchmark.jsonl`. This delegates to the
environment's preparation script, so the split comes from the same fail-closed
path (pinned commit, dataset digest, 21 × 5 class structure, unique ids,
parseable gold). `--limit` takes a positive integer and produces a smoke subset.

## Example artifacts are two-stage

The environment's tracked example artifacts are regenerated in two stages, and
the schemas differ:

1. `python resources_servers/nmrarena/scripts/make_example_data.py` writes
   `resources_servers/nmrarena/data/example.jsonl` — five synthetic rows in the
   preparer's agent-routed schema (`responses_create_params`, `verifier_metadata`,
   `agent_ref`).
2. `gym dataset collate "+config_paths=[resources_servers/nmrarena/configs/nmrarena.yaml]"
   +output_dirpath=resources_servers/nmrarena/data +mode=example_validation`
   rewrites the routing key (`agent_ref` → `task_source`) and writes
   `example_metrics.json`.

The tracked `example.jsonl` carries the **collated** schema (`task_source`); a
regeneration that stops after stage 1 yields a different file. The benchmark
split written by `prepare.py` carries the preparer's schema and is collated at
run time.

## Running

```bash
gym eval run \
    --config benchmarks/nmrarena/config.yaml \
    --model-type openai_model \
    --agent nmrarena_benchmark_agent \
    --split benchmark \
    --output <output>.jsonl
```

A full run is 105 molecules × 3 repeats = 315 rollouts. The 24K output budget
covers reasoning as well as the answer, and some gateways report an exhausted
budget as `status: completed` with empty text; read `mean/output_tokens` and
`mean/response_incomplete` together.

## Licensing

Code: Apache 2.0. The benchmark split is declared `TBD`: the repository and its
dataset file are MIT ("Copyright (c) 2026 OdanChem"), but the spectra are curated
from a third-party database whose rights are not determined here. No benchmark
data is committed.
