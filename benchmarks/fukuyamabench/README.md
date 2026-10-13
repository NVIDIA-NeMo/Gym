# FukuyamaBench benchmark

Benchmark registration for the `fukuyamabench` environment: step-by-step organic
reaction-mechanism prediction over upstream's sets B and C (241 cases).

The environment, its scoring semantics, the three deliberate divergences from
upstream's scorer, and the model-free validation are documented in
[`resources_servers/fukuyamabench/README.md`](../../resources_servers/fukuyamabench/README.md).
This directory only adds what makes it a *benchmark* rather than an environment.

## What this adds

**`num_repeats: 8`.** Upstream reports pass@k for k ∈ {1, 3, 5, 8}, which needs at
least eight draws per case. The environment's own config defaults to a single
draw, which reports a different quantity from the published one. With eight
draws the aggregate emits the full `pass@1` … `pass@8` family.

**Per-tier metrics come from the server** and are prefixed by `case_set`, so a
run over both tiers yields `B/pass@1/accuracy` and `C/pass@1/accuracy` — the two
columns upstream publishes — rather than one pooled number. The pooled
`mean/reward` is deliberately excluded from the headline set; see the server
README.

## Preparing the data

Run from the repository root. The split is not committed — `benchmarks/.gitignore`
excludes it — so prepare it before running:

```bash
python benchmarks/fukuyamabench/prepare.py
```

→ `Wrote 241 rows` (B=131, C=110) to `data/fukuyamabench_benchmark.jsonl`.

This delegates to the environment's own preparation script rather than
duplicating it, so the benchmark split comes from the same tested path — including
the manifest check that refuses to write a short corpus. `--limit` takes a
positive integer and produces a smoke subset.

## Running

```bash
gym eval run \
    --config benchmarks/fukuyamabench/config.yaml \
    --model-type openai_model \
    --agent fukuyamabench_benchmark_agent \
    --split benchmark \
    --output <output>.jsonl
```

A full run is 241 cases × 8 draws = 1,928 rollouts. Note that on reasoning
models the output-token budget covers reasoning as well as the visible answer;
too small a budget returns nothing extractable and scores zero, so check the
response status rather than assuming truncation surfaces as an error.

## Licensing

The benchmark split is declared `TBD`, not Apache-2.0. Upstream declares its
dataset Apache-2.0, but the mechanisms are transcribed from a commercial
textbook and no permission is stated. See the server README's Licensing section
— this is an open rights gate, and no benchmark data is committed here.
