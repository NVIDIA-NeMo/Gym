# VeriSoftBench benchmark

Benchmark registration for the `verisoftbench` environment: 500 Lean 4 proof
obligations from 23 repositories, kernel-verified inside each repository's own
toolchain, with upstream's error-feedback repair loop (three rounds per candidate).

The environment, the verifier (upstream's own harness code driven from a pinned
checkout), the repair loop, the model-free validation and the licensing are
documented in
[`resources_servers/verisoftbench/README.md`](../../resources_servers/verisoftbench/README.md).
This directory only adds what makes it a *benchmark* rather than an environment.

## What this adds

**`num_repeats: 8`.** Upstream's estimand is "a task is solved if any of k = 8
candidates or their repairs verify". The environment's own config defaults to a
single draw, which reports a different quantity from the published figures. With
eight draws the aggregate's `any_of_k/solved` is that estimand; Gym's `pass@8`
coincides with it when every task has exactly eight rollouts.

**Curated context only.** Rows are upstream's `filtered_context` prompt; the
full-repository condition is not registered (the environment README explains what it
would take).

## Preparing the data

Run from the repository root with the server's environment active. The split is not
committed; prepare it before running (the pinned upstream checkout is cloned on
first use, no Lean container is needed for preparation):

```bash
python benchmarks/verisoftbench/prepare.py
```

→ `Wrote 500 rows` to `data/verisoftbench_benchmark.jsonl`. Fails closed if the
dataset digest, row count or task ids differ from the pin. `--limit N` prepares a
deliberate subset.

## Running

The Lean container must be running (see the server README's Setup).

```bash
gym eval run \
    --config benchmarks/verisoftbench/config.yaml \
    --model-type openai_model \
    --agent verisoftbench_benchmark_agent \
    --split benchmark \
    --temperature 0.7 --max-output-tokens 8192 \
    --output <output>.jsonl
```

A full run is 500 tasks × 8 candidates, each with up to three repair rounds: up to
16,000 model calls and as many Lean compilations. The paper publishes no decoding
parameters; the values above are upstream's example-config values and should be
declared with any result.

## Licensing

Harness and dataset: MIT. The 23 source Lean repositories retain their own licences
and their content is downloaded at run time, not committed; see the server README.
