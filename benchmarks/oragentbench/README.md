# ORAgentBench benchmark

Benchmark registration for the `oragentbench` environment: 107 executable operations-research
tasks (32 easy / 41 medium / 34 hard) solved end to end by an agent inside a per-task
container and scored by upstream's own validators. The environment, its reward (the paper's
pass predicate), the Harbor multi-step handling, the deliberate divergences from upstream and
the model-free validation are documented in
[`resources_servers/oragentbench/README.md`](../../resources_servers/oragentbench/README.md).
This directory only adds what makes it a *benchmark* rather than an environment.

## What this adds

**`num_repeats: 1`.** Upstream's main-table protocol is "one complete run per model-agent
configuration" (pass@1, single run). The one row upstream averaged over three runs is
DeepSeek V4 Pro; set `num_repeats: 3` to match it.

**Per-stratum metrics come from the server**: `pass_rate/easy`, `pass_rate/medium`,
`pass_rate/hard` (with feasibility rates, mean quality and counts) are promoted to
`key_metrics` alongside the pooled `mean/reward`, because published baselines differ by more
than 2x across strata. The agent proxies `/aggregate_metrics` to the server so this selection
is what a benchmark run reports.

## Preparing the data

Run from the repository root. The rows are not committed (`benchmarks/.gitignore` excludes
`data/`) and the task content is third-party; prepare before running:

```bash
python benchmarks/oragentbench/prepare.py --build-images
```

-> `Loaded 107 tasks at c9eb952435a4352f33daa2a35efe0f8c76d31b28` and
`Wrote 107 rows to benchmarks/oragentbench/data/oragentbench_benchmark.jsonl`. This delegates
to the environment's preparation script, so the rows come from the same fail-closed path
(exactly 107 tasks with the published strata, or nothing is written). `--limit` writes a
sorted prefix for a smoke subset; `--build-images` builds the base image and the 107 task
images the rows reference (needs a Docker daemon; about 15 minutes cold).

## Running

```bash
ORAGENTBENCH_POLICY_MODEL=<served model id> gym eval run \
    --config benchmarks/oragentbench/config.yaml \
    --agent oragentbench_benchmark_agent \
    --split benchmark \
    --concurrency 6 \
    --output <output>.jsonl
```

A full run is 107 tasks x `num_repeats` rollouts, each up to 45 minutes of agent time
(single-step) or 45 + 3 x 20 minutes (multi-step), in a 4-CPU / 8 GiB container with no
network. Six concurrent containers took 3 h 26 min for one GPT-5.4 run. Only
`reasoning.effort=high` is sent as a decoding parameter; upstream publishes nothing else.

## Example artifacts

The tracked example rows live with the environment, not here:
`resources_servers/oragentbench/data/example.jsonl` (five synthetic rows against the fixture
task under `resources_servers/oragentbench/tests/fixtures/toy_assignment`). Regeneration is
two stages, and the schemas differ: `scripts/make_example_data.py` writes preparer-schema rows
(`agent_ref`), then `gym dataset collate ... +mode=example_validation` rewrites the file in
place into the collated schema (`task_source`) and writes `example_metrics.json`. The tracked
`example.jsonl` carries the collated schema. See the environment README for the commands.

## Licensing

The benchmark rows are declared `TBD`. Upstream's code is MIT and its README declares the
data CC BY 4.0, but 73% of tasks derive from published OR papers and IndustryOR seeds whose
rights upstream asserts without itemising; a licence declaration is not a rights determination.
See the environment README's Licensing section.
