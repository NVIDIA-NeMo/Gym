---
name: add-benchmark
description: >
  Add or review a NeMo Gym benchmark, evaluation, training environment, resources
  server, agent loop, or external harness integration. Covers choosing the integration
  topology, defining dataset/prompt/verifier/metric contracts, preparing data, composing
  config, writing behavior-focused tests, running real rollouts, and checking parity with
  an upstream evaluator.
---

# Add or Review a NeMo Gym Benchmark

## Start With the Contract

Before changing code, inspect the upstream benchmark specification, the closest Gym
implementations, and every runtime consumer of the prepared row. Do not infer a row
schema from a neighboring benchmark alone.

Write down this mapping for the integration:

| Concern | Producer | Consumer | Invariant |
| --- | --- | --- | --- |
| task identity | source / `prepare.py` | rollout grouping and metrics | stable and unique |
| prompt fields | `prepare.py` | `prompt_config` or custom agent | every placeholder exists |
| verifier fields | `prepare.py` | resources-server request model | names and types match exactly |
| selector fields | source / `prepare.py` | verifier and scorer | identify the same tests/groups |
| reward | verifier | training and profiling | range and success meaning are explicit |
| aggregate metrics | verifier `compute_metrics()` | benchmark report | reproduce the official aggregation |

For grouped, weighted, or partially overlapping tests, also specify the evaluation
unit: one problem, one subtask, one test group, or one full episode. A selector is not
just metadata if it changes which tests run or how scores are pooled.

## Choose the Integration Topology

Use the smallest topology that preserves the benchmark's behavior:

1. **Benchmark over an existing verifier** — add `benchmarks/<name>/prepare.py` and
   `config.yaml`, then inherit an existing resources server and agent. This is the
   common path for math, MCQ, translation, and code-generation evals.
2. **Custom Gym verifier** — add a resources server when existing request, reward, or
   metric contracts cannot express the benchmark.
3. **Custom Gym agent loop** — add an agent when prompting, tool use, correction turns,
   or state transitions cannot be handled by an existing agent.
4. **External agent loop or rollout driver** — adapt an upstream harness when its
   orchestration is part of the benchmark. Establish upstream results before adapting it.
5. **Eval suite** — compose several benchmark configs with `config_paths`. A suite is
   not a single benchmark: benchmark discovery requires exactly one locally declared
   `type: benchmark` dataset per benchmark config.

Read [references/patterns.md](references/patterns.md) for repository examples and
config shapes before selecting a topology.

## Scaffold the Right Artifact

Use the manifest-backed scaffold when the benchmark fits its publication contract:

```bash
# New verifier with the default custom-gym-verifier profile
gym env init --benchmark my_benchmark

# Reuse an existing verifier that exports VERIFIER_FIXTURE
gym env init --benchmark my_benchmark \
  --reuse-verifier shared_verifier \
  --reward-range 0 1 \
  --higher-is-better

# Other extension points
gym env init --benchmark my_benchmark --profile custom-gym-agent-loop
gym env init --benchmark my_benchmark --profile external-agent-loop
gym env init --benchmark my_benchmark --profile external-rollout-driver
```

Use `--environment` instead of `--benchmark` for a training environment. Use
`gym env init --resources-server ...` only when the requested artifact is a standalone
resources server rather than a complete benchmark/environment.

The generated `config.yaml` is runtime-authoritative. Keep `manifest.yaml` aligned with
it and replace all scaffold placeholders. Existing config-only benchmarks do not need
an unrelated migration to a manifest. `--reuse-verifier` requires the selected server
to export `VERIFIER_FIXTURE`. A manifest-backed benchmark also currently requires a
standard prompt config; follow an existing config-only pattern when a self-contained
custom-agent dataset must use `prompt_config: null` rather than inventing a dummy prompt.

## Implement Data and Prompt Preparation

A benchmark dataset declaration has this core contract:

```yaml
- name: my_benchmark
  type: benchmark
  jsonl_fpath: benchmarks/my_benchmark/data/my_benchmark.jsonl
  prepare_script: benchmarks/my_benchmark/prepare.py
  prompt_config: benchmarks/my_benchmark/prompts/default.yaml  # or null
  num_repeats: 1
```

`prepare.py` must:

- expose a synchronous `prepare()` callable that works with no arguments by default;
- be importable from the repository-root environment used by `gym eval prepare`;
- return a `Path` exactly matching the configured `jsonl_fpath`;
- create deterministic rows from a pinned or otherwise auditable source;
- validate source invariants that could otherwise produce a runnable but incorrectly
  scored dataset, such as split, task count, IDs, labels, rubric weights, and test groups;
- avoid replacing a known-good output with a partial result when preparation fails;
- keep generated benchmark data ignored unless the repository intentionally tracks it.

If preparation also emits NeMo Skills or another external format, validate that
consumer's naming, registry, and discovery path with a clean invocation. Producing a
directory is not sufficient if the downstream tool cannot resolve its public name.

Choose exactly one prompt representation:

- **Raw semantic fields plus `prompt_config`**: the template fills top-level row fields
  at rollout time. Do not pre-populate `responses_create_params.input`.
- **Materialized `responses_create_params.input` plus `prompt_config: null`**: use this
  for multimodal content, upstream-owned tool schemas/prompts, or custom agents that
  require a self-contained request.

`verifier_metadata` is not universally required. The real contract is the selected
agent/resources-server request model: some servers consume top-level fields, some use
`verifier_metadata`, and custom agents may consume a self-contained request. Preserve
provenance separately from fields that affect scoring.

For a training environment, declare the needed `train`, `validation`, and small
`example` datasets with auditable `source` and license metadata. Check train/eval split
leakage and reward density. Use `gym dataset collate` for those dataset types;
`type: benchmark` preparation goes through `gym eval prepare` instead.

## Implement Verification and Metrics

Trace one prepared row through prompt materialization, the agent request, the resources
server request model, `verify()`, and `compute_metrics()`. Validate the materialized row
against the real request models where practical.

- Define malformed-output behavior; bad model output should score or return a useful
  error response rather than crash the service.
- Declare the actual reward range. Do not assume every verifier is binary.
- Keep per-rollout reward separate from benchmark-level aggregation.
- For subtasks/groups, prove that a row runs exactly the intended tests and can receive
  no more than its declared maximum. Synthetic selector names require an explicit
  verifier mapping; otherwise use identifiers present in verifier metadata.
- Test overlapping tests, missing groups, partial credit, duplicate outputs, ties, and
  best-of/repeat pooling when those cases exist.
- When adapting an upstream evaluator, compare both per-example verdicts and aggregate
  metrics. Similar headline scores can hide different examples being accepted.

Follow the repository `AGENTS.md` for async HTTP, cookie propagation, concurrency,
subprocess isolation, external-tool setup, licensing, and source-file requirements.

## Test in Layers

Tests should pin behavior at each boundary:

1. **Preparation** — import the module, replace network/data sources with fixtures,
   assert the exact output path and semantic row contents, and cover source drift and
   failed/partial writes.
2. **Config and discovery** — resolve the config, assert the selected agent/resources
   server, dataset, prompt mode, repeat count, and any verifier asset paths.
3. **Prompt and request** — materialize at least one row and validate it against the
   agent/resources-server request contract.
4. **Verifier** — cover full reward, zero reward, malformed output, exceptions/timeouts,
   and partial/grouped cases that affect reward.
5. **Metrics** — test grouping and aggregation independently from `verify()`.
6. **End to end** — run a known-good/reference solution and a known-bad output through
   the real execution path. For agent/environment changes, collect real model rollouts
   and inspect agent and verifier behavior; green unit tests are not sufficient.

Network access and large downloads do not belong in unit tests. Use small fixtures that
preserve the scoring-relevant structure.

For a review task, read [references/review-checklist.md](references/review-checklist.md)
and report concrete findings in severity order, with tight file/line references.

## Validate the Integration

Use the target checkout's CLI help as the source of truth. A typical benchmark sequence is:

```bash
python -m pytest benchmarks/my_benchmark/tests -q
python -m pytest resources_servers/my_verifier/tests -q

gym list benchmarks my_benchmark
gym eval prepare --benchmark my_benchmark
gym env validate --benchmark my_benchmark
gym eval run --benchmark my_benchmark --model-type <model_type>
```

For a manifest-backed entry, also run:

```bash
gym env validate my_benchmark --kind benchmark
gym env test my_benchmark --kind benchmark
gym env publish my_benchmark --kind benchmark
```

Profile repeated rollouts with the `nemo-gym-reward-profiling` skill. Inspect individual
rollouts, error rates, reward distribution, per-category/group metrics, and variance—not
only the headline mean. Compare against official or previously reproduced results when
an upstream benchmark exists.

Before handoff, run focused pre-commit checks on changed files, then the broader checks
appropriate to the change. Commits require DCO sign-off with `git commit -s`;
cryptographic `-S` signing is optional.

## Reference Loading

- [references/patterns.md](references/patterns.md) — choose a topology and borrow a
  current repository pattern.
- [references/review-checklist.md](references/review-checklist.md) — review data,
  template, verifier, scoring, metrics, tests, and operational readiness.
- Use `nemo-gym-reward-profiling` for rollout/profile commands and artifact semantics.
- Use `nemo-gym-debugging` when preparation, serving, rollout collection, or judging fails.
