---
name: nemo-gym-integrate-benchmark
description: >-
  Integrates an existing third-party benchmark into NeMo Gym while preserving its
  tasks, harness semantics, scoring, dependencies, and published results. Use when
  the user says "integrate benchmark", "wrap benchmark", "port benchmark", "add
  existing benchmark", "integrate X into Gym", "wrap X library", or "add X benchmark
  to Gym".
---

# Integrate an Existing Benchmark

Assume the user has selected a benchmark and needs the Gym-specific integration workflow. Preserve benchmark fidelity; do not redesign its tasks or scoring unless the user explicitly asks.

## Establish the upstream contract

Before editing Gym:

1. Record the exact upstream version, task split, prompt/harness settings, scorer, dependencies, sampling settings, and reported metric.
2. Run the original implementation and reproduce a public or maintainer-approved baseline.
3. Save the commands, model identity, result, variance, and known deviations.

Do not use a Gym result to debug an upstream reproduction that has never worked.

Read [Benchmarks](../../../fern/versions/latest/pages/evaluation/benchmarks.mdx) and [Add a benchmark](../../../fern/versions/latest/pages/contribute/environments/adding-a-benchmark.mdx).

## Choose the integration seam

Inspect where upstream owns orchestration:

- **Tasks + scorer, but no required harness:** implement or reuse a resources server for scoring/tools/state, pair it with a Gym agent, and add a fixed config under `benchmarks/<name>/`.
- **Required harness or end-to-end runner:** wrap the upstream runner in a Responses API agent server `/run` endpoint. Convert Gym input into upstream input and upstream trajectory/reward into `BaseVerifyResponse`.
- **Reusable environment framework:** use the seam-selection guidance in [Integrate external libraries](../../../fern/versions/latest/pages/environment-tutorials/integrate-external-environments.mdx); do not assume every framework belongs on the agent side.

Keep environment behavior in a resources server when it can be shared across harnesses. Keep an inseparable upstream harness/environment/scorer pipeline together at the agent-server seam.

## Implement the adapter

1. Add a pinned or otherwise reproducible dependency to the server's `requirements.txt`. Include non-Python setup when required.
2. Preserve upstream task IDs and privileged scorer metadata during data conversion.
3. Preserve prompts, stop conditions, tool semantics, timeouts, and reward aggregation.
4. Keep `/run` async. Offload blocking upstream work explicitly and bound it; do not block the event loop.
5. Route model calls through the configured Gym model server and preserve training fields such as prompt token IDs, generation token IDs, and generation log probabilities.
6. Return upstream tool/model errors as diagnosable rollout results where possible; infrastructure failures should remain visible as failures.
7. Add a benchmark config with a `type: benchmark` dataset, reproducible `prepare_script`, documented `num_repeats`, agent selection, and resources-server/model references as applicable.
8. Document upstream license, data license, version, patches, unsupported modes, and reproduction commands.

Use [Configuration](../../../fern/versions/latest/pages/reference/configuration.mdx) for schemas and [Async patterns and performance](../../../fern/versions/latest/pages/infrastructure/async-patterns-and-performance.mdx) for client and concurrency rules.

## Verify fidelity

- Unit-test schema conversion in both directions.
- Test scorer parity on known passing, failing, malformed, and boundary cases.
- Run a small end-to-end smoke test before full reproduction.
- Re-run the same model/settings used for the upstream baseline.
- Compare task count, missing/error rate, aggregate score, per-category scores, and variance.
- Investigate differences at the trajectory/task level. Do not accept matching aggregate scores that hide different task outcomes.
- Record unavoidable deviations and their expected metric impact.

The integration is complete only when another contributor can install dependencies, prepare the fixed split, run the benchmark, and explain any delta from upstream.
