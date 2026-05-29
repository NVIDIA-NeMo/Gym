---
name: nemo-gym-debugging
license: Apache-2.0
description: >-
  Debug a Nemo Gym run or reward-profiling job: rollout collection failures,
  empty or partial JSONL outputs, stale materialized inputs, verifier and schema
  errors, Ray or Slurm issues, vLLM readiness, judge failures, tool and sandbox
  failures, cache problems, and throughput bottlenecks. Not for adding new
  benchmarks (use add-benchmark) or routine profiling setup (use
  nemo-gym-reward-profiling).
metadata:
  author: NVIDIA <nemo-gym@nvidia.com>
  tags:
    - debugging
    - rollouts
    - reward-profiling
    - troubleshooting
    - observability
---

# Nemo Gym Debugging

## Purpose

Diagnose and resolve failures in a Nemo Gym run or reward-profiling job by
classifying the failing layer (infra, model serving, config, data/schema,
verifier/runtime, cache/resume, or throughput) before changing code or data.

## Invocation Check

Use this skill when something failed or looks suspicious in a Nemo Gym run. If the task is adding a new benchmark or environment, use the `add-benchmark` skill; if it is changing profiling behavior, use the `nemo-gym-reward-profiling` skill.

Debug by classification, not by guessing. The first goal is to decide whether the issue is:

- infra: Slurm, Ray, container, filesystem, network, ports
- model serving: vLLM startup/readiness/throughput
- config: wrong config bundle, missing agent, wrong extra args
- data/schema: JSONL fields do not match verifier/resource server expectations
- verifier/runtime: resource server exception or malformed verify response
- cache/resume: stale materialized inputs or partial rollout output
- throughput/resources: concurrency too high, judge bottleneck, tool/sandbox latency

## Instructions

Work through these checks in order:

1. Check Slurm/Ray job state and logs.
2. Check vLLM readiness and `/models` availability.
3. Check Gym server readiness: all expected servers started.
4. Check tool routing if the env uses tools; check sandbox readiness only if a sandbox is configured.
5. Check materialized inputs and source data timestamps.
6. Check rollout output and profiling/metrics output counts.
7. Inspect the first real verifier exception, not shutdown noise.
8. Compare failing row schema against the resource server request model.

## High-Value Suspects

- If data changed and `resume_from_cache` was enabled, stale materialized inputs are a first-class suspect.
- If rollout output has a few rows and profiling is empty, inspect verifier errors and partial-output cache.
- If all servers are ready but verifier returns 422/500, inspect request body schema before debugging infra.
- If tool envs hang or partially work, check tool ownership/loading before changing model settings; check sandbox readiness only when a sandbox is actually part of the env.
- If tool-call rows fail before generation with vLLM grammar/schema errors, read `references/vllm-tool-call-schema-checks.md` and run a static tool-schema check before changing Gym wrappers.
- If logs only show nested "inner server" 500s without the real provider/verifier body, first enable existing request-boundary visibility with `++global_aiohttp_client_request_debug=True`. Read `references/request-boundary-visibility.md` before changing code.

## Reference Loading

- Read `references/error-profiles.md` to classify the failing layer before changing code or data.
- Read `references/diagnostic-snippets.md` when you need copy-paste commands to inspect logs, output counts, materialized inputs, rollout JSONL shape, server readiness, or reward summaries without mutating run state.
- Read `references/vllm-tool-call-schema-checks.md` when a tool-call dataset may be rejected by vLLM/Outlines grammar compilation before any meaningful generation happens.
- Read `references/request-boundary-visibility.md` when `/run` 500s hide row identity or nested Gym 500s hide the inner model/verifier/provider error. It covers the existing Gym debug flag, shipped request-boundary markers, empty provider bodies, and vLLM provider-side escalation.

## Examples

Empty reward-profiling output with a populated rollouts file: confirm the
`rollouts.jsonl` row count, then inspect the first real verifier exception rather
than shutdown noise. If the data changed and `resume_from_cache` was enabled,
suspect stale materialized inputs and compare source-data and materialized-input
timestamps before rerunning.

Tool-call rows failing before generation: run the static tool-schema check in
`references/vllm-tool-call-schema-checks.md` before modifying Gym wrappers, since
vLLM and Outlines reject malformed tool schemas during grammar compilation, ahead
of any meaningful generation.

## Communication Pattern

When reporting back, state:

- observed symptom
- failing layer
- evidence from logs/files
- likely cause
- next concrete action
