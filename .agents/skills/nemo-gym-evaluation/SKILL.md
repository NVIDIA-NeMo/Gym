---
name: nemo-gym-evaluation
description: >-
  Evaluates models or agents with existing or custom NeMo Gym environments and
  benchmarks, including rollout collection, reward profiling, comparisons, and
  pass@k analysis. Use for "evaluate model", "evaluate agent", "run benchmark",
  "collect rollouts", "reward profiling", "benchmark results", "compare models",
  "compare agents", "analyze results", "pass@k", or "why is reward 0". For crashed,
  incomplete, or infrastructure-failed jobs, use Nemo Gym debugging guidance.
---

# Evaluate Models and Agents

Assume the user knows what capability or systems they want to compare. Make the evaluation contract explicit: model, agent harness, tasks/benchmark, scorer, sampling settings, and metrics.

## Define the comparison

1. State the evaluation question and the one variable being changed.
2. Select an existing benchmark or environment with `gym list benchmarks`, `gym list environments`, and `gym search <query>`.
3. Fix:
   - dataset split and task count;
   - resources server and verifier;
   - agent harness unless it is the comparison variable;
   - model unless it is the comparison variable;
   - temperature, output limit, repeat count, and seed policy.
4. Choose metrics before running. Use reward/pass@k for task success and add cost, latency, token, or tool metrics only when relevant.

Read [Evaluate](../../../fern/versions/latest/pages/evaluation/index.mdx), [Benchmarks](../../../fern/versions/latest/pages/evaluation/benchmarks.mdx), and [Aggregate metrics](../../../fern/versions/latest/pages/evaluation/aggregate-metrics.mdx).

## Run

1. Validate the config with `gym env validate`.
2. Start the required servers or use the evaluation command's serving mode.
3. Run a small smoke subset.
4. Collect full rollouts with `gym eval run`, explicit output paths, and an explicit repeat count.
5. Resume interrupted collection with the supported `--resume` flow; do not concatenate incompatible runs.
6. Aggregate shards when applicable.
7. Run `gym eval profile` against the materialized inputs and rollout artifacts.

Use the current commands and flags in [CLI commands](../../../fern/versions/latest/pages/reference/cli-commands.mdx), not remembered syntax.

## Interpret

- Verify expected task and rollout counts before interpreting scores.
- Report the exact config identity and changed variable.
- Distinguish average reward/pass@1 from pass@k and from average@k.
- Inspect variance and confidence, especially for small or stochastic datasets.
- Compare per-task outcomes, not only aggregate scores.
- Read representative always-pass, sometimes-pass, and always-fail trajectories.
- Treat a reward of zero first as a task-level outcome: inspect model output, tool/state trace, verifier inputs, and verifier reason. If rows are missing, requests failed, or files are partial, switch to failure diagnosis rather than score analysis.
- Check that the verifier is invariant to irrelevant formatting and sensitive to actual task success.

For output semantics, read [Reward profiling CLI reference](../../../fern/versions/latest/pages/reference/cli-commands.mdx#eval-profile). For deeper result diagnosis, read [Diagnose results](../../../fern/versions/latest/pages/evaluation/diagnose-results.mdx).

## Compare agents fairly

When comparing harnesses such as OpenHands, KiloCode, OpenCode, or a native Gym agent:

- keep model, tasks, environment-side tools/state, verifier, and sampling fixed;
- document agent-owned tools and prompts, because they are part of the treatment;
- normalize timeout and resource budgets where possible;
- report unsupported tasks and infrastructure failures separately from reward zero.

## Deliverables

Return:

- the exact runnable command/config;
- task and rollout counts;
- aggregate and per-group metrics;
- uncertainty or repeat information;
- material failure examples;
- a conclusion limited to what the controlled comparison supports.
