---
name: nemo-gym-reward-profiling
license: Apache-2.0
description: >-
  Get started with Nemo Gym reward profiling: ng_run, ng_collect_rollouts, and
  ng_reward_profile. For failed jobs, prefer nemo-gym-debugging.
metadata:
  author: NVIDIA <nemo-gym@nvidia.com>
  tags:
    - reward-profiling
    - rollouts
    - evaluation
    - metrics
---

# Nemo Gym Reward Profiling

## Purpose

Run and understand Nemo Gym reward profiling: start servers with `ng_run`,
collect rollout artifacts with `ng_collect_rollouts`, and produce profiling
output with `ng_reward_profile`, then inspect the resulting rows and metrics.

## Prerequisites

- NeMo Gym installed with the `ng_run`, `ng_collect_rollouts`, and `ng_reward_profile` CLIs.
- An environment config bundle and an input JSONL dataset.
- A reachable model server (an OpenAI-compatible endpoint or a local vLLM model server).
- Enough disk for rollout and materialized-input artifacts.

## Invocation Check

Use this skill when the user wants to run, understand, or lightly modify Nemo Gym reward profiling. Keep the answer oriented around the normal workflow:

`ng_run` starts model/resource servers, `ng_collect_rollouts` writes rollout artifacts, and `ng_reward_profile` generates profiling output from those artifacts.

If the user is primarily debugging a failed job or stack trace, use the `nemo-gym-debugging` skill first.

Do not activate this skill for these adjacent tasks:

- Debugging a failed or crashed run (Ray/vLLM stack traces, empty output). Use `nemo-gym-debugging`.
- Adding or scaffolding a new benchmark, evaluation, or training environment. Use `add-benchmark`.

## Instructions

1. Identify the environment config paths and input JSONL.
2. Start Gym servers with `ng_run`.
3. Collect rollouts with `ng_collect_rollouts`; this writes `rollouts.jsonl` and `*_materialized_inputs.jsonl`.
4. Run `ng_reward_profile` on the materialized inputs and rollout JSONL to generate `*_reward_profiling.jsonl`.
5. Inspect line counts and profile rows.

Repeated rollouts are the main profiling lever. `num_repeats=1` is valid, but per-task averages and variance are only meaningful with multiple rollouts per task.

## Core Concepts

- `*_materialized_inputs.jsonl`: expanded collection inputs after repeat expansion, agent defaults, and task/rollout id assignment.
- `rollouts.jsonl`: one completed rollout/result per materialized input row.
- `*_reward_profiling.jsonl`: one summarized profile row per original task with at least one completed rollout.
- `_ng_task_index`: original task/sample id.
- `_ng_rollout_index`: repeated rollout id for that task.
- `rollout_infos`: compact per-rollout info inside each task profile row, including reward, token usage, and numeric rollout metrics when available.

Keep reward-to-length or reward-to-token analysis keyed by both `_ng_task_index` and `_ng_rollout_index`.

## Reference Loading

Load references only when the user needs that detail:

- Read `references/quick-start.md` for a generic command template and the minimal run sequence.
- Read `references/output-format.md` to explain materialized inputs, rollout JSONL, reward profile rows, `rollout_infos`, and partial profiling.

## Practical Defaults

- Treat `ng_reward_profile` as the reward profiling step; rollout collection does not write reward profile files.
- Run strict profiling by default. If rollout collection stopped early, use `++allow_partial_rollouts=True` to profile completed rollouts and drop original input rows with no completed rollout.
- Trust the target checkout's CLI help and `nemo_gym/reward_profile.py` over memory if flags differ.

## Examples

Profiling a single config: run `ng_run` for the environment, collect rollouts
with `+num_repeats` greater than one so per-task averages and variance are
meaningful, then run `ng_reward_profile` on the materialized inputs and rollout
JSONL and compare line counts across the artifacts.

Recovering from an interrupted collection: rerun `ng_reward_profile` with
`++allow_partial_rollouts=True` to profile completed rollouts and drop original
input rows that have no completed rollout.

## Limitations

- Per-task averages and variance are only meaningful with multiple rollouts per task; single-repeat runs give point estimates.
- This step summarizes existing rollout artifacts; it does not collect rollouts or fix failed runs.
- Reward semantics are defined by the resource server, not by this workflow.

## Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| No reward profile file produced | Expected it from rollout collection | Reward profiling is a separate step; run it on the materialized inputs and rollout JSONL |
| Profile rows fewer than input tasks | Rollout collection stopped early | Rerun profiling with partial rollouts allowed (see Practical Defaults) |
| CLI flags differ from this guide | Target checkout version differs | Trust the checkout's CLI help and `nemo_gym/reward_profile.py` |
