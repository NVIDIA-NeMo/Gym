# Reward Profiling Output Format

Reward profiling is built around joining materialized inputs with rollout results by task and rollout identity.

## Files

- `*_materialized_inputs.jsonl`: expanded inputs after repeat expansion. Each row should have `_ng_task_index` and `_ng_rollout_index`.
- `rollouts.jsonl`: completed rollout results. Each row should have matching `_ng_task_index` and `_ng_rollout_index`.
- `*_reward_profiling.jsonl`: task-level summaries produced by `gym eval profile`.
- `*_agent_metrics.json`: agent/global aggregate metrics.
- `*_repeat_level_metrics.json`: per-repeat summaries.

## Task Profile Rows

Task-level profile rows include:

- `_ng_task_index`: original task/sample id.
- `sample`: representative materialized input row for the task, with task/rollout ids removed from the sample copy.
- `num_rollouts`: number of rollout results summarized for the task.
- `expected_num_rollouts`: number of materialized rollout rows expected for the task.
- `missing_num_rollouts`: number of expected rollout rows missing from the profile.
- `reward_profile_completion_pct`: percent of expected rollout rows included for the task.
- `rollout_infos`: compact per-rollout records sorted by `_ng_rollout_index`.
- aggregate metric keys such as `mean/reward`, `max/reward`, `min/reward`, `median/reward`, and `std/reward`.
- token usage aggregate keys such as `mean/input_tokens`, `std/output_tokens`, and `mean/total_tokens`, when those fields are present in `response.usage`.

## Repeat-Level Metrics

`*_repeat_level_metrics.json` holds one entry per `(agent, _ng_rollout_index)`. Agents with a single repeat are skipped entirely.

Each entry includes:

- `agent_ref`, `_ng_rollout_index`
- `sample_count`: tasks with a completed rollout in this repeat.
- `missing_count`: tasks present in some *other* repeat but not this one. Not a count against the expected task list — see below.
- per numeric field: `mean/`, `median/`, `std/` (ddof=1), `sem/`, `min/`, `max/`, `p25/`, `p75/`, and `ci_low_95/`+`ci_high_95/` (Student's t, omitted when `n <= 1`).

`missing_count` derives its denominator from the tasks that completed in at least one repeat, not from the materialized inputs. A task that fails in *every* repeat therefore appears in no repeat, is counted nowhere, and leaves `missing_count` at `0`. Those rollouts live in `<output>_failures.jsonl`, or nowhere at all for `kill_shaped` failures (Slurm SIGTERM, Ray actor died, OOM), where the absent row is itself the signal.

This makes the worst case silent: when the same tasks fail in every repeat, all repeats share an identical sample count, `missing_count` is `0` throughout, and the unequal-sample-size warning never fires — even though every statistic was computed over only the tasks that finished. For true coverage read `expected_num_rollouts`/`missing_num_rollouts` in the task profile rows, or the completion summary printed at the end of `gym eval profile`; both are measured against the materialized inputs.

The per-repeat `mean/{field}` values are also summarized across repeats and merged into `*_agent_metrics.json` as `mean_across_repeats/mean/{field}`, `median_across_repeats/mean/{field}`, `se_across_repeats/mean/{field}`, and `ci_low_95_across_repeats/mean/{field}`+`ci_high_95_across_repeats/mean/{field}`.

Read these keys outward-in: `se_across_repeats/mean/reward` is the standard error, across repeats, of the per-repeat mean reward. The `_across_repeats` half is the cross-repeat stat; the half after the slash is the per-repeat estimate being aggregated. 

Two cases emit a `UserWarning`:

- Unequal `sample_count` across repeats — statistics come from different task sets and are not directly comparable; agent metrics skew toward whichever tasks completed. Fires off `missing_count`, so it inherits the blind spot above: tasks that failed in every repeat do not trigger it.
- Zero `sem` (all values identical) — the CI collapses to `(mean, mean)`. Reported explicitly because SciPy's `t.interval` returns `NaN` at `scale=0`.

Confidence intervals are unbounded, so a 95% CI on a 0–1 reward over few tasks can fall outside `[0, 1]`. That is expected for a t-interval.

`rollout_infos` are intentionally compact. They can include:

- `rollout_id`, usually `task_idx:rollout_idx`
- `_ng_task_index`
- `_ng_rollout_index`
- `reward`
- token usage fields from `response.usage`, when present
- numeric verifier/result fields, when present

Full model responses stay in `rollouts.jsonl`. Join back to full rows with `(_ng_task_index, _ng_rollout_index)` when needed.

## Partial Profiles

Strict profiling is the default. If materialized inputs and rollout results do not have the same rollout keys, `gym eval profile` fails and suggests:

```bash
++allow_partial_rollouts=True
```

With partial profiling enabled, rows with no matching materialized input still fail, but missing rollout results are allowed. The profile includes original input tasks with at least one completed rollout and drops original input tasks with zero completed rollouts.

At the end, the command prints rollout completion and input-task status counts:

- completed rollout rows / expected rollout rows and percentage
- complete input tasks
- partial input tasks
- input tasks without rollouts that were dropped
