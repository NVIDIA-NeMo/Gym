# Harbor parity

Compares a native-path run with a Harbor-path run of the same tasks after an environment is converted to Harbor task
folders. Run both paths with the same harness, model, task subset and number of repeats, then compare them.
Run it from the repository root, in the Gym environment (`uv sync`); it reads the row rules from `nemo_gym`.

```bash
python -m scripts.harbor_parity \
    --native <native run dir or rollouts.jsonl> --native-key verifier_metadata.instance_id \
    --harbor results/harbor/<taskset>/ \
    --json parity.json --markdown parity.md
```

The markdown report is printed and, with `--markdown`, written to a file; `--json` writes the full report. Exit code:
0 PASS, 1 FAIL, 2 input error (including unmatched tasks), 3 INCONCLUSIVE. Reports print the input paths as passed,
so when a report will be pasted into a pull request, run from the run folder and pass relative paths.

## Inputs

Each side is one `gym eval run` output: `<name>.jsonl` (scored rows), `<name>_failures.jsonl` (rollouts with no
result) and `<name>_materialized_inputs.jsonl` (one row per expected rollout). Pass the JSONL or a directory that holds
exactly one `rollouts.jsonl`, directly or one level down. Files are streamed line by line and each row is reduced to
its reward, masking and failure fields, so large trajectories are not held in memory.

## Task matching

Matching is explicit; nothing is guessed.

- `--native-key`: a dotted field naming the task on a native row. A row that lacks it (native results often carry
  only `_ng_task_index`) is keyed through its materialized input row with the same `_ng_task_index`, so dataset
  fields such as `verifier_metadata.instance_id` work.
- `--harbor-key`: default `_ng_task_id.task_id`, the task folder name.
- `--id-map map.json`: a JSON object `{native id: Harbor task id}` for a converter that renames tasks.
- A native dataset with no id field: use `--native-key _ng_task_index` (the 0-based dataset row) with `--id-map`
  mapping each index, written as a string (`"0"`), to its Harbor task folder name. Write the map from the same rows,
  in the same order, that the adapter converted.

A task on only one side is an error that lists the unmatched tasks on each side. `--allow-unmatched` compares the
common tasks and makes the verdict INCONCLUSIVE. A key value that names several dataset rows is an error.

## What is counted

Row semantics are those of `gym dev compare`:

| Count | Meaning |
| --- | --- |
| scored | A row with a finite reward and no `mask_sample`. A `failure_kind` on such a row (a measured failure, such as a scored verifier timeout) still counts and is listed under failure kinds. |
| masked | A row with `mask_sample`, no reward or a non-finite reward. Its reward is left out of the mean. |
| failed | A standing failures-sidecar record: the last record per (task index, rollout index), and none whose rollout was scored on a retry. |
| missing | An expected rollout (materialized input) with no record at all: the run is unfinished. |

Unlike `gym dev compare`, which keeps every main-file row, only the last main-file row per (task index, rollout
index) stands, so a resumed run that rewrites a row is counted once.

## Report

Per task: mean reward on each path, scored and expected rollouts, masked and failed counts (missing rollouts are
shown beside the failed count, as `0 (+1 missing)`), and the difference.

Overall: each path's mean over the paired tasks with a bootstrap interval and its task-level standard error (the
spread of per-task means over tasks, which sets how large a shift a run of this many tasks can resolve), and the
difference (Harbor minus native) with a paired bootstrap interval and standard error. When no paired task's mean
differs, the report says the interval is degenerate: it is `[0, 0]` because nothing changed, not because noise is
small, so read the task-level standard errors for the run's power.

Supplementary lines, which never change the verdict:

- the number of tasks whose means differ in each direction, with an exact two-sided sign test on them;
- each path's repeat-to-repeat spread (each repeat's mean by rollout index, and the largest minus the smallest),
  printed beside the between-path difference, so a reader sees whether the paths differ more than a path differs
  from itself;
- the repeat-to-repeat standard error of each path's mean over the paired tasks: `sqrt(sum(var_i / k_i)) / n`, with
  `var_i` the variance of task i's `k_i` scored rewards and `n` the number of paired tasks (a task with one scored
  repeat contributes 0);
- failure kinds and `verifier_rewards` keys seen on only one path.

## Verdict

Statistical mode (default, a model in the loop):

- PASS when every matched task has a scored rollout on both paths, neither run is unfinished, the masked rates and
  the failed rates of the two paths differ by at most `--max-rate-gap` (default 0.05), and the `--confidence` (default
  0.95) paired bootstrap interval of the mean difference contains 0 or lies within `+/- --tolerance` (default 0).
- FAIL when the interval excludes that, or a rate gap is larger.
- INCONCLUSIVE when a task is unscored on a path, a run is unfinished, tasks were left unmatched, or fewer than
  `--min-tasks` (default 10) tasks are paired; a would-be FAIL on too few tasks is reported as such.

Pairing over tasks cancels task difficulty, so the interval measures run-to-run noise of both paths plus any real
per-task change. Bootstrap resamples (`--bootstrap`, default 10000) use `--seed` (default 0), so a report is
reproducible.

### What the verdict shows

A statistical PASS bounds a shift in mean reward on these tasks, with this harness and model: the interval says how
large a difference between the paths the run cannot rule out. It does not prove the paths are equivalent. It says
nothing about tasks that were not run, a small shift can hide inside a wide interval, and when the model solves
almost every task (or none) on every repeat, the run has little power to see a shift that affects only some tasks.
A FAIL says the mean moved by more than the run's noise; the per-task table and the failure kinds show where.

### Choosing `--tolerance`

With the default `--tolerance 0`, a PASS needs an interval that contains 0. When the model is nearly deterministic,
a few tasks that differ in the same direction can put an edge of the interval at or just past 0, so the verdict is
sensitive at that edge; the sign test and the repeat-to-repeat spread show whether such a split is unusual. For
scale: with 2 repeats, a task whose one repeat flips moves by 0.5; on about 50 tasks, 3 such flips in one direction
and none in the other still PASS at `--tolerance 0` (sign test p = 0.25), while 4 FAIL (interval [+0.01, +0.08])
although the exact sign test gives p = 0.125. When a FAIL rests on a few such flips, first diff the first model
request of a differing task on both paths to rule out a real difference in what the model saw; then add repeats
(or tasks) and run again, or use a tolerance that was stated before the run. Do not change the tolerance after
looking at the results. Before the run, choose it as the smallest shift in mean reward
that would matter to the people who use the converted environment (for example, a change that would alter a
training signal or a reported score), and record it with the task subset, the number of repeats and the reason.
Then check that the run can resolve it: the half-width of a 95% interval is about twice the paired standard error
of the difference, so pick enough tasks (tasks the model solves only sometimes carry the most information) and
repeats that this half-width is below the tolerance; a pilot run gives the standard error to plan with.

### Strict mode

Strict mode (`--strict`, model-free checks such as the reference solution or a negative control): PASS when every
rollout of every task is scored on both paths and all rewards of a task, on both paths, are the same value (and equal
`--expect` when given). FAIL on any differing reward; INCONCLUSIVE when a rollout is masked, failed or missing.

```bash
# Reference solutions: 1.0 everywhere on both paths
python -m scripts.harbor_parity --native native_oracle/ --native-key verifier_metadata.instance_id \
    --harbor results/harbor/validation/<taskset>/ --strict --expect 1
```

## Tests

```bash
pytest tests/unit_tests/tooling/test_harbor_parity.py
```
