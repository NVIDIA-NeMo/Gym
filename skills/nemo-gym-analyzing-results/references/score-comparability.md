# Score comparability

Every factor that changes a score, where to check it in a Gym run, how Gym accounts for failures, and how to quantify run-to-run spread. Section 1 is the checklist, section 2 the variance guide.

## 1. Comparability checklist

Each row of the table below is one factor that can change a score by itself, even with the same model. Before comparing two numbers, check that both sides match on every row. When a reference does not publish a setting, write it down as unknown instead of assuming it matches.

| Factor | Why it moves the score | Where to check in a Gym run |
|---|---|---|
| Gym commit and images | Scorer, agent and failure-handling fixes change results between commits. | `driver.gym_install.ref` in the run directory's `resolved-config.yaml`, or the Gym in `driver.container` when that is unset (`gym_commit` in `gym-job.json` is only the Gym that submitted the job); the launcher log for `gym eval run`; `git log --left-right <a>...<b>` over the server, agent, benchmark and core paths; image tags in the launcher config. |
| Dataset version and subset | Different questions, date windows, or a public subset of a larger private set. Data is prepared separately from the rollouts, so it can differ between runs on the same commit. | The `prepare_script` named in the benchmark config (revision pins, filters); task sets and prompts in both runs' `*_materialized_inputs.jsonl`. |
| Prompt and system prompt | The instructions and answer format decide what the extractor can read. | `prompt_config` in the benchmark config; `responses_create_params.input` in the materialized inputs. |
| Metric and aggregation | pass@1 averaged over k, pass@k and majority@k are different numbers. Rollout-weighted and task-weighted means differ once rollouts are missing. Scales differ (0–1 vs 0–100), and some benchmarks report a normalized Elo. | Key names in `*_aggregate_metrics.json`. |
| Repeats | Fewer repeats give a wider spread, and pass@k needs k rollouts per task. | Count rollouts per task in `*_materialized_inputs.jsonl` rather than trusting a flag. Since #5042, `gym eval run --num-repeats` replaces a benchmark's own `num_repeats`; on older commits it multiplied the two (#5041). |
| Sampling | Temperature, top_p, top_k and seeds change outputs. | `responses_create_params` in the materialized inputs, plus any sampling overrides on the model server. |
| Token limits and context length | Truncated reasoning or answers score as wrong. | Max output tokens in the request; the server's max model length; `usage` against the limit in rollouts. |
| Reasoning settings | Answers that land only in the reasoning field score 0. A parser can also move part of an answer into the reasoning field. | Reasoning effort, chat-template arguments, and the serving engine's reasoning parser. |
| Judge | Different judge models or prompts grade differently. Parse failures can count as wrong. | The judge model server in the config; parsing-issue metrics; `judge_failed` rows. |
| User simulator and tools | The simulator model and the search provider change agentic outcomes. | Agent and resources-server config. |
| Timeouts and step limits | Wall-clock and step caps decide how much work fits into a task. | Agent and sandbox timeouts; `max_steps` or turn limits. |
| Failure accounting | Failures can be counted as 0, left out of the denominator, or retried. | See the next section. |
| Agent harness | Different harnesses solve different fractions of the same tasks, and can treat the same timeout differently. | The agent named in the config and in the rollouts. |
| Serving | Precision, engine version, parsers, parallelism and throughput all change results. | Serving config and logs. |

### Failure Accounting in Gym

Read this before comparing a Gym number with a reference that counts failures differently.

- A rollout result that carries `_ng_failure_class` is written to `<output>_failures.jsonl` instead of the rollouts JSONL, and is left out of aggregate metrics. Classes include `agent_request_failed` (429, 502, 503, 504 or no reply), `agent_run_error`, `environment_server_failed`, `judge_failed`, and benchmark-specific classes such as `infrastructure_error`.
- The failures file has one row per failed attempt, so a rollout that failed and then succeeded on `--resume` appears in both files. Only rollouts with no success count as failed.
- A failed agent `/run` ends the whole run unless `route_failures_to_sidecar=true`.
- Rollouts killed mid-flight (SIGTERM, a dead Ray actor, OOM) and rows drained by a dispatch budget write no row to either file by default. With `+record_omitted_rollouts=true` they leave a failures-file row marked `_ng_omitted_rollout`, which Gym counts as missing, not failed.
- `+count_failure_classes_as_zero=[<class>]` scores a rollout as 0 when its last attempt failed with one of those classes, unless that failure row already carries a reward. `+count_missing_rollouts_as_zero=true` scores materialized rollouts that left no row anywhere as 0. Both change only the metric input. Neither flag is recorded in `*_aggregate_metrics.json`, so check the command, `*_resolved_config.yaml` or `resolved-config.yaml`, or the log lines `Counting N failure row(s) as scored zeros` and `Counting N materialized rollout(s) with no row as scored zeros`.
- With either flag on, treat `mean/reward` as the safest number to compare. At the time of writing, imputed rows crash some benchmarks' own metric hooks and are silently mis-scored by others (#3593).
- `require_complete` defaults to false, so a partial run still publishes a score. The Slurm executor of `gym eval submit` turns it on unless the benchmark's `run` overrides it; the Kubernetes executor does not. With it on, `*_aggregate_metrics.json` is still written for the finished rollouts before the run fails with `EVAL FAILED: <done>/<expected> samples completed`, so check the job status, not just the metrics file.
- Rows with `mask_sample` are left out of quality metrics. A failed judge call is masked and tagged `judge_failed`; rerun those rows with `gym eval reverify --judge-failed-only`.
- A judge reply that arrives but cannot be parsed usually counts as a wrong answer, not as `judge_failed`.
- Some servers score an incomplete verification as 0 inside the denominator instead of classifying it: Terminal-Bench 2.1 verifier errors, Terminal-Bench 4.0 verifier timeouts and missing rewards (since #5038), and SWE-bench evaluation failures. An infrastructure problem then lowers the score instead of shrinking the denominator. On Terminal-Bench 2.1, Terminal-Bench 4.0 and SWE-bench, count rows with `evaluation_completed: false` to size it.

### Health Reports

At the end of a run, failed runs included, `gym eval run` writes `quality_summary.json` and `rollout_verdicts.jsonl` beside the rollouts. `--no-health-check` skips them, and with `--disable-aggregation` they come from `gym eval aggregate` instead. `fern/versions/latest/pages/evaluation/rollout-health.mdx` defines every field and check. What they contain:

- `run.artifacts`: since #4788, the run accounting against `*_materialized_inputs.jsonl`: `expected`, `failed` (`total`, `final`, `not_final` and `by_class`; a resume does not retry `final` ones), `missing`, `healthy` and `unhealthy_reasons`. `expected`, `missing` and `healthy` are null when the inventory is missing, and `failed` is null when there is no failures file. `coverage` shows how many rollouts each check evaluated, could not observe or ignored.
- `run.verdicts`: how many saved rollouts are `healthy`, `unhealthy` or `unobserved`. Failed and missing rollouts are not checked; only `run.artifacts` counts them. `run.issues` counts findings per check; `run.findings` lists the run-level ones.
- `run.stats`: failed model calls by HTTP status, token totals and, since #4841, timeouts (rollouts that timed out, by limit, and the time split between the policy model, other models and tools).
- `tasks`: verdict counts and flags per task, such as `task_consistently_unhealthy`.
- `rollout_verdicts.jsonl`: one row per saved rollout, keyed by `_ng_task_index` and `_ng_rollout_index`, with its verdict, its findings and the checks it could not run. Failed and missing rollouts have no row.

Where they mislead:

- `run.artifacts.healthy: true` means only that nothing is missing and no rollout is recorded twice. Failure limits are unset by default, so a run with many failed rollouts still shows `true`; read `failed`.
- `unobserved` is not healthy: the rollout lacked evidence a check needs, such as a trajectory, agent turns, model calls, or the references that tie calls to turns. When most rollouts are `unobserved`, the report says little about them; `coverage` shows which checks could not run.
- The printed `rollout quality:` line counts only the saved rollouts' verdicts, and the `run health` line above it is `run.artifacts.healthy`; `gym --json eval health-check` prints the whole result.
- Each run or `--resume` deletes the previous reports first, so they describe the latest job. A job stopped by SIGTERM leaves only a provisional summary (`partial: true`): the run accounting, without verdicts. A benchmark with its own `rollout_collection_driver`, such as GDPval, runs the full check instead, so its summary does not show that the job was stopped.
- Reports from before #4788 have no run accounting, those from before #4841 have no timeouts, and check IDs change between commits. To compare the health of runs from different commits, rebuild both runs' reports with the same Gym checkout.
- `gym eval health-check <run_dir>` rebuilds both reports and overwrites them in the run directory, so copy a shared or read-only run first.

### Quick Checks

Use this when `quality_summary.json` is missing or predates #4788, or to cross-check it. Assume the run wrote `results/rollouts.jsonl`; Gym names the siblings from that stem. This counts expected rollouts, scored rows, masked rows, rollouts whose last attempt failed (by class), and rollouts that left no row. Rows that `+record_omitted_rollouts=true` writes count as no row, as Gym counts them. A `gym eval aggregate` output has no sibling files, so run it on each shard instead:

```bash
python - results/rollouts.jsonl <<'EOF'
import collections, json, sys
from pathlib import Path
out = Path(sys.argv[1])
def rows(p):
    if not p.exists():
        return []
    result = []
    with p.open("rb") as f:
        for line in f:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # Blank lines, and the truncated last line a killed writer can leave.
            if isinstance(row, dict):
                result.append(row)
    return result
key = lambda r: (r.get("_ng_task_index"), r.get("_ng_rollout_index"))
done = rows(out); scored = {key(r) for r in done}
# A drained rollout never started; an interrupted one counts as missing, not failed.
fails = [r for r in rows(out.with_name(out.stem + "_failures.jsonl"))
         if not (r.get("_ng_omitted_rollout") and r.get("_ng_dispatch_drained"))]
last = {key(r): r for r in fails}
lost = {k for k, r in last.items() if r.get("_ng_omitted_rollout")}
expected = {key(r) for r in rows(out.with_name(out.stem + "_materialized_inputs.jsonl"))}
skip = scored | lost
failed = collections.Counter(r.get("_ng_failure_class") for k, r in last.items() if k not in skip)
print(f"expected={len(expected)} rows={len(scored)} masked={sum(bool(r.get('mask_sample')) for r in done)} "
      f"failed={sum(failed.values())} no_row={len(expected - scored - (last.keys() - lost))} {dict(failed)}")
EOF
```

`gym eval profile` also prints completion against the materialized inputs. Its task rows carry `expected_num_rollouts` and `missing_num_rollouts` only for tasks with at least one unmasked rollout, and `expected_num_rollouts` leaves masked rollouts out, so take totals from the completion line or the counts above. Add `++allow_partial_rollouts=True` when rollouts are missing; strict profiling fails otherwise.

## 2. Run-to-run variance

Use this to decide whether a gap is larger than the run-to-run spread, and how many repeats a decision needs.

### Where Gym Reports Spread

- With two or more repeats, `*_aggregate_metrics.json` already holds `repeat_level_metrics` (per repeat, across that repeat's tasks: mean, SD, SEM and a 95% Student-t interval) and, in each agent's metrics, `std_across_repeats/mean/reward`, `se_across_repeats/mean/reward` and `ci_low_95_across_repeats/mean/reward` / `ci_high_95_across_repeats/mean/reward`. `gym eval profile` writes the same to `*_repeat_level_metrics.json` and `*_agent_metrics.json`.
- Servers that compute pass@k also write `pass@1[avg-of-k]/<score>/std_dev_across_runs` and `pass@1[avg-of-k]/<score>/std_err_across_runs`, on a 0–100 scale. Not every server does; when they are missing, use the `*_across_repeats` keys.
- `gym eval compare --baseline <rollouts A> --candidates <rollouts B>` reports a Welch 95% confidence interval on the difference of per-repeat means (two or more repeats on each side) and, when every reward is 0 or 1, per-task pass/fail flips. It reads each run's aggregate metrics file, so both runs must use the same failure-accounting flags. It joins tasks on `_ng_task_index` alone, which assumes the same dataset, split, limit and order.

### Definitions

- `std_across_repeats/mean/reward`: the sample standard deviation of the per-repeat mean rewards, on 0–1. This is `sd1` below: how much a single repeat's score varies. A repeat's own `std/reward` is the spread across tasks and is much larger.
- `std_dev_across_runs`: the sample standard deviation of the k per-repeat scores.
- `std_err_across_runs`: `std_dev_across_runs / sqrt(k)`, the standard error of the reported mean. It is not a binomial standard error over all samples.
- `avg_sample_std_dev`: the mean, over tasks, of each task's standard deviation across repeats. It is on the 0–1 score scale, unlike the two keys above.
- Standard error of the gap between two independent numbers: `sqrt(se_a^2 + se_b^2)`, with both on the same scale. `se_across_repeats/mean/reward` is on 0–1; the `*_across_runs` keys are on 0–100.

### Rules of Thumb

- On a fixed task set with independent tasks, one run's standard error is at most `sqrt(p * (1 - p) / n)` for n tasks at pass rate p: about 5 points for 100 tasks at p = 0.5, 2.2 for 500 and 1.1 for 2,000. It is lower when most tasks always pass or always fail. The same formula is the extra error a reference picks up from scoring a different sample of tasks. Shared failures, such as an outage that hits many tasks in one run, push the spread above it, so measure it with repeats.
- To reach a standard error s, you need about `k = (sd1 / s)^2` repeats, where `sd1` is the standard deviation of a single repeat's score (`std_across_repeats/mean/reward`). A benchmark whose single repeat varies by 4 points needs about 16 repeats for a 1-point standard error.
- A gap smaller than about `2 * sqrt(se_a^2 + se_b^2)`, two standard errors of the difference, is not evidence of a difference. Report the interval, not only the means.
- Agentic benchmarks with tens of tasks move several points between single runs. Average three or more runs before comparing; the Terminal-Bench 4.0 README describes Artificial Analysis averaging three.

### Spread That More Repeats Won't Remove

- Different clusters or load: wall-clock timeouts bite harder on slower serving, and different parallelism or GPU types change numerics.
- Resumes and retries: `--resume` re-dispatches retryable failures and lost rollouts up to `NEMO_GYM_MAX_ROLLOUT_ATTEMPTS` (default 3), and a later success counts. A reference that allows one attempt per rollout is not counting the same thing. Restarted rollouts can also behave differently from uninterrupted ones.
- Infrastructure nondeterminism: environment setup, patch application and sandbox restarts. For deterministic verifiers, identical outputs must get identical rewards; check that across runs.
- Judges: an unpinned judge model, or judge sampling at a nonzero temperature, adds its own noise. Re-judge a sample with `gym eval reverify` and measure agreement.
