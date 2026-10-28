# Comparability Checklist

Each row is something that changes a score. Before comparing two numbers, check that both sides match on every row. When a reference does not publish a setting, write it down as unknown instead of assuming it matches.

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

## Failure Accounting in Gym

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

## Quick Checks

Assume the run wrote `results/rollouts.jsonl`; Gym names the siblings from that stem. This counts expected rollouts, scored rows, masked rows, rollouts whose last attempt failed (by class), and rollouts that left no row. Rows that `+record_omitted_rollouts=true` writes count as no row, as Gym counts them. A `gym eval aggregate` output has no sibling files, so run it on each shard instead:

```bash
python - results/rollouts.jsonl <<'EOF'
import collections, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = lambda p: [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []
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

Then check rollout health: empty or runaway generations, failed model calls, and tasks that are unhealthy in every repeat.

```bash
gym eval health-check results
```

`gym eval profile` also prints completion against the materialized inputs. Its task rows carry `expected_num_rollouts` and `missing_num_rollouts` only for tasks with at least one unmasked rollout, and `expected_num_rollouts` leaves masked rollouts out, so take totals from the completion line or the counts above. Add `++allow_partial_rollouts=True` when rollouts are missing; strict profiling fails otherwise.
