---
name: gym-eval
description: >
  Take a NeMo Gym evaluation from a ready config to a trusted result: submit it, let the resumable
  Slurm chain continue through walltime limits, preemption and node loss, inspect and recover the
  run from its saved evidence with `gym eval resubmit --recover`, check health and the acceptance
  gates, and decide when to stop a chain or start a fresh run. Use when a working Gym config
  (benchmark, model, compute) exists and the run has to finish with a result you can rely on. Not
  for building a benchmark or harness (use add-benchmark or add-harness) or for setting up a
  model's serving config (use gym-add-model-config). Triggered by: "run the eval", "submit this
  benchmark", "the job timed out", "the chain stopped", "resume the run", "recover the run",
  "is the run done", "inspect the run", "gate the score", "restart the evaluation".
---

# Run a Gym Evaluation to Its Result

The Gym pages are the reference; when they and this skill disagree, the pages win:
[Configuration](../../../fern/versions/latest/pages/orchestration/configuration.mdx),
[Resumable Runs](../../../fern/versions/latest/pages/orchestration/how-to/resumable-runs.mdx),
[Job Records](../../../fern/versions/latest/pages/orchestration/job-records.mdx) and
[Resume](../../../fern/versions/latest/pages/evaluation/resume.mdx). This skill adds the order of
work and who decides what.

For running against servers you started yourself (`gym env start`, then `gym eval run --no-serve`)
and for reading health while you build a benchmark or a harness, follow
[add-benchmark](../add-benchmark/SKILL.md) and [add-harness](../add-harness/SKILL.md).

## 1. Before you submit

1. Pin Gym. Set `driver.gym_install.ref` to a full commit, not a branch. Recovery reads a run with
   the Gym it was written by, and refuses an unpinned ref.
2. Make every long benchmark resumable: `resumable: true`, or
   `resumable: {max_retries: N, max_walltime: "D-HH:MM:SS"}` under `driver.benchmarks.<name>`.
3. Check the config without submitting anything:

   ```bash
   gym eval submit --config submit.yaml --resolve-only   # the resolved config, validated
   gym eval submit --config submit.yaml --dry-run        # the job scripts it would queue
   ```

4. Smoke the benchmark small first. `--limit` counts **tasks**, not rows: each kept task brings
   every copy saved in the data and its `--num-repeats`, so rows = tasks × repeats. The run log
   says it: `Limiting to N tasks (M rows with their repeats)`. In a submit config, put the limit
   in the benchmark's `run` (`run: {limit: 5, num_repeats: 1}`).

## 2. Submit and let the chain run

```bash
gym eval submit --config submit.yaml            # prints the GYM_JOB_ID; --select a,b submits only those
```

A resumable benchmark is a chain of Slurm jobs that continues itself. Before doing work, each job
queues its successor, then reads its predecessor's final state:

| Predecessor ended | The chain |
| --- | --- |
| `TIMEOUT` | Continues from the saved rows, within `max_walltime`. |
| `PREEMPTED`, `NODE_FAIL`, `BOOT_FAIL` | Retries, spending one of `max_retries`. |
| `COMPLETED` | Ends. |
| `CANCELLED`, `FAILED`, `OUT_OF_MEMORY`, `DEADLINE`, unknown accounting | Stops. |
| `TIMEOUT` before the task inventory was written | Stops and names the fresh-restart command (step 6). |

A run that hangs instead of failing is not left to hold its allocation: a head server that never
answers, or a run stopped by a typed non-resumable condition, exits 76, and the chain queues no
follow-up for it.

Do not resubmit while a job of the run is live; recovery answers NOOP then anyway.

## 3. Where the evidence lives

| Location | What |
| --- | --- |
| Run directory | `gym-job.json` (the submission record), `resolved-config.yaml` (the config as submitted), and the journals `walltime-extension-journal.jsonl`, `repin-journal.jsonl`, `unsafe-resume-journal.jsonl`. |
| Benchmark directory | `job.sh`, `logs/driver-<job id>.log` and `logs/<service>-<job id>.log`, the chain files `.gym_job_chain`, `.gym_accumulated_walltime`, `.gym_infra_retries`, and `.gym_chain_incomplete` (why the chain stopped, when it records a reason). |
| `artifacts/` | `rollouts.jsonl` (results), `rollouts_failures.jsonl` (failed attempts), `rollouts_materialized_inputs.jsonl` (the task inventory), `rollouts_manifest.json` with its identity and resume-drift journal, `rollouts_coverage.json`, `rollouts_aggregate_metrics.json`, `quality_summary.json` (health), and `rollouts_mlflow_export_done` (the post-run export receipt). |
| Your machine | A copy of each submission record under the Gym jobs cache (see Job Records). |

Read these files; do not edit or delete them. Recovery decides from them, and a changed file makes
it refuse.

## 4. Inspect first, always

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover --inspect \
  --score-agent <agent name in the aggregate> \
  --score-key '["agent_metrics", "mean/reward"]'
```

`--inspect` reads the saved evidence and evaluates the gates without dispatching anything. Read:

- the decision's action and `reason`. When the run's latest resume crossed operational
  configuration drift, the reason ends with `resumed across operational drift: <settings>`.
- `gate_task.status`: `RUNNING` (a job is live), `PLANNED` (eligible work, nothing dispatched),
  `FAILED` (stopped, unsupported, or a gate failed: read `errors`), `SUCCESS` (complete and every
  gate passed).
- `gate_task.trial_counts`: `observed_healthy`, `unhealthy`, `unobserved`. Unobserved trials are
  unverified, not healthy.

## 5. Act on the decision

| The inspection says | Do |
| --- | --- |
| Eligible work, `PLANNED` | Run the same command without `--inspect`. |
| The chain's walltime cap is spent | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`. Never below the recorded walltime; journaled. |
| The infra-retry budget is spent | `--recover --extend-retries N`. The spent count is kept; journaled. |
| The run's chain accounting is unknown | An explicit budget continues it: `--max-walltime` with `--walltime-reason`, optionally `--extend-retries`. Without it, the STOP names these options. |
| A Gym fix is needed to continue | Install that commit as the launcher, then `--recover --repin <full sha> --repin-reason "<why>"`. Saved rows and budgets stay. |
| The resume check refuses a configuration difference | Operational differences (timeouts, retries, ports, concurrency, paths, tracking) already resume by themselves, with a warning and a line in the resume-drift journal. A **semantic** difference (data, limits, repeats, model, sampling, judge or grading) is refused, naming the settings. Usually the right answer is a fresh run (step 7). Only when the operator accepts that exact difference: `--recover --allow-unsafe-resume --unsafe-resume-reason "<why>"`, which records the reason and the user. It never bypasses corrupt or conflicting history. |
| `fresh-restart` (no task inventory) | Step 6. |
| `SUCCESS` | Done. Keep the run directory as the record. |

Plain `gym eval resubmit` without `--recover` resets the retry and walltime budgets and the
per-rollout attempt allowance. It is not a substitute for a recovery STOP.

## 6. When nothing can be recovered

A segment that ended before the task inventory was written has nothing to resume. Recovery answers
`fresh-restart` and names the command:

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover \
  --max-walltime HH:MM:SS --walltime-reason "<why the longer walltime>"
```

`--max-walltime` must be longer than the recorded walltime. The run starts fresh with fresh budgets
in the same directory; the saved files are kept, and the plan is journaled.

## 7. Who decides what

The product owns the evaluation retry; you own the model retry. A resubmit picks up only rollouts
that are not terminal: successes and model failures are never run again. Infrastructure failures
are retried within a bounded budget, so when that budget is spent, or the cluster was down, you run
`gym eval resubmit <id> <benchmark> --recover` again (after `--inspect`), and nothing else.

Resubmit by hand after a cluster outage, a crashed server, an expired credential, or a budget stop.
The chain's own stop text (in the driver log and `.gym_chain_incomplete`) and the inspection's
reason say which one it was.

| The product decides | The operator decides |
| --- | --- |
| Retrying infrastructure failures (transport, server start, sandbox start) within the retry budget. | Raising the walltime cap or the retry budget, with a reason. |
| That model failures and final zeros are final: they are scored under the benchmark's protocol, not retried. | Moving a run to another Gym commit (`--repin`). |
| Re-judging saved answers whose judge failed, at the end of a run, without calling the policy again. | Accepting a semantic configuration difference (`--allow-unsafe-resume`), or starting a fresh run instead. |
| Health verdicts and the gate results. | The gate thresholds for this evaluation, and whether a result is accepted. |
| Resuming across operational configuration drift, and saying so. | When to stop a chain, and when to start over in a new location. |

## 8. Overriding evaluation parameters

Only the number of repeats (`num_repeats`) and the client parallelism (`--concurrency`, or
`num_samples_in_parallel`) may be changed from the command line or a benchmark's `run`. Never
change the benchmark, the judge, or the benchmark's token budgets and timeouts. Those are the
evaluation's identity, and changing them makes a different evaluation. The model section is yours:
see [gym-add-model-config](../gym-add-model-config/SKILL.md).

A slow or under-provisioned deployment produces timeouts and truncated answers, and they count as
model failures under each benchmark's rules. Optimize the deployment; do not raise the evaluation's
limits.

## 9. Health and acceptance

The health summary and the gates are the contract you read after a run: the response-failure and
trial-failure rates, coverage, and the unhealthy rollouts. A run can be healthy and still score
low. That is the model or its deployment: fix the deployment and run again, not the evaluation.

- Health is written to `quality_summary.json` after the run. `gym eval health-check <run dir>`
  re-runs the checks; it reports, it does not fail the run for you.
- Gates are flags on `--recover`: `--score-key` (a JSON path) with `--score-agent` and, when an
  agent has several entries, `--score-environment`; `--max-unhealthy-fraction` (per-rollout
  health); `--max-response-failure-fraction` and `--max-trial-failure-fraction` (default 0.02;
  response passes strictly below, trial fails at or above).
- Pass `--mlflow-run-id <id> --mlflow-status FINISHED` only after you observed that run and status
  yourself. They do not export anything; acceptance also needs a current export receipt.
- `SUCCESS` needs complete results and every gate. A NOOP, a finished job, or a full score alone is
  not acceptance.

## 10. Stopping a chain

1. Cancel only the running job of the chain (`scancel <slurm job id>`). Its queued successor sees
   `CANCELLED` and stops without doing work.
2. Then inspect (step 4) before deciding anything else.
3. Keep the run directory. To evaluate something different, submit a new run to a new location
   rather than reusing this one; a fresh `gym eval run` at an existing output path replaces it.

## Exit checks

- [ ] `--inspect` shows `SUCCESS`, or the remaining STOP is explained and accepted by the operator.
- [ ] Every operator action (walltime, retries, repin, unsafe resume) has a reason in its journal.
- [ ] `trial_counts` and `quality_summary.json` were read: unobserved and unhealthy counts are
  explained.
- [ ] Any `resumed across operational drift` settings are listed with the result.
