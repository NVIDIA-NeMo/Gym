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
   says it: `Limiting to N tasks (M rows with their repeats)`; a Gym pinned before that line
   prints `Limiting the number of tasks to N`, and `expected` in
   `artifacts/rollouts_coverage.json` gives the rows. In a submit config, put the limit in the
   benchmark's `run` (`run: {limit: 5, num_repeats: 1}`).
5. Submitting from a machine that is not the login node: set `compute.cluster.hostname` to the
   login node, and list every `runtime:` credential in `compute.cluster.forward_env`; only listed
   names reach the job. Site `#SBATCH` directives go in `compute.cluster.extra_args` without
   leading dashes (`{qos: <qos>, exclude: <node>}`).
6. `gym eval submit` also takes Hydra overrides after its flags, though its `--help` does not
   list them: `++key=value` sets or replaces a key, `+key=value` only adds a new one. For example
   `gym eval submit --config submit.yaml --dry-run ++compute.cluster.walltime=00:40:00
   ++driver.benchmarks.<name>.run.limit=2`. Check the result with `--resolve-only` or `--dry-run`.
7. An evaluation you do not track in MLflow on purpose: set `mlflow_tracking_required: false` in
   each benchmark's `run`. It waives only the delivery requirement of the acceptance gate (step
   9); health checks still run, and a configured MLflow export still happens. Without it, the
   gate of a complete, healthy run stays `FAILED` on "no MLflow run evidence". A Gym pinned
   before this skill does not know the key.

## 2. Submit and let the chain run

```bash
gym eval submit --config submit.yaml            # --select a,b submits only those
```

The GYM_JOB_ID is the last part of the `Run directory:` line it prints (`gym-job-<time>-<hex>`);
the `submitted <benchmark> → Slurm job <id>` line gives the first Slurm job.

A resumable benchmark is a chain of Slurm jobs that continues itself. Before doing work, each job
queues its successor (`afternotok:<this job>`) and a sentinel (`afterany:<successor>`) that takes
over only if the successor dies before its own prologue, then reads its predecessor's final state:

| Predecessor ended | The chain |
| --- | --- |
| `TIMEOUT` | Continues from the saved rows, within `max_walltime`. |
| `PREEMPTED`, `NODE_FAIL`, `BOOT_FAIL` | Retries, spending one of `max_retries`. |
| `COMPLETED` | Ends. |
| `CANCELLED`, `FAILED`, `OUT_OF_MEMORY`, `DEADLINE`, unknown accounting | Stops. |
| `TIMEOUT` before the task inventory was written | Stops and names the fresh-restart command (step 6). |

A run that hangs instead of failing is not left to hold its allocation. A driver that stops on a
typed condition exits at once (76 when another attempt cannot help), and the next segment's guard
ends the chain and writes the kind and the way out to `.gym_chain_incomplete`:

| Typed stop | What happened | The one command |
| --- | --- | --- |
| `server_spinup_timeout` | The Gym servers were not ready within `server_spinup_timeout_seconds` (600 s unless the benchmark's `run` sets it). | `--recover --server-spinup-timeout-seconds N --spinup-timeout-reason "<why>"`: `N` above the recorded limit, 0 for no limit; journaled. |
| `driver_install_failed` | The job could not install Gym (the installer, the checkout or the package). The stop quotes the installer's error; the rest is in `logs/driver-<job id>.log`. | Fix the cause (the ref, the network, the image), then `--recover`. |
| `provider_credentials_rejected` | The sandbox provider answered 401 or 403. | Fix the credentials, then `--recover`. |
| `collection_resume_refused` | The saved rollouts do not verify against this run's inputs and configuration. | A fresh run (step 10), or, only when the operator accepts the difference, `--recover --allow-unsafe-resume --unsafe-resume-reason "<why>"`. |
| A resumed segment saved no rollout | Its walltime was shorter than the setup (Gym install, server start) plus one rollout, so the next segment would repeat it. | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`. |

While the chain runs, `gym eval status "$GYM_JOB_ID"` prints each benchmark's saved progress as the
Store recorded it, without taking the run lock: `updated_at` and `tracking`, the `export receipt`
per exporter (present or missing, with the live run id), and per agent `completed`, `remaining`,
`expected`, `aggregation`, `failures by class`, `by kind` and `model-final by kind` (`--json` gives the
same as data).

Do not resubmit while a job of the run is live; recovery answers NOOP then anyway.

## 3. Where the evidence lives

| Location | What |
| --- | --- |
| Run directory | `gym-job.json` (the submission record), `resolved-config.yaml` (the config as submitted), and the journals `walltime-extension-journal.jsonl`, `repin-journal.jsonl`, `unsafe-resume-journal.jsonl`. |
| Benchmark directory | `job.sh`, `logs/driver-<job id>.log` and `logs/<service>-<job id>.log`, the chain files `.gym_job_chain`, `.gym_accumulated_walltime`, `.gym_infra_retries`, `.gym_infra_retry_extension`, `.gym_followup_queued.<job id>`, `.gym_followup_sentinel.<job id>` and `.gym_chain_incomplete` (why the chain stopped, when it records a reason), and the journals `retry-extension-journal.jsonl`, `spinup-timeout-journal.jsonl` and `operator-continuation-journal.jsonl`. |
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

The agent name is the rows' `agent_ref.name`, the agent entry in
`artifacts/rollouts_aggregate_metrics.json`. Without the score flags the score gate reports
"score unavailable: no requested metric was read".

`--inspect` reads the saved evidence and evaluates the gates without dispatching anything. The
read runs on the run's host; the first time, it installs the run's pinned Gym for its reader with
a private copy of uv and Python, which takes minutes and replaces nothing in your home. Later reads
on that host reuse the cache. Read:

- the decision's action and `reason`. When the run's latest resume crossed operational
  configuration drift, the reason ends with `resumed across operational drift: <settings>`.
- `gate_task.status`: `RUNNING` (a job is live), `PLANNED` (eligible work, nothing dispatched),
  `FAILED` (stopped, unsupported, or a gate failed: read `errors`), `UNVERIFIED` (complete and
  nothing failed, but acceptance is not proven, such as evaluation health UNVERIFIED: read
  `warnings`), `SUCCESS` (complete and every gate passed). Only `SUCCESS` exits 0.
- `gate_task.trial_counts`: `observed_healthy`, `unhealthy`, `unobserved`. Unobserved trials are
  unverified, not healthy.

## 5. Act on the decision

| The inspection says | Do |
| --- | --- |
| Eligible work, `PLANNED` | Run the same command without `--inspect`. |
| `reverify`, `PLANNED` (only judge failures remain) | Run the same command without `--inspect`: the saved answers are judged again and the policy is not called. The executor runs this today only for a submission that deploys no model service; otherwise the outcome says so (`no existing CPU-only judge execution profile`) and the judge failures stay as they are. |
| NOOP, `RUNNING` (`continuation <job> live; nothing to decide`) | A job of the run is live. Wait for it to end, then inspect again. |
| STOP typed `server_spinup_timeout` | `--recover --server-spinup-timeout-seconds N --spinup-timeout-reason "<why>"` (the table in step 2). |
| The chain's walltime cap is spent | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`. Never below the recorded walltime; journaled. |
| The infra-retry budget is spent | `--recover --extend-retries N`. The spent count is kept; journaled. |
| The run's chain accounting is unknown | An explicit budget continues it: `--max-walltime` with `--walltime-reason`, optionally `--extend-retries`. Without it, the STOP names these options. |
| A Gym fix is needed to continue | Install that commit as the launcher, then `--recover --repin <full sha> --repin-reason "<why>"`. Saved rows and budgets stay. |
| The resume check refuses a configuration difference | Operational differences (timeouts, retries, ports, concurrency, paths, tracking) already resume by themselves, with a warning and a line in the resume-drift journal. A **semantic** difference (data, limits, repeats, model, sampling, judge or grading) is refused, naming the settings. Usually the right answer is a fresh run (step 7). Only when the operator accepts that exact difference: `--recover --allow-unsafe-resume --unsafe-resume-reason "<why>"`, which records the reason and the user. It never bypasses corrupt or conflicting history. |
| `fresh-restart` (no task inventory) | Step 6. |
| NOOP, `FAILED`, and the only error starts with `no MLflow run evidence` | The results are complete, but no delivery is on record: no live MLflow run id and no post-run export receipt. The rest of the error says which case it is. "although the saved config snapshot names a tracking URI, experiment and run name": tracking was configured and required, and the export did not finish. Read the exporter's failure in the driver log, repair the delivery, and export the run again so that the receipt `rollouts_mlflow_export_done` exists (`gym eval export` with the `mlflow` exporter writes it once every upload returned); then inspect again. "does not name all of the tracking URI, experiment and run name": the run was never tracked; read `trial_counts`, coverage and the score yourself, and set `mlflow_tracking_required: false` next time (step 1). |
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

Which saved failures run again is decided by the run's recovery outcome policy, not by a fixed
model-versus-infrastructure split. A resubmit never reopens a settled result or a failure marked
final. For every other saved failure the policy picks one action from the failure's HTTP status,
else its failure kind, else the default: `retry` (another attempt, within the identity's attempt
cap and the run's infra-retry cap), `verify` (re-judge the saved answer, never the policy) or
`fail` (leave it ungraded). The default recovery policy retries, re-judges `judge_failed`, and
keeps what producers settled as the model's final result (an answered agent deadline, a scored
zero). `--inspect` lists the identities in each state. When the budget is spent, or the cluster
was down, run `gym eval resubmit <id> <benchmark> --recover` again after `--inspect`; running the
same command does not extend an exhausted budget, `--extend-retries` and `--max-walltime` do.

The CLI extends budgets; it does not change the policy. The Python API takes a partial override
(the defaults stay for everything it does not name, kinds must be registered, and `count_as_zero`
is refused here):

```python
from nemo_gym.agent_outcome import OutcomeAction
from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig
from nemo_gym.orchestration.submit import recover

recover(gym_job_id, benchmark, outcome_policy=RecoveryOutcomePolicyConfig(by_status_code={401: OutcomeAction.FAIL}))
```

A CLI or config path for the override does not exist yet.

Resubmit by hand after a cluster outage, a crashed server, an expired credential, or a budget stop.
The chain's own stop text (in the driver log and `.gym_chain_incomplete`) and the inspection's
reason say which one it was.

| The product decides | The operator decides |
| --- | --- |
| Retrying the failures the outcome policy marks `retry` (by default every infrastructure failure: transport, server start, sandbox start) within the retry budget. | Raising the walltime cap or the retry budget, with a reason. |
| That model failures and final zeros are final: they are scored under the benchmark's protocol, not retried. | Moving a run to another Gym commit (`--repin`). |
| Re-judging saved answers whose judge failed, at the end of a run, without calling the policy again. | Accepting a semantic configuration difference (`--allow-unsafe-resume`), or starting a fresh run instead. |
| Health verdicts, the gate's fixed 2% limit and the gate results. | Whether a result is accepted, on the gate's facts. |
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

- Health is written to `quality_summary.json` after the run.
  `gym eval health-check <run dir>/<benchmark>/artifacts` re-runs the checks on a local path; it
  reports, it does not fail the run for you. It rewrites `quality_summary.json` in that directory,
  so run it on a copy, never on the run's own `artifacts/` (step 3).
- What decides acceptance: complete results, the score gate (`--score-key`, a JSON path, with
  `--score-agent` and, when an agent has several entries, `--score-environment`), a readable
  health summary, the MLflow delivery when it is required (step 1), and evaluation health.
  Evaluation health is `gate_task.evaluation_health`: `PASS` when evaluation-side trial failures
  (judge, verifier, sandbox and harness faults) are strictly below 2% of the expected trials and
  every trial's grading is traceable; `FAIL` at or above 2%; `UNVERIFIED` below 2% when some
  grading is not traceable, so the known count is a lower bound. The 2% is fixed in Gym; no flag
  changes it.
- What is reported and not applied: `--max-response-failure-fraction` and
  `--max-trial-failure-fraction` are validated and echoed under `gate_task.criteria` with
  `applied: false`; `--max-unhealthy-fraction` (per-rollout health) is judged in `warnings`. None
  of them changes the verdict. `gate_task.failures` lists the response failures (truncated, empty,
  malformed) and the trial failures by cause, with their rates: facts for you to judge, not
  verdicts. Model failures count in the score and never fail health.
- Pass `--mlflow-run-id <id> --mlflow-status FINISHED` only after you observed that run and status
  yourself. They do not export anything; acceptance also needs a current export receipt.
- With MLflow configured, the post-run export logs the same facts as metrics: `health/response/*`,
  `health/trial/*` (`known`, `passed`, `counts_known`, `expected`, `failed` or
  `failed_lower_bound`, the rates), `health/upstream_recovery/*` and `health/grading_recovery/*`.
  Read `known` before a `passed` value: `known=0` means unknown, not failed.
- `SUCCESS` needs complete results and every gate. A NOOP, a finished job, or a full score alone is
  not acceptance.

## 10. Stopping a chain

1. Cancel the running job of the chain and the jobs it queued (`scancel <ids>`): its successor
   (in `.gym_followup_queued.<job id>`) and the successor's sentinel (in
   `.gym_followup_sentinel.<job id>`). A successor that is left to start sees `CANCELLED` and stops
   without doing work, but only after it got an allocation. Cancel a pending sentinel after a
   completed run too: once Slurm cancels the successor, the sentinel's dependency is met. If you
   cancel only the running job, its follow-up starts when it gets an allocation, sees `CANCELLED`,
   writes the stop to `.gym_chain_incomplete` and exits; cancelling the successor and the sentinel
   too saves that allocation and leaves no stop record (the inspection reads the state from the
   scheduler either way).
2. Then inspect (step 4) before deciding anything else.
3. Keep the run directory. To evaluate something different, submit a new run to a new location
   rather than reusing this one; a fresh `gym eval run` at an existing output path replaces it.

## Exit checks

- [ ] `--inspect` shows `SUCCESS`, or the remaining STOP is explained and accepted by the operator.
- [ ] Every operator action (walltime, retries, repin, unsafe resume) has a reason in its journal.
- [ ] `trial_counts` and `quality_summary.json` were read: unobserved and unhealthy counts are
  explained.
- [ ] Any `resumed across operational drift` settings are listed with the result.
- [ ] The run is final on its receipts: `rollouts_coverage.json` says complete, the export
  receipt `rollouts_mlflow_export_done` is current (or `mlflow_tracking_required: false` was set
  on purpose), and `.gym_chain_incomplete` is absent or explained.
