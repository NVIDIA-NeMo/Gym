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
   benchmark's `run` (`run: {limit: 5, num_repeats: 1}`). A smoke that must stay in one allocation
   also caps its chain, `resumable: {max_retries: 0, max_walltime: <the walltime>}` beside
   `compute.cluster.walltime`; otherwise a segment that times out continues under the config's cap.
5. Submitting from a machine that is not the login node: set `compute.cluster.hostname` to the
   login node, and list every `runtime:` credential in `compute.cluster.forward_env`; only listed
   names reach the job. Site `#SBATCH` directives go in `compute.cluster.extra_args` without
   leading dashes (`{qos: <qos>, exclude: <node>}`).
6. `gym eval submit` also takes Hydra overrides after its flags, though its `--help` does not
   list them: `++key=value` sets or replaces a key, `+key=value` only adds a new one. For example
   `gym eval submit --config submit.yaml --dry-run ++compute.cluster.walltime=00:40:00
   ++driver.benchmarks.<name>.run.limit=2`. Check the result with `--resolve-only` or `--dry-run`.
7. An evaluation you do not track in MLflow on purpose: set `mlflow_tracking_required: false` in
   each benchmark's `run`. Delivery to MLflow is reported beside the acceptance gate as
   `gate_task.mlflow_delivery` (step 4), and this key makes its state `not_required`: no delivery
   evidence is expected. Health checks still run, and a configured MLflow export still happens.
   A Gym pinned before this skill does not know the key.

## 2. Submit and let the chain run

```bash
gym eval submit --config submit.yaml            # --select a,b submits only those
```

The GYM_JOB_ID is `gym_job_id` in the run directory's `gym-job.json`, or in the record that
`gym eval submit --json` prints. The `Run directory:` line ends with it (`gym-job-<time>-<hex>`)
but wraps when stdout is not a terminal; the `submitted <benchmark> → Slurm job <id>` line gives the
first Slurm job.

A resumable benchmark is a chain of Slurm jobs that continues itself. Before doing work, each job
queues its successor (`afternotok:<this job>`) and a sentinel (`afterany:<successor>`) that takes
over only if the successor dies before its own prologue, then reads its predecessor's final state:

| Predecessor ended | The chain |
| --- | --- |
| `TIMEOUT` | Continues from the saved rows, within `max_walltime`. A rollout in flight when its allocation ends is rerun from the start: the first such cut is free, every later one charges one attempt of that identity as the failure kind `rollout_allocation_ended` (evaluation-side, counted in health; at the attempt cap it is final). The Store cannot tell a walltime cut from a preemption or a node failure, so the rule covers any ended allocation. |
| `PREEMPTED`, `NODE_FAIL`, `BOOT_FAIL` | Retries, spending one of `max_retries`. |
| `COMPLETED` | Ends. |
| `CANCELLED`, `FAILED`, `OUT_OF_MEMORY`, `DEADLINE`, unknown accounting | Stops. |
| `TIMEOUT` before the task inventory was written | Stops and names the fresh-restart command (step 6). |

A job whose driver finishes the run ends the chain itself. After the driver step, it writes the
record the next job would have written (the run complete, or its missing export receipt), cancels
the sentinel and the successor it queued, and journals both, with their ids and the `scancel`
result, in `chain-end-journal.jsonl`. No allocation starts for a finished run.

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

The driver's own exit code tells the chain what happened: 75, unfinished work, the chain continues;
1, no progress in the segment, one infra retry is spent; 76, nothing eligible, a typed stop. Only
`PREEMPTED`, `NODE_FAIL` and `BOOT_FAIL` spend an infra retry. A segment the scheduler requeued
leaves its elapsed time unknown, and the chain stops with unknown accounting (step 5).

Do not resubmit while a job of the run is live; recovery answers NOOP then anyway.

## 3. Where the evidence lives

| Location | What |
| --- | --- |
| Run directory | `gym-job.json` (the submission record), `resolved-config.yaml` (the config as submitted), and the journals `walltime-extension-journal.jsonl`, `repin-journal.jsonl`, `unsafe-resume-journal.jsonl`. |
| Benchmark directory | `job.sh`, `logs/driver-<job id>.log` and `logs/<service>-<job id>.log`, the chain files `.gym_job_chain`, `.gym_accumulated_walltime`, `.gym_infra_retries`, `.gym_infra_retry_extension`, `.gym_followup_queued.<job id>`, `.gym_followup_sentinel.<job id>` and `.gym_chain_incomplete` (why the chain stopped, when it records a reason; while a continuation starts it may hold a provisional line, "Continuation <job> started and had not queued its follow-up ...", which is removed once the follow-up is queued: act on it only if it is still there after the segment ended), and the journals `retry-extension-journal.jsonl` (one line per `--extend-retries` or `--extend-attempts`, with its `kind`), `spinup-timeout-journal.jsonl`, `operator-continuation-journal.jsonl` and `chain-end-journal.jsonl`. |
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

On a machine that did not submit the run, the inspect fails with `No submission '<id>' in this
machine's job index`. Add `--record <run dir>/gym-job.json` (inspect only; copy the file where the
run directory is not mounted): it reads the run's own submission record without writing the local
index. A resubmit that dispatches needs the index entry: where the run directory is mounted but the
index lost it (a recycled machine), add `--run-dir <run dir>` and the entry is restored from the run's
`gym-job.json`, checked to be that submission's, and journaled in the index's `index-restores.jsonl`
(this writes the index, also with `--inspect`).

`--inspect` reads the saved evidence and evaluates the gates without dispatching anything. The
read runs on the run's host; the first time, it installs the run's pinned Gym for its reader with
a private copy of uv and Python, which takes minutes and replaces nothing in your home. Later reads
on that host reuse the cache. An inspection and a recovery decide the same from the same evidence,
and the read hashes its sources when it opens, parses and closes them: a file that changes under
it makes the decision stop ("unstable sources"), so never edit a run while you read it.

The read runs inside the run's pinned Gym, so a run pinned before a recovery rule existed does not
have that rule: its inspect decides by the older code. To read it by a newer Gym, add
`--repin <full commit of the Gym you run the command from> --repin-reason "<why>"` to the inspect.
It reads the Store at that commit, previews the repinned plan and dispatches nothing; dispatch it
with the same options without `--inspect`. The repin target must be the Gym that runs the command.

Where readers run in Slurm steps (`NEMO_GYM_REMOTE_READER_SRUN`), an inspect that could not read
returns `UNKNOWN` with "nothing was read or decided". That is not a problem of the run: read the
reason on stderr (`gym: the reader did not run: ...`). When it says that reader steps are already
running or pending, the per-user limit is reached: raise `NEMO_GYM_REMOTE_READER_MAX` (default 8)
and run the same command again. Other reasons (the step did not
start within its wait) have their own remedy.

The output is one JSON line, the recovery outcome, and nothing is written; the exit code is 0 only
when `gate_task.status` is `SUCCESS`. Keep that line: it is your record of the read. A dispatching call (`--recover` without `--inspect`)
that carries a score, MLflow or health-limit option exits 1 even when it succeeds, because the gate
cannot be `SUCCESS` right after a dispatch; without one it exits 0. Read `ok` and `dispatched`, not
the exit code. A call handed to a queue can also exit 0 while the gate still reads `RESUMING`:
take `ok` and `job_id` from the JSON line, not from the exit code. Its top-level
keys are `decision` (`action`, `reason`), `gate_task`, `ok`, `dispatched`, `job_id`, `unsupported`,
`mlflow` (`receipt`: `run_id`, `current`, `uploaded_rollouts`, `rollout_projection_version`),
`reverify`, `judge_job`, `walltime_plan`, `scheduler_refusal` and `reader_token`. Read:

- the decision's `action` (`noop`, `reverify`, `resubmit`, `fresh-restart`, `stop`) and `reason`.
  When the run's latest resume crossed operational configuration drift, the reason ends with
  `resumed across operational drift: <settings>`.
- `gate_task.status`: `RUNNING` (a job is live), `PLANNED` (eligible work, nothing dispatched; a
  dispatching call reports `RESUMING`), `FAILED` (stopped, unsupported, or something failed: read
  `errors`), `UNVERIFIED` (complete, nothing failed, acceptance not proven: read `warnings`),
  `SUCCESS` (complete and every gate passed). A chain error turns `SUCCESS` or
  `UNVERIFIED` into `FAILED`; MLflow delivery never does.
- `gate_task.evaluation_health` (null until the run is complete): `status` (`PASS`, `FAIL`,
  `UNVERIFIED`), `evaluation_failure_rate`, `exact`, `trials`, `evaluation` (`count`, `fraction`,
  `kinds`, `server`), `model` (`count`, `fraction`, `kinds`), `policy_timeout_counts_as_model`
  (present only when false) and `trajectory_evidence`.
- `gate_task.failures`: `expected`, `omitted`, `response_failures` and `trial_failures` (each with
  `count`, `rate` and `by_cause`, a `count` and `rate` per cause: `truncated`, `empty`, `malformed`
  for responses; `request`, `judge`, `sandbox`, `harness`, `timeout`, `other` for trials) and
  `metric_coverage` (`trials`, `expected`, `source`). Facts; no limit is applied to them.
- `gate_task.trial_counts` (null until complete): `observed_healthy`, `unhealthy`, `unobserved`,
  `verifier_recorded` (the observed-healthy trials known only through the verifier's own grading
  record) and `unobserved_reason` when `unobserved` is not 0. Unobserved trials are unverified,
  not healthy.
- `gate_task.mlflow_delivery`: `state` and `reasons`. `not_required` (`mlflow_tracking_required` is
  false), `verified` (no delivery problem), `pending` (no run evidence yet, or only the run's
  status unread) or `unverified` (with its reasons). It is a fact, not a verdict: it changes
  neither the status nor evaluation health, and `pending` and `unverified` add the warning
  `MLflow delivery <state>: <reasons>`. A run that stated no `tracking_required` is read from its
  saved config.
- `gate_task.health_verdict` (one line), `criteria`, `mlflow_tracking_required`,
  `expected_samples`, `scored_samples`, `omitted_samples`, `score`, `errors`, `warnings`, and where
  the read ran (`reader_host`, `collector_host`, each with a `*_fallback_reason` when a reader
  stayed on the login host).

## 5. Act on the decision

| The inspection says | Do |
| --- | --- |
| Eligible work, `PLANNED` | Run the same command without `--inspect`, with the variables the saved config forwards (`compute.forward_env`, `HF_TOKEN` for instance) set in that shell; a missing one stops the dispatch. |
| `reverify`, `PLANNED` (only rows with a saved answer remain: a judge failure, or a verify call lost in transport) | Run the same command without `--inspect`: the saved answers are judged again and the policy is not called. A run that served its own model, and nothing else beside it, gets a judge-only job on one CPU node. Its partition is the first of: `--judge-partition <partition>`, a node pool of the saved config with `gpus_per_node: 0`, or the partition that `NEMO_GYM_REMOTE_READER_SRUN` names. The job also needs a QOS that matches that partition: `--judge-qos <qos>`, else the QOS of the saved CPU pool, else the `-q` or `--qos` option of `NEMO_GYM_REMOTE_READER_SRUN` when it names this partition or none, else the partition's default. Slurm refuses a GPU QOS on a CPU partition (`QOSMinGRES`); pass `--judge-qos` then. On a site where the saved config has no CPU pool and no reader placement is set, pass `--judge-partition` with the resubmit, or the recovery stops with nothing queued (next row). The judge-only job runs `gym eval reverify --settle-receipt`: the Gym that runs it must define that option, or the queued job stops at argument parsing. A run that serves its judge on a node pool of its own gets a judge-only job on that saved pool instead: the judge's services start on their saved pools and the job keeps the run's QOS; `--judge-partition` and `--judge-qos` are not used (when you pass them, the report says `--judge-partition/--judge-qos apply to the CPU judge profile only; not used`). A run with both judge failures and policy work is resubmitted; the resumed run re-judges the saved judge failures before its policy work, while its judge servers are up, and again at the end of the run. |
| STOP on a `reverify` of a GPU-served run, `... has no existing CPU-only judge execution profile and none can be derived: no CPU partition is known (... no --judge-partition was given)` | Nothing was queued and no row was lost. Run the same command again with `--recover --judge-partition <cpu partition>` (and `--judge-qos <qos>` when Slurm refuses the job with `QOSMinGRES`), a partition of your site that runs CPU jobs. If the message says instead `this submission has no served-judge execution profile: service(s) <names> run on the policy's nodes (its node pool, or no node_pool of their own, which is the whole allocation)`, no partition helps: the judge-only job does not start the policy's allocation and policy services are not restarted. Give each of those services a node pool of its own, beside the policy's, in the submission config, or report that the saved answers cannot be re-judged. |
| NOOP, `RUNNING` (`continuation <job> live; nothing to decide`) | A job of the run is live. Wait for it to end, then inspect again. |
| STOP typed `server_spinup_timeout` | `--recover --server-spinup-timeout-seconds N --spinup-timeout-reason "<why>"` (the table in step 2). |
| The chain's walltime cap is spent | `--recover --max-walltime HH:MM:SS --walltime-reason "<why>"`. Never below the recorded walltime; journaled. |
| The infra-retry budget is spent | `--recover --extend-retries N`. The spent count is kept; journaled. It raises infra retries only: it does not raise the per-identity attempt cap, so a STOP that says identities used every attempt is not a lost extension (see "When a run stops"). |
| Identities used every attempt (`exhausted`, `max_rollout_attempts`) | `--recover --extend-attempts N` gives each of them N more attempts. Only rollouts the Store would run again are reopened (infrastructure failures, allocation cuts); a model failure or a saved result never is. The spent attempts are kept; recorded in `artifacts/rollouts_attempt_extension` and journaled with the infra extension. A run pinned to a Gym without it needs `--repin` in the same resubmit. |
| The run's chain accounting is unknown (a segment the scheduler requeued ends this way: the requeued allocation's elapsed time is unknown) | An explicit budget continues it: `--max-walltime` with `--walltime-reason`, optionally `--extend-retries`. Without it, the STOP names these options. |
| A Gym fix is needed to continue | Install that commit as the launcher, then `--recover --repin <full sha> --repin-reason "<why>"`. Saved rows and budgets stay. |
| The resume check refuses a configuration difference | Operational differences (timeouts, retries, ports, concurrency, paths, tracking) already resume by themselves, with a warning and a line in the resume-drift journal. A **semantic** difference (data, limits, repeats, model, sampling, judge or grading) is refused, naming the settings. Usually the right answer is a fresh run (step 10, item 3). Only when the operator accepts that exact difference: `--recover --allow-unsafe-resume --unsafe-resume-reason "<why>"`, which records the reason and the user. It never bypasses corrupt or conflicting history. |
| `fresh-restart` (no task inventory) | Step 6. |
| NOOP, `SUCCESS` or `UNVERIFIED`, and `gate_task.mlflow_delivery.state` is `pending` or `unverified` (the warning reads `MLflow delivery <state>: <reasons>`) | The results are complete and the status is whatever they decide: delivery is reported beside the gate, never inside it. Close the delivery: run `gym eval export mlflow --rollouts <run dir>/<benchmark>/artifacts/rollouts.jsonl +mlflow_tracking_uri=<uri> +mlflow_experiment_name=<name> +mlflow_run_name=<name>` with the current Gym, the three values taken from the run's `artifacts/rollouts_resolved_config.yaml` (the export does not read that file itself), the tracking token given as `+mlflow_tracking_token=<token>` or in the `MLFLOW_TRACKING_TOKEN` environment variable, and `--rollouts PATH` the same as `+rollouts_jsonl_fpath=PATH` (a repeat on an exported run verifies without touching the run lock); `--recover` on a complete run that requires tracking and has no receipt runs the export too. The reasons say which case it is. "although the saved config snapshot names a tracking URI, experiment and run name": tracking was configured and required, and the export did not finish. Read the exporter's failure in the driver log, repair the delivery, then export. "does not name all of the tracking URI, experiment and run name": the saved config names no complete tracking target, so nothing tracked the run; set `mlflow_tracking_required: false` next time (step 1). The inspection asks the tracking server the saved config names for the status of the receipt's run (up to about 10 seconds, one retry); from a host without access it reports `unverified` with `status query failed: ...`, so inspect with `--mlflow-run-id <the receipt's run_id> --mlflow-status FINISHED` once you observed that status yourself (the run's page, or `GET <tracking uri>/api/2.0/mlflow/runs/get?run_id=<id>`, field `info.status`); a `status query failed: ModuleNotFoundError` means the reader's environment lacks the `mlflow` package, not that the host has no access. Passing the two flags avoids the call. |
| `SUCCESS` | Done. Keep the run directory as the record. Cancel the last segment's pending sentinel (`scancel` the id in `.gym_followup_sentinel.<job id>`): once Slurm cancels the successor, the sentinel starts only to exit, on a full allocation (step 10). |

Plain `gym eval resubmit` without `--recover` resets the retry and walltime budgets and the
per-rollout attempt allowance. It is not a substitute for a recovery STOP.

## When a run stops: budgets and resubmit

- The chain runs one Slurm segment after another until the work is done, the chain cap
  (`max_walltime`, the walltime summed over segments) is spent, or the infra-retry budget
  (`max_retries`) is spent. A typed stop ends it with its kind in `.gym_chain_incomplete`. At a
  stop nothing is lost: every finished row is saved.
- Look first, then act. `gym eval resubmit <id> <benchmark> --recover --inspect` says what a
  recovery would do and why, and dispatches nothing. The same command without `--inspect` does it.
- Three limits, each counting something different. The chain cap is raised with
  `--max-walltime HH:MM:SS --walltime-reason "<why>"`, never below the recorded walltime. The
  job's infra-retry budget (preemptions, node failures) is widened with `--extend-retries N`; the
  spent count is kept. `--max-walltime` takes `--walltime-reason`; `--extend-retries` and
  `--extend-attempts` take no reason flag, and their journal line records the count and its kind, so
  put the reason in your report. The third limit belongs to
  each task row, not to the job: the attempt cap (`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`, default 3),
  and `--extend-retries` does not raise it. The variable is read from the environment of the job: the
  job script does not set it, so a submitted run gets the value in the environment of the process that
  runs `sbatch`, where Slurm exports it (its default), and the manifest records it as `max_rollout_attempts`.
- A cut that ends an allocation while a row is in flight charges that row an attempt, except the
  first cut, so a long task that spans several segments can reach the cap without any model
  failure. A row at the cap is masked: the stop reads "N identity(ies) used every attempt since the
  last resubmit" and says that `--extend-retries` raises infra retries only, not the per-identity
  attempt cap (`max_rollout_attempts=N`). It is never scored as the model's result, and a resubmit
  does not revive it. Only `--recover --extend-attempts N` does (below); a pin without that option
  needs `--repin` in the same resubmit.
- A resume keeps every finished row and runs only the missing and retriable ones. A rollout in
  flight when an allocation ends starts again from its beginning: the first such cut is free, and
  each later one charges one attempt of that row.
- `--recover --extend-attempts N` raises the per-identity attempt cap for the rows
  that spent their attempts on infrastructure or on allocation cuts, never for rows that ended on
  a model failure. Like `--extend-retries`, it is journaled like it, with its `kind`, and keeps the spent
  counts. Inspect first: it names the rows it would reopen.
- A settled result is final and none of this reopens it: a scored zero, including a typed answer
  failure (a truncated, empty or malformed answer), and a clean agent timeout. A saved failure that
  kept no answer is retried by the recovery policy's default action, within the attempt caps,
  unless its kind or HTTP status is listed otherwise (`judge_failed` is re-judged; step 7); a failure
  the Environment Server marked terminal (a typed model failure such as `model_context_limit`) is
  never retried.
- When the launcher's Gym revision differs from the run's pin, the recovery needs
  `--repin <full sha> --repin-reason "<why>"` (step 5).
- If a resume is refused because the saved configuration fingerprint no longer matches, do not set
  `allow_unsafe_resume` yourself: continue the run at its own pinned version and report it.
  A resume on a newer Gym than the one that saved the run can be refused after setup, which costs
  an allocation: `gym eval submit --resolve-only` does not compare the saved identity. Continue at
  the run's own pin and report it.
- Never use plain `gym eval resubmit` without `--recover` to continue: it resets the budgets.

For example, a long-task benchmark whose 16-hour chain cap was spent with rows missing is
continued with `--max-walltime 24:00:00 --walltime-reason "<why>"`, and a run preempted until its
retry budget was spent is continued with `--extend-retries 1`. Both are journaled.

## 6. When nothing can be recovered

A segment that ended before the task inventory was written has nothing to resume. Recovery answers
`fresh-restart` and names the command:

```bash
gym eval resubmit "$GYM_JOB_ID" <benchmark> --recover \
  --max-walltime HH:MM:SS --walltime-reason "<why the longer walltime>"
```

`--max-walltime` must be longer than the recorded walltime. The run starts fresh with fresh budgets
in the same directory; the saved files are kept, and the plan is journaled.

## 7. Retriable and final

A saved failure is retriable or final, and the run's recovery outcome policy decides which, one
action per failure: `retry` (retriable: another attempt, within the identity's attempt cap and the
run's infra-retry cap), `verify` (re-judge the saved answer without calling the policy) or `fail`
(final and ungraded: recovery stops for an owner decision). The action comes from the failure's
HTTP status, else its failure kind, else the default. The defaults: `judge_failed` is re-judged,
`collection_dispatch_failed` and `harness_termination_unconfirmed` are retried, and every other
saved failure is retried too (`default_action: retry`), within the caps. Settled results are final
and never reopened: a scored answer or zero, and a clean agent timeout, which is the model's
result under the benchmark's time limit (a harness that records its agent deadline as a graded
zero has settled it). That limit is the benchmark's own: for Terminal-Bench 4.0, 8 h per task is
the official agent budget (TB 4.0 release notes, "Calibrating resources"), and Gym's default
`agent_timeout_sec` of 28800 matches it. An `agent_timeout` that kept no answer is a saved failure
the policy decides, retried by default. Successes and final failures are never run again. For
GDP-PDF, a policy answer that comes back incomplete because its output budget was spent is the
row's result, graded as a truncated answer and not regenerated; the HTTP 4xx input-limit
negotiation (the DPI step-down to 72, the image cap, the text-only fallback) is the benchmark's
delivery protocol, counted per row (`document_delivery.input_limit_rejections`,
`output_truncation_regenerations`) and in the run facts; `regenerate_on_incomplete_output=true`
reproduces the old main behaviour, for comparison only, so its cells differ from main by design.
A benchmark agent may also recover from an output cut inside the same episode, as part of its own
protocol: the Stirrup agent's `truncation_recovery` setting follows a turn that spent its whole
completion budget without a valid tool call with one more turn that has thinking off and a
one-time instruction to act now, not recorded in the trajectory. That is the
agent's protocol, not a retry of a model failure by recovery; regenerating an incomplete answer
from scratch after a model-side failure stays excluded.
`--inspect` lists the identities in each state and the policy in force (`outcome_policy`, with its
`source`: `default`, `config` or `python`). When the budget is spent, or the cluster was down, run
`gym eval resubmit <id> <benchmark> --recover` again after `--inspect`; running the same command
does not extend an exhausted budget, `--extend-retries`, `--extend-attempts` and `--max-walltime` do.

Retries happen in three layers before a failure reaches you. In a call: the policy client retries
HTTP 429, 502, 503, 504, 520, 404, 408 and 500, the judge client 408, 429, any 5xx, a timeout or a lost connection (three attempts in all), and the sandbox
client 408, 409, 425, 429 and the 5xx codes. In a run: a resumed segment dispatches again every
identity that is not settled, within the attempt cap (`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`, default 3);
a judge failure saved as a failure (`judge_failed`), or a verify call lost in transport, is re-verified from
its saved answer, never regenerated, and is never the model's zero: masked, out of the score, counted on the evaluation
side. A row whose judge returned no usable verdict and that was saved as a scored 0.0 with no
failure record (`judge_unparseable` in the health counts) is not that either: Gym plans a re-judge for it. A run
whose only open work is such rows inspects `reverify`, `PLANNED` and exits 1 until the re-judge settles
it (any status but `SUCCESS` exits 1). Meanwhile evaluation health still counts the row as an
evaluation-side fault ("judge N of M", the 2% limit is unchanged), `gate_task.scored_samples` leaves it out,
`errors` lists the judge failure awaiting re-judgment, and the MLflow export waits for the run to complete. Run
`gym eval resubmit <id> <benchmark> --recover` without `--inspect`: the saved answer is judged again and the model is not called. A judge
served on a node pool of its own is re-judged by a judge-only job on that pool; one served on the policy's own nodes still cannot be (step 5). Report it with the health count. In the chain: only `PREEMPTED`,
`NODE_FAIL` and `BOOT_FAIL` spend an infra retry, `TIMEOUT` continues, and `FAILED`,
`OUT_OF_MEMORY`, `DEADLINE` and `CANCELLED` stop the chain; `--recover` is the way on, and it
dispatches only the missing and retriable identities.

A request the provider rejects for an unsupported parameter value is an evaluation-side fault: it
invalidates the run and is never the model's zero. Fix the parameter and run again.

What is retriable is one question; whether the score can be trusted is another. The split between
model faults and evaluation-side faults belongs to the second only: `gate_task.evaluation_health`
(step 9) counts the evaluation-side faults against the fixed 2% limit, reports the model faults by
kind and never fails on them. The outcome policy never changes evaluation health, and evaluation
health never changes what is retried.

A benchmark config changes what is retriable with `recovery_outcome_policy` under the benchmark's
`resumable`, in its mapping form beside `max_retries` and `max_walltime`:

```yaml
resumable:
  max_retries: 3
  max_walltime: "16:00:00"
  recovery_outcome_policy:
    by_failure_kind: {agent_timeout: fail}
    by_status_code: {429: retry}
    default_action: retry
```

Entries it does not name keep their defaults. Kinds must be registered (or `unknown`), statuses
400 to 599, `count_as_zero` is refused and so is an unknown key; a mistake fails at config load
with `invalid recovery_outcome_policy: <field>: <problem>`. The run saves the policy in its
`resolved-config.yaml`, the chain's own continuations use it, and a change between the inspection
and the dispatch refuses the resubmit, because the policy is a semantic setting. The CLI extends
budgets and has no flag for the policy (extending a budget is not choosing which failures are
retriable); a Python caller can pass `outcome_policy` to `recover(...)`, which replaces the config's
section as a whole:

```python
from nemo_gym.agents import OutcomeAction
from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig
from nemo_gym.orchestration.submit import recover

recover(job_id, benchmark, outcome_policy=RecoveryOutcomePolicyConfig(by_status_code={401: OutcomeAction.FAIL}))
```

Resubmit by hand after a cluster outage, a crashed server, an expired credential, or a budget stop.
The chain's own stop text (in the driver log and `.gym_chain_incomplete`) and the inspection's
reason say which one it was.

| The product decides | The operator decides |
| --- | --- |
| Retrying the failures the outcome policy marks retriable, within the retry budget. | Raising the walltime cap or the retry budget, with a reason; what the benchmark config marks retriable (`recovery_outcome_policy`), before the run. |
| That settled results are final: a scored zero, a saved answer, a clean agent timeout; they are scored under the benchmark's protocol, not retried. | Moving a run to another Gym commit (`--repin`). |
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
model failures by default: a policy-model response timeout is the model's fault unless the
inspection or recovery call sets `--policy-timeout-counts-as-model false`
(`policy_timeout_counts_as_model`, for a self-hosted evaluation that owns its serving; it moves the
timeout to the evaluation side of health and changes nothing else). Optimize the deployment; do not
raise the evaluation's limits.

## 9. Health and acceptance

The health summary and the gates are the contract you read after a run: the response-failure and
trial-failure rates, coverage, and the unhealthy rollouts. A run can be healthy and still score
low. That is the model or its deployment: fix the deployment and run again, not the evaluation.

- The verdict is not a file: it is the inspection's output line (step 4). The one persisted health
  record is `_ng_run_health_metrics` in the first entry of `artifacts/rollouts_aggregate_metrics.json`,
  the same facts the export sends to MLflow. `quality_summary.json` is the per-rollout health
  summary and one of the gate's hashed sources; `gym eval health-check <copy>/artifacts` re-runs
  its checks and rewrites it, so run it on a copy, never on the run's own `artifacts/` (step 3).
- What decides acceptance: complete results, the score gate (`--score-key`, a JSON path, with
  `--score-agent` and, when an agent has several entries, `--score-environment`), a readable
  health summary, and evaluation health.
  Evaluation health is `gate_task.evaluation_health`: `PASS` when evaluation-side trial failures
  (judge, verifier, sandbox and harness faults, a server that did not start, transport faults and
  model-server 5xx replies) are strictly below 2% of the expected trials and every trial's
  grading is traceable; `FAIL` at or above 2%; `UNVERIFIED` below 2% when some grading is not
  traceable, so the known count is a lower bound. The 2% is fixed in Gym; no flag changes it.
  The classes are Gym's shared lists in `nemo_gym/core/failure_kinds.py`: `MODEL_FAULT_KINDS`
  (`model_request_error`, the policy's own 4xx; `model_policy_response_timeout`;
  `model_context_limit`; `agent_timeout`; `agent_command_not_allowed`; `agent_limit`;
  `agent_invalid_output`; `truncated_answer`; `empty_answer`; `malformed_answer`) and
  `SERVER_FAULT_KINDS` (`server_start_failed`, `model_transport_error`, `model_server_error`,
  `agent_timeout_infrastructure`),
  with every `provider_*` kind, any 5xx and another model's 4xx (`non_policy_model_rejection`)
  on the evaluation side. `criteria.evaluation` records the limit and
  `policy_timeout_counts_as_model` beside the verdict.
- Whether a trial counts as observed depends on the benchmark: its server recorded the grading
  operation, or saved a grading artifact bound to the answer (`judge_record`, `rubric_verdict`,
  `symbolic_match`, `metric_record`, `test_execution`, `environment_reward`, `repository_test_run`,
  `substep_tests`), or the model's own failed answer settles it. Most servers grade through the
  shared verify wrapper, which records the grading for them. A verifier that calls a judge without
  recording it and saves no recognised artifact is unobserved by construction, and its runs read
  `UNVERIFIED` with `trial_counts.unobserved_reason` naming the server; so does a run whose saved
  config does not name its resources server. At this head that is the case for `aegis_v4_safety`,
  `agentif`, `beam`, `deepsearchqa`, `finance_agent_v2`, `finance_sec_search`, `gdp_pdf`,
  `imo_proofbench`, `longmemeval`, `simpleqa`, `widesearch`, `litmus-bench`, the `abstention`,
  `inverse_if` and `multichallenge` parts of `nemotron_3.5_super`, `frontierscience_olympiad` and
  `automationbench` (the benchmarks graded by the verifiers agent, AutomationBench among them).
  `UNVERIFIED` means the answers were scored, but the grading evidence was not recorded on this
  stack version. A resubmit cannot change it, and no flag does: it needs a run on a stack version
  that records grading. Report it with the reason and do not present the run as verified.
- Evaluation health is the gate on the collected results. Delivery to MLflow is reported separately
  as `mlflow_delivery` (pending until the export receipt exists, then verified or unverified with
  reasons) and does not fail the gate. A run with `mlflow_tracking_required: false` needs no
  delivery evidence. To close a pending delivery on a finished run, run `gym eval export mlflow --rollouts
  <run dir>/<benchmark>/artifacts/rollouts.jsonl` with the current Gym and the run's saved tracking
  settings (step 5).
  The fix is that server's grading record, in Gym.
- What is reported and not applied: `--max-response-failure-fraction` and
  `--max-trial-failure-fraction` are validated and echoed under `gate_task.criteria` with
  `applied: false`; `--max-unhealthy-fraction` (per-rollout health) is judged in `warnings`. None
  of them changes the verdict. `gate_task.failures` lists the response failures (truncated, empty,
  malformed) and the trial failures by cause (`request`, `judge`, `sandbox`, `harness`, `timeout`,
  `other`), with their rates: facts for you to judge, not verdicts. Model failures count in the
  score and never fail health.
- `warnings` also carries facts that fail nothing: `no response-health criterion configured` when
  `--max-unhealthy-fraction` was not given, and `health checks not fully evaluated` with the checks
  a small run gave too little to judge.
- MLflow delivery is `verified` only with a current export receipt (`mlflow.receipt.current`, with
  `uploaded_rollouts` true and `rollout_projection_version` 1) and a `FINISHED` status of the
  receipt's run, read from the tracking server the saved config names, or given by you with
  `--mlflow-status FINISHED` for `--mlflow-run-id` equal to the receipt's `run_id`. A live run id
  alone is never accepted, and the two flags export nothing. The receipt, `artifacts/rollouts_mlflow_export_done`,
  binds the sha256 of five run files; a file that changes after the export makes it stale.
- With MLflow configured, the post-run export logs the same facts as metrics: `health/response/*`,
  `health/trial/*` (`known`, `passed`, `counts_known`, `expected`, `failed` or
  `failed_lower_bound`, the rates), `health/response_failures/*` and `health/trial_failures/*` (a
  rate and a count per cause), `health/metric_coverage/*`, `health/upstream_recovery/*` and
  `health/grading_recovery/*`. Read `known` before a `passed` value: `known=0` means unknown, not
  failed. The MLflow trial metrics always count a policy timeout as the model's, whatever
  `--policy-timeout-counts-as-model` the inspection used.
- `SUCCESS` needs complete results and every gate. A NOOP, a finished job, or a full score alone is
  not acceptance.

## 10. Stopping a chain

1. Cancel the running job of the chain and the jobs it queued (`scancel <ids>`): its successor
   (in `.gym_followup_queued.<job id>`) and the successor's sentinel (in
   `.gym_followup_sentinel.<job id>`). A successor that is left to start sees `CANCELLED` and stops
   without doing work, but only after it got an allocation. A job that finished the run cancels
   both itself (see `chain-end-journal.jsonl`); a `job.sh` rendered before that did not, so after
   such a run cancel a pending sentinel too: once Slurm cancels the successor, the sentinel's
   dependency is met. A continuation job also has the sentinel its predecessor queued for it
   (`afterany:<this job>`), which becomes eligible when you cancel the job: cancel it too
   (`squeue --me -o '%i %j %T %E'` shows it by its dependency). If you
   cancel only the running job, its follow-up starts when it gets an allocation, sees `CANCELLED`,
   writes the stop to `.gym_chain_incomplete` and exits; cancelling the successor and the sentinel
   too saves that allocation and leaves no stop record (the inspection reads the state from the
   scheduler either way).
2. Then inspect (step 4) before deciding anything else.
3. Keep the run directory. To evaluate something different, submit a new run to a new location
   rather than reusing this one; a fresh `gym eval run` at an existing output path replaces it.

## Exit checks

- [ ] `--inspect` shows `SUCCESS`, or the remaining STOP is explained and accepted by the operator.
- [ ] Every operator action (walltime, retries, repin, unsafe resume) has a reason in its journal or, for `--extend-retries` and `--extend-attempts`, in your report.
- [ ] `trial_counts` and `quality_summary.json` were read: unobserved and unhealthy counts are
  explained.
- [ ] Any `resumed across operational drift` settings are listed with the result.
- [ ] The run is final on its receipts: `rollouts_coverage.json` says complete, the export
  receipt `rollouts_mlflow_export_done` is current and `gate_task.mlflow_delivery.state` is
  `verified` (or `not_required`, with `mlflow_tracking_required: false` set on purpose), and
  `.gym_chain_incomplete` is absent or explained.
