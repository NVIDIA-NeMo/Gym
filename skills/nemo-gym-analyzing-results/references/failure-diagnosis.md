# Why Did Tasks Fail?

Use this reference to explain which tasks fail, why, and what to change, for one run or for the tasks that changed between two comparable runs. It follows the BLADE method (Benchmark Level Analysis and Diagnostics Engine): counts first, then reading rollouts, then a cause for each task, backed by evidence.

## Order of Work

1. Confirm coverage (step 2 of `SKILL.md`). Failed, masked and missing rollouts are outcomes too; list them separately instead of dropping them.
2. Sort tasks into outcome groups across repeats.
3. Count where rollouts stop in the benchmark's workflow, and find the largest drop.
4. Read tasks that sometimes pass: compare a passing and a failing repeat of the same task.
5. Read tasks that never pass: check the verifier, the label and the output format before blaming the model.
6. Give each failing task one root-cause label with evidence.
7. Map labels to fixes, and measure a fix before rerunning everything.
8. Write the report.

## Outcome Groups

This sorts every planned task by its measured rewards and lists the repeats of tasks whose reward varies, so passing and failing repeats can be read side by side. It assumes the run wrote `results/rollouts.jsonl`:

```bash
python3 - results/rollouts.jsonl <<'EOF'
import collections, json, sys
from pathlib import Path
out = Path(sys.argv[1])
def rows(p):
    if not p.exists():
        sys.exit(f"missing {p}: point this at the run's rollouts file, next to its materialized inputs")
    result = []
    with p.open("rb") as f:
        for line in f:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                result.append(row)
    return result
planned = collections.Counter(r.get("_ng_task_index") for r in rows(out.with_name(out.stem + "_materialized_inputs.jsonl")))
measured = collections.defaultdict(dict)
for r in rows(out):
    if isinstance(r.get("reward"), (int, float)) and not r.get("mask_sample"):
        measured[r.get("_ng_task_index")][r.get("_ng_rollout_index")] = r["reward"]
groups = collections.defaultdict(list)
for task in sorted(planned.keys() | measured.keys(), key=lambda t: (t is None, t)):
    got = measured.get(task, {})
    rewards = list(got.values())
    label = f"{task}{'*' if len(got) < planned.get(task, 0) else ''}"
    if not rewards:
        groups["not measured"].append(label)
    elif min(rewards) == max(rewards):
        groups[f"same reward {rewards[0]} on every measured repeat"].append(label)
    else:
        groups["reward varies"].append(label)
        print(f"task {task}: " + ", ".join(f"r{k}={v}" for k, v in sorted(got.items(), key=lambda kv: (kv[0] is None, kv[0]))))
for name, tasks in groups.items():
    print(f"{name} ({len(tasks)}): {', '.join(tasks)}")
print("* = fewer measured repeats than planned")
EOF
```

For rewards of 0 or 1, "same reward 1" is always pass, "same reward 0" is never pass and "reward varies" is sometimes pass. For fractional rewards, a task with the same middle reward every time is not a task that sometimes passes. `gym eval profile --inputs <stem>_materialized_inputs.jsonl --rollouts <stem>.jsonl` writes the same per-task statistics to `<stem>_reward_profiling.jsonl` in the run folder; add `++allow_partial_rollouts=True` when rollouts failed or are missing, or it stops with `Missing rollout results`.

Tasks that sometimes pass are the most useful: the same task succeeded under some conditions and failed under others, so the difference between its repeats points at the cause.

## Where Rollouts Stop

Define the stages a rollout passes through for this benchmark, in order, and count how many rollouts reach each one. The largest drop is the first thing to explain.

- A single answer (multiple choice, math, short QA): a response was produced → it ended normally rather than at the token limit → the final message has text (not only reasoning) → the verifier extracted an answer → the answer is correct.
- An agent task (tools, code, terminal): the run started → the agent used its tools → it produced a submission or final answer → the verifier or tests ran to completion → they passed.

Fields to read in a rollout row of `<stem>.jsonl`:

- `response.output`: the items the model produced, in order: `reasoning`, `message` (its text is the final answer) and tool calls.
- `response.status` and `response.incomplete_details`: `incomplete` with a reason such as `max_output_tokens` means the output was cut off.
- `response.metadata.ng_termination_reason`: Gym's simple agent marks a response that ended with no final message and no tool call: `incomplete_reasoning` (reasoning only) or `empty_output`. A run stopped by a step limit carries no such mark; compare its step count with the limit.
- `response.usage`: output tokens next to the request's limit in `responses_create_params` or the materialized inputs.
- The verifier's fields next to `reward`, for example the extracted and expected answer, a judge verdict, test results, or `evaluation_completed`.
- `<run_dir>/rollout_verdicts.jsonl`: health findings for the same `_ng_task_index` and `_ng_rollout_index`, such as `model_call_runaway_generation` (cut off at the token limit with no answer) or `rollout_ended_on_failed_model_call`.

## Rules for Evidence

- Tie every claim to evidence: task and rollout ids (`_ng_task_index`, `_ng_rollout_index`), the relevant output text, verifier fields, a failure class or a health finding.
- Read each rollout in order. An error that appears before the model's consequential action was not caused by that action; ask instead whether the model failed to retry, check or recover.
- Separate the model's own checks from the final verification. Passing its own test is not passing the benchmark.
- Treat failed, masked and missing rollouts, timeouts and malformed outputs as outcomes, reported in their own counts.
- Say when evidence is missing or redacted. Do not fill gaps from memory.
- Before sharing a report, remove private endpoints, credentials, user names, unreleased benchmark names and raw data that is not cleared for release.

## Signs and Likely Causes

| What the rollouts show | Likely cause | Check |
|---|---|---|
| Output cut off at the token limit with no final answer | The limit is too low for this model's reasoning | Output tokens equal to the limit; `response.status` `incomplete`; the limit in the config |
| No final message, and the answer is only in the reasoning | Reasoning parser or chat template, or the model stops without a final message | `reasoning` items hold the answer; no `message` item, or an empty one; `ng_termination_reason` `incomplete_reasoning` |
| The right answer in a format the verifier does not accept | Answer extraction is stricter than the prompt or the model's habit | The extracted answer is empty or differs from what the text says; compare with the expected answer |
| The same wrong answer on every repeat | A knowledge gap, or a wrong label | Check the expected answer before calling it a model problem |
| Different wrong answers across repeats, with occasional passes | Unreliable knowledge | Passing and failing repeats differ in the answer, not in format or tooling |
| Identical outputs with different rewards | A nondeterministic verifier or environment | Compare rewards of repeated outputs |
| Failure rows with HTTP 429 or 5xx, sandbox errors, timeouts | Infrastructure | `_ng_failure_class` and `_ng_failure_message` (or `failure_reason`) in `<stem>_failures.jsonl`; `run.stats` in `<run_dir>/quality_summary.json` counts errors of saved rollouts only |
| Tool errors, loops, giving up early, ignoring feedback | Agent behavior | Tool calls and their outputs, in order |

## Root-Cause Labels

Use one primary label per failing or mixed task. Use a combined label such as `BI+KG` only when the evidence needs both, not to avoid making a call.

| Label | Meaning | Typical fix |
|---|---|---|
| `KG` knowledge gap | The model lacks the domain, API, tool or task knowledge | Targeted examples or training data, better task documentation |
| `UK` unreliable knowledge | Some repeats show the knowledge and others do not | Repeated-rollout training or RL, self-checking, data from passing repeats |
| `BI` behavior issue | The model could do it but skips steps, gives up, loops, ignores feedback, or puts the answer where it is not read | Change the agent loop or prompt, add checks, reward intermediate verification |
| `TI` task, verifier or setup issue | The task, expected answer, verifier, token limit, timeout or harness setting is wrong for the task | Fix the task, verifier or setting, re-score or rerun, and report scores from before and after the fix separately |
| `IR` infrastructure | Endpoint, sandbox, scheduler, network or storage failures | Fix the infrastructure, rerun the affected rollouts, keep infrastructure failures out of model quality |
| `DA` data artifact | Duplicated or mislabeled tasks, leaked answers, missing files | Fix the data and regenerate before comparing models |

A task that never passes is not automatically a knowledge gap, and an early failure is not automatically a behavior issue. Labels explain a mechanism, not a symptom.

## Measuring a Fix

Recommend how to measure each fix before rerunning everything; running and resuming jobs belongs to the evaluation-running skill.

- **Verifier or answer extraction.** Re-score the saved outputs with the changed verifier. This makes no policy model calls:

  ```bash
  gym eval reverify --config <config with the changed resources server> \
      --inputs results/rollouts_materialized_inputs.jsonl \
      --rollouts results/rollouts.jsonl \
      --output results/rescored/rollouts.jsonl
  ```

  It starts every server in the config, so pass one with just the resources server, plus the judge's model server on judged benchmarks, and it refuses servers that do not support re-scoring unless forced. Both scorings cover the same rollouts, so their difference is exact: count the rollouts and tasks whose reward changed rather than reading an interval from `gym eval compare`, which assumes two independent runs.

  ```bash
  python3 - results/rollouts.jsonl results/rescored/rollouts.jsonl <<'EOF'
  import json, sys
  def rewards(path):
      result = {}
      with open(path) as f:  # Exits with an error when the file is missing.
          for line in f:
              try:
                  r = json.loads(line)
              except json.JSONDecodeError:
                  continue  # Blank lines, and a cut-off last line.
              result[(r.get("_ng_task_index"), r.get("_ng_rollout_index"))] = r.get("reward")
      return result
  old, new = rewards(sys.argv[1]), rewards(sys.argv[2])
  changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
  print(f"{len(changed)} of {len(old.keys() & new.keys())} rollouts changed reward, in tasks {sorted({t for t, _ in changed})}")
  EOF
  ```
- **Prompt, token limit, sampling or harness.** A rerun with the change on the same tasks and repeats, then a comparison of the two runs.
- **Infrastructure failures.** Resuming the run so the failed rollouts are measured, then comparing again.

To rank fixes, estimate what removing one failure mode is worth: for each task, keep the rollout least affected by that failure mode and average those rewards. Never drop tasks for this estimate. The gap between the best estimate and the best rollout per task shows failures that the labels do not yet explain.

## Report Structure

Use this structure for a full written analysis, unless the benchmark defines its own. It is BLADE's report layout; keep the section names, because BLADE's `make-shallow` check reads the Aggregate Results and Workflow Funnel sections:

```markdown
# <Benchmark> analysis: <model or baseline vs candidate>

## Executive Summary          verdict and the two or three findings that matter
## Artifact Inventory         sources, settings, planned vs measured, failures by class
## Aggregate Results          scores with intervals; the comparison and what it supports
## Workflow Funnel            how many rollouts reach each stage, and the largest drop
## Task Outcome Buckets       always, sometimes, never pass, not measured
## Dominant Failure Modes     each with counts and 2 or 3 examples (task and rollout ids)
## Sometimes-Pass Deep Dives  what differs between passing and failing repeats
## Never-Pass Deep Dives      verifier, label and format checks, then capability
## Cross-Model Comparison     for two or more runs: what changed, task by task
## Recommendations            fixes tied to labels, in order of expected gain
## Reproducibility Notes      commands, versions, file paths
```

A useful report is not a table of metrics. It finds the largest drop, proves the main failure mode with examples, explains the tasks that sometimes pass, and ends with fixes that follow from the evidence.
