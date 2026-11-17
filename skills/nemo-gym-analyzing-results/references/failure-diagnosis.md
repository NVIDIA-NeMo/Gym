# Diagnose a run: which tasks fail and why

Use this when the runs are comparable (see [score-comparability.md](score-comparability.md)) and the question is which tasks fail and why, or when you must write up a benchmark's results for others. Each section is a step of the analysis in the order you do it.

## 1. Workflow

1. Count tasks, rollouts, completed rows, repeats per task, and missing rows.
2. Compute pass@1 and pass@k from rewards or reward profiles.
3. Build a workflow funnel appropriate to the benchmark.
4. Split tasks into always-pass, sometimes-pass, never-pass, and missing.
5. Compare passing and failing trajectories for sometimes-pass tasks.
6. Inspect representative never-pass trajectories in chronological order.
7. Separate model capability gaps, agent behavior issues, verifier/task issues,
   and data or infrastructure problems.
8. Map findings to concrete actions: data, prompts, agent workflow, verifier
   repair, environment reliability, SFT, RL, or benchmark documentation.

For multi-repeat benchmarks, sometimes-pass tasks are the highest-signal slice:
they show the conditions under which the same task can succeed or fail.

## 2. Report structure

Use this structure unless the benchmark already defines a report format:

```markdown
# <Benchmark> results analysis report

## Executive Summary
## Artifact Inventory
## Aggregate Results
## Workflow Funnel
## Task Outcome Buckets
## Dominant Failure Modes
## Sometimes-Pass Deep Dives
## Never-Pass Deep Dives
## Cross-Model Comparison
## Recommendations
## Reproducibility Notes
```

Keep the executive summary short and evidence-backed. A useful report explains
what changed, why it changed, and what to do next.

## 3. Core metrics

- `pass@1`: average rollout success rate.
- `pass@k`: fraction of tasks with at least one successful rollout across `k`
  repeats.
- consistency: fraction of tasks where every rollout succeeds.
- coverage: completed rollout rows divided by expected rows.
- retry value: pass@k minus pass@1, useful for spotting instability.
- variance across repeats, seeds, model versions, or task categories.

For benchmark-specific metrics, keep the original names and definitions. Do not
rename verifier outputs unless the report includes a mapping table.

## 4. Evidence rules

- Tie every major claim to row counts, task ids, rollout ids, logs, verifier
  messages, tool calls, or report files.
- Read trajectories in order. Avoid attributing an early failure to evidence
  that only appears later.
- Distinguish self-test success from verifier success.
- If an error appears before the model's consequential action, do not attribute
  that error to the later action; diagnose whether the model failed to retry,
  verify, or recover instead.
- Treat sometimes-pass tasks as primary diagnostic evidence, but inspect whether
  success came from real understanding, lucky output, or workflow variance.
- Treat missing rows, timeouts, and malformed outputs as first-class outcomes.
- Mark redacted or unavailable evidence explicitly.
- Do not include private source code, private endpoints, credentials, user names,
  or unreleased benchmark names in a shareable report.

## 5. Root-cause taxonomy

Use one primary label per failed or mixed task when possible:

- `KG` knowledge gap: the model lacks domain, API, tool, or verifier knowledge.
- `UK` unreliable knowledge: some repeats show the needed knowledge and others
  do not.
- `BI` behavioral issue: the model appears capable but skips key steps, gives up,
  thrashes, ignores feedback, or uses tools poorly.
- `TI` task/verifier issue: the task, harness, timeout, dependency, or expected
  answer is suspect.
- `IR` infrastructure reliability: failures come from service availability,
  sandbox startup, scheduler behavior, network, storage, or provider errors.
- `DA` data artifact: duplicated rows, bad metadata, prompt leakage, missing
  files, or inconsistent labels affect the result.

Prefer a mixed label only when the evidence genuinely requires it, such as
`BI+KG`. Do not use a mixed label to avoid making a call.

## 6. Recommendation mapping

- `KG`: add targeted examples, domain SFT, better task docs, or verifier-facing
  explanations.
- `UK`: add repeated rollout training, RL, self-checking prompts, or comparison
  data from passing trajectories.
- `BI`: shape the agent loop, add workflow checks, reward intermediate
  verification behavior, or simplify tool affordances.
- `TI`: repair the task, verifier, timeout, or dependency; rerun baselines after
  repair.
- `IR`: fix infrastructure, isolate flaky rows, and keep flake rates separate
  from model quality.
- `DA`: correct dataset metadata and regenerate artifacts before model
  comparison.

## 7. Quality bar

A good analysis is not a metrics dump. It should identify the largest
drop-off, prove the dominant failure mode with examples, explain sometimes-pass
behavior, and end with an intervention plan that follows from the evidence.
