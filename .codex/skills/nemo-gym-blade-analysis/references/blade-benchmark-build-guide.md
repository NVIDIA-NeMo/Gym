# BLADE Benchmark Build Guide

Load this reference when the user wants to build, validate, submit, or review a
BLADE-ready benchmark. Keep benchmark-specific examples optional; this guide is
generic and should apply across NeMo Gym environments.

## Required Deliverables

A BLADE-ready benchmark needs four deliverables:

| ID | Deliverable | Purpose |
|----|-------------|---------|
| D1 | Analysis skill | Teaches an agent how to analyze the benchmark's rollout data. |
| D2 | Rollout data | Provides comparable model runs for analysis and judge calibration. |
| D3 | Golden reports | Establishes curated, evidence-backed analysis outputs. |
| D4 | Benchmark-specific judge | Scores candidate analysis reports against deterministic facts and qualitative depth. |

Do not call a benchmark BLADE-ready until all four deliverables exist and have
been checked for consistency.

## D1: Analysis Skill

The analysis skill must include substantive sections for:

- Overview: what the benchmark measures, task shape, reward computation, and
  verifier behavior.
- Input data schema: field names, types, examples, task id, rollout id, reward
  signal, trajectory/output structure, and slicing dimensions.
- Failure taxonomy: hierarchical categories with detection rules, not just
  labels.
- Analysis workflow: deterministic metrics first, qualitative trajectory reading
  second, then causal synthesis.
- Output report structure: the markdown report expected from the skill.

The taxonomy should usually have four layers:

| Layer | Question | Typical Output |
|-------|----------|----------------|
| Pipeline stage | Where did the rollout fail? | Exhaustive phase/funnel labels. |
| Error pattern | What symptom appeared? | Programmatic or semi-programmatic subcodes. |
| Behavioral pattern | What did the model do wrong? | Evidence from trajectories or outputs. |
| Root cause | Why did it fail and what improves it? | Knowledge, behavior, task, infra, or data label. |

Keep a clear split between code and model judgment:

| Analysis Step | Owner | Reason |
|---------------|-------|--------|
| Aggregate metrics | Script | Deterministic counts. |
| Pipeline funnel | Script | Field-based classification. |
| Error pattern counts | Script | Regex or structured-field rules. |
| Behavioral patterns | Analyst/LLM | Requires reading trajectories. |
| Root-cause labels | Analyst/LLM | Requires task-level judgment across repeats. |
| Causal narrative | Analyst/LLM | Requires synthesis, counterfactuals, and examples. |

## D2: Rollout Data

Use rollout data from at least two models, and preferably three, with meaningful
performance spread. A useful set often has one weak model, one medium model, and
one strong model. Without spread, the judge has less signal for distinguishing
metric-only reports from real diagnosis.

Rollout files should be valid JSONL and should have consistent task sets where
comparisons are expected. Each row should expose or allow derivation of:

- task identifier
- rollout identifier or repeat index
- reward or score
- model output or trajectory
- verifier output or failure signal
- task category, difficulty, domain, or other useful slices
- token/step/runtime metadata when available

Do not include large raw rollouts inside a skill reference unless they are
explicitly needed and appropriate for the target repository. Prefer pointers to
existing in-repo example rollouts or external benchmark artifacts.

## D3: Golden Reports

Golden reports are curated analysis reports. They are not just script output.
They should combine deterministic metrics with diagnostic reasoning.

A golden report should include:

- aggregate metrics and workflow funnel
- per-slice breakdowns with totals and denominators
- root-cause classification at task level when repeats exist
- 3-5 concrete examples with task id, rollout id, and verifier/log evidence
- within-task comparisons for sometimes-pass tasks
- cross-cutting patterns not visible from a single table
- non-obvious findings that require reading trajectories or generated outputs
- a concise improvement plan tied to the diagnosis

Every percentage should state or imply its denominator. Every major diagnosis
should cite evidence.

Golden report metrics belong in a JSON sidecar. Include at least:

```json
{
  "model_name": "<model>",
  "benchmark": "<benchmark>",
  "pass_at_1": 0.0,
  "total_tasks": 0,
  "total_rollouts": 0
}
```

Add benchmark-specific metrics such as pass@k, consistency, oracle ceiling,
pipeline counts, error-pattern counts, per-category breakdowns, or token stats.

Anchor facts are optional but useful for judge calibration. They should capture
important findings with enough detail to verify whether a candidate report found
the same pattern.

## D4: Benchmark-Specific Judge

The judge should combine deterministic checks and qualitative scoring.

Deterministic checks should validate facts such as:

- pass@1, pass@k, or benchmark-native metric values within tolerance
- total tasks, rollout counts, and coverage
- pipeline/funnel counts
- key per-category or per-difficulty breakdowns
- required report sections
- cited task or rollout ids, when the data is available to the judge

Qualitative scoring should reward:

- dominant failure-mode identification
- root-cause distinction, especially behavior vs knowledge vs task issue
- causal narrative, not a flat list of symptoms
- evidence citation with task ids, rollout ids, logs, tool calls, or generated
  output snippets
- within-task comparison for sometimes-pass tasks
- cross-cutting insight across categories, domains, or model versions
- non-obvious findings that cannot be derived from aggregate metrics alone

The judge should also detect shortcut analysis:

| Shortcut | Signal | Catch |
|----------|--------|-------|
| Template filling | Generic structure with plausible labels | Missing anchor facts and concrete examples. |
| Hallucinated evidence | Specific-looking citations | Cross-check cited ids/logs against data. |
| Script-only report | Accurate metrics, shallow diagnosis | No causal narrative or trajectory evidence. |
| Confirmation bias | Finds only expected patterns | Misses anchor facts or contrary examples. |
| Lucky heuristic | Correct headline, wrong details | Fails per-task or per-category checks. |

## Counterfactual Methodology

Use within-task comparisons when repeats exist. Compare rollouts with and
without a suspected failure mode within the same task, not across tasks. This
controls for task difficulty.

For ablation-style estimates, never drop tasks. For each task, select the least
affected rollout under a severity function and average those rewards. If all
rollouts exhibit a failure mode, the least severe rollout still represents that
task.

Useful rows for an ablation table:

- baseline average reward
- one row per individual failure mode
- all failure modes combined
- oracle best rollout per task
- reference model, if available

The gap between a failure-mode ablation and the oracle identifies failures not
explained by the current taxonomy.

## Readiness Checklist

Use this checklist before marking a benchmark ready:

- Analysis skill has overview, schema, taxonomy, workflow, and report template.
- Rollouts exist for multiple models and use comparable task sets.
- Metrics can be recomputed or verified from rollout data.
- Golden reports contain causal diagnosis and concrete examples, not only
  aggregate tables.
- Metrics JSON sidecars exist and parse.
- Judge instructions define deterministic checks and qualitative rubric.
- Judge catches template filling, fabricated evidence, and script-only reports.
- Sanitization pass removes private source, endpoints, credentials, personal
  names, unreleased benchmark names, and raw data not cleared for sharing.
