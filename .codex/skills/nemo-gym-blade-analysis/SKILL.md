---
name: nemo-gym-blade-analysis
description: >-
  Use when analyzing NeMo Gym benchmark rollouts for BLADE-style reports,
  writing benchmark methodology notes, checking whether a benchmark is
  BLADE-ready, comparing model runs, or explaining why a benchmark report
  passed, failed, or changed. Covers aggregate metrics, rollout evidence,
  report structure, root-cause taxonomy, judge expectations, and improvement
  recommendations. For generic reward profiling commands, prefer
  nemo-gym-reward-profiling; for failed infrastructure jobs, prefer
  nemo-gym-debugging.
---

# NeMo Gym BLADE Analysis

## Invocation Check

Use this skill when the user wants to turn NeMo Gym rollout outputs into an
analysis report, benchmark card, model comparison, benchmark-improvement
recommendation, or BLADE-ready benchmark package.

Load `references/blade-benchmark-build-guide.md` when the user asks how to
build, validate, submit, or review a BLADE benchmark or asks whether a benchmark
has all required BLADE deliverables.

Do not load benchmark-specific examples by default. Load
`references/cvdp-report-example.md` only when the user explicitly asks for a
CVDP example, the original CVDP report layout, or this optional reference, or
when the agent is confused about the goal and needs one concrete example to
re-anchor on what a BLADE-style report is supposed to look like.

Nemotron-only golden analysis artifacts are available under
`references/nemotron-analysis-artifacts/` as original-CVDP example artifacts.
Load them only when the user explicitly asks to study an example completed
BLADE-style report, asks for CVDP artifacts, or the agent is confused about the
goal and needs a concrete completed example. Do not load those files by default.

## Inputs To Gather

Start by identifying the artifact set:

- rollout JSONL from `ng_collect_rollouts`
- aggregate metrics JSON, if present
- reward profile JSONL from `ng_reward_profile`, if present
- benchmark-specific report directory, if present
- optional golden analysis artifacts, if the user asks to compare against a
  curated report
- config paths, agent name, model name, repeat count, and sampling settings
- source dataset metadata, license, and known redaction limits

If artifacts are missing, state which claims cannot be supported rather than
filling gaps from memory.

If the task is benchmark construction rather than report analysis, first build
an inventory of BLADE deliverables: analysis skill, rollout data, golden reports
with metrics, and benchmark-specific judge. Missing deliverables are blocking
work items, not footnotes.

## Analysis Workflow

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

## Report Structure

Use this structure unless the benchmark already defines a report format:

```markdown
# <Benchmark> BLADE Analysis Report

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

## Core Metrics

- `pass@1`: average rollout success rate.
- `pass@k`: fraction of tasks with at least one successful rollout across `k`
  repeats.
- consistency: fraction of tasks where every rollout succeeds.
- coverage: completed rollout rows divided by expected rows.
- retry value: pass@k minus pass@1, useful for spotting instability.
- variance across repeats, seeds, model versions, or task categories.

For benchmark-specific metrics, keep the original names and definitions. Do not
rename verifier outputs unless the report includes a mapping table.

## Evidence Rules

- Tie every major claim to row counts, task ids, rollout ids, logs, verifier
  messages, tool calls, or report files.
- Read trajectories in order. Avoid attributing an early failure to evidence
  that only appears later.
- Distinguish self-test success from verifier success.
- Treat missing rows, timeouts, and malformed outputs as first-class outcomes.
- Mark redacted or unavailable evidence explicitly.
- Do not include private source code, private endpoints, credentials, user names,
  or unreleased benchmark names in a shareable report.

## Root-Cause Taxonomy

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

## Recommendation Mapping

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

## Quality Bar

A good BLADE analysis is not a metrics dump. It should identify the largest
drop-off, prove the dominant failure mode with examples, explain sometimes-pass
behavior, and end with an intervention plan that follows from the evidence.
