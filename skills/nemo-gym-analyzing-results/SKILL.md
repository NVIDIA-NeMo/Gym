---
name: nemo-gym-analyzing-results
description: >
  Read, compare and explain finished NeMo Gym evaluation results. Compares a baseline run with a
  candidate (for example BF16 vs a quantized checkpoint, two models, two harnesses) with confidence
  intervals, checks whether two numbers are comparable at all, explains a gap to a reference score
  (model card, tech report, leaderboard, earlier run), and finds which tasks fail and why with
  evidence from the rollouts. Use when asked "why is this score different", "is this difference
  real", "my number does not match the paper", "compare these two runs", "which tasks fail",
  "why did the score change", "what should we fix next", "is the variance too high" or "write up
  these results". Not for runs that crashed or produced no rollouts (use nemo-gym-debugging), for
  running the evaluation (use nemo-gym-running-evals) or for the meaning of the profiling files
  (use nemo-gym-reward-profiling).
---

# Analyze and compare Gym evaluation results

Most reported gaps come from comparing unlike things (another code version, benchmark version, method or failure accounting) or from noise. Rule those out first, then compare, then diagnose. Report evidence, not impressions.

## 1. Pick the question

| The question is | Do | Read |
| --- | --- | --- |
| "Why does my number differ from a reference or another run?" | Comparability triage (section 3) | [references/score-comparability.md](references/score-comparability.md) |
| "Is the candidate better than the baseline?" | Paired comparison (section 4) | `gym eval compare --help`, section 4 |
| "Which tasks fail, and why?" | Failure diagnosis (section 5) | [references/failure-diagnosis.md](references/failure-diagnosis.md) |
| "The run crashed or has no rollouts" | Not this skill | `nemo-gym-debugging` |

## 2. Collect first

For both sides of the comparison, get the numbers with their source (document and section, leaderboard page and date read, or a run directory) and the run directory itself: the rollouts JSONL, `*_failures.jsonl`, `*_materialized_inputs.jsonl`, `*_aggregate_metrics.json`, `*_resolved_config.yaml` and the health reports `quality_summary.json` and `rollout_verdicts.jsonl`. Also: the Gym commit and images, the model and its precision, the serving engine and parsers, sampling and limits, the agent harness, judge, tools, timeouts and the repeat count. A reference without a config or a date is a rough target, not a contract. If an artifact is missing, say which claim it leaves unsupported; do not fill the gap from memory.

## 3. Comparability triage

Work from the top; run steps 2, 3 and 5 even when the gap looks like noise. Stop when the findings account for the gap.

1. **Noise.** Get each side's repeat count and standard error. A gap under about twice the combined standard error is within noise (`sqrt(se_a^2 + se_b^2)`); with few repeats use the interval from `gym eval compare`. Spread across repeats is `std_across_repeats/mean/reward` in `*_aggregate_metrics.json`, not a repeat's own `std/reward` (the spread across tasks).
2. **Code.** Same Gym commit and images? If not, list the changes between them. To separate a scorer change from a generation change, re-score the old rollouts on the new commit with `gym eval reverify` where the resources server supports it.
3. **Benchmark and reference.** Same dataset version, subset, date window and metric definition? Data comes from a prepare step, so two runs on one commit can still differ: compare the task sets and prompts in both `*_materialized_inputs.jsonl`.
4. **Method.** Diff the two `*_resolved_config.yaml`: sampling, token limits, prompts, judge, tools, timeouts, max steps, harness, repeats.
5. **Completeness and failure accounting.** Compare scored rollouts with materialized inputs; count the failure class of each unscored rollout; check whether either side counts failures as zero (a reference may count timeouts as failures where Gym leaves them out, or the reverse).
6. **Rollout health.** Read `quality_summary.json` and `rollout_verdicts.jsonl`. If they are missing or old, run `gym eval health-check <run_dir>` on a copy. Look yourself for answers that sit only in the reasoning field and for judge failures.
7. **Scoring.** Read a sample of rollouts whose reward disagrees with what a person would judge; for judged benchmarks re-judge a sample with `gym eval reverify` and measure agreement.
8. **Serving.** Compare precision, engine, parsers and throughput. Timeouts turn slower serving into lower agentic scores. For quantized vs baseline, both runs need the same stack, config and Gym commit.

The checks, where to find each factor in a run directory and how Gym counts failures are in [references/score-comparability.md](references/score-comparability.md).

## 4. Compare a baseline and a candidate

Run both on the same benchmark, split, task order, limit and verifier settings, with repeats, each into its own directory. Comparison reads the saved aggregate files: no servers, no model calls.

```bash
gym eval compare \
    --baseline results/baseline/rollouts.jsonl \
    --candidates results/candidate/rollouts.jsonl \
    --agent <agent name> \
    --output-dir results/comparison --report-format both
```

It writes `compare_report.md` and `compare_report.json`: metric deltas with confidence intervals (95% by default, `--confidence-level` to change) and the tasks that flipped between pass and fail. `--candidates` currently takes one run. Both runs need at least two repeats for a repeat-based interval. For several benchmarks or models use `gym eval compare-suite --suite suite.yaml`. Do not call a regression from one run of an agentic benchmark: average several, and prefer paired comparisons on the same tasks, commit and config.

## 5. Find which tasks fail and why

When the runs are comparable and the question is the failure itself, follow [references/failure-diagnosis.md](references/failure-diagnosis.md): count tasks, rollouts and missing rows; compute pass@1 and pass@k; split tasks into always-pass, sometimes-pass, never-pass and missing; compare passing and failing trajectories of sometimes-pass tasks; read never-pass trajectories in order; label each failure with one primary cause (knowledge gap, unreliable knowledge, behavioral issue, task or verifier issue, infrastructure reliability, data artifact); map each cause to an action. Sometimes-pass tasks are the highest-signal slice for multi-repeat runs.

## 6. Report back

```text
Summary: within noise | explained | not yet explained
Compared: <number A> (<source, date>) vs <number B> (<source, date>)
Checked: noise, code, benchmark/reference, method, accounting, health, scoring, serving
What explains the gap: <one finding with its evidence: a file, a key, a log line or a config diff>
Next step: <rerun on commit X, add repeats to N, align setting Y, or escalate to the benchmark owner>
```

A failure analysis follows the report structure in [references/failure-diagnosis.md](references/failure-diagnosis.md), section 2. Tie every major claim to row counts, task ids, rollout ids or logs. Do not include private source code, endpoints, credentials or user names in a shareable report.

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| `gym eval compare` has no confidence interval | A run has fewer than two repeats, or its `*_aggregate_metrics.json` is missing: run `gym eval aggregate` on it. |
| Agent names differ between the runs | Select each side with `--baseline-agent` and `--candidate-agents`. |
| Health reports are missing or provisional | `gym eval health-check <copy of run_dir>`; a run stopped by SIGTERM has only a provisional summary. |
| The same output gets different rewards | A judged benchmark: re-judge a sample (`gym eval reverify`) and measure agreement; check the judge model and its settings. |
| The gap is large and nothing in section 3 explains it | Say so: "not yet explained", with what was checked, and escalate to the benchmark owner. |
| The run itself failed or is empty | `nemo-gym-debugging`. |

## Exit checks

- [ ] Both numbers have a source and a run directory or an explicit "not available".
- [ ] Noise was quantified (repeats, standard error or interval) before any change was called real.
- [ ] Code, data, method and failure accounting were compared, or each unchecked item is named.
- [ ] Every claim in the report cites a file, key, row count, task id or config difference.
- [ ] The summary says `within noise`, `explained` or `not yet explained`, with one next step.

## References

- [references/score-comparability.md](references/score-comparability.md): every factor that changes a score, failure accounting, health reports, quick checks, variance and repeats.
- [references/failure-diagnosis.md](references/failure-diagnosis.md): workflow, report structure, metrics, evidence rules, cause taxonomy, recommendations.
- [references/benchmark-notes.md](references/benchmark-notes.md): known differences between Gym benchmarks and the published methods they follow.
