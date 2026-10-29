---
name: nemo-gym-score-triage
description: >-
  Use when a NeMo Gym score does not match a reference number (tech report, model card, Artificial
  Analysis, a leaderboard, an earlier run), when two runs or two harnesses disagree, when repeat-to-repeat
  variance looks too high, when the same output gets different rewards, or when someone asks whether a
  baseline-vs-candidate difference (for example BF16 vs a quantized checkpoint) is real. Decides whether
  the numbers are comparable before any deep analysis. For failed or empty runs, prefer nemo-gym-debugging;
  for task-level failure analysis of comparable runs, prefer nemo-gym-blade-analysis.
---

# NeMo Gym Score Triage

## Invocation Check

Use this skill when the question is "why is this number different?" rather than "why did this run fail?". Most reported gaps come from comparing unlike things: another code version, another benchmark version, another method, other failure accounting, or ordinary noise. Rule those out before treating the gap as a change in the model.

- If the run crashed, produced no rollouts, or the metrics file is missing, use the `nemo-gym-debugging` skill first.
- If both runs are comparable and the question becomes which tasks fail and why, hand off to the `nemo-gym-blade-analysis` skill.
- For the meaning of profiling files and keys, see the `nemo-gym-reward-profiling` skill.

## Collect First

Get these for both sides of the comparison. A reference without a config or a date is a rough target, not a contract.

- Both numbers, and where each came from: document and section, leaderboard page and the date it was read, or a run directory.
- The run directory: rollouts JSONL, `*_failures.jsonl`, `*_materialized_inputs.jsonl`, `*_aggregate_metrics.json`, `*_resolved_config.yaml`, and the health reports `quality_summary.json` and `rollout_verdicts.jsonl`.
- The Gym commit and config:
  - Since #4908, every `gym eval run` saves its merged config, secrets masked, beside the rollouts as `<stem>_resolved_config.yaml`. It keeps `${...}` interpolations as written, so values read from environment variables (GDPval's `JUDGE_MODEL_NAME`, for example) are not in it; take those from the launcher's environment or `driver.env` in `resolved-config.yaml`. For older runs, rebuild the config with `gym env resolve --config <benchmark config> --config <model config> ++key=value`, which resolves today's YAML rather than what ran and fails on an unset environment variable without a default (set a dummy value).
  - Plain `gym eval run` records no commit, so use the launcher's log or `git rev-parse HEAD` of the checkout that ran it.
  - `gym eval submit` also writes `gym-job.json` and the submit config `resolved-config.yaml` into the run directory. `gym_commit` in `gym-job.json` is the Gym that submitted the job (a `-dirty` suffix means uncommitted changes or untracked files). The job itself runs `driver.gym_install.ref` from `resolved-config.yaml` (a branch or tag resolves when each job starts), or the Gym inside the driver container when that is unset. That is the commit that scored the run.
  - Note container or image tags too.
- Model, checkpoint and precision; serving engine and version; reasoning and tool-call parsers; parallelism and GPU type.
- Sampling and limits: temperature, top_p, max output tokens, context length, reasoning effort or mode, chat-template arguments, system prompt.
- Agent harness, judge model, user simulator, tools, timeouts, max steps or turns, and repeats.

## Triage Order

Work from the top. Run steps 2, 3 and 5 even when the gap looks like noise, because a systematic difference can hide inside a wide interval. Stop when the findings account for the gap, and report the evidence: a file, a key, a log line or a config diff.

1. **Noise.** Get the repeat count and standard error of both numbers being compared: the Gym run being checked and the reference (a published number or another run). A Gym run with two or more repeats has them in the across-repeat statistics in `*_aggregate_metrics.json`. If the reference publishes no standard error, estimate it as `sd1 / sqrt(k_ref)`, where `sd1` is the standard deviation of a single repeat's score in the Gym run (`std_across_repeats/mean/reward`, not a repeat's own `std/reward`, which is the spread across tasks) and `k_ref` is the reference's repeat count (1 if unknown). The gap's standard error is `sqrt(se_a^2 + se_b^2)`, with both on the same scale, and a gap under about twice that is within noise. When a standard error comes from only a few repeats, the multiplier is larger than 2, so for two Gym runs use the Welch interval from `gym eval compare`. See `references/variance.md`.
2. **Code.** Check whether both sides ran the same Gym commit and images. If not, list the changes between them: `git log --oneline --left-right <a>...<b> -- nemo_gym/ resources_servers/<server> responses_api_agents/<agent> responses_api_models/<model> environment_servers/<env server> environments/<env> benchmarks/<benchmark>`. To separate a scorer change from a generation change, re-score the old rollouts on the new commit with `gym eval reverify`, where the resources server supports it.
3. **Benchmark and reference.** Check that the dataset version, subset, date window and metric definition match. Data comes from a separate prepare step, so two runs on one commit can still use different data: compare the task sets and prompts in both runs' `*_materialized_inputs.jsonl`. See `references/comparability-checklist.md` and `references/benchmark-notes.md`.
4. **Method.** Diff the two runs' `*_resolved_config.yaml` files: sampling, token limits, prompts, judge, user simulator, tools, timeouts, max steps, harness and repeats.
5. **Completeness and failure accounting.** Compare scored rollouts with materialized inputs, count the failure class of each unscored rollout's last attempt, and check whether either side counts failures as zero. Since #4788, `run.artifacts` in `quality_summary.json` holds these counts (`expected`, `failed` by class, `missing`); for older runs, use the Quick Checks script in `references/comparability-checklist.md`. A reference may count timeouts as failures where Gym leaves them out, or the reverse.
6. **Rollout health.** Read `quality_summary.json` and `rollout_verdicts.jsonl`, which `gym eval run` writes beside the rollouts, even for a failed run. `references/comparability-checklist.md` says what they contain and where they mislead. The checks flag empty or runaway generations and failed model calls; `unobserved` means a check could not run. If the reports are missing, provisional or from before #4788, run `gym eval health-check <run_dir>` (add `--rollouts-file <name>` if the file isn't `rollouts.jsonl`); it writes its reports into the run directory, so copy a shared or read-only run first. Look for answers that sit only in the reasoning field, and for judge failures, in the rollouts and the failures file yourself.
7. **Scoring.** Read a sample of rollouts whose reward disagrees with what a person would judge. For code and patch benchmarks, validate the environment with known-good solutions, and check that identical outputs get identical rewards. For judged benchmarks, re-judge a sample with `gym eval reverify` and measure agreement.
8. **Serving.** Compare precision, engine, parsers and throughput. Wall-clock timeouts turn slower serving into lower agentic scores. For a quantized-vs-baseline question, run both on the same stack with identical configs and the same Gym commit.

## Report Back

```text
Summary: within noise | explained | not yet explained
Compared: <number A> (<source, date>) vs <number B> (<source, date>)
Checked: noise, code, benchmark/reference, method, accounting, health, scoring, serving
What explains the gap: <one finding with its evidence>
Next step: <rerun on commit X, add repeats to N, align setting Y, or escalate to the benchmark owner>
```

## Practical Defaults

- Don't call a regression from a single run of an agentic benchmark; average several runs.
- Prefer paired comparisons on the same tasks, commit and config. They remove task-sampling and setup differences; rollout randomness remains.
- When a reference publishes its methodology, write down which of its settings this run does not follow, even if they look minor.
- Trust the target checkout's code and CLI help over this skill when they disagree.

## Reference Loading

Load references only when the step needs them:

- `references/comparability-checklist.md`: every factor that changes a score, where to check it in a Gym run, and how Gym accounts for failures.
- `references/variance.md`: how to quantify run-to-run spread with Gym outputs and how many repeats a decision needs.
- `references/benchmark-notes.md`: known differences between Gym benchmarks and the published methods they follow.
