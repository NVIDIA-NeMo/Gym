# Benchmark Notes

Known differences between Gym benchmarks and the published methods they follow, and traps seen when comparing their numbers. Code and config in the target checkout win over these notes and over benchmark READMEs, which can lag behind the code.

## Terminal-Bench 2.1 (`benchmarks/terminal_bench_2_1`)

- 8 repeats. The Terminus 2 and OpenCode sandboxed agents use a 10,800 s sandbox timeout, and the verifier's evaluation timeout is 1,800 s.
- Prepared rows carry no per-task agent timeout, so every task gets the sandbox timeout. Harbor-based runs that read each task's own timeout can differ.
- `prepare.py` clones the Terminal-Bench 2.1 repository at its head when no checkout exists, while the README's steps check out a pinned commit first. The script checks only that there are 89 tasks, so ask which commit a run used.
- A verifier error or a missing reward file scores 0 and stays in the denominator.
- Timeouts are wall-clock, so throughput and concurrency change scores. Before raising a timeout to finish more tasks, check whether the reference used the same one.
- Artificial Analysis publishes its own timeout and episode limits. Compare them with the run's settings before comparing scores.

## Terminal-Bench 4.0 (`benchmarks/terminal_bench_4`)

- The profile follows Artificial Analysis's version, prompts, step limit and command timeout, as described in the README.
- It defaults to 1 repeat. Set `++num_repeats=3` to follow AA's protocol of averaging pass@1 over three runs.
- Since #3624, the model sees mini-SWE's own output limit: long command output is cut to its first and last 5,000 characters, and the full output stays in `trajectory.json`. Earlier runs showed full output, so they are not comparable with later ones. The note on observations in `benchmarks/terminal_bench_4/README.md` predates #3624.
- A verifier timeout, or a missing, empty or malformed reward file, scores 0 and stays in the denominator. Only real infrastructure errors (`infrastructure_error`) leave the score. Runs from before #5038 left all of these out of the denominator, so they are not comparable with later runs. That README's note on verifier timeouts predates #5038.

## tau2 and tau3 (`benchmarks/tau2`)

- `banking_bm25_grep_artificial_analysis.yaml` runs the 97 banking tasks for 8 repeats, with a `gpt-5.4-mini-2026-03-17` user simulator at medium reasoning and at most 200 steps.
- Prepared rows drop `NL_ASSERTION` from the reward basis, so there is no NL-assertion judge. Only one banking task uses it, but scores are not strictly identical to Artificial Analysis.
- Agentic tau runs vary by several points between runs. Check repeats before reading a gap.

## GDPval (`benchmarks/gdpval`)

- Rubric mode is the default; the reward is the judge's overall score.
- Comparison mode reports an Elo against reference models, with `normalized_elo = (elo - 500) / 2000`. Compare only with numbers computed against the same reference models and the same reference outputs.
- The judge is sampled from a panel per call. The default is 1 repeat.
- In comparison mode, judge failures are environment-server failures, and `--judge-failed-only` does not cover them.
- In either mode, a payload the judge will always refuse becomes a terminal `permanent_judge_error` failure; in comparison mode, one that no judge can take becomes `transport_ineligible`. `--resume` reruns neither.
- Harnesses treat timeouts differently. A Stirrup timeout (`stop_reason: wall_time`) becomes `agent_timeout` in the failures file, leaves the score and reruns on `--resume`. A Pi timeout is judged as an empty submission and scored. Count `agent_timeout` as zero before comparing the two harnesses.

## HLE (`benchmarks/hle`)

- 2,158 text-only questions, 1 repeat. The dataset revision is not pinned.
- The judge is the policy model unless the config overrides it, and no judge temperature is set. Ask which judge produced each number before comparing.
- The server emits `mean/reward` and a judge parsing-issue rate. An unparseable verdict counts as wrong; a failed judge call becomes `judge_failed`.
- HLE-Diamond (`benchmarks/hle_diamond`) is a 1,000-question subset of HLE, 272 of them with images; its text presets run 728. Compare it only with HLE-Diamond numbers over the same rows.

## LiveCodeBench (`benchmarks/livecodebench`)

- v5 covers problems dated 2024-08-01 to 2025-03-01 (279 problems); v6 covers 2024-08-01 to 2025-06-01. Both run 3 repeats, from a dataset pinned to a Hugging Face PR ref.
- Think tags left in the final answer, or more than one `<think>` or `</think>` in the reasoning, replace the reward with `reasoning_format_penalty` (default 0), so a correct solution scores 0.
- Grader fixes changed scores: stdin handling (#2508, #2824), `sys.stdout.buffer` writes and a 1e-6 float tolerance (#2956), and case-insensitive Yes/No (#3051). Runs from before #3051 are not comparable with later ones.
- Other LCB harnesses may still fail correct solutions that use fast I/O (`sys.stdin.buffer`, `sys.stdout.buffer`).

## GPQA (`benchmarks/gpqa`)

- GPQA-Diamond, with options shuffled by an md5(question) seed, the Artificial Analysis-style prompt `benchmarks/prompts/eval/aai/mcq-4choices.yaml`, `Answer: X` grading, `max_steps: 1` and 8 repeats.
- Other implementations differ in prompt, option order and answer parsing. Compare like with like.
- `resources_servers/gpqa_diamond` is a separate training environment with a different verifier.

## SWE-bench Verified and Pro (`benchmarks/swebench`)

- Reward is whether the instance resolved. A failed or inconclusive evaluation scores 0 and stays in the denominator. Pro first re-runs an inconclusive verification on a new sandbox up to 2 more times, each attempt capped at 1,200 s and all of them at 2,700 s.
- Patch application falls back from `git apply` variants to `patch` with fuzz 5.
- The Verified dataset is not pinned; Pro (731 instances) is. The OpenCode profiles run 3 repeats; `pro/hermes.yaml` runs 1.
- Validate the environments with golden patches before chasing a gap to a published number, and check that identical patches get identical rewards across runs.

## AA-Omniscience (`benchmarks/omniscience`)

- The dataset revision is pinned. The default judge is `openai/gpt-oss-20b` at temperature 0, with 8 repeats. A different judge model gives different scores.
- The grader expects the judge to reply with a single letter. When neither the reply nor its last line is a bare letter, it looks for A, B, C and D in that order anywhere in the reply, so any capital A (even in `Answer: B`) reads as CORRECT; with no letter at all it returns B. Check a sample of judge replies before trusting a surprising number.

## AA-LCR (`benchmarks/aalcr`)

- 100 tasks × 16 repeats. The default (v1.0) judge is Qwen3-235B-A22B-Instruct-2507; `config_v1_1` pins a newer dataset revision and uses a different judge.
- An empty answer scores 0 without a judge call.
- v1.0 counts a malformed judge reply as wrong; v1.1 marks it `judge_failed`, which leaves the score. An empty judge reply is `judge_failed` in both.

## MMLU-Pro (`benchmarks/mmlu_pro`)

- A 10-choice prompt with `Answer: X` grading. Other prompts or answer extraction give different numbers, and MMLU-Pro is not MMLU.
