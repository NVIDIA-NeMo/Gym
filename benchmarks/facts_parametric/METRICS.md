# FACTS Parametric — metrics and reading guide

## 1. What the benchmark measures

FACTS Parametric is one of the four sub-leaderboards of the FACTS Benchmark Suite (Google DeepMind, Google Research,
Kaggle). It asks a model closed-book factoid questions ("trivia-style" queries drawn from real user interest) whose
answers are confirmed to exist in Wikipedia, and measures how often the model recalls the fact from its parameters
without tools. Questions were adversarially filtered so that five open-weight models all failed them, then verified
by three human annotators (paper section 4.1.2).

- Threat/measurement model: parametric knowledge recall and calibrated abstention; no retrieval, no tools.
- Unit of analysis: one question, answered once; each answer receives three sampled grades.
- Public protocol scope: the 1,052-question public half. The 1,052-question private half is held by Kaggle and is
  only scored by Kaggle; nothing in this branch is a leaderboard submission and no private score is claimed.
- Slices: the public CSV's `topic` label (54 values; "other" covers 401 rows), used for diagnostic slices only.

## 2. Sources, pins, and licenses

| Item | Value |
|---|---|
| Paper | Cheng et al., *The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality*, arXiv:2512.10791v1, 11 Dec 2025, CC BY 4.0 (`paper/PAPER.md`; PDF SHA-256 `db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be`; local copy via `fetch_paper.py`) |
| Public data | Kaggle dataset `kaggle/facts-parametric-public-examples`, version 2 (2025-12-08), Apache 2.0; `FACTS-Parametric-public.csv` SHA-256 `23fdc39d681656c87c6790b158f3d3a54f335903691a3285599c67a1578439f8`, 1,052 rows, columns `url, query, answer, topic` |
| Reference implementation | Kaggle starter notebook `yulongt/facts-parametric-benchmark-starter-code`, version 10 (Kaggle staff author listed on the paper); grader template SHA-256 `9d6a61f9ce3b875b5f97d25305abe6c073911f97e98297780483dd84fd9b548c`; helper cell SHA-256 `ddb61f29a532e57d0dbf7701471b5aecbe3446fb7bfde5d33ccc6e44d138e033` (vendored verbatim in `upstream_control.py`) |
| Judge | Gemini 2.5 Pro (paper section 4.2: "standardize on Gemini-2.5-Pro as our sole judge", three sampled grades averaged) |
| Gym revision | `1e668906d2e69a9e8ee9aaafc60050a4025d9688` (upstream/main) plus this branch |

## 3. Metric dictionary

Grades: every grader reply is mapped to one of `correct`, `incorrect` (the prompt's MISTAKE), `not_attempted`,
`unknown` with the starter's substring rule (`INCORRECT` > `MISTAKE` > `CORRECT` > `NOT_ATTEMPTED` > else `UNKNOWN`).
Let G be the multiset of all grades over scored answers (3 per answer), N the number of scored answers.

| Metric | Formula / decision rule | Range, direction | Denominator, exclusions | Role |
|---|---|---|---|---|
| `accuracy` (paper "Accuracy") | #correct grades / \|G\| (= mean of per-answer CORRECT fractions) | 0-1, higher better | all grades of scored answers; answers whose grader call failed are in the failures sidecar and excluded | primary |
| `hedging_rate` (paper "Hedging rate") | #not_attempted / \|G\| | 0-1, neutral | same | component |
| `attempted_accuracy` (paper "Attempted accuracy") | #correct / (\|G\| - #not_attempted) = accuracy / (1 - hedging) | 0-1, higher better | grades that were not NOT_ATTEMPTED (MISTAKE and UNKNOWN count as attempted; this reproduces Table 6 to 0.1 pt) | component |
| `f1` (paper "F1") | harmonic mean of accuracy and attempted accuracy | 0-1, higher better | - | component |
| `mistake_rate`, `unknown_rate` | #incorrect / \|G\|, #unknown / \|G\| | 0-1, lower better / neutral | same | component / diagnostic |
| `strict_all_correct_rate` (starter `calculate_score`) | share of answers whose three grades are all CORRECT | 0-1, higher better | N | component (the starter's per-example score; the paper averages grades instead) |
| `reward` (per rollout) | CORRECT fraction of the three grades: 0, 1/3, 2/3, 1 | 0-1 | - | Gym reward; `mean/reward` = accuracy when every answer has 3 grades |
| `accuracy_ci95_low/high` | bootstrap (2,000 resamples, fixed seed) of the mean per-answer reward | - | N | uncertainty |
| `judge_parse_agreement_rate` | answers where the starter parse equals the closing `Output: [LABEL]` parse on all 3 grades | 0-1 | N | diagnostic (grader validity) |
| `judge_empty_grade_rate`, `judge_valid_rate` | empty grader replies / \|G\| (counted as UNKNOWN, as the starter does) | 0-1 | \|G\| | diagnostic |
| `generation_truncated_rate`, `generation_empty_rate` | answers cut by the output-token cap / with no visible text | 0-1, lower better | N | operational (the grader still grades them) |
| `accuracy/topic/<t>` | accuracy restricted to topic t | 0-1 | grades of that topic | slice |

Invalid handling: a grader transport/HTTP failure raises `JudgeError`; the row goes to `<run>_failures.jsonl` with
`_ng_failure_class=judge_failed`, is excluded from every metric, and is retried on `--resume`. Received-but-empty
grader content is labelled UNKNOWN (starter behaviour) and counted in `judge_empty_grade_rate`.

## 4. Paper/source field to Gym mapping

| Source | Gym |
|---|---|
| CSV `query` | row `question`; prompt `benchmarks/prompts/generic/default.yaml` (`user: "{question}"`) == starter `QUERY_TEMPLATE` |
| CSV `answer` | row `expected_answer`; grader placeholder `{gold_answer}` |
| CSV `url`, `topic` | row `source_url`, `topic` (slice) |
| model reply | `generation` = last assistant message after the model server splits reasoning off |
| starter grader loop (3 x `judge.prompt(..., seed=i)`) | `verify()` issues 3 chat-completion calls with `seed` 0/1/2 and provider-default sampling; receipts in `judge_receipts` |
| starter `extract_classification` | `judge_labels_starter` (canonical `judge_labels`) |
| closing `Output: [...]` line | `judge_labels_output_line` (diagnostic) |
| starter `calculate_score` | `all_correct` |
| paper "average of three sampled grades" | `reward`, `accuracy` |

## 5. Calibration

`calibrate.py` writes `calibration/upstream-vs-gym.jsonl` and `calibration/summary.json`:

- `replay`: every grader receipt of the run is re-parsed with the vendored starter functions and the starter score is
  recomputed; the Gym labels, reward and `all_correct` must agree case by case (this isolates parsing/aggregation
  from grader sampling).
- `live`: a deterministic stratified subset (all-correct, mixed, all-mistake, hedge, unknown, truncated/empty) is
  re-graded with the starter loop against the same grader endpoint; label multisets and scores are compared. The
  grader samples, so this measures label stability rather than exact equality.

Results for the Kimi K3 run are recorded in section 6 and in the run package.

## 6. Kimi K3 run fingerprint

_Filled from the run package (`manifest/run-manifest.json`, `calibration/summary.json`)._

<!-- fingerprint:start -->
- Run id `20260914a`; model `moonshotai/Kimi-K3`; finished `2026-09-14T06:51:57Z`; NeMo Gym revision `1e668906d2e69a9e8ee9aaafc60050a4025d9688`.
- Endpoint: OpenAI-compatible chat completions via a loopback proxy on 127.0.0.1 to the shared Modal Kimi K3 endpoint (moonshotai/Kimi-K3, 1,048,576-token context, reasoning enabled by the proxy).
- Sampling: temperature 0.0, top_p 1.0, reasoning enabled by the shared endpoint; limits: max_output_tokens 16384 per answer (reasoning tokens count against it on this endpoint); harness `simple_agent`; repeats 1.
- Judge/verifier: Gemini 2.5 Pro grader (three samples, seed 0/1/2, provider-default sampling), official starter GRADER_TEMPLATE (sha256 9d6a61f9...), starter substring label parse; judge identities observed: {'gemini-2.5-pro': 3156}.
- Coverage: 1052 of 1052 expected rollouts scored over 1052 tasks; judge failed 0, simulator failed 0, infrastructure failed 0, missing 0, duplicates 0, superseded sidecar attempts 0.

| Metric | Value | Numerator / denominator | Role |
|---|---:|---:|---|
| `accuracy` Accuracy (mean of three grades) | 65.7% (95% CI 62.9%–68.5%) | 2075 / 3156 | primary |
| `hedging_rate` Hedging rate | 16.6% | 523 / 3156 | component |
| `attempted_accuracy` Attempted accuracy | 78.8% | 2075 / 2633 | component |
| `f1` F1 (accuracy, attempted accuracy) | 0.72 | - | component |
| `mistake_rate` Mistake rate | 17.6% | 557 / 3156 | component |
| `unknown_rate` Grader-unknown rate | 0.0% | 1 / 3156 | diagnostic |
| `strict_all_correct_rate` All-three-grades-correct rate (starter score) | 64.8% | 682 / 1052 | component |
| `judge_parse_agreement_rate` Grader parse agreement (starter vs closing line) | 97.4% | 1025 / 1052 | diagnostic |
| `judge_empty_grade_rate` Empty grader replies | 0.0% | 0 / 3156 | diagnostic |
| `generation_truncated_rate` Truncated answers | 0.5% | 5 / 1052 | operational |
| `generation_empty_rate` Empty answers | 0.4% | 4 / 1052 | operational |
| `mean_output_tokens` Mean output tokens (reasoning + answer) | 1120.15 | - | operational |

- Calibration: replay of every grader receipt through the starter's verbatim extract_classification/calculate_score; plus a live stratified re-grade with the starter loop: 1084 of 1092 cases agree; replay: 1052 of 1052 cases agree; live: 32 of 40 cases agree; label-level agreement 92.5%; score agreement 36 of 40.
- Canary (run `canary-20260914`): Accuracy (mean of three grades) = 0.6667 over 24 of 24 rollouts; settings max_output_tokens 8192 (reasoning tokens count against it).
- Package: `facts_parametric-kimi-k3-20260914a`; checksums.sha256 SHA-256 `cd2830f6802113521d8c08aa204c118b8464a9126bb167701f67033c7aa86c53`; status: complete public protocol run, `verified: false` (single model baselined).
<!-- fingerprint:end -->

## 7. How to read the result

- The number to quote is `accuracy` with its bootstrap interval, over the public half only. It is the paper's
  primary metric computed with the paper's grader and prompt, so it is comparable in *protocol* to Table 6, but Table 6
  pools the public and private halves.
- `attempted_accuracy` rewards abstention; a model that hedges more can have lower accuracy and higher attempted
  accuracy at the same time (the paper's GPT-5 vs o3 example).
- Common false readings: treating `strict_all_correct_rate` as the paper metric (it is the starter's stricter
  per-example score); reading `unknown` grades as model errors (they are grader indecision or unparseable grader
  replies); comparing to leaderboard numbers that include the private half; interpreting truncated/empty answers as
  refusals (they are an output-cap artefact and are reported separately).

## 8. BLADE mapping

- D1 metrics: `pass_at_1` N/A (the official metric is a mean of three graded samples, not a binary pass);
  `primary` -> `accuracy`; consistency/oracle N/A (one rollout per question). Funnel: expected -> materialized ->
  answered -> graded -> scored.
- D2 anchor facts: outcome, coverage, invalid (grader parse/empty), infrastructure (truncation), slice (topic),
  example (correct / mistake / hedge / mixed grades), calibration; every fact cites task, rollout, and receipt ids.
- D3: `d3-shallow-baseline.md` (script tables) vs `BLADE-report.md` (tables + trace-linked examples + calibration).
- Not applicable, marked N/A rather than zero: pass@k, tool-call funnel, task root-cause taxonomy.

## 9. What belongs in the 1-2 page report

Primary `accuracy` with numerator/denominator and CI; `hedging_rate`, `attempted_accuracy`, `f1`,
`strict_all_correct_rate`; grader validity (`judge_parse_agreement_rate`, empty replies) and truncation; two to four
trace-linked examples (a correct answer, a mistake, a hedge, a grader disagreement if any); the replay/live calibration
counts; the Table 6 references labelled as public+private; the limitations in section 7; and the provenance footer
(dataset version and hash, grader identity, prompt hash, sampling, Gym revision, package hash).
