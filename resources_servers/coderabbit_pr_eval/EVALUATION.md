# LLM-as-Judge Integration Evaluation

## Overview

This document summarizes the evaluation performed to validate the replacement of ROUGE-L with an LLM-as-judge for the `coderabbit_pr_eval` resource server. The new reward formula is:

```
reward = 1.0 if (tag_correct AND judge_score >= 4.0) else 0.0
```

The judge scores summaries on a 1-5 scale using the same strategy already validated in the `coderabbit-pr-eval-harness`.

## What Changed

| Component | Before | After |
|---|---|---|
| Summary metric | ROUGE-L F1 (threshold 0.3) | LLM judge score (threshold 4.0) |
| Reward formula | `tag_correct AND rouge_l >= 0.3` | `tag_correct AND judge_score >= 4.0` |
| Dependencies | `rouge-score` | None (uses existing model server) |
| Config fields | `rouge_l_threshold` | `judge_score_threshold`, `judge_model_server`, `judge_responses_create_params`, etc. |
| Response field | `rouge_l_f1: float` | `judge_score: Optional[float]` |

## Evaluation Methodology

### 1. Parsing Equivalence

Verified that the Gym implementation uses identical parsing logic to the harness:

- **TAG_PATTERN regex**: MATCH — `\[(review_needed_senior_swe|review_needed_junior_swe|skip_review)\]`
- **SUMMARY_PATTERN regex**: MATCH — extracts text between `## AI-generated summary` header and the next `##` or tag
- **`extract_tag()`**: MATCH
- **`extract_summary()`**: MATCH (including fallback to content-before-tag)
- **`strip_thinking()`**: MATCH (handles `<think>` and `<thinking>` blocks)

### 2. Judge Prompt Equivalence

Compared the Gym prompts against the harness constants in `pr_eval_harness/metrics/llm_judge.py`:

- **System prompt** (`JUDGE_SYSTEM_PROMPT`): MATCH — identical 1-5 scoring rubric
- **User prompt template**: MATCH — identical `{reference}` / `{candidate}` template
- **Score parsing regex** (`Score:\s*(\d+)`): MATCH

### 3. Reward Formula Validation Against Harness Benchmark Data

Applied the new Gym reward formula to the harness's per-sample results from the full benchmark run (`eval_results_super_full.json`, 4,040 samples, `nvidia/nemotron-3-super-preview`).

#### Benchmark Numbers (successful inferences: 2,435 / 4,040)

| Metric | Count | Rate |
|---|---|---|
| Tag correct | 1,731 / 2,435 | 71.1% |
| Judge score >= 4.0 | 2,263 / 2,435 | 92.9% |
| **Gym reward = 1.0** (both pass) | **1,624 / 2,435** | **66.7%** |
| **Gym pass@1** | | **0.6669** |

#### Judge Score Distribution (tag-correct samples)

| Score | Count |
|---|---|
| 5 (Excellent) | 922 |
| 4 (Good) | 702 |
| 3 (Adequate) | 94 |
| 2 (Poor) | 13 |
| 1 (Very Poor) | 0 |

#### Old vs New Reward Comparison

| Formula | Pass Count | pass@1 |
|---|---|---|
| OLD: `tag_correct AND rouge_l >= 0.3` | 1,161 (47.7%) | 0.4768 |
| NEW: `tag_correct AND judge_score >= 4` | 1,624 (66.7%) | 0.6669 |

#### Disagreement Analysis

| Category | Count | Interpretation |
|---|---|---|
| Both pass | 1,115 | Agreement |
| Both fail | 765 | Agreement |
| OLD pass, NEW fail | 46 | ROUGE matched on surface but judge found quality lacking |
| OLD fail, NEW pass | 509 | Good summaries with different wording that ROUGE missed |

The new judge is more permissive because it captures semantic quality rather than lexical overlap. The 509 samples that now pass are cases where the model produced good-quality summaries using different wording — exactly the limitation ROUGE-L has. The 46 samples that now fail had surface-level n-gram overlap without real quality.

### 4. Unit Tests

26 tests pass covering:

- Helper functions: `extract_tag`, `extract_summary`, `strip_thinking` (12 tests)
- Correct tag + judge score >= 4.0 (score 5) -> reward 1.0
- Correct tag + judge score == 4.0 (threshold boundary) -> reward 1.0
- Wrong tag -> reward 0.0, judge not called
- Correct tag + judge score < 4.0 -> reward 0.0
- Empty output -> reward 0.0
- Whitespace-only output -> reward 0.0
- Unparseable output (no tag) -> reward 0.0
- Judge parse failure (no "Score: X" in response) -> reward 0.0
- Judge exception (model server error) -> reward 0.0
- Thinking block stripping before parsing
- Missing verifier_metadata handled gracefully
- Response fields match expected schema
- Custom threshold is respected
- Server instantiation sanity check

### 5. Lint

`ruff check` and `ruff format --check` both pass clean.

## Integration Readiness Checklist

- [x] Parsing logic identical to harness (`extract_tag`, `extract_summary`, `strip_thinking`)
- [x] Judge system prompt matches harness `JUDGE_SYSTEM_PROMPT`
- [x] Judge user template matches harness `JUDGE_USER_TEMPLATE`
- [x] Score parsing regex matches harness `_parse_score`
- [x] Reward formula validated against 2,435 benchmark samples
- [x] Reward behavior is sensible: 66.7% pass rate, stricter than tag-only (71.1%), more permissive than ROUGE-L (47.7%)
- [x] Judge call skipped when tag is wrong (saves inference cost)
- [x] Graceful error handling: judge failures -> reward 0.0
- [x] Concurrency control via `asyncio.Semaphore` (configurable, default 64)
- [x] All 26 unit tests pass
- [x] Lint clean (ruff check + ruff format)
- [x] `rouge-score` dependency removed from `requirements.txt`
- [x] YAML config updated with judge fields
- [x] README needs update to reflect new reward formula

## Files Modified

1. `app.py` — Replaced ROUGE scorer with LLM judge client
2. `configs/coderabbit_pr_eval.yaml` — Added judge config fields
3. `requirements.txt` — Removed `rouge-score`
4. `tests/test_app.py` — Rewrote tests for judge-based reward
5. `prompt_templates/pr_summary_judge.txt` — New judge user prompt template
