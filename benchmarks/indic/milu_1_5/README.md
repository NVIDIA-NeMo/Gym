<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MILU-1.5 post-training evaluation

`indic/milu_1_5` evaluates generated answers on the cleaned MILU-1.5 test split:
76,575 questions in 11 languages. It uses the single-turn `simple_agent` and the
`mcqa` verifier's `boxed_and_normalized` mode. Each response receives separate
strict boxed and normalized scores. It is a zero-shot post-training benchmark;
pre-training likelihood evaluation is outside this integration's scope.

## Protocol

- Source: `ai4bharat/MILU-1.5`, pinned to the immutable revision recorded by
  `SOURCE_REVISION` in `prepare.py`, license CC-BY-4.0.
- Zero-shot, one generated answer per question, chat formatting. The validation
  split is unused.
- The prompt uses an English instruction requesting an explanation ending in
  `the answer is \boxed{X}`, then the original-language question and options A–D.
  Only the question is stripped; option text and order are preserved. Line
  separators are real newlines.
- Gold labels map `option1`–`option4` to A–D. Stop strings are `</s>`,
  `<|im_end|>`, and `<eos>`.
- **Strict boxed:** reuse Gym's `strict_single_letter_boxed` parser. It reads the
  last `\boxed{...}` and accepts its supported letter formats, including
  `\boxed{B}`, `\boxed{\text{B}}`, and `\boxed{B: option text}`. An unboxed
  answer is invalid. This is Gym's strict boxed convention, not byte-exact matching
  of only `\boxed{B}`.
- **Normalized:** preserve any valid strict answer; otherwise normalize the last
  boxed payload, or the last `Answer:`/`answer is` payload, or a standalone answer.
  Reuse Gym's Unicode/letter/payload helpers to accept lowercase/fullwidth letters,
  Markdown and parentheses, and supported letter labels. A unique exact option-text
  match is checked after case/whitespace normalization and before letter substitutions,
  so letters inside option words are not treated as competing choices. A validated
  leading label is preserved when an explanation mentions other labels. Do not match
  an option as a substring of unrelated prose. Ambiguous alternatives are rejected,
  and an invalid explicit final payload does not fall back to an earlier answer.
- Both extractors operate independently of the gold label. A valid wrong boxed
  answer stays wrong in both scores. Correct letters score 1; wrong or unparseable
  answers score 0. Both metrics use the same questions and denominator.
- Parsers see assistant output text; separate reasoning items are not graded.
  Gym's default reasoning parser also moves inline `<think>...</think>` blocks
  into reasoning items. Keep the endpoint's reasoning parser enabled for thinking
  models so both Gym and the reference harness score the final content field.
- Gym reports `pass@1/strict_boxed_accuracy` and `pass@1/normalized_accuracy` on a
  0–100% scale. The verifier uses Gym's `BaseMultiRewardVerifyResponse` and shared
  pass/majority aggregation, with a separate extracted-answer key for each metric.
  `reward_components` contains both scores; the scalar `reward`, `extracted_answer`,
  and generic `pass@1/accuracy` use strict boxed scoring. The combined result is
  question-weighted. Prepare/run one language at
  a time for language scores. A table's language-macro average is the unweighted
  mean of all 11 language accuracies, which differs from the combined Gym result.
  Label averages over fewer languages provisional and record the completed count.

The historical `milu_v2` mode remains available explicitly for reference parity.
It uses the original first-match regex (boxed letter or `answer is B`), so old
MILU scores must not be relabeled as strict boxed or normalized scores. Prompts,
dataset, and generation settings are unchanged by the new grading mode.

The single user message has this exact format (no system message or few-shot
examples are added by the benchmark):

```text
Below is a multiple choice question. After explanation finish your answer with statement "the answer is \boxed{X}" where X is the correct letter from the below choices.
Please read the question carefully and choose the correct answer from the given options.
Question: {question}
Options:
A. {option1}
B. {option2}
C. {option3}
D. {option4}
```

The expected response is an explanation followed by, for example,
`the answer is \boxed{B}`. Evaluation compares the extracted letter against the
gold label; it does not score the explanation or use an LLM judge.

`checksums.yaml` contains SHA-256 digests from the verified download manifest.
Preparation checks both checksums and row counts, whether reading a local snapshot
or downloading the pinned revision. It processes one language at a time and only
replaces the destination after all selected languages have been written successfully,
so a failed preparation preserves the previous dataset. Prepared rows retain
language, domain, subject, translation status, source revision, and test row index.
Generated datasets are ignored by git; five synthetic examples are included. The
stop sequence list is a row-scoped request default in `metadata.extra_body`, so it
survives selecting `--model-type` and remains isolated from other benchmarks in a
combined run. Explicit run-level sampling overrides still take precedence.

## Prepare and run

Run from the `Indic_Gym` checkout with its virtual environment activated:

```bash
gym eval prepare --benchmark indic/milu_1_5 \
  '+prepare_script_args={dataset_dir:/path/to/MILU-1.5}' \
  +use_cached_prepared_benchmarks=false

gym eval run --benchmark indic/milu_1_5 --model-type vllm_model \
  --model YOUR_INSTRUCT_OR_THINKING_MODEL --model-url http://HOST:PORT/v1 \
  --model-api-key dummy --split benchmark --concurrency 8 \
  --num-repeats 1 --temperature 0 --max-output-tokens 16384 \
  --output results/milu_1_5/rollouts.jsonl
```

The endpoint must support `/v1/chat/completions`. Its tokenizer applies the
model's chat template. Choose generation limits and sampling settings appropriate
for the model and record them with the results; the example uses greedy decoding.
Thinking models may need a larger output budget. Configure the model adapter's
`chat_template_kwargs` for the endpoint's thinking mode when needed.
The benchmark does not prescribe `top_k`, `min_p`, or repetition penalty because
those settings are model-specific. When using vLLM-only request fields such as
`top_k`, pass them through `metadata.extra_body` together with the benchmark's
existing stop list; they are not standard Responses API fields.

Omit `dataset_dir` to download the pinned release. To select Hindi, add
`languages:[hi]` inside `prepare_script_args`. Supported codes are
`en bn hi ta te ml kn mr gu pa or` (Odia is `or`). To prepare a smoke subset,
also add `question_ids:["0","1","2"]`. IDs are zero-based row indices within
each selected language, and remain stable between subsets and the full run.
Subset preparation replaces the configured benchmark file; re-prepare with the
cache disabled before a full evaluation. Prepare separate files or use separate
checkouts when preparing languages concurrently; the configured destination is
shared within a checkout.

Direct preparation supports a separate output file:

```bash
python -m benchmarks.indic.milu_1_5.prepare \
  --dataset-dir /path/to/MILU-1.5 \
  --languages en hi --question-ids 0 1 2 --output-path /tmp/milu_1_5_smoke.jsonl
```

## Validation

```bash
PYTHONPATH=resources_servers/mcqa:. pytest --import-mode=importlib \
  tests/unit_tests/test_indic_milu_1_5.py \
  tests/unit_tests/test_milu_1_5_grading.py \
  tests/unit_tests/test_milu_1_5_reference_grading.py \
  tests/unit_tests/test_milu_1_5_rescore.py \
  resources_servers/mcqa/tests \
  responses_api_models/vllm_model/tests/test_milu_1_5.py
ruff check benchmarks/indic/milu_1_5 resources_servers/mcqa/app.py
ruff format --check benchmarks/indic/milu_1_5 resources_servers/mcqa/app.py
```

Tests cover prompt rendering, label mapping, subset stability, local and pinned
download preparation, checksum/count guards, native benchmark discovery, and
reference extraction edge cases, including stop-string forwarding through the
agent request schema and chat adapter in the vLLM server suite. Keeping server
integration tests in their owning suite prevents partial server imports from
distorting the separate core coverage report. Failure tests check that preparation errors
after partial serialization preserve the previous output and clean temporary files.

Dual-score tests additionally cover ambiguous/incorrect answers, normalization,
reasoning exclusion, HTTP serialization of both reward components, and independent
majority aggregation. Saved-output tests check both Gym and lm-eval formats, reject
incomplete/duplicate samples and legacy-score mismatches, and preserve source files.

All 76,575 rendered prompts, gold labels, and stop strings matched the supplied
reference, as did 107 historical `milu_v2` regex extraction cases. A real Gym
smoke run with `google/gemma-4-26B-A4B-it` (thinking enabled, temperature 0,
8,192 output-token limit) completed English, Hindi, and Bengali test row 0:
three boxed answers, three correct scores, no request failures or truncation.
Transport records confirmed the prompt, stop strings, and scores. Gym's generic
trajectory health checks reported `unobserved` because capture records were not
enabled; the actual requests and responses were checked separately.

This integration remains `verified: false` until full model baselines have
been reviewed. A smoke run is not a full score.

## Rescore saved responses

The MCQA verifier remains stateless and supports Gym's native reverification flow.
To also handle historical lm-eval files and build separate language-wise tables,
use the offline helper; it calls the same extractors as the live verifier and
does not generate new model responses:

```bash
python -m benchmarks.indic.milu_1_5.rescore \
  --manifest /path/to/saved-runs.json --output-dir /path/to/new-score-directory
```

The manifest identifies `dataset` and `dataset_revision` (the pinned values above)
and a `models` list. Each model has `key`, `model`, `settings`, and a `languages`
mapping keyed by the language codes above. Each language entry contains `format`
(`gym` or `lm_eval`) and `rollouts`; Gym entries also need `inputs`, the matching
`rollouts_materialized_inputs.jsonl`. Use absolute paths. Omit languages with no
finalized saved output.

The helper validates unique source indices, exact language coverage, original gold
labels, and the old reference rewards before publishing scores. It records source
hashes and per-question extracted answers, writes `strict_boxed.csv` and
`normalized.csv`, and includes provenance and example format differences in
`summary.json`. Full `Avg` is blank unless all 11 languages are complete; a separate
column reports the provisional average over completed languages. Original response
files are never rewritten. The two tables rescore each model's original run, so
differences in prompting or generation settings between those runs still apply.
