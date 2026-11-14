<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic JEEBench

Evaluate [`ai4bharat/indic-jee-bench`](https://huggingface.co/datasets/ai4bharat/indic-jee-bench)
with the authors' zero-shot CoT prompt and partial-credit scoring. Reuses Gym's
`simple_agent`, model adapters, and metric utilities.
The small `jeebench` resources server handles all four question types.

## Author protocol

Reference: [dair-iitd/jeebench](https://github.com/dair-iitd/jeebench/tree/a9af8bf354887043bd280a617c8716ef54e7aaec)
and [Arora et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.468/).
The four type-specific instructions and question whitespace normalization match
`inference.py` exactly. Instructions remain English; the question, including its
options, comes from the selected language. Options are never shuffled.
Defaults match the authors' CoT settings: temperature 0, 2,048 output tokens,
one response per question. No extra boxed-answer instruction is added.

Generation defaults are stored in each prepared JEEBench row, so combined runs
preserve other benchmarks' sampling and metadata. Explicit run-level CLI flags
override these defaults. Regenerate older prepared inputs with
`+use_cached_prepared_benchmarks=false` to include the row-scoped defaults.

| Type | Score |
| --- | --- |
| `MCQ` | 1 for the correct option, otherwise 0 |
| `MCQ(multiple)` | 1 for the complete correct set; 0.25 per selected correct option for a proper subset with no wrong option; otherwise 0 |
| `Integer`, `Numeric` | 1 if the absolute difference is at most 0.01, otherwise 0 |

The scorer preserves `compute_metrics.py`'s Python float boundary behavior and
tolerance for **both** numeric types. The paper describes Integer as exact match;
its manual extraction rejects fractional or negative Integer answers. Our
extraction performs that type check before scoring. Exam negative marking,
self-refinement, few-shot prompting, and CoT+SC are separate experiments.

The authors **manually extracted answers**. Our conservative automated extraction
is an adaptation: it reads the final boxed value, explicit `Answer:` marker, or
bare final line, excluding reasoning. An explicit single-choice label may be
followed by its option text; ambiguous option lists are rejected. It does not scan arbitrary solution text
for a matching gold. Unparsed answers score zero and set `extraction_failed`;
all automatic results set `provisional: true`. Translated prose answers may need
manual extraction. These scores are not a reproduction of the paper's results.

Primary metric: `pass@1[avg-of-1]/jeebench_score`, the mean partial-credit score
on a **0–100** scale. `full_credit` and `extraction_failure` are reported separately,
with language, subject, and question-type breakdowns. If repeats are requested,
they are independent samples; generic majority voting is deliberately omitted
because it differs from the authors' per-option CoT+SC aggregation.

## Prepare

```bash
gym eval prepare --benchmark indic/jee_bench

# One language, or an English baseline using languages:[en].
gym eval prepare --benchmark indic/jee_bench \
  '+prepare_script_args={languages:[hi]}' \
  +use_cached_prepared_benchmarks=false
```

The source is pinned to `5ce209f9900a434ead4f040a52004f6fae8aa1f7`, split `test`.
Default languages: `as bn gu hi kn ml mr ne or pa sa ta te ur`. The snapshot
contains 515 original questions and 14 translation columns, producing 7,210
translated tasks. English (`en`, 515 rows) is explicitly selectable. Each language
uses `question_<Language>_translation`; English uses `question`. Paper, index,
subject, type, and gold are shared across languages on the source row.
IDs combine paper and question number, e.g. `JEE Adv 2016 Paper 1/1`.

The dataset includes human-reviewed translation corrections; see its
[dataset card](https://huggingface.co/datasets/ai4bharat/indic-jee-bench)
for coverage and caveats. It does not publish per-row `judge_pass_stage` values;
these remain null rather than implying a quality audit passed. Null translations
are reported as missing in the generated manifest; empty or malformed text is
rejected. No English fallback or quality filtering is applied. An explicitly
requested missing question fails preparation.
Generated data and manifests remain local and are gitignored.

Use `hf auth login` if Hub access requires authentication. A local download with
`test.parquet` at its root is also supported:

```bash
python -m benchmarks.indic.jee_bench.prepare --languages hi \
  --source-dir /path/to/indic-jee-bench
```

The source directory is configurable, not hard-coded to a cluster path. The local
manifest records the parquet SHA-256 as well as the requested revision; when using
a different snapshot, supply its commit with `--revision`.

This replaces the earlier `anushakamathofficial/indic_jee_bench` source and its
per-language parquet layout. Regenerate prepared inputs with
`+use_cached_prepared_benchmarks=false`. Revised translations and the two filled
Odia/Sanskrit cells mean earlier campaign scores must retain their old dataset
provenance; rescoring old responses does not evaluate the updated questions.

`--question-ids`, `--revision`, and `--output-fpath` are also available, or pass
their underscore equivalents through `prepare_script_args`.

## Evaluate and smoke-test

```bash
gym eval run --benchmark indic/jee_bench --split benchmark \
  --model-type vllm_model --model google/gemma-4-31B-it \
  --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --output results/indic_jee_bench/rollouts.jsonl
```

For a bounded smoke, prepare the four types in Hindi and English:

```bash
gym eval prepare --benchmark indic/jee_bench \
  '+prepare_script_args={languages:[hi,en],question_ids:["JEE Adv 2016 Paper 1/1","JEE Adv 2016 Paper 1/24","JEE Adv 2016 Paper 1/50","JEE Adv 2018 Paper 1/7"]}' \
  +use_cached_prepared_benchmarks=false

gym eval run --benchmark indic/jee_bench --split benchmark \
  --model-type vllm_model --model google/gemma-4-31B-it \
  --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --num-repeats 1 --concurrency 4 --max-output-tokens 16384 \
  --output results/jee-smoke-rollouts.jsonl
```

The smoke token limit is an explicit override of the authors' 2,048-token budget.
Model thinking mode and endpoint configuration must be recorded alongside results.
Inspect extraction failures and truncation in the saved rollouts before reporting
scores. This integration remains experimental (`verified: false`).
After the smoke, prepare the full dataset again with
`gym eval prepare --benchmark indic/jee_bench +use_cached_prepared_benchmarks=false`
before collecting full benchmark results.

Run focused checks with:

```bash
pytest tests/unit_tests/test_indic_jee_bench.py
gym env test --resources-server jeebench +should_validate_data=true
gym env test +entrypoint=responses_api_models/vllm_model
```

Preparation, CLI overrides, combined-benchmark routing, and repeat seed tests live
under `tests/unit_tests/`, where core CI collects them. vLLM request translation
tests live in the model server suite; verifier and rescore integrity tests live
in the JEEBench server suite. This keeps server imports outside core coverage.

## Validation in this checkout

The evidence below used the previous `anushakamathofficial/indic_jee_bench`
snapshot, not the current AI4Bharat translations.

On 2026-10-11, `google/gemma-4-31B-it` (revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`) completed the eight-question Hindi/English
smoke through native `gym eval run`, with thinking disabled, temperature 0,
16,384 maximum output tokens, concurrency 4, and one sample per question.
The local source parquet snapshot supplied the questions. Every question type
was exercised in each language. Responses used 659–1,222 output tokens; all eight
answers were extracted from final-channel text and scored full credit, with no
truncation or extraction failures. This bounded automatic score remains
provisional and is not a full benchmark result.

The focused preparation, CLI, verifier, rescore-audit, and vLLM request suites
passed 199 tests. Preparation produced 7,208 translated rows, with missing
translations and source hashes recorded in the generated manifest. The PR
description records exact commands, live rollout hashes, and broader CI status.
