<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# JEEBench verifier

Deterministic implementation of the authors' partial-credit `compute_score`.
See [Indic JEEBench](../../benchmarks/indic/jee_bench/README.md) for data preparation,
generation defaults, metric interpretation, and the manual-extraction limitation.
Upstream attribution and MIT terms are in [NOTICE](NOTICE).

Requests carry `verifier_metadata` with `task_id`, `language`, `subject`,
`question_type` (`MCQ`, `MCQ(multiple)`, `Integer`, or `Numeric`), and `gold`.
The optional `description` and `index` preserve original question identity.

Responses retain the raw generation and metadata, reward, extracted answer,
extraction method/failure, provisional status, and truncation status. Missing,
malformed, refusal-only, and reasoning-only responses receive zero. Infrastructure
failures remain governed by Gym's normal rollout error handling.

For manual extraction, native re-verification accepts `reviewed_answer` (including
the string `None` for no answer), a nonempty `answer_review_id`, and
`reviewed_response_sha256` produced by `response_fingerprint(response)`. Reviews
are rejected if the response has changed. These fields must describe a real
reviewer's extraction of the model's answer, not a correction to the gold answer.

```bash
gym env test --resources-server jeebench
```

The five committed `data/example.jsonl` rows are synthetic schema examples, not
benchmark questions. Live benchmark smoke tests use the selected source dataset.

## Corrected automatic extraction

`extraction.py` supports explicit Indic answer/option labels and conjunctions,
ordinary English answer declarations, LaTeX/Markdown formatting, numeric units,
fractions, scientific notation, Roman-numbered option lists, explicit
positive/negative option lists, and localized scalar conclusions. Single-choice answers given as option text
can match a unique option in the supplied question after formatting normalization.
Reference answers are never passed to the extractor. Units are stripped without
conversion; unsupported expressions and ambiguous conclusions require review.
The published scoring formula and the manual-review mechanism are unchanged.

Rescore every audited, completed language shard in an existing campaign:

```bash
python -m resources_servers.jeebench.rescore_campaign /path/to/campaign \
    --output /path/to/new-corrected-results
```

To extend an already audited correction without reinterpreting its successful
extractions, supply it as a monotonic baseline:

```bash
python -m resources_servers.jeebench.rescore_campaign /path/to/campaign \
    --baseline-corrected /path/to/audited-corrected-results \
    --output /path/to/new-corrected-results
```

This mode preserves every successfully parsed baseline answer and applies the
new parser only to baseline failures. The baseline must be complete and have a
passing `integrity_audit.json`; its hashes and parser version are recorded in
the new summary.

The output directory must be new. Smoke runs and shards without `DONE` are
excluded. Corrected rollouts retain the exact original response and an
`original_scoring` record. Per-shard changes and remaining-review files omit gold
labels for blind inspection. The summary includes coverage, partial-credit
scores, full-credit rates, extraction failures, regressions, and SHA-256
provenance. Original campaign artifacts and its frozen source are preserved.

Validate a completed corrected campaign independently:

```bash
python -m resources_servers.jeebench.audit_rescore /path/to/new-corrected-results
```

The audit checks source/output hashes, every preserved non-scoring field, original
score provenance, the unchanged scoring rubric, and aggregate arithmetic. This
integrity audit does not replace human review of ambiguous extracted answers.
