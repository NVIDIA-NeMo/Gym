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

The dependency-light [`task_data.py`](task_data.py) schema describes these fields
for Gym's dataset validation, including their required `verifier_metadata`
placement. It also declares prepared prompt fields and translation provenance.

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

Version `indic-formats-v2.10` also reads explicit option selections and true/false
verdicts within a concluding paragraph, accounting for every option label and
rejecting conflicting or conditional selections. Roman statement annotations do
not replace an explicitly selected A–D option. Numeric handling accepts agreeing
literal equality chains and a limited set of complete affirmative annotations
(for example, “as required” or “This is the required coefficient”). Unknown prose,
retractions, and conditions are not discarded. Every gap around an option verdict
must be a supported connective or complete annotation: a positive fragment cannot
license an unknown remainder, including an unrecognized Indic negation. A selected option followed by a
description must repeat the unique matching option text from the question, or
use a supported annotation, scalar value, or Roman statement list. If the
question explicitly ends by asking for the value of one typeset symbol, complete,
agreeing assignments to that symbol can identify the answer before a trailing
explanation of a different quantity expressed as a supported complete numeric relation.
Each assignment must start its line with the symbol or an explicit concluding
connective, and every other line in that extraction must be supported. This also
applies to unmarked final equations: blank lines do not turn a trial into a final
answer. Unsupported derivations need an explicit final-answer boundary or manual
review. Answer-marker payloads include subsequent lines, and text following a box
is checked rather than discarded. Assumptions and rejected or conditional
assignments remain unparsed. No equations are solved during extraction.
These rules apply to all models; remaining unsupported conclusions need review.
The stricter rules can lower extraction coverage and provisional scores relative
to earlier versions. Recompute from original responses without `--baseline-corrected` when
applying this fix; preserving earlier extractions would also preserve their bugs.

Rescore every audited, completed language shard in an existing campaign:

The campaign layout is an external evaluation convention, not the directory
produced automatically by `gym eval run`. Prepare one file per language with
the [benchmark preparation workflow](../../benchmarks/indic/jee_bench/README.md),
then collect each model/language run into the corresponding shard:

```text
campaign/
  data/
    hi.jsonl                 # Prepared input rows with unique uuid values
    bn.jsonl
  model-key/
    hi/
      rollouts.jsonl         # One automatic response per input uuid/task_id
      validation.json       # Operator's audited completion record
      DONE                  # Empty marker written only after the checks below
    bn/                     # No DONE while incomplete; excluded from rescoring
```

Only two-letter language filenames under `data/` are selected. Each rollout's
`verifier_metadata.task_id` must match an input `uuid`. Model keys are directory
names, not model IDs containing slashes. For each finished shard, the campaign
operator must validate all of the following before writing `DONE`:

1. Input and response identities agree exactly, without duplicates, missing
   rows, masked samples, or manually reviewed answers.
2. Re-extract each response's final-channel text with the **original run's frozen
   parser**, using `visible_answer`; exclude structured/tagged thinking. Confirm
   the stored answer, extraction flags and reward against that parser and
   `compute_score`. Record this as `post_thinking_scoring_audit: "passed"`.
3. Inspect the resolved configuration and captured outgoing model requests to
   verify the intended model, sampling parameters, token budget and thinking
   setting for the run. Record `sampling_audit: "passed"` only after checking
   that evidence; response counts alone do not establish sampling correctness.
4. Record the number of responses and the sum of their existing `reward` values
   (not a percentage) in `validation.json`, then create `DONE`.

For example, a three-response shard with rewards `1`, `0.25`, `0` has:

```json
{
  "responses": 3,
  "score_sum": 1.25,
  "post_thinking_scoring_audit": "passed",
  "sampling_audit": "passed"
}
```

Keep the original configuration, parser snapshot and request evidence with the
campaign. Gym does not generate these two operator attestations. Do not copy
the example's `"passed"` values without performing the audits. The rescoring
utility checks the attestations, identities, counts and source score sum; its
independent integrity audit below checks artifact preservation and score
arithmetic, not the original live sampling behavior.

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
the new summary. The baseline is re-audited at invocation, each source and
validation digest must match the current shard, and reused answers come from
the hash-verified corrected rollouts. `changes.jsonl` is a review aid and is
never a source of reusable answers.

Parser `indic-formats-v2.7` corrects option-selection and numeric-prefix bugs
in v2.6 and adds whole-word Indic uncertainty matching. To correct already
parsed v2.6 answers, rescore the original campaign **without**
`--baseline-corrected`; monotonic mode intentionally preserves successful
baseline extractions and therefore cannot repair their interpretation.

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
