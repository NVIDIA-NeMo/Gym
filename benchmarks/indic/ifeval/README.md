<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IndicIFEval-Trans

Evaluates the translated track of
[`ai4bharat/IndicIFEval`](https://huggingface.co/datasets/ai4bharat/IndicIFEval),
using English IFEval's `instruction_following` resources server, JSONL schema,
and `simple_agent`. The `indicifeval_trans` backend selects the language-specific
checkers and strict/loose scoring functions from the
[official harness](https://github.com/AI4Bharat/IndicIFEval/tree/1bb5f1bc4936cb20b8544e185bfd7cbed8f31464/lm-evaluation-harness/custom_configs/indicifeval-trans).
The existing English backend remains the default for other configurations.

## Dataset and protocol

Only `indicifeval-trans` is downloaded. The supported languages are Bengali
(`bn`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Marathi (`mr`), Malayalam
(`ml`), Nepali (`ne`), Odia (`or`), Punjabi (`pa`), Tamil (`ta`), Telugu (`te`),
and Urdu (`ur`). Preparation accepts `ka`, `mar`, and `mal` as aliases for
`kn`, `mr`, and `ml`. English, Assamese, Sanskrit, and the Ground track are
outside this integration.

The dataset is pinned to `4343c1b174322e32cc785ef89fe117038fdad7a9`.
Following the upstream recommendation, preparation defaults to the `correct`
translation-quality tag. `translation_quality:all` includes all released
Trans rows; `translation_quality:parallel` requires both `correct` and
`parallel`. Original prompts, instruction IDs, kwargs (including nulls),
source keys, response-language hints, and tags are preserved. Split language
selects the checker; a prompt may explicitly request a different response
language, which its original kwargs continue to enforce.

| Language | Default `correct` rows | All Trans rows |
| --- | ---: | ---: |
| bn | 454 | 490 |
| gu | 482 | 490 |
| hi | 487 | 490 |
| kn | 457 | 490 |
| mr | 483 | 490 |
| ml | 490 | 490 |
| ne | 468 | 490 |
| or | 473 | 490 |
| pa | 474 | 490 |
| ta | 489 | 490 |
| te | 488 | 490 |
| ur | 486 | 490 |
| **Total** | **5,731** | **5,880** |

Prompts are zero-shot user messages without an added system instruction.
Generation defaults match the upstream task YAML: temperature 0 and a
1,280-token output limit. The model server controls chat-template handling.
Dataset `uuid` combines revision, language, and source key; numeric `id`
retains the original key. `subset_for_metrics` identifies the language.

## Usage

Run from the Gym repository root after the usual Gym installation:

```bash
gym eval prepare --benchmark indic/ifeval

# A language subset; the aliases ka, mar, mal also work.
gym eval prepare --benchmark indic/ifeval \
  '+prepare_script_args={languages:[hi,kn,mr,ml],translation_quality:correct}' \
  +use_cached_prepared_benchmarks=false

# All 5,880 released Trans rows across the 12 supported languages.
gym eval prepare --benchmark indic/ifeval \
  '+prepare_script_args={translation_quality:all}' \
  +use_cached_prepared_benchmarks=false

gym eval run --benchmark indic/ifeval --model-type vllm_model \
  --model YOUR_MODEL --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --split benchmark --concurrency 8 \
  --output results/indic_ifeval/rollouts.jsonl
```

The checked-in `data/example.jsonl` contains one real `correct` Trans prompt
per supported language. Use `--split example` for a 12-prompt smoke run.
Bulk prepared JSONL is ignored by git and can be reproduced from the pinned
source. Clear the prepared-data cache with
`+use_cached_prepared_benchmarks=false` when changing preparation options.

The scorer downloads and checksums the pinned harness archive once at server
startup. It caches only the selected Trans checker files under
`resources_servers/instruction_following/.indicifeval_trans/`, together with
upstream license notices. Subsequent starts reuse this cache. An offline
machine needs this cache, the prepared dataset, dependencies, and NLTK's
`punkt_tab` data provisioned beforehand. See
[scorer provenance](../../../resources_servers/instruction_following/indicifeval-provenance.md).

## Scores

Reward is prompt-level strict accuracy: 1 only when all instructions pass.
Setting a row's `verifier_metadata.grading_mode` to `fraction` instead returns
its fraction of strictly satisfied instructions, as in English IFEval.

Verification also returns all four upstream metrics:

- `prompt_level_strict_acc` and `prompt_level_loose_acc`: per-response booleans.
- `inst_level_strict_acc` and `inst_level_loose_acc`: per-instruction booleans.

Gym's aggregate metrics include their means overall and under
`language/<code>/`. Instruction accuracy divides total passed instructions
by total instructions, rather than averaging per-prompt fractions. Loose
scoring uses the upstream eight response variants. Empty responses fail;
checker exceptions fail the affected instruction and are recorded in
`instruction_errors` and `checker_error_count`. Completed `<think>` and
`<thinking>` blocks are removed before scoring; other text is preserved.

The upstream checker behavior, including its language-detection limitations
and handling of falsy kwargs, is retained. This integration has not yet been
baselined against model runs (`verified: false`).

## Checks

```bash
gym env test --resources-server instruction_following
pytest benchmarks/indic/ifeval/tests resources_servers/instruction_following/tests
```

The tests cover the requested language scope and aliases, Trans-only pinned
downloads, quality filters, metadata preservation, Indic sentence and keyword
checks, strict/loose differences, empty outputs, checker failures, HTTP metric
serialization, aggregation, and English regression behavior.

## Attribution

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), AI4Bharat.
Cite Thanmay Jayakumar, Mohammed Safi Ur Rahman
Khan, Raj Dabre, Ratish Puduppully, and Anoop Kunchukuttan,
*IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in
14 Indic Languages* (2026), [arXiv:2602.22125](https://arxiv.org/abs/2602.22125).
The harness repository uses MIT; its Google Research-derived checker files
retain Apache-2.0 notices.
