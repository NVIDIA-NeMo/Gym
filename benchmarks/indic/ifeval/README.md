<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IndicIFEval-Trans

Evaluates the translated track of
[`ai4bharat/IndicIFEval`](https://huggingface.co/datasets/ai4bharat/IndicIFEval),
using the existing English IFEval server, JSONL schema, and `simple_agent`.
The config inherits their shared settings. The `indicifeval_trans` backend selects language-specific
checkers and strict/loose scoring functions from the
[official harness](https://github.com/AI4Bharat/IndicIFEval/tree/1bb5f1bc4936cb20b8544e185bfd7cbed8f31464/lm-evaluation-harness/custom_configs/indicifeval-trans).
The existing English backend remains the default for other configurations.
The Indic config copies the shared verifier settings, keeping the English
verifier available when `--benchmark ifeval --benchmark indic/ifeval` are combined.

## Dataset and protocol

Only `indicifeval-trans` is downloaded. The supported languages are Bengali
(`bn`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Marathi (`mr`), Malayalam
(`ml`), Nepali (`ne`), Odia (`or`), Punjabi (`pa`), Tamil (`ta`), Telugu (`te`),
and Urdu (`ur`). Preparation accepts `ka`, `mar`, and `mal` as aliases for
`kn`, `mr`, and `ml`. English, Assamese, Sanskrit, and the Ground track are
outside this integration.

Preparation downloads the dataset's default revision without a version pin
or fixed row count. It defaults to the `correct` translation-quality tag;
`translation_quality:all` includes every Trans row, and
`translation_quality:parallel` requires both `correct` and `parallel`.
Original prompts, instruction IDs, kwargs (including nulls), source keys,
response-language hints, and tags are preserved. The split selects the checker;
the original kwargs still govern any requested response language.

Prompts are zero-shot user messages without an added system instruction.
Generation defaults match the upstream task YAML: temperature 0 and a
1,280-token output limit. The model server controls chat-template handling.
These defaults are stored on each Indic row, so selecting `--model-type` keeps
them and combining benchmarks does not change another benchmark's requests.
Explicit `--temperature`, `--top-p`, and `--max-output-tokens` flags override
the corresponding fields for the whole run. There is one attempt per prompt
in both the config and manifest; the canonical split is `benchmark`.
Dataset `uuid` combines track, language, and source key; numeric `id`
retains the original key. `subset_for_metrics` identifies the language.

## Usage

Run from the Gym repository root after the usual Gym installation:

```bash
gym eval prepare --benchmark indic/ifeval

# A language subset; the aliases ka, mar, mal also work.
gym eval prepare --benchmark indic/ifeval \
  '+prepare_script_args={languages:[hi,kn,mr,ml],translation_quality:correct}' \
  +use_cached_prepared_benchmarks=false

gym eval run --benchmark indic/ifeval --model-type vllm_model \
  --model YOUR_MODEL --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --split benchmark --concurrency 8 \
  --output results/indic_ifeval/rollouts.jsonl
```

The checked-in `data/example.jsonl` contains one real `correct` Trans prompt
per supported language. To run these 12 samples, start the configured servers,
then collect against the example file in a second terminal:

```bash
gym env start --config benchmarks/indic/ifeval/config.yaml --model-type vllm_model \
  --model YOUR_MODEL --model-url http://HOST:PORT/v1 --model-api-key dummy

gym eval run --no-serve --agent indic_ifeval_instruction_following_simple_agent \
  --input benchmarks/indic/ifeval/data/example.jsonl \
  --output results/indic_ifeval/smoke.jsonl --concurrency 2
```

For thinking models, explicitly choose a sufficient output budget, for example
`--max-output-tokens 50000`, and configure the model's reasoning parser at the
inference endpoint. Record that override and the endpoint's sampling settings
with the results: a 1,280-token default run is not comparable to a 50k-token run.

Bulk prepared JSONL is ignored by git. Use
`+use_cached_prepared_benchmarks=false` to refresh the dataset or change
preparation options. Upstream updates can change the prompts and row counts;
retain the prepared JSONL alongside results for reproducibility and use a new
rollout output path after refreshing it.

The scorer retains its separate harness revision and checksum to keep grading
rules stable. At startup it caches the selected Trans checker files under
`resources_servers/instruction_following/.indicifeval_trans/`, together with
upstream license notices. Subsequent starts reuse this cache. An offline
machine needs this cache, the prepared dataset, dependencies, and NLTK's
`punkt_tab` data provisioned beforehand. See
[scorer provenance](../../../resources_servers/instruction_following/indicifeval-provenance.md).

## Scores

Reward is prompt-level strict accuracy: 1 only when all instructions pass.
Setting a row's `verifier_metadata.grading_mode` to `fraction` instead returns
its fraction of strictly satisfied instructions, as in English IFEval.

Verification returns `prompt_level_{strict,loose}_acc` booleans and
`inst_level_{strict,loose}_acc` boolean lists. Aggregates include all four
metrics overall and under `language/<code>/`. The run summary's `key_metrics`
selects the four overall accuracies. Instruction accuracy is total passed
instructions divided by total instructions. Empty responses fail;
checker exceptions fail the affected instruction and appear in
`instruction_errors` and `checker_error_count`. Completed `<think>` and
`<thinking>` blocks are removed before scoring. Other checker behavior follows
the upstream harness, including its eight loose-scoring response variants.

## Checks

```bash
pytest tests/unit_tests/test_indic_ifeval.py tests/unit_tests/test_rollout_seed_metadata.py
gym env test --resources-server instruction_following +should_validate_data=true
gym env test +entrypoint=responses_api_models/vllm_model
pre-commit run --all-files
```

Preparation, config composition, CLI overrides, and metadata isolation are
tested in the core CI suite. Checker tests run in the instruction-following
server suite; request conversion tests run in the vLLM server suite. See
[validation evidence](validation.md) for preparation and real-model smoke results.

## Attribution

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), AI4Bharat.
Cite Thanmay Jayakumar, Mohammed Safi Ur Rahman
Khan, Raj Dabre, Ratish Puduppully, and Anoop Kunchukuttan,
*IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in
14 Indic Languages* (2026), [arXiv:2602.22125](https://arxiv.org/abs/2602.22125).
The harness repository uses MIT; its Google Research-derived checker files
retain Apache-2.0 notices.
