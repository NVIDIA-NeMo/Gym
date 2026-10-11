<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond

GPQA Diamond translated into 14 Indic languages, with 198 multiple-choice questions
per language from [ai4bharat/indic-gpqa](https://huggingface.co/datasets/ai4bharat/indic-gpqa).
The source is `train.parquet`, split `train`, at revision
`c3c32b0a0ec7aeebe884c4c55c46730b4274a612`. Its SHA-256 is
`34cd810650109f4c08b842e69dcadcff17da3a23c37a0f75d0d6a667b1e2c0cf`.
Both Hub and local-file preparation require these exact bytes.

Preparation validates all 198 source records, unique Record IDs, and nonempty
question/answer translations for each selected language. Missing translations
are errors, never English fallbacks. Source whitespace is preserved. Explanations
and translation-review fields are not included in model input. Stable row UUIDs
are derived from the language and source Record ID.

## Configuration

This benchmark uses the `mcqa` resource server and `simple_agent`, matching
[English GPQA](../../gpqa/config.yaml).

- **Grading mode**: `lenient_answer_colon_md`, with recognized localized answer
  labels enabled only on non-English Indic rows.
- **Prompt**: `benchmarks/prompts/eval/aai/mcq-4choices.yaml`, zero-shot and a
  single user message. This is Gym's English GPQA prompt, not the original
  GPQA repository's baseline prompt or parser.
- **Responses per question**: 8.
- **Metric**: `pass@1[avg-of-8]/accuracy` (mean accuracy over all question/attempt
  pairs, 0–100; not best-of-eight). For one complete language the denominator is
  198 × 8 = 1,584. Unextractable answers remain incorrect, not excluded.

The user prompt is:

```text
Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

{question}
A: {choice A}
B: {choice B}
C: {choice C}
D: {choice D}
```

Choice positions are shuffled deterministically using the English question included
in the dataset, so corresponding choices stay aligned across languages. Generation
settings come from the model configuration and explicit run arguments: this
benchmark does not hard-code temperature, top-p, top-k, thinking, or token limits.
Record these with any reported score. The 120,000-token command below is an
explicit evaluation setting, not a preparation default. Reasoning extraction is
the model server's job; its model-specific vLLM reasoning parser is distinct from
MCQA final-answer extraction.

### Localized extraction and compatibility

Prepared non-English rows set
`template_metadata.allow_localized_answer_labels: true`. Only literal Boolean
`true` enables the fallback. English rows and existing English GPQA, MMLU-Pro,
SuperGPQA, and MMLU-ProX consumers do not opt in.

Existing custom-regex, English `Answer:`, row-provided answer-prefix, and
prefix-enabled boxed fallbacks keep their precedence. Only if these fail, the
final nonempty line can supply one allowed choice after a recognized label:
`উত্তৰ`, `উত্তর`, `જવાબ`, `उत्तर`, `ಉತ್ತರ`, `ഉത്തരം`, `ଉତ୍ତର`, `ਉੱਤਰ`,
`उत्तरम्`, `उत्तरं`, `பதில்`, `விடை`, `సమాధానం`, or `جواب`.
Markdown wrappers and a fullwidth colon are accepted. Arbitrary labels such as
`Eliminated option: B`, `(A): B`, multi-choice payloads, and trailing prose are not.

This is **format-normalized accuracy** for opted-in Indic rows. To report the
unchanged English-harness/strict extraction accuracy on the same generations,
reverify with that opt-in absent or `false`; do not regenerate responses or change
the denominator. Keep the two metrics separately labeled. Prepared data from
before version 2.2.0 must be refreshed to add the row opt-in. Existing cached
rewards are not automatically rescored.

## Prepare data

Use the development setup in the repository's contributing guide, then run from
the repository root. Generated data and `.manifest.json` provenance files under
`data/` are gitignored; do not commit or redistribute restricted dataset contents.

### Authorized Hub access

An account must actually have access to `ai4bharat/indic-gpqa`; logging in alone
does not grant permission. A 404 / `DatasetNotFoundError` can indicate unavailable
or inaccessible data. Ask the dataset owner for access if needed. Do not silently
substitute a different GPQA translation release.

```bash
hf auth login
hf download ai4bharat/indic-gpqa train.parquet --repo-type dataset \
  --revision c3c32b0a0ec7aeebe884c4c55c46730b4274a612 \
  --local-dir /path/to/authorized/indic-gpqa
gym eval prepare --benchmark indic/gpqa_diamond \
  +use_cached_prepared_benchmarks=false
```

### Offline, hash-verified local copy

Use `train.parquet` obtained through an authorized download or approved transfer
of the pinned revision. No Hub call is made with `source_path`. The hash check is
mandatory, and a mismatch fails before prepared data are written.

```bash
export GPQA_SOURCE=/path/to/authorized/indic-gpqa/train.parquet
export HF_DATASETS_CACHE=/tmp/indic-gpqa-datasets-cache
sha256sum "$GPQA_SOURCE"  # must match the hash above
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  gym eval prepare --benchmark indic/gpqa_diamond \
  "+prepare_script_args={source_path:'${GPQA_SOURCE}'}" \
  +use_cached_prepared_benchmarks=false
gym env validate indic/gpqa_diamond
```

The sidecar `data/gpqa_diamond_benchmark.manifest.json` records source revision,
source/prepared/adapter/prompt SHA-256 values, language selection, question IDs,
and per-language coverage. This makes local and Hub preparation auditable.

By default, preparation includes `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`,
`or`, `pa`, `sa`, `ta`, `te`, and `ur`. English (`en`) is optional.

Select one language for per-language scores or refresh an existing prepared file:

```bash
gym eval prepare --benchmark indic/gpqa_diamond \
  "+prepare_script_args={source_path:'${GPQA_SOURCE}',languages:[hi]}" \
  +use_cached_prepared_benchmarks=false
```

## Collect rollouts

For example, save this model-run configuration as `gpqa-runtime.yaml`:

```yaml
config_paths:
  - benchmarks/indic/gpqa_diamond/config.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
num_repeats_add_seed: true
policy_model:
  responses_api_models:
    vllm_model:
      chat_template_kwargs:
        enable_thinking: true
      sampling_overrides:
        temperature: 1.0
        top_p: 0.95
        top_k: -1
```

Use the model's recommended sampling and reasoning parser. For new Sarvam105b,
the requested sampling is temperature 1.0 / top-p 1.0 / top-k off; for the Gnani
thinking smoke it is temperature 0.6 / top-p 0.95 / top-k off / repetition penalty
1.0. These are model-run choices, not benchmark-wide overrides.

```bash
gym eval run \
    --config gpqa-runtime.yaml \
    --model MODEL_NAME \
    --model-url http://HOST:PORT/v1 \
    --model-api-key dummy \
    --split benchmark \
    --max-output-tokens 120000 \
    --output results/indic_gpqa/rollouts.jsonl
```

The outer runtime config includes the benchmark and model configs so its sampling
overrides take precedence. Do not append another `--benchmark` or `--model-type`
selector to this command: later sibling configs can overwrite runtime settings.

With seed injection enabled, the eight attempts use seeds 0–7. Select one
language at preparation time for per-language scores. A complete 14-language
run has 22,176 responses; a 12-language run excluding Assamese and Sanskrit has
19,008. Any small smoke or reduced output cap must be labeled separately from a
full 120k evaluation. See [rollout evidence](rollout-evidence.md) for validation
details and the limitations of the recorded examples.
