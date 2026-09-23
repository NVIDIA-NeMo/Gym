<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic AIME 2026

Translated AIME 2026 from `ai4bharat/indic-aime-2026`: 30 problems per language
across 14 Indic languages. Uses the English [`aime26`](../../aime26) benchmark's
math prompt, `simple_agent`, and symbolic math verifier with the LLM judge disabled.
Preparation loads the dataset's `train` split with Hugging Face `load_dataset`,
then validates problem IDs, translations, and integer answers.

## Configuration

Defaults: four independently seeded responses per question, 120,000 output tokens,
thinking enabled, temperature 1.0, top-p 0.95, and top-k 64.

Generation defaults live in `responses_create_params`, so selecting `--model-type`
preserves them and `--temperature`, `--top-p`, and `--max-output-tokens` can override
them. vLLM receives top-k through `metadata.extra_body` and thinking through
`metadata.chat_template_kwargs`; both metadata values are JSON strings.

The dataset supplies four repeats. With a raw `--input` file instead of the
benchmark split, pass `--num-repeats 4` and the shared math `--prompt-config`.

`pass@4/symbolic_accuracy` measures questions answered correctly at least once in
four responses. `pass@1[avg-of-4]/symbolic_accuracy` measures average accuracy.
Both use a 0–100 scale. Use identical model and generation settings when comparing
English and translated results.

## Prepare data

```bash
gym eval prepare --benchmark indic/aime_2026
```

Defaults to `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`,
`te`, and `ur` (420 questions). Select a single language for per-language scores;
English is available as `en`:

```bash
gym eval prepare --benchmark indic/aime_2026 \
  '+prepare_script_args={languages:[hi]}' \
  +use_cached_prepared_benchmarks=false
```

## Collect rollouts

```bash
gym eval run --benchmark indic/aime_2026 \
  --model-type vllm_model \
  --model MODEL_NAME \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic_aime_2026/rollouts.jsonl
```

The AI4Bharat dataset declares Apache-2.0; the underlying MathArena data retains
CC-BY-NC-SA-4.0. Prepared data and its provenance manifest are generated locally.
