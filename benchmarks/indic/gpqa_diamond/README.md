<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond

GPQA Diamond translated into 14 Indic languages, with 198 multiple-choice questions
per language from [ai4bharat/indic-gpqa](https://huggingface.co/datasets/ai4bharat/indic-gpqa).

## Configuration

This benchmark uses the `mcqa` resource server and `simple_agent`, matching
[English GPQA](../../gpqa/config.yaml).

- **Grading mode**: `lenient_answer_colon_md` (`Answer: A/B/C/D` extraction).
- **Prompt**: `benchmarks/prompts/eval/aai/mcq-4choices.yaml`.
- **Responses per question**: 8.
- **Metric**: `pass@1[avg-of-8]/accuracy` (mean accuracy, 0–100).

Choice positions are shuffled deterministically using the English question included
in the dataset, so corresponding choices stay aligned across languages. Generation
settings come from the model configuration.

## Usage

```bash
# Prepare data
gym eval prepare --benchmark indic/gpqa_diamond

# Run against a vLLM endpoint
gym eval run \
    --benchmark indic/gpqa_diamond \
    --model-type vllm_model \
    --model MODEL_NAME \
    --model-url http://HOST:PORT/v1 \
    --model-api-key dummy \
    --output results/indic_gpqa/rollouts.jsonl
```

## Language Selection

By default, preparation includes `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`,
`or`, `pa`, `sa`, `ta`, `te`, and `ur`. English (`en`) is optional.

Prepare one language for per-language scores, then run the evaluation command above:

```bash
gym eval prepare --benchmark indic/gpqa_diamond \
    "+prepare_script_args={languages:[hi]}" \
    +use_cached_prepared_benchmarks=false
```
