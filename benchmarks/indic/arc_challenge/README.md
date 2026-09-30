<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Indic ARC-Challenge

This experimental integration evaluates
[`anushakamathofficial/indic_ARC-Challenge`](https://huggingface.co/datasets/anushakamathofficial/indic_ARC-Challenge)
with the base-model protocol from the reference `ai2_arc` task: five
demonstrations, raw completion prompts, and conditional likelihood scoring of
answer letters.

## Protocol

- Multilingual test data is pinned to revision
  `7d8d96ef635a28a51809deeb6490257eecfe1611`. The default covers 1,172
  questions in each of 14 Indic languages: Assamese, Bengali, Gujarati, Hindi,
  Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu,
  and Urdu. English is available with `languages:[en]`.
- The canonical `allenai/ai2_arc` ARC-Challenge train and test files are pinned
  to revision `210d026faf9955653af8916fad021475a3f00453`. The multilingual
  dataset has test splits only, so five demonstrations come from the canonical
  English train split. The canonical test split is also checked against the
  published English rows before data is written.
- The default random sampler is reset to seed 42 for each language and advances
  in complete test-set order. Every prompt uses five training examples, even
  when `question_ids` prepares a subset.
- Prompts use actual newline characters and the headings `Question:` and
  `Answer:`. Questions and option text are stripped. Demonstrations end with the
  correct answer letter and are separated by two newlines.
- Choices are scored as continuations with one leading space (` A`, ` B`, ...).
  Their token log probabilities are summed without length normalization; the
  highest score wins and ties select the first choice.
- Raw prompts are the default (`render_chat_template=false`). The same
  `vllm_model` server exposes both generation routes and `/loglikelihood`.

ARC contains a few three- and five-choice rows and some source labels are
numeric. The adapter keeps every row and maps source labels to positional
letters (`A` through `E`) so the gold answer remains correct. This adapts the
released dataset to the reference YAML, which assumes four choices.

## Usage

Prepare Hindi from the Gym repository root. Omit `prepare_script_args` to
prepare all 14 Indic languages.

```bash
gym eval prepare --benchmark indic/arc_challenge \
  '+prepare_script_args={languages:[hi],fewshot_seed:42}' \
  +use_cached_prepared_benchmarks=false
```

Run a base model against a vLLM completions endpoint:

```bash
gym eval run --benchmark indic/arc_challenge --model-type vllm_model \
  --model YOUR_MODEL --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --split benchmark --concurrency 8 --output results/indic_arc_challenge/rollouts.jsonl
```

For Gemma base models, append
`+policy_model.responses_api_models.vllm_model.likelihood_add_special_tokens=true`
to match tokenizer special-token handling in the reference evaluation. The
endpoint must support `/tokenize` and `/v1/completions` with echoed log
probabilities.

Use `question_ids:[Mercury_7175875]` for a small preparation smoke test. To opt
into the model tokenizer's chat template, append
`+policy_model.responses_api_models.vllm_model.render_chat_template=true`; the
default remains the raw base-model prompt.

Run the focused checks with:

```bash
pytest --import-mode=importlib benchmarks/indic/arc_challenge/tests \
  responses_api_agents/multiple_choice/tests \
  responses_api_models/vllm_model/tests/test_loglikelihood.py
```
