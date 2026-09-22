<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic MBPP

Original MBPP from `ai4bharat/indic-mbpp`: 500 Python tasks in each of 14 Indic
languages. Includes an English baseline (`en`) from the same source records.
Reuses Gym's [English MBPP prompt](../../mbpp/prompts/default.yaml), `simple_agent`,
and the existing [SciCodePile assertion verifier](../../../resources_servers/scicodepile/README.md).

## Configuration

Each prompt contains the translated description, task setup when present, and the
first original assertion as an example. Four spaces are replaced with a tab, as
in English MBPP preparation. Reference solutions and the other two assertions are
kept out of the prompt. Each response must provide complete Python code.

Reward is 1 only when all three original `test_list` assertions pass. The existing
verifier runs each response in a fresh subprocess with a 120-second timeout.
`challenge_test_list` is excluded from the standard score. Default repeats: 1.
Use `pass@1/accuracy` for a single attempt; repeated runs also report pass@k and
`pass@1[avg-of-k]/accuracy`. Inspect coverage and `harness_failure` alongside scores.

Gym's existing `mbpp` benchmark is MBPP+ (378 EvalPlus tasks), while this dataset
contains the 500 original MBPP test tasks. Their scores are different benchmarks:
only 224 task IDs overlap, and 115 of those have revised English descriptions.
Use `indic/mbpp` with `languages: [en]` for a comparison on the same 500 tasks,
with identical model and generation settings. The existing `mbpp` benchmark and
its EvalPlus base/plus metrics are unchanged.

## Prepare data

Install Gym and follow the shared verifier's execution requirements. Generated
code runs as a subprocess; use an isolated evaluation environment as described
in the verifier README.

```bash
gym eval prepare --benchmark indic/mbpp
```

Defaults to `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`,
`te`, and `ur` (7,000 tasks). For per-language results, prepare and run one language
at a time; use `en` for the matching baseline:

```bash
gym eval prepare --benchmark indic/mbpp \
  '+prepare_script_args={languages:[hi]}' \
  +use_cached_prepared_benchmarks=false
```

Use `task_ids` in `prepare_script_args` to select smoke-test tasks. Data is pinned
to revision `64e7f7cecd2a6b66a0bcbc8c4c02200d40ae520c`. Missing translations are
rejected and generated data remains local.

## Collect rollouts

```bash
gym eval run --benchmark indic/mbpp \
  --model-type vllm_model \
  --model MODEL_NAME \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic_mbpp/rollouts.jsonl
```

The dataset declares Apache-2.0.
