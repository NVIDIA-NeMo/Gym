<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic MBPP

Original MBPP from `ai4bharat/indic-mbpp`: 500 Python tasks in each of 14 Indic
languages. Includes an English baseline (`en`) from the same source records.
Reuses Gym's [English MBPP prompt](../../mbpp/prompts/default.yaml), `simple_agent`,
and the existing [SciCodePile assertion verifier](../../../resources_servers/scicodepile/README.md).
The `indic_mbpp_environment_server` uses Gym's `legacy_agent` relay to route
episodes to `indic_mbpp_simple_agent`, following the upstream benchmark layout.
The relay does not change the prompt or scoring contract.

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
Saved verifier results retain a language-qualified `task_id`, such as
`Mbpp/hi/11`, so results can be grouped by language without joining the inputs.
The top-level input `language` and `uuid` are not returned by SciCodePile.
Older prepared files use `Mbpp/11`; regenerate them to obtain qualified IDs.
This metadata change does not change prompts, assertions, or rewards.

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
The source repository requires approved Hugging Face access. Authenticate with
`hf auth login` (or supply `HF_TOKEN`) using an account that can read
`ai4bharat/indic-mbpp`; authentication alone does not grant dataset access.

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

Use `task_ids` in `prepare_script_args` to select smoke-test tasks. Data is downloaded
from the dataset's default revision. Missing translations are rejected and generated
data remains local.
Preparation also writes an adjacent `.provenance.json` file recording the resolved
dataset commit, source parquet SHA-256, selected languages/task IDs, and row count.
The revision comes from the downloaded snapshot path, not a separate lookup of
the latest commit. Non-snapshot paths are recorded as `null`, with the checksum
still retained. No source revision is pinned. Keep this sidecar with the inputs
and results; cached prepared data does not automatically refresh upstream changes.

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

The canonical config does not set sampling, thinking, a seed, or an output-token
limit. Specify these in the model/run configuration, and use the same settings
for each language and the English baseline. The verifier timeout is not a model
generation-token limit.

## Paired real-model smoke

The standalone [smoke config](smoke.yaml) selects one attempt each for two task
IDs in English, Hindi, and Kannada after preparing the inputs below. It uses
`google/gemma-4-31B-it`, thinking enabled, temperature 1.0, top-p 0.95, top-k 64,
seed 0, and at most 8,192 output tokens (including reasoning).
See the [inspected smoke results](smoke-results.md) for the source/model revisions,
server settings, outputs, and rewards. This small check is not a benchmark score.

```bash
python -m benchmarks.indic.mbpp.prepare \
  --languages en hi kn --task-ids 11 367 \
  --output-fpath benchmarks/indic/mbpp/data/review-smoke.jsonl

gym eval run --config benchmarks/indic/mbpp/smoke.yaml \
  --agent indic_mbpp_simple_agent \
  --model google/gemma-4-31B-it \
  --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --split benchmark --concurrency 3 \
  --output results/indic_mbpp_smoke/rollouts.jsonl
```

Start a compatible vLLM endpoint with the `gemma4` reasoning parser. Do not append
`--model-type vllm_model` to this smoke command: the config already includes the
adapter, and reloading its defaults afterward can erase the sampling overrides.
Sampling/seed and thinking are scoped to the smoke model adapter, not global
request metadata. The smoke has one repeat; use a separate run configuration
with independent seeds when estimating multi-attempt metrics.

The dataset declares Apache-2.0.
