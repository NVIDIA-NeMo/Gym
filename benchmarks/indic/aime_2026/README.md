<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Indic AIME 2026

This benchmark evaluates the translated
[`anushakamathofficial/indic_aime_2026`](https://huggingface.co/datasets/anushakamathofficial/indic_aime_2026)
dataset with the prompt, format-repair flow, and deterministic answer parser from
[MathArena AIME 2026](https://github.com/eth-sri/matharena/tree/b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b).
It uses no model judge.

## Dataset

The adapter pins both sources to immutable revisions:

| Source | Revision | Split |
| --- | --- | --- |
| `anushakamathofficial/indic_aime_2026` | `938c1c90c23b25ca0f43d1bfc5103b332e8c033e` | `train` |
| `MathArena/aime_2026` | `d2de22f3c656b4f56cf8981212186377d1e23bc3` | `train` |

The publisher calls the dataset split `train`; Gym uses it only for benchmark
evaluation. The default selection has 358 questions across Bengali, Gujarati,
Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Tamil, Telugu, and
Urdu. Odia lacks question 12 and Punjabi lacks question 15. English is available
only when explicitly selected. The source license is CC-BY-NC-SA-4.0.

Preparation validates every translated problem ID and answer against the pinned
MathArena dataset. It writes a companion manifest with source hashes, selected
languages, coverage, and prompt and adapter hashes. Missing translations are
reported and are never replaced with English.

## Protocol

Each question receives four independently seeded attempts. The prompt is the
MathArena boxing and integer-range instruction followed by the translated
problem. If strict parsing cannot find an answer, the agent makes one additional
call with MathArena's formatting-repair prompt and the full conversation. A
parseable but incorrect answer is not retried.

The final response is graded with MathArena's non-strict answer extraction and
symbolic comparison. The primary metric is `pass@4/accuracy`: a question counts
as correct when at least one of its four attempts is correct. Values use Gym's
0–100 percentage scale. Per-language metrics are reported as
`matharena_aime/language/<code>/pass@4/accuracy`, and
`matharena_aime/macro_pass@4/accuracy` weights languages equally.

Incomplete repeat coverage, parser failures, warnings, or truncated responses
make the result provisional. The server retains observed metrics and detailed
coverage counters but withholds the completed headline metric until all selected
measurements are present and review-free.

Defaults are four attempts, distinct repeat seeds, a 120,000-token output limit,
thinking enabled, temperature 1.0, top-p 0.95, and top-k 64. Override top-k when
the selected model requires a different sampling profile. A formatting repair is
a second call with the same sampling parameters and seed.

## Usage

From the repository root:

```bash
# Prepare Hindi. Omit --languages to prepare all 12 default languages.
python -m benchmarks.indic.aime_2026.prepare --languages hi

gym eval run \
  --benchmark indic/aime_2026 \
  --model-type vllm_model \
  --model-url http://POLICY_HOST:PORT/v1 \
  --model MODEL_NAME \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic-aime-2026.jsonl
```

The vLLM endpoint must use the reasoning parser appropriate for the model. Gym
adds repeat seeds 0, 1, 2, and 3 by default; do not add a fixed model-level seed.
Downloaded source data and prepared JSONL files remain ignored runtime artifacts.

Run the focused validation with:

```bash
pytest --import-mode=importlib \
  benchmarks/indic/aime_2026/tests \
  responses_api_agents/matharena_aime/tests \
  resources_servers/matharena_aime/tests
```
