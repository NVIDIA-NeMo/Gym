<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic AIME 2026

Translated AIME 2026 using the same evaluation components as Gym's
[`aime26`](../../aime26/config.yaml): the shared generic math prompt,
`simple_agent`, and `math_with_judge` resource with the judge disabled by default.
Only the problem text changes across languages. Each attempt generates one answer;
there is no formatting-repair call or separate Indic answer parser.

The defaults are four independently seeded attempts, a 120,000-token output limit,
thinking enabled, temperature 1.0, top-p 0.95, and top-k 64. Model-specific sampling
overrides must be identical for the English and translated runs.

## Metrics

Use Gym's `pass@4/symbolic_accuracy` for the fraction of questions with at least
one correct answer in four attempts. `pass@1[avg-of-4]/symbolic_accuracy` is the
average accuracy across those attempts. Both use a 0–100 scale. Extraction,
symbolic verification, majority metrics, and aggregation come directly from the
English AIME resource. Check rollout coverage before reporting a complete score.
Run each language separately for per-language results.

This replaces the earlier MathArena-specific prompt, format repair, parser, and
`matharena_aime/*` metrics. Previously prepared rows and results are not comparable
to this profile. Re-prepare with caching disabled and rerun evaluations.

## Dataset and alignment

| Source | Pinned revision | Split |
| --- | --- | --- |
| `anushakamathofficial/indic_aime_2026` | `938c1c90c23b25ca0f43d1bfc5103b332e8c033e` | `train` |
| `MathArena/aime_2026` | `d2de22f3c656b4f56cf8981212186377d1e23bc3` | `train` |

Preparation validates English text and answers against the canonical source and
checks translated problem IDs and answers. English is selected explicitly with
`languages: [en]`. The default 12 Indic languages contain 358 questions: Odia lacks
question 12 and Punjabi lacks question 15. Missing translations are reported in
the companion manifest and never replaced with English. Use the same question-ID
subset in each language when comparing paired scores. Translation quality flags
remain available for review. The source dataset license is CC-BY-NC-SA-4.0.

## Comparable English and translated runs

Run both through this configuration to keep inference settings identical. The
`en` source is checked against canonical English AIME, and the shared prompt and
verifier are exactly those used by `benchmarks/aime26`.

```bash
for language in en hi; do
  gym eval prepare --benchmark indic/aime_2026 \
    "+prepare_script_args={languages:[$language]}" \
    +use_cached_prepared_benchmarks=false

  gym eval run --benchmark indic/aime_2026 \
    --model-type vllm_model \
    --model MODEL_NAME \
    --model-url http://HOST:PORT/v1 \
    --model-api-key dummy \
    --split benchmark \
    --output "results/aime_2026/$language/rollouts.jsonl"
done
```

Use the same model/tokenizer revision and endpoint settings for both runs,
including the reasoning parser appropriate for the model. Gym supplies distinct
repeat seeds; do not pin a single seed at the model level. If using
`--benchmark aime26` directly, explicitly apply the same repeats, seed handling,
token limit, thinking, and sampling settings as this configuration.

```bash
pytest --import-mode=importlib benchmarks/indic/aime_2026/tests
gym env test --resources-server math_with_judge
```
