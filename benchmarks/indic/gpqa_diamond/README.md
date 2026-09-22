<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond

This benchmark evaluates the 198 GPQA Diamond questions translated into 14 Indic languages in [`anushakamathofficial/Indic_GPQA_Diamond`](https://huggingface.co/datasets/anushakamathofficial/Indic_GPQA_Diamond). English is available as an explicit option and is excluded from the default language set. The source is gated and licensed under CC-BY-4.0; accept its Hugging Face access conditions before preparing data. Do not publish dataset examples or generated JSONL.

The preparer pins both the translated dataset and [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa). It checks that the published English rows match all five canonical question/answer fields in source order, then uses the canonical `Record ID` for stable identity. The translations do not publish independent record IDs, so their alignment relies on that validated row order.

The evaluation follows the original GPQA zero-shot chat baseline:

- choice order starts with the three incorrect answers followed by the correct answer;
- Python's seeded shuffle runs sequentially over all 198 rows and resets for each language;
- the upstream system and user prompts are reproduced exactly, preserving source whitespace;
- one deterministic response is sampled with temperature 0, seed 0, a 1,000-token cap, and model thinking disabled;
- answer extraction uses the upstream ordered, case-sensitive regular expressions.

Unlike the original script's special handling for one long chain-of-thought prompt, row 69 is evaluated because this profile uses the zero-shot prompt. The Responses API carries structured reasoning separately; inline `<think>` or `<thinking>` blocks are removed before applying the upstream answer patterns so hidden reasoning is not scored.

Prepare all default Indic languages:

```bash
uv run python benchmarks/indic/gpqa_diamond/prepare.py
```

Prepare a small gated-data smoke sample:

```bash
uv run python benchmarks/indic/gpqa_diamond/prepare.py \
  --languages hi \
  --question-ids 0 69 \
  --output-path /tmp/indic_gpqa_smoke.jsonl
```

Run the benchmark after starting or configuring the vLLM policy endpoint:

```bash
uv run gym eval run \
  --config benchmarks/indic/gpqa_diamond/config.yaml \
  --agent indic_gpqa_diamond_agent \
  --model <model-name-or-path>
```

The existing `benchmarks/gpqa` and `benchmarks/gpqa-x` integrations use different prompts, option shuffles, repetition counts, and answer extractors. Their results are not directly comparable to this original-protocol profile.
