<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond

This benchmark evaluates the 198 GPQA Diamond questions translated into 14 Indic languages in [`anushakamathofficial/Indic_GPQA_Diamond`](https://huggingface.co/datasets/anushakamathofficial/Indic_GPQA_Diamond). English is available as an explicit option and is excluded from the default language set. The source is gated and licensed under CC-BY-4.0; accept its Hugging Face access conditions before preparing data. Do not publish dataset examples or generated JSONL.

The preparer pins both the translated dataset and [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa). It checks that the published English rows match all five canonical question/answer fields in source order, then uses the canonical `Record ID` for stable identity. The translations do not publish independent record IDs, so their alignment relies on that validated row order.

The evaluation reuses `benchmarks/gpqa`: the shared English row formatter, `eval/aai/mcq-4choices.yaml` prompt, `simple_agent`, and `mcqa` verifier with `lenient_answer_colon_md` grading. Each question has eight responses, matching English GPQA. Report mean accuracy (`pass@1[avg-of-8]/accuracy`); the shared verifier also reports pass and majority metrics.

Choices start with the correct answer followed by the three distractors and use the English preparer's MD5-seeded shuffle. Translations seed that same helper with the canonical English question so corresponding options occupy the same positions in every language. The question and option text remain translated. All 198 questions, including row 69, are retained.

Generation settings, token budget, and thinking mode come from the selected model configuration, just as for English GPQA. Use identical model settings for both runs. Run each language separately for per-language scores; a combined run reports aggregate metrics.

This replaces the previous original-repository zero-shot profile. Regenerate prepared JSONL: old prompts, the custom parser, the `--shuffle-seed` option, and the one-response/1,000-token/no-thinking defaults have been removed. Earlier results are not directly comparable.

Prepare all default Indic languages:

```bash
uv run python -m benchmarks.indic.gpqa_diamond.prepare
```

Prepare a small gated-data smoke sample:

```bash
uv run python -m benchmarks.indic.gpqa_diamond.prepare \
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
