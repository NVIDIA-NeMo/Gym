<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond

This benchmark evaluates the 198 GPQA Diamond questions translated into 14 Indic languages in [`ai4bharat/indic-gpqa`](https://huggingface.co/datasets/ai4bharat/indic-gpqa). English is available as an explicit option and is excluded from the default language set. The translated dataset card declares Apache-2.0; the canonical English GPQA source is gated and licensed under CC-BY-4.0. Authenticate with a Hugging Face account that can access both repositories before preparing data. Do not publish dataset examples or generated JSONL.

The preparer pins both the translated dataset and [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa). It reads `train.parquet` once, aligns rows by `Record ID`, and checks all five English question/answer fields against the canonical source. Each language uses columns such as `Question_Hindi_translation`. Source explanations, annotator details, and other auxiliary columns are excluded from prepared tasks. The translated source is pinned to `c3c32b0a0ec7aeebe884c4c55c46730b4274a612`.

The evaluation reuses `benchmarks/gpqa`: `eval/aai/mcq-4choices.yaml` prompt, `simple_agent`, and `mcqa` verifier with `lenient_answer_colon_md` grading. Each question has eight responses, matching English GPQA. Report mean accuracy (`pass@1[avg-of-8]/accuracy`); the shared verifier also reports pass and majority metrics.

Choices start with the correct answer followed by the three distractors and use the English preparer's MD5-seeded shuffle. Translations use the canonical English question so corresponding options occupy the same positions in every language. The question and option text remain translated. All 198 questions, including row 69, are retained.

The English GPQA code is unchanged; the Indic preparer matches its row format and shuffle.

Generation settings, token budget, and thinking mode come from the selected model configuration, just as for English GPQA. Use identical model settings for both runs. Run each language separately for per-language scores; a combined run reports aggregate metrics.

This replaces the previous original-repository zero-shot profile. Regenerate prepared JSONL: old prompts, the custom parser, the `--shuffle-seed` option, and the one-response/1,000-token/no-thinking defaults have been removed. Earlier results are not directly comparable.

Regenerate existing JSONL after changing the dataset source: this release includes translation corrections, so scores can change.

Prepare all default Indic languages:

```bash
.venv/bin/python -m benchmarks.indic.gpqa_diamond.prepare
```

Prepare a small gated-data smoke sample:

```bash
.venv/bin/python -m benchmarks.indic.gpqa_diamond.prepare \
  --languages hi \
  --question-ids 0 69 \
  --output-path /tmp/indic_gpqa_smoke.jsonl
```

Run the benchmark after starting or configuring the vLLM policy endpoint:

```bash
.venv/bin/gym eval run \
  --config benchmarks/indic/gpqa_diamond/config.yaml \
  --agent indic_gpqa_diamond_agent \
  --model <model-name-or-path>
```
