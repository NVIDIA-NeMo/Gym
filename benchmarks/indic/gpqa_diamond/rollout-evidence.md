<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic GPQA Diamond validation evidence

Validation date: 2026-10-11. The branch was rebased on upstream
`4a9236e48c82f0dc23365a4c0088b1e54d00d3a8`. These are integration checks, not full
benchmark results or estimates of extraction-error frequency.

## Real-data preparation

The pinned `ai4bharat/indic-gpqa` revision is
`c3c32b0a0ec7aeebe884c4c55c46730b4274a612`. Hub access still returned HTTP 404
with the available account. Validation therefore used the existing authorized
local `train.parquet`, with the downloaded revision and content hash checked
against its Hugging Face download metadata. No alternative dataset or synthetic
replacement was used for preparation or model rollouts.

- Source SHA-256: `34cd810650109f4c08b842e69dcadcff17da3a23c37a0f75d0d6a667b1e2c0cf`.
- Production `gym eval prepare`: 198 source records, 14 languages × 198 = 2,772
  prepared rows, with nonempty selected translations and no English fallback.
- Prepared JSONL SHA-256: `d1b329df74de1d63171b8c1aed4b02d04f40705220e48d3961d6a3a151667b2d`.
- Adapter SHA-256: `30344cbf5f93cca75696331cdedd0c876c5eb80c0c5a2614e93cbc80e8ac46c7`.
- Prompt SHA-256: `6f8b8d2d17e32c4fc314c8ff43534896cbf019556406b258e049c880c1b0e2e5`.
- `gym env validate indic/gpqa_diamond`: passed, including manifest/config
  agreement on `grading_mode: lenient_answer_colon_md` and all prepared rows.

See the [dataset-access instructions](README.md#prepare-data) for authenticated
Hub preparation and the hash-verified offline path. Authentication is not a
substitute for dataset permission; reviewers without access must obtain it from
the owner or arrange an approved transfer of the pinned file.

## Fresh native-model smoke

The smoke uses Bengali and Hindi source row 0, eight attempts per language,
through native `gym eval run`, `simple_agent`, the vLLM model adapter, and the
final MCQA verifier. It uses the unchanged English GPQA prompt with translated
question/choices, not a hand-written smoke prompt.

- Model: `gnani/gnani-evon-v3.3-30B-A3B`.
- Model revision: `fb9ddb6b58f2d235684388a9260671fca5ebccaf`.
- Serving: vLLM 0.21.0, model-specific `nano_v3` reasoning parser, one H100,
  TP=1, BF16 weights, FP8 KV cache, float32 Mamba SSM cache, context 131,072.
- Thinking enabled; temperature 0.6, top-p 0.95, top-k off (`-1`), repetition
  penalty 1.0; concurrency 4; seeds 0–7 per language.
- Explicit smoke output cap: 16,384 tokens, **not** the 120,000-token production
  evaluation setting. Reasoning and final-message output are inspected separately.

Final job `20020824` completed in 11m12s and released its GPU. **16/16 responses
completed, 16/16 answers extracted, 13/16 correct, 0 capped, 0 failed requests**.
All backend finish reasons were `stop`; output lengths were 712–11,195 tokens.
The two parsers give identical results on these English-labeled final answers:

| Language / source row | Seeds | Final line | Expected | Strict correct | Normalized correct | Extraction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Bengali / 0 | 0–7 | Five `Answer: B`, three `Answer: D` | B | 5/8 | 5/8 | 8/8 |
| Hindi / 0 | 0–7 | Eight `Answer: B` | B | 8/8 | 8/8 | 8/8 |

Captured outbound requests confirm every sampling value above and seeds 0–7
**per language**. All 16 responses contain separate reasoning items and final
assistant messages. Reverification with the final source code reproduces every
native extracted answer and reward, including the three incorrect Bengali
answers. Source hashes were checked before and after collection.

Artifact SHA-256 values:

- Rollouts: `561fd61ae0aace24d64c860dc092f053eabd1b88221caaa7243b4a211280f3c1`.
- Materialized inputs: `ee3dccfeabe1fb065baf5d6fa32b9445d452d7a00b69b9f48bc3198f11e7942e`.
- Transport: `de6d0faac6bb4d14c4d6f90f37070d1148ef4602d2bc9e16153935f54bdb154e`.
- Runtime config: `a57e088ffdb38f47607fbc46fc14aff7e22895547acd4a2e171d3c46c77c9ad0`.

Gym's automatic health report marked the 16 rollouts **unobserved**, not healthy:
the separate transport log is not Gym trajectory-capture input. The completion,
token, reasoning/final separation and extraction findings above come from the
explicit transport/materialized-input/response audit, not that health verdict.

Two earlier launcher attempts are excluded: one failed because `uv` was missing
from the child-server PATH; another was stopped after one response when a later
benchmark selector overwrote the small dataset/runtime config. The final attempt
asserted the resolved two-row dataset and sampling settings before model startup,
and added a 16-response safety limit. Neither required a shared Gym runtime change.

### Reproduce the bounded smoke

With the authorized source and a model endpoint using the serving settings above:

```bash
python benchmarks/indic/gpqa_diamond/prepare.py \
  --source-path "$GPQA_SOURCE" --languages bn hi --question-ids 0 \
  --output-path results/gpqa-smoke-inputs.jsonl
```

Save the following outer configuration as `gpqa-smoke.yaml`. The dataset override
must be in the outer config; adding a later benchmark selector can overwrite it.

```yaml
config_paths:
  - benchmarks/indic/gpqa_diamond/config.yaml
  - responses_api_models/vllm_model/configs/vllm_model.yaml
num_repeats_add_seed: true
policy_model:
  responses_api_models:
    vllm_model:
      chat_template_kwargs:
        enable_thinking: true
      sampling_overrides:
        temperature: 0.6
        top_p: 0.95
        top_k: -1
        repetition_penalty: 1.0
indic_gpqa_diamond_agent:
  responses_api_agents:
    simple_agent:
      datasets:
        - name: indic_gpqa_diamond
          type: benchmark
          jsonl_fpath: results/gpqa-smoke-inputs.jsonl
          prompt_config: benchmarks/prompts/eval/aai/mcq-4choices.yaml
          num_repeats: 8
```

```bash
NEMO_GYM_VLLM_TRANSPORT_LOG=results/gpqa-smoke-transport.jsonl \
  gym eval run --config gpqa-smoke.yaml \
  --model gnani/gnani-evon-v3.3-30B-A3B \
  --model-url http://HOST:PORT/v1 --model-api-key dummy \
  --split benchmark --output results/gpqa-smoke-rollouts.jsonl \
  --max-output-tokens 16384 --concurrency 4 --limit 16
```

Verify the resolved dataset path before allocating model compute. Inspect the
materialized-input identities and captured outbound requests: there should be
exactly 16 distinct language/seed pairs, with seeds 0–7 for each language and the
settings above. Capture files contain dataset inputs and model outputs: keep them
with the authorized run artifacts rather than committing them to this repository.

## Real cached Sarvam localized-answer evidence

Two existing real Bengali responses from `sarvam-external/sarvam105b-05102026`
(original job `20010832`) were reverified against the final PR verifier. They were
**not newly generated** for this review. Original settings: thinking enabled,
vLLM `qwen3` reasoning parser, temperature 1, top-p 1, top-k off, max output
120,000 tokens, seed 0. Both requests finished with `stop`, not a token cap.

The audit regenerated source rows 0 and 1 from the pinned Parquet, checked exact
prompt content and expected answers against the cached requests, and restored
options from the matched prepared inputs (verification responses do not retain
options). It then scored the same final content with the row opt-in absent and
present. The verifier makes no model or judge call.

| Language / row | Seed | Final line | Expected | Unchanged parser: extraction / reward | Opted-in parser: extraction / reward | Output tokens |
| --- | ---: | --- | --- | --- | --- | ---: |
| Bengali / 0 | 0 | `উত্তর: B` | B | None / 0 | B / 1 | 1,963 |
| Bengali / 1 | 0 | `উত্তর: B` | B | None / 0 | B / 1 | 7,786 |

This establishes a real localized-label case. It does not measure how often
localization occurs across a complete model evaluation.

Artifact SHA-256 values:

- Original Sarvam rollouts: `e268e68b3e9872cb2f719aa847499afe164f57c870fed2b1a218de653b6833d8`.
- Original Sarvam transport: `fdea7ca1105c01ea61a3c95521a365c6d3a4ac969058c9a9535ec2a07386dfbc`.
- Final MCQA verifier: `1a3fe52458359dc3d04bec330b121552bc4fa7d879bb5924fda4f0ac5ee666ab`.

## Compatibility checks

Regression tests cover the reviewer's `Eliminated option: B`, `(A): B`, and
prefix-enabled boxed-answer precedence examples, as well as ambiguous payloads,
trailing prose, and an opt-in that is missing, false, or the string `"true"`.
Existing English GPQA, MMLU-Pro, SuperGPQA, and MMLU-ProX configurations retain
their prior extraction and rewards. Synthetic tests establish these contracts;
they do not establish the frequency of these shapes in real outputs.

### Follow-up Environment Server and rescoring validation

The follow-up review's strict-routing failure was reproduced with English and
Indic GPQA composed in either order. The explicit
`indic_gpqa_diamond_environment_server` now fronts only
`indic_gpqa_diamond_agent`, inheriting the same `legacy_agent` implementation as
English GPQA. All six routing cases (Indic alone and both composition orders,
with strict routing on/off) pass without an automatically generated relay.
The 29 preparation/routing tests now live in
`tests/unit_tests/test_indic_gpqa_diamond.py` for normal core CI collection.

Validation of the follow-up:

- Targeted preparation, MCQA, global config, reverification, manifest/validation,
  prompt, benchmark discovery, and seed-metadata tests: **813 passed**.
- `gym env validate indic/gpqa_diamond`: passed against all 2,772 prepared rows.
- The README's strict-input transformation was checked for flat, `task_input`,
  and legacy `task_input.task_data` rows.
- Both documented `gym eval reverify` commands were exercised against the same
  16 saved Gnani smoke responses above: **16/16 reverified, 13/16 correct** in
  each mode, with no failed reverifications. Task/attempt sets, response values,
  and the original input/rollout hashes were unchanged. The input copy differed
  only in `allow_localized_answer_labels: false`.

No new model generation was needed for this follow-up: it declares routing for
the existing relay and documents CPU-only rescoring; it does not change the
prompt, parser, agent execution, generation settings, or benchmark scores.

Artifact location on the validation machine:
`commits/GPQA/pr3603-review-20261011/`, alongside the `commits/GPQA/Indic_Gym`
checkout. Raw authorized data, model outputs, and logs are intentionally not
included in Git.
