# VoiceCodeBench

Transcription of code-like spoken entities (URLs, paths, commands, identifiers, IDs, amounts) in 300 recordings (5.6
hours, 85 speakers, 1,482 target entities of 26 types, 8 workflow domains). Scoring is format-invariant WER plus an LLM
entity verifier (CTEM: the share of target entities whose canonical value is recovered; TSR: the share of recordings
with every entity recovered).

- **Data:** [`besimple-ai/voice-code-bench`](https://huggingface.co/datasets/besimple-ai/voice-code-bench)
  `data/metadata.jsonl` (SHA-256 `a69c40387303ea70fe46160239266cdb64b6ad9a68ca160c7e12640b4afb74f5`) and
  `data/audio/*.wav`, revision `8edfa398aa5575816d940793e4e2b8c6430184da`.
- **Scorer:** the benchmark's own `scripts/voice_code_bench/{metrics,verifier}.py` at the same revision, vendored in
  `resources_servers/audio_eval/voice_code_bench/` (function bodies unchanged; see `ATTRIBUTIONS.md`).
- **License:** the `LICENSE` file of the dataset repository (the source of the scorer) makes everything except the audio
  MIT, and lets the audio be used and redistributed with the license text, but not for voice cloning. The manifest
  declares `MIT AND LicenseRef-VoiceCodeBench-Audio`. The dataset card's metadata says `license: other` and points to
  that same `LICENSE` file.

## Preparation

`prepare()` checks the metadata hash, the row count, ids, transcript layers, entity counts and per-domain counts against
the pinned release. The default is all 300 recordings (1.96 GB of wav, written unchanged to
`data/voice-code-bench/data/audio/`). `domains` (`technical_it_dev`, `contact_routing`, `retail_logistics_order`,
`finance_billing`, `healthcare_admin`, `legal_insurance_government`, `education_workplace`, `dense_mixed_stress`)
narrows the set, and `max_samples_per_profile=N` keeps the first N rows in file order (smoke runs; the audio of a
narrowed or capped set is fetched file by file). A cap or a domain selection gives a subset: its scores are not
comparable with the full benchmark.

## Profile

One profile, `voice-code-bench`: the transcription prompt in `profiles.json` (it lists the entity kinds as context
terms), the `greedy` request preset (temperature 0, thinking off, at most 256 output tokens), one repeat. The target
entities and transcript layers stay in the scorer-only source row. Generations are cleaned of runaway repetition
before scoring (`cleanup_generation_loops`).

## Metrics

Published as `audio/voice-code-bench/pass@1/<metric>` and, per domain, as `audio/voice-code-bench-<domain>/pass@1/<metric>`.
All rates are percentages.

| Key | Meaning |
| --- | --- |
| `wer` | Format-invariant WER of the pooled recordings (errors over reference words) |
| `ctem` | Percent of target entities the verifier found in the transcript; `ctem::<entity type>` per type |
| `tsr` | Percent of recordings in which every target entity was found |
| `substitutions`, `deletions`, `insertions`, `reference_words`, `hypothesis_words` | The pooled word counts behind `wer` |
| `correct_entities`, `total_entities`, `successful_tasks`, `entity_verified_entries` | The counts behind `ctem` and `tsr` |
| `verifier_id`, `verifier_config_digest`, `official_revision` | What produced the numbers (strings) |
| `no_answer` | Percent of recordings with an empty generation. Unlike the audio metrics of the other benchmarks (see the [audio_eval README](../../resources_servers/audio_eval/README.md)), it is not halved |

## Judge

The entity verifier is an LLM judge: the packaged verifier config `openai_gpt_5_5_v1` (reference model `gpt-5.5`,
structured JSON output, one call per recording). The server requires a `judge_model_server`; `config.yaml` points it at
a model server named `judge_model`, which you provide through any OpenAI-compatible endpoint by setting
`judge_base_url`, `judge_api_key` and `judge_model_name` (`gpt-5.5` for the reference setting) in `env.yaml`. The judge
request carries no model name or sampling parameters of its own beyond the packaged config's. A failed or unusable
verifier reply masks the row and the row is left out of the metrics (it is not scored as a wrong answer); scores from
another judge model are not comparable with the reference.

## Run

```bash
gym eval prepare --benchmark voice_code_bench
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, smoke runs with `+prepare_script_args.max_samples_per_profile=<n>`).

Scores have not been compared with the numbers published by the benchmark's authors; the WER lattice and the verifier
request and parsing are their code, but the transport (a Gym judge model server instead of their client) and the
generation settings are this repository's.
