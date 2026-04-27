# LibriSpeech-PC

Deterministic ASR-with-punctuation-and-capitalization scoring (Word Error Rate) ported from
NeMo-Skills' `nemo_skills/evaluation/evaluator/audio.py::evaluate_asr_pc`. Operates on the
[LibriSpeech-PC](https://www.openslr.org/145/) test splits (manifests from OpenSLR-145, audio from OpenSLR-12).

## What it scores

Per-rollout: `wer`, `wer_c`, `wer_pc`, `per`, plus a binary `is_correct = wer_pc < 0.5`.
Corpus-level (in `compute_metrics`): `corpus_wer@k=N`, `corpus_wer_c@k=N`, `corpus_wer_pc@k=N`
— matches Skills' `AudioMetrics.get_metrics` aggregation (`jiwer.wer(refs, hyps)` over the
whole corpus, not the mean of per-sample WERs).

Standard WER uses Whisper's English text normalizer + lowercase + punctuation strip.
WER_PC tokenizes punctuation as separate tokens so word boundaries and punctuation
errors both count.

## Audio plumbing (v1)

Audio is base64-inlined into `responses_create_params.input` at prepare time as an
`input_audio` (or `audio_url` / data-URI) content block. Mirrors the `circle_click`
pattern; no model-runner code change needed. The follow-up is to factor path→base64
into a `responses_api_models/vllm_audio_model` for benchmarks with longer clips
(audiobench, asr-leaderboard).

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/librispeech_pc/configs/librispeech_pc.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=librispeech_pc_simple_agent \
    +input_jsonl_fpath=resources_servers/librispeech_pc/data/example.jsonl \
    +output_jsonl_fpath=results/librispeech_pc_rollouts.jsonl \
    +num_repeats=1
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3-Nano)
so `<think>…</think>` blocks are stripped before WER scoring; without it, truncated
rollouts can leak reasoning into the hypothesis text and inflate WER asymmetrically.

## Regenerating example data

```bash
python resources_servers/librispeech_pc/generate_example_data.py
```

The committed `data/example.jsonl` uses 1-second silence WAVs as audio placeholders —
small enough to commit, sufficient for unit tests and schema smoke tests. The actual
benchmark JSONL (~270 MB with real LibriSpeech audio) is built by
`benchmarks/librispeech_pc/prepare.py` on the cluster.
