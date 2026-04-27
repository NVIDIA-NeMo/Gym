# LibriSpeech-PC

ASR with Punctuation and Capitalization on the LibriSpeech-PC test splits
(test-clean ~2.6k + test-other ~2.9k). Direct port of the Skills benchmark at
`nemo_skills/dataset/librispeech-pc/`. Pairs with the
[`librispeech_pc`](../../resources_servers/librispeech_pc/) resource server,
which lifts WER computation from `nemo_skills/evaluation/evaluator/audio.py`.

## Audio handling (v1)

Audio WAVs are base64-inlined into `responses_create_params.input` at prepare
time as `input_audio` content blocks. Mirrors the `circle_click` pattern of
prep-time-baked multimodal content; no Gym model-runner change needed. Final
JSONL is ~270 MB across both splits.

The follow-up cleanup is to factor path→base64 inlining out of prepare.py
into a `responses_api_models/vllm_audio_model` so longer-clip audio benchmarks
(audiobench, asr-leaderboard) don't bloat their JSONLs at prep time.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/librispeech_pc/config.yaml]"
```

Downloads OpenSLR-145 manifests and OpenSLR-12 audio (~1 GB total over both
splits) into `benchmarks/librispeech_pc/data/`, then writes the base64-inlined
JSONL alongside.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/librispeech_pc/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=librispeech_pc_simple_agent \
    +input_jsonl_fpath=benchmarks/librispeech_pc/data/librispeech_pc_benchmark.jsonl \
    +output_jsonl_fpath=results/librispeech_pc_rollouts.jsonl \
    +num_repeats=4
```

Start vLLM with `--reasoning-parser <name>` (e.g. `deepseek_r1` for Nemotron-3-Nano)
so the WER scorer doesn't see `<think>…</think>` reasoning blocks.

## Verification

Deterministic: `wer`, `wer_c`, `wer_pc`, `per` per rollout, plus corpus-level
`corpus_wer`, `corpus_wer_c`, `corpus_wer_pc` aggregated via
`jiwer.wer(refs, hyps)` over all rollouts at each k. The corpus-level numbers
match Skills' `AudioMetrics.get_metrics` output exactly.
