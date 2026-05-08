# ASR-Leaderboard

The 8-subset HuggingFace Open ASR Leaderboard test set
(`hf-audio/esb-datasets-test-only-sorted`): librispeech-clean,
librispeech-other, voxpopuli, tedlium, gigaspeech, spgispeech,
earnings22, ami. Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server's
`task_type=ASR` mode (Whisper-normalized WER).

## Audio handling

Audio FLACs are downloaded by `prepare.py` to the cluster-mounted
`/dataset/asr-leaderboard/data/<dataset>/<id>.flac` path. Each row
references the file via `responses_create_params.metadata.audio_path`,
and `vllm_model`'s audio sidechannel reads the file at request time and
splices it into the user message before forwarding to vLLM Chat
Completions.

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml).

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/asr_leaderboard/config.yaml]"
```

Downloads the 8 ESB subsets (~tens of GB of FLAC) and writes
`benchmarks/asr_leaderboard/data/asr_leaderboard_benchmark.jsonl`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/asr_leaderboard/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=asr_leaderboard_asr_with_pc_simple_agent \
    +output_jsonl_fpath=results/asr_leaderboard_rollouts.jsonl \
    +num_repeats=1
```

## Verification

Per-rollout: standard WER (Whisper-normalized) and binary
`is_correct = wer < 0.5`. Aggregated: corpus-level `wer` and per-rollout
`pass@k`/`majority@k` are produced by `asr_with_pc.compute_metrics()`.
