# MUSAN

Hallucination detection on the MUSAN (Music, Speech, and Noise) corpus.
The model is given non-target audio (music, noise, or non-target speech)
and prompted to "Transcribe the speech in this audio. If there is no
speech, do not output anything." Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server in
`task_type=Hallucination` mode, which scores
`char_rate = (len(hyp) / audio_duration) * 60`. A char-rate above 1500
chars/min counts as a hallucination.

## Categories

`prepare.py` materializes all three MUSAN categories (`noise`, `music`,
`speech`) into a single `musan_benchmark.jsonl`; per-row
`subset_for_metrics` is `musan_noise` / `musan_music` / `musan_speech`
for category-level breakdown.

## Audio handling

WAVs are downloaded from OpenSLR-17 (~11 GB, no API key) and referenced
from each row by absolute path on
`responses_create_params.metadata.audio_path`. `vllm_model`'s
file-path audio sidechannel reads that field and splices an `audio_url`
content block into the user message before forwarding to vLLM Chat
Completions. Base64 inlining would inflate the JSONL to tens of GB and
is avoided.

The default `audio_path` is `/data/musan/<category>/audio/musan_<category>_<idx>.wav`
which mirrors the layout `nemo_skills/dataset/musan/prepare.py` writes
(both pipelines share the same on-disk WAVs via the lustre `/data` mount).

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml).
`prompt_config` materializes them into `responses_create_params.input` at
rollout time, so `prepare.py` doesn't need to bake the messages into each
row.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/musan/config.yaml]"
```

Downloads the MUSAN OpenSLR archive, extracts it under
`benchmarks/musan/data/musan_openslr/musan/<cat>/`, and writes
`benchmarks/musan/data/musan_benchmark.jsonl`. Each row's `audio_path`
points at `/data/musan/<cat>/audio/musan_<cat>_NNNNNN.wav` by default
(matching Skills' container mount). Pass `--audio-root <path>` to
`prepare.py` if you need a different mount layout at rollout time.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/musan/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=musan_asr_with_pc_simple_agent \
    +output_jsonl_fpath=results/musan_rollouts.jsonl \
    +num_repeats=4
```

## Verification

Per-rollout: `char_rate`, `hallucination_rate`, and a binary
`is_correct = char_rate <= 1500`. Aggregate `hallucination_rate` and
pass@k metrics are computed by the `asr_with_pc` server's
`compute_metrics()`.
