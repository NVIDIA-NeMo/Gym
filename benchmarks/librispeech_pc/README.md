# LibriSpeech-PC

ASR with Punctuation and Capitalization on the LibriSpeech-PC test splits
(`test-clean` ~2.4k utterances, `test-other` ~2.9k). Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server,
which provides the WER scoring.

## Splits

This benchmark exposes the `test_clean` split (~2.4k utterances).
`ng_prepare_benchmark` enforces one benchmark dataset per agent, so the
harder `test_other` split is left for a sibling benchmark dir as a future
PR. `prepare.py` accepts `--splits test-other` on the command line and
writes a separate `librispeech_pc_test_other.jsonl` if you want to
evaluate against that split via a custom config.

## Audio handling

Audio WAVs are downloaded by `prepare.py`, base64-encoded, and stored on
`responses_create_params.metadata.audio_url`. `vllm_model`'s audio
sidechannel reads that field and splices an `audio_url` content block
into the user message before forwarding to vLLM Chat Completions. The
Responses API content union has no audio variant, so audio cannot ride
in `input.content` directly — the metadata sidechannel is the workaround
until the schema is extended.

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml).
`prompt_config` materializes them into `responses_create_params.input` at
rollout time, so `prepare.py` doesn't need to bake the messages into each
row.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/librispeech_pc/config.yaml]"
```

Downloads OpenSLR-145 manifests and OpenSLR-12 test-clean audio
and writes the JSONL into `benchmarks/librispeech_pc/data/`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/librispeech_pc/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=librispeech_pc_asr_with_pc_simple_agent \
    +output_jsonl_fpath=results/librispeech_pc_rollouts.jsonl \
    +num_repeats=4
```

## Verification

Per-rollout: `wer`, `wer_c`, `wer_pc`, `per`, and a binary
`is_correct = wer_pc < 0.5`. Corpus-level `wer` and sample-mean
`wer_c` / `wer_pc` / `per` are aggregated by `compute_metrics()` on
the `asr_with_pc` server.
