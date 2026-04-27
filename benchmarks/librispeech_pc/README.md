# LibriSpeech-PC

ASR with Punctuation and Capitalization on the LibriSpeech-PC test splits
(`test-clean` ~2.4k utterances, `test-other` ~2.9k). Direct port of the Skills
benchmark at `nemo_skills/dataset/librispeech-pc/`. Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server, which
lifts WER computation from `nemo_skills/evaluation/evaluator/audio.py`.

## Splits

The benchmark exposes two `datasets:` entries:

- `test_clean` → `data/librispeech_pc_test_clean.jsonl`
  (Skills' `EVAL_SPLIT` default; the parity comparison runs against this one)
- `test_other` → `data/librispeech_pc_test_other.jsonl`
  (harder acoustic conditions, higher WER)

Select one at rollout time via `+dataset_name=test_clean` (or `test_other`).

## Audio handling

Audio WAVs are downloaded by `prepare.py`, base64-encoded, and stored on
`responses_create_params.metadata.audio_url`. The
[`vllm_audio_model`](../../responses_api_models/vllm_audio_model/) wrapper
reads that field and splices an `audio_url` content block into the user
message before forwarding to vLLM Chat Completions. The Responses API content
union has no audio variant, so audio cannot ride in `input.content` directly —
the metadata sidechannel is the workaround until the schema is extended.

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml).
`prompt_config` materializes them into `responses_create_params.input` at
rollout time, so `prepare.py` doesn't need to bake the messages into each row.
Strings match Skills' `nemo_skills/dataset/librispeech-pc/prepare.py`
byte-for-byte.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/librispeech_pc/config.yaml]"
```

Downloads OpenSLR-145 manifests and OpenSLR-12 audio (~1 GB over both
splits) and writes per-split JSONLs into `benchmarks/librispeech_pc/data/`.

## Running servers

```bash
config_paths="responses_api_models/vllm_audio_model/configs/vllm_audio_model.yaml,\
benchmarks/librispeech_pc/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=librispeech_pc_asr_with_pc_simple_agent \
    +dataset_name=test_clean \
    +output_jsonl_fpath=results/librispeech_pc_test_clean_rollouts.jsonl \
    +num_repeats=4
```

Pass `+dataset_name=test_other` to evaluate the harder split.

## Verification

Per-rollout: `wer`, `wer_c`, `wer_pc`, `per`, and a binary
`is_correct = wer_pc < 0.5`. Corpus-level `wer` (Skills' headline) and
sample-mean `wer_c` / `wer_pc` / `per` are aggregated by
`compute_metrics()` to match Skills' `AudioMetrics.get_metrics` exactly.
