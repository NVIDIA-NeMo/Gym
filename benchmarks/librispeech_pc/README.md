# LibriSpeech-PC

ASR with Punctuation and Capitalization on the LibriSpeech-PC test splits
(`test-clean` ~2.4k utterances, `test-other` ~2.9k). Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server,
which provides the WER scoring.

## Splits

This benchmark exposes the `test_clean` split (~2.4k utterances).
`gym eval prepare` enforces one benchmark dataset per agent, so the
harder `test_other` split is left for a sibling benchmark dir as a future
PR. `prepare.py` accepts `--splits test-other` on the command line and
writes a separate `librispeech_pc_test_other.jsonl` if you want to
evaluate against that split via a custom config.

## Audio handling

Audio WAVs are downloaded by `prepare.py`, base64-encoded, and stored on
`responses_create_params.metadata.audio_data`. The `vllm_model` adapter
removes that metadata field and splices an `audio_url` content block into
the user message before forwarding it to vLLM Chat Completions.

This benchmark requires an audio-capable vLLM endpoint. The
`--model-type vllm_model` flag selects the Gym adapter; it does not launch
vLLM or change `policy_base_url`. Set `--model-url` to the vLLM server rather than
`https://api.openai.com/v1`. OpenAI Chat Completions uses `input_audio`,
while this adapter emits vLLM's `audio_url` content block.

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml).
`prompt_config` materializes them into `responses_create_params.input` at
rollout time, so `prepare.py` doesn't need to bake the messages into each
row.

## Prepare benchmark data

```bash
gym eval prepare --benchmark librispeech_pc
```

Downloads OpenSLR-145 manifests and OpenSLR-12 test-clean audio
and writes the JSONL into `benchmarks/librispeech_pc/data/`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --model-url http://<vllm-host>:<port>/v1 \
    --model <audio-capable-model> \
    --benchmark librispeech_pc
```

## Collecting rollouts

```bash
gym eval run --no-serve \
    --agent librispeech_pc_asr_with_pc_simple_agent \
    --output results/librispeech_pc_rollouts.jsonl \
    --num-repeats 4
```

## Verification

Per-rollout: `wer`, `wer_c`, `wer_pc`, `per`, and a binary
`is_correct = wer_pc < 0.5`. Corpus-level `wer` and sample-mean
`wer_c` / `wer_pc` / `per` are aggregated by `compute_metrics()` on
the `asr_with_pc` server.
