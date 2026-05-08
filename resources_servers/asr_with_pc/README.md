# ASR with PC (Word Error Rate)

Generic ASR / ASR-PC scoring for audio benchmarks. Reusable across benchmarks
that score Word Error Rate; the LibriSpeech-PC benchmark is the first
consumer, with `asr-leaderboard`, `numb3rs`, and `audiobench` as natural
follow-ups.

## What it scores

Server config dispatches per row on `task_type` (default at the server level,
overridable per-row in the verify request body):

- **`task_type: ASR-PC`** — full WER + WER_C + WER_PC + PER.
  `is_correct = wer_pc < 0.5`.
- **`task_type: ASR`** — standard WER only (Whisper-normalized, lowercased,
  no punctuation). `is_correct = wer < 0.5`.
- **`task_type: Hallucination`** — char-rate based (used by MUSAN). The
  request must carry `audio_duration` (seconds); the metric flags
  `char_rate > 1500` chars/min as hallucination. `expected_answer` is
  typically empty for this mode. `is_correct = not is_hallucinating`.
- **`task_type: ASR_LEADERBOARD`** — primary standard WER against
  `expected_answer` plus per-reference WER against each entry in the
  request's `reference_fields` (e.g. `["text_tn", "text_itn"]`). Emits
  `wer_<suffix>` and `is_correct_<suffix>` per field; suffix is the field
  name with leading `text_` stripped (`text_tn` → `tn`).

Aggregation:

- `wer` (corpus-level, the headline) via `jiwer.wer(refs, hyps)` over the
  whole eval set.
- `wer_c`, `wer_pc`, `per` are mean-of-per-sample.
- `hallucination_rate` (Hallucination) is sample-mean.
- `wer_<suffix>` (ASR_LEADERBOARD) is corpus-level over each row's
  `text_<suffix>` reference.

Standard WER uses Whisper's English text normalizer + lowercase + punctuation
strip. WER_PC tokenizes punctuation as separate tokens so word boundaries and
punctuation errors both count.

## Audio plumbing

Audio is carried separately from text content via
`responses_create_params.metadata.audio_url` (a data-URI string). The
`vllm_model` audio sidechannel reads that field and splices an `audio_url`
content block into the user message before forwarding to vLLM Chat
Completions. The Responses API content union has no audio variant, so audio
cannot ride in `input.content` directly — the metadata sidechannel is the
workaround until the schema is extended.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/asr_with_pc/configs/asr_with_pc.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=asr_with_pc_simple_agent \
    +input_jsonl_fpath=resources_servers/asr_with_pc/data/example.jsonl \
    +output_jsonl_fpath=results/asr_with_pc_rollouts.jsonl \
    +num_repeats=1
```

## Regenerating example data

```bash
python resources_servers/asr_with_pc/generate_example_data.py
```

The committed `data/example.jsonl` uses 1-second silence WAVs as audio
placeholders — small enough to commit, sufficient for unit tests and schema
smoke tests. The actual benchmark JSONLs (with real audio) are built by each
benchmark's own `prepare.py`.
