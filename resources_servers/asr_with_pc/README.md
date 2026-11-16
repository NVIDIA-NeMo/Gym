# ASR with PC (Word Error Rate)

Generic ASR / ASR-PC scoring for audio benchmarks. Reusable across benchmarks
that score word or character error rates; a benchmark selects the task type of
its rows (see below) in its prepared data.

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
  name with leading `text_` stripped (`text_tn` → `tn`). A row with
  `normalization: numb3rs` and a `category` is scored with the conservative
  category-based Numb3rs normalizer (case, prose punctuation, `oh`/`o` zeros and
  conventional initialisms are ignored; number words and numeric symbols are kept)
  instead of Whisper's, which would turn spoken numbers into digits on both sides.
- **`task_type: OpenASR`** — the Hugging Face Open ASR Leaderboard English
  protocol (its text normalizer, compound-word merging). Rows with
  `parent_id` and `chunk_index` are chunks of one long session: aggregation
  concatenates the raw hypotheses of a session in `chunk_index` order and
  scores them against the session reference once.
- **`task_type: Multilingual-ASR`** — language-aware WER, or CER when
  `use_cer` is set. Row fields: `language` (required, e.g. `de` or `de_de`),
  `use_cer`, `remove_diacritics` (default true), `normalize_compound`
  (default true unless English), `merge_compounds` (default false) and
  `cer_whitespace_mode` (`keep`, `all`, or `han_adjacent`, which ignores spaces
  next to Han characters).

An empty reference gives `wer: null` and `is_correct: null` for the edit-count
task types (`OpenASR`, `Multilingual-ASR`), and the row is left out of the corpus
rates.

The responses of `ASR-PC`, `ASR`, `Hallucination` and `ASR_LEADERBOARD` are the
the same as in the original server; the response only gained additive fields
(`raw_pred_text` and the edit counts below). The edit-count task types report
`wer_c`, `wer_pc` and `per` as `null` (the four original task types report
`0.0` for the ones they do not compute).

## Metrics

Per row, the edit-count task types return additive edit counts (`wer_errors` and
`wer_ref_words`; likewise `cer_*`) so that corpus rates pool edits instead of
averaging utterance rates: `100 * sum(errors) / sum(reference words)`.

- `corpus_wer` (the headline `wer`) pools `wer_errors` over the eval set; for
  rows without edit counts it is `jiwer.wer(refs, hyps)` over the normalized
  `text` / `pred_text`.
- `cer` pools its counts the same way.
- `wer::<name>`, `cer::<name>` break the pooled rates down by
  `benchmark_profile`, `subset_for_metrics` and `language`.
- `multilingual_macro_wer` is the Open ASR multilingual score: corpus WER per
  dataset, averaged over the datasets of a language, then averaged over
  languages (it is computed for rows whose `benchmark_profile` is
  `asr-leaderboard-multilingual`).
- `wer_c`, `wer_pc`, `per` and `hallucination_rate` are means over samples.
- `wer_<suffix>` (ASR_LEADERBOARD) is corpus-level over the matching
  `reference_fields` entry of every row.

Standard WER uses Whisper's English text normalizer + lowercase + punctuation
strip. WER_PC tokenizes punctuation as separate tokens so word boundaries and
punctuation errors both count.

## Generation-loop guardrail

Set `cleanup_generation_loops: true` in the server config to collapse runaway
repetition in the model output before scoring: a character repeated 20 or
more times, or a block of 1-4096 characters repeated 20 or more times, is
replaced by one copy. It is off by default and applies to every task type; the
unmodified output stays in `raw_pred_text`.

## Audio plumbing

Audio is carried separately from text content in `responses_create_params.metadata`.
The `vllm_model` adapter accepts one of three mutually exclusive fields:

- `audio_data`: a `data:audio/...;base64,...` URI embedded in the row.
- `audio_path`: one audio file path resolved by the model server.
- `audio_paths`: multiple audio file paths resolved by the model server.

The adapter removes the selected metadata field and splices an `audio_url`
content block into the user message before forwarding it to vLLM Chat
Completions. The committed example data uses `audio_data`.

This path requires an audio-capable vLLM endpoint. `--model-type vllm_model`
selects the Gym adapter; it does not launch vLLM or change `policy_base_url`.
Set `--model-url` to the vLLM server rather than `https://api.openai.com/v1`.
OpenAI Chat Completions uses `input_audio`, while this adapter emits vLLM's
`audio_url` content block.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --model-url http://<vllm-host>:<port>/v1 \
    --model <audio-capable-model> \
    --resources-server asr_with_pc
```

## Collecting rollouts (5-example smoke test)

```bash
gym eval run --no-serve \
    --agent asr_with_pc_simple_agent \
    --input resources_servers/asr_with_pc/data/example.jsonl \
    --output results/asr_with_pc_rollouts.jsonl \
    --num-repeats 1
```

## Regenerating example data

```bash
python resources_servers/asr_with_pc/generate_example_data.py
```

The committed `data/example.jsonl` uses 1-second silence WAVs as audio
placeholders — small enough to commit, sufficient for unit tests and schema
smoke tests. The actual benchmark JSONLs (with real audio) are built by each
benchmark's own `prepare.py`.

## Licensing

The server is Apache-2.0. `open_asr_normalizer.py` is adapted from the Hugging
Face Open ASR Leaderboard (Apache-2.0) at a pinned commit; see its file header
and `ATTRIBUTIONS.md`.
Scoring numbers depend on the pinned `jiwer`, `whisper-normalizer` and
`kaldialign` versions in `pyproject.toml`.
