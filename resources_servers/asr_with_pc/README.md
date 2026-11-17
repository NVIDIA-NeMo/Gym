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
  conventional initialisms are ignored; number words and numeric symbols are kept;
  a dash between words is a space, so `twenty-five` equals `twenty five`)
  instead of Whisper's, which would turn spoken numbers into digits on both sides.
  The `category` is required (a row without one is rejected with HTTP 422).
- **`task_type: OpenASR`** — the Hugging Face Open ASR Leaderboard English
  protocol (its text normalizer, compound-word merging). Rows with
  `parent_id` and `chunk_index` (a non-negative integer, required with `parent_id`) are
  chunks of one long session: aggregation concatenates the raw hypotheses of a
  session in `chunk_index` order and scores them against the session reference once.
  Repeats are paired by `_ng_rollout_index`. A session attempt that lacks a chunk
  (a masked rollout, a gap or a repeated index in the chunk indices) is left out and
  counted in `open_asr_incomplete_sessions` instead of failing the aggregation; a
  session whose last chunks are all missing cannot be detected. Only `OpenASR` rows
  are merged. Each chunk row carries the whole session reference, so the per-row
  `reward` of a chunk row is not meaningful: use the aggregate metrics.
- **`task_type: Multilingual-ASR`** — language-aware WER, or CER when
  `use_cer` is set. Row fields: `language` (required, e.g. `de` or `de_de`),
  `use_cer`, `remove_diacritics` (default false, as in the Open ASR leaderboard's
  `MultilingualNormalizer(remove_diacritics=False)`), `normalize_compound`
  (default true unless English), `merge_compounds` (default false) and
  `cer_whitespace_mode` (`keep`, `all`, or `han_adjacent`, which ignores spaces
  next to Han characters; any other value is rejected at parse time). Locale codes
  such as `de_de` and `de-DE` select the language by their first part. Differences from
  the leaderboard's multilingual protocol: English rows (`en`, `en_us`, `en-US`) use
  Whisper's English normalizer, not the leaderboard's, and the number words of a
  language come from `num2words` at the locked version. A row without `language` is
  rejected with HTTP 422. Rows scored with `use_cer` count characters in `wer_*` and
  report `error_unit: char`.

An empty reference gives `wer: null` and `is_correct: null` for the edit-count
task types (`OpenASR`, `Multilingual-ASR`), and the row is left out of the corpus
rates.

Malformed rows (unknown `reference_fields` entry, missing `language` or `category`, a
`parent_id` without a valid `chunk_index`) answer HTTP 422 so that they are not retried
as infrastructure failures.

The per-row responses of the utterance-WER task types (`ASR-PC`, `ASR`,
`Hallucination`, `ASR_LEADERBOARD`) are the same as before; the response only gained
additive fields (`raw_pred_text`, `error_unit` and the edit counts below). `ASR` and
`ASR_LEADERBOARD` rows return edit counts as well. The edit-count task types
(`OpenASR`, `Multilingual-ASR`) report `wer_c`, `wer_pc` and `per` as `null` (the
utterance-WER task types report `0.0` for the ones they do not compute). The scoring
options of a row (`use_cer`, `remove_diacritics`, `normalize_compound`,
`merge_compounds`, `cer_whitespace_mode`) are not echoed into the response.

## Metrics

Per row, the edit-count task types return additive edit counts (`wer_errors` and
`wer_ref_words`; likewise `cer_*`) so that corpus rates pool edits instead of
averaging utterance rates: `100 * sum(errors) / sum(reference words)`.

- `corpus_wer` (the headline `wer`) pools word edits over the eval set: rows with
  edit counts add their `wer_errors` / `wer_ref_words`, and rows without counts
  (`ASR-PC`, `Hallucination`) add counts computed from their normalized `text` /
  `pred_text`, so a set that mixes task types is one corpus. Rows scored with `use_cer`
  count characters and are left out of it.
- `cer` pools the character counts of the rows that report them.
- `mean_utterance_wer`, `mean_utterance_cer` average the per-utterance rates of the
  rows with edit counts (chunked sessions count once, after merging).
- `wer::profile::<name>`, `wer::subset::<name>`, `wer::language::<name>` (and `cer::...`)
  break the pooled rates down by `benchmark_profile`, `subset_for_metrics` and
  `language`; a group's rate is in the unit its rows are scored in.
- `multilingual_macro_wer` is the Open ASR multilingual score: corpus WER per
  dataset, averaged over the datasets of a language (`macro_wer::language::<name>`),
  then averaged over languages. It is computed for rows whose `benchmark_profile` is
  listed in the server config's `multilingual_macro_profiles`, and replaces
  `corpus_wer` when every row belongs to such a profile.
- `wer_c`, `wer_pc`, `per` and `hallucination_rate` are means over samples.
- `wer_<suffix>` (ASR_LEADERBOARD) is corpus-level over the matching
  `reference_fields` entry of every row.

Standard WER uses Whisper's English text normalizer + lowercase + punctuation
strip. WER_PC tokenizes punctuation as separate tokens so word boundaries and
punctuation errors both count.

A reference that normalizes to empty (a filler such as "Uh.") has no edit counts and is
left out of `corpus_wer`, as in the Open ASR leaderboard. Before the edit counts were
added to `ASR` and `ASR_LEADERBOARD` rows, the hypothesis words of such a row counted as
insertions, so the corpus WER of a benchmark with fillers (for example AMI in
`asr_leaderboard`) is now lower. `Hallucination` rows (MUSAN, whose reference is empty) no
longer report `corpus_wer` / `wer`; they report `hallucination_rate`.

The server declares `REVERIFY_MODE = STATELESS`: `gym eval reverify` re-grades saved
results of this server without `++force`.

## Generation-loop guardrail

Set `cleanup_generation_loops: true` in the server config to collapse runaway
repetition in the model output before scoring: a character repeated 21 or
more times, or a block of 1-4096 characters repeated 20 or more times, is
replaced by one copy. It is off by default and applies to every task type; the
unmodified output stays in `raw_pred_text`. The block rule is a backtracking
regular expression, so it runs in a worker thread.

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

The server is Apache-2.0. `open_asr_normalizer.py` and parts of `speech_metrics.py`
are adapted from the Hugging Face Open ASR Leaderboard (Apache-2.0) at a pinned
commit; see their file headers and `ATTRIBUTIONS.md`.
Scoring numbers depend on the versions of `whisper-normalizer`, `kaldialign` and
`num2words` pinned in `pyproject.toml`, and on `jiwer`, which `pyproject.toml` leaves open
and `uv.lock` holds at 4.0.0.
