# HALAS

Transcription of the 745 test segments of HALAS, a human-annotated dataset of ASR hallucinations on Earnings-22 call
recordings. The benchmark measures how much a model adds to what was said: word error rate, and the share of words and of
clips that contain inserted words.

- **Annotations:** [`MatBar99/HALAS`](https://huggingface.co/datasets/MatBar99/HALAS) `test.csv`, revision
  `317cef3a10d1097edc37e8eb1c007f415e3d0c55`.
- **Audio:** [`distil-whisper/earnings22`](https://huggingface.co/datasets/distil-whisper/earnings22) `chunked` test split,
  revision `0a034f9ed86d33a3859d9025d3e621cf243773ab`, converted to 16 kHz mono 16-bit WAV. Only the referenced clips are read.
- **License:** CC-BY-SA-4.0 for the annotations (stated in the dataset card, whose metadata says `unknown`); CC-BY-4.0
  for the Earnings-22 audio.
- **Reference:** the corrected transcript, falling back to the original Earnings-22 one. Preparation fails if the row count
  or the duplicated-clip census differs from the released one (745 rows; one clip, `23_4484146.wav`, is used twice).

## Profile

One profile, `halas`: the `greedy` request preset (temperature 0, top-p 1, seed 0, thinking off, at most 256 output
tokens), one repeat, and the prompt "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and
repetitions. Formatting: punctuation and capitalization. Output: transcript only." Generation loops are collapsed before
scoring (`cleanup_generation_loops`). Text is normalized with the standard (Whisper) normalizer, so case and punctuation do
not count as errors.

## Metrics

Published as `audio/halas/pass@1/*`, and again under `audio/halas-test/pass@1/*` (the group of the `test` subset, which is
every row):

| Key | Meaning |
| --- | --- |
| `wer` | Corpus WER in percent (total errors over total reference words) |
| `insertion_rate` | Inserted words over reference words, percent |
| `insertion_sample_rate`, `hallucination_proxy_rate` | Percent of clips with at least one inserted word (the same number under two names) |
| `pred_words_per_minute` | Predicted words per minute of audio |
| `substitutions`, `insertions`, `deletions`, `ref_words` | The pooled edit counts |
| `no_answer` | Half of the share of empty generations, as in the other audio benchmarks (see the [audio_eval README](../../resources_servers/audio_eval/README.md)) |

An insertion-based proxy is not the human hallucination label: it flags any inserted word, including a harmless one.
The released per-system hallucination labels are carried in each row (`halas_released_system_*_count`) but are not scored.

## Run

```bash
gym eval prepare --benchmark halas
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, smoke runs with `+prepare_script_args.max_samples_per_profile=<n>`).

Scores have not been compared with the numbers of the HALAS release; the scorer is the generic audio WER evaluator
plus the insertion counts above.
