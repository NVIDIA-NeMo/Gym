# Preference-ASR

Speech recognition that follows a natural-language preference instruction (number style, entity spellings,
disfluencies, casing), scored with a preference-aware word error rate next to the standard one.

- Source: [`nvidia/preference-asr-bench`](https://huggingface.co/datasets/nvidia/preference-asr-bench), revision
  `038bd44722b25a17abc9ad53265c6a96be8195ab`: manifest `data/preference_asr_testset.jsonl` (SHA-256
  `134214055ef2b6f4ff88b5526072036be17a59e6a238472fd1f71d9b0c81c2d9`, identical to the one in the
  [companion repository](https://github.com/nithinraok/preference-asr-bench) at
  `efa5cae9aecb56ce28aeea6105425ded654acbdd`) and the audio of AMI, LibriSpeech and VoxPopuli. Common Voice audio is
  not redistributed by the release: it is read from [`hf-audio/open-asr-leaderboard`](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
  `common_voice` test split, revision `b6bdcd0beb34f8975dc659796176d88f43aff502`.
- License: CC-BY-NC-4.0, non-commercial use only (the license of the dataset card on Hugging Face). The companion
  repository states CC-BY-SA-4.0 for the annotations and Apache-2.0 for its code; the stricter non-commercial terms are
  the ones declared here. The source corpora are AMI and LibriSpeech (CC-BY-4.0), VoxPopuli and Common Voice (CC0).
- Only the public release is used (the larger seven-corpus draft of the paper was replaced): 2,822 instruction rows over
  1,997 clips (AMI 632, Common Voice 477, LibriSpeech 749, VoxPopuli 964; 21,600 s). Preparation checks the manifest
  hash, the per-corpus, per-category and per-direction census and the durations, and writes 16 kHz mono 16-bit WAV
  (about 0.7 GB). The reference is the preference text.
- Default: the whole release in both conditions, 5,644 rows. Profiles:
  `preference-asr-public.default` (prompt `Transcribe the English audio into text`) and
  `preference-asr-public.instructed` (the row's preference instruction, with two openings that made models stop at once
  reworded; the preference suffix is never changed). `corpora` (`ami`, `common_voice`, `librispeech`, `voxpopuli`),
  `categories` (`case`, `disfluencies`, `entities`, `normalization`, `standard`) and `variants` (`default`,
  `instructed`) narrow the set:

  ```bash
  gym eval prepare --benchmark preference_asr '+prepare_script_args.corpora=[ami,librispeech]' \
      '+prepare_script_args.categories=[entities]'
  ```

- Metrics, published as `audio/<profile>/pass@1/*` and again per preference category as
  `audio/<profile>-<category>/pass@1/*` (the group of the rows of that category): `standard_wer` and `preference_wer`
  (corpus WER in percent, total errors over total reference words, after the standard or the preference-aware
  normalization), their substitution/insertion/deletion/reference-word counts, `preference_wer_<category>` and
  `preference_wer_<corpus>` (and the `standard_` twins) for those present, `entries_<category>`, `coverage` (percent of
  rows with a prediction), `missing_predictions`, `scored_entries`, and `no_answer`, which is half of the share of empty
  generations (the convention of the other audio benchmarks, see the
  [audio_eval README](../../resources_servers/audio_eval/README.md)). Default and instructed rows are never pooled: each
  profile is scored alone.
- Scorer: the preference-aware normalizer skips only the step the instruction tests (number and symbol normalization,
  filler removal when fillers are to be kept, lowercasing for the case category). It is rebuilt from the
  `whisper-normalizer` package and has not been compared number for number with the released scoring script.
- Run: see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark);
  a smoke run still downloads and checks the manifest in full, clips are fetched one by one, and Common Voice clips need
  the matching row groups of the Open ASR leaderboard parquet files.
