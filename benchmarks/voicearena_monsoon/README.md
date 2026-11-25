# VoiceArena Monsoon

Transcription of the public test splits of VoiceArena Monsoon, conversational Indian English (2,102 clips, 1,444
speakers) and Hindi (753 clips, 468 speakers) recorded on phone handsets. Each language is one profile; both run by
default.

- `voicearena-monsoon.en-in.public`: [`VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN`](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN)
  test split, revision `bc1da7b42ef6e2853123c97bf6d22067e4802d11` (plain-text reference).
- `voicearena-monsoon.hi-in.public`: [`VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN`](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN)
  test split, revision `a4b7375e9adea16c16100295976cdb0aabd9fb6d` (reference: a lattice of accepted spellings per
  span; its first spelling of each span forms the plain `expected_answer`, which is informational only because the
  score uses the lattice).
- License: CC-BY-4.0 for both (dataset card metadata). The speaker-disjoint private halves and the separately licensed
  hosted VoiceArena corpus are not part of this benchmark.
- Preparation checks the row and speaker counts against the pinned revisions and converts each clip to 16 kHz mono
  16-bit WAV. The prompt asks for Latin script for English and Devanagari for Hindi words, with English words in their
  original spelling. `max_samples_per_profile` keeps the first N rows of each profile (smoke runs); it must be a
  positive integer.
- Metrics (`audio/<profile>/pass@1/*`): `wer` for English (Open ASR Leaderboard normalizer) and `oiwer` for Hindi (the
  [`voi-oiwer`](https://pypi.org/project/voi-oiwer) 0.1.4 lattice-aware error rate the dataset card prescribes), both as
  total errors over total reference words (corpus micro average, in percent), with the substitution, insertion and
  deletion counts, `coverage` and `missing_predictions`. The two profiles use different metrics and are never mixed.
  `exact` utterance accuracy is not reported. `num_entries`, `avg_tokens` and `gen_seconds` come from
  the shared metrics base class.
- Limits: a single greedy generation per clip (`max_output_tokens` 256); there is no judge.
- Run: see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark);
  a full run streams about 2 GB of parquet from the Hub.
