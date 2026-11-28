# CS-FLEURS

Transcription of the four test sets of CS-FLEURS, a massively multilingual code-switched speech corpus (113 language
pairs over 52 languages): a matrix language with embedded English or, in `xtts-test2`, embedded other languages. Each
test set is one profile; all four are prepared by default (49,297 rows, about 150 hours of audio).

- Data: [`byan/cs-fleurs`](https://huggingface.co/datasets/byan/cs-fleurs), revision
  `0cdbf166c5517ae4b6eb1c54248522eedec53017`. Preparation checks the SHA-256 and row count of each pinned
  `metadata.jsonl` and downloads the clips unchanged (16 kHz audio, tens of GB). Origin: Yan et al., "CS-FLEURS: A
  Massively Multilingual and Code-Switched Speech Dataset", 2025.
- License: CC-BY-NC-4.0 (non-commercial). The dataset card's metadata says `cc-by-nc-4.0`; its text says "CC BY 4.0
  for Non-Commercial use". The manifest declares the metadata value. Check that your use is non-commercial.
- Profiles (rows): `cs-fleurs.read` (5,818, read speech), `cs-fleurs.mms` (14,932, concatenative TTS),
  `cs-fleurs.xtts-test1` (11,321) and `cs-fleurs.xtts-test2` (17,226), both generative TTS. Only the first is human
  speech; read the synthetic sets as stress tests of code-switching, not as a measure of natural speech. The reference is
  the code-switched transcript, and every profile uses the fixed prompt "Produce a verbatim transcript of the audio."
  with the `greedy` request preset.
- Scoring: the generic audio evaluator, task type `Multilingual-ASR` with the multilingual text normalizer (combining
  marks are kept so Thai, Lao, Burmese and Khmer vowel signs and tone marks count) and the generation-loop cleanup.
  The mixed error rate counts one unit per character for scripts without word boundaries (Chinese, Japanese, Korean,
  Thai, Lao, Burmese, Khmer) and one per word for the rest, within the same utterance; Thai, Lao, Burmese and Khmer
  (as matrix or embedded language) are segmented by grapheme cluster instead of by code point.
- Metrics, published as `audio/<profile>/pass@1/*` and `audio/<profile>-<pair>/pass@1/*` for each language pair such
  as `ara-eng`: `wer` is the mixed error rate (pooled edit counts, stored under the `wer` key, so it is not a word
  error rate), `wer::<pair>` the per-pair value, and `cer` a parallel character error rate over every row computed
  after lowercasing and removing punctuation (`lower_nopunct`). `no_answer` is half the share of empty generations, as
  in the other audio benchmarks. The overall number pools language pairs of very different sizes; compare per pair.
- Run: `gym eval prepare --benchmark cs_fleurs` (a smoke run, `+prepare_script_args.max_samples_per_profile=20`,
  downloads only the clips of the first rows of each test set), then see the
  [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark).
  `max_samples_per_profile` must be a positive integer.
- Limits: scores have not been compared with the CS-FLEURS paper; the first rows of each test set are not a random
  sample; `yue` and `ceb` have no ISO 639-1 code, so their rows are normalized without a language.
