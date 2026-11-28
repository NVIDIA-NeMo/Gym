# Coval STT v1

Transcription of the fixed 50-utterance LibriSpeech test-clean set (50 utterances, 40 speakers, 393.265 s) of Coval's
public STT benchmark `stt-v1`.

- Definition: [`coval-ai/benchmarks`](https://github.com/coval-ai/benchmarks)
  `runner/src/coval_bench/datasets/manifests/stt-v1.json`, revision `db1e3d4c731b9051292fa36e73d9d2dc859372d3`, SHA-256
  `8c35e1d080e5482bb64eb8b9f39fd870ca334f9f5ca6d188680750715b6605e3`. Audio: 50 public 16 kHz mono WAV objects under
  `https://storage.googleapis.com/coval-benchmarks-datasets/stt-v1/`, each verified against the manifest SHA-256 and
  its WAV header.
- License: CC-BY-4.0 (stated by the manifest; LibriSpeech is CC BY 4.0 on [OpenSLR](https://www.openslr.org/12/));
  the Coval code is Apache-2.0.
- Protocol: the one `coval-stt-v1` profile uses the `greedy` request preset (temperature 0, top-p 1, seed 0, thinking off,
  at most 256 output tokens), one repeat, and the prompt "Audio transcription. Language: as spoken. Wording: verbatim,
  including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only." Generation
  loops are collapsed before scoring. Reference and hypothesis go through `whisper-normalizer==0.1.12`
  (`EnglishTextNormalizer`, the normalizer of the Coval runner), then one WER ratio is computed per utterance. The scorer
  is an independent implementation (own Levenshtein counts) of that protocol, not Coval code.
- Metrics (`audio/coval-stt-v1/pass@1/*`): `wer` is Coval's published quality metric, the mean of the per-utterance WERs in
  percent (utterances weigh equally); `corpus_wer_diagnostic` is the pooled WER (total errors over total reference words),
  kept as a diagnostic because it is not Coval's aggregation. Also `coverage` (percent of non-empty generations) and
  `missing_predictions`, `scored_entries`, and the pooled `wer_errors`/`wer_ref_words`/`wer_substitutions`/
  `wer_insertions`/`wer_deletions`. This scorer does not publish `no_answer`.
- `prepare()` downloads the pinned manifest and the 50 WAV objects, verifying every SHA-256, the 16 kHz mono PCM16 header,
  the duration of each file, and the census (50 rows, 40 speakers, 393.265 s total); any drift fails preparation.
  `+prepare_script_args.max_samples_per_profile=<n>` keeps the first n utterances (smoke runs; fetches only those).
- Limits: 50 utterances of read English speech make a small, easy test; each utterance carries 2% of the
  utterance-averaged WER, so compare models with care. Scores have not been compared with Coval's published numbers.
- Run: `gym eval prepare --benchmark coval_stt_v1`, then see the
  [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark).
