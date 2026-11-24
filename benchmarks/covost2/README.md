# CoVoST 2

Speech recognition and translation on the CoVoST 2 test set (Common Voice 4 read speech). Two profiles share one
prepared file: `covost2.asr` transcribes each clip, and `covost2.st` translates between English and 21 other languages.

- Data: [`fixie-ai/covost2`](https://huggingface.co/datasets/fixie-ai/covost2) `test` split of the 36 language-pair
  configurations (Parquet, original MP3 clips), revision `17c8c81e331e7a6929118121771a58c7ef7331d8`. Origin: CoVoST 2
  ([facebookresearch/covost](https://github.com/facebookresearch/covost)) over Common Voice 4; Wang, Wu and Pino, 2020.
- License: CC0-1.0, the license the CoVoST repository states for the data ("CoVoST data: CC0"); the audio is Common
  Voice, which is CC0. The `fixie-ai/covost2` mirror's dataset card declares no license of its own.
- Preparation checks every pair against the released clip count and unique ids, decodes each MP3, resamples 48 kHz to
  16 kHz with `soxr` (HQ) and writes a 16-bit WAV. English audio is taken once (from `en_de`) and shared by all
  `en->xx` rows. Decoding MP3 needs a `libsndfile` with MP3 support (the `soundfile` wheels bundle one). Parquet shards
  are read with range requests (about 5.2 GB of test audio for all languages, 4.3 GB for the card languages; listing
  all 36 pairs takes a few minutes); clips take about 20 GB (16.8 GB for the card languages).
- Default: **all 22 languages**: ASR for each language (112,311 clips) and translation for every pair with English
  that exists (21 `xx->en` and 15 `en->xx` pairs, 329,745 rows). The model-card protocol is 17 languages
  (`ar de en es et fr id it ja lv nl pt ru sl sv-SE tr zh-CN`, constant `CARD_LANGUAGES`): 92,901 ASR rows and the 26
  pairs among them, 232,680 translation rows. Corpus numbers over 22 languages are not comparable with the card:

  ```bash
  gym eval prepare --benchmark covost2 +prepare_script_args.languages=card
  ```

  `languages` also takes a list (`'+prepare_script_args.languages=[de,en]'`); a pair is kept when both languages are
  selected, so a translation direction needs `en`. Smoke runs: `+prepare_script_args.max_samples_per_subset=5` (a few
  rows of every language and pair) or `max_samples_per_profile=20`, usually with `languages=`.
- Prompts: ASR uses the fixed verbatim transcription prompt, translation `Please translate the given speech to
  <language>.`. Texts use the multilingual normalizer; BLEU is sacreBLEU corpus BLEU tokenized per target language
  (`13a`, or `zh`, `ja-mecab`).
- Metrics: for ASR, `audio/covost2.asr/pass@1/wer::<language>` is the per-language error rate, a character error rate
  for `ja` and `zh-CN`. The overall `wer` pools the edit counts of every selected language, counting characters for
  `ja` and `zh-CN` and words for the others, so over a mixed set read the per-language values. For translation,
  `audio/covost2.st/pass@1/bleu` is the corpus BLEU and `bleu::<src>-><tgt>` the per-pair value; the accuracy-style
  fields (`correct`, `success_rate`) published next to it are not meaningful for translation. Every language and pair
  also has its own `audio/covost2.asr-<language>/pass@1/*` and `audio/covost2.st-<src>-><tgt>/pass@1/*` group.
- Run: see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark).
