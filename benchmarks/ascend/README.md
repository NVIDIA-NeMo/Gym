# ASCEND

Transcription of the 1,315 test utterances of ASCEND, a spontaneous multi-turn Mandarin-English code-switching
conversation corpus recorded in Hong Kong. The score is a mixed error rate: characters for Mandarin, words for English.

- **Data:** [`CAiRE/ASCEND`](https://huggingface.co/datasets/CAiRE/ASCEND) test split
  (`main/test-00000-of-00001.parquet`), revision `737e9800ae31be9932ba8464c80366559bd28424`.
- **License:** CC-BY-SA-4.0, as stated in the dataset card. The data is downloaded by `prepare()` and not copied into
  the repository. The scorer follows the official ASCEND code (MIT License, HKUST); see the header of
  `resources_servers/audio_eval/_scoring/evaluation/evaluator/ascend.py` and `ATTRIBUTIONS.md`.
- **Preparation:** checks the size and SHA-256 of the pinned parquet and its row count (1,315), decodes each clip to a
  mono WAV file at its recorded sampling rate (channels averaged, no resampling) and checks every written file.
  `max_samples_per_profile=<n>` keeps only the first `n` rows (smoke runs).
- **Reference:** the released `transcription`.

## Profile

One profile, `ascend`: the `greedy` request preset (temperature 0, top-p 1, seed 0, thinking off, at most 256 output
tokens), one repeat, and a prompt that asks for a verbatim, code-switch-preserving transcript in simplified Chinese
characters and the Latin alphabet, without sentence punctuation (see `profiles.json`). Generation loops are collapsed
before scoring (`cleanup_generation_loops`).

## Metrics

Published as `audio/ascend/pass@1/*`:

| Key | Meaning |
| --- | --- |
| `mer` | Headline mixed error rate in percent: corpus errors over reference units, where a unit is a Latin-alphabet word or a single Mandarin character (jieba segmentation, as in the official scorer) |
| `cer` | Corpus character error rate in percent (all non-space characters) |
| `mer_upstream_literal`, `cer_upstream_literal` | The same two rates with the raw hypothesis tokenized, as the official code does |
| `mer_errors`, `mer_ref_units`, `cer_errors`, `cer_ref_chars` (and the `_upstream_literal` counterparts) | The pooled counts behind the rates |
| `success_rate` | Percent of utterances with no mixed error |
| `no_answer` | Half of the share of empty generations, as in the other audio benchmarks (see the [audio_eval README](../../resources_servers/audio_eval/README.md)) |

`mer` and `cer` lowercase and remove the official punctuation list from both the reference and the hypothesis. The
official code removes it only from the reference, because its CTC models can only emit that cleaned alphabet; a free-form
audio-language-model transcript can contain case and punctuation, which the `_upstream_literal` rates therefore count as
errors. Read `mer` as the adapted score for such models and `_upstream_literal` as the strict upstream behavior. Scoring
requires `jieba==0.42.1` and fails otherwise.

## Run

```bash
gym eval prepare --benchmark ascend
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, smoke runs with `+prepare_script_args.max_samples_per_profile=<n>`).

Scores have not been compared with published ASCEND numbers: those come from CTC models scored by the official code,
not from an audio language model under this prompt.
