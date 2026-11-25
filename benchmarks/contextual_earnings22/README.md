# Contextual Earnings-22

Transcription of 772 Earnings-22 call segments (15-second windows), with and without a list of keywords in the
prompt. Every clip is used in three conditions, all prepared by default (2,316 rows).

- **Source:** [`argmaxinc/contextual-earnings22`](https://huggingface.co/datasets/argmaxinc/contextual-earnings22) `test`
  split (`data/test-00000-of-00001.parquet`, 357 MB, SHA-256 `4d43471525d8df72930629da0b3dfe20d03b5442f04fd7abedf9df9d301661ec`),
  revision `59d7f94771cc197984d8038e494c83abeff87282` (the 100-row `dev` split is not used). Paper: Contextual
  Earnings-22: A Speech Recognition Benchmark with Custom Vocabulary in the Wild, arXiv:2604.07354.
- **License:** CC-BY-SA-4.0. The dataset card gives no license of its own: it states that the license is that of
  Earnings-22, which Rev publishes under CC BY-SA 4.0.
- **Preparation:** checks the parquet hash and row count and writes 16 kHz mono 16-bit WAV clips (about 0.37 GB).
  The whole parquet file is downloaded even for a capped smoke run. The reference is the dataset's normalized text,
  and `expected_answer_cased` carries the cased one.

## Profiles

One profile per prompt context (`greedy` request preset, one repeat, at most 256 output tokens, empty system prompt;
generation loops are collapsed before scoring):

| Profile | Prompt context |
| --- | --- |
| `contextual-earnings22.contextless` | none |
| `contextual-earnings22.local` | the clip's own keyword dictionary |
| `contextual-earnings22.global` | the keyword list of the whole call, which may include words absent from the clip |

Keyword scoring always uses the clip's own dictionary, whatever the prompt lists. `modes` narrows the set:

```bash
gym eval prepare --benchmark contextual_earnings22 '+prepare_script_args.modes=[contextless,local]'
```

## Scoring

Text is normalized with the Open ASR English normalizer (`normalization_mode=open_asr`: hypothesis, cased reference
and keywords), so a spoken number written as words is not charged against a digit reference. A keyword is a true
positive only if its reference occurrence aligns, by minimum edit distance, to identical hypothesis tokens; other
reference occurrences are false negatives and unmatched hypothesis occurrences are false positives. Counts are pooled
over the corpus.

Metrics are published as `audio/<profile>/pass@1/*` and, per call (`file_id`), as `audio/<profile>-<file_id>/pass@1/*`:
`wer` and `keyword_precision`, `keyword_recall`, `keyword_f1` (all in percent), plus `success_rate` (share of clips
with WER below 50%) and `no_answer`, which as in the other audio benchmarks is half the share of empty generations
(see the [audio_eval README](../../resources_servers/audio_eval/README.md)). A keyword metric is absent when its
denominator is zero. Only one generation per clip is supported.

## Run

```bash
gym eval prepare --benchmark contextual_earnings22
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, smoke runs with `+prepare_script_args.max_samples_per_profile=<n>`).

Scores have not been compared with the numbers of the paper; the metric definition follows the paper's, but the
normalizer and prompts are this benchmark's own.
