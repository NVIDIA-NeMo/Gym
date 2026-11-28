# MUCS 2021

Code-switched recognition on the blind test of the Interspeech 2021 Multilingual and Code-Switching ASR Challenge,
Subtask 2: Hindi-English (4,034 segments, 35 recordings) and Bengali-English (3,130 segments, 32 recordings). Each
language is its own profile and is scored on its own; the release defines no cross-language aggregate. By default both
languages are prepared.

- **Source:** [OpenSLR SLR104](https://www.openslr.org/104/) `subtask2_blindtest_wReadme.tar.gz` (1,008,263,116 bytes),
  SHA-256 `b134ba007e6a310a2ca2a89619257ea6e7de744e135043c8c517cc05a404889a`.
- **License:** CC BY-SA 4.0, as stated on the OpenSLR page. The scoring maps (`inference/` in the same archive) carry the
  same license.
- **Reference:** the release's `text` file of each language. Eight Hindi-English and four Bengali-English references are
  empty; the official scorer skips them, and so does this one (they are counted in `skipped_empty_references`).

## Preparation

`prepare()` downloads the archive once (about 1 GB, kept in `data/source`), checks its SHA-256, and checks the Kaldi
metadata of each language (`text`, `segments`, `wav.scp`, `utt2spk`) against pinned digests and the released row,
recording and empty-reference counts. It then cuts each segment out of its recording, without resampling, into a 16 kHz
mono 16-bit WAV under `data/<language>/audio/` (about 1.4 GB for both languages). Segment boundaries must fall on whole
samples, or preparation fails.

Narrow what is prepared with `languages` (`hindi-english`, `bengali-english`; a list or a comma-separated string) and
`max_samples_per_profile` (the first N segments of each language, for smoke runs):

```bash
gym eval prepare --benchmark mucs2021 +prepare_script_args.languages=hindi-english
```

Data prepared with a narrowing option is not reused by a default run.

## Scoring maps are streamed on first use

The scorer needs three maps from the release: the punctuation map and one word map per language. They are not
redistributed in this repository. `prepare()` installs them from the archive it has already downloaded; otherwise the
scorer streams the pinned OpenSLR archive on its first use, reads only the three `inference/` members, checks each
against its pinned SHA-256 and caches it under Gym's cache directory (`audio_eval/mucs2021`). Later runs read the cache.
The three maps are the last members of the archive, so the stream reads all of it: the first scoring run on a machine
that has not run `prepare()` needs network access to `openslr.org` and downloads about 1 GB (a long time on a slow
link; the OpenSLR server delivered roughly 15 MB/min in our test). A corrupted cache file is fetched again.

## Protocol

The scorer follows the release's `compute_wer.py`:

1. The reference and the hypothesis are lowercased and split on whitespace; no other normalization is applied.
2. **WER (normal):** the punctuation map is applied to the hypothesis only, then the word error rate is computed.
3. **tWER (transliterated):** the language's word map (`hin2engwords.txt` or `ben2engwords.txt`) is applied to the
   already punctuation-mapped hypothesis, then the word error rate is computed.
4. Rows with an empty reference are skipped, so their inserted words count toward neither score.

Both profiles use the `greedy` request preset (temperature 0, top-p 1, seed 0, thinking off, at most 256 output tokens),
an instruction to transcribe verbatim, write the native script and the English words in the Latin alphabet, and use no
sentence punctuation, and collapse generation loops before scoring (`cleanup_generation_loops`).

## Metrics

Published per language as `audio/mucs2021.hindi-english/pass@1/*` and `audio/mucs2021.bengali-english/pass@1/*`:

| Key | Meaning |
| --- | --- |
| `wer`, `twer` | Normal and transliterated WER of the language, in percent: the summed errors over the summed reference words of the non-empty references (the release's utterance-weighted average) |
| `wer_errors`, `twer_errors`, `ref_words` | The pooled counts behind them |
| `scored_entries`, `skipped_empty_references` | Rows scored and rows skipped for an empty reference |
| `no_answer` | Percent of rows with an empty generation. This is the full share, not the halved share that the other `audio_eval` metrics publish |

## Run

See the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, smoke runs).

Scores have not been compared with published MUCS 2021 results; only the scorer's arithmetic is tested, on hand-counted
rows.
