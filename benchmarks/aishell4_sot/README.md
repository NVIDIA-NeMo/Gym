# AISHELL-4 SOT

Speaker-attributed Mandarin meeting transcription on the 20 test meetings of AISHELL-4 (12.7 hours, 4 to 8 speakers
each). The model transcribes a whole meeting (37 to 40 minutes, so the model server must accept long audio) and tags every
speaker turn with `<spk:N>`. Scoring is the session-level concatenated minimum-permutation character error rate (cpCER).

- **Data:** [`AISHELL/AISHELL-4`](https://huggingface.co/datasets/AISHELL/AISHELL-4) `test/`, revision
  `aada72727856313b19d4a030383c426364931dbf`, a copy of the test set of [OpenSLR SLR111](https://www.openslr.org/111/).
  The scorer follows the text normalization of [`felixfuyihui/AISHELL-4`](https://github.com/felixfuyihui/AISHELL-4)
  revision `bad82b77c3753df1b232c5c6491cd3e2f2e32d24`.
- **License:** CC-BY-SA-4.0, as stated by the source corpus (OpenSLR SLR111). The metadata of the Hugging Face copy says
  `apache-2.0` and carries no further text; this benchmark follows the source repository.
- **Audio:** each recording has 8 channels. Preparation writes channel zero as a 16 kHz mono 16-bit WAV (the single distant
  microphone condition) and deletes the downloaded FLAC (about 1.4 GB of WAV remain; the downloads total about 5 GB).
- **Reference:** every non-empty TextGrid interval, ordered by start time and prefixed with `<spk:N>`, numbering the
  speakers by first appearance. Overlapping speech is therefore serialized by interval start, not interleaved.

## Profile

One profile, `aishell4-sot`: the `greedy` request preset (temperature 0, top-p 1, seed 0, thinking off) with at most
16,384 output tokens, one repeat, and a prompt that asks for a distinct `<spk:N>` tag before every speaker turn. Generation
loops are collapsed before scoring (`cleanup_generation_loops`). A 40-minute meeting can exceed the output limit; a
transcript cut at the limit is scored as it stands, so the missing tail counts as deletions.

## Scoring

Reference and answer are split into one text stream per speaker tag. Each stream keeps only Han characters (U+4E00 to
U+9FA5, the released AISHELL-4 rule) and is split into characters. The reference and answer speakers are then matched
one to one (padding with empty streams when their numbers differ) to give the fewest character errors, so the numbering
the model chose does not matter, but a merged or split speaker costs errors. A reference that has no Han character after
normalization is rejected.

## Metrics

Published as `audio/aishell4-sot/pass@1/*`:

| Key | Meaning |
| --- | --- |
| `cpcer` | Character errors over reference characters, summed over all meetings, in percent |
| `cpcer_macro` | Mean of the per-meeting rates, in percent |
| `cpcer_errors`, `cpcer_substitutions`, `cpcer_insertions`, `cpcer_deletions`, `cpcer_ref_chars` | The pooled counts |
| `success_rate` | Percent of meetings scored without a single error |
| `no_answer` | Half of the share of empty generations, as in the other audio benchmarks (see the [audio_eval README](../../resources_servers/audio_eval/README.md)) |

## Run

```bash
gym eval prepare --benchmark aishell4_sot
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`; smoke run with `+prepare_script_args.max_samples_per_profile=2`, the first meetings in sorted order).

Scores have not been compared with published AISHELL-4 results: the metric is the generic cpCER applied to the model's
own turn tags, on a single distant channel and without diarization or the official segment boundaries.
