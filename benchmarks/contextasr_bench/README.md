# ContextASR-Bench

Contextual speech recognition on entity-rich, TTS-synthesized speech, with no context, the domain, or the domain plus
a list of entities in the prompt. Scored for WER and the named-entity metrics NE-WER and NE-FNR.

- **Source:** [`MrSupW/ContextASR-Bench`](https://huggingface.co/datasets/MrSupW/ContextASR-Bench), revision
  `7906fd98d1d093bf244df80e726e07fc8afed6a6`; preparation checks the SHA-256 and row count of the four metadata
  files. Paper: Wang et al., [arXiv:2507.05727](https://arxiv.org/abs/2507.05727).
- **License:** MIT. The code repository
  ([`MrSupW/ContextASR-Bench`](https://github.com/MrSupW/ContextASR-Bench)) carries an MIT `LICENSE` and the dataset
  card metadata says `mit` as well. The audio is synthesized speech; the dataset is downloaded at preparation time and
  not stored in this repository.
- **Default preparation:** all four test sets, each clip in three conditions: 41,329 clips, 123,987 rows, about 100 GB
  of 16 kHz WAV under `data/audio/<set>/<language>`, streamed from the dataset's tar archives (it takes hours):

  | Test set | Clips | Hours | Archives | Profile prefix |
  | --- | --- | --- | --- | --- |
  | `speech` / `English` | 15,326 | 188.0 | 8, 22 GB | `contextasr-bench` |
  | `speech` / `Mandarin` | 15,498 | 197.6 | 8, 23 GB | `contextasr-bench-mandarin` |
  | `dialogue` / `English` | 5,273 | 221.9 | 6, 26 GB | `contextasr-bench-dialogue` |
  | `dialogue` / `Mandarin` | 5,232 | 230.4 | 6, 27 GB | `contextasr-bench-dialogue-mandarin` |

- **Conditions** (profile suffixes): `<prefix>.contextless`, `<prefix>.coarse` (the domain label of a speech clip, the
  movie title of a dialogue) and `<prefix>.fine` (the label and the entity list). The prepare arguments `subsets`
  (`speech`, `dialogue`), `languages` (`English`, `Mandarin`) and `modes` (`contextless`, `coarse`, `fine`) narrow the
  set. Metrics are per profile, so narrowing does not change them. The English speech set alone:

  ```bash
  gym eval prepare --benchmark contextasr_bench '+prepare_script_args.subsets=[speech]' '+prepare_script_args.languages=[English]'
  ```

- **Prompts and limits.** All prompts are written for this benchmark; they have not been compared with the prompts of
  the original evaluation code, and the scores have not been compared with the paper's. The task sentence is
  "Transcribe the `<language>` audio into text, ensuring all punctuation marks are included.", preceded in the coarse
  condition by "This audio belongs to the `<domain>` field." (speech) or "This audio is a dialogue about the movie
  `<title>`." (dialogue), and in the fine condition also followed by the entity list. The English speech profiles use the
  `greedy` request preset (256 output tokens); Mandarin speech allows 1,024 tokens and the dialogues 4,096, as the clips
  are longer. Those limits are chosen sizes, not validated ones. No system prompt is sent.
- **Scoring:** the benchmark's own normalization (contractions expanded, punctuation removed, hyphens and apostrophes
  split, single letters merged, Chinese, Japanese and Korean text split into characters) and its entity matching
  (exact for NE-FNR, edit-distance fuzzy for NE-WER), after the generation-loop cleanup. Only entities that occur in the
  reference count.
- **Metrics** (`audio/<profile>/pass@1/*`, and the same per domain label of the speech sets under
  `audio/<profile>-<domain>/pass@1/*`): `wer`, `ne_wer` and `ne_fnr` in percent, pooled over the corpus (total errors
  over total reference words or entities); `success_rate` (= `correct`) is the share of clips with a per-clip WER below
  0.5; `no_answer` is half of the share of empty generations, as in the other audio benchmarks. WER of Mandarin counts
  characters and of English counts words, so the two sets are not comparable with each other. Exactly one generation
  per clip is supported.
- **Run:** see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark).
  A smoke run (`+prepare_script_args.max_samples_per_profile=<n>`, a positive integer) keeps the first clips of every
  selected set, but the archives are name-sorted, not in metadata order, so each archive that holds a wanted clip is
  read up to that clip (up to 3 to 5 GB) and a smoke run takes minutes per set.
