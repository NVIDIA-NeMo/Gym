# AIR-Bench Foundation

Multiple-choice audio understanding on AIR-Bench Foundation. By default all 24,683 questions in 25 task/dataset leaves
are prepared: speech (10,973 questions in 11 leaves, 4.9 GB), sound (5,896 in 6 leaves, 5.2 GB) and music (7,814 in 8
leaves, 36.7 GB). The 11 speech leaves can be prepared on their own.

- Data: [`qyang1021/AIR-Bench-Dataset`](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset)
  `Foundation/Foundation_meta.json` and the audio under `Foundation/<leaf>/`, revision
  `8fa643aefd1d7248fd55fcf3e7ad22c81bbac1fa`. Prompt and scorer follow [`OFA-Sys/AIR-Bench`](https://github.com/OFA-Sys/AIR-Bench)
  (`Inference_Foundation.py`, `score_foundation.py`), revision `983247666e2a25519b5c122810c2bcd91f26fd8d`.
- License: Apache-2.0, as stated in the README of the AIR-Bench repository (which has no license file). The dataset
  card on Hugging Face states CC-BY-NC-4.0 (non-commercial use only); the two statements disagree and the benchmark
  declares the repository's. The clips come from public corpora (speech: Common Voice 13, IEMOCAP, MELD, SLURP,
  VoxCeleb1, LibriSpeech, CoVoST 2, Fake-or-Real; sound: CochlScene, TUT 2017, Audio Grounding, AVQA, Clotho-AQA,
  VocalSound; music: MUSIC-AVQA, FMA, MTG-Jamendo, NSynth) that keep their own terms, listed on the dataset card; some
  are share-alike or non-commercial.
- Preparation checks the metadata hash, row count, per-leaf counts and audio inventories against the released
  metadata. The release has no mixed-audio task (mixed audio belongs to the separate, LLM-judged AIR-Bench Chat). The
  public task names `Speaker_Emotion_Recontion`, `Speech_Entity_Reconition` and `Music_Instruments_Classfication` are
  misspelled in the release; the scorer matches them verbatim, so they are kept. The metadata names the
  `Audio_Grounding` clips `.wav` but the release stores `.flac`; like the official inference script, preparation uses
  the `.flac` files.
- The default downloads about 46.7 GB of audio (wav, flac, mp3, written unchanged to `data/Foundation/<leaf>/`); the
  four MTG-Jamendo music leaves alone are about 36 GB (roughly 9 MB per clip), so check that the model server accepts
  clips of that length. `modalities` (`speech`, `sound`, `music`) and `tasks` (task names such as `Speech_Grounding`)
  narrow the set and intersect; for example the 10,973 speech questions:

  ```bash
  gym eval prepare --benchmark air_bench_foundation +prepare_script_args.modalities=speech
  ```

- Request: the official instruction (question and lettered options) plus a system prompt asking for a single
  uppercase option letter; reasoning preset (temperature 0.6, top-p 0.95, seed 0, thinking enabled, 20,480 output
  tokens). The scorer reimplements the published first-character parser from the published protocol (no upstream
  code is copied) and applies it unchanged to all 25 leaves;
  outputs it cannot parse are excluded from the leaf's denominator and counted in `fail_num`. The benchmark defines no
  overall accuracy, so none is reported: compute per-modality averages from the leaf accuracies.
- Metrics: `audio/air-bench-foundation-<task>_<dataset>/pass@1/accuracy` for each prepared leaf.
- Row caps: `max_samples_per_profile=N` keeps the first N rows in metadata order (all speech), `max_samples_per_subset=N`
  the first N rows of each of the 25 leaves (a task-stratified sample); both fetch the audio file by file.
- Run: see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark);
  a smoke run with `max_samples_per_profile=20` covers only the first speech leaves, so use
  `max_samples_per_subset` to touch every leaf.
