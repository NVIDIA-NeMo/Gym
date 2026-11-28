# MMAR

Multiple-choice audio reasoning on MMAR: questions needing multi-step reasoning over speech, sound, music and their
mixtures. By default all 1,000 questions are prepared; the 294 speech-only questions can be selected alone.

- Data: [`BoJack/MMAR`](https://huggingface.co/datasets/BoJack/MMAR) test split (`MMAR-meta.json`,
  `mmar-audio.tar.gz`), revision `3bd051123480e80d273ae9e8e9f1653f49010ac7` (pinned because MMAR publishes all labels
  and may update its single test set in place). License: CC-BY-NC-4.0 (non-commercial use only), as stated by the dataset card; the official repository
  states no license for the data.
- Scorer: the official set-token match of [`code/evaluation.py`](https://github.com/ddlBoJack/MMAR/blob/3bce090a967db576c8ad433b290a8da582d6d2a0/code/evaluation.py)
  (official repository revision `3bce090a967db576c8ad433b290a8da582d6d2a0`, no license file; adapted from the
  Apache-2.0 MMAU `evaluation.py`, see `ATTRIBUTIONS.md`): the answer's words must all occur in the prediction and no
  word that appears only in a wrong option may occur in it. Prompt and scorer are the same for every modality (the
  official repository prescribes no inference prompt).
- Preparation checks the metadata against its per-`modality` census. The audio archive is streamed (not stored) and
  read until the last wanted clip, so even a speech-only run reads up to the whole archive (about 3 GB); the default
  writes the 1,000 clips unchanged (about 3 GB).
- Modalities (questions): `speech` 294, `music` 206, `mix-sound-speech` 218, `sound` 165, `mix-music-speech` 82,
  `mix-sound-music-speech` 24, `mix-sound-music` 11. `modalities` narrows the set; the speech-only subset
  (`SPEECH_MODALITIES`) is the 294 speech questions, and accuracy over it is not comparable with accuracy over all 1,000:

  ```bash
  gym eval prepare --benchmark mmar +prepare_script_args.modalities=speech
  ```

- Request: the question and lettered options, with a system prompt asking for the full text of exactly one option
  without its letter; reasoning preset (temperature 0.6, top-p 0.95, seed 0, thinking enabled, 20,480 output tokens).
- Metrics: `audio/mmar/pass@1/accuracy` over all prepared rows and `audio/mmar-<category>/pass@1/accuracy` for each
  category (Signal, Perception, Semantic, Cultural Layer). The official report also gives accuracy per modality;
  metrics are grouped by category only, but every rollout row keeps its `modality`, so per-modality accuracy can be
  computed from the rollouts file.
- `no_answer` is the full share of empty generations (unlike the speech-transcription metrics, which halve it).
- Run: see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark).
