# ASR Leaderboard

English recognition on the test sets of the Hugging Face [Open ASR Leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard),
scored with the [`asr_with_pc`](../../resources_servers/asr_with_pc/) server's `OpenASR` task type (the leaderboard's
text normalizer and compound-aware corpus WER). All eight subsets are prepared by default.

| Subset | Source | License |
| --- | --- | --- |
| `librispeech_clean`, `librispeech_other` | `hf-audio/open-asr-leaderboard` config `librispeech` | CC-BY-4.0 |
| `voxpopuli_aa` | config `voxpopuli_cleaned_aa` | CC0-1.0 |
| `gigaspeech` | config `gigaspeech_cleaned` | Apache-2.0 (see the note below) |
| `spgispeech` | config `spgispeech` | Kensho SPGISpeech user agreement |
| `ami` | config `ami_cleaned` | CC-BY-4.0 |
| `urgent2024` | config `urgent2024` (1,000 noisy rows, resampled to 16 kHz) | none stated |
| `earnings22_aa` | [`ArtificialAnalysis/Earnings22-Cleaned-AA-chunked`](https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA-chunked) with references from `ArtificialAnalysis/Earnings22-Cleaned-AA` | Apache-2.0 (cleaned transcripts), CC-BY-SA-4.0 (Earnings-22 audio) |

- **Licenses.** Each subset's license is the one stated by its source: the Open ASR Leaderboard dataset card for
  LibriSpeech, VoxPopuli, AMI and SPGISpeech; the Earnings-22 repository (`revdotcom/speech-datasets`) and the
  Artificial Analysis cards for `earnings22_aa`. For GigaSpeech the source repository
  ([`SpeechColab/GigaSpeech`](https://github.com/SpeechColab/GigaSpeech)) carries the Apache-2.0 license, as does the
  leaderboard card; the gated `speechcolab/gigaspeech` Hugging Face mirror separately requires accepting its own terms
  of access, so check them before use. The URGENT 2024 rows come from the leaderboard's `urgent2024` config, which
  states no data license (the challenge repository's Apache-2.0 license covers its code). The manifest expresses
  these as an SPDX expression with `LicenseRef-` identifiers for the two terms that have no SPDX identifier.
- **Pinned revisions.** `hf-audio/open-asr-leaderboard` `de79e57adf9983993bcc72586d7313460825f5eb` (adds
  `urgent2024`), `Earnings22-Cleaned-AA-chunked` `27f7d0056d593231974a752ce03c1e965da69b3f`, `Earnings22-Cleaned-AA`
  `5999936ee3d8d3cde160e3c65a8cc6ff5b7c5276`. GigaSpeech and SPGISpeech are gated: accept their terms on Hugging Face
  and log in before preparing.
- **Earnings22** is transcribed as 341 VAD chunks but scored on its six parent calls: each chunk row carries
  `parent_id` and `chunk_index`, and the server concatenates a call's raw hypotheses in chunk order before normalizing
  and scoring once. Each chunk row carries the whole call's reference, so the per-row `reward` of a chunk row is not
  meaningful; use the aggregate metrics.
- **Changes from the previous version** of this benchmark (eight ESB test subsets, Whisper-normalized `ASR` task
  type): the cleaned `ami`, `gigaspeech` and `voxpopuli_aa` configs, chunked Artificial Analysis Earnings22 with
  parent-level scoring, `urgent2024` in place of TED-LIUM, the `OpenASR` task type, the prompt stored in each prepared
  row (no `prompts/` file), the request settings in `config.yaml` (reasoning off, temperature 0, top-p 1, seed 0,
  16,384 output tokens), and the server's generation-loop guardrail (`cleanup_generation_loops`).
- **Prompt.** A verbatim-transcription prompt with punctuation and capitalization; SPGISpeech uses the variant that
  omits fillers, because its references omit them.
- **Reference filter.** As in the leaderboard, rows whose reference is `ignore time segment in scoring` or normalizes
  to empty are dropped by `prepare()`.
- **Metrics.** The headline `wer` is the corpus-level WER in percent; per-subset values are namespaced as
  `wer::subset::<subset>`. The per-row `reward` is `1.0` when the row's WER is below 0.5 and `0.0` otherwise; it is
  not the leaderboard score. See the server README for the full metric list.
- **Reading data.** Rows are read from the pinned parquet shards (and the Earnings22 JSON Lines files) with
  `benchmarks/hf_parquet.py`, one row group at a time, so a capped smoke run fetches only the first row groups and the
  process exits cleanly (`datasets` streaming leaves reader threads that hang the interpreter at exit).
- **Limits.** `prepare()` downloads tens of gigabytes of audio for the full set and writes FLAC files under
  `data/audio`. The test suite checks `prepare()` offline against synthetic sources; it does not download data.

Run (`+prepare_script_args.audio_root=/data/asr_leaderboard` when the model server sees the audio elsewhere):

```bash
gym eval prepare --benchmark asr_leaderboard
# smoke run
gym eval prepare --benchmark asr_leaderboard \
    '+prepare_script_args.datasets=[librispeech_clean,earnings22_aa]' +prepare_script_args.max_samples_per_subset=20
gym eval run --benchmark asr_leaderboard --model-type vllm_model --split benchmark --num-repeats 1 \
    --output results/asr_leaderboard_rollouts.jsonl \
    ++policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=false \
    ++policy_model.responses_api_models.vllm_model.extra_body.seed=0
```

The two trailing overrides are required: `--model-type vllm_model` loads after the benchmark config and resets the
config's `chat_template_kwargs` and `extra_body` to `null`, so without them the request carries neither
`enable_thinking: false` nor `seed: 0` (temperature, top-p and the token limit come from `responses_create_params` and
are unaffected).
