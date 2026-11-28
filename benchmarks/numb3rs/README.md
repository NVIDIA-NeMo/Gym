# Numb3rs

Text-normalization (TN) and inverse-text-normalization (ITN) ASR on
[`nvidia/Numb3rs`](https://huggingface.co/datasets/nvidia/Numb3rs): 10,131 synthetic clips (about 4.9 hours, TTS audio
checked by human annotators) in 12 number-shaped categories (`ADDRESS`, `CARDINAL`, `DATE`, `DECIMAL`, `DIGIT`,
`FRACTION`, `MEASURE`, `MONEY`, `ORDINAL`, `PLAIN`, `TELEPHONE`, `TIME`), each with a written reference (`text_tn`,
`"$100"`) and a spoken reference (`text_itn`, `"one hundred dollars"`).

- **Source:** the dataset `manifest.jsonl` and WAV files at revision `192908075e1bd293914cc7b508f4a183ba6ef2b8`.
  `prepare()` downloads data files only and fails if the manifest does not hold 10,131 rows or a clip path is
  unexpected.
- **License:** CC-BY-NC-SA-4.0, as stated in the dataset card. The dataset links no separate source repository. The card
  says the text pairs derive from the Google Text Normalization dataset; its terms were not checked here. The
  non-commercial and share-alike terms apply to anything that redistributes the data.

## Profiles

One JSONL holds three profiles over the same clips, told apart by `benchmark_profile`:

| Profile | Prompt | Scored against |
|---|---|---|
| `numb3rs` | neutral transcription | `text_itn` |
| `numb3rs.tn` | written form, digits | `text_tn` |
| `numb3rs.itn` | spoken form, numbers spelled out | `text_itn` |

Each row carries its prompt text in `prompt`; [`prompts/default.yaml`](prompts/default.yaml) renders it, with the
system message `You are a helpful assistant. /no_think`, at rollout time. [`prompts/structured.yaml`](prompts/structured.yaml)
is an alternative structured instruction for the `numb3rs` profile (`--prompt-config`). The rows set no sampling
controls: pass them to `gym eval run`, for example `--temperature 0 --top-p 1 --max-output-tokens 256`, with
`chat_template_kwargs.enable_thinking: false` on the policy model.

## Scoring

The [`asr_with_pc`](../../resources_servers/asr_with_pc/) server scores each row in `ASR_LEADERBOARD` mode with
`normalization: numb3rs` (the `category` is required; a row without one is rejected with HTTP 422). That normalizer is
conservative: it ignores case, prose punctuation, `oh`/`o` zeros and conventional initialisms, but keeps number words and
numeric symbols (`$`, `%`, `/`, `:`, signs). Whisper's normalizer would turn spoken numbers into digits on both sides and hide
the errors this benchmark measures. The server also collapses runaway repetition before scoring
(`cleanup_generation_loops`).

Corpus-level metrics (percent):

| Key | Meaning |
|---|---|
| `wer::profile::<profile>@k=1` | WER of one profile, against that profile's reference |
| `wer::subset::<profile>_<CATEGORY>@k=1` | WER of one category within one profile |
| `wer`, `wer_tn`, `wer_itn` | Headline WER and the WER against each form, **pooled over all profiles run**: read the per-profile keys |

## Run

```bash
gym eval prepare --benchmark numb3rs
gym eval run --benchmark numb3rs --model-type vllm_model --split benchmark --num-repeats 1 \
    --output results/numb3rs_rollouts.jsonl
```

`+prepare_script_args.profiles='[numb3rs]'` writes one profile; `+prepare_script_args.audio_root=/data/numb3rs` is for a
model server that sees `data/` elsewhere. `+prepare_script_args.max_samples_per_profile=<n>` keeps the first `n` clips in
category order, so a small cap covers only the first categories (`ADDRESS` has 885 clips); it is for smoke runs.

No score has been compared with a published number for this benchmark.
