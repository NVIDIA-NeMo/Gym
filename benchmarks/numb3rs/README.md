# Numb3rs

ASR benchmark for text normalization (TN) and inverse text normalization
(ITN). Each row from
[`nvidia/Numb3rs`](https://huggingface.co/datasets/nvidia/Numb3rs)
carries paired written and spoken references — `text_tn` (e.g.
`"$100"`) and `text_itn` (e.g. `"one hundred dollars"`) — across 12
number-shaped categories: `ADDRESS`, `CARDINAL`, `DATE`, `DECIMAL`,
`DIGIT`, `FRACTION`, `MEASURE`, `MONEY`, `ORDINAL`, `PLAIN`,
`TELEPHONE`, `TIME`. Pairs with the
[`asr_with_pc`](../../resources_servers/asr_with_pc/) resource server
in `task_type=ASR_LEADERBOARD` mode, which scores standard WER against
the primary `expected_answer` plus per-reference `wer_tn` / `wer_itn`
against the dual references.

## Variant: neutral only

The upstream benchmark exposes three prompt variants (`neutral`, `tn`,
`itn`) but the canonical Gym dataset is the **neutral** variant only —
the same default Skills evaluates against (`EVAL_SPLIT = "test_neutral"`).
For the `tn` / `itn` prompt variants, re-run `prepare.py` with a
different prompt template and a different `expected_answer` policy:

* `tn`: prompt asks for written form; `expected_answer = text_tn`.
* `itn`: prompt asks for spoken form; `expected_answer = text_itn`
  (same as neutral).

Those alternate variants are reproducible from the same HF source and
are not committed here to keep the JSONL lean.

## Audio handling

`prepare.py` writes the WAV tree under
`benchmarks/numb3rs/data/Numb3rs/<CATEGORY>/<filename>.wav` and the
JSONL stores
`responses_create_params.metadata.audio_path =
"<audio_prefix>/Numb3rs/<CATEGORY>/<filename>.wav"`. The
`vllm_model` audio sidechannel reads `audio_path`, base64-encodes the
file at rollout time, and splices an `audio_url` content block into the
user message before forwarding to vLLM Chat Completions.

The default `--audio-prefix` (`/data/numb3rs`) is the path Skills'
prepare also writes, so the same shared mount works for both
pipelines.

## Prompt

System + user templates live in [`prompts/default.yaml`](prompts/default.yaml)
and match Skills' `PROMPT_NEUTRAL` character-for-character.
`prompt_config` materializes them into `responses_create_params.input`
at rollout time, so `prepare.py` doesn't bake the messages into each
row.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/numb3rs/config.yaml]"
```

Downloads `nvidia/Numb3rs` (split=`test`), iterates the 12 categories,
writes WAVs under `benchmarks/numb3rs/data/Numb3rs/<CAT>/` and a single
combined `benchmarks/numb3rs/data/numb3rs_benchmark.jsonl`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/numb3rs/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=numb3rs_asr_with_pc_simple_agent \
    +output_jsonl_fpath=results/numb3rs_rollouts.jsonl \
    +num_repeats=4
```

## Verification

`task_type=ASR_LEADERBOARD` runs standard Whisper-normalized WER against
`expected_answer` (the spoken form), then iterates
`reference_fields = ["text_tn", "text_itn"]` and emits per-reference
`wer_tn` / `wer_itn` plus `is_correct_<suffix>` flags. Corpus-level
`wer`, `wer_tn` and `wer_itn` are aggregated across the eval set by
`compute_metrics()` on the `asr_with_pc` server; per-category
breakdowns ride on `subset_for_metrics = "numb3rs_<CATEGORY>"`.
