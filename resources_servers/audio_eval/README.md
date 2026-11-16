# Audio evaluation resources server

One resources server scores the speech and audio benchmarks under `benchmarks/` that declare `audio_eval` as their
resources server (ASR, multi-speaker ASR, speech translation, language identification, audio question answering,
entity recovery). A benchmark ships a `profiles.json` and a Gym `config.yaml` that inherits this server's config and
sets `profile_contracts` to its profile file. The benchmarks are listed in the "Speech and audio benchmarks" section
of the [root README](../../README.md).

## Scoring model

Many audio metrics (corpus WER and BLEU, session merging, per-subset micro averages) are not averages of per-rollout
scores, so scoring is corpus-level:

- `verify` checks the request, extracts the assistant text, runs the profile's judge call if it has one, and returns
  a **placeholder reward of 0.0**. Do not read the per-row reward as a quality signal.
- `aggregate_metrics` scores all unmasked rows per profile and publishes flat scalar metrics named
  `audio/<profile>/pass@1/<metric>` (and `audio/<profile>-<subset>/...` for each `subset_for_metrics` value).
  These are the benchmark results.
- A response that is neither `completed` nor truncated at `max_output_tokens` is a masked row
  (`mask_sample`, `failure_kind: audio_eval:generation_failed`) and is excluded from the metrics. A failed or unusable
  judge reply masks the row with `failure_kind: judge_failed`.
- Run with one rollout per task (`num_repeats=1`): corpus metrics are defined per source row and
  `compute_metrics` rejects repeats. Every benchmark config declares `num_repeats: 1`, but Gym does not pass the
  repeat count to a resources server at startup, so `--num-repeats N` with N above 1 collects all rollouts and only
  then fails in `aggregate_metrics`; keep it at 1.

## Profile contract

`profiles.json` maps a profile name to its contract; the server reads it once at startup and fails fast if it is
empty or inconsistent.

| Key | Meaning |
|---|---|
| `skills.EVAL_ARGS` | `++eval_type=<evaluator>` and `++eval_config.<key>=<value>` settings (for example `cleanup_generation_loops=true`, which collapses runaway repetition before scoring). |
| `skills.METRICS_TYPE` | Metrics aggregator for the profile. |
| `judge_prompt_template` | Optional `system`/`user` templates, filled from the source row and the generation, for a judged profile. |
| `request`, `prompt_override` | Used by benchmark preparation, not by the server. |

To avoid repeating identical contracts, a `profiles.json` may add a top-level `defaults` object that every profile
inherits (an object-valued key such as `request` or `skills` merges one level deep with the profile's own, any other
value is replaced by the profile's), and `judge_prompt_templates`, named judge templates that a profile selects with
`"judge_prompt_template": "<name>"`. `load_profiles` resolves both, so consumers always see complete contracts.

### Scorers are found by name

`++eval_type=<name>` and `METRICS_TYPE` name modules; there is no registry file, so adding a scorer means adding a module:

| Name | Module | Defines |
|---|---|---|
| evaluator `foo-bar` or `foo_bar` | `_scoring/evaluation/evaluator/foo_bar.py` | a function `eval_foo_bar`, or exactly one `BaseEvaluator` subclass |
| metrics `foo-bar` or `foo_bar` | `_scoring/evaluation/metrics/foo_bar_metrics.py` | exactly one `BaseMetrics` subclass |
| either, `nemo_skills.dataset.<dataset>.scorer::<Class>` | `_scoring/dataset/<dataset>/scorer.py` | `<Class>` |

A metrics name may carry a variant suffix: `foo_bar_open_ended` uses `foo_bar_metrics.py` when no module matches the full
name. The generic evaluator is `audio` (WER, CER, mixed error rate, BLEU, punctuation, multi-speaker cpWER, selected by
the row's `task_type`); `audio` metrics aggregate its output and judge ratings. A benchmark-specific scorer lives in
its own modules, and a module that needs a text normalization mode or a task type of the generic evaluator adds it with
`register_normalization_mode` or `register_task_type` (in `audio.py`) when it is imported.

A scorer whose judge call is not a prompt template defines a module-level `JUDGE` in its evaluator module. The object has
`validate(settings)` (raise `ValueError` for settings it cannot judge), `request(row)` (the chat request for a scored
row) and `fields(completion, row)` (the row fields the reply contributes), as described in `judges.py`; a profile that
names that evaluator is then judged without a `judge_prompt_template`.

## Example data

`data/example.jsonl` holds five synthetic rows of an ASR profile (`example-asr` in `data/example_profiles.json`), each
with a 0.1-second silent WAV as a `data:` URI in `metadata.audio_data`, so no audio file is committed.
`create_examples.py` writes them. To run the example, set the profile file, since the shared config leaves it empty
for a benchmark to fill in:

```bash
gym env start --config resources_servers/audio_eval/configs/audio_eval.yaml --model-type vllm_model \
    +audio_eval.resources_servers.audio_eval.profile_contracts=resources_servers/audio_eval/data/example_profiles.json
```

The reply of the model is not meaningful (the audio is silence); the rows only exercise the request, verify and
corpus-scoring path. `data/example_rollouts.jsonl` was collected with a stub chat model that returns a fixed
transcript per request, so its metrics (`audio/example-asr/pass@1/wer` of 8.57) are not a model result.
`data/example_metrics.json` comes from `gym dataset collate` in `example_validation` mode, which also adds
`task_source` to `example.jsonl`.

## Prepared rows

Each row carries the Gym request fields plus `benchmark_profile`, `sample_id`, the immutable source row
`ng_source_row` and its digest `ng_source_row_sha256` (`resources_servers.audio_eval.profiles.source_hash`). The
server rejects a request whose source row no longer matches its digest and scores the source row, not the request
copy. `resources_servers.audio_eval.profiles.load_profiles` is the shared profile loader.

## Preparing and running an audio benchmark

Every benchmark below `benchmarks/` is prepared and run the same way (the benchmark README gives its sizes, narrowing
arguments and caveats):

```bash
gym eval prepare --benchmark <name>
gym eval run --benchmark <name> --model-type vllm_model --split benchmark --num-repeats 1 \
    --output results/<name>_rollouts.jsonl
```

- The model server must accept audio (`vllm_model` or a compatible audio-capable endpoint).
- Audio is written under `benchmarks/<name>/data/` and rows reference it by absolute local path. If the model server
  sees that directory elsewhere (a container mount), prepare with the path it will see, for example
  `+prepare_script_args.audio_root=/data/<name>`.
- `+prepare_script_args.max_samples_per_profile=20` keeps only the first rows of each profile (a smoke run). A few
  benchmarks cap per subset, locale or pair instead (`max_samples_per_subset`, `max_samples_per_locale`,
  `max_samples_per_pair`); their READMEs say so.
- The default of every narrowing argument is the complete benchmark. Narrowed, capped or `audio_root`-remapped data is
  never reported as already prepared, so a later default `prepare` rebuilds the full set.
- Per-row rewards are placeholders (`0.0`); read the corpus metrics, published as `audio/<profile>/pass@1/<metric>` in
  `rollouts_aggregate_metrics.json`, and keep `num_repeats` at 1.
- Unless a benchmark README says otherwise, requests use the `greedy` preset (temperature 0, seed 0, thinking
  disabled, at most 256 output tokens) with the benchmark's fixed prompt, and `cleanup_generation_loops=true` collapses
  repeated character runs and repeated blocks in the output before scoring.

## Judged profiles

Judged profiles (a `judge_prompt_template`, or an evaluator with `JUDGE` hooks) call a chat-completions judge through
Gym's judge mechanism. Point the server at a model server defined in `env.yaml`:

```bash
gym env start ... \
    +audio_eval.resources_servers.audio_eval.judge_model_server.name=judge_model
```

The server fails at startup if a judged profile is configured without `judge_model_server`, or if its endpoint is still
the `http://unset.local/v1` placeholder of the judge model config. The judge's model is the one the model server is
configured with; a benchmark's README names the judge its reference results use, and scores from other judges are not
comparable. `judge_max_concurrency` (default 40) bounds in-flight judge calls.

## Dependencies and data

`pyproject.toml` pins `kaldialign`, `whisper-normalizer`, `jiwer` and `jieba`, which define the scoring numerics, and lists
the libraries the scorers of the benchmarks import, so that one environment serves every benchmark.

## Provenance and licensing

The scorers adapt code from other repositories, recorded in `ATTRIBUTIONS.md` and in each file header:

- `_scoring/evaluation/`: evaluators and metrics adapted from NVIDIA-NeMo/Skills (Apache-2.0) and the NeMo cpWER
  implementation (Apache-2.0).
- `_scoring/chime8_normalizer.py` and `_scoring/chime8_normalizer_data/`: the CHiME-8 text normalizer adapted from
  chime-utils (MIT, derived from the OpenAI Whisper English normalizer, MIT) at a pinned commit; see its file header and
  `CHIME_UTILS_LICENSE`.

`_scoring/` is excluded from the repository coverage gate where it vendors third-party source; the rest of the server is
covered by `tests/`.
