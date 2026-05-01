# AudioBench — Alpaca Audio (judge)

Speech-instruction-following on AudioBench's
[`alpaca_audio_test`](https://huggingface.co/datasets/AudioLLMs/alpaca_audio_test)
split (100 prompts). The model hears a spoken instruction and has to
respond. Open-ended responses are scored by an LLM judge on a 0–5 rating
scale, paired with the
[`audiobench_judge`](../../resources_servers/audiobench_judge/) resource
server.

This is the first AudioBench dataset migrated from
[NeMo Skills](https://github.com/NVIDIA/NeMo-Skills/tree/main/nemo_skills/dataset/audiobench)
to NeMo Gym. The remaining 33 judge-scored datasets and 32 nonjudge
(WER/BLEU/CER) datasets follow the same pattern (judge prompt + audio
sidechannel) and are deferred to follow-on PRs.

## Data layout on disk

```
benchmarks/audiobench_alpaca_audio/
├── data/
│   ├── audiobench_alpaca_audio_test.jsonl   # JSONL with audio_path references
│   └── audio/
│       └── alpaca_audio_test_000000.wav     # one WAV per row
```

Both the JSONL and the audio dir are gitignored. `prepare.py` recreates
them from `AudioLLMs/alpaca_audio_test`.

## Audio handling

Rows use the **file-path** audio sidechannel
(`responses_create_params.metadata.audio_path`) introduced in
[#1170](https://github.com/NVIDIA-NeMo/Gym/pull/1170). `prepare.py` writes
**absolute** paths into the JSONL — since the JSONL is regenerated from
HuggingFace on every machine the path doesn't need to be portable, and
absolute paths sidestep the cwd-mismatch between `ng_run`'s launch dir
and each server's per-venv cwd.

Why path-mode rather than inline base64 (`audio_data`)? The 100 sample
JSONL is ~50× smaller (~600 KB instead of ~30 MB), and per-seed
materialized rollouts shrink proportionally — keeps multi-seed audio
benchmarks tractable.

## Prompt

`prompts/default.yaml` materializes the system+user messages into
`responses_create_params.input` at rollout time. Skills uses
`"You are a helpful assistant. /no_think"` as the system message and the
row's `instruction` field ("Please follow the instruction in the
speech.") as the user message — kept byte-equivalent here.

## Prepare benchmark data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/audiobench_alpaca_audio/config.yaml]"
```

Downloads the 100 alpaca_audio_test samples from HuggingFace, writes one
WAV per sample to `data/audio/`, and emits
`audiobench_alpaca_audio_test.jsonl`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/audiobench_alpaca_audio/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=audiobench_alpaca_audio_judge_simple_agent \
    +output_jsonl_fpath=results/audiobench_alpaca_audio_rollouts.jsonl \
    +num_repeats=1
```

## Verification

Per-rollout: an LLM judge is asked to rate the model's answer 0–5 against
the reference. `is_correct = rating >= 3`. Aggregated:

* `accuracy` (pass@k of `is_correct`) — headline percent.
* `judge_score` — `avg(rating) * 20`, AudioBench's headline 0–100 number.

Same scoring as NeMo Skills' `AudioMetrics._extract_judge_result` and
upstream AudioBench. See
[`resources_servers/audiobench_judge/`](../../resources_servers/audiobench_judge/)
for details.
