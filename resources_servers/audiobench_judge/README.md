# AudioBench Judge

LLM-as-a-judge scoring for the open-ended half of [AudioBench](https://github.com/AudioLLMs/AudioBench)
(speech instruction following, audio QA, emotion / gender / accent recognition
phrased as open-ended QA, music understanding, etc.).

Pairs an audio benchmark row's `question` + `expected_answer` with the model's
generated answer and asks an LLM judge to score the model's answer on a 0–5
scale of alignment with the reference. Rating 0–5 is mapped to a
`judge_score = avg(rating) * 20` headline (0–100) for parity with AudioBench's
reporting convention; per-rollout `is_correct = rating >= 3` feeds pass@k.

## What it scores

For each rollout the server posts a single judge call with the AudioBench
rating prompt:

```
Score0: 'cannot decide' / completely misaligned
Score1: minimal alignment, mostly wrong
Score2: recognizes the topic but diverges
Score3: aligns generally, lacks detail
Score4: mostly accurate, could be clearer
Score5: highly accurate, matches the reference
```

Output is parsed for `Rating: X` (0–5). Falls back to `Judgement: Yes/No`
(legacy `audiobench_binary` prompt → 5.0 / 0.0) and finally a plain
`yes`/`no` anywhere in the text — same precedence as NeMo Skills'
[`AudioMetrics._extract_judge_result`](https://github.com/NVIDIA/NeMo-Skills/blob/main/nemo_skills/evaluation/metrics/audio_metrics.py).

## Audio plumbing

The server itself doesn't touch audio — the model server (`vllm_model`)
handles the audio sidechannel via `responses_create_params.metadata`.
Three mutually exclusive metadata keys are supported:

* `audio_data`  — pre-built `data:audio/...;base64,...` URI inlined into the JSONL
* `audio_path`  — single file path; resolved against `audio_root` and base64-encoded at request time
* `audio_paths` — list of file paths; encoded and spliced in order

For the audiobench benchmark JSONLs we use `audio_path` to keep the on-disk
JSONL small (~50 KB / WAV vs ~50× expansion when inlined as base64).

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/audiobench_judge/configs/audiobench_judge.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collecting rollouts (5-example smoke test)

```bash
ng_collect_rollouts \
    +agent_name=audiobench_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/audiobench_judge/data/example.jsonl \
    +output_jsonl_fpath=results/audiobench_judge_rollouts.jsonl \
    +num_repeats=1
```

## Regenerating example data

```bash
python resources_servers/audiobench_judge/generate_example_data.py
```

The committed `data/example.jsonl` uses 1-second silence WAVs as audio
placeholders — small enough to commit, sufficient for unit tests and schema
smoke tests. The actual benchmark JSONLs (with real audio) are built by each
benchmark's own `prepare.py`.
