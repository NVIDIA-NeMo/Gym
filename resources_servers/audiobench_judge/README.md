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

## Judge model

The judge is a separate model server, `judge_model`, so the audio model under
test isn't asked to score itself. It defaults to `openai/gpt-oss-20b` on
`https://integrate.api.nvidia.com/v1`, called through `/v1/chat/completions`
(`use_chat_completions_for_judge: true`) at temperature 0 with 512 output
tokens. Set `judge_api_key` in `env.yaml`, and `judge_base_url` /
`judge_model_name` to use a different judge. `judge_max_concurrency` (default
32) bounds concurrent judge calls.

A failed judge call (transport error, timeout, or non-2xx reply) raises
`JudgeError`, so the row is masked and routed to `<output>_failures.jsonl`
instead of being scored as a wrong answer. A judge reply with no parseable
rating is a legitimate rating of 0. An empty model answer is rated 0 without
calling the judge.

## Metrics

- `judge_score`: mean rating × 20 (0–100), the AudioBench headline.
- `accuracy`: share of rollouts rated ≥ 3, reported as pass@k.
- `no_answer`: share of rollouts with an empty answer.
- Rows that carry `subset_for_metrics` also get per-subset metrics
  (`<subset>/pass@1[avg-of-k]/judge_score`, ...).

## Audio plumbing

Audio is carried separately from text content in `responses_create_params.metadata`.
The server itself doesn't read audio; the `vllm_model` adapter accepts one of
three mutually exclusive fields:

- `audio_data`: a `data:audio/...;base64,...` URI embedded in the row.
- `audio_path`: one audio file path resolved by the model server.
- `audio_paths`: multiple audio file paths resolved by the model server.

The committed example data uses `audio_data`; the
[`audiobench/judge`](../../benchmarks/audiobench/) benchmark uses `audio_path`
to keep its JSONL small.

This server only accepts vLLM-based model adapters (`allowed_model_types`).
`--model-type vllm_model` selects the Gym adapter; set `--model-url` to an
audio-capable vLLM server.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --model-url http://<vllm-host>:<port>/v1 \
    --model <audio-capable-model> \
    --resources-server audiobench_judge
```

## Collecting rollouts (5-example smoke test)

```bash
gym eval run --no-serve \
    --agent audiobench_judge_simple_agent \
    --input resources_servers/audiobench_judge/data/example.jsonl \
    --output results/audiobench_judge_rollouts.jsonl \
    --num-repeats 1
```

## Regenerating example data

```bash
python resources_servers/audiobench_judge/generate_example_data.py
```

The committed `data/example.jsonl` uses 1-second silence WAVs as audio
placeholders — small enough to commit, sufficient for unit tests and schema
smoke tests. The benchmark JSONLs (with real audio) are built by
`benchmarks/audiobench/<benchmark>/prepare.py`.

## Licensing information

- Code: Apache-2.0
- Data: the committed example data is synthetic. Benchmark data licenses vary
  per AudioBench sub-dataset; see [`benchmarks/audiobench`](../../benchmarks/audiobench/).
- Dependencies: nemo-gym (Apache-2.0)
