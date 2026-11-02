# RAGTruth (gym-native)

[RAGTruth](https://github.com/ParticleMedia/RAGTruth) is a **hallucination
detection** benchmark: the model is shown a `(reference context, candidate
response)` pair and must emit a `{"hallucination list": [...]}` JSON object
listing any hallucinated spans. Reward is `1.0` when the model's binary "any
hallucination?" verdict matches the gold label, else `0.0`.

This entry runs RAGTruth through the **gym-native** eval path over all three
task slices at once and reports the `ragtruth` resources server's
`compute_metrics`: `mean_reward` (the headline accuracy), corpus-level
`precision` / `recall` / `f1`, the per-slice `task_type/<slice>/*` breakdown,
and `parse_fail_rate`.

## Relationship to the resources server

Prompt templates, JSON parsing, scoring and all aggregation live in the
`ragtruth` resources server (`resources_servers/ragtruth/`) — see its
[README](../../resources_servers/ragtruth/README.md) for the slice table,
metrics and the deviations from upstream. This benchmark only supplies data and
wiring; it chains to `resources_servers/ragtruth/configs/ragtruth.yaml` and
inherits `ragtruth_simple_agent`.

No judge and no model server are needed on the Gym side — `verify()` scores the
model's JSON deterministically.

## Data shape

RAGTruth rows need **no re-shaping**: `prepare_ragtruth.py` already emits
Responses API shape — a single user message with the slice's prompt template
applied (context + candidate response baked in) — so `prompt_config` is `null`
and the pre-built `responses_create_params.input` is used untouched.

`prepare.py` concatenates the three splits (`test_qa.jsonl`,
`test_summary.jsonl`, `test_data2txt.jsonl`) into one whole-dataset file and
tags each row with the benchmark `agent_ref` (`ragtruth_benchmark_simple_agent`)
so rows align with the agent selected at eval time. `task_type` rides on every
row, so concatenating does not lose the per-slice breakdown.

## Prepare data

```bash
gym eval prepare --benchmark ragtruth
```

Builds `resources_servers/ragtruth/data/test_{qa,summary,data2txt}.jsonl` if any
is missing (invokes `prepare_ragtruth.py`, which downloads the upstream
`response.jsonl` / `source_info.jsonl` into `$XDG_CACHE_HOME/byob_ragtruth` on
first run), then writes the tagged
`benchmarks/ragtruth/data/ragtruth_benchmark.jsonl`. All are gitignored.

Set `RAGTRUTH_DATASET_DIR=/path/to/dir` to read a pre-staged copy, or
`RAGTRUTH_NO_FETCH=1` to disable network fetches (air-gapped clusters).

## Running servers

```bash
# Against a vLLM deployment you already serve
gym env start --benchmark ragtruth --model-type vllm_model \
    --model <served-model-name> \
    --model-url http://<vllm-host>:8000/v1 \
    --model-api-key dummy

# Against a hosted API model
gym env start --benchmark ragtruth --model-type openai_model \
    --model gpt-4o-2024-05-13 \
    --model-url https://api.openai.com/v1 \
    --model-api-key "$OPEN_AI_KEY"
```

The policy endpoint is not optional: `--model` / `--model-url` /
`--model-api-key` set `policy_model_name` / `policy_base_url` / `policy_api_key`,
and the model server config fails to resolve without them.

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent ragtruth_benchmark_simple_agent \
    --input benchmarks/ragtruth/data/ragtruth_benchmark.jsonl \
    --output results/ragtruth_rollouts.jsonl \
    --num-repeats 1
```

Check `parse_fail_rate` before trusting a score: a model that never emits valid
JSON scores as "no hallucination" on every row. For a reasoning model, serve it
with `--reasoning-parser <name>` — `verify()` also strips a leading
`<think>…</think>` block and ```json fences as a fallback.

## License

Dataset: MIT — [`ParticleMedia/RAGTruth`](https://github.com/ParticleMedia/RAGTruth).
