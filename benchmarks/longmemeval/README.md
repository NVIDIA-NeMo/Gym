# LongMemEval (gym-native)

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a **long-term-memory
QA** benchmark: each task hands the model a multi-session chat haystack and asks
a question that can only be answered by recalling, reasoning over, updating or
abstaining from information buried in that history.

This entry runs LongMemEval through the **gym-native** eval path against the
whole **oracle** split and reports the upstream headline `accuracy` — computed
by the `longmemeval` resources server's `compute_metrics` and surfaced via
`get_key_metrics` (alongside `abstention/accuracy` and `n_excluded`).

`accuracy` excludes rows the judge could not grade (`judge_call_failed`,
`unknown_question_type`), matching upstream's skip path. A plain per-row-mean
eval reports something closer to `accuracy_strict` (the full denominator) and no
per-question-type breakdown; only the gym-native `compute_metrics` path produces
the headline number and the buckets.

## Relationship to the resources server

Scoring, the per-question-type rubrics, the judge client and all aggregation
live in the `longmemeval` resources server (`resources_servers/longmemeval/`) —
see its [README](../../resources_servers/longmemeval/README.md) for the rubric
table, error taxonomy, metrics and the deviations from upstream. This benchmark
only supplies data and wiring; it chains to
`resources_servers/longmemeval/configs/longmemeval.yaml` and inherits
`longmemeval_simple_agent`.

## Data shape

Unlike some benchmarks here, LongMemEval rows need **no re-shaping**:
`prepare_longmemeval.py` already emits Responses API shape — a single user
message carrying the rendered JSON-format session history plus the question — so
`prompt_config` is `null` and the pre-built `responses_create_params.input` is
used untouched. `prepare.py` only tags each row with the benchmark `agent_ref`
(`longmemeval_benchmark_simple_agent`) so rows align with the agent selected at
eval time.

## Prepare data

```bash
gym eval prepare --benchmark longmemeval
```

Builds `resources_servers/longmemeval/data/oracle.jsonl` if missing (invokes
`prepare_longmemeval.py` with its defaults — `--split oracle --topk-context 50`
— which downloads `xiaowu0162/longmemeval-cleaned` from HuggingFace into
`$XDG_CACHE_HOME/longmemeval` on first run), then writes the tagged
`benchmarks/longmemeval/data/longmemeval_benchmark.jsonl`. Both are gitignored.

To evaluate a different split (`s` / `m`), build it yourself and point the
dataset at it — `prepare.py` is hardwired to the oracle split.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark longmemeval \
    +judge_base_url=https://api.openai.com/v1 \
    +judge_api_key=$OPEN_AI_KEY \
    +judge_model_name=gpt-4o-mini
```

**A judge endpoint is required** — LongMemEval is LLM-as-judge scored and cannot
verify without one. `gpt-4o-mini` is the upstream default.

## Collecting rollouts and scoring

```bash
gym eval run --no-serve \
    --agent longmemeval_benchmark_simple_agent \
    --input benchmarks/longmemeval/data/longmemeval_benchmark.jsonl \
    --output results/longmemeval_rollouts.jsonl \
    --num-repeats 1
```

Check `n_excluded` / `n_judge_call_failed` in the reported metrics before
trusting a score: a rate-limited judge shrinks the `accuracy` denominator. Raise
`judge_max_retries` / `judge_retry_base_delay` (or lower
`judge_endpoint_max_concurrency`) if either is non-zero.

## License

Dataset: MIT — [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
derived from [`github.com/xiaowu0162/LongMemEval`](https://github.com/xiaowu0162/LongMemEval) (MIT).
