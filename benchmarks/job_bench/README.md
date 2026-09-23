# Job-Bench

Runs the Job-Bench `main` split with OpenCode 1.14.18 and an upstream-derived weighted-rubric evaluator.
The `job-bounded-utf8-v1` input-preparation variant is documented in
[the resources server](../../resources_servers/job_bench/README.md); it is not exact published-leaderboard reproduction.
The example explicitly selects a judge on NVIDIA's OpenAI-compatible API; no XAI key is required.

```bash
uv run gym eval prepare --benchmark job_bench

export NVIDIA_API_KEY=...
export POLICY_MODEL=... # Exact model ID available on your endpoint.
export POLICY_BASE_URL=https://inference-api.nvidia.com/v1
export JOB_BENCH_JUDGE_BASE_URL="$POLICY_BASE_URL"
export JOB_BENCH_JUDGE_API_KEY="$NVIDIA_API_KEY"
export JOB_BENCH_JUDGE_MODEL="$POLICY_MODEL"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run gym eval run \
  --benchmark job_bench \
  --model-type vllm_model \
  --model-url "$POLICY_BASE_URL" \
  --model "$POLICY_MODEL" \
  --model-api-key "$NVIDIA_API_KEY" \
  --split benchmark \
  --output results/job_bench.jsonl \
  +default_host="$ROUTABLE_HOST_IP"
```

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and a routable `ROUTABLE_HOST_IP`. Use `JOB_BENCH_JUDGE_*` to
override the judge. Set `JOB_BENCH_SPLIT=easy` for the smaller non-leaderboard split.
Without explicit judge overrides, the upstream-derived YAML still defaults to Grok 4.3 on XAI.
Keep one judge fixed across model comparisons, including its revision, sampling, and input protocol.
The default dataset configuration runs one repeat; set `num_repeats: 4` for avg@4 and include every planned trial.
