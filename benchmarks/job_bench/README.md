# Job-Bench

Runs the official Job-Bench `main` split with OpenCode 1.14.18 and its weighted-rubric evaluator. Grok 4.3 is the
default judge.

```bash
uv run gym eval prepare --benchmark job_bench

export NVIDIA_API_KEY=...
export XAI_API_KEY=...
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run gym eval run \
  --benchmark job_bench \
  --model-type inference_provider \
  --model-url https://inference-api.nvidia.com/v1 \
  --model nvidia/moonshotai/kimi-k3 \
  --model-api-key "$NVIDIA_API_KEY" \
  --split benchmark \
  --output results/job_bench_kimi_k3.jsonl \
  +default_host="$ROUTABLE_HOST_IP"
```

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and a routable `ROUTABLE_HOST_IP`. Use `JOB_BENCH_JUDGE_*` to
override the judge. Set `JOB_BENCH_SPLIT=easy` for the smaller non-leaderboard split.
