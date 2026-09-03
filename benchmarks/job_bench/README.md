# Job-Bench

This integration runs the official Job-Bench `main` split in an isolated sandbox with OpenCode 1.14.18,
then applies the upstream weighted-rubric evaluator using the official Grok 4.3 judge default. The upstream
evaluator is vendored in `resources_servers/job_bench/vendor/judge.py` to preserve its scoring behavior.

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

The published Kimi K3 target is 54.3% weighted score. Run all 65 `main` tasks for a comparable score. The default
judge matches the official harness. Set `JOB_BENCH_JUDGE_*` to override it.
`JOB_BENCH_SPLIT=easy` runs the smaller easy split. Easy-split scores are not leaderboard results.

The benchmark selects Gym's OpenSandbox provider. Export `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` before
starting a run, and set `ROUTABLE_HOST_IP` to an address the remote sandbox can use to reach Gym's model server.
Before a full run, validate one task end to end and confirm that the sandbox can install OpenCode, access the
proxied model endpoint, read the task files, write `/workspace/output`, and return the output archive.
