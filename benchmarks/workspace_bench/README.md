# Workspace-Bench

Runs the public English Workspace-Bench-Lite tasks with OpenCode in a fresh OpenSandbox. Rubrics remain on the
resources server. The verifier uses the pinned upstream agent-as-a-judge and dependency-graph builder.

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `NVIDIA_API_KEY`. The policy defaults to
`nvidia/moonshotai/kimi-k3` on NVIDIA's internal API. Override it with `WORKSPACE_BENCH_POLICY_MODEL`.

Set `WORKSPACE_BENCH_JUDGE_BASE_URL`, `WORKSPACE_BENCH_JUDGE_API_KEY`, and `WORKSPACE_BENCH_JUDGE_MODEL` to an
Anthropic-compatible judge endpoint. Set `WORKSPACE_BENCH_UPSTREAM_DIR` to reuse an existing upstream checkout.

Build the upstream `evaluation/docker/Dockerfile` and set `WORKSPACE_BENCH_IMAGE` to its OpenSandbox image reference.

Set `default_host` to an address that OpenSandbox can use to reach the model server.

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run gym eval prepare --benchmark workspace_bench
uv run gym eval run --benchmark workspace_bench --model-type inference_provider \
  --split benchmark \
  --output results/workspace_bench_lite.jsonl \
  +default_host="$ROUTABLE_HOST_IP"
```
