# Workspace-Bench

This recipe runs the public English Workspace-Bench-Lite tasks with the sandboxed OpenCode agent. Each task
gets a fresh OpenSandbox instance containing only its source files. Rubric metadata remains on the resources server.
The reward is the upstream metric: passed rubrics divided by total rubrics.

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `NVIDIA_API_KEY`. The policy defaults to
`nvidia/moonshotai/kimi-k3` on NVIDIA's internal API. Override it with `WORKSPACE_BENCH_POLICY_MODEL`. The judge
defaults to `nvidia/openai/gpt-oss-20b`. Override it with `WORKSPACE_BENCH_JUDGE_MODEL`.

Workspace-Bench tasks depend on office tooling. Build and publish the upstream `evaluation/docker/Dockerfile`, then set
`WORKSPACE_BENCH_IMAGE` to that OpenSandbox-accessible image reference. The default `node:24-bookworm-slim` image
supports tasks that only need code or text. A preinstalled OpenCode binary avoids installing it once per task. Set
`remote_opencode_binary_path` to its path in the image.

The model server must be reachable from OpenSandbox. Set `default_host` to a routable host address (or use
`use_absolute_ip=true` where hostname resolution returns one). Loopback addresses only work with local sandboxes.

```bash
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run gym eval prepare --benchmark workspace_bench
uv run gym eval run --benchmark workspace_bench --model-type inference_provider \
  --split benchmark \
  --output results/workspace_bench_lite.jsonl \
  +default_host="$ROUTABLE_HOST_IP"
```

Source: [OpenDataBox/Workspace-Bench](https://github.com/OpenDataBox/Workspace-Bench). This minimal integration uses
task-local provided inputs rather than materializing the optional multi-gigabyte persona workspace archives.
