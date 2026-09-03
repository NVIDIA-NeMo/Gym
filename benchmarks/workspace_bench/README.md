# Workspace-Bench

This recipe runs the public English Workspace-Bench-Lite tasks with the sandboxed OpenCode agent. Each task
gets a fresh OpenSandbox instance containing only its source files. Rubric metadata remains on the resources server.
The reward is the upstream metric: passed rubrics divided by total rubrics.

Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY`, and `NVIDIA_API_KEY`. The policy defaults to
`nvidia/moonshotai/kimi-k3` on NVIDIA's internal API. Override it with `WORKSPACE_BENCH_POLICY_MODEL`.

The verifier runs the official Workspace-Bench agent-as-a-judge at commit
`3fbd0f1a136720fece86786545983e26642c3db2`. Set `WORKSPACE_BENCH_JUDGE_BASE_URL`,
`WORKSPACE_BENCH_JUDGE_API_KEY`, and `WORKSPACE_BENCH_JUDGE_MODEL` to an Anthropic-compatible judge endpoint.
The resources server installs the pinned evaluator automatically. Set `WORKSPACE_BENCH_UPSTREAM_DIR` to reuse an
existing checkout.

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

Source: [OpenDataBox/Workspace-Bench](https://github.com/OpenDataBox/Workspace-Bench). The OpenCode rollout uses the
official task-local inputs, sandbox image, office skills, agentic rubric evaluator, and dependency-graph builder.
