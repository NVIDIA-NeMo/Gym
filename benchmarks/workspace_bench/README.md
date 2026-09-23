# Workspace-Bench

The 100-task English Workspace-Bench-Lite split, run with `harness_agent` using Claude Code, Codex, Pi, or Hermes.
Outputs are graded by the pinned upstream agent-as-a-judge. See
[the resources server](../../resources_servers/workspace_bench/README.md) for how grading works.

```bash
gym eval prepare --benchmark workspace_bench/claude_code

gym eval run \
  --benchmark workspace_bench/claude_code \
  --split benchmark \
  --model-type vllm_model \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY" \
  --output results/workspace_bench_lite.jsonl
```

Replace `claude_code` with `codex`, `pi`, or `hermes` to change the harness.

- Sandboxes run on OpenSandbox: set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`.
- Set `WORKSPACE_BENCH_JUDGE_BASE_URL`, `WORKSPACE_BENCH_JUDGE_API_KEY`, and `WORKSPACE_BENCH_JUDGE_MODEL` to an
  Anthropic-compatible judge endpoint reachable from the sandbox. Keep it fixed across compared runs.
- Set `WORKSPACE_BENCH_IMAGE` to an image built from upstream `evaluation/docker/Dockerfile` at commit
  `3fbd0f1a136720fece86786545983e26642c3db2`, with that commit's `evaluation/` directory at
  `/workspace/Workspace-Bench/evaluation` (plus `npm ci` there) and `python3` with Gym's dependencies for
  `harness_agent`.
