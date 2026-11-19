# Workspace-Bench

The 100 English Workspace-Bench-Lite tasks, run with `harness_agent`.

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

Use `codex`, `pi`, or `hermes` in place of `claude_code` to change the harness.

Set these before running:

- `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the sandboxes.
- `WORKSPACE_BENCH_JUDGE_BASE_URL`, `WORKSPACE_BENCH_JUDGE_API_KEY`, and `WORKSPACE_BENCH_JUDGE_MODEL` for an
  Anthropic-compatible judge. Keep the judge fixed across compared runs.
- `WORKSPACE_BENCH_IMAGE` for an image built from upstream `evaluation/docker/Dockerfile` at commit
  `3fbd0f1a136720fece86786545983e26642c3db2`. It also needs that commit's `evaluation/` directory at
  `/workspace/Workspace-Bench/evaluation` with `npm ci` run there, and `python3` with Gym's dependencies.
