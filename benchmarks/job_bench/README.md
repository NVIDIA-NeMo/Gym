# Job-Bench

The 65-task Job-Bench `main` split, run with `harness_agent` using Claude Code, Codex, Pi, or Hermes. Deliverables are
graded by the upstream Job-Bench rubric judge. See [the resources server](../../resources_servers/job_bench/README.md)
for how grading differs from upstream.

```bash
gym eval prepare --benchmark job_bench/claude_code

gym eval run \
  --benchmark job_bench/claude_code \
  --split benchmark \
  --model-type vllm_model \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY" \
  --output results/job_bench.jsonl
```

Replace `claude_code` with `codex`, `pi`, or `hermes` to change the harness.

- Sandboxes run on OpenSandbox: set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY`.
- The judge defaults to Grok 4.3 on xAI (`XAI_API_KEY`), as upstream does. Override it with `JOB_BENCH_JUDGE_BASE_URL`,
  `JOB_BENCH_JUDGE_API_KEY`, and `JOB_BENCH_JUDGE_MODEL`, and keep it fixed across compared runs.
- `harness_agent` runs the agent inside the task sandbox, so `JOB_BENCH_SANDBOX_IMAGE` must include `python3` and Gym's
  dependencies.
- Set `JOB_BENCH_SPLIT=easy` before `prepare` for the smaller split.
