# Workspace-Bench

The 100 English Workspace-Bench-Lite tasks, run with Claude Code, Codex, Pi, or Hermes.

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
  `/workspace/Workspace-Bench/evaluation` with `npm ci` run there. The judge runs in a sandbox of the same image.

`codex`, `pi` and `hermes` run through `single_agent_turn_legacy`: the resources server lends each task's sandbox to
`codex_agent`, `pi_agent` or `hermes_agent`, which installs its pinned runtime there, so the image needs `bash`, `tar`
and glibc. Codex and Pi download Node.js and their CLI into the sandbox at each seed, so the sandbox needs internet
access; Hermes uploads its runtime. The agent calls the Gym model server from inside the sandbox, so these configs set
`use_absolute_ip: true` and the servers advertise this host's address: pass `+default_host=<address the sandbox
reaches>` when the hostname does not resolve. `claude_code` still runs through `harness_agent` behind
`legacy_agent`, which runs the agent inside the task sandbox; for it the image also needs `python3` with Gym's
dependencies.
