# TerminalBench

This integration runs Claude Code in a sandbox created by the TerminalBench resources server. The
agent server contains only Claude Code harness behavior. The resources server owns the task image,
hidden tests, and reward.

```bash
uv run gym eval prepare --benchmark terminalbench

uv run gym eval run \
  --benchmark terminalbench \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type vllm_model \
  --model-url "$POLICY_MODEL_URL" \
  --model-api-key "$POLICY_MODEL_API_KEY" \
  --split benchmark \
  --output results/terminalbench/claude_code.jsonl
```
