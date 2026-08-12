# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`

## Example usage

```bash
# Prepare benchmark data
uv run gym eval prepare --config benchmarks/swebench/verified/claude_code.yaml

# Running servers
uv run gym env start \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config benchmarks/swebench/verified/claude_code.yaml

# Collecting rollouts
uv run gym eval run --no-serve \
    --agent claude_code_sandboxed_agent_swebench_verified \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/swebench_verified/rollouts.jsonl
```

The Claude Code config matches AnySWE's 100-turn limit and 1800-second Claude Code timeout. Both paths
extract the working-tree patch and verify it in a fresh task image. This config delegates that work
to the shared SWEBench resources server.
