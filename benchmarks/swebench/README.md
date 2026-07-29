# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

TODO @bxyu-nvidia: Currently this is fixed to use Mini SWE agent harness.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --config benchmarks/swebench/verified/config.yaml

# Running servers
gym env start \
    --model-type vllm_model \
    --config benchmarks/swebench/verified/config.yaml

# Collecting rollouts
gym eval run --no-serve \
    --agent mini_swe_agent_2_swebench_verified \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/swebench_verified/rollouts.jsonl
```
