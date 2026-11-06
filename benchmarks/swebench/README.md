# SWE Bench

Adds multiple SWE benchmarks to Gym. These are useable across agents/harnesses.

## Details

- Data source: `princeton-nlp/SWE-bench_Verified`
- SWE Bench Multimodal (`benchmarks/swebench/multimodal/opencode.yaml`): the dev split of
  `SWE-bench/SWE-bench_Multimodal`, 100 JavaScript tasks. The prepare script downloads each
  issue's images and attaches them to the prompt, and stores the test patch's reference images
  for the verifier, so it needs internet access; rollouts and verification do not. See
  [SWE Bench Multimodal grading](../../resources_servers/swebench/README.md#swe-bench-multimodal-grading).
  The test split is not prepared, because the pinned SWE-bench harness (swebench 4.1.0) has no
  environment specs for its repositories.

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# Running servers
gym env start \
    --model-type vllm_model \
    --config benchmarks/swebench/verified/opencode.yaml

# Collecting rollouts
gym eval run --no-serve \
    --agent mini_swe_agent_2_swebench_verified \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/swebench_verified/rollouts.jsonl
```
