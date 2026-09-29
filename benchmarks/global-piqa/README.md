# Global-PIQA

Adds the `global_piqa` benchmark to Gym on top of the shared `mcqa` resource
server.

## Details

- Data source: `mrlbenchmarks/global-piqa-nonparallel`
- Evaluation: 2-choice multiple choice
- Prompt uses the benchmark-local template matching the original Global-PIQA
  format
- Each row carries the original Skills regex list via
  `template_metadata.output_regex`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark global-piqa

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark global-piqa

# Collecting rollouts
gym eval run --no-serve \
    --agent global_piqa_mcqa_simple_agent \
    --input benchmarks/global-piqa/data/global-piqa_benchmark.jsonl \
    --output results/global-piqa/rollouts.jsonl \
    --prompt-config benchmarks/global-piqa/prompts/default.yaml
```
