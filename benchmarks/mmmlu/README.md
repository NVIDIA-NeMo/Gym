# MMMLU

Migrates NeMo Skills' `mmmlu` benchmark to Gym on top of the shared `mcqa`
resource server.

## Details

- Data source: OpenAI simple-evals public CSV files
- Default languages: Skills' multilingual set, excluding English by default
- Evaluation: multiple choice with multilingual answer extraction regexes
- Prompt: shared passthrough prompt, matching Skills' `generic/default`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark mmmlu

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark mmmlu

# Collecting rollouts
gym eval run --no-serve \
    --agent mmmlu_mcqa_simple_agent \
    --input benchmarks/mmmlu/data/mmmlu_benchmark.jsonl \
    --output results/mmmlu/rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/default.yaml
```
