# MMLU-Redux

Migrates NeMo Skills' `mmlu-redux` benchmark to Gym on top of the shared
`mcqa` resource server.

## Details

- Data source: `edinburgh-dawg/mmlu-redux-2.0` on HuggingFace
- Default split: `test`
- Evaluation: multiple choice, boxed answer letter
- Prompt: mirrors Skills' `generic/general-boxed`
- `wrong_groundtruth` rows use the dataset's corrected answer label

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark mmlu-redux

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark mmlu-redux

# Collecting rollouts
gym eval run --no-serve \
    --agent mmlu-redux_mcqa_simple_agent \
    --input benchmarks/mmlu-redux/data/mmlu-redux_benchmark.jsonl \
    --output results/mmlu-redux/rollouts.jsonl \
    --prompt-config benchmarks/prompts/generic/general-boxed.yaml
```
