# SuperGPQA

Adds the `supergpqa` benchmark to Gym on top of the shared `mcqa` resource
server.

## Details

- Data source: `m-a-p/SuperGPQA`
- Evaluation: multiple choice with up to 10 answer choices
- Prompt mirrors Skills' `eval/aai/mcq-10choices`
- Verification uses `mcqa` with `Answer: X` extraction

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark supergpqa

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark supergpqa

# Collecting rollouts
gym eval run --no-serve \
    --agent supergpqa_mcqa_simple_agent \
    --input benchmarks/supergpqa/data/supergpqa_benchmark.jsonl \
    --output results/supergpqa/rollouts.jsonl \
    --prompt-config benchmarks/prompts/eval/aai/mcq-10choices.yaml
```
