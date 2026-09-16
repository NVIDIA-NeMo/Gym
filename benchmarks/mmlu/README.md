# MMLU

Migrates NeMo Skills' `mmlu` benchmark to Gym on top of the shared `mcqa`
resource server.

## Details

- Data source: `https://people.eecs.berkeley.edu/~hendrycks/data.tar`
- Default split: `test`
- Evaluation: multiple choice, boxed answer letter
- Prompt: mirrors Skills' `eval/aai/mcq-4choices-boxed`

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark mmlu

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark mmlu

# Collecting rollouts
gym eval run --no-serve \
    --agent mmlu_mcqa_simple_agent \
    --input benchmarks/mmlu/data/mmlu_benchmark.jsonl \
    --output results/mmlu/rollouts.jsonl \
    --prompt-config benchmarks/prompts/eval/aai/mcq-4choices-boxed.yaml
```

## NOOA agent flavor

The `mmlu/config_nooa` flavor evaluates the same MMLU split with the NOOA
agent adapter and the same MCQA verifier. The benchmark prompt is mapped to
the typed NOOA `GymResourceAgent.answer(task: str)` entrypoint.

```bash
# Prepare the shared MMLU dataset
gym eval prepare --benchmark mmlu/config_nooa

# Start the model, NOOA agent, and MCQA resources servers and collect rollouts
gym eval run \
    --benchmark mmlu/config_nooa \
    --model-type openai_model \
    --model-url "$MODEL_URL" \
    --model-api-key "$MODEL_API_KEY" \
    --model "$MODEL_NAME" \
    --output results/mmlu/nooa_rollouts.jsonl
```
