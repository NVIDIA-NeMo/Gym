# HLE Benchmark

Benchmark wrapper for [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle), a
2158-question (text-only subset) exam covering graduate-level STEM and humanities knowledge.

- **Tasks**: 2158 text-only questions (image questions filtered at prepare time)
- **Reward**: binary; LLM judge checks whether the model's response matches the ground-truth answer
- **Metrics**: `pass@1/judge_accuracy` — fraction of questions judged correct

The judge uses the official HLE evaluation prompt adapted from
[`centerforaisafety/hle`](https://github.com/centerforaisafety/hle), which extracts the model's
final answer and checks it against the expected answer with a yes/no verdict. The policy model
serves as the judge — no separate judge server is needed.

## Dataset access

`cais/hle` is a gated HuggingFace dataset. Request access at
[https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle), then
authenticate:

```bash
huggingface-cli login
```

## Prepare benchmark data

```bash
gym eval prepare --benchmark hle
```

Downloads `cais/hle`, filters to text-only questions, and writes
`benchmarks/hle/data/hle_benchmark.jsonl`.

## Running servers

```bash
gym env start \
    --model-type vllm_model \
    --benchmark hle
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collect rollouts

```bash
gym eval run --no-serve \
    --agent hle_equivalence_llm_judge_simple_agent \
    --input benchmarks/hle/data/hle_benchmark.jsonl \
    --output results/hle_rollouts.jsonl \
    --prompt-config benchmarks/hle/prompts/default.yaml \
    --num-repeats 1 \
    --temperature 0.0
```

Use `temperature: 0.0` to match the nemo-skills evaluation setup and ensure reproducible scores.

## Metrics

`pass@1/judge_accuracy` is the headline metric.
