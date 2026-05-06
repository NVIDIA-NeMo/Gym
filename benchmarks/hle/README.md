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
ng_prepare_benchmark "+config_paths=[benchmarks/hle/config.yaml]"
```

Downloads `cais/hle`, filters to text-only questions, and writes
`benchmarks/hle/data/hle_benchmark.jsonl`.

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/hle/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

Requires `policy_base_url` / `policy_api_key` / `policy_model_name` in
`env.yaml` (or passed as CLI overrides).

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=hle_equivalence_llm_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/hle/data/hle_benchmark.jsonl \
    +output_jsonl_fpath=results/hle_rollouts.jsonl \
    +prompt_config=benchmarks/hle/prompts/default.yaml \
    +num_repeats=1 \
    "++responses_create_params={temperature: 0.0}"
```

Use `temperature: 0.0` to match the nemo-skills evaluation setup and ensure reproducible scores.

## Metrics

`pass@1/judge_accuracy` is the headline metric.
