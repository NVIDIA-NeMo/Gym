# AIME 2026

AIME 2026 (American Invitational Mathematics Examination) — 30 competition math problems requiring integer answers in [0, 999]. Verification uses symbolic math equivalence (`math_verify`) with an LLM judge fallback.

## Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/aime26/config.yaml]"
```

## Run servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,benchmarks/aime26/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=aime26_math_with_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/aime26/data/aime26_benchmark.jsonl \
    +output_jsonl_fpath=results/aime26_rollouts.jsonl \
    +num_repeats=4
```
