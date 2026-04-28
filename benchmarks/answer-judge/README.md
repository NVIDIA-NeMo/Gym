# Answer-Judge

Adds the `answer-judge` benchmark to Gym. Each row contains a math problem, a
predicted answer, an expected answer, and the gold `Judgement: Yes/No` label.

## Verification

This benchmark reuses `math_proof_judgement` because the verifier logic needed
here is the same deterministic `Judgement: Yes/No` parsing used by Skills'
`answer-judgement` metric.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/answer-judge/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/answer-judge/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=answer_judge_math_proof_judgement_simple_agent \
    +input_jsonl_fpath=benchmarks/answer-judge/data/answer-judge_benchmark.jsonl \
    +output_jsonl_fpath=results/answer-judge/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/math_answer_judge.yaml
```
