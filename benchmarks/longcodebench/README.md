# LongCodeBench (LongCodeQA)

[LongCodeBench](https://huggingface.co/datasets/Steefano/LCB) is a multi-choice
question-answering benchmark over long code contexts. Each row presents a
long code prompt with options A/B/C/D and asks the model to pick the correct
letter; the prompt postfix instructs the model to emit `Answer: \boxed{X}`.

This benchmark reuses the existing `mcqa` resource server with
`grading_mode=strict_single_letter_boxed`. Each row's `question` field carries
the long code prompt plus the postfix; the shared
`benchmarks/prompts/generic_default.yaml` template (`user: "{question}"`)
wraps it as a single user message, mirroring NeMo Skills' `prompt_format=openai`
behaviour.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/longcodebench/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/longcodebench/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=longcodebench_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/longcodebench/data/longcodebench_benchmark.jsonl \
    +output_jsonl_fpath=results/longcodebench_rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic_default.yaml \
    +num_repeats=4
```
