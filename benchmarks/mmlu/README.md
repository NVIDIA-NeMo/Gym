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
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/mmlu/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=mmlu_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/mmlu/data/mmlu_benchmark.jsonl \
    +output_jsonl_fpath=results/mmlu/rollouts.jsonl \
    +prompt_config=benchmarks/mmlu/prompts/default.yaml
```
