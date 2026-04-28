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
ng_prepare_benchmark "+config_paths=[benchmarks/supergpqa/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/supergpqa/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=supergpqa_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/supergpqa/data/supergpqa_benchmark.jsonl \
    +output_jsonl_fpath=results/supergpqa/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/mcq_10choices.yaml
```
