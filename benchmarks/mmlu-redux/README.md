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
ng_prepare_benchmark "+config_paths=[benchmarks/mmlu-redux/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/mmlu-redux/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=mmlu-redux_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/mmlu-redux/data/mmlu-redux_benchmark.jsonl \
    +output_jsonl_fpath=results/mmlu-redux/rollouts.jsonl \
    +prompt_config=benchmarks/mmlu-redux/prompts/default.yaml
```
