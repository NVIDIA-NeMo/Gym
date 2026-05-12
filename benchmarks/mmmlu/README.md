# MMMLU

Migrates NeMo Skills' `mmmlu` benchmark to Gym on top of the shared `mcqa`
resource server.

## Details

- Data source: OpenAI simple-evals public CSV files
- Default languages: Skills' multilingual set, excluding English by default
- Evaluation: multiple choice with multilingual answer extraction regexes
- Prompt: shared passthrough prompt, matching Skills' `generic/default`

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/mmmlu/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/mmmlu/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=mmmlu_mcqa_simple_agent \
    +input_jsonl_fpath=benchmarks/mmmlu/data/mmmlu_benchmark.jsonl \
    +output_jsonl_fpath=results/mmmlu/rollouts.jsonl \
    +prompt_config=benchmarks/prompts/generic_default.yaml
```
